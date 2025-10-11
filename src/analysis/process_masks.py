#!/usr/bin/env python3
"""
process_masks.py
- Reads multispectral .czi images and matching binary masks (<name>_mask.png)
- Builds instance masks via watershed
- Extracts morphology + per-channel intensities (DAPI, CD3, pSMAD, CD8, Ki67, Caspase)
- Saves one Excel with all cells across images

Defaults resolve relative to this file's folder:
  images:  <repo>/src/analysis/Test_images
  masks:   <repo>/src/analysis/Test_images/LoGSAGE-CBAM_masks
  out:     <repo>/src/analysis/Processed_Images_Data.xlsx
"""

from __future__ import annotations
import argparse
from pathlib import Path
import json
import os
import numpy as np
import pandas as pd
import cv2
from scipy import ndimage
from scipy.ndimage import distance_transform_edt, binary_erosion
from scipy.spatial import Delaunay
from skimage import measure, morphology, segmentation
from skimage.morphology import area_opening, disk
from skimage.feature import peak_local_max
from czifile import imread as read_czi


# ---------- Paths default to the script’s folder ----------
BASE = Path(__file__).resolve().parent
DEFAULT_IMAGES = BASE / "Test_images"
DEFAULT_MASKS  = DEFAULT_IMAGES / "LoGSAGE-CBAM_masks"
DEFAULT_OUT    = BASE / "Processed_Images_Data.xlsx"


# ---------- I/O helpers ----------
def read_czi_channels(path: Path) -> np.ndarray:
    """
    Return image as (C, H, W) float32.
    Handles common CZI axis squeezes.
    """
    arr = read_czi(str(path))
    arr = np.squeeze(arr)

    # Heuristics: if first dim looks like channels (small) -> (C,H,W)
    if arr.ndim == 3:
        if arr.shape[0] <= 12 and arr.shape[1] >= 64 and arr.shape[2] >= 64:
            out = arr.astype(np.float32)
        elif arr.shape[2] <= 12:
            # (H,W,C) -> (C,H,W)
            out = np.moveaxis(arr, -1, 0).astype(np.float32)
        else:
            raise ValueError(f"Unexpected CZI shape {arr.shape} for {path}")
    elif arr.ndim == 4:
        # Often (C,H,W,Z) with Z=1
        if arr.shape[-1] == 1:
            out = arr[..., 0].astype(np.float32)
        else:
            # (C,H,W, something) -> pick first
            out = arr[..., 0].astype(np.float32)
    else:
        raise ValueError(f"Unsupported CZI ndim={arr.ndim} for {path}")

    return out


def load_mask_png(mask_path: Path) -> np.ndarray:
    """Load a binary mask PNG -> uint8 {0,1}."""
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Mask not found or unreadable: {mask_path}")
    return (m > 0).astype(np.uint8)


def align_mask_to_image(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """
    Ensure mask (H,W) matches target_shape (H,W).
    If different, center-crop or pad with zeros; no scaling.
    """
    Ht, Wt = target_shape
    Hm, Wm = mask.shape[:2]

    # Crop if larger
    top = max((Hm - Ht) // 2, 0)
    left = max((Wm - Wt) // 2, 0)
    mask_c = mask[top:top+Ht, left:left+Wt]

    # Pad if smaller
    pad_t = max((Ht - mask_c.shape[0]) // 2, 0)
    pad_b = Ht - mask_c.shape[0] - pad_t
    pad_l = max((Wt - mask_c.shape[1]) // 2, 0)
    pad_r = Wt - mask_c.shape[1] - pad_l

    if any(x > 0 for x in (pad_t, pad_b, pad_l, pad_r)):
        mask_c = np.pad(mask_c, ((pad_t, pad_b), (pad_l, pad_r)), mode="constant", constant_values=0)
    return mask_c.astype(np.uint8)


# ---------- Instances from binary mask ----------
def mask_to_instances(
    mask_bin: np.ndarray,
    min_area: int = 50,
    erode_iters: int = 1,
    min_peak_dist: int = 6,
) -> np.ndarray:
    """
    Binary -> cleaned -> seeds (distance peaks) -> watershed -> relabel-seq
    Returns labels int32 in [0..N].
    """
    mask_bin = (mask_bin > 0).astype(np.uint8)

    # clean
    filled  = ndimage.binary_fill_holes(mask_bin).astype(np.uint8)
    cleaned = area_opening(filled, min_area, connectivity=1).astype(np.uint8)

    # distance & seeds
    eroded  = binary_erosion(cleaned, structure=np.ones((3,3)), iterations=erode_iters).astype(np.uint8)
    dist    = distance_transform_edt(eroded)
    peaks   = peak_local_max(dist, min_distance=min_peak_dist, labels=eroded, footprint=np.ones((3,3)))
    seed_img = np.zeros_like(eroded, dtype=np.uint8)
    if peaks.size > 0:
        seed_img[tuple(peaks.T)] = 1

    markers = measure.label(seed_img)
    if markers.max() == 0:
        # Fallback: use connected components if no peaks found
        markers = measure.label(eroded)

    labels = segmentation.watershed(-dist, markers, mask=cleaned)
    labels = segmentation.relabel_sequential(labels.astype(np.int32))[0]
    return labels.astype(np.int32)


# ---------- Feature extraction ----------
def _ring_mask(binary: np.ndarray, radius: int = 8) -> np.ndarray:
    """Dilated ring = dilate(binary, radius) - binary. Returns uint8 {0,1}."""
    dil = morphology.dilation(binary.astype(np.uint8), disk(radius)).astype(np.uint8)
    ring = (dil - binary.astype(np.uint8))
    ring[ring < 0] = 0
    return (ring > 0).astype(np.uint8)


def _stats_positive(vals: np.ndarray) -> tuple[float, float]:
    """Mean/median on positive values; returns (nan, nan) if empty."""
    v = vals[vals > 0]
    if v.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(v)), float(np.median(v))


def _json_list_int(xs) -> str:
    """Safe JSON for a list/array of integer-like values (handles numpy types)."""
    return json.dumps([int(x) for x in xs])

def _json_list_float(xs) -> str:
    """Safe JSON for a list/array of float-like values (handles numpy types)."""
    return json.dumps([float(x) for x in xs])


def measure_one_image(
    czi_path: Path,
    mask_dir: Path,
    chan_map: dict[str, int],
    ring_radius: int = 8,
    min_area: int = 50,
    erode_iters: int = 1,
    min_peak_dist: int = 6,
) -> pd.DataFrame:
    """
    Extracts features for one CZI + matching mask.
    chan_map keys expected (defaults): DAPI, CD3, pSMAD, CD8, Ki67, Caspase
    """
    name = czi_path.name
    # Load image channels
    CxHW = read_czi_channels(czi_path)  # (C,H,W)
    H, W = CxHW.shape[1], CxHW.shape[2]

    # Load mask (binary) and align
    mask_path = mask_dir / name.replace(".czi", "_mask.png")
    mask_bin = load_mask_png(mask_path)
    mask_bin = align_mask_to_image(mask_bin, (H, W))

    # Instances
    labels = mask_to_instances(mask_bin, min_area=min_area, erode_iters=erode_iters, min_peak_dist=min_peak_dist)
    if labels.max() == 0:
        # No instances -> return empty DF with columns
        return pd.DataFrame(columns=[
            "image_id","nuclei_id","area","solidity","elongation",
            "mean_intensity_DAPI","total_intensity_DAPI",
            "mean_cd3","median_cd3",
            "mean_psmad","median_psmad",
            "mean_cytoplasmic_cd8","median_cytoplasmic_cd8",
            "mean_nuclear_cd8","median_nuclear_cd8",
            "mean_ki67","median_ki67",
            "mean_caspase","median_caspase",
            "neighbors","neighbor_lengths","centroid_y","centroid_x","bbox_min_row","bbox_min_col","bbox_max_row","bbox_max_col"
        ])

    # Channels (with defaults)
    DAPI    = CxHW[chan_map.get("DAPI", 0), ...]
    CD3     = CxHW[chan_map.get("CD3", 1), ...]
    pSMAD   = CxHW[chan_map.get("pSMAD", 2), ...]
    CD8     = CxHW[chan_map.get("CD8", 3), ...]
    Ki67    = CxHW[chan_map.get("Ki67", 4), ...]
    Caspase = CxHW[chan_map.get("Caspase", 5), ...]

    # Regionprops on DAPI for morphology
    props = measure.regionprops(labels, intensity_image=DAPI)

    # Build centroids for graph
    centroids = np.array([p.centroid for p in props], dtype=float) if len(props) else np.empty((0,2))
    neighbors = [[] for _ in props]
    neighbor_lengths = [[] for _ in props]
    if len(centroids) >= 3:
        tri = Delaunay(centroids)
        # Build neighbor lists from triangulation
        graph = {}
        for tri_idx in tri.simplices:
            a, b, c = [int(x) for x in tri_idx]
            for u, v in [(a,b),(b,c),(c,a)]:
                graph.setdefault(u, set()).add(v)
                graph.setdefault(v, set()).add(u)
        for i in range(len(props)):
            if i in graph:
                neigh = sorted(int(n) for n in graph[i])
                neighbors[i] = neigh
                neighbor_lengths[i] = [float(np.linalg.norm(centroids[i] - centroids[j])) for j in neigh]

    # Measurements
    rows = []
    for i, p in enumerate(props):
        # binary for this nucleus
        rr, cc = p.coords[:,0], p.coords[:,1]
        nuc_mask = np.zeros((H,W), dtype=np.uint8)
        nuc_mask[rr, cc] = 1
        ring = _ring_mask(nuc_mask, radius=ring_radius)

        # Per-channel stats
        dapi_vals = DAPI[rr, cc]
        mean_dapi = float(np.mean(dapi_vals))
        total_dapi = float(dapi_vals.sum())

        cd3_mean,  cd3_med  = _stats_positive((ring * CD3).astype(np.float32)[ring > 0])
        psm_mean,  psm_med  = _stats_positive((nuc_mask * pSMAD).astype(np.float32)[nuc_mask > 0])
        cd8_c_mean, cd8_c_med = _stats_positive((ring * CD8).astype(np.float32)[ring > 0])
        cd8_n_mean, cd8_n_med = _stats_positive((nuc_mask * CD8).astype(np.float32)[nuc_mask > 0])
        ki67_mean, ki67_med = _stats_positive((nuc_mask * Ki67).astype(np.float32)[nuc_mask > 0])
        casp_mean, casp_med = _stats_positive((nuc_mask * Caspase).astype(np.float32)[nuc_mask > 0])

        rows.append({
            "image_id": name,
            "nuclei_id": int(i),
            "area": int(p.area),
            "solidity": float(p.solidity) if p.solidity is not None else float("nan"),
            "elongation": float(p.eccentricity) if p.eccentricity is not None else float("nan"),

            "mean_intensity_DAPI": mean_dapi,
            "total_intensity_DAPI": total_dapi,

            "mean_cd3": cd3_mean,
            "median_cd3": cd3_med,

            "mean_psmad": psm_mean,
            "median_psmad": psm_med,

            "mean_cytoplasmic_cd8": cd8_c_mean,
            "median_cytoplasmic_cd8": cd8_c_med,

            "mean_nuclear_cd8": cd8_n_mean,
            "median_nuclear_cd8": cd8_n_med,

            "mean_ki67": ki67_mean,
            "median_ki67": ki67_med,

            "mean_caspase": casp_mean,
            "median_caspase": casp_med,

            # cast to JSON-safe Python types
            "neighbors": _json_list_int(neighbors[i]) if i < len(neighbors) else "[]",
            "neighbor_lengths": _json_list_float(neighbor_lengths[i]) if i < len(neighbor_lengths) else "[]",

            "centroid_y": float(p.centroid[0]),
            "centroid_x": float(p.centroid[1]),
            "bbox_min_row": int(p.bbox[0]),
            "bbox_min_col": int(p.bbox[1]),
            "bbox_max_row": int(p.bbox[2]),
            "bbox_max_col": int(p.bbox[3]),
        })

    return pd.DataFrame(rows)


# ---------- CLI ----------
def parse_args():
    ap = argparse.ArgumentParser(description="Extract morphology + intensities from CZI + binary masks.")
    ap.add_argument("--images", default=str(DEFAULT_IMAGES), help="Folder with .czi images")
    ap.add_argument("--masks",  default=str(DEFAULT_MASKS),  help="Folder with *_mask.png")
    ap.add_argument("--ext",    default="czi",              help="Image extension (default: czi)")
    ap.add_argument("--out",    default=str(DEFAULT_OUT),   help="Output Excel path")
    ap.add_argument("--ring-radius", type=int, default=8,   help="Pixels for cytoplasmic ring")
    ap.add_argument("--min-area",    type=int, default=50,  help="Min area to keep in mask cleaning")
    ap.add_argument("--erode-iters", type=int, default=1,   help="Binary erosion iterations before DT")
    ap.add_argument("--min-peak-dist", type=int, default=6, help="Min distance for peak_local_max")

    # channel indices (0-based)
    ap.add_argument("--chan-DAPI",    type=int, default=0)
    ap.add_argument("--chan-CD3",     type=int, default=1)
    ap.add_argument("--chan-pSMAD",   type=int, default=2)
    ap.add_argument("--chan-CD8",     type=int, default=3)
    ap.add_argument("--chan-Ki67",    type=int, default=4)
    ap.add_argument("--chan-Caspase", type=int, default=5)

    # filter a single image by basename (no extension needed)
    ap.add_argument("--only", type=str, default="", help="Process only this file basename (e.g., A1819_P0203_4MGLTumor_1)")
    return ap.parse_args()


def main():
    args = parse_args()
    images_dir = Path(args.images).expanduser().resolve()
    masks_dir  = Path(args.masks).expanduser().resolve()
    out_path   = Path(args.out).expanduser().resolve()

    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # collect image files (case-insensitive ext)
    ext_lower = "." + args.ext.lower()
    files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(ext_lower)])
    if not files:
        raise FileNotFoundError(f"No *.{args.ext} files found in {images_dir}")

    # optional: restrict to a single basename via --only (without extension)
    if args.only:
        target_base = args.only.lower()
        # exact basename match (case-insensitive)
        matches = [f for f in files if os.path.splitext(f)[0].lower() == target_base]
        if not matches:
            # allow a partial match fallback (e.g., substring)
            matches = [f for f in files if target_base in os.path.splitext(f)[0].lower()]
        if not matches:
            raise FileNotFoundError(
                f"'{args.only}' not found in {images_dir}. "
                f"Checked among {len(files)} *.{args.ext} files."
            )
        files = sorted(matches)
        print(f"[info] --only matched {len(files)} file(s): {files}")

    chan_map = {
        "DAPI": args.chan_DAPI,
        "CD3": args.chan_CD3,
        "pSMAD": args.chan_pSMAD,
        "CD8": args.chan_CD8,
        "Ki67": args.chan_Ki67,
        "Caspase": args.chan_Caspase,
    }

    all_rows = []
    for fname in files:
        czi_path = images_dir / fname
        try:
            df = measure_one_image(
                czi_path=czi_path,
                mask_dir=masks_dir,
                chan_map=chan_map,
                ring_radius=args.ring_radius,
                min_area=args.min_area,
                erode_iters=args.erode_iters,
                min_peak_dist=args.min_peak_dist,
            )
            if not df.empty:
                all_rows.append(df)
            else:
                print(f"[warn] No instances in {fname}")
        except Exception as e:
            print(f"[error] Failed on {fname}: {e}")

    if not all_rows:
        # create empty file with columns if nothing processed
        empty = pd.DataFrame(columns=[
            "image_id","nuclei_id","area","solidity","elongation",
            "mean_intensity_DAPI","total_intensity_DAPI",
            "mean_cd3","median_cd3",
            "mean_psmad","median_psmad",
            "mean_cytoplasmic_cd8","median_cytoplasmic_cd8",
            "mean_nuclear_cd8","median_nuclear_cd8",
            "mean_ki67","median_ki67",
            "mean_caspase","median_caspase",
            "neighbors","neighbor_lengths","centroid_y","centroid_x","bbox_min_row","bbox_min_col","bbox_max_row","bbox_max_col"
        ])
        empty.to_excel(out_path, index=False)
        print(f"[done] Wrote empty table to {out_path} (no usable images).")
        return

    out_df = pd.concat(all_rows, ignore_index=True)
    out_df.to_excel(out_path, index=False)
    print(f"[done] {len(out_df)} rows saved -> {out_path}")


if __name__ == "__main__":
    main()
