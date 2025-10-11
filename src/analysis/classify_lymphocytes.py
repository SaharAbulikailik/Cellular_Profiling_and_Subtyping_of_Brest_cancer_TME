#!/usr/bin/env python3
# classify_lymphocytes.py (robust sheet handling + pre-filter by area)
import argparse, os, json, numpy as np, pandas as pd, joblib
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ---------- column utils ----------
REQ_KEYS = ["area","pleomorphism","elongation","mean_intensity_DAPI","total_intensity_DAPI"]

def find_col(df, needle):
    for c in df.columns:
        if str(c).lower().strip() == needle.lower(): return c
    for c in df.columns:
        if needle.lower() in str(c).lower(): return c
    return None

def build_feature_frame(df):
    if isinstance(df, dict):
        if not df:
            raise ValueError("No sheets found in provided Excel (dict was empty).")
        first_name = next(iter(df.keys()))
        print(f"[info] build_feature_frame: dict detected; using first sheet '{first_name}'")
        df = df[first_name]
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected DataFrame, got {type(df)}")

    colmap = {}
    for nm in REQ_KEYS:
        col = find_col(df, nm)
        if col is None and nm == "pleomorphism":
            col = find_col(df, "solidity")
        if col is None:
            raise ValueError(
                f"Missing column for '{nm}' (or 'solidity' for pleomorphism). "
                f"Found: {list(df.columns)[:12]} ..."
            )
        colmap[nm] = col

    X_df = df[[colmap["area"], colmap["pleomorphism"], colmap["elongation"],
               colmap["mean_intensity_DAPI"], colmap["total_intensity_DAPI"]]].apply(pd.to_numeric, errors="coerce")
    valid = ~X_df.isna().any(axis=1)
    dropped = int((~valid).sum())
    if dropped:
        print(f"[warn] Dropping {dropped} rows with NaNs in required features.")
    return X_df.loc[valid], valid

def read_excel_resolving_sheet(path, sheet=None):
    if sheet is None:
        obj = pd.read_excel(path, sheet_name=None)
        if isinstance(obj, dict):
            if not obj:
                raise ValueError(f"No sheets found in Excel file: {path}")
            first_name = next(iter(obj.keys()))
            print(f"[info] Using first sheet: '{first_name}'")
            return obj[first_name].copy()
        return obj.copy()

    if isinstance(sheet, int):
        obj = pd.read_excel(path, sheet_name=None)
        if not isinstance(obj, dict) or not obj:
            raise ValueError("Expected multiple sheets to index by int, but did not find any.")
        names = list(obj.keys())
        if sheet < 0 or sheet >= len(names):
            raise IndexError(f"Sheet index {sheet} out of range (0..{len(names)-1}).")
        sel = names[sheet]
        print(f"[info] Using sheet index {sheet}: '{sel}'")
        return obj[sel].copy()
    else:
        return pd.read_excel(path, sheet_name=sheet).copy()

# ---------- PyTorch model (for .pth) ----------
import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim1=64, hidden_dim2=32, output_dim=2, p_drop=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p_drop)
    def forward(self, x):
        x = self.fc1(x); x = self.bn1(x); x = self.relu(x); x = self.dropout(x)
        x = self.fc2(x); x = self.bn2(x); x = self.relu(x); x = self.dropout(x)
        x = self.fc3(x)
        return x

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Classify lymphocytes with either sklearn .pkl or PyTorch .pth model.")
    ap.add_argument("--data", required=True, help="Excel file with cells")
    ap.add_argument("--model", required=True, help="Model path: .pkl (sklearn bundle) or .pth (PyTorch state_dict)")
    ap.add_argument("--meta", default=None, help="Joblib meta (scaler + feature_names) required if --model is .pth")
    ap.add_argument("--sheet", default=None, help="Excel sheet name or index; default: first sheet")
    ap.add_argument("--out", default=None, help="Output Excel path (default: overwrite input)")
    ap.add_argument("--pred-col", default="lympho_pred", help="Predicted class column")
    ap.add_argument("--prob-col", default="lympho_prob", help="Probability for class=1 column")
    # --- NEW: pre-filter controls ---
    ap.add_argument("--min-area", type=float, default=100, help="Min area to keep before classification")
    ap.add_argument("--max-area", type=float, default=3000, help="Max area to keep before classification")
    ap.add_argument("--save-filtered", default=None, help="Optional path to save the filtered table before classification")
    args = ap.parse_args()

    if not os.path.exists(args.data): raise FileNotFoundError(args.data)
    if not os.path.exists(args.model): raise FileNotFoundError(args.model)

    # Resolve sheet
    if args.sheet is None:
        df_raw = read_excel_resolving_sheet(args.data, None)
    else:
        sheet_resolved = int(args.sheet) if str(args.sheet).isdigit() else args.sheet
        df_raw = read_excel_resolving_sheet(args.data, sheet_resolved)

    # ---- Pre-filter by area (using robust column matching) ----
    area_col = find_col(df_raw, "area")
    if area_col is None:
        raise ValueError("Could not find 'area' column to filter on.")
    n_before = len(df_raw)
    df_filtered = df_raw[(pd.to_numeric(df_raw[area_col], errors="coerce") >= args.min_area) &
                         (pd.to_numeric(df_raw[area_col], errors="coerce") <= args.max_area)].copy()
    n_after = len(df_filtered)
    print(f"[info] Area filter: kept {n_after}/{n_before} rows in [{args.min_area}, {args.max_area}].")

    if args.save_filtered:
        df_filtered.to_excel(args.save_filtered, index=False)
        print(f"[info] Saved filtered table -> {args.save_filtered}")

    df_out = df_filtered.copy()

    # Build features & valid mask on the *filtered* table
    X_df, valid = build_feature_frame(df_out)
    X = X_df.values
    out_path = args.out if args.out else (args.save_filtered or args.data)
    ext = os.path.splitext(args.model)[1].lower()

    if ext == ".pkl":
        bundle = joblib.load(args.model)
        if not isinstance(bundle, dict) or "model" not in bundle or "scaler" not in bundle:
            raise ValueError("The provided .pkl does not look like a bundle with {'model','scaler'}.")
        clf = bundle["model"]
        scaler = bundle["scaler"]
        Xs = scaler.transform(X)
        if hasattr(clf, "predict_proba"):
            prob1 = clf.predict_proba(Xs)[:, 1]
        else:
            s = clf.decision_function(Xs)
            smin, smax = float(s.min()), float(s.max())
            prob1 = (s - smin) / (smax - smin + 1e-8)
        pred = (prob1 >= 0.5).astype(int)

    elif ext == ".pth":
        if not args.meta or not os.path.exists(args.meta):
            raise FileNotFoundError("--meta is required and must exist when using a .pth model")
        meta = joblib.load(args.meta)
        scaler = meta["scaler"]
        Xs = scaler.transform(X)

        model = SimpleNN(input_dim=Xs.shape[1], hidden_dim1=64, hidden_dim2=32, output_dim=2, p_drop=0.5)
        state = torch.load(args.model, map_location="cpu")
        model.load_state_dict(state)
        model.eval()

        with torch.no_grad():
            feats = torch.tensor(Xs, dtype=torch.float32)
            logits = model(feats)
            probs = torch.softmax(logits, dim=1).numpy()
            prob1 = probs[:, 1]
            pred = probs.argmax(axis=1).astype(int)
    else:
        raise ValueError(f"Unsupported model extension: {ext} (use .pkl or .pth)")

    # Write predictions back to the *filtered* DataFrame
    df_out.loc[valid, args.pred_col] = pred
    df_out.loc[valid, args.prob_col] = prob1
    df_out.to_excel(out_path, index=False)

    print(json.dumps({
        "n_rows_filtered": int(len(df_out)),
        "n_classified": int(valid.sum()),
        "pred_col": args.pred_col,
        "prob_col": args.prob_col,
        "out": out_path,
        "model_type": "sklearn(.pkl)" if ext==".pkl" else "torch(.pth)"
    }, indent=2))

if __name__ == "__main__":
    main()
