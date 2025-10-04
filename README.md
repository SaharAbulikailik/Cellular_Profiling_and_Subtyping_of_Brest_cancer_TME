# Cellular Profiling and Subtyping of Breast Cancer Tumor Microenvironment (TME)

This repository contains the implementation for **cellular profiling and subtyping of breast cancer TME**, leveraging the **LoGSAGE-CBAM** model for robust nuclear segmentation and downstream analysis to uncover tumor subtypes and immune phenotypes.

---

## 🔬 Overview

This project addresses the complexity of the tumor microenvironment by:

- **Nuclear Segmentation**: Accurate segmentation of densely packed nuclei in multispectral immunofluorescence images using LoGSAGE-CBAM.
- **Feature Extraction**: Morphometric and protein expression-based profiling of each cell.
- **Lymphocyte Classification**: Classification of lymphocytes using a multi-layer perceptron (MLP).
- **Tumor Subtyping**: Phenotype-driven tumor subtype discovery and association with growth latency and molecular expression patterns.

---

## 🌟 Highlights

- Introduces **LoGSAGE-CBAM**, a dual-encoder segmentation model combining LoG-based saliency and Swin Transformer encoding.
- Incorporates **curvature-aware loss** to enhance biological accuracy in nuclear boundaries.
- Enables **cell classification and spatial profiling** using extracted cellular indices.
- Reveals **subtype-specific immune and morphological signatures** predictive of growth and latency.

---

## 🔁 Pipeline

The pipeline integrates imaging, segmentation, feature extraction, classification, clustering, and statistical analysis.

<p align="center">
  <img src="docs/Pipeline.png" alt="Pipeline Overview" width="700"/>
</p>

1. Tumor harvesting, staining, and multispectral imaging  
2. Nuclear segmentation using LoGSAGE-CBAM  
3. Feature extraction and cellular classification  
4. Tumor subtyping and clinical association  
5. Visualization and downstream analysis  

---

## 🧠 Model Architecture

<p align="center">
  <img src="docs/Model.png" alt="LoGSAGE-CBAM Architecture" width="700"/>
</p>

LoGSAGE-CBAM consists of two parallel encoders:
- A **saliency encoder** using Laplacian-of-Gaussian filtered DAPI to highlight nuclear structures based on UNet encoder
- A **Multi-spectral images encoder** based on a Swin Transformer
- A **CBAM-based fusion block** to adaptively merge both representations
- A UNet decoder
- Model trained with a **composite loss**:
  - **Dice Loss** for overlap accuracy
  - **Curvature Loss** for smooth, accurate boundaries

---

## 🧪 Analysis Workflow

````markdown
## 1)  **Segment Tumor Microenvironment Images (LoGSAGE-CBAM)**

**Goal:** train the nuclear segmentation model on paired TIFF images/masks.

#### a) Create & activate the environment
```bash
conda create -n CellProfiling python=3.10 -y
conda activate CellProfiling
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

#### b) Train

python /home/sahar/Cellular_Profiling_and_Subtyping_of_Brest_cancer_TME/src/segmentation_model/train.py \
  --images_dir "$IMAGES" \
  --masks_dir  "$MASKS" \
  --output_dir "$OUT" \
  --model swin_T_1 \
  --epochs 100 \
  --batch_size 4 \
  --lr 1e-4 \
  --val_split 0.2 \
  --seed 42

# from the repo root
export PYTHONPATH="$(pwd)/src:${PYTHONPATH}"

# put your .czi files in: src/analysis/Test_images/
python src/analysis/Generate_masks.py


## 2) Extract Morphological & Protein Features

* Compute area, elongation, solidity, and intensities for DAPI, CD3, CD8, Ki67, Caspase, and pSMAD.

```bash
(CellProfiling) analysis$> python /home/sahar/Cellular_Profiling_and_Subtyping_of_Brest_cancer_TME/src/analysis/process_masks.py
```
This reads `src/analysis/Test_images/*.czi` and matching masks in `src/analysis/Test_images/LoGSAGE-CBAM_masks/`, then writes `src/analysis/Processed_Images_Data.xlsx` with area, elongation, solidity, and per-channel intensities (DAPI, CD3, CD8, Ki67, Caspase, pSMAD).

**Test on a single image (by basename, no extension):**

```bash
(CellProfiling) analysis$> python /home/sahar/Cellular_Profiling_and_Subtyping_of_Brest_cancer_TME/src/analysis/process_masks.py --only A1819-P0203-4MGLTumor-1
```

## 3) Classify Lymphocytes (MLP)

**Goal:** Train a simple MLP to classify cells as lymphocyte vs. non-lymphocyte using morphology + DAPI features.

### Input

Excel file with columns (case-insensitive):

* `area`, `pleomorphism` (or `solidity`), `elongation`, `mean_intensity_DAPI`, `total_intensity_DAPI`, `TARGET` (0/1)

> If `TARGET` is not 0/1, the script maps the smallest label → 0 and largest → 1.

### Train & evaluate

```bash
# from repo root (example)
python run_mlp_and_lazy.py --data /home/sahar/Cellular_Profiling_and_Subtyping_of_Brest_cancer_TME/src/analysis/final_traindata.xlsx --out-model /home/sahar/Cellular_Profiling_and_Subtyping_of_Brest_cancer_TME/src/analysis/lymphocyte_mlp.pkl
```

### What you’ll see

```txt
=== MLP Test Metrics (3 d.p.) ===
{
  "Model": "MLPClassifier(64,32)",
  "Accuracy": 0.970,
  "Precision": 0.964,
  "Recall": 0.980,
  "F1": 0.972,
  "AUC": 0.996,
  "N_test": 929
}

=== LazyPredict-style (fallback) Baseline (3 d.p.) ===
             Model  Accuracy  Precision  Recall    F1   AUC
LogisticRegression     0.969      0.956   0.986 0.971 0.995
         LinearSVC     0.969      0.956   0.986 0.971 0.995
      RandomForest     0.966      0.956   0.980 0.968 0.995
               KNN     0.961      0.952   0.976 0.964 0.992
      DecisionTree     0.955      0.946   0.969 0.958 0.954
        GaussianNB     0.953      0.946   0.965 0.956 0.988

[Saved] MLP model -> src/analysis/lymphocyte_mlp.pkl
```

4. **Localize Lymphocytes**  
   - Apply Delaunay triangulation for lymphocytes localization.

5. **Cluster indecies via Consensus Clustering**  
   - Perform clustering on each feature to define high and low values.

6. **Quantify Feature Frequencies per Tumor**  
   - Summarize morphological and expression profiles across tumors.

7. **Visualize Subtype Patterns**  
   - Generate heatmaps, t-SNE plots, and Agressiveness curves (Latency and Growth) for subtype comparison.

---

## 📁 Repository Structure

```
.
├── docs
│   ├── Model.png
│   ├── Pipeline.png
│   └── Results.png
├── lymphocyte_env
│   ├── bin/
│   ├── lib/
│   └── ...
├── README.md
├── requirements.txt
├── saved_model/
├── saved_models/
└── src/
    ├── analysis/
    │   ├── consensus_clustering.py
    │   ├── lymphocyte_association.ipynb
    │   ├── Lymphocytes_classification.py
    │   ├── nuclear_subtyping.ipynb
    │   ├── process_masks.py
    │   └── simK_perweek.ipynb
    └── segmentation_model/
        ├── dataset/
        ├── losses/
        ├── models/
        ├── __pycache__/
        └── train.py
```

---

## 📊 Results

<p align="center">
  <img src="docs/Results.png" alt="Segmentation Results" width="700"/>
</p>

| Task                     | Metric                     | Score   |
|--------------------------|----------------------------|---------|
| **Segmentation**         | Dice Score                 | 95.5    |
|                          | Relative Count Error (RCE) | 86.6    |
| **Lymphocyte Detection** | Accuracy                   | 97.0    |
|                          | Precision                  | 98.0    |
|                          | Recall                     | 97.0    |
|                          | AUC                        | 99.0    |
| **Tumor Subtyping**      | # of Subtypes              | 4       |

> **Note:** RCE (Relative Count Error) evaluates segmentation performance by comparing the number of predicted nuclei (N_pred) to the number of ground truth nuclei (N_true). It emphasizes biologically meaningful object-level accuracy.
>
> <img src="docs/metric.png" alt="RCE formula" width="320"/>

---

## 📦 Install Environment and Train

```bash
conda create -n logsage_cbam python=3.10 -y
conda activate logsage_cbam
pip install -r requirements.txt

python src/segmentation_model/train.py
```


---

## 📬 Contact

For questions or collaborations, contact:  
📧 **saharabulikailik@gmail.com**
