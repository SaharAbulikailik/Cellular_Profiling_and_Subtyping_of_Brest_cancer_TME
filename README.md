# Cellular Profiling and Subtyping of Breast Cancer Tumor Microenvironment (TME)

This repository contains the implementation for **cellular profiling and subtyping of breast cancer TME**, leveraging the **MCC-UNet** model for robust nuclear segmentation and downstream analysis to uncover tumor subtypes and immune phenotypes.

---

## 🔬 Overview

This project addresses the complexity of the tumor microenvironment by:

- **Nuclear Segmentation**: Accurate segmentation of nuclei in multispectral immunofluorescence images using MCC-UNet.
- **Feature Extraction**: Morphometric and protein expression-based profiling of each cell.
- **Lymphocyte Classification**: Classification of lymphocytes using a multi-layer perceptron (MLP).
- **Tumor Subtyping**: Phenotype-driven tumor subtype discovery and association with clinical variables.

---

## 🔁 Pipeline

The pipeline integrates imaging, segmentation, feature extraction, classification, clustering, and statistical analysis.

<p align="center">
  <img src="docs/Pipeline.png" alt="Pipeline Overview" width="700"/>
</p>

1. Tumor harvesting, staining, and multispectral imaging  
2. Nuclear segmentation using MCC-UNet  
3. Feature extraction and cellular classification  
4. Tumor subtyping and clinical association  
5. Visualization and downstream analysis  

---

## 🧪 Analysis Workflow

1. **Segment Tumor Microenvironment Images**  
   - Segment nuclei using MCC-UNet.

2. **Extract Morphological & Protein Features**  
   - Compute area, elongation, solidity, and intensities for DAPI, CD3, CD8, Ki67, Caspase, and pSMAD.

3. **Classify Lymphocytes**  
   - Use MLP to classify cells as lymphocytes or non-lymphocytes.

4. **Localize Lymphocytes**  
   - Apply Delaunay triangulation to map lymphocyte distribution.

5. **Cluster Tumors via Consensus Clustering**  
   - Perform clustering on tumor-level feature frequencies to define subtypes.

6. **Quantify Feature Frequencies per Tumor**  
   - Summarize morphological and expression profiles across tumors.

7. **Visualize Subtype Patterns**  
   - Generate heatmaps, t-SNE plots, and survival curves for subtype comparison.

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


---

## 📦 Installation

```bash
conda create -n logsage_cbam python=3.10 -y
conda activate logsage_cbam
pip install -r requirements.txt
```

---

## 📚 Citation

If you use this repository, please cite:

**Title**: *Robust Cellular Profiling of the Tumor Microenvironment Unveils Subtype-specific Growth Patterns*  
**Author**: Sahar Mohammed  
**DOI**: [AfterAccpetance]

---

## 📬 Contact

For questions or collaborations, contact:  
📧 **saharabulikailik@gmail.com**
