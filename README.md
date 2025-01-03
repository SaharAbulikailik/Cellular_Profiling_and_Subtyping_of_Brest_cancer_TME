# Cellular_Profiling_and_Subtyping_of_Brest_cancer_TME
=======
---

### **`README.md`**

```markdown
# Cellular Profiling and Subtyping of Breast Cancer Tumor Microenvironment (TME)

This repository accompanies our study on cellular profiling and subtyping of the breast cancer tumor microenvironment (TME). The project leverages the MCC-UNet model for robust nuclear segmentation and subsequent computational analysis, providing insights into tumor composition and immune profiles.

---

## Table of Contents

1. [Overview](#overview)
2. [Pipeline](#pipeline)
3. [Results](#results)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Data](#data)
7. [Citation](#citation)

---

## **Overview**

The breast cancer TME evolves dynamically during tumorigenesis, influenced by systemic inflammation and treatment interventions. This repository focuses on:
- **Accurate Nuclear Segmentation**: Using MCC-UNet to segment nuclei in multispectral immunofluorescence images.
- **Computational Analysis**: Deriving morphometric and protein expression features for tumor subtyping and immune profiling.

---

## **Pipeline**

The pipeline integrates data acquisition, segmentation, feature extraction, and tumor subtyping into a cohesive workflow:

![Pipeline Overview](figures/pipeline_figure.png) <!-- Replace with the actual file path -->

1. **Data Acquisition**:
   - Imaging of tumor sections stained for nuclear and cytoplasmic protein markers.
2. **Segmentation**:
   - Application of MCC-UNet for nuclei segmentation.
3. **Feature Extraction**:
   - Calculation of morphometric and protein expression indices.
4. **Subtype Analysis**:
   - Tumor subtyping using clustering algorithms and visualization.

---

## **Results**

### **Segmentation**
- **Dice Score**: 95.71
- **Panoptic Quality**: 82.53
- MCC-UNet surpasses other models, such as StarDist and CellPose, in segmenting dense nuclear regions.

![Segmentation Results](figures/segmentation_results.png) <!-- Replace with actual file path -->

### **Lymphocyte Classification**
- Utilized a Multi-layer Perceptron (MLP) classifier for lymphocyte identification.
- **Accuracy**: 92%
- **Precision**: 91%
- **Recall**: 93%

### **Tumor Subtyping**
- Identified three distinct tumor subtypes:
  1. **Subtype A**: High immune infiltration, slower growth.
  2. **Subtype B**: Aggressive tumor behavior with low lymphocyte frequency.
  3. **Subtype C**: Intermediate properties with distinct molecular signatures.

![Subtype Analysis](figures/tumor_subtyping.png) <!-- Replace with actual file path -->

---

## **Installation**

Clone the repository and set up the environment:
```bash
git clone https://github.com/username/Cellular_Profiling_and_Subtyping_of_Breast_Cancer_TME.git
cd Cellular_Profiling_and_Subtyping_of_Breast_Cancer_TME
conda env create -f environment.yml
conda activate lymphocyte_env
```

---

## **Usage**

### **1. Segmentation**

1. Navigate to the segmentation directory:
   ```bash
   cd src/segmentation
   ```
2. Train the MCC-UNet model:
   ```bash
   python train.py
   ```
3. Generate segmentation masks:
   ```bash
   python generate_masks.py
   ```

### **2. Lymphocyte Classification**

1. Navigate to the classification directory:
   ```bash
   cd src/classification
   ```
2. Run the classification pipeline:
   ```bash
   python classify_lymphocytes.py
   ```

---

## **Data**

Organize your dataset in the following structure:
```plaintext
data/
├── Train/
│   ├── augmented_images/   # Training images
│   ├── augmented_masks/    # Corresponding masks
├── Test/
```

---

## **Citation**

If you use this repository in your work, please cite our study:

[Paper Title Here]

**Authors**: [List of authors here]  
**DOI**: [DOI Link Here]  

---

## **Contact**

For questions or collaborations, contact Sahar Mohammed at saharabulikailik@gmail.com 
```

