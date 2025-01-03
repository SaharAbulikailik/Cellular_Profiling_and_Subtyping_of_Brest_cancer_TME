# Cellular Profiling and Subtyping of Breast Cancer Tumor Microenvironment (TME)

This repository contains the implementation for cellular profiling and subtyping of breast cancer TME, leveraging MCC-UNet for robust nuclear segmentation and downstream analysis for tumor subtyping and immune profiling.

---

## **Overview**

This project addresses the dynamic nature of the tumor microenvironment by:
- **Segmenting Nuclei**: Using MCC-UNet for accurate segmentation of nuclei in multispectral immunofluorescence images.
- **Extracting Features**: Computing morphometric and protein expression indices.
- **Classifying Cells**: Classifying lymphocytes using machine learning models.
- **Tumor Subtyping**: Clustering tumor features to identify subtypes and their biological significance.

---

## **Pipeline**

The pipeline integrates tumor growth, imaging, segmentation, feature extraction, classification, and subtyping into a streamlined workflow:

![Pipeline Overview](figures/pipeline_figure.png)

1. **Tumor Sectioning, Staining, and Imaging**
2. **Multi-Spectral Image Segmentation** with MCC-UNet
3. **Feature Aggregation, Subtyping, and Analysis**
4. **Visualization of Results**

---

## **Steps to Use the Repository**

### **1. Prepare the Environment**
Install all required dependencies:
```bash
conda env create -f environment.yml
conda activate lymphocyte_env
```

### **2. Train the MCC-UNet Model**
Organize your training images and masks under the `data/Train` directory. Train the MCC-UNet model to segment nuclei.

### **3. Generate Segmentation Masks**
Use the trained MCC-UNet model to predict and save segmentation masks for test images.

### **4. Extract Features**
Compute morphometric and protein expression features from segmented masks.

### **5. Perform Tumor Subtyping**
Cluster the extracted features to identify tumor subtypes and their characteristics.

### **6. Classify Lymphocytes**
Use the trained classifier to classify lymphocytes and compute lymphocyte-related metrics.

---

## **Repository Structure**

```
Cellular_Profiling_and_Subtyping_of_Brest_cancer_TME/
├── data/
│   ├── Train/                # Training images and masks
│   ├── Test/                 # Testing images
├── results/                  # Generated results (features, classifications, subtypes)
├── src/
│   ├── segmentation/         # MCC-UNet model and training scripts
│   ├── feature_extraction/   # Feature extraction scripts
│   ├── classification/       # Lymphocyte classification scripts
│   ├── analysis/             # Tumor subtyping and clustering scripts
├── figures/                  # Pipeline and result figures
├── environment.yml           # Conda environment configuration
├── README.md                 # Documentation
```

---

## **Results**

- **Segmentation**:
  - Dice Score: 95.71%
  - Panoptic Quality: 82.53%

- **Tumor Subtyping**:
  - Identified three tumor subtypes with distinct characteristics.

- **Lymphocyte Classification**:
  - Accuracy: 92%
  - Precision: 91%
  - Recall: 93%

---

## **Citation**

If you use this repository, please cite our work:

**Paper Title**: Robust Cellular Profiling of the Tumor Microenvironment Unveils Subtype-specific Growth Patterns  
**Authors**: [Sahar Mohammed]  
**DOI**: [Link Here]

---

## **Contact**

For questions or collaborations, contact **[Sahar Mohammed]** at **[saharabulikailik@gmail.com]**.

