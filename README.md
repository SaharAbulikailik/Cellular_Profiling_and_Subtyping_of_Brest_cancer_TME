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
│   ├── images/                                  # Directory containing input images for training and testing
│   ├── masks/                                   # Directory containing ground truth masks for training and evaluation
├── docs/                                        # Documentation and related files, including the paper
├── src/
│   ├── segmentation/                            # Scripts for MCC-UNet segmentation pipeline
│   │  ├── custom_loss.py                        # Implementation of the custom loss function
│   │  ├── dataset_loader.py                     # Dataset loader for images and masks
│   │  ├── mask_generation.py                    # Script for generating segmentation masks using the trained model
│   │  ├── train.py                              # Script to train the MCC-UNet model
│   │  ├── unet3d.py                             # Definition of the MCC-UNet 3D architecture
│   ├── analysis/                                # Scripts for tumor subtyping, clustering, and analysis
│   │  ├── consensus_clustering.py               # Script for performing consensus clustering
│   │  ├── lymphocyte_association.ipynb          # Notebook for analyzing lymphocyte associations
│   │  ├── Lymphocytes_classification.py         # Script for classifying lymphocytes
│   │  ├── process_masks.py                      # Script for processing masks for feature extraction
│   │  ├── nuclear_subtyping.py                  # Script for nuclear subtyping and feature aggregation
├── lymphocyte_env/                              # Virtual environment directory
├── README.md                                    # Documentation and project overview
├── requirements.txt                             # List of required libraries and dependencies
```

---

## **Results**

- **Segmentation**:
  - Dice Score: 95.71%
  - Panoptic Quality: 82.53%

- **Tumor Subtyping**:
  - Identified three tumor subtypes with distinct characteristics.

- **Lymphocyte Classification**:
  - Accuracy: 97.1%
  - Precision: 97.9%
  - Recall: 97.1%

---

## **Citation**

If you use this repository, please cite our work:

**Paper Title**: Robust Cellular Profiling of the Tumor Microenvironment Unveils Subtype-specific Growth Patterns  
**Authors**: [Sahar Mohammed]  
**DOI**: [Link Here]

---

## **Contact**

For questions or collaborations, contact **[Sahar Mohammed]** at **[saharabulikailik@gmail.com]**.

