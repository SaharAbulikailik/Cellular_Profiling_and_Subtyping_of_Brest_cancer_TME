# Cellular Profiling and Subtyping of Breast Cancer Tumor Microenvironment (TME)

This repository contains the implementation for cellular profiling and subtyping of breast cancer TME, leveraging MCC-UNet for robust nuclear segmentation and downstream analysis for tumor subtyping and immune profiling.

---

## **Overview**

This project addresses the dynamic nature of the tumor microenvironment by:
- **Segmenting Nuclei**: Using MCC-UNet for accurate segmentation of nuclei in multispectral immunofluorescence images.
- **Extracting Features**: Computing morphometric and protein expression indices.
- **Classifying Cells**: Classifying lymphocytes using MLP model.
- **Tumor Subtyping**: Tumor subtyping and association to clinical variables.

---

## **Pipeline**

The pipeline integrates tumor growth, imaging, segmentation, feature extraction, classification, and subtyping into a streamlined workflow:

![Pipeline Overview](docs/Figure1.png)

1. **Tumor Sectioning, Staining, and Imaging**
2. **Multi-Spectral Image Segmentation** with MCC-UNet
3. **Feature Aggregation, Subtyping, and Analysis**
4. **Visualization of Results**

---


## **Analysis Steps**

The downstream analysis proceeds as follows:

1. **Segment Tumor Microenvironment (TME) Images**  
   Apply MCC-UNet to segment nuclei from multispectral immunofluorescence images.

2. **Measure Morphology and Protein Expression**  
   Extract cellular features such as area, elongation, solidity, and intensity-based protein expression indices from segmented masks.

3. **Classify Lymphocytes**  
   Use a subset of the extracted features to classify cells into lymphocytes and non-lymphocytes using an MLP model.

4. **Localize Lymphocytes**  
   Apply Delaunay triangulation to the classified lymphocytes for spatial localization. Add lymphocyte labels and their localization results to the spreadsheet for each image.

5. **Subtype Tumors via Consensus Clustering**  
   Perform consensus clustering on the aggregated feature matrix to identify tumor subtypes with distinct characteristics.

6. **Measure Feature Frequencies per Tumor**  
   Compute the frequency of morphometric and protein expression features for each tumor to represent phenotype prevalence.

7. **Visualize Subtypes**  
   Use frequency tables to generate heatmaps and other visualizations to highlight subtype-specific profiles.


---

## **Repository Structure**

```
Cellular_Profiling_and_Subtyping_of_Brest_cancer_TME/
├── data/
│   ├── augmented_images/                                  # Directory containing input images for training and testing
│   ├── augmented_masks/                                   # Directory containing ground truth masks for training and evaluation
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
  - Identified four tumor subtypes with distinct characteristics using phenotypic indices.

- **Lymphocyte Classification**:
  - Accuracy: 97%
  - Precision: 98%
  - Recall: 97%

---

## **Citation**

If you use this repository, please cite our work:

**Paper Title**: Robust Cellular Profiling of the Tumor Microenvironment Unveils Subtype-specific Growth Patterns  
**Authors**: [Sahar Mohammed]  
**DOI**: [Link Here]

---

## **Contact**

For questions or collaborations, contact **[Sahar Mohammed]** at **[saharabulikailik@gmail.com]**.

