# Adaptive Subtype and Stage Inference for Alzheimer's Disease (SSED)

This repository contains the implementation of the **Subtype-Specific Events Discovery (SSED)** algorithm, as presented in the paper *"Adaptive Subtype and Stage Inference for Alzheimer's Disease"*.

## Overview

**SSED** is an extension of the Subtype and Stage Inference (SuStaIn) model. While SuStaIn is powerful for capturing temporal and phenotypical heterogeneity in neurodegenerative diseases, it typically assumes that biomarkers follow a fixed set of "events" (z-score thresholds) across all subtypes.

However, disease subtypes often exhibit **different progression rates**. For example, a specific biomarker might reach a different maximum abnormality level in one subtype compared to another.

**SSED addresses this by:**
* **Adaptive Learning:** Learning subtype-specific z-score events during the inference process rather than fixing them beforehand.
* **Handling Rate Heterogeneity:** Capturing subtypes that have distinct progression rates across regions of interest (ROIs).
* **Improved Stratification:** Providing a more precise stratification of patients based on both phenotypic and temporal aspects.

## Algorithm Description

The SSED algorithm iterates between fitting the model and refining the event definitions:
1.  **Initialization:** Initializes using standard SuStaIn assignments.
2.  **Fitting Step:** Runs SuStaIn separately on data assigned to each subtype to learn subtype-specific event sequences.
3.  **Subtyping Step:** Re-calculates the probability of each subject belonging to each subtype/stage using the new specific trajectories.
4.  **Evaluation:** Uses the **Average Stage-wise Silhouette Score (ASSS)** to evaluate clustering performance and determine convergence.

## Dependencies

* **Python 3.x**
* **pySuStaIn:** [https://github.com/ucl-pond/pySuStaIn](https://github.com/ucl-pond/pySuStaIn)
* `numpy`
* `pandas`
* `matplotlib`
* `scikit-learn`

## Usage

The provided script (`SSED.py` / `SSED.ipynb`) demonstrates how to run SSED on biomarker data (e.g., Tau PET SUVR z-scores).

### 1. Data Preparation
Input data should be a matrix of Z-scored biomarker values (rows = subjects, columns = biomarkers). The paper utilizes 5 ROIs: Temporal, Frontal, Parietal, Occipital, and Medial Temporal Lobe.

### 2. Running the Model
The core function `Run_SE_Recurse` (or the SSED logic in the script) performs the iterative discovery. You can run the script via terminal:

```bash
python SSED.py
```

## Citation

If you use this code or the SSED algorithm in your research, please cite our paper:

Adaptive Subtype and Stage Inference for Alzheimer's Disease > Xinkai Wang, Yonggang Shi

Medical Image Computing and Computer Assisted Intervention (MICCAI) 2024 > DOI: 10.1007/978-3-031-72384-1_5

PMID: 39376664 | PMCID: PMC11632966
