# Adaptive Subtype and Stage Inference for Alzheimer's Disease (SSED)

This repository contains the implementation of the **Subtype-Specific Events Discovery (SSED)** algorithm, as presented in the paper *"Adaptive Subtype and Stage Inference for Alzheimer's Disease"*.

## Overview

**SSED** is an extension of the Subtype and Stage Inference (SuStaIn) model. [cite_start]While SuStaIn is powerful for capturing temporal and phenotypical heterogeneity in neurodegenerative diseases, it assumes that biomarkers follow a fixed set of "events" (z-score thresholds) across all subtypes[cite: 11, 12]. 

However, disease subtypes often exhibit **different progression rates**. [cite_start]For example, a specific biomarker might reach a different maximum abnormality level in one subtype compared to another[cite: 80].

**SSED addresses this by:**
* [cite_start]**Adaptive Learning:** Learning subtype-specific z-score events during the inference process rather than fixing them beforehand[cite: 13].
* [cite_start]**Handling Rate Heterogeneity:** Capturing subtypes that have distinct progression rates across regions of interest (ROIs)[cite: 86, 87].
* [cite_start]**Improved Stratification:** Providing a more precise stratification of patients based on both phenotypic and temporal aspects[cite: 35].

## Algorithm Description

The SSED algorithm iterates between fitting the model and refining the event definitions:
1.  [cite_start]**Initialization:** Initializes using standard SuStaIn assignments[cite: 91].
2.  [cite_start]**Fitting Step:** Runs SuStaIn separately on data assigned to each subtype to learn subtype-specific event sequences[cite: 93].
3.  [cite_start]**Subtyping Step:** Re-calculates the probability of each subject belonging to each subtype/stage using the new specific trajectories[cite: 95].
4.  [cite_start]**Evaluation:** Uses the **Average Stage-wise Silhouette Score (ASSS)** to evaluate clustering performance and determine convergence[cite: 118].

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
Input data should be a matrix of Z-scored biomarker values (rows = subjects, columns = biomarkers). [cite_start]The paper utilizes 5 ROIs: Temporal, Frontal, Parietal, Occipital, and Medial Temporal Lobe[cite: 151].

### 2. Running the Model
The core function `Run_SE_Recurse` (or `SSED` logic in the script) performs the iterative discovery:

```python
# Example pseudo-call based on provided script
import SSED

# Load your data
data = np.loadtxt("your_data.csv", delimiter=",")

# Run SSED logic (iterative refinement)
# See SSED.py for the full recursive implementation
