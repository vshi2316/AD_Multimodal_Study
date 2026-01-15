# Alzheimer's Disease Multimodal Deep Phenotyping and Subtype Discovery

A comprehensive computational pipeline for discovering biologically distinct Alzheimer's disease subtypes through multimodal deep learning, validated across four independent cohorts (ADNI, AIBL, HABS, A4).

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![R](https://img.shields.io/badge/R-4.0%2B-blue)](https://www.r-project.org/)

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Data Requirements](#data-requirements)
- [Usage](#usage)
- [Output Files](#output-files)
- [Citation](#citation)
- [License](#license)

---

## Overview

This repository implements a novel framework for identifying Alzheimer's disease subtypes using:

- **Multimodal Data Integration**: APOE genotype, CSF biomarkers, clinical assessments, structural MRI, and PET imaging
- **Deep Learning Clustering**: Variational Autoencoder (VAE) for unsupervised subtype discovery
- **Multi-Cohort Validation**: External validation in AIBL, HABS, and A4 cohorts
- **Statistical Characterization**: Differential analysis, predictive modeling, and meta-analysis
- **Biological Interpretation**: Neuroimaging endotype characterization and clinical-MRI heterogeneity analysis

**Key Features**:
- 22-step end-to-end analysis pipeline
- Discovery cohort (ADNI) with 3-cohort external validation
- Random-effects meta-analysis across cohorts
- SHAP-based model interpretability
- Bootstrap validation with 1000 iterations
- Publication-ready visualizations (300 DPI)

---

## Repository Structure

```
AD_Multimodal_Study/
│
├── Data Preprocessing (Steps 1-6) - Python
│   ├── step1_preprocess_APOE.py              # APOE genotype extraction
│   ├── step2_preprocess_CSF.py               # CSF biomarker integration
│   ├── step3_preprocess_Clinical.py          # Clinical cognitive scores
│   ├── step4_preprocess_sMRI.py              # Structural MRI features
│   ├── step5_preprocess_PET.py               # PET imaging quantification
│   └── step6_create_outcome.py               # AD conversion outcomes
│
├── Cohort Integration & Clustering (Steps 7-9C) - Python
│   ├── step7_integrate_cohorts.py            # Multimodal data integration
│   ├── step8_vae_clustering.py               # VAE deep clustering
│   ├── step9A_cross_cohort_analysis.py        # Cross-cohort validation
│   ├── step9B_biomarker_validation.py        # Biomarker validation
│   
│
├── Statistical Analysis (Steps 10-13) - R
│   ├── step10_differential_analysis.R        # Limma differential analysis
│   ├── step11_predictive_modeling.R          # Multi-algorithm ML models
│   ├── step12_cluster_signatures.R           # Cluster signature visualization
│   └── step13_conversion_differential.R      # Converter vs non-converter analysis
│
├── Cluster Validation (Steps 14-15) - R
│   ├── step14_consensus_clustering.R         # Consensus clustering (PAC)
│   └── step15_cross_modal_validation.R       # CSF & MRI validation
│
├── External Validation (Steps 16-18) - R
│   ├── step16_habs_validation.R              # HABS cohort validation
│   ├── step17_meta_analysis.R                # Random-effects meta-analysis
│   └── step18_shap_analysis.R                # SHAP explainability
│
├── Discovery-Validation Chain (Steps 19-21) - R
│   ├── step19_adni_discovery.R               # ADNI discovery & classifier
│   ├── step20_aibl_preprocessing.R           # AIBL validation preprocessing
│   └── step21_a4_validation.R                # A4 large-sample validation
│
├── Biological Characterization (Steps 22) - R
│   └── step22_neuroimaging_endotypes.R       # Clinical-MRI heterogeneity
│


## Prerequisites

### Software Requirements

- **Python**: 3.8 or higher
- **R**: 4.0 or higher
- **Operating System**: Windows, macOS, or Linux

### Python Dependencies

```bash
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
tensorflow>=2.8.0
keras>=2.8.0
matplotlib>=3.5.0
seaborn>=0.11.0
```

### R Packages

```R
# Data manipulation
dplyr, tidyverse

# Statistical analysis
limma, tableone, survival, survminer

# Machine learning
randomForest, caret, glmnet, xgboost

# Clustering & validation
ConsensusClusterPlus, cluster, mclust, mice

# Meta-analysis
meta, metafor

# Interpretability
shap (via reticulate)

# Visualization
ggplot2, pheatmap, patchwork, RColorBrewer, ggrepel
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/vshi2316/AD_Multimodal_Study.git
cd AD_Multimodal_Study
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install R Packages

```R
# In R console
install.packages(c("dplyr", "tidyverse", "ggplot2", "survival", "survminer",
                   "randomForest", "caret", "glmnet", "xgboost",
                   "pheatmap", "patchwork", "RColorBrewer", "ggrepel",
                   "tableone", "cluster", "mclust", "mice"))

# Bioconductor packages
if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
BiocManager::install(c("limma", "ConsensusClusterPlus"))

# Meta-analysis packages
install.packages(c("meta", "metafor"))
```

---

## Data Requirements

### Input Data Format

All input files should be CSV format with the following structure:

#### 1. APOE Genotyping (`ApoE_Genotyping_Results.csv`)
- **Required columns**: `ID`, `APOE_Genotype`
- **Example**: ID=001, APOE_Genotype=E3/E4

#### 2. CSF Biomarkers (`CSF_*.csv`)
- **Required columns**: `ID`, `ABETA`, `TAU`, `PTAU`
- **Optional**: Additional CSF markers

#### 3. Clinical Data (`Clinical_Assessments.csv`)
- **Required columns**: `ID`, `ADAS13`, `CDRSB`, `MMSE_Baseline`, `Age`, `Gender`, `Education`
- **Optional**: FAQTOTAL, RAVLT scores

#### 4. Structural MRI (`FreeSurfer_*.csv`)
- **Required columns**: `ID`, MRI features (e.g., `ST102TA`, `ST103CV`, etc.)
- **Format**: FreeSurfer ROI measurements

#### 5. PET Imaging (`PET_SUVR_Data.csv`)
- **Required columns**: `ID`, regional SUVR values
- **Format**: Normalized to reference region

#### 6. Longitudinal Data (`CDR_Longitudinal.csv`)
- **Required columns**: `ID`, `Visit_Date`, `CDR`, `AD_Conversion`
- **Purpose**: Outcome generation

### Data Organization

```
data/
├── ADNI/
│   ├── ApoE_Genotyping_Results.csv
│   ├── CSF_Biomarkers.csv
│   ├── Clinical_Assessments.csv
│   ├── FreeSurfer_ROI.csv
│   ├── PET_SUVR_Data.csv
│   └── CDR_Longitudinal.csv
│
├── AIBL/
│   └── AIBL_Baseline_Integrated.csv
│
├── HABS/
│   └── HABS_Baseline_Integrated.csv
│
└── A4/
    └── A4_Baseline_Integrated.csv
```

---

## Usage

### Quick Start (Recommended Order)

#### Phase 1: Data Preprocessing (Python)

```bash
# Run preprocessing scripts sequentially
python step1_preprocess_APOE.py
python step2_preprocess_CSF.py
python step3_preprocess_Clinical.py
python step4_preprocess_sMRI.py
python step5_preprocess_PET.py
python step6_create_outcome.py
```

**Output**: Individual modality CSV files (e.g., `APOE_genetics.csv`, `metabolites.csv`)

#### Phase 2: Cohort Integration & Clustering (Python)

```bash
python step7_integrate_cohorts.py
python step8_vae_clustering.py
python step9A_cross_cohort_analysis.py
python step9B_biomarker_validation.py
```

**Output**: 
- `Cohort_A_Integrated.csv`, `Cohort_B_Integrated.csv`
- `VAE_latent_embeddings.csv`, `cluster_results.csv`

#### Phase 3: Statistical Analysis (R)

```R
# Differential analysis (use integrated version)
source("step10_differential_analysis_INTEGRATED.R")

# Predictive modeling
source("step11_predictive_modeling.R")

# Cluster signatures
source("step12_cluster_signatures.R")

# Conversion analysis
source("step13_conversion_differential.R")
```

**Output**: 
- Differential expression results (`DiffExpr_*.csv`)
- ML model performance (`Model_Performance_Comparison.csv`)
- Signature heatmaps

#### Phase 4: Validation (R)

```R
# Cluster validation (use integrated version)
source("step14_cluster_validation_INTEGRATED.R")

# Cross-modal validation
source("step15_cross_modal_validation.R")

# External cohort validation
source("step16_habs_validation.R")
```

**Output**:
- Stability metrics (ARI, Jaccard, Silhouette)
- Validation AUC, confusion matrices

#### Phase 5: Meta-Analysis & Interpretability (R)

```R
# Enhanced meta-analysis (use new version)
source("step17_meta_analysis_NEW.R")

# SHAP explainability
source("step18_shap_analysis.R")
```

**Output**:
- Forest plots, funnel plots
- SHAP feature importance plots

#### Phase 6: Discovery-Validation Chain (R)

```R
# ADNI discovery
source("step19_adni_discovery.R")

# AIBL validation
source("step20_aibl_preprocessing.R")

# A4 validation
source("step21_a4_validation.R")
```

**Output**:
- Trained classifier (`ADNI_Classifier.rds`)
- Validation survival curves

#### Phase 7: Biological Characterization (R)


# Neuroimaging endotypes 
source("step22_neuroimaging_endotypes_GITHUB.R")
```

**Output**:
- Subtype naming tables (HP, CD, TAD)
- Clinical-MRI heterogeneity analysis

---

## Advanced Usage

### Option 1: Use Integrated Scripts (Recommended)

For cleaner workflow, use the integrated versions:

```R
# Instead of step10 + step10B separately
source("step10_differential_analysis_INTEGRATED.R")

# Instead of step14 + step14B + step14_bootstrap separately
source("step14_cluster_validation_INTEGRATED.R")

# Enhanced meta-analysis with sensitivity analysis
source("step17_meta_analysis_NEW.R")

# GitHub-ready endotype analysis
source("step22_neuroimaging_endotypes_GITHUB.R")
```

### Option 2: Parallel Processing

For large datasets, enable parallel processing in R scripts:

```R
# In step14_cluster_validation_INTEGRATED.R
library(parallel)
n_cores <- detectCores() - 1
cl <- makeCluster(n_cores)
# ... parallel bootstrap code ...
stopCluster(cl)
```

### Option 3: Custom Cohort Analysis

To analyze your own cohort:

1. Format data according to specifications above
2. Run steps 1-6 for preprocessing
3. Run step 7-8 for integration and clustering
4. Apply the trained classifier from step19:

```R
# Load your data
new_cohort <- read.csv("Your_Cohort_Data.csv")

# Load trained classifier
classifier <- readRDS("ADNI_Classifier.rds")

# Predict subtypes
predictions <- predict(classifier, newdata = new_cohort)
```

---

## Reproducibility

### Random Seeds

All scripts use fixed random seeds for reproducibility:
- Python scripts: `np.random.seed(42)`
- R scripts: `set.seed(42)`

### Session Info

To ensure reproducibility, save session information:

```R
# At end of analysis
writeLines(capture.output(sessionInfo()), "sessionInfo.txt")
```

### Docker Support (Optional)

For complete reproducibility, consider using Docker:

```dockerfile
FROM rocker/tidyverse:4.2
RUN apt-get update && apt-get install -y python3-pip
COPY requirements.txt .
RUN pip3 install -r requirements.txt
# ... additional setup ...
```

---
