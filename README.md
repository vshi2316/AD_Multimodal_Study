A comprehensive computational pipeline for discovering biologically distinct Alzheimer's disease subtypes through multimodal deep learning, validated across four independent cohorts (ADNI, AIBL, HABS, A4).

---

## Overview

This repository implements a novel framework for identifying Alzheimer's disease subtypes using:

- **Multimodal Data Integration**: CSF biomarkers (p-tau181, Aβ42/Aβ40 ratio), APOE ε4 genotype, MMSE, age, and sex
- **Deep Learning Clustering**: 6-domain Variational Autoencoder (VAE) with β-annealing for unsupervised subtype discovery
- **Multi-Cohort Validation**: External validation in A4 (primary), AIBL (exploratory), and HABS (real-world generalizability)
- **AI vs Expert Comparison**: Rigorous comparison with 5 neurologists on independent test set (n=196)
- **Meta-Analysis**: Random-effects meta-analysis across validation cohorts

### Key Features

- 22-step end-to-end analysis pipeline + 4-step AI vs Expert comparison module
- Strict data leakage prevention with "frozen pipeline" strategy
- Bootstrap stability assessment (100 iterations) with PAC, ARI, and Jaccard Index
- Publication-ready visualizations (300-600 DPI)

---

## Study Design

The study comprises four phases as described in Methods 2.1:

| Phase | Cohort | N | Purpose |
|-------|--------|---|---------|
| Discovery | ADNI | Variable | VAE training, subtype identification |
| Primary Validation | A4 | 1,147 | Cognitively normal, Aβ-PET positive |
| Exploratory Validation | AIBL | 48 | Independent replication |
| Real-world Validation | HABS | 4,134 | Generalizability, plasma p-tau217 |
| Clinical Comparison | ADNI (independent) | 196 | AI vs 5 neurologists |

---

## Repository Structure

```
AD_Multimodal_Study/
│
├── Data Preprocessing (Steps 1-7) ─────────────────────── Python
│   ├── step1_preprocess_APOE.py              # APOE ε4 genotype extraction
│   ├── step2_preprocess_CSF.py               # CSF biomarkers (p-tau181, Aβ42/Aβ40)
│   ├── step3_preprocess_Clinical.py          # Clinical scores (MMSE, ADAS-Cog, CDR-SB)
│   ├── step4_preprocess_sMRI.py              # Structural MRI (FreeSurfer 6.0)
│   ├── step5_preprocess_PET.py               # PET imaging quantification
│   ├── step6_create_outcome.py               # AD conversion outcomes
│   └── step7_integrate_cohorts.py            # Multimodal integration + frozen pipeline
│
├── VAE Clustering (Steps 8-9) ─────────────────────────── Python
│   ├── step8_vae_clustering.py               # 6-domain β-VAE (PyTorch 1.12.0)
│   ├── step9A_cross_cohort_analysis.py       # Stability: PAC, ARI, Jaccard
│   └── step9B_biomarker_validation.py        # Biomarker validation
│
├── Statistical Analysis (Steps 10-13) ─────────────────── R
│   ├── step10_differential_analysis.R        # Limma differential analysis
│   ├── step11_predictive_modeling.R          # Multi-algorithm ML models
│   ├── step12_cluster_signatures.R           # Cluster signature visualization
│   └── step13_conversion_differential.R      # Converter vs non-converter analysis
│
├── Cluster Validation (Steps 14-15) ───────────────────── R
│   ├── step14_cluster_validation.R           # Consensus clustering, bootstrap
│   └── step15_cross_modal_validation.R       # CSF & MRI cross-validation
│
├── External Validation (Steps 16-18) ──────────────────── R
│   ├── step16_habs_validation.R              # HABS: Firth regression, p-tau217
│   ├── step17_shap_analysis.R                # SHAP explainability
│   └── step18_meta_analysis.R                # Random-effects meta-analysis
│
├── Discovery-Validation Chain (Steps 19-21) ───────────── R
│   ├── step19_adni_discovery.R               # ADNI discovery classifier
│   ├── step20_aibl_preprocessing.R           # AIBL validation
│   └── step21_a4_validation.R                # A4 large-sample validation
│
├── Biological Characterization (Step 22) ──────────────── R
│   └── step22_neuroimaging_endotypes.R       # Clinical-MRI heterogeneity (η²)
│
├── AI vs Expert Comparison ────────────────────────────── R/Python
│   └── AI_vs_Clinician_Analysis/
│       ├── Step 1 Prepare Test.R             # Independent test set (n=196)
│       ├── Step 2 AI Prediction.py           # Frozen AI pipeline prediction
│       ├── Step 3 Expert Assessment Workflow.R   # Expert data collection
│       └── Step 4 AI vs Expert Comparison Analysis.R  # DeLong, NRI, IDI, DCA
│
└── Documentation
    ├── README.md
    ├── LICENSE
    ├── requirements.txt
```

---

## Prerequisites

### Software Requirements

| Software | Version | Purpose |
|----------|---------|---------|
| Python | ≥3.9 | Preprocessing, VAE |
| R | ≥4.3 | Statistical analysis |
| PyTorch | 1.12.0 | Deep learning (Methods 2.9) |

### Python Dependencies

```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
torch>=1.12.0
matplotlib>=3.5.0
seaborn>=0.11.0
scipy>=1.7.0
```

### R Packages

```r
# Core
tidyverse, dplyr, readr

# Statistical
survival, survminer, tableone, mice

# Machine Learning
caret, randomForest, glmnet, xgboost

# Clustering
cluster, mclust, ConsensusClusterPlus

# Meta-analysis
meta, metafor

# Visualization
ggplot2, pheatmap, patchwork, RColorBrewer

# AI vs Expert
pROC, irr, BlandAltmanLeh, dcurves
```

---

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/vshi2316/AD_Multimodal_Study.git
cd AD_Multimodal_Study
```

### 2. Python Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
pip install -r requirements_pytorch.txt
```

### 3. R Packages

```r
# CRAN packages
install.packages(c(
  "tidyverse", "survival", "survminer", "tableone", "mice",
  "caret", "randomForest", "glmnet", "xgboost",
  "cluster", "mclust", "meta", "metafor",
  "ggplot2", "pheatmap", "patchwork", "RColorBrewer",
  "pROC", "irr", "optparse"
))

# Bioconductor
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
BiocManager::install(c("limma", "ConsensusClusterPlus"))
```

---

## Data Requirements

### Input Data Format

All input files should be CSV format with subject ID column.

#### 1. APOE Genotyping
- **Required**: `ID`, `GENOTYPE` (e.g., E3/E4)
- **Output**: `APOE4_STATUS` (0/1), `APOE4_DOSAGE` (0/1/2)

#### 2. CSF Biomarkers
- **Required**: `ID`, `PTAU181`, `ABETA42`, `ABETA40`
- **Derived**: `ABETA42_ABETA40_RATIO`

#### 3. Clinical Data
- **Required**: `ID`, `AGE`, `SEX`, `MMSE`
- **Optional**: `ADAS13`, `CDRSB`, `FAQTOTAL`, `EDUCATION`

#### 4. Structural MRI (FreeSurfer 6.0)
- **Required**: `ID`, regional volumes/thickness (ST* columns)
- **Processing**: ICV-corrected, log-transformed

#### 5. Outcome Data
- **Required**: `ID`, `AD_Conversion` (0/1)
- **Optional**: `Time_to_Event`, `Followup_Months`

### Data Organization

```
data/
├── ADNI/
│   ├── APOE/ApoE_Genotyping_Results.csv
│   ├── CSF/UPENN_CSF_Biomarkers.csv
│   ├── LINES/Clinical_Assessments.csv
│   ├── sMRI/FreeSurfer_ROI.csv
│   └── Outcomes/AD_Conversion.csv
├── A4/
│   └── A4_Baseline_Integrated.csv
├── AIBL/
│   └── AIBL_Baseline_Integrated.csv
└── HABS/
    └── HABS_Baseline_Integrated.csv
```

---

## Usage

### Phase 1: Data Preprocessing (Python, Steps 1-7)

```bash
# Step 1: APOE genotype preprocessing
python step1_preprocess_APOE.py --input_file ./data/APOE.csv --output_dir ./processed

# Step 2: CSF biomarkers preprocessing
python step2_preprocess_CSF.py --input_file ./data/CSF.csv --output_dir ./processed

# Step 3: Clinical data preprocessing
python step3_preprocess_Clinical.py --input_file ./data/Clinical.csv --output_dir ./processed

# Step 4: Structural MRI preprocessing
python step4_preprocess_sMRI.py --input_file ./data/MRI.csv --output_dir ./processed

# Step 5: PET imaging preprocessing
python step5_preprocess_PET.py --input_file ./data/PET.csv --output_dir ./processed

# Step 6: Outcome variable creation
python step6_create_outcome.py --input_file ./data/Outcomes.csv --output_dir ./processed

# Step 7: Cohort integration with frozen pipeline (Methods 2.6)
python step7_integrate_cohorts.py \
    --data_dir ./processed \
    --output_dir ./processed \
    --test_size 196 \
    --mice_iterations 15
```

**Output**: 
- `ADNI_Discovery_Scaled.csv` (training set)
- `ADNI_Test_Scaled.csv` (independent test set)
- `pipeline_imputer.pkl`, `pipeline_scaler.pkl` (frozen processors)

### Phase 2: VAE Clustering (Python, Steps 8-9)

```bash
# Step 8: 6-domain β-VAE clustering (Methods 2.3)
python step8_vae_clustering.py \
    --input_file ./processed/ADNI_Discovery_Scaled.csv \
    --output_dir ./results \
    --n_clusters 3 \
    --latent_dim 10 \
    --epochs 200

# Step 9A: Stability assessment - PAC, ARI, Jaccard (Methods 2.7)
python step9A_cross_cohort_analysis.py \
    --integrated_file ./processed/ADNI_Discovery_Scaled.csv \
    --latent_file ./results/latent_encoded.csv \
    --output_dir ./results \
    --n_bootstrap 100

# Step 9B: Biomarker validation
python step9B_biomarker_validation.py \
    --latent_file ./results/latent_encoded.csv \
    --output_dir ./results
```

**Output**:
- `cluster_results.csv` (ID, Cluster_Labels, AD_Conversion)
- `latent_encoded.csv` (latent representations)
- `vae_model.pth` (trained PyTorch model)
- `Consensus_Matrix.png`, `cluster_evaluation.png`

### Phase 3: Statistical Analysis (R, Steps 10-13)

```r
# Step 10: Differential analysis (limma)
source("step10_differential_analysis.R")

# Step 11: Predictive modeling (RF, SVM, XGBoost)
source("step11_predictive_modeling.R")

# Step 12: Cluster signature visualization
source("step12_cluster_signatures.R")

# Step 13: Converter vs non-converter analysis
source("step13_conversion_differential.R")
```

### Phase 4: Cluster Validation (R, Steps 14-15)

```r
# Step 14: Consensus clustering and bootstrap validation
source("step14_cluster_validation.R")

# Step 15: Cross-modal validation (CSF & MRI)
source("step15_cross_modal_validation.R")
```

### Phase 5: External Validation (R, Steps 16-18)

```r
# Step 16: HABS validation with Firth regression and p-tau217
source("step16_habs_validation.R")

# Step 17: SHAP explainability analysis
source("step17_shap_analysis.R")

# Step 18: Random-effects meta-analysis (Methods 2.7)
source("step18_meta_analysis.R")
```

### Phase 6: Discovery-Validation Chain (R, Steps 19-21)

```r
# Step 19: ADNI discovery classifier training
source("step19_adni_discovery.R")

# Step 20: AIBL external validation
source("step20_aibl_preprocessing.R")

# Step 21: A4 large-sample validation (Methods 2.5)
source("step21_a4_validation.R")
```

### Phase 7: Biological Characterization (R, Step 22)

```r
# Step 22: Neuroimaging endotypes and clinical-MRI heterogeneity (Methods 2.8)
source("step22_neuroimaging_endotypes.R")
```

### Phase 8: AI vs Expert Comparison (R/Python, Methods 2.6 & 2.9)

```bash
# Navigate to AI vs Clinician Analysis folder
cd AI_vs_Clinician_Analysis

# Step 1: Prepare independent test set (n=196 MCI patients)
Rscript "Step 1 Prepare Test.R" \
    --adni_dir ../data/ADNI \
    --train_file ../results/cluster_results.csv \
    --output_dir ./Test_Data \
    --target_n 196

# Step 2: AI prediction using frozen pipeline
python "Step 2 AI Prediction.py" \
    --test_file ./Test_Data/independent_test_set.csv \
    --model_dir ../results \
    --output_dir ./AI_Predictions

# Step 3: Expert assessment workflow (5 neurologists, 2-stage protocol)
Rscript "Step 3 Expert Assessment Workflow.R" \
    --test_file ./Test_Data/independent_test_set.csv \
    --output_dir ./Expert_Data

# Step 4: AI vs Expert comparison analysis
Rscript "Step 4 AI vs Expert Comparison Analysis.R" \
    --ai_file ./AI_Predictions/ai_predictions.csv \
    --expert_file ./Expert_Data/expert_assessments.csv \
    --output_dir ./Comparison_Results
```

**AI vs Expert Comparison Output**:
- ROC curves with DeLong test (2000 bootstrap iterations)
- Calibration plots with Hosmer-Lemeshow test
- Decision Curve Analysis (DCA)
- Bland-Altman plots
- NRI and IDI statistics
- ICC and Fleiss' Kappa for inter-rater reliability

---

## Output Files

### Key Outputs by Step

| Step | Script | Output | Description |
|------|--------|--------|-------------|
| 7 | step7_integrate_cohorts.py | `pipeline_*.pkl` | Frozen MICE imputer and StandardScaler |
| 8 | step8_vae_clustering.py | `cluster_results.csv` | Subtype assignments |
| 8 | step8_vae_clustering.py | `vae_model.pth` | Trained PyTorch VAE model |
| 9A | step9A_cross_cohort_analysis.py | `Consensus_Matrix.png` | Stability visualization |
| 14 | step14_cluster_validation.R | `Bootstrap_Stability.csv` | ARI, Jaccard, PAC metrics |
| 18 | step18_meta_analysis.R | `Forest_Plot.png` | Meta-analysis results |
| 22 | step22_neuroimaging_endotypes.R | `Subtype_Naming.csv` | HP, CD, TAD characterization |
| AI-4 | Step 4 AI vs Expert Comparison Analysis.R | `AI_vs_Expert_ROC.png` | Performance comparison |

---

## Methods Alignment

This codebase strictly implements the manuscript methods:

### Methods 2.3: VAE Architecture
- ✓ Encoder: 3 hidden layers (128→64→32→latent)
- ✓ 6 independent decoders (CSF p-tau181, Aβ ratio, MMSE, age, sex, APOE)
- ✓ β-annealing: 0.001→1.0
- ✓ Adam optimizer (lr=0.001, batch=64)
- ✓ Early stopping (patience=20)
- ✓ Latent dimension: 10 (grid search optimized)

### Methods 2.6: Data Leakage Prevention
- ✓ Frozen pipeline strategy
- ✓ MICE fitted on training only (15 iterations)
- ✓ StandardScaler fitted on training only
- ✓ Independent test set (n=196) physically sequestered

### Methods 2.7: Stability Assessment
- ✓ Bootstrap resampling (100 iterations)
- ✓ Jaccard Index and ARI
- ✓ Consensus Clustering with PAC (<0.05 threshold)
- ✓ Sample-level stability (>0.85 threshold)

### Methods 2.8: Neuroimaging Subtype Analysis
- ✓ Eta-squared (η²) effect sizes
- ✓ ANCOVA adjusting for MMSE and age
- ✓ Stage-independent verification
- ✓ W-score residual analysis

### Methods 2.9: AI vs Expert Comparison
- ✓ 2000 bootstrap iterations for CI
- ✓ DeLong test for AUC comparison
- ✓ Hosmer-Lemeshow calibration test
- ✓ Brier score for probabilistic accuracy
- ✓ ICC and Fleiss' Kappa for inter-rater reliability
- ✓ Bland-Altman plots for systematic bias
- ✓ NRI and IDI for incremental value
- ✓ Decision Curve Analysis (DCA)

---


## Reproducibility

### Random Seeds

All scripts use fixed random seeds:
- Python: `np.random.seed(42)`, `torch.manual_seed(42)`
- R: `set.seed(42)`

### Session Information

```r
# Save session info for reproducibility
writeLines(capture.output(sessionInfo()), "sessionInfo.txt")
```
---

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

Data used in preparation of this article were obtained from:
- Alzheimer's Disease Neuroimaging Initiative (ADNI)
- Anti-Amyloid Treatment in Asymptomatic AD (A4) Study
- Australian Imaging, Biomarkers and Lifestyle Study (AIBL)
- Harvard Aging Brain Study (HABS)
