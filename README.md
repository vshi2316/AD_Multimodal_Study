# AD Multimodal Study

This repository contains the analysis code for multimodal subtype discovery, prognostic modeling, external cohort evaluation, longitudinal imaging validation, and AI-versus-clinician benchmarking in Alzheimer's disease research.

## Overview

The manuscript-aligned workflow addresses five linked components:

1. Shared preprocessing and multimodal feature integration in the discovery cohort
2. Latent subtype discovery and discovery-stage subtype characterization
3. Discovery-stage prognostic modeling
4. Independent holdout AI-versus-clinician benchmarking in ADNI
5. External contextualization across A4, AIBL, and HABS, followed by final evidence synthesis

Although the scientific workflow is branch-based, the current public repository keeps most scripts in the repository root for convenience, with the AI benchmarking workflow under `AI_vs_Clinician_Analysis/`.

## Current Repository Layout

```text
AD_Multimodal_Study/
├── preprocess_APOE.py
├── preprocess_CSF.py
├── preprocess_Clinical.py
├── preprocess_sMRI.py
├── preprocess_PET.py
├── create_outcome.py
├── Cohort Integration.py
├── vae_clustering.py
├── cross_cohort_analysis.py
├── biomarker_validation.py
├── differential_analysis.R
├── predictive_modeling.R
├── cluster_signatures.R
├── conversion_differential.R
├── cluster_validation.R
├── cross_modal_validation.R
├── habs_validation.R
├── shap_analysis.R
├── evidence_synthesis.R
├── ADNI_discovery.R
├── AIBL_Validation.R
├── A4_validation.R
├── neuroimaging_endotypes.R
├── requirements.txt
├── LICENSE
└── AI_vs_Clinician_Analysis/
    ├── Prepare Test.R
    ├── AI Prediction.py
    ├── Expert Assessment Workflow.R
    └── AI vs Expert Comparison Analysis
Manuscript-Aligned Workflow
A. Shared discovery input preparation
These scripts create the harmonized multimodal discovery inputs:

preprocess_APOE.py
preprocess_CSF.py
preprocess_Clinical.py
preprocess_sMRI.py
create_outcome.py
Cohort Integration.py
vae_clustering.py
Key expected outputs include discovery-ready integrated matrices, imputation and scaling artifacts, latent representations, subtype assignments, VAE summaries, and subtype centroids.

B. Discovery subtype characterization
These scripts characterize the discovered subtypes biologically, clinically, and longitudinally:

biomarker_validation.py
cluster_signatures.R
conversion_differential.R
cluster_validation.R
cross_modal_validation.R
neuroimaging_endotypes.R
This branch supports subtype biology, conversion heterogeneity, stability analyses, longitudinal MRI or BSI context, and structural endotype characterization.

C. Discovery prognostic modeling
predictive_modeling.R
This script supports discovery-stage conversion prediction benchmarking.

D. Independent holdout AI-versus-clinician benchmarking
AI_vs_Clinician_Analysis/Prepare Test.R
AI_vs_Clinician_Analysis/AI Prediction.py
AI_vs_Clinician_Analysis/Expert Assessment Workflow.R
AI_vs_Clinician_Analysis/AI vs Expert Comparison Analysis
This branch implements the frozen holdout benchmark used for AI-versus-clinician comparison.

E. External contextualization
A4 and AIBL direct subtype transportability
cross_cohort_analysis.py --cohort_name AIBL
AIBL_Validation.R
cross_cohort_analysis.py --cohort_name A4
A4_validation.R
HABS framework adaptation
habs_validation.R
shap_analysis.R
F. Final manuscript-facing synthesis
evidence_synthesis.R
This script should be run only after the discovery, holdout, and external contextualization branches have finished.

Scripts Retained in the Repository but Not Part of the Primary Manuscript Path
The following scripts are retained for traceability or exploratory back-compatibility, but they are not part of the canonical primary manuscript workflow:

preprocess_PET.py
differential_analysis.R
ADNI_discovery.R
They should not be treated as required steps for reproducing the final manuscript-aligned analysis path.

Data Access
This repository does not redistribute participant-level cohort data.

Data access must be obtained directly from the original sources and used under the relevant data use agreements, including:

Alzheimer's Disease Neuroimaging Initiative (ADNI)
A4 Study
Australian Imaging, Biomarker and Lifestyle Study (AIBL)
Harvard Aging Brain Study (HABS)
Fox Laboratory longitudinal BSI-derived imaging outputs, where applicable
The code is licensed separately from the cohort datasets. No data-use rights are conveyed by this repository license.

Software Environment
Python
Recommended Python version:

Python 3.10 or later
Install Python dependencies:

pip install -r requirements.txt
R
Recommended R version:

R 4.2 or later
The R scripts rely on the following packages across the repository:

optparse
dplyr
tidyr
ggplot2
jsonlite
stringr
randomForest
pROC
caret
mice
glmnet
xgboost
corrplot
ResourceSelection
limma
ggrepel
pheatmap
RColorBrewer
data.table
survival
survminer
lme4
lmerTest
emmeans
cluster
mclust
logistf
PRROC
tidyverse
readr
readxl
writexl
patchwork
multcomp
In addition, some scripts require Bioconductor packages:

ConsensusClusterPlus
limma
A minimal installation example is:

install.packages(c(
  "optparse", "dplyr", "tidyr", "ggplot2", "jsonlite", "stringr",
  "randomForest", "pROC", "caret", "mice", "glmnet", "xgboost",
  "corrplot", "ResourceSelection", "ggrepel", "pheatmap", "RColorBrewer",
  "data.table", "survival", "survminer", "lme4", "lmerTest", "emmeans",
  "cluster", "mclust", "logistf", "PRROC", "tidyverse", "readr",
  "readxl", "writexl", "patchwork", "multcomp"
))

if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
BiocManager::install(c("ConsensusClusterPlus", "limma"))
Important Repository Notes
AI_vs_Clinician_Analysis/AI vs Expert Comparison Analysis may not carry a file extension. Keep the filename exactly as stored in the repository unless you intentionally rename it everywhere.
predictive_modeling.R currently uses multiple imputation by chained equations and carries forward one completed dataset rather than pooled Rubin-style inference.
habs_validation.R is used for framework adaptation in HABS rather than direct subtype transportability.
cross_cohort_analysis.py is used for direct latent projection and subtype transportability in A4 and AIBL.
evidence_synthesis.R is the final manuscript-facing aggregation step and should not be run first.
Representative Outputs
Representative outputs across the workflow include:

subtype assignments
latent representations
VAE summary files
subtype centroid files
subtype biomarker summaries
conversion differential tables
stability and clustering validation outputs
longitudinal MRI or BSI summaries
discovery predictive modeling results
holdout AI prediction outputs
expert benchmarking comparison outputs
A4 and AIBL transportability summaries
HABS validation summaries
final manuscript-facing synthesis tables
Reproducibility Notes
Discovery subtype identification is performed in the discovery cohort and then reused across downstream analyses.
A4 and AIBL are treated as direct subtype transportability analyses.
HABS is treated as framework adaptation rather than direct subtype transfer.
Final manuscript-facing summary outputs should be generated only after all required upstream branches are complete.
Citation
If you use this repository, please cite the associated manuscript together with the originating cohort studies.

License
This repository is released under the MIT License. See LICENSE.
