# AD Multimodal Study

Code and analysis workflow for multimodal subtype discovery, prognostic modelling, external validation, and AI-versus-clinician benchmarking in Alzheimer's disease research.

## Overview

This repository contains the analysis scripts used to support a multimodal Alzheimer's disease study built around:

- modality-weighted variational autoencoder subtype discovery in ADNI
- multimodal MCI-to-AD prediction modelling
- independent ADNI holdout benchmarking against neurologists
- external projection to A4 and AIBL
- framework adaptation to HABS with plasma p-tau217
- longitudinal validation with Fox Lab boundary-shift integral measures

The repository mixes manuscript-aligned scripts and several legacy or intermediate scripts retained for traceability. The sections below identify the recommended execution order and the main caveats that should be understood before reuse.

## Repository layout

### Root analysis scripts

- `step1_preprocess_APOE.py`
- `step2_preprocess_CSF.py`
- `step3_preprocess_Clinical.py`
- `step4_preprocess_sMRI.py`
- `step5_preprocess_PET.py`
- `step6_create_outcome.py`
- `step7_Cohort Integration.py`
- `step8_vae_clustering.py`
- `step9A_cross_cohort_analysis.py`
- `step9B_biomarker_validation.py`
- `step10_differential_analysis.R`
- `step11_predictive_modeling.R`
- `step12_cluster_signatures.R`
- `step13_conversion_differential.R`
- `step14_cluster_validation.R`
- `step15_cross_modal_validation.R`
- `step16_habs_validation.R`
- `step17_shap_analysis.R`
- `step18_evidence_synthesis.R`
- `step19_ADNI_discovery.R`
- `step20_AIBL _Validation.R`
- `step21_A4_validation.R`
- `step22_neuroimaging_endotypes.R`

### AI-versus-clinician workflow

Directory: `AI_vs_Clinician_Analysis`

- `Step 1 Prepare Test.R`
- `Step 2 AI Prediction.py`
- `Step 3 Expert Assessment Workflow.R`
- `Step 4 AI vs Expert Comparison Analysis`

## Recommended analysis path

The manuscript-oriented workflow is:

1. `step1_preprocess_APOE.py`
2. `step2_preprocess_CSF.py`
3. `step3_preprocess_Clinical.py`
4. `step4_preprocess_sMRI.py`
5. `step6_create_outcome.py`
6. `step7_Cohort Integration.py`
7. `step8_vae_clustering.py`
8. `step9B_biomarker_validation.py`
9. `step10_differential_analysis.R`
10. `step11_predictive_modeling.R`
11. `step12_cluster_signatures.R`
12. `step13_conversion_differential.R`
13. `step14_cluster_validation.R`
14. `step15_cross_modal_validation.R`
15. `step16_habs_validation.R`
16. `step17_shap_analysis.R`
17. `step19_ADNI_discovery.R`
18. `step9A_cross_cohort_analysis.py`
19. `step20_AIBL _Validation.R`
20. `step21_A4_validation.R`
21. `step22_neuroimaging_endotypes.R`
22. `step18_evidence_synthesis.R`

The AI-versus-clinician branch should then be run separately:

1. `AI_vs_Clinician_Analysis\Step 1 Prepare Test.R`
2. `AI_vs_Clinician_Analysis\Step 2 AI Prediction.py`
3. `AI_vs_Clinician_Analysis\Step 3 Expert Assessment Workflow.R`
4. `AI_vs_Clinician_Analysis\Step 4 AI vs Expert Comparison Analysis`

## Data requirements

This repository does not redistribute cohort data. Access must be obtained directly from the source studies under their respective data-use agreements.

### Cohorts used

- ADNI
- A4
- AIBL
- HABS
- Fox Lab BSI longitudinal MRI measures

### Typical expected input files

Examples of filenames referenced in the code include:

- `Clinical_data.csv`
- `metabolites.csv`
- `RNA_plasma.csv`
- `subtype_assignments.csv`
- `latent_representations.csv`
- `vae_summary.json`
- `independent_test_set.csv`
- `HABS_Baseline_Integrated.csv`
- `AIBL_Baseline_Integrated.csv`
- `A4_Baseline_Integrated.csv`
- `FOXLABBSI_02Mar2026.csv`

Because the raw source files differ by cohort and local preprocessing, you should inspect each script's `--help` output before execution.

## Software environment

### Python

Recommended:

- Python 3.10+
- `pip install -r requirements.txt`

Python packages used in the repository include:

- `numpy`
- `pandas`
- `scikit-learn`
- `scipy`
- `matplotlib`
- `seaborn`
- `torch`
- `statsmodels`

### R

Recommended:

- R 4.2+

R packages used across scripts include:

- `optparse`
- `dplyr`
- `tidyr`
- `ggplot2`
- `survival`
- `survminer`
- `caret`
- `mice`
- `glmnet`
- `randomForest`
- `xgboost`
- `pROC`
- `ResourceSelection`
- `ConsensusClusterPlus`
- `cluster`
- `mclust`
- `data.table`
- `lme4`
- `lmerTest`
- `emmeans`
- `logistf`
- `PRROC`
- `jsonlite`
- `patchwork`
- `multcomp`
- `corrplot`
- `psych`
- `irr`
- `writexl`
- `readxl`
- `kernelshap`

## What each key script does

### Discovery and subtype derivation

- `step7_Cohort Integration.py` creates discovery and holdout cohorts using a frozen preprocessing strategy.
- `step8_vae_clustering.py` trains the modality-weighted VAE, generates latent representations, and performs K-means clustering.
- `step14_cluster_validation.R` evaluates conversion gradients, consensus clustering, bootstrap stability, and subtype reproducibility.
- `step22_neuroimaging_endotypes.R` maps subtype-discriminative MRI features to anatomical and network-level signatures.

### Biomarker and predictive modelling

- `step9B_biomarker_validation.py` evaluates AT-composite and neuroinflammation-related CSF signatures.
- `step10_differential_analysis.R` compares subtype-level features using standardized mean differences and FDR control.
- `step11_predictive_modeling.R` trains discovery-stage supervised models and exports the best-performing configuration.
- `step12_cluster_signatures.R` visualizes multimodal subtype signatures.
- `step13_conversion_differential.R` performs converter-versus-non-converter differential analysis.

### Longitudinal and external validation

- `step15_cross_modal_validation.R` validates subtype differences using longitudinal BSI trajectories and mixed-effects models.
- `step9A_cross_cohort_analysis.py` projects external cohorts into the ADNI latent space and assigns subtypes by centroid distance.
- `step20_AIBL _Validation.R` evaluates projected subtype performance in AIBL.
- `step21_A4_validation.R` evaluates projected subtype performance in A4.
- `step16_habs_validation.R` fits cohort-specific Firth logistic regression models in HABS.
- `step17_shap_analysis.R` computes model-agnostic SHAP values for the HABS complete model.

### AI-versus-clinician comparison

- `AI_vs_Clinician_Analysis\Step 1 Prepare Test.R` builds the independent ADNI test set.
- `AI_vs_Clinician_Analysis\Step 2 AI Prediction.py` runs the frozen AI holdout pipeline.
- `AI_vs_Clinician_Analysis\Step 3 Expert Assessment Workflow.R` generates neurologist assessment forms and templates.
- `AI_vs_Clinician_Analysis\Step 4 AI vs Expert Comparison Analysis` computes ROC, DeLong, calibration, reliability, NRI/IDI, and decision-curve outputs.


## Example commands

### Python examples

```bash
python step8_vae_clustering.py --input_dir ./processed_data --output_dir ./vae_output
python step9A_cross_cohort_analysis.py --external_file ./A4_Baseline_Integrated.csv --vae_dir ./vae_output --output_dir ./a4_projection --cohort_name A4
python AI_vs_Clinician_Analysis/"Step 2 AI Prediction.py" --base_dir ./AI_vs_Clinician_Analysis --output_dir ./AI_vs_Clinician_Analysis/results
```

### R examples

```bash
Rscript step11_predictive_modeling.R --vae_dir ./vae_output --data_dir ./processed_data --output_dir ./step11_results
Rscript step16_habs_validation.R --habs_file ./HABS_Baseline_Integrated.csv --output_dir ./step16_results
Rscript "step20_AIBL _Validation.R" --projection_file ./aibl_projection/AIBL_projected_subtypes.csv --baseline_file ./AIBL_Baseline_Integrated.csv --output_dir ./step20_results
```

## Outputs to expect

Representative outputs produced by the pipeline include:

- subtype assignments and latent representations
- VAE summary JSON and subtype centroids
- feature-importance tables and model-comparison CSV files
- calibration, ROC, and confusion-matrix plots
- BSI longitudinal analysis tables and figures
- external-cohort projected subtype files
- HABS performance summaries, decision curves, and SHAP outputs
- HABS, AIBL, and A4 manuscript summary CSV files used for cross-cohort synthesis
- Cox time-source metadata for discovery survival analyses
- AI-versus-clinician comparison tables, ROC curves, calibration plots, and reliability statistics

## Reproducibility and reporting

- Set seeds are embedded in the scripts where applicable.
- Cohort-specific preprocessing assumptions remain script-dependent and should be documented in any derivative publication.
- Before manuscript submission, verify that the text matches the executable code for endpoint definition, imputation strategy, and time-to-event handling.

## Citation

If you use this repository, please cite the associated manuscript and the originating cohort studies.

## License

This repository is released under the MIT License. See `LICENSE`.
