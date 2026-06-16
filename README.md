# AD Multimodal Study

This repository contains the manuscript-aligned analysis code for multimodal subtype discovery, prognostic modeling, independent AI-versus-neurologist benchmarking, human-AI second-reader workflow evaluation, reduced-feature AIBL external validation, external cohort framework-extension analyses, and final evidence synthesis in Alzheimer's disease research.

The repository does not contain participant-level data. All cohort data must be obtained from the original study repositories under the relevant data-use agreements.

## Scientific Scope

The current manuscript-facing workflow has five linked aims:

1. Build harmonized multimodal ADNI discovery inputs from clinical, CSF, APOE, and structural MRI data.
2. Derive and characterize VAE-based latent structural profiles as discovery-stage heterogeneity layers.
3. Train a frozen multimodal AI risk model and benchmark it against masked neurologist assessments in an independent ADNI holdout cohort.
4. Evaluate a prespecified human-AI Rule C workflow in which the AI model acts as a specificity-oriented second reader for expert gray-zone cases.
5. Add external evidence through AIBL reduced-feature external validation and separate A4/HABS framework-extension analyses.

The final manuscript interpretation is intentionally hierarchical:

- **Primary clinical utility evidence:** independent ADNI holdout AI-versus-neurologist benchmark plus prespecified Rule C gray-zone second-reader workflow.
- **Secondary external validation:** AIBL reduced-feature validation of an ADNI-trained clinical model using harmonizable variables.
- **Framework-extension evidence:** A4 preclinical/trial-screening transportability and HABS cohort-specific biomarker-enriched adaptability.
- **Supporting structural context:** VAE latent profiles and BSI longitudinal atrophy analyses.

The AIBL reduced-feature analysis externally validates a prespecified clinical model. It is not presented as direct external validation of the full multimodal MRI/CSF/VAE AI pipeline, because same-pipeline multimodal feature equivalence was not available in AIBL at the time of this analysis.

## Repository Layout

```text
AD_Multimodal_Study/
├── 0_shared_input_preparation/
│   ├── Cohort Integration.py
│   ├── Create_outcome.py
│   ├── Preprocess_APOE.py
│   ├── Preprocess_Clinical.py
│   ├── Preprocess_CSF.py
│   └── Preprocess_sMRI.py
├── 1_discovery_subtype_model/
│   └── vae_clustering.py
├── 2_discovery_characterization/
│   ├── ADNI_discovery.R
│   ├── Biomarker_validation.py
│   ├── Cluster_signatures.R
│   ├── Cluster_validation.R
│   ├── Conversion_differential.R
│   ├── Cross_modal_validation.R
│   ├── Neuroimaging_endotypes.R
│   └── Predictive_modeling.R
├── 3_AI_vs_Clinician_Analysis/
│   ├── Prepare Test.R
│   ├── AI Prediction.py
│   ├── Expert Assessment Workflow.R
│   ├── AI vs Expert Comparison Analysis.R
│   ├── Human_AI_RuleC_Workflow_Extension.R
│   └── Human_AI_RuleC_Posthoc_Refinements.R
├── 4_external_contextualization/
│   ├── Cross_cohort_analysis.py
│   ├── A4_validation.R
│   ├── AIBL _Validation.R
│   ├── AIBL_Feasibility_Gate.R
│   ├── AIBL_Reduced_Feature_External_Validation.R
│   ├── HABS_validation.R
│   └── SHAP_analysis.R
├── 5_final_evidence_synthesis/
│   └── Evidence_synthesis.R
├── requirements.txt
├── LICENSE
└── README.md
```

## Data Access and Expected Local Data Organization

This repository does not redistribute ADNI, A4, AIBL, HABS, Fox Lab BSI outputs, or expert-reader data.

Users must obtain access directly from:

- Alzheimer's Disease Neuroimaging Initiative (ADNI)
- Anti-Amyloid Treatment in Asymptomatic Alzheimer's Disease (A4) Study
- Australian Imaging, Biomarker and Lifestyle Flagship Study (AIBL)
- Harvard Aging Brain Study (HABS)
- Fox Lab longitudinal BSI-derived imaging outputs, where applicable

The code is licensed separately from the cohort datasets. No data-use rights are conveyed by this repository license.

Several scripts use command-line arguments, so the exact local data structure can be customized. The default examples assume a project-level data root with folders such as:

```text
<data_root>/
├── ADNI_Raw_Data/
├── ADNI原始数据/
├── Phase1_ADNI_Discovery/
├── AI_vs_Clinician_Test/
├── aibl_19Sep2019/
│   └── Data_extract_3.3.0/
├── AIBL验证/
├── step11_results/
├── step12_results/
├── step14_results/
├── step16_results/
├── step20_results/
├── step21_results/
├── step22_results/
├── longitudinal_bsi_validation/
├── PET_cohort_analysis/
└── vae_revised_output/
```

If your data are stored elsewhere, pass explicit paths using the script arguments shown below.

## Software Environment

### Python

Recommended Python version: 3.10 or later.

Install Python dependencies:

```bash
pip install -r requirements.txt
```

The Python workflow uses common scientific packages, including `numpy`, `pandas`, `scikit-learn`, `tensorflow`, `keras`, `matplotlib`, `seaborn`, and `scipy`.

### R

Recommended R version: 4.2 or later.

Install CRAN dependencies:

```r
install.packages(c(
  "optparse", "dplyr", "tidyr", "ggplot2", "jsonlite", "stringr",
  "randomForest", "pROC", "caret", "mice", "glmnet", "xgboost",
  "corrplot", "ResourceSelection", "ggrepel", "pheatmap", "RColorBrewer",
  "data.table", "survival", "survminer", "lme4", "lmerTest", "emmeans",
  "cluster", "mclust", "logistf", "PRROC", "tidyverse", "readr",
  "readxl", "writexl", "patchwork", "multcomp", "broom", "purrr",
  "tibble", "scales", "irr", "gridExtra", "lubridate"
))
```

Install Bioconductor dependencies:

```r
if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
BiocManager::install(c("ConsensusClusterPlus", "limma"))
```

Some scripts attempt to install missing packages automatically. For reproducible manuscript reruns, pre-installing dependencies is recommended.

## Workflow Overview

The workflow is modular. Run only the branches required for the analysis you want to reproduce.

### 0. Shared Input Preparation

These scripts harmonize raw cohort inputs into analysis-ready tables.

```bash
python 0_shared_input_preparation/Preprocess_APOE.py \
  --input_file ./ADNI_Raw_Data/APOE/ApoE_Genotyping_Results.csv \
  --output_file ./processed_data/APOE_genetics.csv \
  --output_dir ./processed_data

python 0_shared_input_preparation/Preprocess_CSF.py
python 0_shared_input_preparation/Preprocess_Clinical.py
python 0_shared_input_preparation/Preprocess_sMRI.py
python 0_shared_input_preparation/Create_outcome.py
python "0_shared_input_preparation/Cohort Integration.py"
```

Expected outputs include harmonized APOE, CSF, clinical, structural MRI, outcome, and integrated multimodal feature tables.

### 1. Discovery Latent Subtype Model

The VAE discovery model learns latent structural/multimodal representations in the ADNI discovery cohort.

```bash
python 1_discovery_subtype_model/vae_clustering.py
```

Representative outputs include:

- latent representations
- VAE reconstruction summaries
- subtype assignments
- subtype centroids
- model configuration artifacts

### 2. Discovery Characterization and Prediction

Discovery-stage scripts characterize the latent subgroups biologically, clinically, structurally, and longitudinally.

```bash
Rscript 2_discovery_characterization/ADNI_discovery.R \
  --subtype_file subtype_assignments.csv \
  --clinical_file Clinical_data.csv \
  --output_dir ./step19_results

Rscript 2_discovery_characterization/Cluster_validation.R
Rscript 2_discovery_characterization/Cluster_signatures.R
Rscript 2_discovery_characterization/Conversion_differential.R
Rscript 2_discovery_characterization/Cross_modal_validation.R
Rscript 2_discovery_characterization/Neuroimaging_endotypes.R
Rscript 2_discovery_characterization/Predictive_modeling.R
python 2_discovery_characterization/Biomarker_validation.py
```

These analyses support discovery-stage heterogeneity, conversion gradients, stability testing, MRI/network characterization, biomarker context, and discovery predictive modeling.

### 3. Independent ADNI Holdout AI-Versus-Clinician Benchmark

This branch builds the independent holdout test set, generates frozen AI predictions, collects or formats expert predictions, and compares AI performance with expert readers.

```bash
Rscript "3_AI_vs_Clinician_Analysis/Prepare Test.R"
python "3_AI_vs_Clinician_Analysis/AI Prediction.py"
Rscript "3_AI_vs_Clinician_Analysis/Expert Assessment Workflow.R"
Rscript "3_AI_vs_Clinician_Analysis/AI vs Expert Comparison Analysis.R"
```

Expected files for the new human-AI workflow extension include:

```text
AI_vs_Clinician_Test/independent_test_set.csv
AI_vs_Clinician_Test/AI_Predictions_Final.csv
AI_vs_Clinician_Test/AI_per_patient_predictions.csv
AI_vs_Clinician_Test/Expert_Predictions_Long.csv
```

The AI prediction script writes both `AI_Predictions_Final.csv` and `AI_per_patient_predictions.csv` for backward compatibility.

### 3A. Human-AI Rule C Workflow Extension

`Human_AI_RuleC_Workflow_Extension.R` implements the primary manuscript-facing human-AI extension analysis:

- expert Stage 2 gray-zone distribution check
- no-refitting AI-expert probability integration
- prespecified Rule A, Rule B, and Rule C workflow simulations
- primary Rule C gray-zone second-reader analysis
- case-level AI/expert discordance groups
- AUC, DeLong tests, confusion matrices, PPV/NPV, and accuracy metrics
- categorical NRI and IDI
- decision-curve net benefit and resource translation
- threshold sweep for descriptive operating points
- optional BSI and VAE mechanism/context layers if linkable files are present

Primary manuscript interpretation should focus on Rule C, not post-hoc fitted stacking.

Run with default relative paths:

```bash
Rscript "3_AI_vs_Clinician_Analysis/Human_AI_RuleC_Workflow_Extension.R" \
  --data_root . \
  --output_dir ./3_AI_vs_Clinician_Analysis/Q1_Human_AI_Extension \
  --ai_file ./AI_vs_Clinician_Test/AI_Predictions_Final.csv \
  --expert_file ./AI_vs_Clinician_Test/Expert_Predictions_Long.csv \
  --test_file ./AI_vs_Clinician_Test/independent_test_set.csv \
  --n_bootstrap 2000 \
  --cv_repeats 200
```

If your AI-vs-clinician data are in a different folder, set `--data_root` and pass file paths relative to that root.

Main outputs include:

```text
3_AI_vs_Clinician_Analysis/Q1_Human_AI_Extension/
├── 00_case_level_master.csv
├── 00_case_level_master_with_combined_predictions.csv
├── 00_gray_zone_distribution_check.csv
├── 01_performance_summary_primary_thresholds.csv
├── 02_delong_auc_tests_primary_leakage_safe.csv
├── 03_case_level_discordance.csv
├── 03_discordance_feature_comparison.csv
├── 03_adjusted_discordance_models_mmse_adjusted.csv
├── 04_workflow_metrics_vs_expert_stage2.csv
├── 05_nri_idi_summary.csv
├── 05_categorical_nri_primary.csv
├── 06_decision_curve_net_benefit.csv
├── 06_dca_resource_translation_per_100.csv
├── 07_threshold_sweep_metrics_descriptive_sensitivity_only.csv
├── 07_clinical_operating_points_descriptive_not_primary.csv
├── README_Q1_extension_outputs.txt
└── figures/
    ├── Figure_Q1_Gray_Zone_Distribution_Check.png
    ├── Figure_Q1_Performance_Human_AI_Workflows.png
    ├── Figure_Q1_Workflow_Error_Profile.png
    ├── Figure_Q1_Decision_Curve_Human_AI.png
    └── Figure_Q1_Threshold_Sweep.png
```

Optional BSI and VAE outputs are generated only when linkable candidate files are available under the supplied `--data_root`, such as:

```text
longitudinal_bsi_validation/individual_bsi_slopes.csv
longitudinal_bsi_validation/bsi_longitudinal_merged.csv
vae_revised_output/latent_representations.csv
vae_revised_output/subtype_assignments.csv
```

### 3B. Human-AI Rule C Post-Hoc Refinements

Run this after `Human_AI_RuleC_Workflow_Extension.R`.

```bash
Rscript "3_AI_vs_Clinician_Analysis/Human_AI_RuleC_Posthoc_Refinements.R" \
  --rulec_dir ./3_AI_vs_Clinician_Analysis/Q1_Human_AI_Extension
```

This script adds three manuscript-refinement analyses:

- z-scored adjusted discordance models
- paired FP/FN comparison for Rule C versus expert Stage 2
- bootstrap confidence intervals for DCA net benefit

Outputs are written by default to:

```text
3_AI_vs_Clinician_Analysis/Q1_Human_AI_Extension/posthoc_refinements/
├── 10_adjusted_discordance_zscore_results.csv
├── 11_ruleC_fp_fn_paired_comparison.csv
├── 12_dca_net_benefit_bootstrap_ci_curve.csv
├── 12_dca_net_benefit_bootstrap_ci_key_thresholds.csv
└── README_posthoc_refinements.txt
```

### 4. External Cohort Framework-Extension Analyses

A4, AIBL VAE transfer, and HABS analyses are retained as external framework-extension components. They should not be described as uniform validation of one fixed full multimodal model.

```bash
python 4_external_contextualization/Cross_cohort_analysis.py --cohort_name AIBL
Rscript "4_external_contextualization/AIBL _Validation.R"

python 4_external_contextualization/Cross_cohort_analysis.py --cohort_name A4
Rscript 4_external_contextualization/A4_validation.R

Rscript 4_external_contextualization/HABS_validation.R
Rscript 4_external_contextualization/SHAP_analysis.R
```

### 4A. AIBL Feasibility Gate

`AIBL_Feasibility_Gate.R` rebuilds the AIBL baseline MCI-to-AD endpoint and determines whether AIBL can support full-feature or reduced-feature validation.

Run with default relative paths:

```bash
Rscript 4_external_contextualization/AIBL_Feasibility_Gate.R \
  --aibl_dir ./aibl_19Sep2019/Data_extract_3.3.0 \
  --adni_holdout_file ./AI_vs_Clinician_Test/independent_test_set.csv \
  --adni_discovery_file ./Phase1_ADNI_Discovery/ADNI_Labeled_For_Classifier.csv \
  --model_config ./step11_results/model_config.rds \
  --feature_importance ./step11_results/Feature_Importance_RF.csv \
  --output_dir ./4_external_contextualization/AIBL_Feasibility_Gate
```

Main outputs include:

```text
4_external_contextualization/AIBL_Feasibility_Gate/
├── 01_aibl_all_baseline_mci_rebuilt.csv
├── 02_aibl_eligible_mci_to_ad_conversion_cohort.csv
├── 03_aibl_prisma_sample_flow.csv
├── 04_aibl_vs_adni_feature_overlap_audit.csv
├── 05_aibl_feature_gate_summary.csv
├── 06_aibl_reduced_feature_missingness.csv
├── 07_aibl_reduced_core_feature_status.csv
├── 08_aibl_gate_decision_summary.csv
├── 09_aibl_reader_study_blinded_case_packet.csv
├── 10_aibl_reader_study_outcome_key_do_not_share.csv
└── README_AIBL_feasibility_gate.txt
```

Interpretation:

- If full multimodal ADNI feature equivalence is available, AIBL can be considered for full-feature frozen-model validation.
- If full feature equivalence is not available but age, sex, MMSE, and APOE epsilon-4 are harmonizable, proceed with reduced-feature external validation.
- The blinded case packet can support a future retrospective external reader study, but it is not itself an expert-reader result.

### 4B. AIBL Reduced-Feature External Validation

`AIBL_Reduced_Feature_External_Validation.R` trains a prespecified reduced clinical model in ADNI discovery and applies the frozen preprocessing, coefficients, and threshold once to AIBL.

Run:

```bash
Rscript 4_external_contextualization/AIBL_Reduced_Feature_External_Validation.R \
  --data_root . \
  --out_dir ./4_external_contextualization/AIBL_Reduced_Feature_External_Validation
```

Expected data under `--data_root` include:

```text
Phase1_ADNI_Discovery/ADNI_Labeled_For_Classifier.csv
ADNI原始数据/LINES/Subject Demographics.csv
ADNI原始数据/LINES/Mini-Mental State Examination (MMSE).csv
ADNI原始数据/LINES/Clinical Dementia Rating.csv
ADNI原始数据/APOE/ApoE Genotyping - Results.csv
aibl_19Sep2019/Data_extract_3.3.0/aibl_pdxconv_01-Jun-2018.csv
aibl_19Sep2019/Data_extract_3.3.0/aibl_mmse_01-Jun-2018.csv
aibl_19Sep2019/Data_extract_3.3.0/aibl_cdr_01-Jun-2018.csv
aibl_19Sep2019/Data_extract_3.3.0/aibl_apoeres_01-Jun-2018.csv
aibl_19Sep2019/Data_extract_3.3.0/aibl_ptdemog_01-Jun-2018.csv
```

Main outputs include:

```text
4_external_contextualization/AIBL_Reduced_Feature_External_Validation/
├── 01_cohort_summary.csv
├── 02_adni_discovery_training_performance.csv
├── 03_aibl_external_validation_performance.csv
├── 04_frozen_model_coefficients.csv
├── 05_aibl_external_predictions.csv
├── 06_aibl_bootstrap_metric_ci.csv
├── 07_aibl_decision_curve.csv
├── 08_frozen_preprocessing_parameters.csv
├── 09_aibl_probability_distribution.png
├── 10_aibl_decision_curve.png
└── 00_README_results_summary.txt
```

Manuscript interpretation:

- This is true external validation of a prespecified reduced-feature clinical model.
- It is not direct validation of the full multimodal AI model.
- Full multimodal AIBL validation would require same-pipeline MRI feature extraction and harmonized multimodal inputs.

### 5. Final Evidence Synthesis

After upstream analyses finish, run the manuscript-facing synthesis script.

```bash
Rscript 5_final_evidence_synthesis/Evidence_synthesis.R \
  --step14_dir ./step14_results \
  --step2_dir ./AI_vs_Clinician_Test \
  --step16_dir ./step16_results \
  --step20_dir ./step20_results \
  --step21_dir ./step21_results \
  --step12_dir ./step12_results \
  --step22_dir ./step22_results \
  --output_dir ./step18_results
```

`Evidence_synthesis.R` is intended for final aggregation. It should not be run before the discovery, holdout, external, and Rule C branches have generated their outputs.

## Manuscript-Facing Evidence Hierarchy

The following hierarchy should be preserved when interpreting outputs:

1. **ADNI holdout benchmark:** independent participant-level AI-versus-neurologist evaluation.
2. **Rule C workflow:** primary translational human-AI analysis, using the AI model as a specificity-oriented second reader in expert Stage 2 gray-zone cases.
3. **AIBL reduced-feature validation:** secondary external validation of an ADNI-trained clinical model using harmonized age, sex, MMSE, and APOE epsilon-4 features.
4. **A4 and HABS:** framework-extension analyses, not co-primary validation cohorts for the full multimodal model.
5. **VAE and BSI:** supporting structural heterogeneity and longitudinal context, not deployment-ready subtype labels.

This distinction is important for avoiding overstatement of external validation and for maintaining the integrity of the frozen holdout benchmark.

## Key Output Files for the Final Manuscript

For the human-AI Rule C analysis:

```text
3_AI_vs_Clinician_Analysis/Q1_Human_AI_Extension/00_gray_zone_distribution_check.csv
3_AI_vs_Clinician_Analysis/Q1_Human_AI_Extension/04_workflow_metrics_vs_expert_stage2.csv
3_AI_vs_Clinician_Analysis/Q1_Human_AI_Extension/05_categorical_nri_primary.csv
3_AI_vs_Clinician_Analysis/Q1_Human_AI_Extension/06_decision_curve_net_benefit.csv
3_AI_vs_Clinician_Analysis/Q1_Human_AI_Extension/06_dca_resource_translation_per_100.csv
3_AI_vs_Clinician_Analysis/Q1_Human_AI_Extension/posthoc_refinements/11_ruleC_fp_fn_paired_comparison.csv
3_AI_vs_Clinician_Analysis/Q1_Human_AI_Extension/posthoc_refinements/12_dca_net_benefit_bootstrap_ci_key_thresholds.csv
```

For the AIBL reduced-feature external validation:

```text
4_external_contextualization/AIBL_Feasibility_Gate/08_aibl_gate_decision_summary.csv
4_external_contextualization/AIBL_Reduced_Feature_External_Validation/03_aibl_external_validation_performance.csv
4_external_contextualization/AIBL_Reduced_Feature_External_Validation/05_aibl_external_predictions.csv
4_external_contextualization/AIBL_Reduced_Feature_External_Validation/06_aibl_bootstrap_metric_ci.csv
4_external_contextualization/AIBL_Reduced_Feature_External_Validation/07_aibl_decision_curve.csv
```

For framework-extension analyses:

```text
step20_results/step20_aibl_summary.csv
step21_results/step21_a4_summary.csv
step16_results/step16_manuscript_summary.csv
```

## Reproducibility Notes

- The 196-case ADNI holdout benchmark is independent at the participant level from the ADNI discovery cohort.
- Rule C uses a fixed AI threshold and an a priori expert gray zone; it does not fit new model weights in the holdout set.
- Simple/rank AI-expert combinations are no-refitting sensitivity analyses.
- Fitted logistic stacking on the holdout set should be interpreted only as exploratory or cross-validated sensitivity analysis, not as the primary validated model.
- The AIBL reduced-feature model derives preprocessing parameters, coefficients, and the operating threshold in ADNI discovery and applies them unchanged to AIBL.
- A4 uses a preclinical cognitive-progression outcome and therefore should not be described as direct MCI-to-AD validation.
- HABS uses cohort-specific modeling with plasma p-tau217 and therefore evaluates framework adaptability rather than direct ADNI model transfer.
- VAE subgroup labels are descriptive latent structural profiles and should not be treated as deployment-ready clinical subtypes.
- BSI analyses provide longitudinal structural context and should be interpreted alongside their borderline and non-monotonic statistical pattern.

## Scripts Retained for Traceability

Some scripts reflect earlier or auxiliary analyses and are retained for transparency:

- `2_discovery_characterization/ADNI_discovery.R`
- `4_external_contextualization/AIBL _Validation.R`
- legacy AIBL/A4 direct VAE-transfer scripts

These remain useful for reproducing supplementary framework-extension analyses, but the final manuscript's main external validation component is `AIBL_Reduced_Feature_External_Validation.R`.

## Troubleshooting

### Missing data files

If a script stops with `Missing file`, either place the required file under the expected default location or pass the correct path through the command-line arguments.

### Different AI prediction filenames

The Rule C script defaults to:

```text
AI_vs_Clinician_Test/AI_Predictions_Final.csv
```

The AI prediction script also writes:

```text
AI_vs_Clinician_Test/AI_per_patient_predictions.csv
AI_vs_Clinician_Test/AI_test_predictions.csv
```

If you prefer one of these files, pass it via `--ai_file`.

### R packages not found

Install the CRAN and Bioconductor packages listed in the Software Environment section. Some scripts attempt automatic installation, but locked institutional computers may require manual installation.

### AIBL full-feature validation fails

This is expected if AIBL lacks same-pipeline MRI/CSF/VAE features. In that case, use the reduced-feature external validation and report it explicitly as reduced-feature external validation.

### Reader study packet

`AIBL_Feasibility_Gate.R` writes a blinded case packet and a separate outcome key. The outcome key should not be shared with readers during a retrospective reader study.

## Citation

If you use this repository, please cite the associated manuscript and the originating cohort studies.

## License

This repository is released under the MIT License. See `LICENSE`.
