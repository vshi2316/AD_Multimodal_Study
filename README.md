[README.md](https://github.com/user-attachments/files/29989673/README.md)
# AD Multimodal Study

Code supporting the manuscript:

**Five-Neurologist Evaluation of Cross-Fitted Multimodal Artificial Intelligence for Three-Year Progression From Mild Cognitive Impairment to Alzheimer Disease**

This repository contains data-preparation scripts, variational autoencoder analyses, supervised prediction models, a five-neurologist reader benchmark, Rule C decision analyses, longitudinal outcome analyses, and external cohort contextualization. Participant-level source data and expert ratings are not distributed because they are governed by study-specific data-use agreements.

## Study objective

The study evaluates three-year progression from mild cognitive impairment to Alzheimer disease and examines whether multimodal model estimates can support neurologist classification when reader probabilities fall within a prespecified uncertainty interval.

Five neurologists provided conversion probabilities before and after structured magnetic resonance imaging review. Rule C substitutes the model probability only when the Stage 2 neurologist probability is between 40% and 60%. The model threshold and Rule C interval are specified in the analysis code.

The evidence structure includes:

1. Multimodal characterization in the Alzheimer Disease Neuroimaging Initiative discovery cohort.
2. Cross-fitted model predictions for reader-study participants who overlap the discovery cohort.
3. Frozen model predictions for reader-study participants without discovery overlap.
4. A participant-independent Alzheimer Disease Neuroimaging Initiative validation cohort with 318 participants and no discovery identifiers.
5. A clinical-proxy validation in the Australian Imaging, Biomarker and Lifestyle cohort.
6. Exploratory analyses in the Anti-Amyloid Treatment in Asymptomatic Alzheimer's Disease and Harvard Aging Brain Study cohorts.
7. Longitudinal boundary shift integral and cognitive outcome sensitivity analyses.

## Main analysis set

The strict endpoint defines conversion as an Alzheimer disease diagnosis within 36 calendar months of the mild cognitive impairment baseline. A participant classified as a non-event must have diagnostic follow-up extending to at least 36 months. Participants without an event and without sufficient follow-up are excluded from strict-endpoint analyses.

The five-reader benchmark contains 153 participants, 77 events, 76 non-events, five neurologists, and 765 case-reader pairs. Outcome-excluded predictions are used for every participant. Cross-fitting supplies predictions for the 124 participants who overlap the discovery cohort. A model fitted in the strict development cohort supplies frozen predictions for the 29 nonoverlapping reader-study participants.

The participant-independent Alzheimer Disease Neuroimaging Initiative benchmark contains 318 participants, including 104 events. The clinical plus magnetic resonance imaging model achieved an area under the receiver operating characteristic curve of 0.752, with a 95% confidence interval from 0.695 to 0.810. The complete clinical, cerebrospinal fluid, and magnetic resonance imaging model achieved an area under the curve of 0.720; cerebrospinal fluid measurements were available for 29 participants, so this estimate requires cautious interpretation.

The Australian Imaging, Biomarker and Lifestyle clinical-proxy analysis contains 34 participants and 16 events. The age, sex, Mini-Mental State Examination, and apolipoprotein E epsilon 4 model achieved an area under the curve of 0.759, with a 95% confidence interval from 0.597 to 0.908. This analysis evaluates transportability of a clinical proxy and does not constitute complete external validation of the magnetic resonance imaging and cerebrospinal fluid model.

Reference values used to check generated aggregate results are stored in `analysis_pipeline/REFERENCE_RESULTS.json`.

## Repository structure

```text
AD_Multimodal_Study/
|-- 0_shared_input_preparation/
|-- 1_discovery_subtype_model/
|-- 2_discovery_characterization/
|-- 3_AI_vs_Clinician_Analysis/
|-- 4_external_contextualization/
|-- 5_final_evidence_synthesis/
|-- analysis_pipeline/
|   |-- 01_define_36m_endpoints.py
|   |-- 02_rulec_statistics_core.py
|   |-- 03_extract_aligned_features.py
|   |-- 04_fit_leakage_controlled_models.py
|   |-- 05_validate_aibl_clinical_proxy.py
|   |-- 06_vae_sensitivity_analysis.py
|   |-- 07_build_nonoverlap_adni_validation.py
|   |-- 08_crossfit_five_reader_benchmark.py
|   |-- 09_multireader_statistics.py
|   |-- 10_generate_figures.py
|   |-- run_analysis_pipeline.py
|   |-- REFERENCE_RESULTS.json
|   |-- OUTPUT_SCHEMA.md
|   `-- .env.example
|-- CITATION.cff
|-- LICENSE
|-- requirements.txt
`-- README.md
```

### Module descriptions

`0_shared_input_preparation` contains preprocessing for clinical variables, apolipoprotein E genotype, cerebrospinal fluid biomarkers, structural magnetic resonance imaging, outcome construction, and cohort integration.

`1_discovery_subtype_model` contains variational autoencoder training, latent representation extraction, and clustering.

`2_discovery_characterization` contains subtype characterization, biomarker analyses, conversion comparisons, predictive modeling, neuroimaging endotypes, and longitudinal boundary shift integral analyses.

`3_AI_vs_Clinician_Analysis` contains the five-neurologist assessment workflow, artificial intelligence predictions, Rule C analyses, no-latent-variable analyses, post hoc analyses, and longitudinal outcome sensitivity analyses.

`4_external_contextualization` contains Australian Imaging, Biomarker and Lifestyle feasibility and clinical-proxy analyses, Anti-Amyloid Treatment in Asymptomatic Alzheimer's Disease analyses, Harvard Aging Brain Study analyses, cross-cohort projections, and feature-attribution analyses.

`5_final_evidence_synthesis` contains evidence-summary code for combining outputs from the component analyses.

`analysis_pipeline` contains the ordered workflow for the strict 36-month endpoint, leakage-controlled supervised models, five-reader benchmark, participant-independent validation, clinical-proxy validation, VAE sensitivity analyses, and manuscript figures.

## Data access

Source data must be obtained from the organizations that govern each cohort. Users are responsible for completing the corresponding applications and complying with all data-use terms.

The analyses use data from:

1. Alzheimer's Disease Neuroimaging Initiative.
2. Australian Imaging, Biomarker and Lifestyle study.
3. Anti-Amyloid Treatment in Asymptomatic Alzheimer's Disease study.
4. Harvard Aging Brain Study.
5. Fox Lab longitudinal boundary shift integral outputs.
6. Five neurologist assessments collected for this study.

No participant-level data, protected health information, imaging files, or individual neurologist ratings should be committed to this repository.

## Data directory

The integrated pipeline expects the following English-language directory structure beneath the path supplied with `--data-root`:

```text
ad_multimodal_data/
|-- ADNI_Raw_Data/
|   |-- LINES/
|   |   |-- Diagnostic Summary.csv
|   |   |-- Subject Demographics.csv
|   |   `-- Mini-Mental State Examination (MMSE).csv
|   |-- APOE/
|   |   `-- ApoE Genotyping - Results.csv
|   |-- CSF/
|   |   |-- UPENN CSF Biomarker Master Alzbio3.csv
|   |   `-- UPENN CSF Biomarkers Roche Elecsys.csv
|   `-- sMRI/
|       `-- UCSF - Cross-Sectional FreeSurfer (7.x).csv
|-- Analysis_Inputs/
|   |-- AI_vs_Clinician_Test/
|   |   |-- independent_test_set.csv
|   |   |-- AI_per_patient_predictions.csv
|   |   `-- Expert_Predictions_Long.csv
|   |-- AIBL_Feasibility_Gate/
|   |   `-- 02_aibl_eligible_mci_to_ad_conversion_cohort.csv
|   `-- VAE_Output/
|       `-- subtype_assignments.csv
`-- Derived_Inputs/
    `-- Discovery_CSF_Cohort/
        |-- Womac_score_pain_function.csv
        |-- Clinical_data.csv
        |-- RNA_plasma.csv
        `-- metabolites.csv
```

The public code contains no machine-specific absolute data path. `AD_MULTIMODAL_DATA_ROOT` may be set directly, or the same path may be supplied with `--data-root`.

## Software requirements

Recommended environment:

- Python 3.10 or later.
- R 4.2 or later.
- At least 16 GB of memory for the complete workflow.

Install Python dependencies:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Linux or macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Principal Python packages include NumPy, pandas, SciPy, scikit-learn, statsmodels, TensorFlow, Keras, matplotlib, seaborn, and joblib.

Install the R packages required by the component scripts:

```r
install.packages(c(
  "optparse", "readr", "dplyr", "tidyr", "purrr", "stringr",
  "ggplot2", "patchwork", "broom", "car", "emmeans", "lme4",
  "lmerTest", "survival", "survminer", "pROC", "mice", "caret",
  "glmnet", "randomForest", "ranger", "cluster", "factoextra"
))
```

Package requirements for an individual R analysis are also declared near the beginning of the corresponding script.

## Run the integrated pipeline

Run all ten steps from the repository root:

```bash
python analysis_pipeline/run_analysis_pipeline.py \
  --data-root "/absolute/path/to/ad_multimodal_data" \
  --output-dir "analysis_pipeline/outputs" \
  --figure-dir "analysis_pipeline/submission_figures"
```

Windows example:

```powershell
python analysis_pipeline\run_analysis_pipeline.py `
  --data-root "C:\data\ad_multimodal_data" `
  --output-dir "analysis_pipeline\outputs" `
  --figure-dir "analysis_pipeline\submission_figures"
```

Run a selected interval with `--start-step` and `--stop-step`:

```bash
python analysis_pipeline/run_analysis_pipeline.py \
  --data-root "/absolute/path/to/ad_multimodal_data" \
  --start-step 8 \
  --stop-step 10
```

The selected interval assumes that required outputs from preceding steps already exist in `--output-dir`.

## Pipeline steps

### Step 1: Define 36-month endpoints

Script: `analysis_pipeline/01_define_36m_endpoints.py`

This step identifies the mild cognitive impairment baseline, calculates time from baseline to Alzheimer disease diagnosis, applies the strict 36-month endpoint, and constructs a visit-window sensitivity endpoint for the Alzheimer Disease Neuroimaging Initiative and Australian Imaging, Biomarker and Lifestyle cohorts.

Principal outputs:

- `adni_discovery_endpoints.csv`
- `adni_holdout_endpoints.csv`
- `aibl_endpoints.csv`
- `endpoint_summary.json`

### Step 2: Construct Rule C analysis tables

Script: `analysis_pipeline/02_rulec_statistics_core.py`

This step aligns strict outcomes, model probabilities, and five-neurologist ratings. It evaluates the prespecified 40% to 60% uncertainty interval, calculates discrimination, Brier score, calibration, sensitivity, specificity, paired classification changes, reader-specific performance, and bootstrap confidence intervals.

Principal outputs:

- `holdout_reader_master.csv`
- `rulec_scenario_performance.csv`
- `rulec_scenario_bootstrap_cis.csv`
- `rulec_paired_changes.csv`
- `rulec_reclassification_tables.csv`
- `rulec_per_reader_performance.csv`
- `expert_ai_reference_correlations.csv`
- `holdout_reader_summary.json`

### Step 3: Extract aligned baseline features

Script: `analysis_pipeline/03_extract_aligned_features.py`

This step extracts baseline clinical, apolipoprotein E, cerebrospinal fluid, and FreeSurfer variables from source tables. It aligns raw features to the endpoint cohorts without using outcome information during feature preprocessing.

Principal outputs:

- `raw_extraction_validation_against_test_file.csv`
- `adni_discovery_raw_aligned_features.csv`
- `adni_holdout_raw_aligned_features.csv`

### Step 4: Fit leakage-controlled supervised models

Script: `analysis_pipeline/04_fit_leakage_controlled_models.py`

This step fits elastic-net logistic models within pipelines containing median imputation, missingness indicators, and standardization. Hyperparameters are selected within inner folds. Out-of-fold predictions and thresholds are generated without access to the corresponding test outcomes.

Candidate feature sets include clinical variables, clinical plus magnetic resonance imaging variables, and the complete clinical plus cerebrospinal fluid plus magnetic resonance imaging panel.

Principal outputs:

- `leakage_free_model_performance.csv`
- `independent41_new_model_predictions.csv`
- `nested_tuning_*.csv`
- `final_coefficients_*.csv`

### Step 5: Evaluate the AIBL clinical proxy

Script: `analysis_pipeline/05_validate_aibl_clinical_proxy.py`

This step fits the age, sex, Mini-Mental State Examination, and apolipoprotein E epsilon 4 model in the Alzheimer Disease Neuroimaging Initiative development data and evaluates frozen predictions in eligible Australian Imaging, Biomarker and Lifestyle participants.

Principal outputs:

- `aibl_harmonized_predictions.csv`
- `aibl_harmonized_nested_tuning.csv`
- `aibl_harmonized_performance.csv`

### Step 6: Evaluate VAE sensitivity analyses

Script: `analysis_pipeline/06_vae_sensitivity_analysis.py`

This step evaluates subtype-associated magnetic resonance imaging variation with Type II analysis of covariance, strict 36-month conversion associations, demographic and education-related separation, and soluble triggering receptor expressed on myeloid cells 2 comparisons.

Principal outputs:

- `type2_ancova_mri_subtype.csv`
- `vae_subtype_conversion_rates_original_vs_strict36.csv`
- `vae_strict36_conversion_association.csv`
- `vae_demographic_confounding_tests.csv`
- `strem2_subtype_descriptive.csv`
- `strem2_pairwise_tests.csv`
- `vae_statistics_summary.json`

### Step 7: Build the nonoverlapping ADNI validation cohort

Script: `analysis_pipeline/07_build_nonoverlap_adni_validation.py`

This step identifies Alzheimer Disease Neuroimaging Initiative participants with a mild cognitive impairment baseline who do not share a discovery identifier, applies the strict endpoint, extracts the required features, and evaluates frozen supervised models.

Principal outputs:

- `new_nonoverlapping_adni_benchmark_predictions.csv`
- `new_nonoverlapping_adni_benchmark_performance.csv`
- `new_nonoverlapping_adni_benchmark_subgroups.csv`

### Step 8: Generate cross-fitted reader-study predictions

Script: `analysis_pipeline/08_crossfit_five_reader_benchmark.py`

For reader-study participants who overlap the discovery cohort, this step produces outer-fold predictions from models that exclude the participant during fitting, tuning, and threshold selection. Frozen development-cohort models supply predictions for reader-study participants without discovery overlap.

Principal outputs:

- `crossfitted_expert_benchmark.csv`
- `crossfitted_expert_summary.csv`
- `crossfitted_expert_performance.csv`
- `crossfitted_expert_ci.csv`
- `crossfitted_expert_per_reader.csv`
- `crossfitted_expert_paired.csv`
- `crossfitted_expert_reclassification.csv`
- `crossfitted_expert_folds.csv`

### Step 9: Estimate multi-reader statistics

Script: `analysis_pipeline/09_multireader_statistics.py`

This step calculates reader-macro performance, pooled performance, participant-cluster bootstrap confidence intervals, exact McNemar tests, reclassification tables, categorical net reclassification improvement, decision-curve estimates, intraclass correlation coefficients, and Fleiss kappa.

Principal outputs:

- `final_multireader_macro_performance.csv`
- `final_multireader_macro_differences.csv`
- `final_expert_reliability_strict153.csv`
- `final_pooled_reader_performance.csv`
- `final_pooled_reader_bootstrap_cis.csv`
- `final_pooled_rulec_paired_changes.csv`
- `final_pooled_rulec_reclassification.csv`
- `final_pooled_rulec_categorical_nri.csv`
- `final_pooled_rulec_exact_mcnemar.csv`
- `final_pooled_rulec_dca.csv`
- `final_multireader_summary.json`

### Step 10: Generate figures

Script: `analysis_pipeline/10_generate_figures.py`

This step generates the main and supplementary statistical figures directly from pipeline outputs. Each figure is written in 600-dpi PNG and vector PDF format.

Output directories:

- `analysis_pipeline/submission_figures/main`
- `analysis_pipeline/submission_figures/supplementary`

## Variational autoencoder analysis

The variational autoencoder is fitted in the discovery cohort to characterize multimodal baseline variation. It is separate from the supervised Rule C model.

Example command:

```bash
python 1_discovery_subtype_model/vae_clustering.py \
  --input_dir "/absolute/path/to/discovery_inputs" \
  --output_dir "/absolute/path/to/VAE_Output" \
  --cohort A \
  --n_clusters 3 \
  --latent_dim 3 \
  --epochs 300 \
  --batch_size 32
```

The principal outputs include latent representations, subtype assignments, subtype centroids, model parameters, training diagnostics, and clustering summaries.

## Five-neurologist assessment data

The five neurologists completed the structured assessment protocol represented in `3_AI_vs_Clinician_Analysis/Expert Assessment Workflow.R`. Stage 1 records the clinical-data estimate. Stage 2 records the estimate after magnetic resonance imaging review. The individual ratings are genuine human assessments and are treated as restricted research data.

Expected long-format variables include:

- case identifier
- neurologist identifier
- Stage 1 probability
- Stage 2 probability
- confidence rating
- observed outcome

The reader analysis preserves neurologist identity during reader-specific estimation and uses participant-level resampling for pooled confidence intervals.

## External and contextual analyses

The scripts in `4_external_contextualization` address distinct questions:

- `AIBL_Feasibility_Gate.R` evaluates feature and outcome availability.
- `AIBL reduced-feature external validation.R` evaluates the clinical proxy.
- `A4_validation.R` evaluates framework extension in the Anti-Amyloid Treatment in Asymptomatic Alzheimer's Disease cohort.
- `HABS_validation.R` evaluates cohort-specific associations in the Harvard Aging Brain Study.
- `Cross_cohort_analysis.py` projects compatible external data into the latent representation.
- `SHAP_analysis.R` evaluates feature attribution in the available external framework.

These analyses should be interpreted according to the variables and outcomes available in each cohort. They do not all represent full external validation of the complete multimodal model.

## Longitudinal analyses

`2_discovery_characterization/Cross_modal_validation.R` evaluates annualized whole-brain and ventricular boundary shift integral measures, subtype comparisons, linear mixed models, continuous latent-dimension correlations, and K equals 2 sensitivity analyses.

`3_AI_vs_Clinician_Analysis/Human_AI_RuleC_Longitudinal_Sensitivity.R` evaluates longitudinal Mini-Mental State Examination, Alzheimer's Disease Assessment Scale 13-item cognitive subscale, Clinical Dementia Rating Sum of Boxes, Functional Activities Questionnaire, and available boundary shift integral outcomes.

These analyses depend on serial observations and may include fewer participants than the baseline cohorts.

## Reproducibility safeguards

The integrated workflow applies the following controls:

1. Strict endpoint construction is separated from model fitting.
2. Discovery identifiers are excluded from the participant-independent validation cohort.
3. Imputation and scaling are fitted within the training data of each model pipeline.
4. Hyperparameter tuning occurs within inner cross-validation folds.
5. Reader-study participants with discovery overlap receive outer-fold predictions.
6. Threshold selection excludes the outcome of the participant receiving the prediction.
7. Random seeds are specified in the analysis scripts.
8. Aggregate reference values are stored separately from participant-level outputs.

## Output handling

Generated `outputs` and `submission_figures` directories are excluded by `.gitignore`. Inspect `analysis_pipeline/OUTPUT_SCHEMA.md` for the output definitions.

Before sharing generated files, verify that they contain no participant identifiers, dates, protected health information, or restricted individual ratings. Aggregate tables and figures may be shared only when permitted by the relevant data-use agreements.

## Result checking

`analysis_pipeline/REFERENCE_RESULTS.json` stores selected aggregate sample sizes and performance estimates reported by the analysis. It is intended for deterministic comparison after a complete run. Minor differences may arise from software versions in iterative optimization, but cohort counts and prespecified endpoint totals should agree exactly.

## Citation

Citation metadata are provided in `CITATION.cff`. When using this code, cite the associated manuscript and the original cohort publications required by the applicable data-use agreements.

## License

The source code is released under the MIT License. The license applies to the code only and does not grant permission to redistribute cohort data, imaging, expert ratings, or other restricted study materials.

## Repository

https://github.com/vshi2316/AD_Multimodal_Study
