# AD Multimodal Study

This repository contains the code used for multimodal subtype discovery, prognostic modelling, external evaluation, longitudinal imaging validation, and AI versus clinician benchmarking in Alzheimer's disease research.

## Study focus

The manuscript aligned analysis addresses four linked questions.

1. Whether clinically similar participants with mild cognitive impairment contain structurally distinct multimodal subtypes.
2. Whether those subtype differences correspond to divergent longitudinal brain atrophy.
3. Whether a frozen holdout prediction pipeline generalizes to an independent Alzheimer's Disease Neuroimaging Initiative test set.
4. Whether the framework extends across external cohorts through subtype transportability in A4 and AIBL and framework adaptation in the Harvard Aging Brain Study.

## Repository structure

### Core scripts

`step1_preprocess_APOE.py`

`step2_preprocess_CSF.py`

`step3_preprocess_Clinical.py`

`step4_preprocess_sMRI.py`

`step5_preprocess_PET.py`

`step6_create_outcome.py`

`step7_Cohort Integration.py`

`step8_vae_clustering.py`

`step9A_cross_cohort_analysis.py`

`step9B_biomarker_validation.py`

`step10_differential_analysis.R`

`step11_predictive_modeling.R`

`step12_cluster_signatures.R`

`step13_conversion_differential.R`

`step14_cluster_validation.R`

`step15_cross_modal_validation.R`

`step16_habs_validation.R`

`step17_shap_analysis.R`

`step18_evidence_synthesis.R`

`step19_ADNI_discovery.R`

`step20_AIBL _Validation.R`

`step21_A4_validation.R`

`step22_neuroimaging_endotypes.R`

### AI versus clinician workflow

Directory `AI_vs_Clinician_Analysis`

`Step 1 Prepare Test.R`

`Step 2 AI Prediction.py`

`Step 3 Expert Assessment Workflow.R`

`Step 4 AI vs Expert Comparison Analysis`

## Recommended execution order

### Discovery and primary analysis path

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
15. `step19_ADNI_discovery.R`
16. `step9A_cross_cohort_analysis.py`
17. `step20_AIBL _Validation.R`
18. `step21_A4_validation.R`
19. `step22_neuroimaging_endotypes.R`
20. `step16_habs_validation.R`
21. `step17_shap_analysis.R`
22. `step18_evidence_synthesis.R`

### Independent holdout branch

1. `AI_vs_Clinician_Analysis\Step 1 Prepare Test.R`
2. `AI_vs_Clinician_Analysis\Step 2 AI Prediction.py`
3. `AI_vs_Clinician_Analysis\Step 3 Expert Assessment Workflow.R`
4. `AI_vs_Clinician_Analysis\Step 4 AI vs Expert Comparison Analysis`

Run `step18_evidence_synthesis.R` only after the independent holdout branch, HABS validation, and A4 and AIBL summary outputs have all been generated.

## Data requirements

This repository does not redistribute cohort data. Access must be obtained directly from the source studies under their respective data use agreements.

### Cohorts used

Alzheimer's Disease Neuroimaging Initiative

Anti Amyloid Treatment in Asymptomatic Alzheimer's Disease study

Australian Imaging, Biomarker and Lifestyle study

Harvard Aging Brain Study

Fox Laboratory boundary shift integral longitudinal magnetic resonance imaging measures

### Common input files referenced by scripts

`Clinical_data.csv`

`metabolites.csv`

`RNA_plasma.csv`

`subtype_assignments.csv`

`latent_representations.csv`

`vae_summary.json`

`independent_test_set.csv`

`HABS_Baseline_Integrated.csv`

`AIBL_Baseline_Integrated.csv`

`A4_Baseline_Integrated.csv`

Because local file names may differ across environments, inspect each script argument list before execution.

## Software environment

### Python

Recommended Python 3.10 or higher

Install dependencies with `pip install -r requirements.txt`

### R

Recommended R 4.2 or higher

Install required packages listed in the script headers before execution.

## Manuscript alignment notes

The repository reflects the current manuscript aligned logic.

1. The variational autoencoder uses a modality weighted reconstruction loss so that cerebrospinal fluid, clinical, and magnetic resonance imaging blocks contribute comparably despite unequal dimensionality.
2. The final variational autoencoder feature list is stored in `vae_summary.json` and should be treated as the executable record of the discovery input matrix.
3. A4 and AIBL are implemented as direct subtype transportability analyses based on latent projection and centroid assignment.
4. The Harvard Aging Brain Study is implemented as framework adaptation with cohort specific Firth logistic regression rather than direct transfer of the Alzheimer's Disease Neuroimaging Initiative subtype model.
5. The holdout workflow now leaves unavailable variables such as `GDS` as missing and lets the discovery fitted imputation pipeline supply the reference value. Holdout preprocessing does not estimate imputation or scaling parameters from the holdout set.
6. `AI_vs_Clinician_Analysis\Step 1 Prepare Test.R` now creates a strict 36 month conversion endpoint so that the public holdout label matches the expert three year assessment task.
7. `step14_cluster_validation.R` now checks archived follow up variables first and writes `Cox_Time_Source_Metadata.csv` so that discovery survival analyses can be reported transparently.
8. `step16_habs_validation.R` writes `step16_manuscript_summary.csv`, which should be used as the manuscript facing source for Harvard Aging Brain Study sample size, event count, event rate, and area under the curve reporting.
9. `step20_AIBL _Validation.R` writes `step20_aibl_summary.csv` and `step21_A4_validation.R` writes `step21_a4_summary.csv` for cross cohort synthesis.
10. `step18_evidence_synthesis.R` uses direct step outputs whenever available and no longer relies on hardcoded holdout metrics.

## Important caveats

1. `step11_predictive_modeling.R` currently uses multiple imputation by chained equations and then carries forward one completed dataset with `complete(mice_obj, 1)`. It does not pool estimates under Rubin's rules.
2. The current public discovery modelling workflow does not implement additional inverse probability class weighting.
3. Several preprocessing scripts are broader than the final primary manuscript path. In particular, `step5_preprocess_PET.py` is retained for traceability but is not part of the final primary analysis path.
4. The file name `step20_AIBL _Validation.R` contains a space and should be called exactly as written.

## Example commands

### Python examples

`python step8_vae_clustering.py --input_dir ./processed_data --output_dir ./vae_output`

`python step9A_cross_cohort_analysis.py --external_file ./A4_Baseline_Integrated.csv --vae_dir ./vae_output --output_dir ./a4_projection --cohort_name A4`

`python AI_vs_Clinician_Analysis/"Step 2 AI Prediction.py" --base_dir ./AI_vs_Clinician_Analysis --output_dir ./AI_vs_Clinician_Analysis/results`

### R examples

`Rscript step11_predictive_modeling.R --vae_dir ./vae_output --data_dir ./processed_data --output_dir ./step11_results`

`Rscript step16_habs_validation.R --habs_file ./HABS_Baseline_Integrated.csv --output_dir ./step16_results`

`Rscript "step20_AIBL _Validation.R" --projection_file ./aibl_projection/AIBL_projected_subtypes.csv --baseline_file ./AIBL_Baseline_Integrated.csv --output_dir ./step20_results`

## Expected outputs

Representative outputs include the following files.

Subtype assignments and latent representations

Variational autoencoder summary files and subtype centroids

Feature importance tables and model comparison files

Calibration, receiver operating characteristic, and confusion matrix plots

Boundary shift integral longitudinal analysis tables and figures

External cohort projected subtype files

Harvard Aging Brain Study performance summaries, decision curves, and kernel SHAP outputs

Independent holdout prediction outputs and AI versus clinician comparison results

Cross cohort manuscript summary tables, including `Cox_Time_Source_Metadata.csv`, `step16_manuscript_summary.csv`, `step20_aibl_summary.csv`, and `step21_a4_summary.csv`

## Reproducibility and reporting

Set seeds are embedded in the scripts where applicable.

Cohort specific preprocessing assumptions remain script dependent and should be documented in any derivative publication.

Before manuscript submission, verify that the text matches the executable code for endpoint definition, imputation strategy, time source handling, and external cohort summary outputs.

## Citation

If you use this repository, please cite the associated manuscript and the originating cohort studies.

## License

This repository is released under the MIT License. See `LICENSE`.
