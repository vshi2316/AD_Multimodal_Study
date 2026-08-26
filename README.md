[README.md](https://github.com/user-attachments/files/31445281/README.md)
# Multimodal Artificial Intelligence for Three-Year Progression From Mild Cognitive Impairment to Alzheimer Disease

This repository contains reproducible code for multimodal representation learning, progression-risk prediction, reader benchmarking, and external cohort evaluation.

The primary discovery analysis integrates cerebrospinal-fluid biomarkers, clinical and genetic variables, and structural MRI measures from ADNI. A modality-weighted variational autoencoder learns a latent representation, followed by clustering to describe multimodal progression-related heterogeneity. The supervised prediction analysis is fitted separately and evaluated on a non-overlapping ADNI holdout set with structured assessments from five neurologists.

External analyses address distinct transportability questions. A4 is evaluated by direct projection into the ADNI latent space when the required feature space is available. AIBL is evaluated with a frozen clinical proxy based on age, sex, and APOE ε4 status because the available AIBL release does not contain the harmonized CSF and structural MRI variables required for direct projection. HABS is evaluated with a cohort-specific Firth logistic framework using age, sex, APOE ε4 status, and plasma p-tau217 when available.

## Repository structure

```text
0_shared_input_preparation/
1_discovery_subtype_model/
2_discovery_characterization/
3_AI_vs_Clinician_Analysis/
4_external_contextualization/
5_final_evidence_synthesis/
analysis_pipeline/
```

## Main workflow

```bash
python analysis_pipeline/run_analysis_pipeline.py \
  --data-root "/absolute/path/to/ad_multimodal_data" \
  --output-dir "analysis_pipeline/outputs" \
  --figure-dir "analysis_pipeline/submission_figures"
```

The pipeline defines endpoints, prepares aligned discovery and holdout data, fits leakage-controlled models, evaluates the AIBL clinical proxy, performs VAE sensitivity analyses, constructs the non-overlapping ADNI validation cohort, and calculates five-reader benchmark statistics.

## External contextualization

The `4_external_contextualization/` directory contains AIBL cohort construction, A4 projection validation, HABS Firth logistic evaluation, SHAP analysis, and the general cross-cohort projection utility. External analyses are interpreted according to each cohort’s available variables and endpoint definitions.

## Software and data

Python 3.10 or later is recommended; install dependencies with `python -m pip install -r requirements.txt`. R 4.2 or later is recommended for the R analyses. Participant-level ADNI, A4, AIBL, and HABS data are not redistributed and must be obtained under the applicable data-use agreements.

## Reproducibility and privacy

Random seeds are set in stochastic analyses. Imputation, scaling, tuning, and threshold selection are estimated from designated training data and frozen before external application. Do not commit participant-level predictions, dates, reader ratings, or other restricted data; share only permitted aggregate outputs.

## License

See `LICENSE`. Underlying cohort data remain subject to their original data-use agreements.
