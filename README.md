[Uploading README.md…]()
# Five-neurologist evaluation of cross-fitted multimodal artificial intelligence for three-year progression from mild cognitive impairment to Alzheimer disease

This repository contains the analysis code for a multimodal study of progression from mild cognitive impairment to Alzheimer disease within 36 months. The primary analysis evaluates leakage-controlled artificial intelligence predictions against structured assessments from five neurologists. Rule C retains the expert probability outside a prespecified 40% to 60% uncertainty interval and substitutes the artificial intelligence output within that interval.

The repository also contains an exploratory variational autoencoder analysis of multimodal heterogeneity. The supervised prediction model and the variational autoencoder address separate questions. Variational autoencoder groups do not enter the supervised model or Rule C.

## Study populations

The strict model-development cohort contains 126 participants. The reader benchmark contains 153 participants, including 77 events and 76 non-events. Among the reader cases, 124 overlap model development and receive cross-fitted predictions. The remaining 29 cases receive predictions from a frozen development model.

Participant-independent Alzheimer Disease Neuroimaging Initiative evaluation includes 318 participants and 104 events. The clinical plus magnetic resonance imaging model achieved an area under the receiver operating characteristic curve of 0.752, with a 95% confidence interval from 0.695 to 0.810.

The Australian Imaging, Biomarkers and Lifestyle clinical-proxy cohort contains 34 participants and 16 events. Its age, sex, Mini-Mental State Examination, and apolipoprotein E epsilon 4 model achieved an area under the curve of 0.759, with a 95% confidence interval from 0.597 to 0.908. The calibration slope was 3.234, with a 95% confidence interval from 1.263 to 7.180. This estimate indicates marked calibration failure and precludes interpretation of the probabilities as calibrated absolute risks.

## Repository structure

```text
0_shared_input_preparation/
1_discovery_subtype_model/
2_discovery_characterization/
3_AI_vs_Clinician_Analysis/
4_external_contextualization/
5_final_evidence_synthesis/
analysis_pipeline/
    01_define_36m_endpoints.py
    02_rulec_statistics_core.py
    03_extract_aligned_features.py
    04_fit_leakage_controlled_models.py
    05_validate_aibl_clinical_proxy.py
    06_vae_sensitivity_analysis.py
    07_build_nonoverlap_adni_validation.py
    08_crossfit_five_reader_benchmark.py
    09_multireader_statistics.py
    10_generate_figures.py
    FEATURE_MANIFEST.json
    OUTPUT_SCHEMA.md
    REFERENCE_RESULTS.json
    run_analysis_pipeline.py
FIGURE_PROVENANCE.md
R_PACKAGES.md
requirements.txt
```

## Data access

Individual-level data are governed by the applicable Alzheimer Disease Neuroimaging Initiative, Australian Imaging, Biomarkers and Lifestyle, Anti-Amyloid Treatment in Asymptomatic Alzheimer's Disease, and Harvard Aging Brain Study agreements. The repository does not distribute restricted participant data or individual reader ratings.

The primary Python workflow expects the following structure beneath `--data-root`.

```text
ADNI_Raw_Data/
    APOE/
    CSF/
    LINES/
    sMRI/
AIBL_Raw_Data/
Analysis_Inputs/
    AI_vs_Clinician_Test/
    Reader_Assessment/
Derived_Inputs/
    Discovery_CSF_Cohort/
```

Input tables must retain the variable names expected by the scripts. The supervised-model variables are fixed in `analysis_pipeline/FEATURE_MANIFEST.json`.

## Software environment

Python 3.10 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows PowerShell activation uses:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

R 4.2 or later is required for the characterization and longitudinal analyses. Package requirements appear in `R_PACKAGES.md`.

## Primary supervised and reader workflow

Run the ordered Python workflow from the repository root.

```bash
python analysis_pipeline/run_analysis_pipeline.py \
  --data-root "/absolute/path/to/ad_multimodal_data" \
  --output-dir "analysis_pipeline/outputs" \
  --figure-dir "analysis_pipeline/submission_figures"
```

The workflow performs the following analyses:

1. Defines strict 36-month, visit-window, and archived original endpoints.
2. Reconstructs the Rule C statistical benchmark.
3. Extracts the fixed clinical, cerebrospinal fluid, and magnetic resonance imaging features.
4. Fits leakage-controlled elastic-net models.
5. Evaluates the Australian Imaging, Biomarkers and Lifestyle clinical proxy.
6. Evaluates prespecified variational autoencoder sensitivity outcomes.
7. Constructs the participant-independent Alzheimer Disease Neuroimaging Initiative cohort.
8. Produces cross-fitted and frozen predictions for the five-reader benchmark.
9. Calculates reader-macro, pooled, paired, reclassification, agreement, and decision-curve statistics.
10. Generates Figures 1, 2, 3, and 5, together with Supplementary Figures 1 to 4 and 23 to 31.

Imputation, missingness indicators, scaling, tuning, and threshold selection occur within their designated training data. Calibration intercept and slope confidence intervals use 2,000 stratified bootstrap samples. Phase-specific estimates are generated for ADNI1 and ADNI3.

## Variational autoencoder analysis

The variational autoencoder uses 37 prestandardized variables after prespecified cerebrospinal fluid quality control. Median imputation and three-standard-deviation winsorization are fitted in the discovery cohort. Saved artifacts contain the fitted imputation values and feature-specific winsorization limits.

```bash
python 1_discovery_subtype_model/vae_clustering.py \
  --input_dir "/path/to/discovery_inputs" \
  --output_dir "/path/to/vae_outputs" \
  --n_clusters 3
```

Cluster stability is evaluated for K = 2 and K = 3 with identical bootstrap samples. Bootstrap labels are aligned to the corresponding full-data solution before Jaccard and participant-level stability calculations.

```bash
Rscript 2_discovery_characterization/Cluster_validation.R \
  --data_dir "/path/to/discovery_inputs" \
  --vae_dir "/path/to/vae_outputs" \
  --output_dir "/path/to/cluster_validation_outputs"
```

The quantitative criteria favor K = 2. The K = 3 solution is retained for exploratory description of finer multimodal patterns. Figure 4 is generated by `2_discovery_characterization/Cluster_signatures.R`. Figure 5 summarizes covariate-adjusted characterization and sensitivity analyses.

## External variational autoencoder projection

External projection requires features expressed on the discovery z-score scale. The frozen discovery imputation values, winsorization limits, encoder, and latent centroids are applied without refitting.

```bash
python 4_external_contextualization/Cross_cohort_analysis.py \
  --external_file "/path/to/external_integrated.csv" \
  --vae_dir "/path/to/vae_outputs" \
  --output_dir "/path/to/projection_outputs" \
  --cohort_name "external" \
  --input_scale discovery_zscore
```

## Output checks

`analysis_pipeline/OUTPUT_SCHEMA.md` defines the principal output tables. `analysis_pipeline/REFERENCE_RESULTS.json` records aggregate values reported in the manuscript. Cohort counts and prespecified endpoint totals should match exactly after a successful run. Numerical optimization can produce small floating-point differences across supported software versions.

`FIGURE_PROVENANCE.md` maps every main figure to its generating script. Figures are exported in 600-dpi PNG and vector PDF formats where supported by the source script.

## Privacy safeguards

Generated output directories are excluded by `.gitignore`. Files must be inspected before public release because prediction tables and analysis outputs may contain participant identifiers, dates, or restricted reader information. Only aggregate outputs permitted by the relevant data-use agreements may be shared.

## License

The code is released under the MIT License. Access to the underlying cohort data remains subject to the original data-use agreements.
