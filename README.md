[Uploading README.md…]()
# Intermediate-risk model substitution for three-year prognosis in mild cognitive impairment

Research code for **Intermediate-risk model substitution increases three-year prognostic specificity in a retrospective five-neurologist study of mild cognitive impairment**.

The primary analysis asks whether replacing neurologists' intermediate-risk forecasts changes the balance of three-year conversion errors. Five neurologists supplied clinical assessments at Stage 1 and assessments after regional MRI information at Stage 2. Rule C was applied retrospectively to the completed Stage 2 probabilities. The supervised model is fitted separately from the exploratory variational autoencoder (VAE).

## Analysis populations and evidence roles

| Analysis | Fixed population | Role |
|---|---|---|
| Multimodal discovery characterization | 157 participants | Exploratory fixed VAE partition and biomarker characterization |
| Supervised development | 126 participants, 66 events | Eligible 36-month conversion outcomes |
| Five-reader comparison | 153 cases, 77 events, 765 paired ratings | Primary retrospective substitution analysis |
| Development overlap within the reader comparison | 124 cases | Five-fold cross-fitted predictions and training-fold thresholds |
| Reader cases outside development | 29 cases | Frozen development model and development-only threshold |
| Additional ADNI evaluation | 318 participants, 104 events | Excludes all 157 discovery identifiers; includes the 29 reader cases outside development |
| AIBL three-variable clinical proxy | 28 participants, 12 events | Submitted age, sex and APOE model, with its original 157-person development labels |
| AIBL four-variable sensitivity analysis | 34 participants, 16 events | Separate harmonized model including MMSE |
| A4 | 1,147 participants | Clinical-class transfer and cognitive progression in a preclinical population |
| HABS-HD | 3,704 clinical complete cases; 2,407 with p-tau217 | Cohort-specific baseline classification and matched biomarker comparison |

A4 uses a frozen four-variable clinical random forest. Its nominal training classes have counts 53, 43 and 61. These classes are distinct from the VAE groups of 26, 59 and 72. HABS-HD fits its own Firth logistic models. Neither analysis constitutes external validation of the five-reader substitution rule.

## Repository layout

```text
analysis_config.json                     Input/output locations and Rscript
run_all.R                                source() entry for R users
install_R_packages.R                     Explicit dependency installation
analysis_pipeline/
  run_analysis_pipeline.py               Main entry and component runner
  INPUT_MANIFEST.json                    Required restricted input files
  FEATURE_MANIFEST.json                  Ordered supervised feature lists
  MRI_LABELS.json                        Verified anatomical display names
  FROZEN_AGGREGATE_CHECKS.json            Manuscript reference aggregates
  01_define_36m_endpoints.py              Date-based endpoint reconstruction
  02_rulec_statistics_core.py            Shared Rule C functions
  03_extract_aligned_features.py          Clinical/CSF extraction and screened MRI alignment
  04_fit_leakage_controlled_models.py     Nested elastic-net training and calibration
  05_validate_aibl_clinical_proxy.py      Four-variable AIBL sensitivity model
  06_vae_sensitivity_analysis.py          Adjusted exploratory analyses
  07_build_nonoverlap_adni_validation.py  Additional ADNI evaluation
  08_crossfit_five_reader_benchmark.py    Per-case predictions, thresholds and fold identifiers
  09_multireader_statistics.py            Reader agreement
  final/                                 Current manuscript statistics and figure modules
0_shared_input_preparation/               Existing cohort preparation utilities
1_discovery_subtype_model/                Existing VAE training implementation
2_discovery_characterization/             Existing exploratory characterization
3_AI_vs_Clinician_Analysis/               Earlier workflows retained for provenance
4_external_contextualization/             A4, AIBL and HABS-HD analyses
5_final_evidence_synthesis/               Earlier synthesis utility
data/                                    Local restricted inputs, excluded from Git
results/                                 Generated results and logs, excluded from Git
```

`run_analysis_pipeline.py` defines the analysis used for this manuscript. Earlier scripts retained in directories 3 and 5 are not invoked by this entry. Do not combine their historical predictions with the current case-level outputs.

## Software installation

Use Python 3.10 or later and R with the packages listed in `install_R_packages.R`. R 4.4.2 is suitable for the supplied entry. Install Python packages in a dedicated environment.

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# Linux or a virtual machine
# source .venv/bin/activate
python -m pip install -r requirements.txt
```

Install R packages explicitly from R.

```r
source("/absolute/path/to/AD_Multimodal_Study/install_R_packages.R", encoding = "UTF-8")
```

The existing Python dependency ranges are retained. Each run records the actual Python package versions, platform and interpreter in `results/python_environment.json`. The R statistics stage records `results/R_sessionInfo.txt` and `results/R_package_versions.csv`. Archive these environment records with a completed run before creating the manuscript's repository release.

## Input preparation

Obtain the cohort data through the applicable ADNI, A4, AIBL and HABS-HD access procedures. Participant-level inputs and trained objects containing restricted material are not distributed in this repository. `INPUT_MANIFEST.json` gives the exact filenames, join keys and analysis roles.

Access entry points are the [ADNI data portal](https://adni.loni.usc.edu/), the [LONI Image and Data Archive](https://ida.loni.usc.edu/) for the applicable ADNI/AIBL collections, the [A4 data-sharing site](https://www.a4studydata.org/), and the [Institute for Translational Research portal](https://apps.unthsc.edu/itr/) for HABS-HD. Request the relevant cohort access before obtaining the input tables. The repository's derived discovery files are prepared with the retained `0_shared_input_preparation` utilities and held under the cohort agreements.

The manifest records byte counts and SHA-256 fingerprints for the author-held input snapshot of 23 November 2026. These identify the files used to assemble this workflow; the date does not describe a new cohort release. Each analysis run records its actual input fingerprints in `results/input_versions.json`.

Place the files below under the `data_root` specified in `analysis_config.json`.

```text
data/
  ADNI_Raw_Data/
    LINES/Diagnostic Summary.csv
    LINES/Subject Demographics.csv
    LINES/Mini-Mental State Examination (MMSE).csv
    APOE/ApoE Genotyping - Results.csv
    CSF/UPENN CSF Biomarker Master Alzbio3.csv
    CSF/UPENN CSF Biomarkers Roche Elecsys.csv
    sMRI/UCSF - Cross-Sectional FreeSurfer (7.x).csv
    BSI/FOXLABBSI_02Mar2026.csv
  Derived_Inputs/Discovery_CSF_Cohort/
    Clinical_data.csv
    RNA_plasma.csv
    metabolites.csv
    Womac_score_pain_function.csv
  Analysis_Inputs/
    AI_vs_Clinician_Test/independent_test_set.csv
    AI_vs_Clinician_Test/Expert_Predictions_Long.csv
    AIBL_Feasibility_Gate/02_aibl_eligible_mci_to_ad_conversion_cohort.csv
    VAE_Output/subtype_assignments.csv
    VAE_Output/latent_representations.csv
    VAE_Output/vae_summary.json
    VAE_Output/vae_artifacts.pkl
    VAE_Output/K2_Subtype_Assignments.csv
    Clinical_Classifier/ADNI_Labeled_For_Classifier.csv
    Clinical_Classifier/ADNI_RF_Classifier.rds
    A4/A4_Baseline_Integrated.csv
    A4/A4_CDR_Longitudinal.csv
    HABS/HABS_Baseline.csv
  AIBL_Raw_Data/Data_extract_3.3.0/
    aibl_pdxconv_01-Jun-2018.csv
    aibl_cdr_01-Jun-2018.csv
    aibl_ptdemog_01-Jun-2018.csv
    aibl_apoeres_01-Jun-2018.csv
  Figure_Templates/Figure 1.tif
  Figure_Templates/Figure 2.tif
```

The four historical discovery filenames are retained because existing scripts read them. `RNA_plasma.csv` contains structural MRI measures, `metabolites.csv` contains CSF measurements, and `Womac_score_pain_function.csv` supplies the fixed discovery identifiers/outcomes. Their filenames do not define their scientific meaning.

The reader input contains 196 unique screened cases identified by `ID` and `RID`. It supplies the archived visit-specific values for the 30 MRI measures. `Expert_Predictions_Long.csv` must contain the observed `CaseID`, `Expert`, `Stage1_Prob` and `Stage2_Prob`, with five ratings per case and probabilities on [0, 1]. The code never generates reader scores. The strict 36-month endpoint is rebuilt from diagnostic dates.

The fixed VAE assignment and latent files have one row for each of 157 participants, keyed by `ID`. `K2_Subtype_Assignments.csv` supplies `K3_Subtype` as the canonical label reference. Its labels are matched by ID and checked for a pure permutation before use. The strict 126-person group counts are 18, 48 and 60. Group labels are never selected from outcome performance.

The VAE any-follow-up analysis reproduces the original visit-based diagnostic follow-up, with 152 participants and 82 events. It translates `m` visits into months and `y` visits into years, retaining observed diagnoses and positive follow-up. Its follow-up time is distinct from the calendar-date calculation used for strict 36-month eligibility. The resulting participant table is generated before the BSI workflow reads it.

## Configuration and execution

Edit only the locations in `analysis_config.json`. Relative locations resolve against that file, independently of the shell's working directory.

```json
{
  "data_root": "data",
  "output_dir": "results",
  "rscript": "Rscript"
}
```

On Windows, an absolute R executable can be entered as `D:/R/R-4.4.2/bin/Rscript.exe`. On Linux, `Rscript` works when R is on PATH.

Check file availability without running analyses.

```bash
python analysis_pipeline/run_analysis_pipeline.py --check-inputs
```

Run the complete workflow.

```bash
python analysis_pipeline/run_analysis_pipeline.py
```

From R, set the interpreter belonging to the Python environment, then use `source()`.

```r
Sys.setenv(AD_MULTIMODAL_PYTHON = "D:/AD_Multimodal_Study/.venv/Scripts/python.exe")
source("D:/AD_Multimodal_Study/run_all.R", encoding = "UTF-8")
```

The R entry uses the current R installation's Rscript executable. In a Linux virtual machine, set the Python interpreter to that environment's `bin/python` and use the corresponding repository path.

Components can be run separately, in this order. The core stage must precede VAE follow-up analyses and figure production. The extension stage also reads the ADNI demographic file produced by the core input preparation.

```bash
python analysis_pipeline/run_analysis_pipeline.py --component core
python analysis_pipeline/run_analysis_pipeline.py --component vae
python analysis_pipeline/run_analysis_pipeline.py --component extensions
python analysis_pipeline/run_analysis_pipeline.py --component figures
```

Each component starts fresh subprocesses, writes named logs and stops if a stage fails. Completion markers are written only after the requested component finishes. Inspect `results/logs/` and `results/run_commands.json` for the actual commands.

## Fixed computational definitions

The supervised model uses the ordered clinical, CSF and MRI features in `FEATURE_MANIFEST.json`. Median imputation with missingness indicators and standardization are fitted within training pipelines. Elastic-net logistic regression uses `saga`, a maximum of 10,000 iterations, C values of 0.05, 0.1, 0.5, 1 and 5, and mixing values of 0, 0.5 and 1. The development assessment uses repeated five-fold outer cross-validation and three-fold inner tuning, with base seed 20260710. The final development search uses five-fold cross-validation. Platt calibration is fitted from out-of-fold logits.

For the reader comparison, five-fold cross-fitting is applied to the 126 development cases. Tuning uses three-fold inner cross-validation. Each held-out case receives the threshold estimated from five-fold predictions within its outer training set. The 29 reader cases outside development receive the full development model and the single five-fold development out-of-fold threshold. The repeated development assessment and the reader cross-fitting procedure are separate calculations.

The 30 screened MRI values replace the corresponding raw extraction values for the eligible development cases and reader cases, matched by RID. Clinical features, CSF extraction, outcomes and reader scores retain their existing definitions. The external ADNI MRI values come from the designated raw extraction.

Rule C triggers inclusively at Stage 2 probability 0.40 through 0.60. Outside that interval, reader classification uses 0.50. Within the interval, the primary classification uses the case's training-only AI threshold. Common-0.50 analyses use 0.50 for both sources. Sensitivity analyses use 0.45–0.55, 0.35–0.65 and 0.30–0.70 intervals. No threshold is selected to maximize performance on the reader cases.

Reader-macro measures average the five readers' measures. Pooled measures first average their probabilities within each case. Case bootstraps retain all five ratings for each sampled case. Marginal intervals in the main table use outcome-stratified case resampling; paired changes use the retained paired case bootstrap. Seeds and replicate counts remain explicit in each module.

External intercept updates are fitted to the evaluation cohort's own outcomes and are reported as apparent recalibration. Their results are stored separately from frozen-model evaluation. The AIBL 28-case proxy retains its own original training labels and threshold procedure, separate from the 34-case four-variable analysis.

## Outputs and manuscript mapping

| Manuscript analysis | Computation | Principal output |
|---|---|---|
| Cohort definitions | 01 and 03 | `results/core/*endpoints.csv`, strict cohort tables |
| Model training and external ADNI evaluation | 04 and 07 | `leakage_free_model_summary.json`, `nonoverlapping_adni_validation_*.csv` |
| Per-case reader model estimates | 08 | `primary_153_case_predictions.csv`, `ablation_153_case_predictions.csv`, fold tuning tables |
| Reader agreement | 09 | `multireader_reliability.csv` |
| Trigger widths and common threshold | `final/reader_intervals.R` | `results/intervals_primary/`, `intervals_ablation/` |
| Overlap strata and paired changes | `final/reader_overlap.R` | `results/reader_overlap/` |
| Study precision | `final/reader_precision.R` | `results/reader_precision/` |
| NRI and decision curves | `final/reader_nri_dca.R` | `results/reader_nri_dca/` |
| Table 1 and calibration bins | `final/reader_main_table.R` | `results/reader_main_table/` |
| Censored follow-up | `final/reader_survival.R`, `discovery_survival.R` | `results/reader_survival/`, `discovery_survival/` |
| Selection frequency and regularization | `final/model_stability.py` | `results/model_stability/` |
| Missingness and intercept updates | `final/external_calibration.R` | `results/external_missingness_calibration/` |
| Missing-indicator ablation | `final/missing_indicator_ablation.py` | `results/missing_indicator_ablation/` |
| VAE fixed groups and adjusted effects | 06 and characterization scripts | `results/core/vae_*.csv`, `vae_signatures/`, `vae_validation/`, `vae_neuroimaging/` |
| VAE longitudinal/CSF analyses | Characterization scripts and `final/vae_followup.R` | `vae_bsi/`, `vae_biomarkers/`, `conversion_contrasts/`, `vae_followup/` |
| AIBL three-variable proxy | `4_external_contextualization/AIBL reduced-feature external validation.R` | `results/aibl_three_feature/` |
| A4 and HABS-HD | Respective external scripts | `results/a4/`, `results/habs_hd/` |
| Extension missingness and AIBL exact intervals | `final/external_context_summary.R` | `results/external_context_summary/` |
| Figures 1–3 | `final/reader_figures.R`, `figure3.R` | `results/figures/` |
| Figure 4/5 scientific panels and related supplementary plots | Existing VAE characterization scripts | `vae_signatures/`, `vae_neuroimaging/`, `vae_validation/` |

The combined benchmark has 306 rows because it contains two models. Downstream reader statistics read the separate **153-row primary file**. Never join all 306 rows to reader ratings without restricting the model.

Figure 1 retains the existing workflow artwork. Its numeric panels and all Figure 2 panels are regenerated from the computed outputs. The templates must retain their original dimensions, 6669 × 4989 and 3214 × 2194 pixels. Figures 4 and 5 and supplementary composites use panels produced by the characterization scripts. Final journal numbering, captions and panel assembly belong to the manuscript files. This code does not recreate the Word submission package or claim identical pagination or pixels for every composite.

## Reproduction checks

The core entry compares newly computed reader and external model results against `FROZEN_AGGREGATE_CHECKS.json` and writes `results/reproduction_comparison.csv`. The extension stage writes `results/extension_reproduction_comparison.csv` for the separate AIBL and HABS-HD results. These aggregate references are used only for verification. They are never supplied to model fitting or probability generation.

Key reference results are 153 reader cases with 77 events and 124 development overlaps. Stage 2 reader-macro specificity is 0.360526 and Rule C specificity is 0.415789. Corresponding sensitivities are 0.883117 and 0.844156. Pooled Stage 2 false positives/false negatives are 50/7, compared with 44/10 for the training-threshold Rule C. Pooled AUC values are 0.732656 and 0.715567, with standalone AI AUC 0.647984. A common 0.50 boundary gives 44 false positives and 9 false negatives.

Successful file checks or syntax checks do not establish a completed numerical reproduction. Confirm the aggregate comparison and review the saved environment records after running the complete workflow with the specified data versions.

## Upstream VAE and classifier regeneration

The main entry uses the frozen VAE fit and clinical random forest used for the manuscript. Training code remains available. To regenerate a VAE from the fixed discovery inputs, use the existing trainer and retain its defaults for three latent dimensions, 300 epochs, batch size 32 and learning rate 0.0005.

```bash
python 1_discovery_subtype_model/vae_clustering.py --cohort A \
  --input_dir data/Derived_Inputs/Discovery_CSF_Cohort \
  --output_dir results/vae_retraining
```

The optional `analysis_pipeline/final/train_a4_classifier.R` reconstructs the clinical random forest from the separate fixed training labels. It requires `AD_MULTIMODAL_DATA_ROOT` and `AD_MULTIMODAL_RESULTS_DIR` to be set to absolute paths before sourcing. It writes to `results/a4_classifier_retraining/` and leaves the primary frozen classifier unchanged.

A newly trained latent solution must be matched by participant ID and checked against the fixed partition before it is used with manuscript group labels. Record its software and hardware environment. Retraining does not automatically replace the frozen manuscript inputs.

## Data, license and contact

Code is distributed under the existing MIT `LICENSE`. Cohort data remain subject to their original access agreements. The repository excludes participant-level ratings, dates, predictions and model artifacts derived from restricted data. Study authors can provide the applicable input preparation details and access instructions through the manuscript's correspondence contacts.
