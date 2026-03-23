[README.md](https://github.com/user-attachments/files/26175126/README.md)
# AD Multimodal Study

This folder is the **GitHub upload-ready manuscript-aligned code package**.

Manuscript title:

**Multimodal Variational Autoencoder-Driven Deep Learning Uncovers Clinically Masked Neuroimaging Subtypes and Progression Risk Stratification in Alzheimer's Disease**

## Upload rule

Use the files in this folder directly for GitHub upload.

## Ordered root scripts

1. `01_step1_preprocess_APOE.py`
2. `02_step2_preprocess_CSF.py`
3. `03_step3_preprocess_Clinical.py`
4. `04_step4_preprocess_sMRI.py`
5. `05_step6_create_outcome.py`
6. `06_step7_integrate_cohorts.py`
7. `07_step8_vae_clustering.py`
8. `08_step9A_cross_cohort_analysis.py`
9. `09_step9B_biomarker_validation.py`
10. `10_step10_differential_analysis.R`
11. `11_step11_predictive_modeling.R`
12. `12_step12_cluster_signatures.R`
13. `13_step13_conversion_differential.R`
14. `14_step14_cluster_validation.R`
15. `15_step15_longitudinal_bsi_validation.R`
16. `16_step16_habs_validation.R`
17. `17_step17_shap_analysis.R`
18. `18_step18_evidence_synthesis.R`
19. `19_step19_adni_discovery.R`
20. `20_step20_aibl_validation.R`
21. `21_step21_a4_validation.R`
22. `22_step22_neuroimaging_endotypes.R`

## AI vs Expert folder

Folder: `AI`

- `AI_01_Step_1_Prepare_Test.R`
- `AI_02_Step_2_AI_Prediction.py`
- `AI_03_Step_3_Expert_Assessment_Workflow.R`
- `AI_04_Step_4_AI_vs_Expert_Comparison_Analysis.R`

## Important manuscript alignment notes

- The primary discovery model uses **37 variables**: 3 CSF, 4 clinical/genetic, and 30 structural MRI features.
- `Age` and `Gender` are excluded from VAE input.
- `FAQ`, `ADAS13`, and `CDRSB` are excluded from primary model features to avoid circularity.
- A4 and AIBL are implemented through **external latent projection + centroid-based subtype assignment**.
- HABS is implemented as **framework adaptation with Firth logistic regression**, not direct subtype transfer.
- PET preprocessing is **not included** in the final upload set because it is not part of the final manuscript analysis path.

## Files from the old GitHub repo that should be deleted

- `step5_preprocess_PET.py`
- `step7_Cohort Integration.py`
- `step15_cross_modal_validation.R`
- `step18_meta_analysis.R`
- `step19_ADNI_discovery.R`
- `step20_AIBL _Validation.R`
- `step21_A4_validation.R`
- `AI_vs_Clinician_Analysis/Step 4 AI vs Expert Comparison Analysis`

See `DELETE_OLD_GITHUB_FILES.md` and `UPLOAD_ORDER.md`.
