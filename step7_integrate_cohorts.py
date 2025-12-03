import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

## Load preprocessed data files
apoe = pd.read_csv("APOE_genetics.csv")
csf = pd.read_csv("metabolites.csv")
clinical = pd.read_csv("clinical_cognitive_scores.csv")
mri = pd.read_csv("structural_MRI_features.csv")
pet = pd.read_csv("PET_imaging_features.csv")
outcome = pd.read_csv("AD_conversion_outcomes.csv")

## Merge all modalities by ID
data = apoe.copy()
for df in [csf, clinical, mri, pet, outcome]:
    if "ID" in df.columns:
        data = pd.merge(data, df, on="ID", how="inner")

print(f"Integrated sample size: {len(data)}")

## Separate ADNI and AIBL cohorts
if "Cohort" not in data.columns:
    data["Cohort"] = data["ID"].apply(lambda x: "ADNI" if str(x).startswith("0") else "AIBL")

cohort_a = data[data["Cohort"] == "ADNI"].copy()
cohort_b = data[data["Cohort"] == "AIBL"].copy()

print(f"Cohort A (ADNI): {len(cohort_a)}")
print(f"Cohort B (AIBL): {len(cohort_b)}")

## Feature harmonization
feature_cols = [col for col in data.columns if col not in ["ID", "Cohort", "AD_Conversion", "Time_to_Event", "Followup_Years"]]
numeric_cols = data[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

## Standardize features within each cohort
scaler_a = StandardScaler()
scaler_b = StandardScaler()

if len(numeric_cols) > 0:
    cohort_a[numeric_cols] = scaler_a.fit_transform(cohort_a[numeric_cols])
    cohort_b[numeric_cols] = scaler_b.fit_transform(cohort_b[numeric_cols])

## Save integrated cohorts
cohort_a.to_csv("Cohort_A_Integrated.csv", index=False)
cohort_b.to_csv("Cohort_B_Integrated.csv", index=False)

print("\nIntegration complete:")
print(f"- Cohort_A_Integrated.csv: {len(cohort_a)} samples, {len(cohort_a.columns)} features")
print(f"- Cohort_B_Integrated.csv: {len(cohort_b)} samples, {len(cohort_b.columns)} features")
