import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

## Load PET imaging data
def read_csv_robust(filepath):
    encodings = ['utf-8-sig', 'gbk', 'latin-1']
    for encoding in encodings:
        try:
            return pd.read_csv(filepath, encoding=encoding)
        except:
            continue
    return pd.read_csv(filepath)

amyloid_pet = read_csv_robust("Amyloid_PET.csv")
tau_pet = read_csv_robust("Tau_PET.csv")

## Extract baseline data
def extract_baseline(df):
    if 'VISCODE' not in df.columns:
        return df
    baseline_codes = ['bl', 'BL', 'sc', 'SC', 'v01', 'V01']
    df_bl = df[df['VISCODE'].isin(baseline_codes)].copy()
    if len(df_bl) == 0:
        id_col = 'PTID' if 'PTID' in df.columns else 'RID'
        df_bl = df.sort_values('VISCODE').groupby(id_col).first().reset_index()
    return df_bl

amyloid_baseline = extract_baseline(amyloid_pet)
tau_baseline = extract_baseline(tau_pet)

## Extract PET SUVR features
def extract_pet_features(df, roi_keywords):
    features = []
    for col in df.columns:
        col_lower = col.lower()
        if 'suvr' in col_lower and any(roi in col_lower for roi in roi_keywords):
            features.append(col)
    if len(features) < 5:
        suvr_cols = [col for col in df.columns if 'suvr' in col.lower()]
        features = suvr_cols[:15]
    return features

amyloid_roi_keywords = ['frontal', 'temporal', 'parietal', 'cingulate', 'precuneus', 'global', 'composite', 'cortical']
amyloid_features = extract_pet_features(amyloid_baseline, amyloid_roi_keywords)

tau_roi_keywords = ['entorhinal', 'hippocampus', 'temporal', 'fusiform', 'parahippocampal', 'global', 'composite']
tau_features = extract_pet_features(tau_baseline, tau_roi_keywords)

## Extract ID and features
amyloid_id_col = 'PTID' if 'PTID' in amyloid_baseline.columns else 'RID'
amyloid_data = amyloid_baseline[[amyloid_id_col] + amyloid_features].copy()
amyloid_data.rename(columns={amyloid_id_col: 'ID'}, inplace=True)
amyloid_data.columns = ['ID'] + [f'Amyloid_{col}' for col in amyloid_features]

tau_id_col = 'PTID' if 'PTID' in tau_baseline.columns else 'RID'
tau_data = tau_baseline[[tau_id_col] + tau_features].copy()
tau_data.rename(columns={tau_id_col: 'ID'}, inplace=True)
tau_data.columns = ['ID'] + [f'Tau_{col}' for col in tau_features]

## Merge Amyloid and Tau PET data
pet_merged = pd.merge(amyloid_data, tau_data, on='ID', how='inner')

all_features = [col for col in pet_merged.columns if col != 'ID']

## Handle missing values
for col in all_features:
    if pet_merged[col].isnull().sum() > 0:
        median_val = pet_merged[col].median()
        pet_merged[col].fillna(median_val, inplace=True)

## Handle negative values for log transformation
features_matrix = pet_merged[all_features].copy()

min_val = features_matrix.min().min()
if min_val <= 0:
    for col in all_features:
        col_min = features_matrix[col].min()
        if col_min <= 0:
            offset = -col_min + 0.01
            features_matrix[col] = features_matrix[col] + offset

## Normalization: sum + log + pareto
sample_sums = features_matrix.sum(axis=1)
median_sum = sample_sums.median()
sum_norm = features_matrix.div(sample_sums, axis=0) * median_sum

log_norm = np.log2(sum_norm + 1)

scaler = StandardScaler()
pareto_norm = scaler.fit_transform(log_norm)
pareto_norm_df = pd.DataFrame(pareto_norm, columns=all_features, index=features_matrix.index)

## Output
final_pet = pd.concat([
    pet_merged[['ID']].reset_index(drop=True),
    pareto_norm_df.reset_index(drop=True)
], axis=1)

final_pet.to_csv("PET_features.csv", index=False)
