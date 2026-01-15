import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

## Load FreeSurfer sMRI data
def read_csv_robust(filepath):
    encodings = ['gbk', 'utf-8-sig', 'latin-1']
    for encoding in encodings:
        try:
            return pd.read_csv(filepath, encoding=encoding)
        except:
            continue
    return pd.read_csv(filepath)

smri_raw = read_csv_robust("FreeSurfer_sMRI.csv")

## Extract baseline data
def extract_baseline(df):
    if 'VISCODE' not in df.columns:
        return df
    baseline_codes = ['bl', 'BL', 'sc', 'SC']
    df_bl = df[df['VISCODE'].isin(baseline_codes)].copy()
    if len(df_bl) == 0:
        id_col = 'PTID' if 'PTID' in df.columns else 'RID'
        df_bl = df.sort_values('VISCODE').groupby(id_col).first().reset_index()
    return df_bl

smri_baseline = extract_baseline(smri_raw)

## Select AD-relevant brain regions (aligned with Methods section)
core_features = []
feature_patterns = {
    'Hippocampus': ['hippocampus'],
    'Entorhinal': ['entorhinal'],
    'Temporal': ['temporal'],
    'Parietal': ['parietal'],
    'Frontal': ['frontal'],
    'Cingulate': ['cingulate'],
    'Precuneus': ['precuneus'],
    'Ventricular': ['ventric'],
    'WholeBrain': ['wholebrain', 'intracranial']
}

for region, patterns in feature_patterns.items():
    matched_cols = []
    for pattern in patterns:
        matched = [col for col in smri_baseline.columns if pattern in col.lower()]
        matched_cols.extend(matched)
    matched_cols = list(set(matched_cols))
    core_features.extend(matched_cols)
    print(f"{region}: {len(matched_cols)} features")

core_features = list(set(core_features))
print(f"\nTotal selected features: {len(core_features)}")

## Extract ID and features
id_col = 'PTID' if 'PTID' in smri_baseline.columns else 'RID'
smri_features = smri_baseline[[id_col] + core_features].copy()
smri_features.rename(columns={id_col: 'ID'}, inplace=True)

## Handle missing values
for col in core_features:
    if smri_features[col].isnull().sum() > 0:
        median_val = smri_features[col].median()
        smri_features[col].fillna(median_val, inplace=True)

## Handle negative values for log transformation
features_matrix = smri_features[core_features].copy()
for col in core_features:
    min_val = features_matrix[col].min()
    if min_val <= 0:
        offset = -min_val + 1
        features_matrix[col] = features_matrix[col] + offset

## Normalization: sum + log + pareto
sample_sums = features_matrix.sum(axis=1)
median_sum = sample_sums.median()
sum_norm = features_matrix.div(sample_sums, axis=0) * median_sum
log_norm = np.log2(sum_norm + 1)

scaler = StandardScaler()
pareto_norm = scaler.fit_transform(log_norm)
pareto_norm_df = pd.DataFrame(pareto_norm, columns=core_features, index=features_matrix.index)

## Output
final_smri = pd.concat([smri_features[['ID']].reset_index(drop=True),
                        pareto_norm_df.reset_index(drop=True)], axis=1)

final_smri.to_csv("sMRI_features.csv", index=False)
print(f"\nsMRI preprocessing complete:")
print(f"- Final sample size: {len(final_smri)}")
print(f"- Features: {len(core_features)}")
