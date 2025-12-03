import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import warnings
warnings.filterwarnings("ignore")

## Load clinical assessment files
def read_csv_robust(filepath):
    encodings = ['utf-8-sig', 'gbk', 'latin-1']
    for encoding in encodings:
        try:
            return pd.read_csv(filepath, encoding=encoding)
        except:
            continue
    return pd.read_csv(filepath)

baseline_changes = read_csv_robust("Diagnostic_Summary_Baseline_Changes.csv")
diagnostic = read_csv_robust("Diagnostic_Summary.csv")
adas = read_csv_robust("ADAS_Cognitive_Behavior.csv")
cdr = read_csv_robust("Clinical_Dementia_Rating.csv")
faq = read_csv_robust("Functional_Activities_Questionnaire.csv")
demographics = read_csv_robust("Subject_Demographics.csv")
gds = read_csv_robust("Geriatric_Depression_Scale.csv")
mmse = read_csv_robust("MMSE.csv")

data_dict = {
    'baseline_changes': baseline_changes,
    'diagnostic': diagnostic,
    'adas': adas,
    'cdr': cdr,
    'faq': faq,
    'demographics': demographics,
    'gds': gds,
    'mmse': mmse
}

## Extract baseline visits
def extract_baseline(df):
    if 'VISCODE' not in df.columns:
        return df
    baseline_codes = ['bl', 'BL', 'sc', 'SC']
    df_bl = df[df['VISCODE'].isin(baseline_codes)].copy()
    if len(df_bl) == 0:
        df_bl = df.sort_values('VISCODE').groupby('PTID').first().reset_index()
    return df_bl

baseline_dict = {key: extract_baseline(df) for key, df in data_dict.items()}

## Extract core features from each dataset
diag_features = baseline_dict['diagnostic'][['PTID', 'DIAGNOSIS']].copy()
diagnosis_map = {'CN': 1, 'MCI': 2, 'AD': 3, 'Dementia': 3}
diag_features['DIAGNOSIS'] = diag_features['DIAGNOSIS'].map(diagnosis_map)

adas_features = baseline_dict['adas'][['PTID', 'TOTAL13']].copy()
adas_features.rename(columns={'TOTAL13': 'ADAS13'}, inplace=True)

cdr_features = baseline_dict['cdr'][['PTID', 'CDRSB']].copy()

faq_features = baseline_dict['faq'][['PTID', 'FAQTOTAL']].copy()

mmse_features = baseline_dict['mmse'][['PTID', 'MMSCORE']].copy()
mmse_features.rename(columns={'MMSCORE': 'MMSE'}, inplace=True)

demo_df = baseline_dict['demographics']
demo_features = demo_df[['PTID']].copy()
demo_features['AGE'] = 2010 - demo_df['PTDOBYY']
demo_features['SEX'] = demo_df['PTGENDER'].map({1: 0, 2: 1})
demo_features['EDUCATION'] = demo_df['PTEDUCAT']

gds_features = baseline_dict['gds'][['PTID', 'GDTOTAL']].copy()
gds_features.rename(columns={'GDTOTAL': 'GDS'}, inplace=True)

## Merge all clinical features
clinical_merged = diag_features
for df in [adas_features, cdr_features, faq_features, mmse_features, demo_features, gds_features]:
    clinical_merged = pd.merge(clinical_merged, df, on='PTID', how='outer')

## Impute missing values using MICE
ptid = clinical_merged['PTID']
features = clinical_merged.drop(columns=['PTID'])

imputer = IterativeImputer(max_iter=10, random_state=42)
features_imputed = imputer.fit_transform(features)
features_imputed_df = pd.DataFrame(features_imputed, columns=features.columns, index=features.index)

clinical_imputed = pd.concat([ptid.reset_index(drop=True), features_imputed_df.reset_index(drop=True)], axis=1)

## Standardize continuous variables
categorical_cols = ['DIAGNOSIS', 'SEX']
continuous_cols = [col for col in clinical_imputed.columns if col not in ['PTID'] + categorical_cols]

scaler = StandardScaler()
clinical_imputed[continuous_cols] = scaler.fit_transform(clinical_imputed[continuous_cols])

## Output
clinical_final = clinical_imputed.rename(columns={'PTID': 'ID'})
final_cols = ['ID'] + [col for col in clinical_final.columns if col != 'ID']
clinical_final = clinical_final[final_cols]

clinical_final.to_csv("Clinical_data.csv", index=False)
