import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

## Load longitudinal diagnostic data
def read_csv_robust(filepath):
    encodings = ['gbk', 'utf-8-sig', 'latin-1']
    for encoding in encodings:
        try:
            return pd.read_csv(filepath, encoding=encoding)
        except:
            continue
    return pd.read_csv(filepath)

diag_data = read_csv_robust("Diagnostic_Summary.csv")

## Extract baseline records
def extract_baseline_records(df):
    if 'VISCODE' not in df.columns:
        return df
    baseline_codes = ['bl', 'BL', 'sc', 'SC']
    baseline_mask = df['VISCODE'].isin(baseline_codes)
    df_bl = df[baseline_mask].copy()
    if len(df_bl) == 0:
        df_bl = df.sort_values(['PTID', 'VISCODE']).groupby('PTID').first().reset_index()
    return df_bl

baseline_data = extract_baseline_records(diag_data)

## Filter baseline MCI patients (aligned with Methods: MCI at baseline)
diagnosis_col = 'DIAGNOSIS'

def is_mci_diagnosis(dx_value):
    if pd.isna(dx_value):
        return False
    if dx_value == 2:
        return True
    dx_str = str(dx_value).upper()
    mci_keywords = ['MCI', 'MILD COGNITIVE IMPAIRMENT', 'EMCI', 'LMCI']
    return any(kw in dx_str for kw in mci_keywords)

mci_baseline = baseline_data[baseline_data[diagnosis_col].apply(is_mci_diagnosis)].copy()
mci_ptid_list = mci_baseline['PTID'].unique()

print(f"Baseline MCI patients: {len(mci_ptid_list)}")

## Extract follow-up visits
all_viscodes = diag_data['VISCODE'].unique()
followup_viscodes = []

for month in [1, 3, 6, 12, 18, 24, 30, 36, 48, 60, 72]:
    for prefix in ['m', 'M', 'v', 'V']:
        code = f"{prefix}{month:02d}" if month < 10 else f"{prefix}{month}"
        if code in all_viscodes:
            followup_viscodes.append(code)

alt_codes = ['m1', 'm3', 'm6', 'm12', 'm18', 'm24', 'm30', 'm36', 
             'M1', 'M3', 'M6', 'M12', 'M18', 'M24', 'M30', 'M36']
for code in alt_codes:
    if code in all_viscodes and code not in followup_viscodes:
        followup_viscodes.append(code)

print(f"Follow-up visit codes: {len(followup_viscodes)}")

## Get follow-up data for MCI patients
followup_data = diag_data[
    (diag_data['PTID'].isin(mci_ptid_list)) &
    (diag_data['VISCODE'].isin(followup_viscodes))
].copy()

## Identify converters (MCI to AD dementia per NIA-AA criteria)
def is_ad_diagnosis(dx_value):
    if pd.isna(dx_value):
        return False
    if dx_value == 3:
        return True
    dx_str = str(dx_value).upper()
    ad_keywords = ['AD', 'ALZHEIMER', 'DEMENTIA']
    exclude_keywords = ['MCI']
    if any(ex in dx_str for ex in exclude_keywords):
        return False
    return any(kw in dx_str for kw in ad_keywords)

conversion_results = []

## Check conversion status for each MCI patient
for ptid in mci_ptid_list:
    patient_followup = followup_data[followup_data['PTID'] == ptid]
    converted = False
    
    if len(patient_followup) > 0:
        for idx, row in patient_followup.iterrows():
            if is_ad_diagnosis(row[diagnosis_col]):
                converted = True
                break
    
    conversion_results.append({
        'ID': ptid,
        'AD_Conversion': 1 if converted else 0
    })

outcome_df = pd.DataFrame(conversion_results)
outcome_final = outcome_df[['ID', 'AD_Conversion']].copy()

## Output with summary statistics
outcome_final.to_csv("AD_Conversion.csv", index=False)

converters = outcome_final['AD_Conversion'].sum()
conversion_rate = converters / len(outcome_final) * 100

print(f"\nOutcome creation complete:")
print(f"- Total MCI patients: {len(outcome_final)}")
print(f"- Converters to AD: {converters} ({conversion_rate:.1f}%)")
print(f"- Non-converters: {len(outcome_final) - converters}")
