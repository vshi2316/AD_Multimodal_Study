import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

## Load CSF data files
def read_csv_robust(filepath):
    encodings = ['utf-8-sig', 'gbk', 'latin-1']
    for encoding in encodings:
        try:
            return pd.read_csv(filepath, encoding=encoding)
        except:
            continue
    return pd.read_csv(filepath, low_memory=False)

strem2_pgrn = read_csv_robust("CSF_sTREM2_PGRN.csv")
roche = read_csv_robust("CSF_Roche_Elecsys.csv")
alzbio3 = read_csv_robust("CSF_Alzbio3.csv")

## Extract core biomarkers
roche_core = roche[["PTID", "RID", "ABETA40", "ABETA42", "TAU", "PTAU", "BATCH"]].copy()
roche_core["ABETA42_ABETA40_RATIO"] = roche_core["ABETA42"] / roche_core["ABETA40"]
roche_core.rename(columns={"TAU": "TAU_TOTAL", "PTAU": "PTAU181"}, inplace=True)

alzbio3_core = alzbio3[["RID", "ABETA", "TAU", "PTAU", "BATCH"]].copy()
alzbio3_core.rename(columns={"ABETA": "ABETA_ALZ",
                              "TAU": "TAU_TOTAL_ALZ",
                              "PTAU": "PTAU181_ALZ"}, inplace=True)

strem2_pgrn_core = strem2_pgrn[["RID", "WU_STREM2CORRECTED", "MSD_PGRNCORRECTED"]].copy()
strem2_pgrn_core.rename(columns={"WU_STREM2CORRECTED": "STREM2",
                                  "MSD_PGRNCORRECTED": "PGRN"}, inplace=True)

## Merge three datasets
merge1 = pd.merge(roche_core, alzbio3_core, on="RID", how="inner")
merge_final = pd.merge(merge1, strem2_pgrn_core, on="RID", how="inner")

drop_cols = ["RID", "ABETA_ALZ", "TAU_TOTAL_ALZ", "PTAU181_ALZ", "BATCH_y"]
merge_final = merge_final.drop(columns=drop_cols).rename(columns={"BATCH_x": "BATCH"})

## Batch correction: center each batch to overall mean
feature_cols = [col for col in merge_final.columns if col not in ["PTID", "BATCH"]]
print(f"Applying batch correction to {len(feature_cols)} features across {merge_final['BATCH'].nunique()} batches")

for feat in feature_cols:
    overall_mean = merge_final[feat].mean()
    batch_mean = merge_final.groupby("BATCH")[feat].transform("mean")
    merge_final[feat] = merge_final[feat] - batch_mean + overall_mean

merge_final = merge_final.drop(columns="BATCH")

## Handle missing values
merge_final = merge_final.replace([np.inf, -np.inf], np.nan)
for feat in feature_cols:
    merge_final[feat] = merge_final.groupby("PTID")[feat].transform(lambda x: x.fillna(x.mean()))
    merge_final[feat] = merge_final[feat].fillna(merge_final[feat].mean())

## Normalization: sum + log + pareto
data_for_norm = merge_final[["PTID"] + feature_cols].copy()
data_for_norm.rename(columns={"PTID": "ID"}, inplace=True)

features = data_for_norm[feature_cols].copy()
sample_sums = features.sum(axis=1)
median_sum = sample_sums.median()
sum_norm = features.div(sample_sums, axis=0) * median_sum
log_norm = np.log2(sum_norm + 1)

scaler = StandardScaler()
pareto_norm = scaler.fit_transform(log_norm)
pareto_norm_df = pd.DataFrame(pareto_norm, columns=feature_cols, index=data_for_norm.index)

final_data = pd.concat([data_for_norm[["ID"]], pareto_norm_df], axis=1)
final_cols = ["ID"] + feature_cols
final_csf = final_data[final_cols].copy()

## Output with quality control summary
final_csf.to_csv("metabolites.csv", index=False)
print(f"\nCSF preprocessing complete:")
print(f"- Final sample size: {len(final_csf)}")
print(f"- Features: {len(feature_cols)}")
print(f"- Missing values: {final_csf.isnull().sum().sum()}")
