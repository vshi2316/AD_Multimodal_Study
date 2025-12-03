import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

## Load APOE genotype data
try:
    apoe_raw = pd.read_csv("ApoE_Genotyping_Results.csv", encoding="gbk")
except:
    apoe_raw = pd.read_csv("ApoE_Genotyping_Results.csv", encoding="utf-8-sig")

## Extract and deduplicate
apoe_core = apoe_raw[["PTID", "GENOTYPE"]].copy()
apoe_dedup = apoe_core.drop_duplicates(subset="PTID", keep="first")
apoe_clean = apoe_dedup.dropna(subset=["GENOTYPE"]).reset_index(drop=True)

## Extract APOE4 features
def extract_apoe4_features(genotype):
    geno_clean = str(genotype).strip().lower().replace(" ", "")
    apoe4_dosage = geno_clean.count("4")
    apoe4_status = 1 if apoe4_dosage >= 1 else 0
    return apoe4_dosage, apoe4_status

apoe_features = apoe_clean["GENOTYPE"].apply(extract_apoe4_features)
apoe_clean[["APOE4_DOSAGE", "APOE4_STATUS"]] = pd.DataFrame(
    apoe_features.tolist(), index=apoe_clean.index
)

## Output
apoe_final = apoe_clean.rename(columns={"PTID": "ID"}).copy()
apoe_final = apoe_final[["ID", "APOE4_DOSAGE", "APOE4_STATUS"]]
apoe_final.to_csv("APOE_genetics.csv", index=False)
