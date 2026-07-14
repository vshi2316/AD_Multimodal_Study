from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("AD_MULTIMODAL_DATA_ROOT", ".")).expanduser().resolve()
OUT = Path(os.environ.get("AD_MULTIMODAL_OUTPUT_DIR", str(HERE / "outputs"))).expanduser()
RAW = ROOT / "ADNI_Raw_Data"
TEST_FILE = ROOT / "Analysis_Inputs" / "AI_vs_Clinician_Test" / "independent_test_set.csv"
DISCOVERY_STANDARDIZED_DIR = (
    ROOT / "Derived_Inputs" / "Discovery_CSF_Cohort"
)

MRI_FILE = RAW / "sMRI" / "UCSF - Cross-Sectional FreeSurfer (7.x).csv"
DEMOG_FILE = RAW / "LINES" / "Subject Demographics.csv"
MMSE_FILE = RAW / "LINES" / "Mini-Mental State Examination (MMSE).csv"
APOE_FILE = RAW / "APOE" / "ApoE Genotyping - Results.csv"
CSF_ALZ_FILE = RAW / "CSF" / "UPENN CSF Biomarker Master Alzbio3.csv"
CSF_ROCHE_FILE = RAW / "CSF" / "UPENN CSF Biomarkers Roche Elecsys.csv"


def numeric(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.replace(r"^[<>]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def first_baseline(frame: pd.DataFrame, visit_col: str) -> pd.DataFrame:
    visit = frame[visit_col].astype(str).str.lower().str.strip()
    return frame.loc[visit.isin(["bl", "sc"])].drop_duplicates("RID", keep="first")


def load_raw_feature_table(mri_features: list[str]) -> pd.DataFrame:
    mri_cols = ["RID", "VISCODE"] + mri_features
    mri = pd.read_csv(MRI_FILE, usecols=mri_cols, low_memory=False)
    mri = first_baseline(mri, "VISCODE")
    for col in mri_features:
        mri[col] = numeric(mri[col])
    mri = mri[["RID"] + mri_features]

    demog = pd.read_csv(
        DEMOG_FILE,
        usecols=["RID", "PTGENDER", "PTEDUCAT"],
        low_memory=False,
    ).drop_duplicates("RID", keep="first")
    demog["Education"] = numeric(demog["PTEDUCAT"])
    demog.loc[demog["Education"].isna() | (demog["Education"] == 0), "Education"] = 15
    gender_text = demog["PTGENDER"].astype(str).str.lower()
    demog["Gender"] = np.where(
        gender_text.isin(["female", "2", "2.0"]),
        1.0,
        np.where(gender_text.isin(["male", "1", "1.0"]), 0.0, np.nan),
    )
    demog = demog[["RID", "Education", "Gender"]]

    mmse = pd.read_csv(MMSE_FILE, usecols=["RID", "VISCODE", "MMSCORE"], low_memory=False)
    mmse = first_baseline(mmse, "VISCODE")
    mmse["MMSE_Baseline"] = numeric(mmse["MMSCORE"])
    mmse = mmse[["RID", "MMSE_Baseline"]]

    apoe = pd.read_csv(APOE_FILE, usecols=["RID", "GENOTYPE"], low_memory=False)
    apoe = apoe.drop_duplicates("RID", keep="first")
    apoe["APOE4_Copies"] = apoe["GENOTYPE"].astype(str).str.count("4").astype(float)
    apoe = apoe[["RID", "APOE4_Copies"]]

    alz = pd.read_csv(
        CSF_ALZ_FILE,
        usecols=["RID", "VISCODE", "ABETA", "TAU", "PTAU"],
        engine="python",
        on_bad_lines="skip",
    )
    alz = first_baseline(alz, "VISCODE")
    alz["ABETA42"] = numeric(alz["ABETA"])
    alz["TAU_TOTAL"] = numeric(alz["TAU"])
    alz["PTAU181"] = numeric(alz["PTAU"])
    alz = alz[["RID", "ABETA42", "TAU_TOTAL", "PTAU181"]]

    roche = pd.read_csv(
        CSF_ROCHE_FILE,
        usecols=["RID", "VISCODE2", "ABETA40"],
        low_memory=False,
    )
    roche = first_baseline(roche, "VISCODE2")
    roche["ABETA40"] = numeric(roche["ABETA40"])
    roche = roche[["RID", "ABETA40"]]

    tables = [demog, mmse, apoe, alz, roche, mri]
    merged = tables[0]
    for table in tables[1:]:
        merged = merged.merge(table, on="RID", how="outer", validate="one_to_one")
    merged["ABETA42_ABETA40_RATIO"] = merged["ABETA42"] / merged["ABETA40"]
    return merged


def compare_extraction(test: pd.DataFrame, extracted: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    joined = test[["RID"] + features].merge(
        extracted[["RID"] + features], on="RID", suffixes=("_stored", "_raw"), how="left"
    )
    rows = []
    for feature in features:
        stored = numeric(joined[f"{feature}_stored"])
        raw = numeric(joined[f"{feature}_raw"])
        both = stored.notna() & raw.notna()
        if both.sum() >= 2:
            corr = float(np.corrcoef(stored[both], raw[both])[0, 1])
            median_abs = float(np.median(np.abs(stored[both] - raw[both])))
            exact = float(np.mean(np.isclose(stored[both], raw[both], rtol=1e-8, atol=1e-8)))
        else:
            corr = median_abs = exact = np.nan
        rows.append(
            {
                "Feature": feature,
                "N_Both": int(both.sum()),
                "Correlation": corr,
                "Median_Absolute_Difference": median_abs,
                "Exact_Match_Fraction": exact,
                "Stored_Missing": int(stored.isna().sum()),
                "Extracted_Missing": int(raw.isna().sum()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    standardized_mri = pd.read_csv(DISCOVERY_STANDARDIZED_DIR / "RNA_plasma.csv", nrows=2)
    mri_features = [col for col in standardized_mri.columns if col.startswith("ST")]
    raw_features = load_raw_feature_table(mri_features)

    discovery_endpoints = pd.read_csv(OUT / "adni_discovery_endpoints.csv")
    holdout_endpoints = pd.read_csv(OUT / "adni_holdout_endpoints.csv")
    test = pd.read_csv(TEST_FILE, low_memory=False)

    clinical_features = [
        "MMSE_Baseline",
        "Education",
        "Gender",
        "APOE4_Copies",
        "ABETA42",
        "ABETA40",
        "TAU_TOTAL",
        "PTAU181",
        "ABETA42_ABETA40_RATIO",
    ]
    all_features = clinical_features + mri_features

    validation = compare_extraction(test, raw_features, all_features)
    validation.to_csv(OUT / "raw_extraction_validation_against_test_file.csv", index=False)

    discovery = discovery_endpoints.merge(raw_features, on="RID", how="left", validate="one_to_one")
    discovery.to_csv(OUT / "adni_discovery_raw_aligned_features.csv", index=False)

    holdout = holdout_endpoints.merge(raw_features, on="RID", how="left", validate="one_to_one")
    overlap_flag = pd.to_numeric(holdout["RID"], errors="coerce").astype("Int64").isin(
        set(pd.to_numeric(discovery_endpoints["RID"], errors="coerce").dropna().astype(int))
    )
    holdout["Overlaps_Discovery"] = overlap_flag
    holdout.to_csv(OUT / "adni_holdout_raw_aligned_features.csv", index=False)

    core_validation = validation.loc[validation["Feature"].isin(["MMSE_Baseline", "Education", "APOE4_Copies"] + mri_features)]
    summary = {
        "mri_feature_count": len(mri_features),
        "raw_feature_rows": int(len(raw_features)),
        "features_with_correlation_at_least_0_999": int((validation["Correlation"] >= 0.999).sum()),
        "features_compared": int(len(validation)),
        "median_exact_match_fraction_core_features": float(core_validation["Exact_Match_Fraction"].median()),
        "discovery_rows": int(len(discovery)),
        "holdout_rows": int(len(holdout)),
        "independent_holdout_rows": int((~overlap_flag).sum()),
        "discovery_strict36_eligible": int(discovery["Strict36_Outcome"].notna().sum()),
        "independent_holdout_strict36_eligible": int(
            ((~overlap_flag) & holdout["Strict36_Outcome"].notna()).sum()
        ),
    }
    with (OUT / "raw_feature_extraction_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nLowest extraction correlations:")
    print(validation.sort_values("Correlation").head(12).to_string(index=False))


if __name__ == "__main__":
    main()
