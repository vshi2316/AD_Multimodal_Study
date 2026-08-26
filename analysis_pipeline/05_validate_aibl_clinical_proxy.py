from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("AD_MULTIMODAL_DATA_ROOT", ".")).expanduser().resolve()
OUT = Path(os.environ.get("AD_MULTIMODAL_OUTPUT_DIR", str(HERE / "outputs"))).expanduser()
DEMOG_FILE = ROOT / "ADNI_Raw_Data" / "LINES" / "Subject Demographics.csv"
AIBL_FILE = ROOT / "Analysis_Inputs" / "AIBL_Feasibility_Gate" / "02_aibl_eligible_mci_to_ad_conversion_cohort.csv"
FEATURES = ["Age", "Gender_Female", "APOE4_Positive"]


def load_model_helpers():
    spec = importlib.util.spec_from_file_location("model_helpers", HERE / "04_fit_leakage_controlled_models.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def build_discovery() -> pd.DataFrame:
    d = pd.read_csv(OUT / "adni_discovery_raw_aligned_features.csv", low_memory=False)
    demog = pd.read_csv(DEMOG_FILE, usecols=["RID", "PTDOBYY"], low_memory=False).drop_duplicates("RID")
    d["Baseline_Date"] = pd.to_datetime(d["Baseline_Date"], errors="coerce")
    d = d.merge(demog, on="RID", how="left")
    d["Age"] = d["Baseline_Date"].dt.year - pd.to_numeric(d["PTDOBYY"], errors="coerce")
    d["Gender_Female"] = pd.to_numeric(d["Gender"], errors="coerce")
    d["APOE4_Positive"] = (pd.to_numeric(d["APOE4_Copies"], errors="coerce") > 0).astype(float)
    return d.loc[d["Strict36_Outcome"].notna()].copy()


def build_aibl() -> pd.DataFrame:
    endpoint = pd.read_csv(OUT / "aibl_endpoints.csv", low_memory=False)
    needed = ["RID", "Age", "APOE4_Positive", "AD_Conversion"]
    source_cols = [c for c in needed if c in pd.read_csv(AIBL_FILE, nrows=0).columns]
    if "Gender_Female" in pd.read_csv(AIBL_FILE, nrows=0).columns:
        source_cols.append("Gender_Female")
    elif "Gender" in pd.read_csv(AIBL_FILE, nrows=0).columns:
        source_cols.append("Gender")
    source = pd.read_csv(AIBL_FILE, usecols=sorted(set(source_cols)), low_memory=False)
    source = source.merge(endpoint[["RID", "Strict36_Outcome", "VisitWindow_Outcome"]], on="RID", how="left", validate="one_to_one")
    if "Gender_Female" not in source:
        g = pd.to_numeric(source["Gender"], errors="coerce")
        source["Gender_Female"] = np.where(g.isin([0, 1]), g, np.where(g == 2, 1.0, np.where(g == 1, 0.0, np.nan)))
    source["Age"] = pd.to_numeric(source["Age"], errors="coerce")
    source["APOE4_Positive"] = pd.to_numeric(source["APOE4_Positive"], errors="coerce")
    return source


def main() -> None:
    helpers = load_model_helpers()
    discovery = build_discovery()
    aibl = build_aibl()
    x = discovery[FEATURES]
    y = discovery["Strict36_Outcome"].astype(int).to_numpy()
    oof, tuning = helpers.nested_oof_probabilities(x, y)
    threshold = helpers.choose_youden_threshold(y, oof)
    platt = helpers.fit_platt_from_oof(y, oof)
    final = helpers.fit_final(x, y)
    p_raw = final.predict_proba(aibl[FEATURES])[:, 1]
    p_cal = helpers.apply_platt(platt, p_raw)
    aibl["Reduced_Clinical_Probability"] = p_raw
    aibl["Reduced_Clinical_Calibrated_Probability"] = p_cal
    aibl["Reduced_Clinical_Prediction"] = (p_raw >= threshold).astype(int)
    aibl.to_csv(OUT / "aibl_reduced_clinical_predictions.csv", index=False)
    tuning.to_csv(OUT / "aibl_reduced_clinical_tuning.csv", index=False)
    rows = []
    oof_row = helpers.point_metrics(y, oof, threshold)
    oof_row.update({"Dataset": "ADNI_discovery_nested_oof", "Calibration": "raw"})
    rows.append(oof_row)
    for dataset, endpoint_name in [("AIBL_strict36", "Strict36_Outcome"), ("AIBL_visit_window", "VisitWindow_Outcome"), ("AIBL_original_endpoint", "AD_Conversion")]:
        ok = aibl[endpoint_name].notna().to_numpy()
        yy = aibl.loc[ok, endpoint_name].astype(int).to_numpy()
        for label, pp in [("raw", p_raw[ok]), ("discovery_oof_platt", p_cal[ok])]:
            row = helpers.point_metrics(yy, pp, threshold)
            row.update(helpers.bootstrap_ci(yy, pp, threshold))
            row.update({"Dataset": dataset, "Calibration": label})
            rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "aibl_reduced_clinical_performance.csv", index=False)
    summary = {"discovery_n": int(len(discovery)), "discovery_events": int(y.sum()), "discovery_nested_oof_auc": float(roc_auc_score(y, oof)), "aibl_n": int(len(aibl)), "aibl_events": int(aibl["AD_Conversion"].sum()), "features": FEATURES, "locked_threshold": float(threshold)}
    (OUT / "aibl_reduced_clinical_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
