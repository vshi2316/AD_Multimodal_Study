from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "python_pkgs"))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(os.environ.get("AD_MULTIMODAL_DATA_ROOT", ".")).expanduser().resolve()
OUT = Path(os.environ.get("AD_MULTIMODAL_OUTPUT_DIR", str(HERE / "outputs"))).expanduser()
DEMOG_FILE = ROOT / "ADNI_Raw_Data" / "LINES" / "Subject Demographics.csv"
AIBL_FILE = (
    ROOT
    / "Analysis_Inputs"
    / "AIBL_Feasibility_Gate"
    / "02_aibl_eligible_mci_to_ad_conversion_cohort.csv"
)
FEATURES = ["Age", "Gender_Female", "MMSE_Baseline", "APOE4_Positive"]


def load_model_helpers():
    path = HERE / "04_fit_leakage_controlled_models.py"
    spec = importlib.util.spec_from_file_location("model_helpers", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def build_discovery() -> pd.DataFrame:
    discovery = pd.read_csv(OUT / "adni_discovery_raw_aligned_features.csv", low_memory=False)
    demog = pd.read_csv(DEMOG_FILE, usecols=["RID", "PTDOBYY"], low_memory=False)
    demog = demog.drop_duplicates("RID", keep="first")
    demog["Birth_Year"] = pd.to_numeric(demog["PTDOBYY"], errors="coerce")
    discovery["Baseline_Date"] = pd.to_datetime(discovery["Baseline_Date"], errors="coerce")
    discovery = discovery.merge(demog[["RID", "Birth_Year"]], on="RID", how="left")
    discovery["Age"] = discovery["Baseline_Date"].dt.year - discovery["Birth_Year"]
    discovery["Gender_Female"] = pd.to_numeric(discovery["Gender"], errors="coerce")
    discovery["APOE4_Positive"] = (
        pd.to_numeric(discovery["APOE4_Copies"], errors="coerce") > 0
    ).astype(float)
    return discovery.loc[discovery["Strict36_Outcome"].notna()].copy()


def build_aibl() -> pd.DataFrame:
    source = pd.read_csv(AIBL_FILE, low_memory=False)
    endpoint_table = pd.read_csv(OUT / "aibl_endpoints.csv", low_memory=False)
    endpoints = endpoint_table[["RID", "Strict36_Outcome", "VisitWindow_Outcome"]].copy()
    source = source.merge(endpoints, on="RID", how="left", validate="one_to_one")
    gender = pd.to_numeric(source["Gender"], errors="coerce")
    source["Gender_Female"] = np.where(gender == 2, 1.0, np.where(gender == 1, 0.0, np.nan))
    source["APOE4_Positive"] = pd.to_numeric(source["APOE4_Positive"], errors="coerce")
    source["MMSE_Baseline"] = pd.to_numeric(source["MMSE_Baseline"], errors="coerce")
    source["Age"] = pd.to_numeric(source["Age"], errors="coerce")
    return source


def main() -> None:
    helpers = load_model_helpers()
    discovery = build_discovery()
    aibl = build_aibl()
    x_discovery = discovery[FEATURES]
    y_discovery = discovery["Strict36_Outcome"].astype(int).to_numpy()

    oof, tuning = helpers.nested_oof_probabilities(x_discovery, y_discovery)
    threshold = helpers.choose_youden_threshold(y_discovery, oof)
    platt = helpers.fit_platt_from_oof(y_discovery, oof)
    final = helpers.fit_final(x_discovery, y_discovery)
    all_probability = final.predict_proba(aibl[FEATURES])[:, 1]
    all_platt = helpers.apply_platt(platt, all_probability)
    aibl["Harmonized_Reduced_Model_Probability"] = all_probability
    aibl["Harmonized_Reduced_Model_Platt_Probability"] = all_platt
    aibl["Harmonized_Reduced_Model_Prediction"] = (all_probability >= threshold).astype(int)
    aibl.to_csv(OUT / "aibl_harmonized_predictions.csv", index=False)
    tuning.to_csv(OUT / "aibl_harmonized_nested_tuning.csv", index=False)

    rows = []
    oof_point = helpers.point_metrics(y_discovery, oof, threshold)
    oof_point.update({"Dataset": "ADNI_discovery_nested_oof", "Calibration": "raw"})
    rows.append(oof_point)

    for dataset, endpoint in [
        ("AIBL_strict36", "Strict36_Outcome"),
        ("AIBL_visit_window", "VisitWindow_Outcome"),
        ("AIBL_original_endpoint", "AD_Conversion"),
    ]:
        eligible = aibl[endpoint].notna().to_numpy()
        y = aibl.loc[eligible, endpoint].astype(int).to_numpy()
        raw = all_probability[eligible]
        calibrated = all_platt[eligible]
        raw_point = helpers.point_metrics(y, raw, threshold)
        raw_point.update(helpers.bootstrap_ci(y, raw, threshold))
        raw_point.update({"Dataset": dataset, "Calibration": "raw"})
        rows.append(raw_point)
        calibrated_threshold = helpers.choose_youden_threshold(
            y_discovery, helpers.apply_platt(platt, oof)
        )
        calibrated_point = helpers.point_metrics(y, calibrated, calibrated_threshold)
        calibrated_point.update(helpers.bootstrap_ci(y, calibrated, calibrated_threshold))
        calibrated_point.update({"Dataset": dataset, "Calibration": "discovery_oof_platt"})
        rows.append(calibrated_point)

    performance = pd.DataFrame(rows)
    performance.to_csv(OUT / "aibl_harmonized_performance.csv", index=False)
    summary = {
        "discovery_n": int(len(discovery)),
        "discovery_events": int(y_discovery.sum()),
        "discovery_nested_oof_auc": float(roc_auc_score(y_discovery, oof)),
        "locked_threshold": threshold,
        "aibl_original_n": int(len(aibl)),
        "aibl_strict36_n": int(aibl["Strict36_Outcome"].notna().sum()),
        "aibl_strict36_events": int(aibl["Strict36_Outcome"].sum(skipna=True)),
        "final_best_C": final.best_params_["model__C"],
        "final_best_l1_ratio": final.best_params_["model__l1_ratio"],
    }
    with (OUT / "aibl_harmonized_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nAIBL performance:")
    print(
        performance.loc[performance["Dataset"].str.startswith("AIBL")][
            [
                "Dataset",
                "Calibration",
                "N",
                "Events",
                "AUC",
                "AUC_CI_Lower",
                "AUC_CI_Upper",
                "Brier",
                "Calibration_Intercept",
                "Calibration_Slope",
                "Sensitivity",
                "Specificity",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
