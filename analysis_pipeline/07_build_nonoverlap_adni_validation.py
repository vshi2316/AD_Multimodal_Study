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


ROOT = Path(os.environ.get("AD_MULTIMODAL_DATA_ROOT", ".")).expanduser().resolve()
OUT = Path(os.environ.get("AD_MULTIMODAL_OUTPUT_DIR", str(HERE / "outputs"))).expanduser()
DEMOG_FILE = ROOT / "ADNI_Raw_Data" / "LINES" / "Subject Demographics.csv"
SEED = 20260710


def load_module(filename: str, name: str):
    path = HERE / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def rebuild_endpoints(endpoint_helpers, discovery_ids: set[str]) -> pd.DataFrame:
    dx = endpoint_helpers.prepare_diagnoses()
    candidates = dx.loc[
        dx["VISCODE_NORM"].isin(["bl", "sc"])
        & dx["IS_MCI"]
        & dx["EXAMDATE_PARSED"].notna()
        & ~dx["PTID"].isin(discovery_ids),
        ["PTID", "RID", "EXAMDATE_PARSED", "VISCODE_NORM", "PHASE"],
    ].copy()
    candidates["Visit_Order"] = candidates["VISCODE_NORM"].map({"bl": 0, "sc": 1})
    candidates = candidates.sort_values(["PTID", "EXAMDATE_PARSED", "Visit_Order"])
    candidates = candidates.drop_duplicates("PTID", keep="first")

    rows = []
    for row in candidates.itertuples(index=False):
        result = endpoint_helpers.classify_adni_participant(
            dx.loc[dx["PTID"] == row.PTID],
            row.EXAMDATE_PARSED,
            np.nan,
        )
        rows.append(
            {
                "ID": row.PTID,
                "RID": row.RID,
                "Baseline_Date": row.EXAMDATE_PARSED,
                "Baseline_PHASE": row.PHASE,
                **result,
            }
        )
    return pd.DataFrame(rows)


def add_age(cohort: pd.DataFrame) -> pd.DataFrame:
    demog = pd.read_csv(DEMOG_FILE, usecols=["RID", "PTDOBYY"], low_memory=False)
    demog = demog.drop_duplicates("RID", keep="first")
    demog["Birth_Year"] = pd.to_numeric(demog["PTDOBYY"], errors="coerce")
    cohort = cohort.merge(demog[["RID", "Birth_Year"]], on="RID", how="left")
    cohort["Baseline_Date"] = pd.to_datetime(cohort["Baseline_Date"], errors="coerce")
    cohort["Age"] = cohort["Baseline_Date"].dt.year - cohort["Birth_Year"]
    return cohort


def main() -> None:
    endpoint_helpers = load_module("01_define_36m_endpoints.py", "endpoint_helpers")
    extraction_helpers = load_module("03_extract_aligned_features.py", "extraction_helpers")
    model_helpers = load_module("04_fit_leakage_controlled_models.py", "model_helpers")

    discovery = pd.read_csv(OUT / "adni_discovery_raw_aligned_features.csv", low_memory=False)
    discovery_ids = set(discovery["ID"].astype(str))
    mri = [col for col in discovery.columns if col.startswith("ST")]
    raw_features = extraction_helpers.load_raw_feature_table(mri)
    endpoints = rebuild_endpoints(endpoint_helpers, discovery_ids)
    cohort = endpoints.merge(raw_features, on="RID", how="left", validate="one_to_one")
    cohort = add_age(cohort)

    core = model_helpers.CLINICAL
    eligible = cohort.loc[
        cohort["Strict36_Outcome"].notna()
        & cohort["Age"].between(50, 95, inclusive="both")
        & cohort[core].notna().all(axis=1)
        & cohort[mri].notna().all(axis=1)
    ].copy()

    feature_sets = {
        "clinical_plus_mri": model_helpers.CLINICAL + mri,
        "primary_transportable_multimodal": model_helpers.CLINICAL + model_helpers.CSF + mri,
    }
    discovery_strict = discovery.loc[discovery["Strict36_Outcome"].notna()].copy()
    y_train = discovery_strict["Strict36_Outcome"].astype(int).to_numpy()
    y_test = eligible["Strict36_Outcome"].astype(int).to_numpy()
    with (OUT / "leakage_free_model_summary.json").open(encoding="utf-8") as handle:
        prior_summary = json.load(handle)

    performance_rows = []
    for name, features in feature_sets.items():
        final = model_helpers.fit_final(discovery_strict[features], y_train)
        probability = final.predict_proba(eligible[features])[:, 1]
        threshold = prior_summary[name]["locked_threshold_from_oof"]
        eligible[f"{name}_probability"] = probability
        eligible[f"{name}_prediction"] = (probability >= threshold).astype(int)
        result = model_helpers.point_metrics(y_test, probability, threshold)
        result.update(model_helpers.bootstrap_ci(y_test, probability, threshold))
        result.update(
            {
                "Model": name,
                "Dataset": "nonoverlapping_ADNI_strict36",
                "Best_C": final.best_params_["model__C"],
                "Best_L1_Ratio": final.best_params_["model__l1_ratio"],
            }
        )
        performance_rows.append(result)

    eligible.to_csv(OUT / "nonoverlapping_adni_validation_predictions.csv", index=False)
    performance = pd.DataFrame(performance_rows)
    performance.to_csv(OUT / "nonoverlapping_adni_validation_performance.csv", index=False)

    phase_summary = (
        eligible.groupby("Baseline_PHASE")["Strict36_Outcome"]
        .agg(N="count", Events="sum", Event_Rate="mean")
        .reset_index()
    )
    phase_summary.to_csv(OUT / "nonoverlapping_adni_validation_subgroups.csv", index=False)
    summary = {
        "baseline_mci_candidates_after_discovery_exclusion": int(len(endpoints)),
        "strict36_endpoint_eligible_before_feature_filter": int(endpoints["Strict36_Outcome"].notna().sum()),
        "final_complete_clinical_mri_benchmark_n": int(len(eligible)),
        "events": int(y_test.sum()),
        "nonevents": int((y_test == 0).sum()),
        "any_csf_marker_available_n": int(eligible[model_helpers.CSF].notna().any(axis=1).sum()),
        "all_three_csf_markers_available_n": int(eligible[model_helpers.CSF].notna().all(axis=1).sum()),
        "phases": phase_summary.to_dict(orient="records"),
    }
    with (OUT / "nonoverlapping_adni_validation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nPerformance:")
    print(
        performance[
            [
                "Model",
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
                "FP",
                "FN",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
