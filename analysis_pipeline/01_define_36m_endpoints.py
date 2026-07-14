from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("AD_MULTIMODAL_DATA_ROOT", ".")).expanduser().resolve()
OUT = Path(os.environ.get("AD_MULTIMODAL_OUTPUT_DIR", str(HERE / "outputs"))).expanduser()

DX_FILE = ROOT / "ADNI_Raw_Data" / "LINES" / "Diagnostic Summary.csv"
DISCOVERY_FILE = (
    ROOT
    / "Derived_Inputs"
    / "Discovery_CSF_Cohort"
    / "Womac_score_pain_function.csv"
)
HOLDOUT_FILE = ROOT / "Analysis_Inputs" / "AI_vs_Clinician_Test" / "independent_test_set.csv"
AIBL_FILE = (
    ROOT
    / "Analysis_Inputs"
    / "AIBL_Feasibility_Gate"
    / "02_aibl_eligible_mci_to_ad_conversion_cohort.csv"
)

DAYS_PER_MONTH = 365.2425 / 12.0


def num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def prepare_diagnoses() -> pd.DataFrame:
    dx = pd.read_csv(DX_FILE, low_memory=False)
    dx["RID"] = num(dx["RID"]).astype("Int64")
    dx["EXAMDATE_PARSED"] = pd.to_datetime(dx["EXAMDATE"], errors="coerce")
    dx["DIAGNOSIS_NUM"] = num(dx["DIAGNOSIS"])
    dx["DXMCI_NUM"] = num(dx["DXMCI"])
    dx["DXAD_NUM"] = num(dx["DXAD"])
    dx["IS_MCI"] = (dx["DIAGNOSIS_NUM"] == 2) | (dx["DXMCI_NUM"] == 1)
    dx["IS_AD"] = (dx["DIAGNOSIS_NUM"] == 3) | (dx["DXAD_NUM"] == 1)
    dx["VISCODE_NORM"] = dx["VISCODE"].astype(str).str.lower().str.strip()
    return dx


def select_discovery_baselines(dx: pd.DataFrame, ptids: pd.Series) -> pd.DataFrame:
    candidates = dx.loc[
        dx["PTID"].isin(ptids)
        & dx["VISCODE_NORM"].isin(["bl", "sc"])
        & dx["IS_MCI"]
        & dx["EXAMDATE_PARSED"].notna(),
        ["PTID", "RID", "EXAMDATE_PARSED", "VISCODE_NORM", "PHASE"],
    ].copy()
    candidates["visit_order"] = candidates["VISCODE_NORM"].map({"bl": 0, "sc": 1})
    candidates = candidates.sort_values(["PTID", "EXAMDATE_PARSED", "visit_order"])
    return candidates.drop_duplicates("PTID", keep="first").rename(
        columns={"EXAMDATE_PARSED": "Baseline_Date_Rebuilt"}
    )


def classify_adni_participant(
    records: pd.DataFrame,
    baseline_date: pd.Timestamp,
    original_outcome: float,
    primary_event_limit: float = 36.0,
    primary_control_minimum: float = 36.0,
    sensitivity_event_limit: float = 39.0,
    sensitivity_control_minimum: float = 33.0,
) -> dict:
    rows = records.loc[records["EXAMDATE_PARSED"].notna()].copy()
    rows["Months_From_Baseline"] = (
        (rows["EXAMDATE_PARSED"] - baseline_date).dt.days / DAYS_PER_MONTH
    )
    rows = rows.loc[rows["Months_From_Baseline"] > 0].sort_values("Months_From_Baseline")

    ad_rows = rows.loc[rows["IS_AD"]]
    first_ad_month = float(ad_rows["Months_From_Baseline"].min()) if len(ad_rows) else np.nan
    last_followup_month = float(rows["Months_From_Baseline"].max()) if len(rows) else 0.0

    def endpoint(event_limit: float, control_minimum: float) -> tuple[float, str]:
        if np.isfinite(first_ad_month) and first_ad_month <= event_limit:
            return 1.0, "converted_within_window"
        if last_followup_month >= control_minimum:
            return 0.0, "event_free_with_adequate_followup"
        return np.nan, "insufficient_followup"

    strict_outcome, strict_reason = endpoint(primary_event_limit, primary_control_minimum)
    visit_window_outcome, visit_window_reason = endpoint(
        sensitivity_event_limit, sensitivity_control_minimum
    )

    late_conversion = bool(np.isfinite(first_ad_month) and first_ad_month > primary_event_limit)
    strict_match = (
        bool(strict_outcome == original_outcome) if np.isfinite(strict_outcome) else False
    )
    return {
        "Original_Outcome": original_outcome,
        "First_AD_Month": first_ad_month,
        "Last_Diagnostic_Followup_Month": last_followup_month,
        "Strict36_Outcome": strict_outcome,
        "Strict36_Reason": strict_reason,
        "VisitWindow_Outcome": visit_window_outcome,
        "VisitWindow_Reason": visit_window_reason,
        "Late_Conversion_After_36m": late_conversion,
        "Strict36_Matches_Original": strict_match,
    }


def define_discovery_endpoints(dx: pd.DataFrame) -> pd.DataFrame:
    original = pd.read_csv(DISCOVERY_FILE)
    original = original[["ID", "AD_Conversion"]].drop_duplicates("ID")
    baseline = select_discovery_baselines(dx, original["ID"])
    cohort = original.merge(
        baseline[["PTID", "RID", "Baseline_Date_Rebuilt", "VISCODE_NORM", "PHASE"]],
        left_on="ID",
        right_on="PTID",
        how="left",
    )

    output = []
    for row in cohort.itertuples(index=False):
        base = row.Baseline_Date_Rebuilt
        if pd.isna(base):
            result = {
                "Original_Outcome": row.AD_Conversion,
                "First_AD_Month": np.nan,
                "Last_Diagnostic_Followup_Month": np.nan,
                "Strict36_Outcome": np.nan,
                "Strict36_Reason": "baseline_not_found",
                "VisitWindow_Outcome": np.nan,
                "VisitWindow_Reason": "baseline_not_found",
                "Late_Conversion_After_36m": False,
                "Strict36_Matches_Original": False,
            }
        else:
            result = classify_adni_participant(
                dx.loc[dx["PTID"] == row.ID], base, row.AD_Conversion
            )
        result.update(
            {
                "Cohort": "ADNI_discovery",
                "ID": row.ID,
                "RID": row.RID,
                "Baseline_Date": base,
                "Baseline_VISCODE": row.VISCODE_NORM,
                "Baseline_PHASE": row.PHASE,
            }
        )
        output.append(result)
    return pd.DataFrame(output)


def define_holdout_endpoints(dx: pd.DataFrame) -> pd.DataFrame:
    original = pd.read_csv(HOLDOUT_FILE, low_memory=False)
    original = original[["ID", "RID", "AD_Conversion", "Baseline_Date"]].copy()
    original["RID"] = num(original["RID"]).astype("Int64")
    original["Baseline_Date"] = pd.to_datetime(original["Baseline_Date"], errors="coerce")

    output = []
    for row in original.itertuples(index=False):
        records = dx.loc[dx["RID"] == row.RID]
        result = classify_adni_participant(
            records, row.Baseline_Date, row.AD_Conversion
        )
        baseline_record = records.loc[
            records["EXAMDATE_PARSED"] == row.Baseline_Date,
            ["VISCODE_NORM", "PHASE", "IS_MCI"],
        ]
        if len(baseline_record):
            viscode = baseline_record.iloc[0]["VISCODE_NORM"]
            phase = baseline_record.iloc[0]["PHASE"]
            baseline_is_mci = bool(baseline_record["IS_MCI"].any())
        else:
            viscode = None
            phase = None
            baseline_is_mci = False
        result.update(
            {
                "Cohort": "ADNI_holdout",
                "ID": row.ID,
                "RID": row.RID,
                "Baseline_Date": row.Baseline_Date,
                "Baseline_VISCODE": viscode,
                "Baseline_PHASE": phase,
                "Baseline_Is_MCI": baseline_is_mci,
            }
        )
        output.append(result)
    return pd.DataFrame(output)


def define_aibl_endpoints() -> pd.DataFrame:
    original = pd.read_csv(AIBL_FILE, low_memory=False)
    event_month = num(original["event_month"])
    last_month = num(original["last_followup_month"])
    old_outcome = num(original["AD_Conversion"])

    strict = np.where(
        event_month.notna() & (event_month <= 36.0),
        1.0,
        np.where(last_month >= 36.0, 0.0, np.nan),
    )
    visit_window = np.where(
        event_month.notna() & (event_month <= 39.0),
        1.0,
        np.where(last_month >= 33.0, 0.0, np.nan),
    )
    reason = np.where(
        event_month.notna() & (event_month <= 36.0),
        "converted_within_window",
        np.where(last_month >= 36.0, "event_free_with_adequate_followup", "insufficient_followup"),
    )
    visit_reason = np.where(
        event_month.notna() & (event_month <= 39.0),
        "converted_within_window",
        np.where(last_month >= 33.0, "event_free_with_adequate_followup", "insufficient_followup"),
    )

    keep_cols = [
        col
        for col in [
            "RID",
            "Baseline_Date",
            "baseline_date",
            "Followup_Years",
            "Time_to_Event_Months",
            "n_followup_visits",
            "event_month",
            "last_followup_month",
        ]
        if col in original.columns
    ]
    result = original[keep_cols].copy()
    result.insert(0, "ID", original["RID"].astype(str))
    result.insert(0, "Cohort", "AIBL")
    result["Original_Outcome"] = old_outcome
    result["First_AD_Month"] = event_month
    result["Last_Diagnostic_Followup_Month"] = last_month
    result["Strict36_Outcome"] = strict
    result["Strict36_Reason"] = reason
    result["VisitWindow_Outcome"] = visit_window
    result["VisitWindow_Reason"] = visit_reason
    result["Late_Conversion_After_36m"] = event_month.notna() & (event_month > 36.0)
    result["Strict36_Matches_Original"] = result["Strict36_Outcome"].eq(old_outcome)
    return result


def summarize(df: pd.DataFrame) -> dict:
    strict_eligible = df["Strict36_Outcome"].notna()
    visit_eligible = df["VisitWindow_Outcome"].notna()
    strict_changed = strict_eligible & ~df["Strict36_Outcome"].eq(df["Original_Outcome"])
    visit_changed = visit_eligible & ~df["VisitWindow_Outcome"].eq(df["Original_Outcome"])
    return {
        "n_original": int(len(df)),
        "original_events": int((df["Original_Outcome"] == 1).sum()),
        "strict36_eligible": int(strict_eligible.sum()),
        "strict36_excluded_insufficient": int((~strict_eligible).sum()),
        "strict36_events": int((df.loc[strict_eligible, "Strict36_Outcome"] == 1).sum()),
        "strict36_nonevents": int((df.loc[strict_eligible, "Strict36_Outcome"] == 0).sum()),
        "strict36_labels_changed_among_eligible": int(strict_changed.sum()),
        "visit_window_eligible": int(visit_eligible.sum()),
        "visit_window_excluded_insufficient": int((~visit_eligible).sum()),
        "visit_window_events": int((df.loc[visit_eligible, "VisitWindow_Outcome"] == 1).sum()),
        "visit_window_nonevents": int((df.loc[visit_eligible, "VisitWindow_Outcome"] == 0).sum()),
        "visit_window_labels_changed_among_eligible": int(visit_changed.sum()),
        "late_conversions_after_36m": int(df["Late_Conversion_After_36m"].sum()),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dx = prepare_diagnoses()
    discovery = define_discovery_endpoints(dx)
    holdout = define_holdout_endpoints(dx)
    aibl = define_aibl_endpoints()

    for name, frame in [
        ("adni_discovery_endpoints.csv", discovery),
        ("adni_holdout_endpoints.csv", holdout),
        ("aibl_endpoints.csv", aibl),
    ]:
        frame.to_csv(OUT / name, index=False)

    summary = {
        "definitions": {
            "strict36": "event by 36.0 calendar months; controls require at least 36.0 months of diagnostic follow-up",
            "visit_window_sensitivity": "event by 39.0 months; controls require at least 33.0 months of diagnostic follow-up",
        },
        "ADNI_discovery": summarize(discovery),
        "ADNI_holdout": summarize(holdout),
        "AIBL": summarize(aibl),
    }
    with (OUT / "endpoint_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
