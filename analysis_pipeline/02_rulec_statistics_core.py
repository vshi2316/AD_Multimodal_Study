from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "python_pkgs"))

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score


ROOT = Path(os.environ.get("AD_MULTIMODAL_DATA_ROOT", ".")).expanduser().resolve()
OUT = Path(os.environ.get("AD_MULTIMODAL_OUTPUT_DIR", str(HERE / "outputs"))).expanduser()
TEST_DIR = ROOT / "Analysis_Inputs" / "AI_vs_Clinician_Test"
TEST_FILE = TEST_DIR / "independent_test_set.csv"
AI_FILE = TEST_DIR / "AI_per_patient_predictions.csv"
EXPERT_FILE = TEST_DIR / "Expert_Predictions_Long.csv"

AI_THRESHOLD = 0.5142
EXPERT_THRESHOLD = 0.50
GRAY_LOWER = 0.40
GRAY_UPPER = 0.60
BOOTSTRAP_REPS = 2000
SEED = 20260710


def clip_probability(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)


def auc_or_nan(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p)) if np.unique(y).size == 2 else np.nan


def calibration(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    logit_p = np.log(clip_probability(p) / (1 - clip_probability(p)))
    design = sm.add_constant(logit_p)
    try:
        fit = sm.GLM(y, design, family=sm.families.Binomial()).fit(disp=0)
        return float(fit.params[0]), float(fit.params[1])
    except Exception:
        return np.nan, np.nan


def metrics(y: np.ndarray, p: np.ndarray, pred: np.ndarray, model: str) -> dict:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pred = np.asarray(pred, dtype=int)
    tp = int(((y == 1) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    intercept, slope = calibration(y, p)
    return {
        "Model": model,
        "N": len(y),
        "Events": int(y.sum()),
        "Nonevents": int((y == 0).sum()),
        "AUC": auc_or_nan(y, p),
        "Brier": float(np.mean((p - y) ** 2)),
        "Calibration_Intercept": intercept,
        "Calibration_Slope": slope,
        "Sensitivity": tp / (tp + fn) if tp + fn else np.nan,
        "Specificity": tn / (tn + fp) if tn + fp else np.nan,
        "PPV": tp / (tp + fp) if tp + fp else np.nan,
        "NPV": tn / (tn + fn) if tn + fn else np.nan,
        "Accuracy": (tp + tn) / len(y),
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
    }


def stratified_bootstrap_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    event = np.flatnonzero(y == 1)
    nonevent = np.flatnonzero(y == 0)
    return np.concatenate(
        [rng.choice(event, size=len(event), replace=True), rng.choice(nonevent, size=len(nonevent), replace=True)]
    )


def bootstrap_metric_cis(y: np.ndarray, models: dict[str, tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    boot = {name: {key: [] for key in ["AUC", "Brier", "Sensitivity", "Specificity"]} for name in models}
    for _ in range(BOOTSTRAP_REPS):
        idx = stratified_bootstrap_indices(y, rng)
        for name, (p, pred) in models.items():
            yy = y[idx]
            pp = p[idx]
            dd = pred[idx]
            event = yy == 1
            nonevent = yy == 0
            boot[name]["AUC"].append(auc_or_nan(yy, pp))
            boot[name]["Brier"].append(float(np.mean((pp - yy) ** 2)))
            boot[name]["Sensitivity"].append(float(np.mean(dd[event] == 1)))
            boot[name]["Specificity"].append(float(np.mean(dd[nonevent] == 0)))
    for name, values in boot.items():
        for metric_name, samples in values.items():
            arr = np.asarray(samples, dtype=float)
            rows.append(
                {
                    "Model": name,
                    "Metric": metric_name,
                    "CI_Lower": float(np.nanpercentile(arr, 2.5)),
                    "CI_Upper": float(np.nanpercentile(arr, 97.5)),
                    "Bootstrap_Reps": BOOTSTRAP_REPS,
                }
            )
    return pd.DataFrame(rows)


def net_benefit(y: np.ndarray, p: np.ndarray, threshold: float) -> float:
    pred = p >= threshold
    tp = ((y == 1) & pred).sum()
    fp = ((y == 0) & pred).sum()
    return float(tp / len(y) - fp / len(y) * threshold / (1 - threshold))


def paired_rulec_statistics(
    y: np.ndarray,
    expert_p: np.ndarray,
    expert_pred: np.ndarray,
    rule_p: np.ndarray,
    rule_pred: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    event = y == 1
    nonevent = y == 0
    point = {
        "Delta_FP_Count": int(((nonevent) & (rule_pred == 1)).sum() - ((nonevent) & (expert_pred == 1)).sum()),
        "Delta_FN_Count": int(((event) & (rule_pred == 0)).sum() - ((event) & (expert_pred == 0)).sum()),
        "Delta_FPR": float(np.mean(rule_pred[nonevent] == 1) - np.mean(expert_pred[nonevent] == 1)),
        "Delta_FNR": float(np.mean(rule_pred[event] == 0) - np.mean(expert_pred[event] == 0)),
    }
    thresholds = [0.20, 0.30, 0.50]
    for threshold in thresholds:
        point[f"Delta_NB_{threshold:.2f}"] = net_benefit(y, rule_p, threshold) - net_benefit(
            y, expert_p, threshold
        )

    rng = np.random.default_rng(SEED + 1)
    samples = {key: [] for key in point}
    for _ in range(BOOTSTRAP_REPS):
        idx = stratified_bootstrap_indices(y, rng)
        yy = y[idx]
        ep = expert_p[idx]
        ed = expert_pred[idx]
        rp = rule_p[idx]
        rd = rule_pred[idx]
        ev = yy == 1
        non = yy == 0
        samples["Delta_FP_Count"].append(((non) & (rd == 1)).sum() - ((non) & (ed == 1)).sum())
        samples["Delta_FN_Count"].append(((ev) & (rd == 0)).sum() - ((ev) & (ed == 0)).sum())
        samples["Delta_FPR"].append(np.mean(rd[non] == 1) - np.mean(ed[non] == 1))
        samples["Delta_FNR"].append(np.mean(rd[ev] == 0) - np.mean(ed[ev] == 0))
        for threshold in thresholds:
            samples[f"Delta_NB_{threshold:.2f}"].append(
                net_benefit(yy, rp, threshold) - net_benefit(yy, ep, threshold)
            )

    result = []
    for key, value in point.items():
        arr = np.asarray(samples[key], dtype=float)
        result.append(
            {
                "Statistic": key,
                "Estimate": value,
                "CI_Lower": float(np.percentile(arr, 2.5)),
                "CI_Upper": float(np.percentile(arr, 97.5)),
            }
        )

    reclassification = pd.DataFrame(
        {
            "Outcome": y,
            "Expert_Pred": expert_pred,
            "RuleC_Pred": rule_pred,
        }
    )
    table = (
        reclassification.groupby(["Outcome", "Expert_Pred", "RuleC_Pred"], dropna=False)
        .size()
        .reset_index(name="N")
    )
    return pd.DataFrame(result), table


def build_master() -> tuple[pd.DataFrame, pd.DataFrame]:
    test = pd.read_csv(TEST_FILE, low_memory=False)
    ai = pd.read_csv(AI_FILE).rename(
        columns={"Actual": "AI_Actual", "Predicted_Prob": "AI_Prob", "Predicted_Class": "AI_Pred"}
    )
    experts = pd.read_csv(EXPERT_FILE)
    expert_wide = (
        experts.groupby("CaseID")
        .agg(
            Expert_Stage1_Prob=("Stage1_Prob", "mean"),
            Expert_Stage2_Prob=("Stage2_Prob", "mean"),
            Expert_Stage2_SD=("Stage2_Prob", "std"),
            N_Experts=("Expert", "nunique"),
        )
        .reset_index()
    )
    holdout_endpoints = pd.read_csv(OUT / "adni_holdout_endpoints.csv")
    discovery_endpoints = pd.read_csv(OUT / "adni_discovery_endpoints.csv")
    discovery_rids = set(pd.to_numeric(discovery_endpoints["RID"], errors="coerce").dropna().astype(int))

    master = (
        test.merge(ai, on="ID", how="inner")
        .merge(expert_wide, left_on="ID", right_on="CaseID", how="inner")
        .merge(
            holdout_endpoints[
                ["ID", "Strict36_Outcome", "VisitWindow_Outcome", "First_AD_Month", "Last_Diagnostic_Followup_Month"]
            ],
            on="ID",
            how="left",
        )
    )
    master["Overlaps_Discovery"] = pd.to_numeric(master["RID"], errors="coerce").astype(int).isin(discovery_rids)
    master["AI_Pred_Locked"] = pd.to_numeric(master["AI_Pred"], errors="coerce").astype(int)
    master["AI_Pred_From_Declared_Threshold"] = (master["AI_Prob"] >= AI_THRESHOLD).astype(int)
    master["Expert_Stage2_Pred"] = (master["Expert_Stage2_Prob"] >= EXPERT_THRESHOLD).astype(int)
    master["RuleC_Prob"] = np.where(
        master["Expert_Stage2_Prob"].between(GRAY_LOWER, GRAY_UPPER, inclusive="both"),
        master["AI_Prob"],
        master["Expert_Stage2_Prob"],
    )
    master["RuleC_Pred"] = np.where(
        master["Expert_Stage2_Prob"].between(GRAY_LOWER, GRAY_UPPER, inclusive="both"),
        master["AI_Pred_Locked"],
        master["Expert_Stage2_Pred"],
    ).astype(int)
    return master, experts


def analyze_scenario(name: str, data: pd.DataFrame, outcome_col: str) -> dict:
    dat = data.loc[data[outcome_col].notna()].copy()
    y = dat[outcome_col].astype(int).to_numpy()
    model_arrays = {
        "AI_locked": (dat["AI_Prob"].to_numpy(), dat["AI_Pred_Locked"].to_numpy()),
        "Pooled_expert_stage2": (
            dat["Expert_Stage2_Prob"].to_numpy(),
            dat["Expert_Stage2_Pred"].to_numpy(),
        ),
        "Pooled_RuleC_simulation": (dat["RuleC_Prob"].to_numpy(), dat["RuleC_Pred"].to_numpy()),
    }
    point = pd.DataFrame([metrics(y, p, pred, model) for model, (p, pred) in model_arrays.items()])
    point.insert(0, "Scenario", name)
    cis = bootstrap_metric_cis(y, model_arrays)
    cis.insert(0, "Scenario", name)
    paired, reclass = paired_rulec_statistics(
        y,
        model_arrays["Pooled_expert_stage2"][0],
        model_arrays["Pooled_expert_stage2"][1],
        model_arrays["Pooled_RuleC_simulation"][0],
        model_arrays["Pooled_RuleC_simulation"][1],
    )
    paired.insert(0, "Scenario", name)
    reclass.insert(0, "Scenario", name)
    return {"point": point, "ci": cis, "paired": paired, "reclass": reclass, "data": dat}


def per_reader_analysis(dat: pd.DataFrame, experts: pd.DataFrame, outcome_col: str, scenario: str) -> pd.DataFrame:
    long = experts.merge(
        dat[["ID", outcome_col, "AI_Prob", "AI_Pred_Locked"]],
        left_on="CaseID",
        right_on="ID",
        how="inner",
    )
    rows = []
    for expert, part in long.groupby("Expert"):
        y = part[outcome_col].astype(int).to_numpy()
        ep = part["Stage2_Prob"].astype(float).to_numpy()
        ed = (ep >= EXPERT_THRESHOLD).astype(int)
        gray = (ep >= GRAY_LOWER) & (ep <= GRAY_UPPER)
        rp = np.where(gray, part["AI_Prob"].to_numpy(), ep)
        rd = np.where(gray, part["AI_Pred_Locked"].to_numpy(), ed).astype(int)
        for label, p, pred in [("Expert_stage2", ep, ed), ("RuleC_simulation", rp, rd)]:
            result = metrics(y, p, pred, label)
            result.update(
                {
                    "Scenario": scenario,
                    "Expert": expert,
                    "Gray_Zone_Cases": int(gray.sum()),
                }
            )
            rows.append(result)
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    master, experts = build_master()
    master.to_csv(OUT / "holdout_reader_master.csv", index=False)

    scenarios = {
        "full_original_endpoint_contaminated": (master, "AD_Conversion"),
        "full_strict36_endpoint_contaminated": (master, "Strict36_Outcome"),
        "independent41_original_endpoint": (master.loc[~master["Overlaps_Discovery"]], "AD_Conversion"),
        "independent_strict36_endpoint": (master.loc[~master["Overlaps_Discovery"]], "Strict36_Outcome"),
        "independent_visit_window_endpoint": (master.loc[~master["Overlaps_Discovery"]], "VisitWindow_Outcome"),
    }

    all_points = []
    all_cis = []
    all_paired = []
    all_reclass = []
    all_reader = []
    scenario_summary = {}
    for name, (source, outcome) in scenarios.items():
        result = analyze_scenario(name, source, outcome)
        all_points.append(result["point"])
        all_cis.append(result["ci"])
        all_paired.append(result["paired"])
        all_reclass.append(result["reclass"])
        all_reader.append(per_reader_analysis(result["data"], experts, outcome, name))
        scenario_summary[name] = {
            "n": int(len(result["data"])),
            "events": int(result["data"][outcome].sum()),
            "nonevents": int((result["data"][outcome] == 0).sum()),
        }

    pd.concat(all_points, ignore_index=True).to_csv(OUT / "rulec_scenario_performance.csv", index=False)
    pd.concat(all_cis, ignore_index=True).to_csv(OUT / "rulec_scenario_bootstrap_cis.csv", index=False)
    pd.concat(all_paired, ignore_index=True).to_csv(OUT / "rulec_paired_changes.csv", index=False)
    pd.concat(all_reclass, ignore_index=True).to_csv(OUT / "rulec_reclassification_tables.csv", index=False)
    pd.concat(all_reader, ignore_index=True).to_csv(OUT / "rulec_per_reader_performance.csv", index=False)

    expert_correlations = []
    if "AI_Prob_Reference" in experts.columns:
        for expert, part in experts.groupby("Expert"):
            complete = part[["AI_Prob_Reference", "Stage1_Prob", "Stage2_Prob"]].dropna()
            for stage in ["Stage1_Prob", "Stage2_Prob"]:
                r, p = pearsonr(complete["AI_Prob_Reference"], complete[stage])
                expert_correlations.append(
                    {"Expert": expert, "Stage": stage, "N": len(complete), "Pearson_r": r, "P": p}
                )
    pd.DataFrame(expert_correlations).to_csv(OUT / "expert_ai_reference_correlations.csv", index=False)

    summary = {
        "holdout_n": int(len(master)),
        "discovery_overlap_n": int(master["Overlaps_Discovery"].sum()),
        "discovery_overlap_percent_of_holdout": float(100 * master["Overlaps_Discovery"].mean()),
        "genuinely_nonoverlapping_n": int((~master["Overlaps_Discovery"]).sum()),
        "ai_locked_class_disagrees_with_declared_threshold_n": int(
            (master["AI_Pred_Locked"] != master["AI_Pred_From_Declared_Threshold"]).sum()
        ),
        "scenarios": scenario_summary,
        "expert_template_contains_ai_reference_columns": bool("AI_Prob_Reference" in experts.columns),
        "expert_assessment_dates": sorted(experts["Assessment_Date"].dropna().astype(str).unique().tolist()),
    }
    with (OUT / "holdout_reader_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nPrimary salvage scenario performance:")
    points = pd.concat(all_points, ignore_index=True)
    print(points.loc[points["Scenario"] == "independent_strict36_endpoint"].to_string(index=False))
    print("\nPrimary salvage paired Rule C changes:")
    paired = pd.concat(all_paired, ignore_index=True)
    print(paired.loc[paired["Scenario"] == "independent_strict36_endpoint"].to_string(index=False))


if __name__ == "__main__":
    main()
