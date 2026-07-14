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
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.inter_rater import fleiss_kappa


ROOT = Path(os.environ.get("AD_MULTIMODAL_DATA_ROOT", ".")).expanduser().resolve()
OUT = Path(os.environ.get("AD_MULTIMODAL_OUTPUT_DIR", str(HERE / "outputs"))).expanduser()
EXPERT_FILE = ROOT / "Analysis_Inputs" / "AI_vs_Clinician_Test" / "Expert_Predictions_Long.csv"
BENCHMARK_FILE = OUT / "crossfitted_expert_benchmark.csv"
REPS = 5000
SEED = 20260710


def load_helpers():
    path = HERE / "02_rulec_statistics_core.py"
    spec = importlib.util.spec_from_file_location("rule_helpers", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def icc2_absolute(matrix: np.ndarray) -> float:
    matrix = np.asarray(matrix, dtype=float)
    n, k = matrix.shape
    grand = matrix.mean()
    row_mean = matrix.mean(axis=1)
    col_mean = matrix.mean(axis=0)
    ss_rows = k * np.sum((row_mean - grand) ** 2)
    ss_cols = n * np.sum((col_mean - grand) ** 2)
    residual = matrix - row_mean[:, None] - col_mean[None, :] + grand
    ss_error = np.sum(residual**2)
    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))
    denominator = ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n
    return float((ms_rows - ms_error) / denominator)


def bootstrap_icc(matrix: np.ndarray) -> tuple[float, float, float]:
    rng = np.random.default_rng(SEED)
    point = icc2_absolute(matrix)
    samples = []
    for _ in range(REPS):
        idx = rng.choice(matrix.shape[0], matrix.shape[0], replace=True)
        samples.append(icc2_absolute(matrix[idx]))
    return point, float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def reliability(experts: pd.DataFrame, eligible_ids: list[str]) -> pd.DataFrame:
    data = experts.loc[experts["CaseID"].isin(eligible_ids)].copy()
    rows = []
    for stage in ["Stage1_Prob", "Stage2_Prob"]:
        matrix = data.pivot(index="CaseID", columns="Expert", values=stage).loc[eligible_ids].to_numpy()
        icc, lower, upper = bootstrap_icc(matrix)
        binary = (matrix >= 0.5).astype(int)
        counts = np.column_stack([(binary == 0).sum(axis=1), (binary == 1).sum(axis=1)])
        rows.append(
            {
                "Stage": stage,
                "N_Cases": matrix.shape[0],
                "N_Readers": matrix.shape[1],
                "ICC_2_1_Absolute_Agreement": icc,
                "ICC_95CI_Lower": lower,
                "ICC_95CI_Upper": upper,
                "Fleiss_Kappa_50pct_Boundary": float(fleiss_kappa(counts, method="fleiss")),
            }
        )
    return pd.DataFrame(rows)


def macro_reader_metrics(y: np.ndarray, probability: np.ndarray, prediction: np.ndarray) -> dict:
    aucs = []
    briers = []
    sensitivities = []
    specificities = []
    for reader_index in range(probability.shape[1]):
        p = probability[:, reader_index]
        pred = prediction[:, reader_index]
        aucs.append(roc_auc_score(y, p))
        briers.append(np.mean((p - y) ** 2))
        sensitivities.append(np.mean(pred[y == 1] == 1))
        specificities.append(np.mean(pred[y == 0] == 0))
    return {
        "Macro_AUC": float(np.mean(aucs)),
        "Macro_Brier": float(np.mean(briers)),
        "Macro_Sensitivity": float(np.mean(sensitivities)),
        "Macro_Specificity": float(np.mean(specificities)),
    }


def macro_bootstrap(
    y: np.ndarray,
    conditions: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED + 1)
    point_rows = []
    samples = {
        condition: {metric: [] for metric in ["Macro_AUC", "Macro_Brier", "Macro_Sensitivity", "Macro_Specificity"]}
        for condition in conditions
    }
    difference_samples = {
        metric: [] for metric in ["Macro_AUC", "Macro_Brier", "Macro_Sensitivity", "Macro_Specificity"]
    }
    event = np.flatnonzero(y == 1)
    nonevent = np.flatnonzero(y == 0)
    for condition, (probability, prediction) in conditions.items():
        row = macro_reader_metrics(y, probability, prediction)
        row["Condition"] = condition
        point_rows.append(row)
    for _ in range(REPS):
        idx = np.concatenate(
            [rng.choice(event, len(event), replace=True), rng.choice(nonevent, len(nonevent), replace=True)]
        )
        current = {}
        for condition, (probability, prediction) in conditions.items():
            result = macro_reader_metrics(y[idx], probability[idx], prediction[idx])
            current[condition] = result
            for metric, value in result.items():
                samples[condition][metric].append(value)
        for metric in difference_samples:
            difference_samples[metric].append(current["RuleC"][metric] - current["Stage2"][metric])
    points = pd.DataFrame(point_rows)
    for index, row in points.iterrows():
        condition = row["Condition"]
        for metric in samples[condition]:
            values = samples[condition][metric]
            points.loc[index, f"{metric}_CI_Lower"] = np.percentile(values, 2.5)
            points.loc[index, f"{metric}_CI_Upper"] = np.percentile(values, 97.5)
    differences = []
    for metric, values in difference_samples.items():
        point = (
            points.loc[points["Condition"] == "RuleC", metric].iloc[0]
            - points.loc[points["Condition"] == "Stage2", metric].iloc[0]
        )
        differences.append(
            {
                "Contrast": "RuleC_minus_Stage2",
                "Metric": metric,
                "Estimate": point,
                "CI_Lower": np.percentile(values, 2.5),
                "CI_Upper": np.percentile(values, 97.5),
            }
        )
    return points, pd.DataFrame(differences)


def categorical_nri(y: np.ndarray, old: np.ndarray, new: np.ndarray) -> dict:
    event = y == 1
    nonevent = y == 0
    event_up = np.mean((old[event] == 0) & (new[event] == 1))
    event_down = np.mean((old[event] == 1) & (new[event] == 0))
    nonevent_down = np.mean((old[nonevent] == 1) & (new[nonevent] == 0))
    nonevent_up = np.mean((old[nonevent] == 0) & (new[nonevent] == 1))
    return {
        "Event_NRI": event_up - event_down,
        "Nonevent_NRI": nonevent_down - nonevent_up,
        "Categorical_NRI": event_up - event_down + nonevent_down - nonevent_up,
    }


def bootstrap_nri(y: np.ndarray, old: np.ndarray, new: np.ndarray) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 2)
    point = categorical_nri(y, old, new)
    samples = {key: [] for key in point}
    event = np.flatnonzero(y == 1)
    nonevent = np.flatnonzero(y == 0)
    for _ in range(REPS):
        idx = np.concatenate(
            [rng.choice(event, len(event), replace=True), rng.choice(nonevent, len(nonevent), replace=True)]
        )
        result = categorical_nri(y[idx], old[idx], new[idx])
        for key, value in result.items():
            samples[key].append(value)
    return pd.DataFrame(
        [
            {
                "Metric": key,
                "Estimate": point[key],
                "CI_Lower": np.percentile(samples[key], 2.5),
                "CI_Upper": np.percentile(samples[key], 97.5),
            }
            for key in point
        ]
    )


def dca_curve(y: np.ndarray, models: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    prevalence = y.mean()
    for threshold in np.arange(0.05, 0.81, 0.01):
        weight = threshold / (1 - threshold)
        rows.append(
            {
                "Model": "Treat_all",
                "Threshold": threshold,
                "Net_Benefit": prevalence - (1 - prevalence) * weight,
            }
        )
        rows.append({"Model": "Treat_none", "Threshold": threshold, "Net_Benefit": 0.0})
        for name, probability in models.items():
            pred = probability >= threshold
            tp = np.sum(pred & (y == 1))
            fp = np.sum(pred & (y == 0))
            rows.append(
                {
                    "Model": name,
                    "Threshold": threshold,
                    "Net_Benefit": tp / len(y) - fp / len(y) * weight,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    helpers = load_helpers()
    benchmark_all = pd.read_csv(BENCHMARK_FILE, low_memory=False)
    benchmark = benchmark_all.loc[
        benchmark_all["Analysis_Model"] == "primary_clinical_csf_mri"
    ].copy()
    benchmark = benchmark.sort_values("ID").reset_index(drop=True)
    experts = pd.read_csv(EXPERT_FILE)
    eligible_ids = benchmark["ID"].tolist()
    reliability_table = reliability(experts, eligible_ids)
    reliability_table.to_csv(OUT / "final_expert_reliability_strict153.csv", index=False)

    long = experts.loc[experts["CaseID"].isin(eligible_ids)].merge(
        benchmark[["ID", "Strict36_Outcome", "AI_Probability", "AI_Prediction"]],
        left_on="CaseID",
        right_on="ID",
        how="inner",
    )
    reader_order = sorted(long["Expert"].unique())
    stage1 = long.pivot(index="CaseID", columns="Expert", values="Stage1_Prob").loc[eligible_ids, reader_order].to_numpy()
    stage2 = long.pivot(index="CaseID", columns="Expert", values="Stage2_Prob").loc[eligible_ids, reader_order].to_numpy()
    ai_probability = benchmark.set_index("ID").loc[eligible_ids, "AI_Probability"].to_numpy()
    ai_prediction = benchmark.set_index("ID").loc[eligible_ids, "AI_Prediction"].astype(int).to_numpy()
    y = benchmark.set_index("ID").loc[eligible_ids, "Strict36_Outcome"].astype(int).to_numpy()
    stage1_prediction = (stage1 >= 0.5).astype(int)
    stage2_prediction = (stage2 >= 0.5).astype(int)
    gray = (stage2 >= 0.4) & (stage2 <= 0.6)
    rule_probability = np.where(gray, ai_probability[:, None], stage2)
    rule_prediction = np.where(gray, ai_prediction[:, None], stage2_prediction).astype(int)

    macro_points, macro_differences = macro_bootstrap(
        y,
        {
            "Stage1": (stage1, stage1_prediction),
            "Stage2": (stage2, stage2_prediction),
            "RuleC": (rule_probability, rule_prediction),
        },
    )
    macro_points.to_csv(OUT / "final_multireader_macro_performance.csv", index=False)
    macro_differences.to_csv(OUT / "final_multireader_macro_differences.csv", index=False)

    pooled_stage1 = stage1.mean(axis=1)
    pooled_stage2 = stage2.mean(axis=1)
    pooled_stage1_pred = (pooled_stage1 >= 0.5).astype(int)
    pooled_stage2_pred = (pooled_stage2 >= 0.5).astype(int)
    pooled_gray = (pooled_stage2 >= 0.4) & (pooled_stage2 <= 0.6)
    pooled_rule_probability = np.where(pooled_gray, ai_probability, pooled_stage2)
    pooled_rule_prediction = np.where(pooled_gray, ai_prediction, pooled_stage2_pred).astype(int)
    pooled_models = {
        "Crossfitted_AI": (ai_probability, ai_prediction),
        "Pooled_Expert_Stage1": (pooled_stage1, pooled_stage1_pred),
        "Pooled_Expert_Stage2": (pooled_stage2, pooled_stage2_pred),
        "Pooled_RuleC": (pooled_rule_probability, pooled_rule_prediction),
    }
    pooled_performance = pd.DataFrame(
        [helpers.metrics(y, probability, prediction, name) for name, (probability, prediction) in pooled_models.items()]
    )
    pooled_cis = helpers.bootstrap_metric_cis(y, pooled_models)
    pooled_performance.to_csv(OUT / "final_pooled_reader_performance.csv", index=False)
    pooled_cis.to_csv(OUT / "final_pooled_reader_bootstrap_cis.csv", index=False)

    paired, reclassification = helpers.paired_rulec_statistics(
        y,
        pooled_stage2,
        pooled_stage2_pred,
        pooled_rule_probability,
        pooled_rule_prediction,
    )
    paired.to_csv(OUT / "final_pooled_rulec_paired_changes.csv", index=False)
    reclassification.to_csv(OUT / "final_pooled_rulec_reclassification.csv", index=False)
    nri = bootstrap_nri(y, pooled_stage2_pred, pooled_rule_prediction)
    nri.to_csv(OUT / "final_pooled_rulec_categorical_nri.csv", index=False)

    fp_table = pd.crosstab(pooled_stage2_pred[y == 0], pooled_rule_prediction[y == 0]).reindex(
        index=[0, 1], columns=[0, 1], fill_value=0
    )
    fn_table = pd.crosstab(pooled_stage2_pred[y == 1], pooled_rule_prediction[y == 1]).reindex(
        index=[0, 1], columns=[0, 1], fill_value=0
    )
    tests = pd.DataFrame(
        [
            {
                "Comparison": "False_positive_classification_among_nonevents",
                "Discordant_Expert0_Rule1": int(fp_table.loc[0, 1]),
                "Discordant_Expert1_Rule0": int(fp_table.loc[1, 0]),
                "Exact_McNemar_P": float(mcnemar(fp_table.to_numpy(), exact=True).pvalue),
            },
            {
                "Comparison": "Positive_classification_among_events",
                "Discordant_Expert0_Rule1": int(fn_table.loc[0, 1]),
                "Discordant_Expert1_Rule0": int(fn_table.loc[1, 0]),
                "Exact_McNemar_P": float(mcnemar(fn_table.to_numpy(), exact=True).pvalue),
            },
        ]
    )
    tests.to_csv(OUT / "final_pooled_rulec_exact_mcnemar.csv", index=False)

    dca = dca_curve(
        y,
        {
            "Crossfitted_AI": ai_probability,
            "Pooled_Expert_Stage2": pooled_stage2,
            "Pooled_RuleC": pooled_rule_probability,
        },
    )
    dca.to_csv(OUT / "final_pooled_rulec_dca.csv", index=False)

    summary = {
        "n_cases": len(y),
        "events": int(y.sum()),
        "nonevents": int((y == 0).sum()),
        "readers": len(reader_order),
        "case_reader_ratings": int(len(y) * len(reader_order)),
        "pooled_gray_zone_cases": int(pooled_gray.sum()),
        "reader_specific_gray_zone_ratings": int(gray.sum()),
        "bootstrap_repetitions": REPS,
    }
    with (OUT / "final_multireader_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nReliability:")
    print(reliability_table.to_string(index=False))
    print("\nPooled performance:")
    print(pooled_performance.to_string(index=False))
    print("\nMacro-reader contrasts:")
    print(macro_differences.to_string(index=False))
    print("\nCategorical NRI:")
    print(nri.to_string(index=False))
    print("\nExact McNemar tests:")
    print(tests.to_string(index=False))


if __name__ == "__main__":
    main()
