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
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict


ROOT = Path(os.environ.get("AD_MULTIMODAL_DATA_ROOT", ".")).expanduser().resolve()
OUT = Path(os.environ.get("AD_MULTIMODAL_OUTPUT_DIR", str(HERE / "outputs"))).expanduser()
EXPERT_FILE = ROOT / "Analysis_Inputs" / "AI_vs_Clinician_Test" / "Expert_Predictions_Long.csv"
SEED = 20260710


def load_module(filename: str, name: str):
    path = HERE / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def crossfit_discovery(model_helpers, x: pd.DataFrame, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    probabilities = np.full(len(y), np.nan)
    predictions = np.full(len(y), -1, dtype=int)
    rows = []
    for fold, (train_idx, test_idx) in enumerate(outer.split(x, y), start=1):
        inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED + fold)
        search = GridSearchCV(
            model_helpers.pipeline(SEED + fold),
            model_helpers.PARAM_GRID,
            scoring="roc_auc",
            cv=inner,
            n_jobs=-1,
            refit=True,
        )
        search.fit(x.iloc[train_idx], y[train_idx])

        threshold_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + 100 + fold)
        train_oof = cross_val_predict(
            search.best_estimator_,
            x.iloc[train_idx],
            y[train_idx],
            cv=threshold_cv,
            method="predict_proba",
            n_jobs=-1,
        )[:, 1]
        threshold = model_helpers.choose_youden_threshold(y[train_idx], train_oof)
        fold_probability = search.predict_proba(x.iloc[test_idx])[:, 1]
        probabilities[test_idx] = fold_probability
        predictions[test_idx] = (fold_probability >= threshold).astype(int)
        rows.append(
            {
                "Fold": fold,
                "Training_N": len(train_idx),
                "Test_N": len(test_idx),
                "Best_C": search.best_params_["model__C"],
                "Best_L1_Ratio": search.best_params_["model__l1_ratio"],
                "Training_Only_Threshold": threshold,
            }
        )
    if np.isnan(probabilities).any() or (predictions < 0).any():
        raise RuntimeError("Cross-fitted prediction generation was incomplete")
    return probabilities, predictions, pd.DataFrame(rows)


def run_model(model_name: str, features: list[str], model_helpers, rule_helpers) -> dict[str, pd.DataFrame]:
    discovery = pd.read_csv(OUT / "adni_discovery_raw_aligned_features.csv", low_memory=False)
    holdout = pd.read_csv(OUT / "adni_holdout_raw_aligned_features.csv", low_memory=False)
    reader_master = pd.read_csv(OUT / "holdout_reader_master.csv", low_memory=False)
    experts = pd.read_csv(EXPERT_FILE)

    discovery = discovery.loc[discovery["Strict36_Outcome"].notna()].copy().reset_index(drop=True)
    strict_holdout = holdout.loc[holdout["Strict36_Outcome"].notna()].copy()
    y_discovery = discovery["Strict36_Outcome"].astype(int).to_numpy()
    cross_probability, cross_prediction, folds = crossfit_discovery(
        model_helpers, discovery[features], y_discovery
    )
    global_threshold = model_helpers.choose_youden_threshold(y_discovery, cross_probability)

    final = model_helpers.fit_final(discovery[features], y_discovery)
    nonoverlap = strict_holdout.loc[~strict_holdout["Overlaps_Discovery"].astype(bool)].copy()
    external_probability = final.predict_proba(nonoverlap[features])[:, 1]
    external_prediction = (external_probability >= global_threshold).astype(int)

    discovery_prediction_map = pd.DataFrame(
        {
            "RID": discovery["RID"].astype(int),
            "AI_Probability": cross_probability,
            "AI_Prediction": cross_prediction,
            "Prediction_Source": "five_fold_crossfitted_discovery_prediction",
        }
    )
    external_prediction_map = pd.DataFrame(
        {
            "RID": nonoverlap["RID"].astype(int),
            "AI_Probability": external_probability,
            "AI_Prediction": external_prediction,
            "Prediction_Source": "frozen_model_nonoverlapping_prediction",
        }
    )
    prediction_map = pd.concat([discovery_prediction_map, external_prediction_map], ignore_index=True)

    pooled = (
        experts.groupby("CaseID")
        .agg(
            Pooled_Stage1_Probability=("Stage1_Prob", "mean"),
            Pooled_Stage2_Probability=("Stage2_Prob", "mean"),
            N_Experts=("Expert", "nunique"),
        )
        .reset_index()
    )
    benchmark = (
        strict_holdout[["ID", "RID", "Strict36_Outcome", "Overlaps_Discovery"]]
        .merge(prediction_map, on="RID", how="inner", validate="one_to_one")
        .merge(pooled, left_on="ID", right_on="CaseID", how="inner", validate="one_to_one")
    )
    benchmark = benchmark.merge(
        reader_master[["ID", "Expert_Stage2_SD"]], on="ID", how="left", validate="one_to_one"
    )
    benchmark["Expert_Prediction"] = (
        benchmark["Pooled_Stage2_Probability"] >= rule_helpers.EXPERT_THRESHOLD
    ).astype(int)
    benchmark["Gray_Zone"] = benchmark["Pooled_Stage2_Probability"].between(
        rule_helpers.GRAY_LOWER, rule_helpers.GRAY_UPPER, inclusive="both"
    )
    benchmark["RuleC_Probability"] = np.where(
        benchmark["Gray_Zone"],
        benchmark["AI_Probability"],
        benchmark["Pooled_Stage2_Probability"],
    )
    benchmark["RuleC_Prediction"] = np.where(
        benchmark["Gray_Zone"],
        benchmark["AI_Prediction"],
        benchmark["Expert_Prediction"],
    ).astype(int)

    y = benchmark["Strict36_Outcome"].astype(int).to_numpy()
    arrays = {
        "Crossfitted_AI": (
            benchmark["AI_Probability"].to_numpy(),
            benchmark["AI_Prediction"].to_numpy(),
        ),
        "Pooled_expert_stage2": (
            benchmark["Pooled_Stage2_Probability"].to_numpy(),
            benchmark["Expert_Prediction"].to_numpy(),
        ),
        "Crossfitted_RuleC_simulation": (
            benchmark["RuleC_Probability"].to_numpy(),
            benchmark["RuleC_Prediction"].to_numpy(),
        ),
    }
    performance = pd.DataFrame(
        [rule_helpers.metrics(y, p, pred, label) for label, (p, pred) in arrays.items()]
    )
    performance.insert(0, "Analysis_Model", model_name)
    ci = rule_helpers.bootstrap_metric_cis(y, arrays)
    ci.insert(0, "Analysis_Model", model_name)
    paired, reclassification = rule_helpers.paired_rulec_statistics(
        y,
        arrays["Pooled_expert_stage2"][0],
        arrays["Pooled_expert_stage2"][1],
        arrays["Crossfitted_RuleC_simulation"][0],
        arrays["Crossfitted_RuleC_simulation"][1],
    )
    paired.insert(0, "Analysis_Model", model_name)
    reclassification.insert(0, "Analysis_Model", model_name)

    long = experts.merge(
        benchmark[["ID", "Strict36_Outcome", "AI_Probability", "AI_Prediction"]],
        left_on="CaseID",
        right_on="ID",
        how="inner",
    )
    reader_rows = []
    for expert, part in long.groupby("Expert"):
        yy = part["Strict36_Outcome"].astype(int).to_numpy()
        ep = part["Stage2_Prob"].astype(float).to_numpy()
        ed = (ep >= rule_helpers.EXPERT_THRESHOLD).astype(int)
        gray = (ep >= rule_helpers.GRAY_LOWER) & (ep <= rule_helpers.GRAY_UPPER)
        rp = np.where(gray, part["AI_Probability"].to_numpy(), ep)
        rd = np.where(gray, part["AI_Prediction"].to_numpy(), ed).astype(int)
        for label, p, pred in [("Expert_stage2", ep, ed), ("Crossfitted_RuleC", rp, rd)]:
            result = rule_helpers.metrics(yy, p, pred, label)
            result.update(
                {
                    "Analysis_Model": model_name,
                    "Expert": expert,
                    "Gray_Zone_Cases": int(gray.sum()),
                }
            )
            reader_rows.append(result)

    folds.insert(0, "Analysis_Model", model_name)
    benchmark.insert(0, "Analysis_Model", model_name)
    summary = pd.DataFrame(
        [
            {
                "Analysis_Model": model_name,
                "N": len(benchmark),
                "Events": int(y.sum()),
                "Nonevents": int((y == 0).sum()),
                "Crossfitted_Overlap_Cases": int(benchmark["Overlaps_Discovery"].sum()),
                "Frozen_Nonoverlap_Cases": int((~benchmark["Overlaps_Discovery"].astype(bool)).sum()),
                "Pooled_Gray_Zone_Cases": int(benchmark["Gray_Zone"].sum()),
                "Global_OOF_Threshold_For_Nonoverlap": global_threshold,
            }
        ]
    )
    return {
        "summary": summary,
        "performance": performance,
        "ci": ci,
        "paired": paired,
        "reclassification": reclassification,
        "per_reader": pd.DataFrame(reader_rows),
        "folds": folds,
        "benchmark": benchmark,
    }


def main() -> None:
    model_helpers = load_module("04_fit_leakage_controlled_models.py", "model_helpers")
    rule_helpers = load_module("02_rulec_statistics_core.py", "rule_helpers")
    discovery = pd.read_csv(OUT / "adni_discovery_raw_aligned_features.csv", nrows=2)
    mri = [col for col in discovery.columns if col.startswith("ST")]
    models = {
        "primary_clinical_csf_mri": model_helpers.CLINICAL + model_helpers.CSF + mri,
        "ablation_clinical_mri": model_helpers.CLINICAL + mri,
    }
    collected: dict[str, list[pd.DataFrame]] = {}
    for model_name, features in models.items():
        print(f"Running {model_name} with {len(features)} features", flush=True)
        result = run_model(model_name, features, model_helpers, rule_helpers)
        for key, frame in result.items():
            collected.setdefault(key, []).append(frame)
    for key, frames in collected.items():
        pd.concat(frames, ignore_index=True).to_csv(
            OUT / f"crossfitted_expert_{key}.csv", index=False
        )

    summary = pd.concat(collected["summary"], ignore_index=True)
    performance = pd.concat(collected["performance"], ignore_index=True)
    paired = pd.concat(collected["paired"], ignore_index=True)
    print("\nSummary:")
    print(summary.to_string(index=False))
    print("\nPerformance:")
    print(
        performance[
            [
                "Analysis_Model",
                "Model",
                "N",
                "Events",
                "AUC",
                "Brier",
                "Sensitivity",
                "Specificity",
                "FP",
                "FN",
            ]
        ].to_string(index=False)
    )
    print("\nPaired Rule C changes:")
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
