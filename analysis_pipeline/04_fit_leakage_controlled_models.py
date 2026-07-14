from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "python_pkgs"))

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore")

OUT = Path(os.environ.get("AD_MULTIMODAL_OUTPUT_DIR", str(HERE / "outputs"))).expanduser()
DISCOVERY_FILE = OUT / "adni_discovery_raw_aligned_features.csv"
HOLDOUT_FILE = OUT / "adni_holdout_raw_aligned_features.csv"

SEED = 20260710
BOOTSTRAP_REPS = 2000

CLINICAL = ["MMSE_Baseline", "Education", "APOE4_Copies"]
CSF = ["ABETA42", "TAU_TOTAL", "PTAU181"]
LEGACY_TRANSPORTABLE = [
    "MMSE_Baseline",
    "Education",
    "ST105CV",
    "ST102TS",
    "ST108TS",
    "ST109CV",
]


def pipeline(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    penalty="elasticnet",
                    solver="saga",
                    max_iter=10000,
                    random_state=seed,
                ),
            ),
        ]
    )


PARAM_GRID = {
    "model__C": [0.05, 0.1, 0.5, 1.0, 5.0],
    "model__l1_ratio": [0.0, 0.5, 1.0],
}


def nested_oof_probabilities(x: pd.DataFrame, y: np.ndarray) -> tuple[np.ndarray, pd.DataFrame]:
    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    prediction_sum = np.zeros(len(y), dtype=float)
    prediction_n = np.zeros(len(y), dtype=int)
    tuning_rows = []
    for split_index, (train_idx, test_idx) in enumerate(outer.split(x, y), start=1):
        inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED + split_index)
        search = GridSearchCV(
            pipeline(SEED + split_index),
            PARAM_GRID,
            scoring="roc_auc",
            cv=inner,
            n_jobs=-1,
            refit=True,
        )
        search.fit(x.iloc[train_idx], y[train_idx])
        prediction_sum[test_idx] += search.predict_proba(x.iloc[test_idx])[:, 1]
        prediction_n[test_idx] += 1
        tuning_rows.append(
            {
                "Outer_Split": split_index,
                "Best_C": search.best_params_["model__C"],
                "Best_L1_Ratio": search.best_params_["model__l1_ratio"],
                "Inner_AUC": search.best_score_,
            }
        )
    if np.any(prediction_n == 0):
        raise RuntimeError("At least one participant lacks an outer-fold prediction")
    return prediction_sum / prediction_n, pd.DataFrame(tuning_rows)


def fit_final(x: pd.DataFrame, y: np.ndarray) -> GridSearchCV:
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    search = GridSearchCV(
        pipeline(SEED),
        PARAM_GRID,
        scoring="roc_auc",
        cv=inner,
        n_jobs=-1,
        refit=True,
    )
    search.fit(x, y)
    return search


def choose_youden_threshold(y: np.ndarray, p: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y, p)
    finite = np.isfinite(thresholds)
    index = np.argmax((tpr - fpr)[finite])
    return float(thresholds[finite][index])


def calibration(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    logit_p = np.log(p / (1 - p))
    try:
        fit = sm.GLM(y, sm.add_constant(logit_p), family=sm.families.Binomial()).fit(disp=0)
        return float(fit.params[0]), float(fit.params[1])
    except Exception:
        return np.nan, np.nan


def fit_platt_from_oof(y: np.ndarray, p: np.ndarray) -> LogisticRegression:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    logit_p = np.log(p / (1 - p)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=10000, random_state=SEED)
    model.fit(logit_p, y)
    return model


def apply_platt(model: LogisticRegression, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    logit_p = np.log(p / (1 - p)).reshape(-1, 1)
    return model.predict_proba(logit_p)[:, 1]


def point_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = p >= threshold
    tp = int(((y == 1) & pred).sum())
    tn = int(((y == 0) & (~pred)).sum())
    fp = int(((y == 0) & pred).sum())
    fn = int(((y == 1) & (~pred)).sum())
    intercept, slope = calibration(y, p)
    return {
        "N": len(y),
        "Events": int(y.sum()),
        "AUC": float(roc_auc_score(y, p)),
        "Brier": float(np.mean((p - y) ** 2)),
        "Calibration_Intercept": intercept,
        "Calibration_Slope": slope,
        "Threshold": threshold,
        "Sensitivity": tp / (tp + fn) if tp + fn else np.nan,
        "Specificity": tn / (tn + fp) if tn + fp else np.nan,
        "PPV": tp / (tp + fp) if tp + fp else np.nan,
        "NPV": tn / (tn + fn) if tn + fn else np.nan,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
    }


def stratified_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    event = np.flatnonzero(y == 1)
    nonevent = np.flatnonzero(y == 0)
    return np.concatenate(
        [rng.choice(event, len(event), replace=True), rng.choice(nonevent, len(nonevent), replace=True)]
    )


def bootstrap_ci(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    rng = np.random.default_rng(SEED)
    values = {key: [] for key in ["AUC", "Brier", "Sensitivity", "Specificity"]}
    for _ in range(BOOTSTRAP_REPS):
        idx = stratified_indices(y, rng)
        yy = y[idx]
        pp = p[idx]
        pred = pp >= threshold
        values["AUC"].append(roc_auc_score(yy, pp))
        values["Brier"].append(np.mean((pp - yy) ** 2))
        values["Sensitivity"].append(np.mean(pred[yy == 1]))
        values["Specificity"].append(np.mean(~pred[yy == 0]))
    output = {}
    for key, samples in values.items():
        output[f"{key}_CI_Lower"] = float(np.percentile(samples, 2.5))
        output[f"{key}_CI_Upper"] = float(np.percentile(samples, 97.5))
    return output


def summarize_coefficients(search: GridSearchCV, features: list[str]) -> pd.DataFrame:
    model = search.best_estimator_.named_steps["model"]
    imputer = search.best_estimator_.named_steps["imputer"]
    transformed_names = list(features)
    if hasattr(imputer, "indicator_") and imputer.indicator_ is not None:
        transformed_names += [f"Missing_{features[index]}" for index in imputer.indicator_.features_]
    coef = model.coef_.ravel()
    return pd.DataFrame({"Feature": transformed_names, "Coefficient": coef}).sort_values(
        "Coefficient", key=lambda s: s.abs(), ascending=False
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    discovery = pd.read_csv(DISCOVERY_FILE, low_memory=False)
    holdout = pd.read_csv(HOLDOUT_FILE, low_memory=False)
    discovery = discovery.loc[discovery["Strict36_Outcome"].notna()].copy()
    independent = holdout.loc[~holdout["Overlaps_Discovery"].astype(bool)].copy()
    mri = [col for col in discovery.columns if col.startswith("ST")]

    feature_sets = {
        "clinical_only": CLINICAL,
        "clinical_plus_csf": CLINICAL + CSF,
        "clinical_plus_mri": CLINICAL + mri,
        "primary_transportable_multimodal": CLINICAL + CSF + mri,
        "legacy_selected_without_gds_abeta40_vae": LEGACY_TRANSPORTABLE,
    }
    y_discovery = discovery["Strict36_Outcome"].astype(int).to_numpy()

    performance_rows = []
    predictions = independent[["ID", "RID", "Strict36_Outcome", "VisitWindow_Outcome", "Original_Outcome"]].copy()
    model_summary = {}

    for model_index, (name, features) in enumerate(feature_sets.items(), start=1):
        print(f"[{model_index}/{len(feature_sets)}] {name}: {len(features)} features", flush=True)
        x_discovery = discovery[features].apply(pd.to_numeric, errors="coerce")
        x_holdout = independent[features].apply(pd.to_numeric, errors="coerce")

        oof, tuning = nested_oof_probabilities(x_discovery, y_discovery)
        tuning.insert(0, "Model", name)
        tuning.to_csv(OUT / f"nested_tuning_{name}.csv", index=False)
        threshold = choose_youden_threshold(y_discovery, oof)
        platt = fit_platt_from_oof(y_discovery, oof)

        final = fit_final(x_discovery, y_discovery)
        holdout_raw = final.predict_proba(x_holdout)[:, 1]
        holdout_platt = apply_platt(platt, holdout_raw)
        predictions[f"{name}_raw_probability"] = holdout_raw
        predictions[f"{name}_platt_probability"] = holdout_platt
        predictions[f"{name}_locked_prediction"] = (holdout_raw >= threshold).astype(int)

        oof_metrics = point_metrics(y_discovery, oof, threshold)
        oof_metrics.update({"Model": name, "Dataset": "discovery_repeated_nested_oof", "Calibration": "raw"})
        performance_rows.append(oof_metrics)

        for endpoint_name, endpoint_col in [
            ("independent_strict36", "Strict36_Outcome"),
            ("independent_visit_window", "VisitWindow_Outcome"),
            ("independent_original", "Original_Outcome"),
        ]:
            eligible = independent[endpoint_col].notna().to_numpy()
            y_holdout = independent.loc[eligible, endpoint_col].astype(int).to_numpy()
            raw_p = holdout_raw[eligible]
            calibrated_p = holdout_platt[eligible]
            raw_result = point_metrics(y_holdout, raw_p, threshold)
            raw_result.update(bootstrap_ci(y_holdout, raw_p, threshold))
            raw_result.update({"Model": name, "Dataset": endpoint_name, "Calibration": "raw"})
            performance_rows.append(raw_result)

            calibrated_threshold = choose_youden_threshold(y_discovery, apply_platt(platt, oof))
            calibrated_result = point_metrics(y_holdout, calibrated_p, calibrated_threshold)
            calibrated_result.update(bootstrap_ci(y_holdout, calibrated_p, calibrated_threshold))
            calibrated_result.update(
                {"Model": name, "Dataset": endpoint_name, "Calibration": "discovery_oof_platt"}
            )
            performance_rows.append(calibrated_result)

        coef = summarize_coefficients(final, features)
        coef.insert(0, "Model", name)
        coef.to_csv(OUT / f"final_coefficients_{name}.csv", index=False)
        model_summary[name] = {
            "n_features": len(features),
            "features": features,
            "nested_oof_auc": float(roc_auc_score(y_discovery, oof)),
            "locked_threshold_from_oof": threshold,
            "final_best_C": final.best_params_["model__C"],
            "final_best_l1_ratio": final.best_params_["model__l1_ratio"],
            "final_inner_cv_auc": float(final.best_score_),
            "platt_intercept": float(platt.intercept_[0]),
            "platt_slope": float(platt.coef_[0, 0]),
            "nonzero_coefficients": int((coef["Coefficient"].abs() > 1e-8).sum()),
        }

    performance = pd.DataFrame(performance_rows)
    performance.to_csv(OUT / "leakage_free_model_performance.csv", index=False)
    predictions.to_csv(OUT / "independent41_new_model_predictions.csv", index=False)
    with (OUT / "leakage_free_model_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(model_summary, handle, indent=2, ensure_ascii=False)

    primary = performance.loc[
        (performance["Model"] == "primary_transportable_multimodal")
        & performance["Dataset"].isin(["discovery_repeated_nested_oof", "independent_strict36"])
    ]
    print("\nPrimary model results:")
    print(
        primary[
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
                "FP",
                "FN",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
