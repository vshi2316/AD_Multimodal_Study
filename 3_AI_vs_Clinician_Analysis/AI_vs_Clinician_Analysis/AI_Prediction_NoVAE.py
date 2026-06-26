"""
No-VAE sensitivity analysis for the AD multimodal human-AI Rule C workflow.

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import binomtest
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler


plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False
sns.set_style("whitegrid")


CLINICAL_FEATURES = ["MMSE", "EDUCATION", "GDS"]
CSF_FEATURES = ["PTAU181", "ABETA42_ABETA40_RATIO", "ABETA40"]
LEAKAGE_PATTERNS = ["FAQ", "FAQTOTAL", "ADAS13", "ADAS", "CDRSB", "CDR"]
EXCLUDED_PATTERNS = LEAKAGE_PATTERNS + [
    "AGE",
    "SEX",
    "GENDER",
    "APOE",
    "DX",
    "DIAGNOSIS",
    "VISCODE",
    "VAE",
    "SUBTYPE",
    "CONVERSION",
    "OUTCOME",
]

TRAIN_TO_TEST_MAP = {
    "MMSE": "MMSE_Baseline",
    "EDUCATION": "Education",
    "GDS": None,
    "PTAU181": "PTAU181",
    "ABETA42_ABETA40_RATIO": "ABETA42_ABETA40_RATIO",
    "ABETA40": "ABETA40",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="No-VAE AI prediction and Rule C sensitivity analysis"
    )
    parser.add_argument("--data_root", type=str, default=".")
    parser.add_argument("--subtype_file", type=str, default=None)
    parser.add_argument("--clinical_file", type=str, default=None)
    parser.add_argument("--smri_file", type=str, default=None)
    parser.add_argument("--csf_file", type=str, default=None)
    parser.add_argument("--test_file", type=str, default=None)
    parser.add_argument("--expert_file", type=str, default=None)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./3_AI_vs_Clinician_Analysis/NoVAE_Sensitivity",
    )
    parser.add_argument("--n_bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260614)
    return parser.parse_args()


def resolve_path(data_root: Path, explicit: Optional[str], candidates: Sequence[str]) -> Path:
    if explicit:
        return Path(explicit)
    for rel in candidates:
        p = data_root / rel
        if p.exists():
            return p
    return data_root / candidates[0]


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def lower_map(columns: Iterable[str]) -> Dict[str, str]:
    return {str(c).lower(): str(c) for c in columns}


def get_col(df: pd.DataFrame, name: str) -> Optional[str]:
    if name in df.columns:
        return name
    m = lower_map(df.columns)
    return m.get(name.lower())


def numeric_series(df: pd.DataFrame, col: Optional[str], n: Optional[int] = None) -> np.ndarray:
    if col is None or col not in df.columns:
        if n is None:
            n = len(df)
        return np.full(n, np.nan)
    return pd.to_numeric(df[col], errors="coerce").to_numpy()


def as_binary(x: Sequence) -> np.ndarray:
    ser = pd.Series(x)
    if ser.dtype == object:
        low = ser.astype(str).str.strip().str.lower()
        return low.isin(["1", "yes", "true", "ad", "converter", "positive", "high"]).astype(int).to_numpy()
    return pd.to_numeric(ser, errors="coerce").fillna(0).astype(int).to_numpy()


def normalize_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "ID" not in out.columns:
        for candidate in ["CaseID", "case_id", "Subject", "PTID", "RID"]:
            if candidate in out.columns:
                out["ID"] = out[candidate].astype(str)
                break
    if "RID" in out.columns:
        out["RID"] = pd.to_numeric(out["RID"], errors="coerce").astype("Int64")
    return out


def collect_train_features(
    subtypes: pd.DataFrame,
    clinical: pd.DataFrame,
    smri: pd.DataFrame,
    csf: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str], Dict[str, List[str]]]:
    for df in [subtypes, clinical, smri, csf]:
        df["ID"] = df["ID"].astype(str)

    y_col = get_col(subtypes, "AD_Conversion")
    if y_col is None:
        raise ValueError("subtype_assignments.csv must include AD_Conversion")

    master = subtypes[["ID", y_col]].rename(columns={y_col: "AD_Conversion"}).copy()

    clinical_used = [c for c in CLINICAL_FEATURES if get_col(clinical, c) is not None]
    for feat in clinical_used:
        c = get_col(clinical, feat)
        master = master.merge(
            clinical[["ID", c]].rename(columns={c: feat}),
            on="ID",
            how="left",
        )

    csf_used = [c for c in CSF_FEATURES if get_col(csf, c) is not None]
    for feat in csf_used:
        c = get_col(csf, feat)
        master = master.merge(
            csf[["ID", c]].rename(columns={c: feat}),
            on="ID",
            how="left",
        )

    mri_used = [
        c for c in smri.columns
        if str(c).startswith("ST") and pd.api.types.is_numeric_dtype(smri[c])
    ]
    if mri_used:
        master = master.merge(smri[["ID"] + mri_used], on="ID", how="left")

    feature_cols = [
        c for c in master.columns
        if c not in ["ID", "AD_Conversion"]
        and not any(pat in c.upper() for pat in EXCLUDED_PATTERNS)
    ]
    feature_cols = [c for c in feature_cols if pd.to_numeric(master[c], errors="coerce").notna().any()]

    feature_groups = {
        "clinical": [c for c in feature_cols if c in clinical_used],
        "csf": [c for c in feature_cols if c in csf_used],
        "mri": [c for c in feature_cols if c.startswith("ST")],
    }
    return master, feature_cols, feature_groups


def collect_holdout_features(test_df: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    test_df = normalize_id_columns(test_df)
    X = pd.DataFrame(index=np.arange(len(test_df)))
    for feat in feature_cols:
        target_col = TRAIN_TO_TEST_MAP.get(feat, feat)
        if target_col is None:
            X[feat] = np.nan
            continue
        actual = get_col(test_df, target_col)
        if actual is None and feat.startswith("ST"):
            actual = get_col(test_df, feat)
        X[feat] = numeric_series(test_df, actual, len(test_df))
    return X


def leakage_check(feature_cols: Sequence[str]) -> None:
    bad = [f for f in feature_cols if any(pat in f.upper() for pat in LEAKAGE_PATTERNS)]
    if bad:
        raise RuntimeError(f"Outcome-proximal leakage features were retained: {bad}")
    z_feats = [f for f in feature_cols if f.upper() in ["Z1", "Z2", "Z3"] or f.upper().startswith("Z")]
    if z_feats:
        raise RuntimeError(f"No-VAE analysis must not contain latent variables: {z_feats}")


def fit_preprocess(X_train: np.ndarray, X_test: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    imputer = IterativeImputer(max_iter=15, random_state=seed, sample_posterior=False)
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_imp)
    X_test_sc = scaler.transform(X_test_imp)
    return X_train_sc, X_test_sc


def lasso_select_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    feature_cols: Sequence[str],
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, List[str], pd.DataFrame]:
    lasso = LassoCV(cv=5, random_state=seed, max_iter=20000)
    lasso.fit(X_train, y_train)
    coefs = np.asarray(lasso.coef_)
    selected_mask = np.abs(coefs) > 1e-8
    if selected_mask.sum() < 3:
        top_idx = np.argsort(np.abs(coefs))[::-1][: min(10, len(coefs))]
        selected_mask = np.zeros(len(coefs), dtype=bool)
        selected_mask[top_idx] = True
    selected = [f for f, keep in zip(feature_cols, selected_mask) if keep]
    table = pd.DataFrame({
        "Feature": feature_cols,
        "Lasso_Coefficient": coefs,
        "Selected": selected_mask.astype(bool),
    }).sort_values("Lasso_Coefficient", key=lambda s: s.abs(), ascending=False)
    return X_train[:, selected_mask], X_test[:, selected_mask], selected, table


def train_elastic_net(X_train: np.ndarray, y_train: np.ndarray, seed: int):
    grid = {
        "C": np.logspace(-3, 1, 20),
        "l1_ratio": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    clf = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        max_iter=20000,
        random_state=seed,
    )
    search = GridSearchCV(clf, grid, cv=cv, scoring="roc_auc", n_jobs=-1, refit=True)
    search.fit(X_train, y_train)
    model = search.best_estimator_
    train_prob = model.predict_proba(X_train)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_train, train_prob)
    idx = int(np.argmax(tpr - fpr))
    threshold = float(thresholds[idx])
    return model, threshold, search.best_params_, train_prob, float(tpr[idx]), float(1 - fpr[idx])


def auc_ci(y: np.ndarray, prob: np.ndarray, reps: int, seed: int) -> Tuple[float, float, float]:
    auc = float(roc_auc_score(y, prob))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(reps):
        idx = rng.choice(np.arange(len(y)), len(y), replace=True)
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(roc_auc_score(y[idx], prob[idx]))
    if len(vals) == 0:
        return auc, np.nan, np.nan
    return auc, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def metrics_from_pred(
    y: np.ndarray,
    prob: np.ndarray,
    pred: np.ndarray,
    label: str,
    threshold: Optional[float],
    reps: int,
    seed: int,
) -> Dict[str, float]:
    y = as_binary(y)
    prob = np.asarray(prob, dtype=float)
    pred = as_binary(pred)
    keep = np.isfinite(prob) & ~pd.isna(y) & ~pd.isna(pred)
    y = y[keep]
    prob = prob[keep]
    pred = pred[keep]
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    auc, lo, hi = auc_ci(y, prob, reps, seed)
    return {
        "model": label,
        "threshold": threshold if threshold is not None else np.nan,
        "auc": auc,
        "auc_ci_lower": lo,
        "auc_ci_upper": hi,
        "sensitivity": tp / (tp + fn) if (tp + fn) else np.nan,
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "ppv": tp / (tp + fp) if (tp + fp) else np.nan,
        "npv": tn / (tn + fn) if (tn + fn) else np.nan,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "brier": float(brier_score_loss(y, prob)) if np.all((prob >= 0) & (prob <= 1)) else np.nan,
    }


def expert_stage2_case_average(expert: pd.DataFrame) -> pd.DataFrame:
    expert = expert.copy()
    case_col = get_col(expert, "CaseID") or get_col(expert, "ID")
    if case_col is None:
        raise ValueError("Expert file must include CaseID or ID")

    prob_col = None
    for candidate in ["Stage2_Conversion_Prob", "Stage2_Prob", "Stage2Probability"]:
        prob_col = get_col(expert, candidate)
        if prob_col is not None:
            break
    if prob_col is None:
        raise ValueError("Expert file must include Stage2_Conversion_Prob or Stage2_Prob")

    out = (
        expert.assign(
            CaseID=expert[case_col].astype(str),
            Stage2_Prob=pd.to_numeric(expert[prob_col], errors="coerce"),
        )
        .groupby("CaseID", as_index=False)
        .agg(expert_stage2_prob=("Stage2_Prob", "mean"))
    )
    return out


def paired_error_table(y: np.ndarray, ref_pred: np.ndarray, new_pred: np.ndarray) -> pd.DataFrame:
    y = as_binary(y)
    ref = as_binary(ref_pred)
    new = as_binary(new_pred)
    non = y == 0
    ev = y == 1

    def exact_mcnemar(mask: np.ndarray, event_label: str) -> Dict[str, float]:
        improved = int(np.sum((ref[mask] == 1) & (new[mask] == 0)))
        worsened = int(np.sum((ref[mask] == 0) & (new[mask] == 1)))
        p = binomtest(min(improved, worsened), improved + worsened, 0.5).pvalue if improved + worsened > 0 else np.nan
        ref_count = int(np.sum(ref[mask] == 1))
        new_count = int(np.sum(new[mask] == 1))
        return {
            "comparison": event_label,
            "expert_count": ref_count,
            "rulec_count": new_count,
            "delta_rulec_minus_expert": new_count - ref_count,
            "improved_pairs": improved,
            "worsened_pairs": worsened,
            "mcnemar_exact_p": p,
        }

    fp = exact_mcnemar(non, "FP change: Rule C no-VAE vs Expert Stage 2")
    fn = exact_mcnemar(ev, "FN change: Rule C no-VAE vs Expert Stage 2")
    fn["expert_count"] = int(np.sum(ref[ev] == 0))
    fn["rulec_count"] = int(np.sum(new[ev] == 0))
    fn["delta_rulec_minus_expert"] = fn["rulec_count"] - fn["expert_count"]
    fn["improved_pairs"] = int(np.sum((ref[ev] == 0) & (new[ev] == 1)))
    fn["worsened_pairs"] = int(np.sum((ref[ev] == 1) & (new[ev] == 0)))
    if fn["improved_pairs"] + fn["worsened_pairs"] > 0:
        fn["mcnemar_exact_p"] = binomtest(
            min(fn["improved_pairs"], fn["worsened_pairs"]),
            fn["improved_pairs"] + fn["worsened_pairs"],
            0.5,
        ).pvalue
    return pd.DataFrame([fp, fn])


def categorical_nri(
    y: np.ndarray,
    ref_pred: np.ndarray,
    new_pred: np.ndarray,
    reps: int,
    seed: int,
) -> pd.DataFrame:
    y = as_binary(y)
    ref = as_binary(ref_pred)
    new = as_binary(new_pred)

    def point(yy, rr, nn):
        ev = yy == 1
        ne = yy == 0
        event_up = np.mean((nn[ev] > rr[ev])) if ev.any() else np.nan
        event_down = np.mean((nn[ev] < rr[ev])) if ev.any() else np.nan
        nonevent_down = np.mean((nn[ne] < rr[ne])) if ne.any() else np.nan
        nonevent_up = np.mean((nn[ne] > rr[ne])) if ne.any() else np.nan
        event_nri = event_up - event_down
        nonevent_nri = nonevent_down - nonevent_up
        return np.array([event_nri + nonevent_nri, event_nri, nonevent_nri, event_up, event_down, nonevent_down, nonevent_up])

    pt = point(y, ref, new)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(reps):
        idx = rng.choice(np.arange(len(y)), len(y), replace=True)
        vals.append(point(y[idx], ref[idx], new[idx])[0])
    vals = np.asarray(vals, dtype=float)
    return pd.DataFrame([{
        "nri": pt[0],
        "nri_ci_lower": np.nanpercentile(vals, 2.5),
        "nri_ci_upper": np.nanpercentile(vals, 97.5),
        "bootstrap_p": 2 * min(np.mean(vals <= 0), np.mean(vals >= 0)),
        "event_nri": pt[1],
        "nonevent_nri": pt[2],
        "event_up": pt[3],
        "event_down": pt[4],
        "nonevent_down": pt[5],
        "nonevent_up": pt[6],
    }])


def net_benefit(y: np.ndarray, prob: np.ndarray, thresholds: Sequence[float], label: str) -> pd.DataFrame:
    y = as_binary(y)
    prob = np.asarray(prob, dtype=float)
    rows = []
    for pt in thresholds:
        pred = prob >= pt
        tp = np.sum(pred & (y == 1))
        fp = np.sum(pred & (y == 0))
        n = len(y)
        nb = tp / n - fp / n * (pt / (1 - pt))
        rows.append({"model": label, "threshold": pt, "net_benefit": nb})
    return pd.DataFrame(rows)


def all_none_net_benefit(y: np.ndarray, thresholds: Sequence[float]) -> pd.DataFrame:
    y = as_binary(y)
    prev = y.mean()
    rows = []
    for pt in thresholds:
        rows.append({"model": "Treat all", "threshold": pt, "net_benefit": prev - (1 - prev) * pt / (1 - pt)})
        rows.append({"model": "Treat none", "threshold": pt, "net_benefit": 0.0})
    return pd.DataFrame(rows)


def save_feature_audit(
    out_dir: Path,
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: Sequence[str],
    feature_groups: Dict[str, List[str]],
    selected_table: pd.DataFrame,
    best_params: Dict,
    train_auc: float,
    threshold: float,
    train_sens: float,
    train_spec: float,
) -> None:
    rows = []
    rows.append({"section": "Supplementary Table 44. No-VAE feature audit and discovery-only training summary"})
    rows.append({"section": "Feature audit"})
    rows.append({
        "section": "Feature audit",
        "n_discovery": len(train),
        "n_holdout": len(test),
        "n_discovery_events": int(train["AD_Conversion"].sum()),
        "n_holdout_events": int(test["AD_Conversion"].sum()),
        "candidate_features_no_vae": len(feature_cols),
        "used_features_no_vae": len(feature_cols),
        "contains_Z_features": any(str(f).upper().startswith("Z") for f in feature_cols),
        "clinical_features_used": json.dumps(feature_groups["clinical"]),
        "csf_features_used": json.dumps(feature_groups["csf"]),
        "mri_feature_count": len(feature_groups["mri"]),
    })
    rows.append({"section": "Discovery-only training summary"})
    rows.append({
        "section": "Discovery-only training summary",
        "best_C": best_params.get("C"),
        "best_l1_ratio": best_params.get("l1_ratio"),
        "discovery_auc_apparent": train_auc,
        "frozen_youden_threshold": threshold,
        "discovery_sensitivity_at_threshold": train_sens,
        "discovery_specificity_at_threshold": train_spec,
    })
    audit = pd.DataFrame(rows)
    selected = selected_table.copy()
    selected.insert(0, "section", "Lasso feature selection")
    pd.concat([audit, selected], ignore_index=True).to_csv(
        out_dir / "44_no_vae_feature_audit_and_training_summary.csv",
        index=False,
    )


def make_figure_30(out_dir: Path, y: np.ndarray, master: pd.DataFrame, dca: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    fpr, tpr, _ = roc_curve(y, master["NoVAE_AI_Probability"])
    axes[0, 0].plot(fpr, tpr, color="#4C78A8", lw=2, label=f"No-VAE AI AUC={roc_auc_score(y, master['NoVAE_AI_Probability']):.3f}")
    fpr2, tpr2, _ = roc_curve(y, master["expert_stage2_prob"])
    axes[0, 0].plot(fpr2, tpr2, color="#F58518", lw=2, label=f"Expert Stage 2 AUC={roc_auc_score(y, master['expert_stage2_prob']):.3f}")
    axes[0, 0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0, 0].set_title("No-VAE AI vs Expert Stage 2")
    axes[0, 0].set_xlabel("1 - Specificity")
    axes[0, 0].set_ylabel("Sensitivity")
    axes[0, 0].legend(loc="lower right")

    perf = pd.DataFrame([
        {"model": "No-VAE AI", "metric": "sensitivity", "value": master.attrs["no_vae_sens"]},
        {"model": "No-VAE AI", "metric": "specificity", "value": master.attrs["no_vae_spec"]},
        {"model": "Expert Stage 2", "metric": "sensitivity", "value": master.attrs["expert_sens"]},
        {"model": "Expert Stage 2", "metric": "specificity", "value": master.attrs["expert_spec"]},
        {"model": "Rule C with no-VAE AI", "metric": "sensitivity", "value": master.attrs["rulec_sens"]},
        {"model": "Rule C with no-VAE AI", "metric": "specificity", "value": master.attrs["rulec_spec"]},
    ])
    sns.barplot(data=perf, x="model", y="value", hue="metric", ax=axes[0, 1], palette=["#4C78A8", "#F58518"])
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_title("Operating profile after removing VAE latent variables")
    axes[0, 1].set_xlabel("")
    axes[0, 1].set_ylabel("Value")
    axes[0, 1].tick_params(axis="x", rotation=15)

    errors = pd.DataFrame([
        {"error": "False positives", "strategy": "Expert Stage 2", "cases": master.attrs["expert_fp"]},
        {"error": "False positives", "strategy": "Rule C no-VAE", "cases": master.attrs["rulec_fp"]},
        {"error": "False negatives", "strategy": "Expert Stage 2", "cases": master.attrs["expert_fn"]},
        {"error": "False negatives", "strategy": "Rule C no-VAE", "cases": master.attrs["rulec_fn"]},
    ])
    sns.barplot(data=errors, x="error", y="cases", hue="strategy", ax=axes[1, 0], palette=["#4C78A8", "#F58518"])
    axes[1, 0].set_title("Error trade-off for Rule C without VAE")
    axes[1, 0].set_xlabel("")
    axes[1, 0].set_ylabel("Cases")

    plot_dca = dca[dca["threshold"] <= 0.80].copy()
    for label, sub in plot_dca.groupby("model"):
        axes[1, 1].plot(sub["threshold"], sub["net_benefit"], lw=2, label=label)
    axes[1, 1].set_title("Decision curve: no-VAE ablation")
    axes[1, 1].set_xlabel("Threshold probability")
    axes[1, 1].set_ylabel("Net benefit")
    axes[1, 1].legend(loc="lower left", fontsize=8)

    for label, ax in zip(["A", "B", "C", "D"], axes.ravel()):
        ax.text(-0.10, 1.08, label, transform=ax.transAxes, fontsize=14, fontweight="bold", va="top")

    fig.tight_layout()
    fig.savefig(out_dir / "Supplementary_Figure_30_NoVAE_Ablation.png", dpi=300)
    fig.savefig(out_dir / "Supplementary_Figure_30_NoVAE_Ablation.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    data_root = Path(args.data_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    subtype_path = resolve_path(data_root, args.subtype_file, ["subtype_assignments.csv"])
    clinical_path = resolve_path(data_root, args.clinical_file, ["Clinical_data.csv"])
    smri_path = resolve_path(data_root, args.smri_file, ["RNA_plasma.csv"])
    csf_path = resolve_path(data_root, args.csf_file, ["metabolites.csv"])
    test_path = resolve_path(
        data_root,
        args.test_file,
        ["AI_vs_Clinician_Test/independent_test_set.csv", "independent_test_set.csv"],
    )
    expert_path = resolve_path(
        data_root,
        args.expert_file,
        ["AI_vs_Clinician_Test/Expert_Predictions_Long.csv", "Expert_Predictions_Long.csv"],
    )

    print("=" * 78)
    print("No-VAE sensitivity: discovery-only retraining + Rule C holdout analysis")
    print("=" * 78)
    for label, path in [
        ("subtype", subtype_path),
        ("clinical", clinical_path),
        ("sMRI", smri_path),
        ("CSF", csf_path),
        ("test", test_path),
        ("expert", expert_path),
    ]:
        print(f"{label:>10s}: {path}")

    subtypes = normalize_id_columns(read_csv_required(subtype_path))
    clinical = normalize_id_columns(read_csv_required(clinical_path))
    smri = normalize_id_columns(read_csv_required(smri_path))
    csf = normalize_id_columns(read_csv_required(csf_path))
    test_df = normalize_id_columns(read_csv_required(test_path))
    expert = read_csv_required(expert_path)

    train, feature_cols, feature_groups = collect_train_features(subtypes, clinical, smri, csf)
    leakage_check(feature_cols)
    test_X_df = collect_holdout_features(test_df, feature_cols)

    y_train = as_binary(train["AD_Conversion"])
    y_test = as_binary(test_df["AD_Conversion"])
    X_train = train[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy()
    X_test = test_X_df[feature_cols].to_numpy()

    X_train_sc, X_test_sc = fit_preprocess(X_train, X_test, args.seed)
    X_train_sel, X_test_sel, selected, selected_table = lasso_select_features(
        X_train_sc, y_train, X_test_sc, feature_cols, args.seed
    )
    model, threshold, best_params, train_prob, train_sens, train_spec = train_elastic_net(
        X_train_sel, y_train, args.seed
    )
    train_auc = float(roc_auc_score(y_train, train_prob))
    test_prob = model.predict_proba(X_test_sel)[:, 1]
    test_pred = (test_prob >= threshold).astype(int)

    save_feature_audit(
        out_dir,
        train,
        test_df,
        feature_cols,
        feature_groups,
        selected_table,
        best_params,
        train_auc,
        threshold,
        train_sens,
        train_spec,
    )

    pred_df = pd.DataFrame({
        "ID": test_df["ID"].astype(str),
        "RID": test_df["RID"] if "RID" in test_df.columns else pd.NA,
        "Actual": y_test.astype(int),
        "Predicted_Prob": test_prob,
        "Predicted_Class": test_pred.astype(int),
        "NoVAE_AI_Probability": test_prob,
        "NoVAE_AI_Predicted_Class": test_pred.astype(int),
        "Frozen_Threshold": threshold,
    })
    pred_df.to_csv(out_dir / "AI_per_patient_predictions_no_vae.csv", index=False)
    pred_df.rename(columns={
        "ID": "CaseID",
        "Predicted_Prob": "AI_Probability",
        "Predicted_Class": "AI_Predicted_Class",
    }).to_csv(out_dir / "AI_Predictions_Final_no_vae.csv", index=False)

    expert_case = expert_stage2_case_average(expert)
    master = test_df[["ID", "RID", "AD_Conversion"]].copy() if "RID" in test_df.columns else test_df[["ID", "AD_Conversion"]].copy()
    master["ID"] = master["ID"].astype(str)
    master["outcome"] = y_test.astype(int)
    master = master.merge(pred_df[["ID", "NoVAE_AI_Probability", "NoVAE_AI_Predicted_Class"]], on="ID", how="left")
    master = master.merge(expert_case, left_on="ID", right_on="CaseID", how="left")
    master["expert_stage2_pred"] = (master["expert_stage2_prob"] >= 0.50).astype(int)
    master["expert_uncertain_40_60"] = master["expert_stage2_prob"].between(0.40, 0.60, inclusive="both")
    master["rulec_no_vae_pred"] = np.where(
        master["expert_uncertain_40_60"],
        master["NoVAE_AI_Predicted_Class"],
        master["expert_stage2_pred"],
    ).astype(int)
    master["rulec_no_vae_prob"] = np.where(
        master["expert_uncertain_40_60"],
        master["NoVAE_AI_Probability"],
        master["expert_stage2_prob"],
    )
    master.to_csv(out_dir / "00_no_vae_case_level_master.csv", index=False)

    y = master["outcome"].to_numpy()
    perf_rows = [
        metrics_from_pred(y, master["NoVAE_AI_Probability"], master["NoVAE_AI_Predicted_Class"], "No-VAE AI", threshold, args.n_bootstrap, args.seed),
        metrics_from_pred(y, master["expert_stage2_prob"], master["expert_stage2_pred"], "Expert Stage 2", 0.50, args.n_bootstrap, args.seed),
        metrics_from_pred(y, master["rulec_no_vae_prob"], master["rulec_no_vae_pred"], "Rule C with no-VAE AI", None, args.n_bootstrap, args.seed),
    ]
    perf = pd.DataFrame(perf_rows)

    paired = paired_error_table(y, master["expert_stage2_pred"], master["rulec_no_vae_pred"])
    nri = categorical_nri(y, master["expert_stage2_pred"], master["rulec_no_vae_pred"], args.n_bootstrap, args.seed)
    perf.to_csv(out_dir / "45_no_vae_holdout_core_metrics.csv", index=False)
    paired.to_csv(out_dir / "45_no_vae_paired_error_change.csv", index=False)
    nri.to_csv(out_dir / "45_no_vae_categorical_nri.csv", index=False)
    pd.concat([
        pd.DataFrame({"section": ["Core operating metrics"]}),
        perf,
        pd.DataFrame({"section": ["Paired error change"]}),
        paired,
        pd.DataFrame({"section": ["Categorical net reclassification improvement"]}),
        nri,
    ], ignore_index=True).to_csv(out_dir / "45_no_vae_holdout_rulec_performance.csv", index=False)

    thresholds = np.round(np.arange(0.05, 0.81, 0.01), 2)
    dca = pd.concat([
        all_none_net_benefit(y, thresholds),
        net_benefit(y, master["NoVAE_AI_Probability"], thresholds, "No-VAE AI"),
        net_benefit(y, master["expert_stage2_prob"], thresholds, "Expert Stage 2"),
        net_benefit(y, master["rulec_no_vae_prob"], thresholds, "Rule C no-VAE"),
    ], ignore_index=True)
    dca.to_csv(out_dir / "46_no_vae_decision_curve.csv", index=False)
    key = dca[dca["threshold"].isin([0.20, 0.30, 0.50])].pivot_table(
        index="threshold", columns="model", values="net_benefit", aggfunc="first"
    ).reset_index()
    key["Avoided_unnecessary_high_risk_per_100_vs_treat_all"] = (
        (key["Rule C no-VAE"] - key["Treat all"]) / (key["threshold"] / (1 - key["threshold"])) * 100
    )
    key.to_csv(out_dir / "46_no_vae_decision_curve_key_thresholds.csv", index=False)

    no_vae_metrics = perf[perf["model"] == "No-VAE AI"].iloc[0]
    expert_metrics = perf[perf["model"] == "Expert Stage 2"].iloc[0]
    rulec_metrics = perf[perf["model"] == "Rule C with no-VAE AI"].iloc[0]
    master.attrs["no_vae_sens"] = no_vae_metrics["sensitivity"]
    master.attrs["no_vae_spec"] = no_vae_metrics["specificity"]
    master.attrs["expert_sens"] = expert_metrics["sensitivity"]
    master.attrs["expert_spec"] = expert_metrics["specificity"]
    master.attrs["rulec_sens"] = rulec_metrics["sensitivity"]
    master.attrs["rulec_spec"] = rulec_metrics["specificity"]
    master.attrs["expert_fp"] = int(expert_metrics["fp"])
    master.attrs["expert_fn"] = int(expert_metrics["fn"])
    master.attrs["rulec_fp"] = int(rulec_metrics["fp"])
    master.attrs["rulec_fn"] = int(rulec_metrics["fn"])
    make_figure_30(out_dir, y, master, dca)

    summary = {
        "analysis": "No-VAE sensitivity analysis",
        "n_discovery": int(len(train)),
        "n_holdout": int(len(test_df)),
        "n_features_no_vae": int(len(feature_cols)),
        "selected_features": selected,
        "best_params": {k: float(v) for k, v in best_params.items()},
        "frozen_youden_threshold": float(threshold),
        "holdout_auc": float(perf.loc[perf["model"] == "No-VAE AI", "auc"].iloc[0]),
        "rulec_fp": int(rulec_metrics["fp"]),
        "rulec_fn": int(rulec_metrics["fn"]),
    }
    (out_dir / "README_no_vae_sensitivity.txt").write_text(
        "\n".join([
            "No-VAE sensitivity analysis outputs",
            "=" * 40,
            "The script excludes Z1-Z3, FAQ, ADAS13, and CDRSB.",
            "Discovery-only preprocessing, feature selection, model tuning, and threshold selection are frozen before holdout evaluation.",
            "Rule C no-VAE applies no-VAE AI only inside the expert Stage 2 40-60% gray zone.",
            "",
            json.dumps(summary, indent=2),
        ]),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(f"Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
