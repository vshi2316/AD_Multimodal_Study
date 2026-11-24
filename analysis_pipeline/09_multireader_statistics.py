from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "python_pkgs"))

import numpy as np
import pandas as pd
from statsmodels.stats.inter_rater import fleiss_kappa


ROOT = Path(os.environ.get("AD_MULTIMODAL_DATA_ROOT", ".")).expanduser().resolve()
OUT = Path(os.environ.get("AD_MULTIMODAL_OUTPUT_DIR", str(HERE / "outputs"))).expanduser()
EXPERT_FILE = ROOT / "Analysis_Inputs" / "AI_vs_Clinician_Test" / "Expert_Predictions_Long.csv"
REPS = 5000
SEED = 20260710


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



def main():
    experts = pd.read_csv(EXPERT_FILE)
    benchmark = pd.read_csv(OUT / "primary_153_case_predictions.csv")
    if len(benchmark) != 153 or not benchmark.ID.is_unique:
        raise ValueError("Reliability requires one row per primary reader case")
    reliability(experts, benchmark.ID.tolist()).to_csv(
        OUT / "multireader_reliability.csv", index=False)


if __name__ == "__main__":
    main()
