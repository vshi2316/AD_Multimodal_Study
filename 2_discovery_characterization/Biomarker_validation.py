"""

Implements AT composite and neuroinflammation analyses described in the
manuscript using discovery-cohort CSF biomarkers.
"""

import argparse
import json
import os
import warnings
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Biomarker validation for subtype pathology")
    parser.add_argument("--subtype_file", required=True, help="Subtype assignment CSV")
    parser.add_argument("--csf_file", required=True, help="CSF biomarker CSV")
    parser.add_argument("--output_dir", default="./step9B_results", help="Output directory")
    parser.add_argument("--id_col", default="ID", help="Subject ID column")
    return parser.parse_args()


def zscore(series):
    values = pd.to_numeric(series, errors="coerce")
    return (values - values.mean()) / values.std(ddof=0)


def eta_squared(values, groups):
    frame = pd.DataFrame({"value": values, "group": groups}).dropna()
    if frame["group"].nunique() < 2:
        return np.nan
    overall_mean = frame["value"].mean()
    grouped = frame.groupby("group")
    ss_between = sum(len(group) * (group["value"].mean() - overall_mean) ** 2 for _, group in grouped)
    ss_total = ((frame["value"] - overall_mean) ** 2).sum()
    return float(ss_between / ss_total) if ss_total > 0 else np.nan


def pairwise_tests(frame, variable, group_col="Subtype"):
    results = []
    groups = sorted(frame[group_col].dropna().unique())
    for g1, g2 in combinations(groups, 2):
        x = pd.to_numeric(frame.loc[frame[group_col] == g1, variable], errors="coerce").dropna()
        y = pd.to_numeric(frame.loc[frame[group_col] == g2, variable], errors="coerce").dropna()
        if len(x) < 3 or len(y) < 3:
            continue
        stat, p_value = stats.mannwhitneyu(x, y, alternative="two-sided")
        pooled_sd = np.sqrt((((len(x) - 1) * x.std(ddof=1) ** 2) + ((len(y) - 1) * y.std(ddof=1) ** 2)) / (len(x) + len(y) - 2))
        cohens_d = (x.mean() - y.mean()) / pooled_sd if pooled_sd > 0 else np.nan
        results.append({
            "Variable": variable,
            "Group1": int(g1),
            "Group2": int(g2),
            "U": float(stat),
            "P": float(p_value),
            "Cohens_d": float(cohens_d),
        })
    return results


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    subtype = pd.read_csv(args.subtype_file)
    csf = pd.read_csv(args.csf_file)
    subtype[args.id_col] = subtype[args.id_col].astype(str)
    csf[args.id_col] = csf[args.id_col].astype(str)

    frame = subtype.merge(csf, on=args.id_col, how="inner")
    subtype_col = "VAE_Subtype" if "VAE_Subtype" in frame.columns else "Projected_Subtype"
    frame = frame.rename(columns={subtype_col: "Subtype"})

    if "ABETA42_ABETA40_RATIO" not in frame.columns or "PTAU181" not in frame.columns:
        raise ValueError("CSF file must contain ABETA42_ABETA40_RATIO and PTAU181")

    frame["AT_Composite"] = (zscore(-pd.to_numeric(frame["ABETA42_ABETA40_RATIO"], errors="coerce")) + zscore(frame["PTAU181"])) / 2.0

    variables = [column for column in [
        "PTAU181", "ABETA42_ABETA40_RATIO", "ABETA40", "AT_Composite", "sTREM2", "PGRN", "PROGRANULIN"
    ] if column in frame.columns]

    global_results = []
    pairwise_results = []

    for variable in variables:
        subset = frame[["Subtype", variable]].dropna()
        if subset["Subtype"].nunique() < 2 or len(subset) < 10:
            continue
        groups = [subset.loc[subset["Subtype"] == group, variable].values for group in sorted(subset["Subtype"].unique())]
        kruskal = stats.kruskal(*groups)
        anova = stats.f_oneway(*groups)
        global_results.append({
            "Variable": variable,
            "N": int(len(subset)),
            "Kruskal_H": float(kruskal.statistic),
            "Kruskal_P": float(kruskal.pvalue),
            "ANOVA_F": float(anova.statistic),
            "ANOVA_P": float(anova.pvalue),
            "Eta_Squared": eta_squared(subset[variable], subset["Subtype"]),
        })
        pairwise_results.extend(pairwise_tests(frame, variable))

        plt.figure(figsize=(7, 5))
        sns.boxplot(data=subset, x="Subtype", y=variable, palette="Set2")
        sns.stripplot(data=subset, x="Subtype", y=variable, color="black", alpha=0.35, size=3)
        plt.title(variable)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, f"{variable}_by_subtype.png"), dpi=300)
        plt.close()

    global_df = pd.DataFrame(global_results)
    if not global_df.empty:
        global_df["Kruskal_Q"] = multipletests(global_df["Kruskal_P"], method="fdr_bh")[1]
        global_df["ANOVA_Q"] = multipletests(global_df["ANOVA_P"], method="fdr_bh")[1]
    global_df.to_csv(os.path.join(args.output_dir, "biomarker_global_results.csv"), index=False)

    pairwise_df = pd.DataFrame(pairwise_results)
    if not pairwise_df.empty:
        pairwise_df["Q"] = multipletests(pairwise_df["P"], method="fdr_bh")[1]
    pairwise_df.to_csv(os.path.join(args.output_dir, "biomarker_pairwise_results.csv"), index=False)

    summary = {
        "n_participants": int(len(frame)),
        "variables_tested": variables,
        "significant_global_kruskal": [] if global_df.empty else global_df.loc[global_df["Kruskal_Q"] < 0.05, "Variable"].tolist(),
    }
    with open(os.path.join(args.output_dir, "biomarker_validation_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Saved biomarker validation outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
