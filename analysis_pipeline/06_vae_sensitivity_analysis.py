from __future__ import annotations

import json
import os
import sys
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "python_pkgs"))

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2_contingency, f_oneway, kruskal, mannwhitneyu
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm


ROOT = Path(os.environ.get("AD_MULTIMODAL_DATA_ROOT", ".")).expanduser().resolve()
OUT = Path(os.environ.get("AD_MULTIMODAL_OUTPUT_DIR", str(HERE / "outputs"))).expanduser()
DISCOVERY_DIR = ROOT / "Derived_Inputs" / "Discovery_CSF_Cohort"
SUBTYPE_FILE = ROOT / "Analysis_Inputs" / "VAE_Output" / "subtype_assignments.csv"


def bh_adjust(p_values: list[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.minimum(adjusted, 1.0)
    return output


def partial_eta_squared(table: pd.DataFrame, effect: str) -> float:
    ss_effect = float(table.loc[effect, "sum_sq"])
    ss_residual = float(table.loc["Residual", "sum_sq"])
    return ss_effect / (ss_effect + ss_residual)


def ancova_for_feature(data: pd.DataFrame, feature: str) -> dict:
    base_cols = [feature, "VAE_Subtype", "MMSE", "AGE", "SEX", "EDUCATION", "GDS"]
    complete = data[base_cols].dropna().copy()
    effect = "C(VAE_Subtype)"
    unadjusted_fit = ols(f"{feature} ~ C(VAE_Subtype)", data=complete).fit()
    adjusted_fit = ols(
        f"{feature} ~ C(VAE_Subtype) + MMSE + AGE + C(SEX)", data=complete
    ).fit()
    expanded_fit = ols(
        f"{feature} ~ C(VAE_Subtype) + MMSE + AGE + C(SEX) + EDUCATION + GDS",
        data=complete,
    ).fit()
    unadjusted = anova_lm(unadjusted_fit, typ=2)
    adjusted = anova_lm(adjusted_fit, typ=2)
    expanded = anova_lm(expanded_fit, typ=2)
    eta_unadjusted = partial_eta_squared(unadjusted, effect)
    eta_adjusted = partial_eta_squared(adjusted, effect)
    eta_expanded = partial_eta_squared(expanded, effect)
    return {
        "Feature": feature,
        "N_Complete": len(complete),
        "Unadjusted_P": float(unadjusted.loc[effect, "PR(>F)"]),
        "Unadjusted_Partial_Eta2": eta_unadjusted,
        "Adjusted_Age_Sex_MMSE_P": float(adjusted.loc[effect, "PR(>F)"]),
        "Adjusted_Age_Sex_MMSE_Partial_Eta2": eta_adjusted,
        "Expanded_Age_Sex_MMSE_Education_GDS_P": float(expanded.loc[effect, "PR(>F)"]),
        "Expanded_Age_Sex_MMSE_Education_GDS_Partial_Eta2": eta_expanded,
        "Eta2_Percent_Change_Age_Sex_MMSE": 100 * (eta_adjusted - eta_unadjusted) / eta_unadjusted
        if eta_unadjusted > 0
        else np.nan,
        "Eta2_Percent_Change_Expanded": 100 * (eta_expanded - eta_unadjusted) / eta_unadjusted
        if eta_unadjusted > 0
        else np.nan,
    }


def subtype_order(data: pd.DataFrame, outcome_col: str) -> tuple[list[int], pd.DataFrame]:
    rates = (
        data.loc[data[outcome_col].notna()]
        .groupby("VAE_Subtype")[outcome_col]
        .agg(["count", "sum", "mean"])
        .reset_index()
        .rename(columns={"count": "N", "sum": "Events", "mean": "Conversion_Rate"})
        .sort_values("Conversion_Rate")
    )
    return rates["VAE_Subtype"].astype(int).tolist(), rates


def strem2_analysis(data: pd.DataFrame, ordered_subtypes: list[int]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    available = data[["ID", "VAE_Subtype", "STREM2"]].dropna().copy()
    groups = [available.loc[available["VAE_Subtype"] == subtype, "STREM2"].to_numpy() for subtype in ordered_subtypes]
    omnibus_h, omnibus_p = kruskal(*groups)
    pair_rows = []
    for first, second in combinations(ordered_subtypes, 2):
        x = available.loc[available["VAE_Subtype"] == first, "STREM2"].to_numpy()
        y = available.loc[available["VAE_Subtype"] == second, "STREM2"].to_numpy()
        statistic, p_value = mannwhitneyu(x, y, alternative="two-sided")
        pooled_sd = np.sqrt(((len(x) - 1) * np.var(x, ddof=1) + (len(y) - 1) * np.var(y, ddof=1)) / (len(x) + len(y) - 2))
        smd = (np.mean(y) - np.mean(x)) / pooled_sd if pooled_sd > 0 else np.nan
        pair_rows.append(
            {
                "Subtype_1": first,
                "Subtype_2": second,
                "N_1": len(x),
                "N_2": len(y),
                "Mean_1": np.mean(x),
                "Mean_2": np.mean(y),
                "SMD_Subtype2_Minus_Subtype1": smd,
                "Mann_Whitney_U": statistic,
                "P": p_value,
            }
        )
    pairwise = pd.DataFrame(pair_rows)
    pairwise["BH_Q"] = bh_adjust(pairwise["P"].tolist())
    descriptive = (
        available.groupby("VAE_Subtype")["STREM2"]
        .agg(N="count", Mean="mean", SD="std", Median="median")
        .reset_index()
    )
    high = ordered_subtypes[-1]
    low = ordered_subtypes[0]
    high_low = pairwise.loc[
        ((pairwise["Subtype_1"] == low) & (pairwise["Subtype_2"] == high))
        | ((pairwise["Subtype_1"] == high) & (pairwise["Subtype_2"] == low))
    ].iloc[0]
    summary = {
        "n_available": int(len(available)),
        "omnibus_kruskal_h": float(omnibus_h),
        "omnibus_p": float(omnibus_p),
        "low_risk_subtype": int(low),
        "high_risk_subtype": int(high),
        "high_vs_low_smd": float(high_low["SMD_Subtype2_Minus_Subtype1"]),
        "high_vs_low_p": float(high_low["P"]),
        "high_vs_low_bh_q": float(high_low["BH_Q"]),
    }
    return descriptive, pairwise, summary


def main() -> None:
    clinical = pd.read_csv(DISCOVERY_DIR / "Clinical_data.csv")
    mri = pd.read_csv(DISCOVERY_DIR / "RNA_plasma.csv")
    metabolites = pd.read_csv(DISCOVERY_DIR / "metabolites.csv")
    subtype = pd.read_csv(SUBTYPE_FILE)
    endpoint = pd.read_csv(OUT / "adni_discovery_endpoints.csv")[["ID", "Strict36_Outcome"]]
    data = (
        subtype[["ID", "VAE_Subtype", "AD_Conversion"]]
        .merge(clinical, on="ID", how="left", validate="one_to_one")
        .merge(mri, on="ID", how="left", validate="one_to_one")
        .merge(metabolites, on="ID", how="left", validate="one_to_one")
        .merge(endpoint, on="ID", how="left", validate="one_to_one")
    )
    mri_features = [col for col in mri.columns if col.startswith("ST")]
    ancova = pd.DataFrame([ancova_for_feature(data, feature) for feature in mri_features])
    ancova["Adjusted_Age_Sex_MMSE_BH_Q"] = bh_adjust(ancova["Adjusted_Age_Sex_MMSE_P"].tolist())
    ancova["Expanded_BH_Q"] = bh_adjust(
        ancova["Expanded_Age_Sex_MMSE_Education_GDS_P"].tolist()
    )
    ancova.to_csv(OUT / "type2_ancova_mri_subtype.csv", index=False)

    original_order, original_rates = subtype_order(data, "AD_Conversion")
    strict_order, strict_rates = subtype_order(data, "Strict36_Outcome")
    original_rates.insert(0, "Endpoint", "Original_any_followup_label")
    strict_rates.insert(0, "Endpoint", "Strict36")
    pd.concat([original_rates, strict_rates], ignore_index=True).to_csv(
        OUT / "vae_subtype_conversion_rates_original_vs_strict36.csv", index=False
    )

    strict = data.loc[data["Strict36_Outcome"].notna()].copy()
    score_map = {subtype: score for score, subtype in enumerate(strict_order)}
    strict["Subtype_Risk_Score"] = strict["VAE_Subtype"].map(score_map)
    contingency = pd.crosstab(strict["VAE_Subtype"], strict["Strict36_Outcome"])
    chi2, chi_p, chi_df, _ = chi2_contingency(contingency)
    trend_rows = []
    for model_name, predictors in [
        ("unadjusted_ordered_trend", ["Subtype_Risk_Score"]),
        (
            "adjusted_ordered_trend",
            ["Subtype_Risk_Score", "AGE", "SEX", "EDUCATION", "MMSE"],
        ),
    ]:
        model_data = strict[["Strict36_Outcome"] + predictors].dropna()
        design = sm.add_constant(model_data[predictors].astype(float))
        fit = sm.GLM(
            model_data["Strict36_Outcome"].astype(float),
            design,
            family=sm.families.Binomial(),
        ).fit()
        beta = float(fit.params["Subtype_Risk_Score"])
        se = float(fit.bse["Subtype_Risk_Score"])
        trend_rows.append(
            {
                "Model": model_name,
                "N": len(model_data),
                "OR_Per_One_Level_Increase": np.exp(beta),
                "CI_Lower": np.exp(beta - 1.96 * se),
                "CI_Upper": np.exp(beta + 1.96 * se),
                "P": float(fit.pvalues["Subtype_Risk_Score"]),
            }
        )
    trend = pd.DataFrame(trend_rows)
    trend["Omnibus_Chi2"] = chi2
    trend["Omnibus_df"] = chi_df
    trend["Omnibus_P"] = chi_p
    trend.to_csv(OUT / "vae_strict36_conversion_association.csv", index=False)

    descriptive, pairwise, strem2_summary = strem2_analysis(data, strict_order)
    descriptive.to_csv(OUT / "strem2_subtype_descriptive.csv", index=False)
    pairwise.to_csv(OUT / "strem2_pairwise_tests.csv", index=False)

    demographic_rows = []
    for variable in ["EDUCATION", "GDS", "AGE", "MMSE"]:
        complete = data[["VAE_Subtype", variable]].dropna()
        arrays = [group[variable].to_numpy() for _, group in complete.groupby("VAE_Subtype")]
        f_stat, p_value = f_oneway(*arrays)
        fit = ols(f"{variable} ~ C(VAE_Subtype)", data=complete).fit()
        table = anova_lm(fit, typ=2)
        eta = partial_eta_squared(table, "C(VAE_Subtype)")
        demographic_rows.append(
            {"Variable": variable, "N": len(complete), "ANOVA_F": f_stat, "P": p_value, "Partial_Eta2": eta}
        )
    demographics = pd.DataFrame(demographic_rows)
    demographics["BH_Q"] = bh_adjust(demographics["P"].tolist())
    demographics.to_csv(OUT / "vae_demographic_confounding_tests.csv", index=False)

    summary = {
        "mri_features": len(mri_features),
        "median_unadjusted_partial_eta2": float(ancova["Unadjusted_Partial_Eta2"].median()),
        "median_age_sex_mmse_adjusted_partial_eta2": float(
            ancova["Adjusted_Age_Sex_MMSE_Partial_Eta2"].median()
        ),
        "median_expanded_adjusted_partial_eta2": float(
            ancova["Expanded_Age_Sex_MMSE_Education_GDS_Partial_Eta2"].median()
        ),
        "median_percent_change_age_sex_mmse": float(
            ancova["Eta2_Percent_Change_Age_Sex_MMSE"].median()
        ),
        "median_percent_change_expanded": float(ancova["Eta2_Percent_Change_Expanded"].median()),
        "features_q_below_0_05_age_sex_mmse": int((ancova["Adjusted_Age_Sex_MMSE_BH_Q"] < 0.05).sum()),
        "features_q_below_0_05_expanded": int((ancova["Expanded_BH_Q"] < 0.05).sum()),
        "strict36_subtype_order_low_to_high": strict_order,
        "strict36_conversion_omnibus_p": float(chi_p),
        "strict36_unadjusted_ordered_trend_p": float(
            trend.loc[trend["Model"] == "unadjusted_ordered_trend", "P"].iloc[0]
        ),
        "strict36_adjusted_ordered_trend_p": float(
            trend.loc[trend["Model"] == "adjusted_ordered_trend", "P"].iloc[0]
        ),
        "strem2": strem2_summary,
    }
    with (OUT / "vae_statistics_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nDemographic differences:")
    print(demographics.to_string(index=False))
    print("\nSTREM2 pairwise tests:")
    print(pairwise.to_string(index=False))


if __name__ == "__main__":
    main()
