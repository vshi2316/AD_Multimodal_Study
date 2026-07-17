from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "python_pkgs"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.special import expit
from sklearn.metrics import auc, roc_curve
from statsmodels.stats.proportion import proportion_confint


OUT = Path(os.environ.get("AD_MULTIMODAL_OUTPUT_DIR", str(HERE / "outputs"))).expanduser()
FIG = Path(os.environ.get("AD_MULTIMODAL_FIGURE_DIR", str(HERE / "submission_figures"))).expanduser()
MAIN = FIG / "main"
SUPP = FIG / "supplementary"

COLORS = {
    "ai": "#007C83",
    "expert": "#D89C2B",
    "rule": "#B7463A",
    "clinical": "#4C6A92",
    "full": "#6B5B95",
    "gray": "#7A7A7A",
    "light": "#D9E4E5",
    "black": "#202020",
    "green": "#4E8A67",
}


def load_module(filename: str, name: str):
    path = HERE / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "legend.fontsize": 8,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def panel_label(ax, label: str) -> None:
    ax.text(-0.12, 1.08, label, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")


def save_figure(fig, name: str, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{name}.png", dpi=600, bbox_inches="tight")
    fig.savefig(directory / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_roc(ax, y, series: list[tuple[str, np.ndarray, str]]) -> None:
    for label, probability, color in series:
        fpr, tpr, _ = roc_curve(y, probability)
        ax.plot(fpr, tpr, color=color, linewidth=1.8, label=f"{label} (AUC {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=0.8, color="#A0A0A0")
    ax.set(xlabel="1 - Specificity", ylabel="Sensitivity", xlim=(0, 1), ylim=(0, 1))
    ax.legend(frameon=False, loc="lower right")
    ax.set_aspect("equal", adjustable="box")


def calibration_points(y: np.ndarray, p: np.ndarray, bins: int = 6) -> pd.DataFrame:
    data = pd.DataFrame({"y": y, "p": p})
    data["bin"] = pd.qcut(data["p"], q=min(bins, data["p"].nunique()), duplicates="drop")
    return data.groupby("bin", observed=True).agg(predicted=("p", "mean"), observed=("y", "mean"), n=("y", "size")).reset_index()


def plot_calibration(ax, y, series: list[tuple[str, np.ndarray, str]]) -> None:
    ax.plot([0, 1], [0, 1], linestyle="--", color="#A0A0A0", linewidth=0.8)
    for label, probability, color in series:
        points = calibration_points(y, probability)
        ax.plot(points["predicted"], points["observed"], marker="o", markersize=4, linewidth=1.4, color=color, label=label)
    ax.set(xlabel="Mean predicted probability", ylabel="Observed event rate", xlim=(0, 1), ylim=(0, 1))
    ax.legend(frameon=False, loc="upper left")
    ax.set_aspect("equal", adjustable="box")


def feature_label(feature: str) -> str:
    map_prefix = {
        "ST101SV": "Left hippocampal volume",
        "ST102": "Right paracentral",
        "ST103": "Right parahippocampal",
        "ST104": "Right pars opercularis",
        "ST105": "Right pars orbitalis",
        "ST106": "Right pars triangularis",
        "ST107": "Right pericalcarine",
        "ST108": "Right postcentral",
        "ST109CV": "Right posterior cingulate volume",
    }
    suffix = {"CV": "volume", "SA": "surface area", "TA": "thickness", "TS": "thickness variability"}
    if feature in map_prefix:
        return map_prefix[feature]
    for prefix, label in map_prefix.items():
        if feature.startswith(prefix) and prefix.startswith("ST") and len(prefix) == 5:
            return f"{label} {suffix.get(feature[-2:], feature[-2:])}"
    return feature.replace("_", " ")


def fit_contribution_model() -> tuple[pd.DataFrame, pd.DataFrame, float]:
    helpers = load_module("04_fit_leakage_controlled_models.py", "model_helpers_fig")
    discovery = pd.read_csv(OUT / "adni_discovery_raw_aligned_features.csv", low_memory=False)
    discovery = discovery.loc[discovery["Strict36_Outcome"].notna()].copy()
    validation = pd.read_csv(OUT / "nonoverlapping_adni_validation_predictions.csv", low_memory=False)
    mri = [col for col in discovery.columns if col.startswith("ST")]
    features = helpers.CLINICAL + mri
    y = discovery["Strict36_Outcome"].astype(int).to_numpy()
    search = helpers.fit_final(discovery[features], y)
    estimator = search.best_estimator_
    imputed = estimator.named_steps["imputer"].transform(validation[features])
    scaled = estimator.named_steps["scaler"].transform(imputed)
    coefficients = estimator.named_steps["model"].coef_.ravel()
    names = features.copy()
    if len(coefficients) > len(names):
        indicator_features = estimator.named_steps["imputer"].indicator_.features_
        names += [f"Missing_{features[index]}" for index in indicator_features]
    contribution = scaled * coefficients
    contribution_df = pd.DataFrame(contribution, columns=names)
    contribution_df.insert(0, "ID", validation["ID"].values)
    coefficient_df = pd.DataFrame({"Feature": names, "Coefficient": coefficients})
    coefficient_df["Display_Label"] = coefficient_df["Feature"].map(feature_label)
    coefficient_df["Absolute_Coefficient"] = coefficient_df["Coefficient"].abs()
    contribution_df.to_csv(OUT / "clinical_mri_logodds_contributions.csv", index=False)
    coefficient_df.to_csv(OUT / "clinical_mri_final_coefficients_for_figure.csv", index=False)
    return validation, contribution_df, float(estimator.named_steps["model"].intercept_[0])


def figure1() -> None:
    benchmark = pd.read_csv(OUT / "crossfitted_expert_benchmark.csv", low_memory=False)
    benchmark = benchmark.loc[benchmark["Analysis_Model"] == "primary_clinical_csf_mri"].copy()
    macro = pd.read_csv(OUT / "final_multireader_macro_performance.csv")
    pooled = pd.read_csv(OUT / "final_pooled_reader_performance.csv")
    macro_diff = pd.read_csv(OUT / "final_multireader_macro_differences.csv")
    dca = pd.read_csv(OUT / "final_pooled_rulec_dca.csv")
    aibl = pd.read_csv(OUT / "aibl_harmonized_predictions.csv", low_memory=False)
    aibl = aibl.loc[aibl["Strict36_Outcome"].notna()].copy()

    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.8))
    ax = axes[0, 0]
    ax.axis("off")
    boxes = [
        (0.04, 0.72, "VAE discovery\nN = 157"),
        (0.54, 0.72, "Strict model development\nN = 126, events = 66"),
        (0.04, 0.28, "Five-reader benchmark\nN = 153, events = 77"),
        (0.54, 0.28, "Nonoverlapping ADNI validation\nN = 318, events = 104"),
        (0.29, -0.14, "AIBL clinical proxy\nN = 34, events = 16"),
    ]
    for x, y, text in boxes:
        ax.text(x, y, text, transform=ax.transAxes, ha="left", va="center", fontsize=8.6,
                bbox=dict(boxstyle="square,pad=0.45", facecolor="#F4F7F7", edgecolor=COLORS["ai"], linewidth=1.0))
    arrows = [((0.42, 0.72), (0.52, 0.72)), ((0.29, 0.63), (0.19, 0.40)), ((0.70, 0.63), (0.70, 0.40)), ((0.53, 0.20), (0.48, 0.02))]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, xycoords="axes fraction", arrowprops=dict(arrowstyle="->", color="#555555", lw=1))
    ax.set_title("Reconstructed analysis cohorts", loc="left")
    panel_label(ax, "A")

    ax = axes[0, 1]
    metrics = ["Macro_Sensitivity", "Macro_Specificity"]
    labels = ["Sensitivity", "Specificity"]
    x = np.arange(2)
    width = 0.24
    for offset, condition, color in [(-width, "Stage2", COLORS["expert"]), (0, "RuleC", COLORS["rule"])]:
        row = macro.loc[macro["Condition"] == condition].iloc[0]
        values = [row[m] for m in metrics]
        lower = [row[m] - row[f"{m}_CI_Lower"] for m in metrics]
        upper = [row[f"{m}_CI_Upper"] - row[m] for m in metrics]
        ax.bar(x + offset, values, width=width, color=color, label="Expert Stage 2" if condition == "Stage2" else "Rule C")
        ax.errorbar(x + offset, values, yerr=[lower, upper], fmt="none", ecolor=COLORS["black"], capsize=3, lw=0.8)
    ax.set_xticks(x - width / 2, labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Reader-macro estimate")
    ax.legend(frameon=False, loc="lower center")
    ax.set_title("Multi-reader performance")
    panel_label(ax, "B")

    ax = axes[0, 2]
    order = ["Pooled_Expert_Stage2", "Pooled_RuleC"]
    p = pooled.set_index("Model").loc[order]
    x = np.arange(2)
    ax.bar(x - 0.18, p["FP"], width=0.36, color=COLORS["expert"], label="False positives")
    ax.bar(x + 0.18, p["FN"], width=0.36, color=COLORS["rule"], label="False negatives")
    ax.set_xticks(x, ["Expert Stage 2", "Rule C"])
    ax.set_ylabel("Number of cases")
    ax.legend(frameon=False)
    ax.set_title("Pooled error trade-off")
    panel_label(ax, "C")

    ax = axes[1, 0]
    for metric, y_pos, color in [("Macro_Specificity", 1, COLORS["green"]), ("Macro_Sensitivity", 0, COLORS["rule"])]:
        row = macro_diff.loc[macro_diff["Metric"] == metric].iloc[0]
        ax.errorbar(row["Estimate"], y_pos, xerr=[[row["Estimate"] - row["CI_Lower"]], [row["CI_Upper"] - row["Estimate"]]],
                    fmt="o", color=color, capsize=4, markersize=6)
    ax.axvline(0, color="#777777", linestyle="--", lw=0.8)
    ax.set_yticks([0, 1], ["Sensitivity", "Specificity"])
    ax.set_xlabel("Rule C minus Expert Stage 2")
    ax.set_xlim(-0.12, 0.14)
    ax.set_title("Paired reader-macro differences")
    panel_label(ax, "D")

    ax = axes[1, 1]
    for model, color, label in [
        ("Crossfitted_AI", COLORS["ai"], "Cross-fitted AI"),
        ("Pooled_Expert_Stage2", COLORS["expert"], "Expert Stage 2"),
        ("Pooled_RuleC", COLORS["rule"], "Rule C"),
        ("Treat_all", "#888888", "Treat all"),
    ]:
        part = dca.loc[dca["Model"] == model]
        style = "--" if model == "Treat_all" else "-"
        ax.plot(part["Threshold"], part["Net_Benefit"], color=color, linestyle=style, lw=1.5, label=label)
    ax.axhline(0, color="#AAAAAA", lw=0.7)
    ax.set(xlabel="Threshold probability", ylabel="Net benefit", xlim=(0.05, 0.70), ylim=(-0.1, 0.55))
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("Decision-curve analysis")
    panel_label(ax, "E")

    ax = axes[1, 2]
    y = aibl["Strict36_Outcome"].astype(int).to_numpy()
    prob = aibl["Harmonized_Reduced_Model_Probability"].to_numpy()
    plot_roc(ax, y, [("AIBL clinical proxy", prob, COLORS["clinical"])])
    ax.set_title("AIBL strict 36-month validation")
    panel_label(ax, "F")
    fig.tight_layout(w_pad=2.2, h_pad=2.0)
    save_figure(fig, "Figure 1", MAIN)


def figure2() -> None:
    benchmark = pd.read_csv(OUT / "crossfitted_expert_benchmark.csv", low_memory=False)
    benchmark = benchmark.loc[benchmark["Analysis_Model"] == "primary_clinical_csf_mri"].sort_values("ID")
    per_reader = pd.read_csv(OUT / "crossfitted_expert_per_reader.csv")
    per_reader = per_reader.loc[per_reader["Analysis_Model"] == "primary_clinical_csf_mri"]
    independent = pd.read_csv(OUT / "nonoverlapping_adni_validation_predictions.csv", low_memory=False)
    y = benchmark["Strict36_Outcome"].astype(int).to_numpy()
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 8.4))
    plot_roc(
        axes[0, 0],
        y,
        [
            ("Cross-fitted AI", benchmark["AI_Probability"].to_numpy(), COLORS["ai"]),
            ("Expert Stage 2", benchmark["Pooled_Stage2_Probability"].to_numpy(), COLORS["expert"]),
            ("Rule C", benchmark["RuleC_Probability"].to_numpy(), COLORS["rule"]),
        ],
    )
    axes[0, 0].set_title("Five-neurologist benchmark")
    panel_label(axes[0, 0], "A")

    ax = axes[0, 1]
    pivot = per_reader.pivot(index="Expert", columns="Model", values="AUC").sort_index()
    for index, expert in enumerate(pivot.index):
        ax.plot([pivot.loc[expert, "Expert_stage2"], pivot.loc[expert, "Crossfitted_RuleC"]], [index, index], color="#BBBBBB", lw=1.2)
        ax.scatter(pivot.loc[expert, "Expert_stage2"], index, color=COLORS["expert"], s=38)
        ax.scatter(pivot.loc[expert, "Crossfitted_RuleC"], index, color=COLORS["rule"], s=38)
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.set_xlim(0.45, 0.9)
    ax.set_xlabel("AUC")
    ax.legend(handles=[Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["expert"], label="Expert Stage 2"),
                       Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["rule"], label="Rule C")], frameon=False)
    ax.set_title("Reader-level discrimination")
    panel_label(ax, "B")

    plot_calibration(
        axes[1, 0],
        y,
        [
            ("Cross-fitted AI", benchmark["AI_Probability"].to_numpy(), COLORS["ai"]),
            ("Expert Stage 2", benchmark["Pooled_Stage2_Probability"].to_numpy(), COLORS["expert"]),
            ("Rule C", benchmark["RuleC_Probability"].to_numpy(), COLORS["rule"]),
        ],
    )
    axes[1, 0].set_title("Calibration in the reader benchmark")
    panel_label(axes[1, 0], "C")

    ax = axes[1, 1]
    for phase, color in [("ADNI1", COLORS["clinical"]), ("ADNI3", COLORS["green"])]:
        part = independent.loc[independent["Baseline_PHASE"] == phase]
        fpr, tpr, _ = roc_curve(part["Strict36_Outcome"], part["clinical_plus_mri_probability"])
        ax.plot(fpr, tpr, color=color, lw=1.8, label=f"{phase} (AUC {auc(fpr, tpr):.3f}, N={len(part)})")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=0.8, color="#A0A0A0")
    ax.set(xlabel="1 - Specificity", ylabel="Sensitivity", xlim=(0, 1), ylim=(0, 1))
    ax.legend(frameon=False, loc="lower right")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Nonoverlapping ADNI validation")
    panel_label(ax, "D")
    fig.tight_layout(w_pad=2.2, h_pad=2.0)
    save_figure(fig, "Figure 2", MAIN)


def figure3() -> None:
    perf = pd.read_csv(OUT / "leakage_free_model_performance.csv")
    validation_perf = pd.read_csv(OUT / "nonoverlapping_adni_validation_subgroups.csv")
    validation, contributions, intercept = fit_contribution_model()
    coefficients = pd.read_csv(OUT / "clinical_mri_final_coefficients_for_figure.csv")
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.8))

    ax = axes[0, 0]
    oof = perf.loc[(perf["Dataset"] == "discovery_repeated_nested_oof") & (perf["Calibration"] == "raw")].copy()
    label_map = {
        "clinical_only": "Clinical",
        "clinical_plus_csf": "Clinical + CSF",
        "clinical_plus_mri": "Clinical + MRI",
        "primary_transportable_multimodal": "Clinical + CSF + MRI",
        "exploratory_six_feature_ablation": "Six-feature ablation",
    }
    oof["Label"] = oof["Model"].map(label_map)
    oof = oof.sort_values("AUC")
    ax.barh(oof["Label"], oof["AUC"], color=[COLORS["clinical"] if "MRI" not in x else COLORS["ai"] for x in oof["Label"]])
    ax.axvline(0.5, color="#888888", linestyle="--", lw=0.8)
    ax.set_xlim(0.5, 0.72)
    ax.set_xlabel("Repeated nested out-of-fold AUC")
    ax.set_title("Discovery model comparison")
    panel_label(ax, "A")

    ax = axes[0, 1]
    sub = validation_perf.loc[
        validation_perf["Subgroup"].isin(["all", "ADNI1", "ADNI3"])
        & validation_perf["Model"].isin(["clinical_plus_mri", "primary_transportable_multimodal"])
    ].copy()
    sub["Label"] = sub["Model"].map({"clinical_plus_mri": "Clinical + MRI", "primary_transportable_multimodal": "Clinical + CSF + MRI"}) + " | " + sub["Subgroup"]
    sub = sub.sort_values("AUC")
    xerr = np.vstack([sub["AUC"] - sub["AUC_CI_Lower"], sub["AUC_CI_Upper"] - sub["AUC"]])
    ax.errorbar(sub["AUC"], np.arange(len(sub)), xerr=xerr, fmt="o", color=COLORS["ai"], ecolor="#555555", capsize=3)
    ax.axvline(0.5, color="#888888", linestyle="--", lw=0.8)
    ax.set_yticks(np.arange(len(sub)), sub["Label"])
    ax.set_xlim(0.5, 0.85)
    ax.set_xlabel("AUC with 95% bootstrap CI")
    ax.set_title("Independent validation and phase sensitivity")
    panel_label(ax, "B")

    ax = axes[1, 0]
    top = coefficients.nlargest(14, "Absolute_Coefficient").sort_values("Coefficient")
    colors = np.where(top["Coefficient"] >= 0, COLORS["rule"], COLORS["ai"])
    ax.barh(top["Display_Label"], top["Coefficient"], color=colors)
    ax.axvline(0, color="#777777", lw=0.7)
    ax.set_xlabel("Standardized logistic coefficient")
    ax.set_title("Clinical + MRI model coefficients")
    panel_label(ax, "C")

    ax = axes[1, 1]
    probabilities = validation.set_index("ID")["clinical_plus_mri_probability"]
    quantiles = [0.1, 0.5, 0.9]
    selected = []
    for quantile in quantiles:
        target = probabilities.quantile(quantile)
        selected.append((probabilities - target).abs().idxmin())
    case_labels = ["Lower risk", "Intermediate risk", "Higher risk"]
    y_positions = []
    values = []
    labels = []
    colors = []
    cursor = 0
    for case_label, case_id in zip(case_labels, selected):
        row = contributions.set_index("ID").loc[case_id].drop(labels=[], errors="ignore")
        row = row.reindex(row.abs().sort_values(ascending=False).head(4).index)
        for feature, value in row.items():
            y_positions.append(cursor)
            values.append(value)
            labels.append(f"{case_label}: {feature_label(feature)}")
            colors.append(COLORS["rule"] if value > 0 else COLORS["ai"])
            cursor += 1
        cursor += 0.8
    ax.barh(y_positions, values, color=colors)
    ax.set_yticks(y_positions, labels, fontsize=7)
    ax.axvline(0, color="#777777", lw=0.7)
    ax.set_xlabel("Contribution to log odds")
    ax.set_title("Representative case-level explanations")
    panel_label(ax, "D")
    fig.tight_layout(w_pad=2.0, h_pad=2.0)
    save_figure(fig, "Figure 3", MAIN)


def figure5() -> None:
    ancova = pd.read_csv(OUT / "type2_ancova_mri_subtype.csv")
    demographics = pd.read_csv(OUT / "vae_demographic_confounding_tests.csv")
    rates = pd.read_csv(OUT / "vae_subtype_conversion_rates_original_vs_strict36.csv")
    rates = rates.loc[rates["Endpoint"] == "Strict36"].sort_values("Conversion_Rate")
    strem2 = pd.read_csv(OUT / "strem2_subtype_descriptive.csv")
    top = ancova.nlargest(15, "Unadjusted_Partial_Eta2").copy()
    top["Label"] = top["Feature"].map(feature_label)
    top = top.sort_values("Unadjusted_Partial_Eta2")
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.0))

    ax = axes[0, 0]
    y = np.arange(len(top))
    ax.scatter(top["Unadjusted_Partial_Eta2"], y, color=COLORS["expert"], label="Unadjusted")
    ax.scatter(top["Adjusted_Age_Sex_MMSE_Partial_Eta2"], y, color=COLORS["ai"], label="Adjusted")
    for index in range(len(top)):
        ax.plot([top.iloc[index]["Unadjusted_Partial_Eta2"], top.iloc[index]["Adjusted_Age_Sex_MMSE_Partial_Eta2"]], [index, index], color="#BBBBBB", lw=0.8)
    ax.set_yticks(y, top["Label"], fontsize=6.8)
    ax.set_xlabel("Partial eta squared")
    ax.legend(frameon=False)
    ax.set_title("MRI subtype effects")
    panel_label(ax, "A")

    ax = axes[0, 1]
    change = top.sort_values("Eta2_Percent_Change_Age_Sex_MMSE")
    ax.barh(change["Label"], change["Eta2_Percent_Change_Age_Sex_MMSE"], color=np.where(change["Eta2_Percent_Change_Age_Sex_MMSE"] < 0, COLORS["ai"], COLORS["rule"]))
    ax.axvline(0, color="#777777", lw=0.7)
    ax.set_xlabel("Relative effect-size change (%)")
    ax.tick_params(axis="y", labelsize=6.8)
    ax.set_title("Covariate sensitivity")
    panel_label(ax, "B")

    ax = axes[0, 2]
    demographics = demographics.sort_values("Partial_Eta2")
    ax.barh(demographics["Variable"], demographics["Partial_Eta2"], color=[COLORS["expert"] if x == "EDUCATION" else COLORS["gray"] for x in demographics["Variable"]])
    ax.set_xlabel("Partial eta squared")
    ax.set_title("Demographic separation")
    panel_label(ax, "C")

    ax = axes[1, 0]
    ci = np.array([proportion_confint(row.Events, row.N, method="wilson") for row in rates.itertuples()])
    x = np.arange(len(rates))
    ax.bar(x, rates["Conversion_Rate"], color=[COLORS["clinical"], COLORS["expert"], COLORS["rule"]])
    ax.errorbar(x, rates["Conversion_Rate"], yerr=[rates["Conversion_Rate"].to_numpy() - ci[:, 0], ci[:, 1] - rates["Conversion_Rate"].to_numpy()], fmt="none", ecolor=COLORS["black"], capsize=4)
    ax.set_xticks(x, ["Lower", "Intermediate", "Higher"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Strict 36-month conversion rate")
    ax.set_title("Exploratory conversion gradient")
    panel_label(ax, "D")

    ax = axes[1, 1]
    strem2 = strem2.sort_values("VAE_Subtype")
    ax.errorbar(strem2["VAE_Subtype"].astype(str), strem2["Mean"], yerr=strem2["SD"], fmt="o", color=COLORS["ai"], capsize=4)
    ax.axhline(0, color="#AAAAAA", lw=0.7)
    ax.set_xlabel("VAE subtype")
    ax.set_ylabel("Standardized sTREM2")
    ax.set_title("No sTREM2 separation (P = 0.839)")
    panel_label(ax, "E")

    ax = axes[1, 2]
    summary_values = [
        ("Median unadjusted", ancova["Unadjusted_Partial_Eta2"].median()),
        ("Age/sex/MMSE adjusted", ancova["Adjusted_Age_Sex_MMSE_Partial_Eta2"].median()),
        ("Expanded adjusted", ancova["Expanded_Age_Sex_MMSE_Education_GDS_Partial_Eta2"].median()),
    ]
    ax.bar([x[0] for x in summary_values], [x[1] for x in summary_values], color=[COLORS["expert"], COLORS["ai"], COLORS["green"]])
    ax.tick_params(axis="x", rotation=25)
    ax.set_ylabel("Median partial eta squared")
    ax.set_title("Summary of adjusted MRI effects")
    panel_label(ax, "F")
    fig.tight_layout(w_pad=2.0, h_pad=2.0)
    save_figure(fig, "Figure 5", MAIN)


def supplementary_figures() -> None:
    benchmark = pd.read_csv(OUT / "crossfitted_expert_benchmark.csv", low_memory=False)
    benchmark = benchmark.loc[benchmark["Analysis_Model"] == "primary_clinical_csf_mri"].sort_values("ID")
    y = benchmark["Strict36_Outcome"].astype(int).to_numpy()
    pooled = pd.read_csv(OUT / "final_pooled_reader_performance.csv")
    macro = pd.read_csv(OUT / "final_multireader_macro_performance.csv")
    per_reader = pd.read_csv(OUT / "crossfitted_expert_per_reader.csv")
    per_reader = per_reader.loc[per_reader["Analysis_Model"] == "primary_clinical_csf_mri"]
    dca = pd.read_csv(OUT / "final_pooled_rulec_dca.csv")
    cross_perf = pd.read_csv(OUT / "crossfitted_expert_performance.csv")
    subgroup = pd.read_csv(OUT / "nonoverlapping_adni_validation_subgroups.csv")
    independent = pd.read_csv(OUT / "nonoverlapping_adni_validation_predictions.csv", low_memory=False)
    aibl = pd.read_csv(OUT / "aibl_harmonized_predictions.csv", low_memory=False)
    aibl = aibl.loc[aibl["Strict36_Outcome"].notna()].copy()

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.7))
    for ax, model, color in zip(axes, ["Crossfitted_AI", "Pooled_Expert_Stage2", "Pooled_RuleC"], [COLORS["ai"], COLORS["expert"], COLORS["rule"]]):
        row = pooled.set_index("Model").loc[model]
        matrix = np.array([[row["TN"], row["FP"]], [row["FN"], row["TP"]]])
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=matrix.max())
        for i in range(2):
            for j in range(2):
                ax.text(j, i, int(matrix[i, j]), ha="center", va="center", fontsize=13, color="white" if matrix[i, j] > matrix.max() / 2 else "black")
        ax.set_xticks([0, 1], ["Predicted 0", "Predicted 1"])
        ax.set_yticks([0, 1], ["Observed 0", "Observed 1"])
        ax.set_title({"Crossfitted_AI": "Cross-fitted AI", "Pooled_Expert_Stage2": "Expert Stage 2", "Pooled_RuleC": "Rule C"}[model])
    fig.tight_layout()
    save_figure(fig, "Supplementary Figure 1", SUPP)

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8))
    plot_calibration(axes[0], y, [("AI", benchmark["AI_Probability"].to_numpy(), COLORS["ai"]), ("Expert", benchmark["Pooled_Stage2_Probability"].to_numpy(), COLORS["expert"]), ("Rule C", benchmark["RuleC_Probability"].to_numpy(), COLORS["rule"])])
    for column, label, color in [("AI_Probability", "AI", COLORS["ai"]), ("Pooled_Stage2_Probability", "Expert", COLORS["expert"]), ("RuleC_Probability", "Rule C", COLORS["rule"])]:
        axes[1].hist(benchmark.loc[benchmark["Strict36_Outcome"] == 0, column], bins=np.linspace(0, 1, 16), histtype="step", linewidth=1.4, color=color, label=f"{label}, non-event")
    axes[1].set(xlabel="Predicted probability", ylabel="Number of cases")
    axes[1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save_figure(fig, "Supplementary Figure 2", SUPP)

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    for model, color in [("Crossfitted_AI", COLORS["ai"]), ("Pooled_Expert_Stage2", COLORS["expert"]), ("Pooled_RuleC", COLORS["rule"]), ("Treat_all", COLORS["gray"])]:
        part = dca.loc[dca["Model"] == model]
        ax.plot(part["Threshold"], part["Net_Benefit"], color=color, linestyle="--" if model == "Treat_all" else "-", label=model.replace("_", " "))
    ax.axhline(0, color="#AAAAAA", lw=0.7)
    ax.set(xlabel="Threshold probability", ylabel="Net benefit", xlim=(0.05, 0.70), ylim=(-0.10, 0.55))
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, "Supplementary Figure 3", SUPP)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    pivot_auc = per_reader.pivot(index="Expert", columns="Model", values="AUC")
    pivot_spec = per_reader.pivot(index="Expert", columns="Model", values="Specificity")
    for ax, pivot, title, xlabel in [(axes[0], pivot_auc, "Reader-level AUC", "AUC"), (axes[1], pivot_spec, "Reader-level specificity", "Specificity")]:
        for index, expert in enumerate(pivot.index):
            ax.plot([pivot.loc[expert, "Expert_stage2"], pivot.loc[expert, "Crossfitted_RuleC"]], [index, index], color="#BBBBBB")
            ax.scatter(pivot.loc[expert, "Expert_stage2"], index, color=COLORS["expert"])
            ax.scatter(pivot.loc[expert, "Crossfitted_RuleC"], index, color=COLORS["rule"])
        ax.set_yticks(range(len(pivot.index)), pivot.index)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
    fig.tight_layout()
    save_figure(fig, "Supplementary Figure 4", SUPP)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    bins = np.linspace(0, 1, 16)
    for outcome, color, label in [(0, COLORS["clinical"], "Non-event"), (1, COLORS["rule"], "Event")]:
        ax.hist(benchmark.loc[benchmark["Strict36_Outcome"] == outcome, "Pooled_Stage2_Probability"], bins=bins, alpha=0.62, color=color, label=label)
    ax.axvspan(0.40, 0.60, color=COLORS["expert"], alpha=0.18, label="Prespecified gray zone")
    ax.set(xlabel="Pooled expert Stage 2 probability", ylabel="Number of cases", xlim=(0, 1))
    ax.legend(frameon=False)
    ax.set_title("Expert gray-zone distribution, N=153")
    fig.tight_layout()
    save_figure(fig, "Supplementary Figure 23", SUPP)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0))
    display_models = ["Crossfitted_AI", "Pooled_Expert_Stage2", "Pooled_RuleC"]
    labels = ["Cross-fitted AI", "Expert Stage 2", "Rule C"]
    colors = [COLORS["ai"], COLORS["expert"], COLORS["rule"]]
    performance = pooled.set_index("Model").loc[display_models]
    axes[0].bar(labels, performance["AUC"], color=colors)
    axes[0].set(ylabel="AUC", ylim=(0.5, 0.8), title="Discrimination")
    axes[1].bar(np.arange(3) - 0.16, performance["Sensitivity"], width=0.32, color=colors, alpha=0.78, label="Sensitivity")
    axes[1].bar(np.arange(3) + 0.16, performance["Specificity"], width=0.32, color=colors, hatch="//", alpha=0.78, label="Specificity")
    axes[1].set_xticks(np.arange(3), labels)
    axes[1].set(ylabel="Proportion", ylim=(0, 1), title="Operating profile")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, "Supplementary Figure 24", SUPP)

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    x = np.arange(2)
    expert_row = performance.loc["Pooled_Expert_Stage2"]
    rule_row = performance.loc["Pooled_RuleC"]
    ax.bar(x - 0.18, [expert_row["FP"], expert_row["FN"]], width=0.36, color=COLORS["expert"], label="Expert Stage 2")
    ax.bar(x + 0.18, [rule_row["FP"], rule_row["FN"]], width=0.36, color=COLORS["rule"], label="Rule C")
    ax.set_xticks(x, ["False positives", "False negatives"])
    ax.set_ylabel("Number of classifications")
    ax.set_title("Paired error profile in the strict reader benchmark")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, "Supplementary Figure 25", SUPP)

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    for model, color, label in [("Crossfitted_AI", COLORS["ai"], "Cross-fitted AI"), ("Pooled_Expert_Stage2", COLORS["expert"], "Expert Stage 2"), ("Pooled_RuleC", COLORS["rule"], "Rule C"), ("Treat_all", COLORS["gray"], "Treat all")]:
        part = dca.loc[dca["Model"] == model]
        ax.plot(part["Threshold"], part["Net_Benefit"], color=color, linestyle="--" if model == "Treat_all" else "-", lw=1.6, label=label)
    ax.axhline(0, color="#AAAAAA", lw=0.7)
    ax.set(xlabel="Threshold probability", ylabel="Net benefit", xlim=(0.05, 0.70), ylim=(-0.10, 0.55))
    ax.legend(frameon=False)
    ax.set_title("Decision-curve analysis")
    fig.tight_layout()
    save_figure(fig, "Supplementary Figure 26", SUPP)

    fig, ax = plt.subplots(figsize=(6.4, 5.3))
    for outcome, color, label in [(0, COLORS["clinical"], "Non-event"), (1, COLORS["rule"], "Event")]:
        part = benchmark.loc[benchmark["Strict36_Outcome"] == outcome]
        ax.scatter(part["Pooled_Stage2_Probability"], part["AI_Probability"], s=28, alpha=0.72, color=color, edgecolor="white", linewidth=0.3, label=label)
    ax.axvline(0.5, color="#999999", linestyle="--", lw=0.8)
    ax.axhline(0.5, color="#999999", linestyle="--", lw=0.8)
    ax.plot([0, 1], [0, 1], color="#BBBBBB", lw=0.8)
    ax.set(xlabel="Pooled expert Stage 2 probability", ylabel="Cross-fitted AI probability", xlim=(0, 1), ylim=(0, 1))
    ax.legend(frameon=False)
    ax.set_title("AI-expert probability concordance")
    fig.tight_layout()
    save_figure(fig, "Supplementary Figure 27", SUPP)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.1))
    for phase, color in [("ADNI1", COLORS["clinical"]), ("ADNI3", COLORS["green"])]:
        part = independent.loc[independent["Baseline_PHASE"] == phase]
        fpr, tpr, _ = roc_curve(part["Strict36_Outcome"], part["clinical_plus_mri_probability"])
        axes[0].plot(fpr, tpr, color=color, lw=1.7, label=f"{phase}, AUC {auc(fpr, tpr):.3f}")
    axes[0].plot([0, 1], [0, 1], color="#999999", linestyle="--", lw=0.8)
    axes[0].set(xlabel="1 - Specificity", ylabel="Sensitivity", xlim=(0, 1), ylim=(0, 1), title="Phase-specific discrimination")
    axes[0].legend(frameon=False)
    plot_calibration(axes[1], independent["Strict36_Outcome"].astype(int).to_numpy(), [("Clinical + MRI", independent["clinical_plus_mri_probability"].to_numpy(), COLORS["ai"])])
    axes[1].set_title("Combined calibration, N=318")
    fig.tight_layout()
    save_figure(fig, "Supplementary Figure 28", SUPP)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.1))
    aibl_probability = aibl["Harmonized_Reduced_Model_Probability"].to_numpy()
    aibl_y = aibl["Strict36_Outcome"].astype(int).to_numpy()
    axes[0].boxplot([aibl_probability[aibl_y == 0], aibl_probability[aibl_y == 1]], tick_labels=["Non-event", "Event"], patch_artist=True, boxprops={"facecolor": "#D9E7EC"})
    axes[0].scatter(np.ones((aibl_y == 0).sum()), aibl_probability[aibl_y == 0], color=COLORS["clinical"], s=24, alpha=0.72)
    axes[0].scatter(np.full((aibl_y == 1).sum(), 2), aibl_probability[aibl_y == 1], color=COLORS["rule"], s=24, alpha=0.72)
    axes[0].set(ylabel="Predicted probability", ylim=(0, 1), title="AIBL strict 36-month risk distribution")
    thresholds = np.linspace(0.05, 0.75, 71)
    prevalence = aibl_y.mean()
    axes[1].plot(thresholds, [prevalence - (1 - prevalence) * t / (1 - t) for t in thresholds], color=COLORS["gray"], linestyle="--", label="Treat all")
    axes[1].axhline(0, color="#AAAAAA", lw=0.8, label="Treat none")
    net_benefit = []
    for threshold in thresholds:
        predicted = aibl_probability >= threshold
        tp = np.sum(predicted & (aibl_y == 1))
        fp = np.sum(predicted & (aibl_y == 0))
        net_benefit.append(tp / len(aibl_y) - fp / len(aibl_y) * threshold / (1 - threshold))
    axes[1].plot(thresholds, net_benefit, color=COLORS["ai"], lw=1.7, label="AIBL clinical proxy")
    axes[1].set(xlabel="Threshold probability", ylabel="Net benefit", xlim=(0.05, 0.70), ylim=(-0.25, 0.55), title="AIBL decision curve")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, "Supplementary Figure 29", SUPP)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    model_labels = {
        "ablation_clinical_mri": "Clinical + MRI ablation",
        "primary_clinical_csf_mri": "Clinical + CSF + MRI",
    }
    for index, metric in enumerate(["AUC", "Specificity"]):
        pivot = cross_perf.pivot(index="Analysis_Model", columns="Model", values=metric)
        pivot = pivot.rename(index=model_labels)
        pivot[["Pooled_expert_stage2", "Crossfitted_RuleC_simulation"]].plot(kind="bar", ax=axes[index], color=[COLORS["expert"], COLORS["rule"]], legend=index == 0)
        axes[index].set_ylabel(metric)
        axes[index].set_xlabel("")
        axes[index].tick_params(axis="x", rotation=0)
        axes[index].set_title("AUC" if metric == "AUC" else "Specificity")
        if index == 0:
            axes[index].legend(["Expert Stage 2", "Rule C"], frameon=False, fontsize=8)
        axes[index].set_ylim(0, 1)
    fig.tight_layout()
    save_figure(fig, "Supplementary Figure 30", SUPP)

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    display = subgroup.loc[subgroup["Subgroup"].isin(["all", "ADNI1", "ADNI3"])].copy()
    display["Label"] = display["Model"].map({"clinical_plus_mri": "Clinical + MRI", "primary_transportable_multimodal": "Clinical + CSF + MRI"}) + " | " + display["Subgroup"]
    display = display.sort_values("AUC")
    ax.errorbar(display["AUC"], np.arange(len(display)), xerr=np.vstack([display["AUC"] - display["AUC_CI_Lower"], display["AUC_CI_Upper"] - display["AUC"]]), fmt="o", color=COLORS["ai"], ecolor="#555555", capsize=3)
    ax.axvline(0.5, color="#888888", linestyle="--", lw=0.8)
    ax.set_yticks(np.arange(len(display)), display["Label"])
    ax.set_xlabel("AUC with 95% bootstrap CI")
    ax.set_title("Nonoverlapping ADNI phase-specific sensitivity")
    fig.tight_layout()
    save_figure(fig, "Supplementary Figure 31", SUPP)


def main() -> None:
    setup_style()
    figure1()
    figure2()
    figure3()
    figure5()
    supplementary_figures()
    print(f"Figures written to {FIG}")


if __name__ == "__main__":
    main()
