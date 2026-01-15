"""

import os
import argparse
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import mannwhitneyu, chi2_contingency, kruskal
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests  
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from itertools import combinations
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


def calculate_cohens_d(group1, group2):
    """
    Calculate Cohen's d effect size.
    
    Methods 2.8: "requiring |SMD| > 0.5 for clinical meaningfulness"
    
    Interpretation:
    - |d| < 0.2: Negligible
    - 0.2 <= |d| < 0.5: Small
    - 0.5 <= |d| < 0.8: Medium (clinically meaningful)
    - |d| >= 0.8: Large
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0
    
    cohens_d = (group1.mean() - group2.mean()) / pooled_std
    return cohens_d


def interpret_cohens_d(d):
    """Interpret Cohen's d effect size."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "Negligible"
    elif abs_d < 0.5:
        return "Small"
    elif abs_d < 0.8:
        return "Medium"
    else:
        return "Large"


def calculate_eta_squared(groups):
    """
    Calculate eta-squared for ANOVA-like effect size.
    
    Methods 2.8: "Eta-squared (η²) quantifies effect sizes: between-group sum 
    of squares divided by total sum of squares, with 0.01, 0.06, and 0.14 
    representing small, medium, and significant effects per Cohen's criteria"
    """
    all_values = np.concatenate(groups)
    grand_mean = all_values.mean()
    
    ss_total = np.sum((all_values - grand_mean) ** 2)
    
    ss_between = 0
    for group in groups:
        group_mean = group.mean()
        ss_between += len(group) * (group_mean - grand_mean) ** 2
    
    eta_squared = ss_between / ss_total if ss_total > 0 else 0
    return eta_squared


def interpret_eta_squared(eta_sq):
    """Interpret eta-squared effect size per Cohen's criteria."""
    if eta_sq < 0.01:
        return "Negligible"
    elif eta_sq < 0.06:
        return "Small"
    elif eta_sq < 0.14:
        return "Medium"
    else:
        return "Large"


def main():
    # ========== Parse Arguments ==========
    parser = argparse.ArgumentParser(
        description="Biomarker Validation with FDR Correction (Methods 2.4/2.8)"
    )
    parser.add_argument(
        "--cluster_file",
        type=str,
        required=True,
        help="Path to cluster_results.csv (from step8)"
    )
    parser.add_argument(
        "--integrated_file",
        type=str,
        required=True,
        help="Path to integrated cohort CSV file with biomarkers"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Output directory (default: ./results)"
    )
    parser.add_argument(
        "--ptau_col",
        type=str,
        default="PTAU181",
        help="Column name for p-tau181 (default: PTAU181)"
    )
    parser.add_argument(
        "--abeta_ratio_col",
        type=str,
        default="ABETA42_ABETA40_RATIO",
        help="Column name for Aβ42/Aβ40 ratio (default: ABETA42_ABETA40_RATIO)"
    )
    parser.add_argument(
        "--fdr_alpha",
        type=float,
        default=0.05,
        help="FDR significance threshold (default: 0.05, as per Methods 2.8)"
    )
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 90)
    print("Step 9B: Biomarker Validation with FDR Correction (Methods 2.4/2.8)".center(90))
    print("=" * 90)

    # ========== [1/7] Load Data ==========
    print("\n[1/7] Loading data...")
    
    if not os.path.exists(args.cluster_file):
        raise FileNotFoundError(f"Cluster file not found: {args.cluster_file}")
    if not os.path.exists(args.integrated_file):
        raise FileNotFoundError(f"Integrated file not found: {args.integrated_file}")
    
    # Load cluster results
    cluster_results = pd.read_csv(args.cluster_file)
    cluster_results['ID'] = cluster_results['ID'].astype(str).str.strip()
    cluster_results = cluster_results.rename(columns={'Cluster_Labels': 'Subtype'})
    
    # Load integrated data with biomarkers
    cohort_data = pd.read_csv(args.integrated_file)
    cohort_data['ID'] = cohort_data['ID'].astype(str).str.strip()
    
    # Merge
    merge_cols = ['ID', 'Subtype']
    if 'AD_Conversion' in cluster_results.columns:
        merge_cols.append('AD_Conversion')
    
    data = cluster_results[merge_cols].merge(
        cohort_data, on='ID', how='inner'
    )
    
    # Handle duplicate AD_Conversion columns
    if 'AD_Conversion_x' in data.columns:
        data['AD_Conversion'] = data['AD_Conversion_x']
        data = data.drop(columns=['AD_Conversion_x', 'AD_Conversion_y'], errors='ignore')
    
    print(f"  Total samples: {len(data)}")
    if 'AD_Conversion' in data.columns:
        n_conv = (data['AD_Conversion'] == 1).sum()
        print(f"  Converters: {n_conv} ({n_conv/len(data)*100:.1f}%)")
    
    # ========== [2/7] Check CSF Biomarkers ==========
    print("\n[2/7] Checking CSF biomarkers...")
    
    # Find columns case-insensitively
    col_map = {c.lower(): c for c in data.columns}
    
    ptau_col = args.ptau_col
    abeta_col = args.abeta_ratio_col
    
    if ptau_col.lower() in col_map:
        ptau_col = col_map[ptau_col.lower()]
    if abeta_col.lower() in col_map:
        abeta_col = col_map[abeta_col.lower()]
    
    if ptau_col not in data.columns:
        raise ValueError(f"p-tau181 column '{args.ptau_col}' not found in data")
    if abeta_col not in data.columns:
        raise ValueError(f"Aβ ratio column '{args.abeta_ratio_col}' not found in data")
    
    print(f"  ✓ {abeta_col}: {data[abeta_col].notna().sum()}/{len(data)} ({data[abeta_col].notna().sum()/len(data)*100:.1f}%)")
    print(f"  ✓ {ptau_col}: {data[ptau_col].notna().sum()}/{len(data)} ({data[ptau_col].notna().sum()/len(data)*100:.1f}%)")
    
    # ========== [3/7] Generate AT Score ==========
    print("\n[3/7] Generating AT Score (PCA-based composite)...")
    
    # Invert Aβ ratio (lower ratio = more pathology)
    data['ABETA_RATIO_INV'] = -data[abeta_col]
    
    # Select features for AT score
    at_features = ['ABETA_RATIO_INV', ptau_col]
    at_data = data[at_features].copy()
    valid_idx = at_data.dropna(how='all').index
    
    # Median imputation for missing values
    for col in at_features:
        at_data[col] = at_data[col].fillna(at_data[col].median())
    
    # PCA to generate composite AT score (Methods 2.4)
    scaler = StandardScaler()
    pca = PCA(n_components=1)
    at_score = pca.fit_transform(scaler.fit_transform(at_data.loc[valid_idx]))
    data.loc[valid_idx, 'AT_Score'] = at_score.flatten()
    
    print(f"  Valid samples: {data['AT_Score'].notna().sum()}/{len(data)}")
    print(f"  PCA variance explained: {pca.explained_variance_ratio_[0]*100:.1f}%")
    
    # ========== [4/7] Statistical Tests with FDR Correction ==========
    print("\n[4/7] Statistical Testing with Benjamini-Hochberg FDR Correction...")
    print(f"  FDR threshold: q < {args.fdr_alpha} (Methods 2.8)")
    
    data_valid = data[data['AT_Score'].notna()].copy()
    subtypes = sorted(data_valid['Subtype'].unique())
    n_subtypes = len(subtypes)
    
    # Global test: Kruskal-Wallis
    groups = [data_valid[data_valid['Subtype'] == s]['AT_Score'].values for s in subtypes]
    H, p_kw = kruskal(*groups)
    eta_sq = calculate_eta_squared(groups)
    
    print(f"\n  Global Test (Kruskal-Wallis):")
    print(f"    H = {H:.3f}, p = {p_kw:.4e}")
    print(f"    η² = {eta_sq:.4f} ({interpret_eta_squared(eta_sq)} effect)")
    if p_kw < 0.05:
        print("    ✓ SIGNIFICANT global difference")
    else:
        print("    ⚠ No significant global difference")

    # Pairwise comparisons with FDR correction (Methods 2.8)
    print(f"\n  Pairwise Comparisons (Mann-Whitney U + FDR):")
    
    pairwise_results = []
    p_values_raw = []
    
    for s1, s2 in combinations(subtypes, 2):
        g1 = data_valid[data_valid['Subtype'] == s1]['AT_Score']
        g2 = data_valid[data_valid['Subtype'] == s2]['AT_Score']
        
        # Mann-Whitney U test
        u_stat, p_raw = mannwhitneyu(g1, g2, alternative='two-sided')
        
        # Cohen's d effect size
        cohens_d = calculate_cohens_d(g1, g2)
        
        pairwise_results.append({
            'Comparison': f'{s1} vs {s2}',
            'N1': len(g1),
            'N2': len(g2),
            'Mean1': g1.mean(),
            'Mean2': g2.mean(),
            'U_statistic': u_stat,
            'p_raw': p_raw,
            'Cohens_d': cohens_d,
            'Effect_Size': interpret_cohens_d(cohens_d),
            'Clinically_Meaningful': 'Yes' if abs(cohens_d) > 0.5 else 'No'
        })
        p_values_raw.append(p_raw)
    
    # Apply Benjamini-Hochberg FDR correction (CRITICAL - Methods 2.8)
    reject, p_adj, _, _ = multipletests(p_values_raw, alpha=args.fdr_alpha, method='fdr_bh')
    
    # Update results with FDR-corrected p-values
    for i, res in enumerate(pairwise_results):
        res['p_adj_FDR'] = p_adj[i]
        res['Significant_FDR'] = 'Yes' if reject[i] else 'No'
        
        sig_marker = '✓' if reject[i] else ''
        clinical_marker = '(clinically meaningful)' if res['Clinically_Meaningful'] == 'Yes' else ''
        print(f"    {res['Comparison']}: p_raw={res['p_raw']:.4f}, q={p_adj[i]:.4f}, "
              f"d={res['Cohens_d']:.2f} {sig_marker} {clinical_marker}")
    
    pairwise_df = pd.DataFrame(pairwise_results)
    
    # ========== [5/7] Subtype Characterization ==========
    print("\n[5/7] Characterizing subtypes...")
    
    subtype_stats = []
    for st in subtypes:
        subset = data_valid[data_valid['Subtype'] == st]
        at_mean = subset['AT_Score'].mean()
        at_std = subset['AT_Score'].std()
        at_median = subset['AT_Score'].median()
        
        stats_dict = {
            'Subtype': st,
            'N': len(subset),
            'AT_Mean': at_mean,
            'AT_Std': at_std,
            'AT_Median': at_median,
            'PTAU181_Mean': subset[ptau_col].mean() if ptau_col in subset.columns else np.nan,
            'ABETA_Ratio_Mean': subset[abeta_col].mean() if abeta_col in subset.columns else np.nan
        }
        
        if 'AD_Conversion' in subset.columns:
            conv = (subset['AD_Conversion'] == 1).sum()
            conv_rate = conv / len(subset) * 100 if len(subset) > 0 else 0
            stats_dict['Converters'] = conv
            stats_dict['Conv_Rate_Pct'] = conv_rate
            print(f"  Subtype {st}: N={len(subset)}, Conv={conv} ({conv_rate:.1f}%), "
                  f"AT={at_mean:.2f}±{at_std:.2f}")
        else:
            print(f"  Subtype {st}: N={len(subset)}, AT={at_mean:.2f}±{at_std:.2f}")
        
        subtype_stats.append(stats_dict)
    
    subtype_stats_df = pd.DataFrame(subtype_stats)
    
    # ========== [6/7] Conversion & Prognostic Analysis ==========
    chi2 = None
    p_chi2 = None
    auc_at = None
    
    if 'AD_Conversion' in data_valid.columns:
        print("\n[6/7] Conversion Rate & Prognostic Analysis...")
        
        # Chi-square test
        conv_table = pd.crosstab(data_valid['Subtype'], data_valid['AD_Conversion'])
        chi2, p_chi2, dof, expected = chi2_contingency(conv_table)
        print(f"  Chi-square test: χ²={chi2:.3f}, p={p_chi2:.4f}")
        
        if p_chi2 < 0.05:
            print("  ✓ Significant association between subtypes and conversion")
        else:
            print("  ⚠ No significant association detected")
        
        # AUC for prognostic value (Methods 2.5)
        if data_valid['AD_Conversion'].nunique() > 1:
            auc_at = roc_auc_score(data_valid['AD_Conversion'], data_valid['AT_Score'])
            print(f"\n  Prognostic Value:")
            print(f"    AT Score AUC for conversion: {auc_at:.3f}")
            if auc_at >= 0.7:
                print("    ✓ Good discriminative ability")
            elif auc_at >= 0.6:
                print("    Moderate discriminative ability")
            else:
                print("    ⚠ Poor discriminative ability")
        else:
            print("\n  ⚠ Cannot calculate AUC (insufficient outcome variation)")
    else:
        print("\n[6/7] Skipping conversion analysis (no outcome data)")

    # ========== [7/7] Save Results & Visualizations ==========
    print("\n[7/7] Saving results and generating visualizations...")
    
    # Save complete data
    complete_path = os.path.join(args.output_dir, "Cohort_Complete.csv")
    data.to_csv(complete_path, index=False)
    print(f"  ✓ Saved: {complete_path}")
    
    # Save subtype stats
    stats_path = os.path.join(args.output_dir, "Subtype_Stats.csv")
    subtype_stats_df.to_csv(stats_path, index=False)
    print(f"  ✓ Saved: {stats_path}")
    
    # Save pairwise comparisons with FDR
    pairwise_path = os.path.join(args.output_dir, "Pairwise_Comparisons_FDR.csv")
    pairwise_df.to_csv(pairwise_path, index=False)
    print(f"  ✓ Saved: {pairwise_path}")
    
    # ========== Comprehensive Visualization ==========
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # 1. AT Score boxplot with significance
    ax1 = fig.add_subplot(gs[0, 0])
    bp = data_valid.boxplot(column='AT_Score', by='Subtype', ax=ax1, patch_artist=True,
                            return_type='dict')
    colors = plt.cm.Set2(np.linspace(0, 1, n_subtypes))
    for patch, color in zip(bp['AT_Score']['boxes'], colors):
        patch.set_facecolor(color)
    ax1.set_title(f'AT Score by Subtype\n(Kruskal-Wallis p={p_kw:.4e}, η²={eta_sq:.3f})', fontsize=11)
    ax1.set_xlabel('Subtype')
    ax1.set_ylabel('CSF AT Score')
    plt.suptitle('')
    
    # 2. Conversion rate (if available)
    ax2 = fig.add_subplot(gs[0, 1])
    if 'Conv_Rate_Pct' in subtype_stats_df.columns:
        conv_rates = subtype_stats_df['Conv_Rate_Pct'].values
        colors_list = plt.cm.Set2(np.linspace(0, 1, n_subtypes))
        bars = ax2.bar(subtypes, conv_rates, color=colors_list)
        ax2.set_xlabel('Subtype')
        ax2.set_ylabel('Conversion Rate (%)')
        title_str = 'MCI→AD Conversion Rate'
        if p_chi2 is not None:
            title_str += f'\n(χ²={chi2:.2f}, p={p_chi2:.4f})'
        ax2.set_title(title_str)
        ax2.set_ylim(0, max(conv_rates) * 1.2 if max(conv_rates) > 0 else 100)
        for bar, rate in zip(bars, conv_rates):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                     f'{rate:.1f}%', ha='center', va='bottom', fontsize=9)
    else:
        ax2.text(0.5, 0.5, 'No conversion data', ha='center', va='center', 
                 transform=ax2.transAxes, fontsize=12)
        ax2.set_title('Conversion Rate (N/A)')
    
    # 3. Effect size heatmap for pairwise comparisons
    ax3 = fig.add_subplot(gs[0, 2])
    n_comparisons = len(pairwise_df)
    effect_matrix = np.zeros((n_subtypes, n_subtypes))
    sig_matrix = np.zeros((n_subtypes, n_subtypes))
    
    for _, row in pairwise_df.iterrows():
        s1, s2 = row['Comparison'].split(' vs ')
        s1_idx = subtypes.index(int(s1))
        s2_idx = subtypes.index(int(s2))
        effect_matrix[s1_idx, s2_idx] = row['Cohens_d']
        effect_matrix[s2_idx, s1_idx] = -row['Cohens_d']
        if row['Significant_FDR'] == 'Yes':
            sig_matrix[s1_idx, s2_idx] = 1
            sig_matrix[s2_idx, s1_idx] = 1
    
    mask = np.triu(np.ones_like(effect_matrix, dtype=bool), k=1)
    sns.heatmap(effect_matrix, mask=~mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, vmin=-2, vmax=2, ax=ax3,
                xticklabels=subtypes, yticklabels=subtypes,
                cbar_kws={'label': "Cohen's d"})
    ax3.set_title("Pairwise Effect Sizes (Cohen's d)\n* = FDR-significant")
    
    # Add significance markers
    for i in range(n_subtypes):
        for j in range(i+1, n_subtypes):
            if sig_matrix[i, j] == 1:
                ax3.text(j + 0.5, i + 0.8, '*', ha='center', va='center', 
                        fontsize=14, fontweight='bold', color='black')
    
    # 4. Biomarker profile heatmap
    ax4 = fig.add_subplot(gs[1, 0])
    biomarker_cols = [abeta_col, ptau_col]
    biomarker_means = data_valid.groupby('Subtype')[biomarker_cols].mean()
    biomarker_std = data_valid.groupby('Subtype')[biomarker_cols].std()
    
    # Z-score normalize for visualization
    biomarker_z = (biomarker_means - biomarker_means.mean()) / biomarker_means.std()
    sns.heatmap(biomarker_z.T, annot=biomarker_means.T.round(2), fmt='.2f', 
                cmap='RdYlBu_r', ax=ax4, center=0,
                cbar_kws={'label': 'Z-score'})
    ax4.set_title('Biomarker Profile by Subtype\n(values shown, colors = z-scores)')
    ax4.set_xlabel('Subtype')
    ax4.set_ylabel('Biomarker')
    
    # 5. Violin plot with individual points
    ax5 = fig.add_subplot(gs[1, 1])
    sns.violinplot(data=data_valid, x='Subtype', y='AT_Score', ax=ax5, 
                   inner='box', palette='Set2')
    ax5.set_title('AT Score Distribution by Subtype')
    ax5.set_xlabel('Subtype')
    ax5.set_ylabel('CSF AT Score')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Sample composition with conversion
    ax6 = fig.add_subplot(gs[1, 2])
    if 'Converters' in subtype_stats_df.columns:
        n_conv = subtype_stats_df['Converters'].values
        n_stable = subtype_stats_df['N'].values - n_conv
        x = np.arange(n_subtypes)
        width = 0.6
        ax6.bar(x, n_stable, width, label='Stable', color='skyblue')
        ax6.bar(x, n_conv, width, bottom=n_stable, label='Converter', color='salmon')
        ax6.set_xlabel('Subtype')
        ax6.set_ylabel('Number of Samples')
        ax6.set_title('Sample Composition by Outcome')
        ax6.set_xticks(x)
        ax6.set_xticklabels(subtypes)
        ax6.legend()
        
        # Add total counts
        for i, (s, c) in enumerate(zip(n_stable, n_conv)):
            ax6.text(i, s + c + 2, f'n={s+c}', ha='center', fontsize=9)
    else:
        n_samples = subtype_stats_df['N'].values
        ax6.bar(subtypes, n_samples, color='steelblue')
        ax6.set_xlabel('Subtype')
        ax6.set_ylabel('Number of Samples')
        ax6.set_title('Sample Distribution')
    
    fig_path = os.path.join(args.output_dir, 'Biomarker_Validation.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {fig_path}")
    
    # ========== Summary Report ==========
    print("\n" + "=" * 90)
    print("Step 9B: Biomarker Validation Complete!".center(90))
    print("=" * 90)
    print("\nOutputs:")
    print(f"  - {complete_path}")
    print(f"  - {stats_path}")
    print(f"  - {pairwise_path}")
    print(f"  - {fig_path}")
    print("\nKey Findings:")
    print(f"  - Sample size: {len(data_valid)}")
    print(f"  - Subtypes: {n_subtypes}")
    print(f"  - Kruskal-Wallis: p={p_kw:.4e}, η²={eta_sq:.4f} ({interpret_eta_squared(eta_sq)})")
    
    # FDR-corrected significant comparisons
    n_sig_fdr = (pairwise_df['Significant_FDR'] == 'Yes').sum()
    n_clinical = (pairwise_df['Clinically_Meaningful'] == 'Yes').sum()
    print(f"  - FDR-significant pairwise comparisons: {n_sig_fdr}/{len(pairwise_df)}")
    print(f"  - Clinically meaningful (|d|>0.5): {n_clinical}/{len(pairwise_df)}")
    
    if p_chi2 is not None:
        print(f"  - Chi-square (conversion): p={p_chi2:.4f} {'(significant)' if p_chi2<0.05 else ''}")
    if auc_at is not None:
        print(f"  - AT Score AUC: {auc_at:.3f}")
    
    print("\nMethods 2.8 Compliance:")
    print(f"  ✓ Benjamini-Hochberg FDR correction applied (q < {args.fdr_alpha})")
    print(f"  ✓ Cohen's d effect sizes calculated")
    print(f"  ✓ Clinical meaningfulness threshold (|SMD| > 0.5) applied")
    print(f"  ✓ Eta-squared effect size for global test")
    print("=" * 90)


if __name__ == "__main__":
    main()
