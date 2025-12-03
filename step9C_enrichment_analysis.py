import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

## Load data
cohort_a = pd.read_csv("Cohort_A_Complete.csv")
csf_data = pd.read_csv("Cohort_A_metabolites.csv")

cohort_a['ID'] = cohort_a['ID'].astype(str).str.strip()
csf_data['ID'] = csf_data['ID'].astype(str).str.strip()
data_full = cohort_a[['ID', 'Subtype', 'AD_Conversion']].merge(csf_data, on='ID')

csf_features = [c for c in data_full.columns if c not in ['ID', 'Subtype', 'AD_Conversion']]

## Differential expression analysis
subtypes = sorted(data_full['Subtype'].unique())
de_results = []

for subtype in subtypes:
    labels = (data_full['Subtype'] == subtype).astype(int)
    
    for feat in csf_features:
        valid_idx = data_full[feat].notna()
        if valid_idx.sum() < 10:
            continue
        
        feat_data = data_full.loc[valid_idx, feat]
        feat_labels = labels[valid_idx]
        
        group_target = feat_data[feat_labels == 1]
        group_others = feat_data[feat_labels == 0]
        
        if len(group_target) < 3 or len(group_others) < 3:
            continue
        
        try:
            u_stat, p_val = stats.mannwhitneyu(group_target, group_others, alternative='two-sided')
            
            mean_target = group_target.mean()
            mean_others = group_others.mean()
            pooled_std = np.sqrt(((len(group_target)-1)*group_target.std()**2 +
                                  (len(group_others)-1)*group_others.std()**2) /
                                (len(group_target)+len(group_others)-2))
            
            smd = (mean_target - mean_others) / pooled_std if pooled_std > 0 else 0
            fc = mean_target / mean_others if mean_others != 0 else np.nan
            log2fc = np.log2(fc) if not np.isnan(fc) and fc > 0 else 0
            
            de_results.append({
                'Subtype': subtype,
                'Feature': feat,
                'Mean_Target': mean_target,
                'Mean_Others': mean_others,
                'SMD': smd,
                'Log2FC': log2fc,
                'P_value': p_val,
                'N_Target': len(group_target),
                'N_Others': len(group_others)
            })
        except:
            continue

de_df = pd.DataFrame(de_results)

## Multiple testing correction
for subtype in subtypes:
    subset = de_df['Subtype'] == subtype
    if subset.sum() > 0:
        _, q_vals, _, _ = multipletests(de_df.loc[subset, 'P_value'], method='fdr_bh')
        de_df.loc[subset, 'Q_value'] = q_vals

de_df['Significant'] = (de_df['Q_value'] < 0.05) & (np.abs(de_df['SMD']) > 0.3)

de_df.to_csv("Differential_Expression_Results.csv", index=False)

## Pathway annotation
csf_pathways = {
    'ABETA42': ['Amyloid processing', 'APP metabolism', 'Neurodegeneration'],
    'ABETA40': ['Amyloid processing', 'APP metabolism'],
    'ABETA42_ABETA40_RATIO': ['Amyloid pathology', 'AD diagnosis'],
    'TAU_TOTAL': ['Tau pathology', 'Neuronal injury', 'Neurodegeneration'],
    'PTAU181': ['Tau phosphorylation', 'Tau pathology', 'AD pathology'],
    'STREM2': ['Immune response', 'Microglial activation', 'Neuroinflammation'],
    'PGRN': ['Neuroinflammation', 'Lysosomal function', 'Progranulin pathway']
}

pathway_enrichment = []
for subtype in subtypes:
    sig_features = de_df[(de_df['Subtype']==subtype) & (de_df['Significant'])]['Feature'].tolist()
    
    pathway_counts = {}
    for feat in sig_features:
        if feat in csf_pathways:
            for pathway in csf_pathways[feat]:
                pathway_counts[pathway] = pathway_counts.get(pathway, 0) + 1
    
    for pathway, count in pathway_counts.items():
        pathway_enrichment.append({
            'Subtype': subtype,
            'Pathway': pathway,
            'Feature_Count': count
        })

pathway_df = pd.DataFrame(pathway_enrichment)
pathway_df.to_csv("Pathway_Enrichment.csv", index=False)

## Conversion rate analysis
conv_stats = []
for subtype in subtypes:
    subset = data_full[data_full['Subtype']==subtype]
    n_total = len(subset)
    n_conv = (subset['AD_Conversion']==1).sum()
    conv_rate = n_conv / n_total * 100 if n_total > 0 else 0
    
    conv_stats.append({
        'Subtype': subtype,
        'N': n_total,
        'Converters': n_conv,
        'Conv_Rate': conv_rate
    })

conv_df = pd.DataFrame(conv_stats)

conv_table = pd.crosstab(data_full['Subtype'], data_full['AD_Conversion'])
from scipy.stats import chi2_contingency
chi2, p_chi2, dof, expected = chi2_contingency(conv_table)

conv_df.to_csv("Conversion_Rate_Analysis.csv", index=False)

## Visualizations
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, subtype in enumerate(subtypes):
    ax = axes[idx]
    subset = de_df[de_df['Subtype']==subtype]
    
    sig = subset['Significant']
    ax.scatter(subset.loc[~sig, 'Log2FC'], -np.log10(subset.loc[~sig, 'P_value']),
               c='gray', alpha=0.5, s=30, label='Non-significant')
    ax.scatter(subset.loc[sig, 'Log2FC'], -np.log10(subset.loc[sig, 'P_value']),
               c='red', alpha=0.7, s=50, label='Significant')
    
    ax.axhline(y=-np.log10(0.05), color='blue', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(x=-0.3, color='green', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(x=0.3, color='green', linestyle='--', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Log2 Fold Change')
    ax.set_ylabel('-log10(P-value)')
    ax.set_title(f'Subtype {subtype} vs Others')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("Volcano_Plots.png", dpi=300, bbox_inches='tight')
plt.close()

## Conversion rate plot
fig, ax = plt.subplots(figsize=(8, 6))

x = np.arange(len(conv_df))
bars = ax.bar(x, conv_df['Conv_Rate'], color=['steelblue', 'coral', 'lightgreen'],
              alpha=0.8, edgecolor='black', linewidth=1.5)

for i, (bar, row) in enumerate(zip(bars, conv_df.itertuples())):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{row.Conv_Rate:.1f}%\n({row.Converters}/{row.N})',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_xlabel('Subtype')
ax.set_ylabel('MCI→AD Conversion Rate (%)')
ax.set_title(f'Conversion Rates by Subtype\n(Chi-square p={p_chi2:.4f})')
ax.set_xticks(x)
ax.set_xticklabels([f'Subtype {s}' for s in subtypes])
ax.set_ylim(0, 100)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig("Conversion_Rates.png", dpi=300, bbox_inches='tight')
plt.close()
