import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import mannwhitneyu, chi2_contingency
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

## Load Cohort A (CSF-based)
try:
    vae_a = pd.read_csv("Cohort_A_cluster_results.csv")
    csf_a = pd.read_csv("Cohort_A_metabolites.csv")
    
    vae_a['ID'] = vae_a['ID'].astype(str).str.strip()
    csf_a['ID'] = csf_a['ID'].astype(str).str.strip()
    vae_a = vae_a.rename(columns={'Cluster_Labels': 'Subtype'})
    
    data_a = vae_a[['ID', 'Subtype', 'AD_Conversion']].merge(csf_a, on='ID')
    
    ## Generate AT Score
    data_a['ABETA_RATIO_INV'] = -data_a['ABETA42_ABETA40_RATIO']
    at_features_a = ['ABETA_RATIO_INV', 'PTAU181']
    
    at_data_a = data_a[at_features_a].dropna(how='all')
    valid_idx_a = at_data_a.index
    
    for col in at_features_a:
        at_data_a[col] = at_data_a[col].fillna(at_data_a[col].median())
    
    scaler_a = StandardScaler()
    pca_a = PCA(n_components=1)
    at_score_a = pca_a.fit_transform(scaler_a.fit_transform(at_data_a))
    data_a.loc[valid_idx_a, 'AT_Score'] = at_score_a.flatten()
    
    ## Statistical tests
    data_a_valid = data_a[data_a['AT_Score'].notna()].copy()
    subtypes_a = sorted(data_a_valid['Subtype'].unique())
    groups_a = [data_a_valid[data_a_valid['Subtype']==s]['AT_Score'].values for s in subtypes_a]
    
    H_a, p_a = stats.kruskal(*groups_a)
    
    ## Subtype characteristics
    subtype_stats_a = []
    for st in subtypes_a:
        subset = data_a_valid[data_a_valid['Subtype']==st]
        conv = (subset['AD_Conversion']==1).sum()
        subtype_stats_a.append({
            'Subtype': st,
            'N': len(subset),
            'Converters': conv,
            'Conv_Rate': conv/len(subset)*100 if len(subset)>0 else 0,
            'AT_Mean': subset['AT_Score'].mean(),
            'AT_Std': subset['AT_Score'].std()
        })
    
    subtype_stats_a_df = pd.DataFrame(subtype_stats_a)
    
    ## Pairwise comparisons
    pairwise_a = []
    for s1, s2 in combinations(subtypes_a, 2):
        g1 = data_a_valid[data_a_valid['Subtype']==s1]['AT_Score']
        g2 = data_a_valid[data_a_valid['Subtype']==s2]['AT_Score']
        u, p = mannwhitneyu(g1, g2)
        
        pooled_std = np.sqrt(((len(g1)-1)*g1.std()**2 + (len(g2)-1)*g2.std()**2) / (len(g1)+len(g2)-2))
        cohens_d = (g1.mean() - g2.mean()) / pooled_std if pooled_std > 0 else 0
        
        pairwise_a.append({
            'Comparison': f'{s1} vs {s2}',
            'U': u,
            'p_value': p,
            'Cohens_d': cohens_d,
            'Significant': 'Yes' if p<0.05 else 'No'
        })
    
    pairwise_a_df = pd.DataFrame(pairwise_a)
    
    ## Chi-square test
    conv_table = pd.crosstab(data_a_valid['Subtype'], data_a_valid['AD_Conversion'])
    chi2, p_chi2, dof, expected = chi2_contingency(conv_table)
    
    ## AUC
    if len(data_a_valid['AD_Conversion'].unique()) > 1:
        auc_at = roc_auc_score(data_a_valid['AD_Conversion'], data_a_valid['AT_Score'])
    
    ## Save results
    data_a.to_csv("Cohort_A_Complete.csv", index=False)
    subtype_stats_a_df.to_csv("Cohort_A_Subtype_Stats.csv", index=False)
    pairwise_a_df.to_csv("Cohort_A_Pairwise.csv", index=False)
    
    cohort_a_success = True
    
except Exception as e:
    cohort_a_success = False

## Load Cohort B (PET-based)
try:
    vae_b = pd.read_csv("Cohort_B_cluster_results.csv")
    pet_b = pd.read_csv("Cohort_B_RNA_synovial.csv")
    
    vae_b['ID'] = vae_b['ID'].astype(str).str.strip()
    pet_b['ID'] = pet_b['ID'].astype(str).str.strip()
    vae_b = vae_b.rename(columns={'Cluster_Labels': 'Subtype'})
    
    data_b = vae_b[['ID', 'Subtype', 'AD_Conversion']].merge(pet_b, on='ID')
    
    ## Find PET biomarkers
    amyloid_cols = [c for c in data_b.columns if 'amyloid' in c.lower() and 'suvr' in c.lower()]
    tau_cols = [c for c in data_b.columns if 'tau' in c.lower() and 'suvr' in c.lower()]
    
    if amyloid_cols and tau_cols:
        amy_col = min(amyloid_cols, key=lambda c: data_b[c].isna().sum())
        tau_col = min(tau_cols, key=lambda c: data_b[c].isna().sum())
        
        ## Generate AT Score
        at_data_b = data_b[[amy_col, tau_col]].dropna(how='all')
        valid_idx_b = at_data_b.index
        
        for col in [amy_col, tau_col]:
            at_data_b[col] = at_data_b[col].fillna(at_data_b[col].median())
        
        scaler_b = StandardScaler()
        pca_b = PCA(n_components=1)
        at_score_b = pca_b.fit_transform(scaler_b.fit_transform(at_data_b))
        data_b.loc[valid_idx_b, 'AT_Score'] = at_score_b.flatten()
        
        ## Statistical tests
        data_b_valid = data_b[data_b['AT_Score'].notna()].copy()
        subtypes_b = sorted(data_b_valid['Subtype'].unique())
        groups_b = [data_b_valid[data_b_valid['Subtype']==s]['AT_Score'].values for s in subtypes_b]
        
        H_b, p_b = stats.kruskal(*groups_b)
        
        ## Subtype characteristics
        subtype_stats_b = []
        for st in subtypes_b:
            subset = data_b_valid[data_b_valid['Subtype']==st]
            subtype_stats_b.append({
                'Subtype': st,
                'N': len(subset),
                'AT_Mean': subset['AT_Score'].mean(),
                'AT_Std': subset['AT_Score'].std()
            })
        
        subtype_stats_b_df = pd.DataFrame(subtype_stats_b)
        
        ## Pairwise comparisons
        pairwise_b = []
        for s1, s2 in combinations(subtypes_b, 2):
            g1 = data_b_valid[data_b_valid['Subtype']==s1]['AT_Score']
            g2 = data_b_valid[data_b_valid['Subtype']==s2]['AT_Score']
            u, p = mannwhitneyu(g1, g2)
            
            pooled_std = np.sqrt(((len(g1)-1)*g1.std()**2 + (len(g2)-1)*g2.std()**2) / (len(g1)+len(g2)-2))
            cohens_d = (g1.mean() - g2.mean()) / pooled_std if pooled_std > 0 else 0
            
            pairwise_b.append({
                'Comparison': f'{s1} vs {s2}',
                'U': u,
                'p_value': p,
                'Cohens_d': cohens_d,
                'Significant': 'Yes' if p<0.05 else 'No'
            })
        
        pairwise_b_df = pd.DataFrame(pairwise_b)
        
        ## Save
        data_b.to_csv("Cohort_B_Complete.csv", index=False)
        subtype_stats_b_df.to_csv("Cohort_B_Subtype_Stats.csv", index=False)
        pairwise_b_df.to_csv("Cohort_B_Pairwise.csv", index=False)
        
        cohort_b_success = True
    else:
        cohort_b_success = False
        
except Exception as e:
    cohort_b_success = False

## Visualizations
if cohort_a_success:
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    data_a_valid.boxplot(column='AT_Score', by='Subtype', ax=ax1)
    ax1.set_title(f'AT Score by Subtype\n(p={p_a:.4f})')
    ax1.set_xlabel('Subtype')
    ax1.set_ylabel('CSF AT Score')
    plt.suptitle('')
    
    ax2 = fig.add_subplot(gs[0, 1])
    conv_rates = [subtype_stats_a_df[subtype_stats_a_df['Subtype']==s]['Conv_Rate'].values[0] for s in subtypes_a]
    ax2.bar(subtypes_a, conv_rates, color=['steelblue', 'coral', 'lightgreen'])
    ax2.set_xlabel('Subtype')
    ax2.set_ylabel('Conversion Rate (%)')
    ax2.set_title(f'Conversion Rate\n(p={p_chi2:.4f})')
    
    ax3 = fig.add_subplot(gs[0, 2])
    for st in subtypes_a:
        subset = data_a_valid[data_a_valid['Subtype']==st]
        ax3.scatter(subset['AT_Score'], subset['AD_Conversion'], label=f'Subtype {st}', alpha=0.6)
    ax3.set_xlabel('CSF AT Score')
    ax3.set_ylabel('Conversion')
    ax3.set_title(f'AT Score vs Conversion\n(AUC={auc_at:.3f})')
    ax3.legend()
    
    plt.savefig('Cohort_A_Comprehensive.png', dpi=300, bbox_inches='tight')
    plt.close()
