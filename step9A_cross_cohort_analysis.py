"""

import os
import argparse
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")


def consensus_clustering_pac(X, k, n_iter=100, subsample_frac=0.8, random_state=42):
    """
    Implements Consensus Clustering and PAC calculation as per.
    
    "Consensus matrices were visualized using hierarchical methods. 
    Proportion of ambiguous clustering (PAC) indices assessed clustering 
    number optimality (PAC<0.05 indicating clear structure)"
    
    Parameters:
    -----------
    X : np.ndarray
        Latent features (N x D)
    k : int
        Number of clusters
    n_iter : int
        Number of bootstrap iterations
    subsample_frac : float
        Fraction of samples to use in each iteration
    random_state : int
        Random seed
        
    Returns:
    --------
    consensus_matrix : np.ndarray
        N x N consensus matrix
    pac : float
        Proportion of Ambiguous Clustering
    sample_stability : np.ndarray
        Per-sample stability scores
    """
    N = X.shape[0]
    consensus_matrix = np.zeros((N, N))
    count_matrix = np.zeros((N, N))
    sample_cluster_counts = np.zeros((N, k))
    
    np.random.seed(random_state)
    print(f"    Running Consensus Clustering (K={k}, {n_iter} iterations)...")
    
    for i in range(n_iter):
        # Subsample
        n_sub = int(N * subsample_frac)
        indices = np.random.choice(N, n_sub, replace=False)
        X_sub = X[indices]
        
        # Cluster
        km = KMeans(n_clusters=k, n_init=10, random_state=i)
        labels_sub = km.fit_predict(X_sub)
        
        # Update consensus matrix
        for ii, idx1 in enumerate(indices):
            for jj, idx2 in enumerate(indices):
                count_matrix[idx1, idx2] += 1
                if labels_sub[ii] == labels_sub[jj]:
                    consensus_matrix[idx1, idx2] += 1
            # Track cluster assignments for sample stability
            sample_cluster_counts[idx1, labels_sub[ii]] += 1
        
        if (i + 1) % 20 == 0:
            print(f"      Completed {i+1}/{n_iter} iterations")
    
    # Normalize consensus matrix
    with np.errstate(divide='ignore', invalid='ignore'):
        consensus_matrix = np.divide(consensus_matrix, count_matrix)
        consensus_matrix[np.isnan(consensus_matrix)] = 0
    
    # Calculate PAC (Proportion of Ambiguous Clustering)
    # Ambiguous range: (0.1, 0.9) as per standard definition
    flat_matrix = consensus_matrix[np.triu_indices(N, k=1)]  # Upper triangle only
    ambiguous_vals = flat_matrix[(flat_matrix > 0.1) & (flat_matrix < 0.9)]
    pac = len(ambiguous_vals) / len(flat_matrix) if len(flat_matrix) > 0 else 0
    
    # Calculate sample-level stability (>0.85 indicating stable assignment)
    sample_stability = np.max(sample_cluster_counts, axis=1) / np.sum(sample_cluster_counts, axis=1)
    sample_stability[np.isnan(sample_stability)] = 0
    
    return consensus_matrix, pac, sample_stability


def calculate_jaccard_index(labels_true, labels_pred):
    """
    Calculates Jaccard Index by aligning clusters using Hungarian algorithm.
    
     "Metrics included Jaccard index measuring overlap between 
    original and bootstrap clustering (range: 0-1, higher indicating greater stability)"
    
    Parameters:
    -----------
    labels_true : np.ndarray
        Original cluster labels
    labels_pred : np.ndarray
        Predicted cluster labels from bootstrap
        
    Returns:
    --------
    jaccard : float
        Mean Jaccard index across aligned clusters
    """
    # Get unique labels
    unique_true = np.unique(labels_true)
    unique_pred = np.unique(labels_pred)
    n_true = len(unique_true)
    n_pred = len(unique_pred)
    
    # Build confusion matrix
    D = max(n_true, n_pred)
    w = np.zeros((D, D), dtype=np.float64)
    
    for i, t_label in enumerate(unique_true):
        for j, p_label in enumerate(unique_pred):
            intersection = np.sum((labels_true == t_label) & (labels_pred == p_label))
            w[i, j] = intersection
    
    # Hungarian algorithm to find best matching
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    
    # Calculate Jaccard for each matched pair
    jaccards = []
    for t_idx, p_idx in zip(row_ind, col_ind):
        if t_idx < n_true and p_idx < n_pred:
            t_label = unique_true[t_idx]
            p_label = unique_pred[p_idx]
            intersection = np.sum((labels_true == t_label) & (labels_pred == p_label))
            union = np.sum((labels_true == t_label) | (labels_pred == p_label))
            jaccards.append(intersection / union if union > 0 else 0)
    
    return np.mean(jaccards) if jaccards else 0


def calculate_eta_squared(values, labels, n_clusters):
    """
    Calculate eta-squared (effect size) for feature importance.
    
     "Eta-squared (η²) quantifies effect sizes: between-group sum 
    of squares divided by total sum of squares"
    """
    grand_mean = values.mean()
    ss_total = np.sum((values - grand_mean) ** 2)
    
    ss_between = 0
    for k in range(n_clusters):
        mask = labels == k
        if mask.sum() > 0:
            group_mean = values[mask].mean()
            ss_between += mask.sum() * (group_mean - grand_mean) ** 2
    
    eta_squared = ss_between / ss_total if ss_total > 0 else 0
    return eta_squared

def main():
    # ========== Parse Arguments ==========
    parser = argparse.ArgumentParser(
        description="Cross-Cohort Cluster Validation"
    )
    parser.add_argument(
        "--integrated_file",
        type=str,
        required=True,
        help="Path to integrated cohort CSV file"
    )
    parser.add_argument(
        "--latent_file",
        type=str,
        required=True,
        help="Path to latent_encoded.csv"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Output directory (default: ./results)"
    )
    parser.add_argument(
        "--n_bootstrap",
        type=int,
        default=100,
        help="Number of bootstrap iterations (default: 100)"
    )
    parser.add_argument(
        "--n_consensus",
        type=int,
        default=100,
        help="Number of consensus clustering iterations (default: 100)"
    )
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("Step 9A: Aligned Stability Assessment".center(80))
    print("(PAC, Consensus Clustering, Jaccard Index, ARI)".center(80))
    print("=" * 80)
    
    # ========== [1/6] Load Data ==========
    print("\n[1/6] Loading data...")
    
    cohort_a = pd.read_csv(args.integrated_file)
    vae_embeddings = pd.read_csv(args.latent_file)
    
    # Standardize ID column
    cohort_a['ID'] = cohort_a['ID'].astype(str).str.strip()
    vae_embeddings['ID'] = vae_embeddings['ID'].astype(str).str.strip()
    
    # Merge cluster assignments
    cohort_a = cohort_a.merge(
        vae_embeddings[['ID', 'Cluster_Labels']], 
        on='ID', 
        how='inner'
    )
    
    n_samples = len(cohort_a)
    n_clusters = cohort_a['Cluster_Labels'].nunique()
    
    print(f"  Samples: {n_samples}")
    print(f"  Clusters: {n_clusters}")
    
    # Cluster distribution
    print("\n  Cluster distribution:")
    for cluster in sorted(cohort_a['Cluster_Labels'].unique()):
        count = (cohort_a['Cluster_Labels'] == cluster).sum()
        print(f"    Cluster {cluster}: {count} samples ({count/n_samples*100:.1f}%)")
    
    # ========== [2/6] Feature Importance (Eta-squared) ==========
    print("\n[2/6] Feature Importance Analysis (Eta-squared)...")
    
    exclude_cols = ["ID", "Cohort", "AD_Conversion", "Time_to_Event", 
                    "Followup_Years", "Cluster_Labels"]
    feature_cols = [col for col in cohort_a.columns if col not in exclude_cols]
    features = cohort_a[feature_cols].select_dtypes(include=[np.number])
    
    cluster_labels = cohort_a['Cluster_Labels'].values
    feature_importance = []
    
    for col in features.columns:
        values = features[col].fillna(features[col].median()).values
        eta_sq = calculate_eta_squared(values, cluster_labels, n_clusters)
        feature_importance.append({
            "Feature": col,
            "Eta_Squared": eta_sq,
            "Effect_Size": "Large" if eta_sq >= 0.14 else ("Medium" if eta_sq >= 0.06 else "Small")
        })
    
    importance_df = pd.DataFrame(feature_importance)
    importance_df = importance_df.sort_values("Eta_Squared", ascending=False)
    
    print("\n  Top 15 discriminative features (by η²):")
    print(importance_df.head(15).to_string(index=False))
    
    importance_path = os.path.join(args.output_dir, "Cross_Cohort_Feature_Importance.csv")
    importance_df.to_csv(importance_path, index=False)
    print(f"\n  ✓ Saved: {importance_path}")

    # ========== [3/6] Bootstrap Stability (ARI & Jaccard) ==========
    print(f"\n[3/6] Bootstrap Stability Assessment ({args.n_bootstrap} iterations)...")
    print("  Computing ARI and Jaccard Index ...")
    
    latent_cols = [col for col in vae_embeddings.columns if col.startswith('Latent_')]
    X = vae_embeddings[latent_cols].values
    original_labels = vae_embeddings['Cluster_Labels'].values
    
    ari_scores = []
    jaccard_scores = []
    
    np.random.seed(42)
    for i in range(args.n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n_samples, n_samples, replace=True)
        X_boot = X[indices]
        labels_true_boot = original_labels[indices]
        
        # Recluster
        km = KMeans(n_clusters=n_clusters, random_state=i, n_init=20)
        labels_pred_boot = km.fit_predict(X_boot)
        
        # Calculate metrics
        ari = adjusted_rand_score(labels_true_boot, labels_pred_boot)
        jaccard = calculate_jaccard_index(labels_true_boot, labels_pred_boot)
        
        ari_scores.append(ari)
        jaccard_scores.append(jaccard)
        
        if (i + 1) % 20 == 0:
            print(f"    Completed {i+1}/{args.n_bootstrap} iterations")
    
    mean_ari = np.mean(ari_scores)
    std_ari = np.std(ari_scores)
    mean_jaccard = np.mean(jaccard_scores)
    std_jaccard = np.std(jaccard_scores)
    
    print(f"\n  Bootstrap Results:")
    print(f"    ARI: {mean_ari:.3f} ± {std_ari:.3f}")
    print(f"    Jaccard Index: {mean_jaccard:.3f} ± {std_jaccard:.3f}")
    
    # Interpret stability 
    if mean_ari > 0.8:
        ari_interpretation = "Highly Stable"
    elif mean_ari > 0.6:
        ari_interpretation = "Moderately Stable"
    else:
        ari_interpretation = "Unstable"
    print(f"    Interpretation: {ari_interpretation}")
    
    # ========== [4/6] Consensus Clustering & PAC ==========
    print(f"\n[4/6] Consensus Clustering & PAC Assessment ({args.n_consensus} iterations)...")
    
    consensus_matrix, pac, sample_stability = consensus_clustering_pac(
        X, n_clusters, n_iter=args.n_consensus, subsample_frac=0.8, random_state=42
    )
    
    # Sample-level stability assessment ( >0.85 indicating stable)
    stable_samples = np.sum(sample_stability > 0.85)
    stable_fraction = stable_samples / n_samples
    
    print(f"\n  Consensus Clustering Results:")
    print(f"    PAC Score: {pac:.4f}", end="")
    if pac < 0.05:
        print(" ✓ (PAC < 0.05 indicates clear cluster structure)")
    elif pac < 0.10:
        print(" (Acceptable)")
    else:
        print(" ⚠ (High ambiguity)")
    
    print(f"    Stable samples (>0.85): {stable_samples}/{n_samples} ({stable_fraction*100:.1f}%)")
    
    # Silhouette score
    overall_silhouette = silhouette_score(X, original_labels)
    print(f"    Silhouette Score: {overall_silhouette:.3f}")
    
    # ========== [5/6] Conversion Rate Analysis ==========
    chi2 = None
    p_chi2 = None
    
    if 'AD_Conversion' in cohort_a.columns:
        print("\n[5/6] Conversion Rate Analysis...")
        
        for cluster in sorted(cohort_a['Cluster_Labels'].unique()):
            subset = cohort_a[cohort_a['Cluster_Labels'] == cluster]
            n_conv = (subset['AD_Conversion'] == 1).sum()
            n_total = len(subset)
            conv_rate = n_conv / n_total * 100 if n_total > 0 else 0
            print(f"    Cluster {cluster}: {n_conv}/{n_total} converters ({conv_rate:.1f}%)")
        
        # Chi-square test
        contingency = pd.crosstab(cohort_a['Cluster_Labels'], cohort_a['AD_Conversion'])
        chi2, p_chi2, dof, expected = chi2_contingency(contingency)
        print(f"\n  Chi-square test: χ²={chi2:.3f}, p={p_chi2:.4f}")
        
        if p_chi2 < 0.05:
            print("  ✓ Significant association between clusters and conversion")
        else:
            print("  ⚠ No significant association detected")
    else:
        print("\n[5/6] Skipping conversion rate analysis (no outcome data)")

    # ========== [6/6] Save Results & Visualizations ==========
    print("\n[6/6] Saving results and generating visualizations...")
    
    # Save validation results
    validation_results = pd.DataFrame({
        "Metric": [
            "Sample_Size",
            "Number_of_Clusters",
            "Mean_Bootstrap_ARI",
            "Std_Bootstrap_ARI",
            "Mean_Bootstrap_Jaccard",
            "Std_Bootstrap_Jaccard",
            "PAC_Score",
            "Stable_Samples_Fraction",
            "Overall_Silhouette",
            "Top_Feature",
            "Chi_Square",
            "Chi_Square_P_Value"
        ],
        "Value": [
            n_samples,
            n_clusters,
            f"{mean_ari:.4f}",
            f"{std_ari:.4f}",
            f"{mean_jaccard:.4f}",
            f"{std_jaccard:.4f}",
            f"{pac:.4f}",
            f"{stable_fraction:.4f}",
            f"{overall_silhouette:.4f}",
            importance_df.iloc[0]["Feature"] if len(importance_df) > 0 else "N/A",
            f"{chi2:.4f}" if chi2 is not None else "N/A",
            f"{p_chi2:.4f}" if p_chi2 is not None else "N/A"
        ]
    })
    
    validation_path = os.path.join(args.output_dir, "Cross_Cohort_Validation_Results.csv")
    validation_results.to_csv(validation_path, index=False)
    print(f"  ✓ Saved: {validation_path}")
    
    # Generate Consensus Matrix Heatmap
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Consensus Matrix
    ax1 = axes[0]
    sns.heatmap(consensus_matrix, cmap='viridis', vmin=0, vmax=1, ax=ax1,
                cbar_kws={'label': 'Co-clustering Probability'})
    ax1.set_title(f'Consensus Matrix (K={n_clusters})\nPAC={pac:.4f}', fontsize=12)
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Sample Index')
    
    # Bootstrap Stability Distribution
    ax2 = axes[1]
    ax2.hist(ari_scores, bins=20, alpha=0.7, label=f'ARI (μ={mean_ari:.3f})', color='steelblue')
    ax2.hist(jaccard_scores, bins=20, alpha=0.7, label=f'Jaccard (μ={mean_jaccard:.3f})', color='coral')
    ax2.axvline(x=0.8, color='green', linestyle='--', label='Stability threshold (0.8)')
    ax2.set_xlabel('Score')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'Bootstrap Stability Distribution (n={args.n_bootstrap})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    consensus_fig_path = os.path.join(args.output_dir, "Consensus_Matrix.png")
    plt.savefig(consensus_fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {consensus_fig_path}")
    
    # Sample Stability Distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(sample_stability, bins=30, color='teal', alpha=0.7, edgecolor='black')
    ax.axvline(x=0.85, color='red', linestyle='--', linewidth=2, 
               label=f'Stability threshold (0.85)\n{stable_fraction*100:.1f}% samples stable')
    ax.set_xlabel('Sample Stability Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Sample-Level Clustering Stability 
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    stability_fig_path = os.path.join(args.output_dir, "Sample_Stability_Distribution.png")
    plt.savefig(stability_fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {stability_fig_path}")
    
    # ========== Summary ==========
    print("\n" + "=" * 80)
    print("Step 9A: Cross-Cohort Validation Complete!".center(80))
    print("=" * 80)
    print("\nOutputs:")
    print(f"  - {importance_path}")
    print(f"  - {validation_path}")
    print(f"  - {consensus_fig_path}")
    print(f"  - {stability_fig_path}")
    print("\nKey Findings:")
    print(f"  - Bootstrap ARI: {mean_ari:.3f} ± {std_ari:.3f} ({ari_interpretation})")
    print(f"  - Bootstrap Jaccard: {mean_jaccard:.3f} ± {std_jaccard:.3f}")
    print(f"  - PAC Score: {pac:.4f} {'(Clear structure)' if pac < 0.05 else ''}")
    print(f"  - Stable samples (>0.85): {stable_fraction*100:.1f}%")
    print(f"  - Silhouette Score: {overall_silhouette:.3f}")
    if len(importance_df) > 0:
        print(f"  - Top discriminative feature: {importance_df.iloc[0]['Feature']} (η²={importance_df.iloc[0]['Eta_Squared']:.3f})")
    print("=" * 80)


if __name__ == "__main__":
    main()


