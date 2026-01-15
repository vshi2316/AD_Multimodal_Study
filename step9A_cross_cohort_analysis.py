import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import mannwhitneyu, chi2_contingency
from sklearn.metrics import adjusted_rand_score, silhouette_score
import warnings
warnings.filterwarnings("ignore")

print("="*70)
print("Cross-Cohort Cluster Validation")
print("="*70)

## Load data
cohort_a = pd.read_csv("Cohort_A_Integrated.csv")
vae_embeddings = pd.read_csv("latent_encoded.csv")

## Merge cluster assignments
cohort_a = cohort_a.merge(vae_embeddings[['ID', 'Cluster_Labels']], on='ID', how='inner')

print(f"\nCohort A samples: {len(cohort_a)}")
print(f"Number of clusters: {cohort_a['Cluster_Labels'].nunique()}")

## Cluster distribution
print("\nCluster distribution:")
for cluster in sorted(cohort_a['Cluster_Labels'].unique()):
    count = (cohort_a['Cluster_Labels'] == cluster).sum()
    print(f"  Cluster {cluster}: {count} samples ({count/len(cohort_a)*100:.1f}%)")

## Feature importance analysis
print("\n" + "="*70)
print("Feature Importance Analysis")
print("="*70)

## Select features for analysis
exclude_cols = ["ID", "Cohort", "AD_Conversion", "Time_to_Event", "Followup_Years", "Cluster_Labels"]
feature_cols = [col for col in cohort_a.columns if col not in exclude_cols]
features = cohort_a[feature_cols].select_dtypes(include=[np.number])

## Calculate between-cluster variance for each feature
n_clusters = cohort_a['Cluster_Labels'].nunique()
cluster_labels = cohort_a['Cluster_Labels'].values

feature_importance = []
for col in features.columns:
    values = features[col].dropna().values
    valid_idx = features[col].notna()
    valid_labels = cluster_labels[valid_idx]
    
    if len(values) > 0:
        # Calculate between-cluster variance
        cluster_means = [values[valid_labels == i].mean() 
                        for i in range(n_clusters) 
                        if (valid_labels == i).sum() > 0]
        overall_mean = values.mean()
        
        between_var = sum([
            (cluster_means[i] - overall_mean)**2 * (valid_labels == i).sum()
            for i in range(len(cluster_means))
        ])
        
        total_var = np.var(values) * len(values)
        variance_explained = between_var / total_var if total_var > 0 else 0
        
        feature_importance.append({
            "Feature": col,
            "Variance_Explained": variance_explained
        })

importance_df = pd.DataFrame(feature_importance)
importance_df = importance_df.sort_values("Variance_Explained", ascending=False)

print("\nTop 20 discriminative features:")
print(importance_df.head(20).to_string(index=False))

## Save results
importance_df.to_csv("Cross_Cohort_Feature_Importance.csv", index=False)

## Cluster stability analysis
print("\n" + "="*70)
print("Cluster Stability Analysis")
print("="*70)

## Bootstrap resampling
n_bootstrap = 100
latent_cols = [col for col in vae_embeddings.columns if col.startswith('Latent_')]
latent_features = vae_embeddings[latent_cols].values
original_labels = vae_embeddings['Cluster_Labels'].values

ari_scores = []

print(f"\nRunning {n_bootstrap} bootstrap iterations...")
for i in range(n_bootstrap):
    # Resample with replacement
    indices = np.random.choice(len(latent_features), size=len(latent_features), replace=True)
    resampled_latent = latent_features[indices]
    
    # Recluster
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=i, n_init=50)
    resampled_labels = kmeans.fit_predict(resampled_latent)
    
    # Calculate ARI
    ari = adjusted_rand_score(original_labels[indices], resampled_labels)
    ari_scores.append(ari)
    
    if (i + 1) % 20 == 0:
        print(f"  Completed {i+1}/{n_bootstrap} iterations")

mean_ari = np.mean(ari_scores)
std_ari = np.std(ari_scores)

print(f"\nBootstrap ARI (n={n_bootstrap}):")
print(f"  Mean: {mean_ari:.3f} ± {std_ari:.3f}")
print(f"  Interpretation: {'Stable' if mean_ari > 0.8 else 'Moderate' if mean_ari > 0.6 else 'Unstable'}")

## Silhouette analysis
overall_silhouette = silhouette_score(latent_features, original_labels)
print(f"\nOverall Silhouette Score: {overall_silhouette:.3f}")

## Save validation results
validation_results = pd.DataFrame({
    "Metric": [
        "Sample_Size",
        "Number_of_Clusters",
        "Mean_Bootstrap_ARI",
        "Std_Bootstrap_ARI",
        "Overall_Silhouette",
        "Top_Feature"
    ],
    "Value": [
        len(cohort_a),
        n_clusters,
        mean_ari,
        std_ari,
        overall_silhouette,
        importance_df.iloc[0]["Feature"]
    ]
})

validation_results.to_csv("Cross_Cohort_Validation_Results.csv", index=False)

## Conversion rate analysis (if applicable)
if 'AD_Conversion' in cohort_a.columns:
    print("\n" + "="*70)
    print("Conversion Rate Analysis")
    print("="*70)
    
    for cluster in sorted(cohort_a['Cluster_Labels'].unique()):
        subset = cohort_a[cohort_a['Cluster_Labels'] == cluster]
        n_conv = (subset['AD_Conversion'] == 1).sum()
        n_total = len(subset)
        conv_rate = n_conv / n_total * 100 if n_total > 0 else 0
        
        print(f"  Cluster {cluster}: {n_conv}/{n_total} converters ({conv_rate:.1f}%)")
    
    # Chi-square test
    contingency = pd.crosstab(cohort_a['Cluster_Labels'], cohort_a['AD_Conversion'])
    chi2, p_value, dof, expected = chi2_contingency(contingency)
    print(f"\n  Chi-square test: χ²={chi2:.3f}, p={p_value:.4f}")

print("\n" + "="*70)
print("Cross-cohort analysis complete!")
print("="*70)
print("\nOutputs saved:")
print("  - Cross_Cohort_Feature_Importance.csv")
print("  - Cross_Cohort_Validation_Results.csv")
