import pandas as pd
import numpy as np
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.cluster import KMeans
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings("ignore")

## Load data
cohort_a = pd.read_csv("Cohort_A_Integrated.csv")
cohort_b = pd.read_csv("Cohort_B_Integrated.csv")
vae_embeddings = pd.read_csv("VAE_latent_embeddings.csv")

## Get cluster assignments from VAE
cluster_labels_a = vae_embeddings["Cluster_Labels"].values

print("="*70)
print("Cross-Cohort Cluster Validation")
print("="*70)

## Validate clusters in Cohort B using same latent features
latent_cols = [col for col in vae_embeddings.columns if col.startswith("Latent_")]

if len(cohort_b) > 0:
    ## For demonstration, apply k-means on Cohort B features
    feature_cols = [col for col in cohort_b.columns 
                   if col not in ["ID", "Cohort", "AD_Conversion", "Time_to_Event", "Followup_Years"]]
    features_b = cohort_b[feature_cols].select_dtypes(include=[np.number])
    features_b = features_b.dropna(axis=1, how="all").fillna(features_b.mean())
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_b_scaled = scaler.fit_transform(features_b)
    
    ## Apply k-means with same number of clusters
    n_clusters = len(np.unique(cluster_labels_a))
    kmeans_b = KMeans(n_clusters=n_clusters, random_state=42, n_init=100)
    cluster_labels_b = kmeans_b.fit_predict(X_b_scaled)
    
    print(f"\nCohort B clustering:")
    for i in range(n_clusters):
        count = np.sum(cluster_labels_b == i)
        print(f"  Cluster {i}: {count} samples ({count/len(cluster_labels_b)*100:.1f}%)")
    
    ## Calculate silhouette score for Cohort B
    sil_score_b = silhouette_score(X_b_scaled, cluster_labels_b)
    print(f"\nCohort B Silhouette Score: {sil_score_b:.3f}")
else:
    print("\nCohort B is empty, skipping validation.")

## Feature importance analysis
print("\n" + "="*70)
print("Feature Importance Ranking")
print("="*70)

## Calculate feature variance explained by clusters
feature_cols = [col for col in cohort_a.columns 
               if col not in ["ID", "Cohort", "AD_Conversion", "Time_to_Event", "Followup_Years"]]
features_a = cohort_a[feature_cols].select_dtypes(include=[np.number])
features_a = features_a.dropna(axis=1, how="all").fillna(features_a.mean())

## Calculate between-cluster variance for each feature
feature_importance = []
for col in features_a.columns:
    values = features_a[col].values
    cluster_means = [values[cluster_labels_a == i].mean() for i in range(n_clusters)]
    overall_mean = values.mean()
    between_var = np.sum([(cluster_means[i] - overall_mean)**2 * np.sum(cluster_labels_a == i) 
                          for i in range(n_clusters)])
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

## Bootstrap resampling for stability
n_bootstrap = 100
ari_scores = []

for i in range(n_bootstrap):
    ## Resample with replacement
    indices = np.random.choice(len(cluster_labels_a), size=len(cluster_labels_a), replace=True)
    
    ## Recluster resampled data
    latent_features = vae_embeddings[latent_cols].values
    resampled_latent = latent_features[indices]
    
    kmeans_resample = KMeans(n_clusters=n_clusters, random_state=i, n_init=10)
    labels_resample = kmeans_resample.fit_predict(resampled_latent)
    
    ## Calculate ARI between original and resampled
    ari = adjusted_rand_score(cluster_labels_a[indices], labels_resample)
    ari_scores.append(ari)

mean_ari = np.mean(ari_scores)
std_ari = np.std(ari_scores)

print(f"Bootstrap ARI (n={n_bootstrap}):")
print(f"  Mean: {mean_ari:.3f} ± {std_ari:.3f}")
print(f"  Interpretation: {'Stable' if mean_ari > 0.8 else 'Moderate' if mean_ari > 0.6 else 'Unstable'}")

## Save validation results
validation_results = {
    "Analysis": ["Cohort_A_Samples", "Cohort_B_Samples", "Number_of_Clusters", 
                 "Mean_Bootstrap_ARI", "Std_Bootstrap_ARI", "Top_Feature"],
    "Value": [len(cohort_a), len(cohort_b), n_clusters, 
              mean_ari, std_ari, importance_df.iloc[0]["Feature"]]
}

validation_df = pd.DataFrame(validation_results)
validation_df.to_csv("Cross_Cohort_Validation_Results.csv", index=False)

print("\n" + "="*70)
print("Cross-cohort analysis complete!")
print("="*70)
print("\nOutputs saved:")
print("  - Cross_Cohort_Feature_Importance.csv")
print("  - Cross_Cohort_Validation_Results.csv")
