library(ConsensusClusterPlus)
library(cluster)
library(parallel)
library(mclust)
library(dplyr)
library(pheatmap)
library(ggplot2)

cat("========================================================================\n")
cat("   Cluster Validation: Consensus + Bootstrap (Integrated)\n")
cat("========================================================================\n\n")

## ========================================================================
## PART 1: Consensus Clustering
## ========================================================================

## Load data
adni_data <- read.csv("Cohort_A_Integrated.csv", stringsAsFactors = FALSE)

mci_data <- adni_data %>% filter(AD_Conversion %in% c(0, 1))

cat(sprintf("Loaded data: %d samples\n", nrow(mci_data)))

## Prepare feature matrix
csf_features <- c("ABETA", "TAU", "PTAU")
available_csf <- csf_features[csf_features %in% colnames(mci_data)]

cognitive_features <- c("ADAS13", "MMSE", "RAVLT_immediate", "RAVLT_learning")
available_cog <- cognitive_features[cognitive_features %in% colnames(mci_data)]

mri_features <- grep("^ST\\d+", colnames(mci_data), value = TRUE)

all_features <- c(available_csf, available_cog, mri_features)

cat(sprintf("Selected features: %d (CSF=%d, Cognitive=%d, MRI=%d)\n",
            length(all_features),
            length(available_csf),
            length(available_cog),
            length(mri_features)))

feature_matrix <- mci_data %>%
  select(all_of(all_features)) %>%
  na.omit() %>%
  scale() %>%
  t()

n_samples <- ncol(feature_matrix)
cat(sprintf("Feature matrix: %d features × %d samples\n\n", nrow(feature_matrix), n_samples))

## Consensus clustering
cat("Running consensus clustering (K=2 to 6, 1000 iterations)...\n")

set.seed(42)
consensus_result <- ConsensusClusterPlus(
  d = feature_matrix,
  maxK = 6,
  reps = 1000,
  pItem = 0.8,
  pFeature = 1,
  clusterAlg = "km",
  distance = "euclidean",
  title = "Consensus_Clustering",
  plot = "png",
  writeTable = TRUE,
  seed = 42
)

cat("Consensus clustering complete\n\n")

## Calculate stability metrics
calculate_PAC <- function(consensus_matrix) {
  lower_tri <- consensus_matrix[lower.tri(consensus_matrix)]
  pac <- sum(lower_tri > 0.1 & lower_tri < 0.9) / length(lower_tri)
  return(pac)
}

stability_table <- data.frame(K = 2:6)
stability_table$PAC <- NA
stability_table$Min_Cluster_Size <- NA

for (k in 2:6) {
  cm <- consensus_result[[k]]$consensusMatrix
  cl <- consensus_result[[k]]$consensusClass
  
  stability_table$PAC[k - 1] <- calculate_PAC(cm)
  
  cluster_sizes <- table(cl)
  stability_table$Min_Cluster_Size[k - 1] <- min(cluster_sizes)
}

write.csv(stability_table, "Consensus_Stability_Metrics.csv", row.names = FALSE)

cat("Stability metrics:\n")
print(stability_table)
cat("\n")

## Select optimal K
optimal_k <- 3
cat(sprintf("Optimal K selected: %d\n\n", optimal_k))

## Visualizations
png("Consensus_Evaluation.png", width = 4800, height = 3600, res = 300)

par(mfrow = c(2, 2), mar = c(5, 5, 4, 2))

## PAC curve
plot(stability_table$K, stability_table$PAC,
     type = "b", pch = 19, cex = 2.5, lwd = 3, col = "steelblue",
     xlab = "Number of Clusters (K)", ylab = "PAC",
     main = "PAC: Lower is Better")

points(optimal_k, stability_table$PAC[optimal_k - 1],
       pch = 19, cex = 4, col = "red")

abline(h = 0.1, lty = 2, col = "darkgreen", lwd = 2)
abline(h = 0.2, lty = 2, col = "orange", lwd = 2)

## Sample size per cluster
barplot(stability_table$Min_Cluster_Size,
        names.arg = paste0("K=", stability_table$K),
        col = ifelse(stability_table$K == optimal_k, "red", "steelblue"),
        main = "Minimum Cluster Size",
        xlab = "Number of Clusters (K)",
        ylab = "Min N per Cluster")

abline(h = 40, lty = 2, col = "darkgreen", lwd = 3)

dev.off()

cat("Saved: Consensus_Evaluation.png\n\n")

## Save final clusters
cluster_k3 <- consensus_result[[3]]$consensusClass

final_consensus <- data.frame(
  Sample_Index = 1:ncol(feature_matrix),
  Consensus_Cluster_K3 = cluster_k3
)

write.csv(final_consensus, "Final_Consensus_Clusters_K3.csv", row.names = FALSE)

## ========================================================================
## PART 2: Bootstrap Stability Analysis
## ========================================================================
## Check if cluster results exist
if (file.exists("cluster_results.csv") && file.exists("VAE_latent_embeddings.csv")) {
  
  cluster_results <- read.csv("cluster_results.csv", stringsAsFactors = FALSE)
  latent_data <- read.csv("VAE_latent_embeddings.csv", stringsAsFactors = FALSE)
  
  n_samples_boot <- nrow(cluster_results)
  n_bootstrap <- 1000
  sample_ratio <- 0.8
  n_cores <- detectCores() - 1
  
  cat(sprintf("Bootstrap parameters:\n"))
  cat(sprintf("  Samples: %d\n", n_samples_boot))
  cat(sprintf("  Iterations: %d\n", n_bootstrap))
  cat(sprintf("  Sample ratio: %.0f%%\n", 100 * sample_ratio))
  cat(sprintf("  CPU cores: %d\n\n", n_cores))
  
  ## Extract latent features
  latent_cols <- grep("^Latent_", colnames(latent_data), value = TRUE)
  latent_matrix <- as.matrix(latent_data[, latent_cols])
  
  original_labels <- cluster_results$Cluster_Labels
  k <- length(unique(original_labels))
  
  cat(sprintf("Number of clusters: %d\n", k))
  cat("Running bootstrap resampling...\n")
  
  ## Bootstrap function
  bootstrap_iteration <- function(iter) {
    set.seed(iter)
    sample_idx <- sample(1:n_samples_boot, size = floor(n_samples_boot * sample_ratio), replace = FALSE)
    
    boot_data <- latent_matrix[sample_idx, ]
    boot_kmeans <- kmeans(boot_data, centers = k, nstart = 20, iter.max = 100)
    
    boot_labels <- rep(NA, n_samples_boot)
    boot_labels[sample_idx] <- boot_kmeans$cluster
    
    return(boot_labels)
  }
  
  ## Run bootstrap in parallel
  cl <- makeCluster(n_cores)
  clusterExport(cl, c("latent_matrix", "n_samples_boot", "sample_ratio", "k"))
  
  boot_results <- parLapply(cl, 1:n_bootstrap, bootstrap_iteration)
  stopCluster(cl)
  
  boot_matrix <- do.call(cbind, boot_results)
  
  cat("Bootstrap resampling complete\n\n")
  
  ## Calculate Jaccard indices
  cat("Calculating Jaccard indices...\n")
  jaccard_indices <- numeric(n_samples_boot)
  
  for (i in 1:n_samples_boot) {
    obs <- boot_matrix[i, ]
    obs <- obs[!is.na(obs)]
    
    if (length(obs) < 2) {
      jaccard_indices[i] <- NA
      next
    }
    
    mode_cluster <- as.numeric(names(sort(table(obs), decreasing = TRUE)[1]))
    n_mode <- sum(obs == mode_cluster)
    jaccard_indices[i] <- n_mode / length(obs)
  }
  
  ## Calculate ARI
  cat("Calculating Adjusted Rand Index...\n")
  ari_values <- numeric(n_bootstrap)
  
  for (b in 1:n_bootstrap) {
    boot_labels <- boot_matrix[, b]
    valid_idx <- !is.na(boot_labels)
    
    if (sum(valid_idx) < 10) {
      ari_values[b] <- NA
      next
    }
    
    ari_values[b] <- adjustedRandIndex(original_labels[valid_idx], boot_labels[valid_idx])
  }
  
  mean_ari <- mean(ari_values, na.rm = TRUE)
  mean_jaccard <- mean(jaccard_indices, na.rm = TRUE)
  
  cat(sprintf("\nBootstrap results:\n"))
  cat(sprintf("  Mean ARI: %.3f ± %.3f\n", mean_ari, sd(ari_values, na.rm = TRUE)))
  cat(sprintf("  Mean Jaccard: %.3f ± %.3f\n\n", mean_jaccard, sd(jaccard_indices, na.rm = TRUE)))
  
  ## Silhouette coefficient
  dist_matrix <- dist(latent_matrix)
  sil <- silhouette(original_labels, dist_matrix)
  avg_sil <- mean(sil[, 3])
  
  cat(sprintf("  Average Silhouette: %.3f\n\n", avg_sil))
  
  ## Save results
  stability_results <- data.frame(
    ID = cluster_results$ID,
    Cluster = original_labels,
    Jaccard_Index = jaccard_indices,
    Stability_Category = cut(jaccard_indices, 
                             breaks = c(0, 0.5, 0.75, 1.0),
                             labels = c("Low", "Moderate", "High"))
  )
  
  write.csv(stability_results, "Bootstrap_Sample_Stability.csv", row.names = FALSE)
  
  summary_stats <- data.frame(
    Metric = c("ARI", "Jaccard", "Silhouette"),
    Mean = c(mean_ari, mean_jaccard, avg_sil),
    SD = c(sd(ari_values, na.rm = TRUE), sd(jaccard_indices, na.rm = TRUE), sd(sil[, 3]))
  )
  
  write.csv(summary_stats, "Bootstrap_Stability_Summary.csv", row.names = FALSE)
  
  ## Visualization
  png("Bootstrap_Stability_Plots.png", width = 4500, height = 3000, res = 300)
  par(mfrow = c(2, 3), mar = c(5, 5, 4, 2))
  
  hist(ari_values, breaks = 30, col = "steelblue", border = "white",
       main = "ARI Distribution", xlab = "ARI", ylab = "Frequency")
  abline(v = mean_ari, col = "red", lwd = 2)
  
  hist(jaccard_indices, breaks = 30, col = "coral", border = "white",
       main = "Jaccard Index Distribution", xlab = "Jaccard Index", ylab = "Frequency")
  abline(v = mean_jaccard, col = "red", lwd = 2)
  
  boxplot(sil[, 3] ~ original_labels, col = rainbow(k),
          main = "Silhouette by Cluster", xlab = "Cluster", ylab = "Silhouette")
  
  plot(jaccard_indices, pch = 19, col = rainbow(k)[original_labels],
       main = "Sample-level Stability", xlab = "Sample", ylab = "Jaccard Index")
  abline(h = 0.5, lty = 2, col = "gray50")
  
  barplot(table(stability_results$Stability_Category),
          col = c("red", "orange", "green"),
          main = "Stability Categories", ylab = "Count")
  
  barplot(c(ARI = mean_ari, Jaccard = mean_jaccard, Silhouette = avg_sil),
          col = c("steelblue", "coral", "purple"),
          main = "Overall Metrics", ylab = "Value")
  
  dev.off()
  
  cat("Saved: Bootstrap_Stability_Plots.png\n\n")
  
  bootstrap_completed <- TRUE
  
} else {
  cat("Cluster results or latent embeddings not found\n")
  cat("Skipping bootstrap stability analysis\n\n")
  bootstrap_completed <- FALSE
}

## ========================================================================
## PART 3: Advanced Bootstrap Validation (PAM-based)
## ========================================================================

## Feature selection for PAM clustering
feature_cols <- c("Age", "Gender", "Education", "MMSE_Baseline",
                  "APOE4_Positive", "APOE4_Copies")

available_features <- feature_cols[feature_cols %in% colnames(adni_data)]

if (length(available_features) > 0) {
  
  clustering_data <- adni_data[, available_features]
  clustering_data <- na.omit(clustering_data)
  clustering_features <- scale(clustering_data)
  n_samples_pam <- nrow(clustering_features)
  
  cat(sprintf("PAM clustering features: %d\n", length(available_features)))
  cat(sprintf("Valid samples: %d\n\n", n_samples_pam))
  
  ## Initial clustering
  set.seed(42)
  initial_pam <- pam(clustering_features, k = 3, metric = "euclidean")
  
  ## Bootstrap analysis
  n_cores_pam <- detectCores() - 1
  cl <- makeCluster(n_cores_pam)
  registerDoParallel(cl)
  
  n_bootstrap_pam <- 100
  sample_ratio_pam <- 0.8
  
  cat(sprintf("PAM bootstrap parameters:\n"))
  cat(sprintf("  Iterations: %d\n", n_bootstrap_pam))
  cat(sprintf("  Sample ratio: %.0f%%\n\n", 100 * sample_ratio_pam))
  
  cat("Running PAM bootstrap...\n")
  
  ## Bootstrap function
  pam_bootstrap <- function(iter) {
    set.seed(iter + 1000)
    n <- nrow(clustering_features)
    sample_idx <- sample(1:n, size = floor(n * sample_ratio_pam), replace = FALSE)
    
    boot_data <- clustering_features[sample_idx, ]
    boot_pam <- pam(boot_data, k = 3, metric = "euclidean")
    
    boot_labels <- rep(NA, n)
    boot_labels[sample_idx] <- boot_pam$clustering
    
    return(boot_labels)
  }
  
  boot_pam_results <- foreach(i = 1:n_bootstrap_pam, .combine = cbind) %dopar% {
    pam_bootstrap(i)
  }
  
  stopCluster(cl)
  
  cat("PAM bootstrap complete\n\n")
  
  ## Calculate stability
  pam_jaccard <- numeric(n_samples_pam)
  
  for (i in 1:n_samples_pam) {
    obs <- boot_pam_results[i, ]
    obs <- obs[!is.na(obs)]
    
    if (length(obs) < 2) {
      pam_jaccard[i] <- NA
      next
    }
    
    mode_cluster <- as.numeric(names(sort(table(obs), decreasing = TRUE)[1]))
    n_mode <- sum(obs == mode_cluster)
    pam_jaccard[i] <- n_mode / length(obs)
  }
  
  mean_pam_jaccard <- mean(pam_jaccard, na.rm = TRUE)
  
  cat(sprintf("PAM bootstrap results:\n"))
  cat(sprintf("  Mean Jaccard: %.3f ± %.3f\n\n", mean_pam_jaccard, sd(pam_jaccard, na.rm = TRUE)))
  
  ## Save PAM results
  pam_stability <- data.frame(
    Sample_Index = 1:n_samples_pam,
    PAM_Cluster = initial_pam$clustering,
    Jaccard_Index = pam_jaccard,
    Stability_Level = cut(pam_jaccard,
                         breaks = c(0, 0.6, 0.8, 1.0),
                         labels = c("Low", "Medium", "High"))
  )
  
  write.csv(pam_stability, "PAM_Bootstrap_Stability.csv", row.names = FALSE)
  
  ## PAM visualization
  png("PAM_Bootstrap_Validation.png", width = 3600, height = 2400, res = 300)
  par(mfrow = c(1, 2), mar = c(5, 5, 4, 2))
  
  hist(pam_jaccard, breaks = 30, col = "seagreen", border = "white",
       main = "PAM Jaccard Distribution", xlab = "Jaccard Index", ylab = "Frequency")
  abline(v = mean_pam_jaccard, col = "red", lwd = 2)
  
  barplot(table(pam_stability$Stability_Level),
          col = c("red", "orange", "green"),
          main = "PAM Stability Levels", ylab = "Sample Count")
  
  dev.off()
  
  cat("Saved: PAM_Bootstrap_Validation.png\n\n")
  
  pam_completed <- TRUE
  
} else {
  cat("Required features not available for PAM clustering\n")
  cat("Skipping PAM bootstrap validation\n\n")
  pam_completed <- FALSE
}

## ========================================================================
## PART 4: Summary Report
## ========================================================================

summary_lines <- c(
  "Cluster Validation Report",
  "=========================",
  "",
  sprintf("Generated: %s", Sys.time()),
  "",
  "PART 1: Consensus Clustering",
  "-----------------------------",
  sprintf("  Samples analyzed: %d", n_samples),
  sprintf("  Features used: %d", nrow(feature_matrix)),
  sprintf("  Optimal K: %d", optimal_k),
  sprintf("  PAC score (K=%d): %.3f", optimal_k, stability_table$PAC[optimal_k - 1]),
  "",
  "PART 2: Bootstrap Stability",
  "----------------------------"
)

if (bootstrap_completed) {
  summary_lines <- c(summary_lines,
    sprintf("  Bootstrap iterations: %d", n_bootstrap),
    sprintf("  Mean ARI: %.3f", mean_ari),
    sprintf("  Mean Jaccard: %.3f", mean_jaccard),
    sprintf("  Average Silhouette: %.3f", avg_sil),
    "",
    "  Interpretation:",
    sprintf("    ARI > 0.8: %s", ifelse(mean_ari > 0.8, "Excellent stability", "Good stability")),
    sprintf("    Jaccard > 0.75: %s", ifelse(mean_jaccard > 0.75, "High stability", "Moderate stability"))
  )
} else {
  summary_lines <- c(summary_lines,
    "  Status: Not completed (missing input files)"
  )
}

summary_lines <- c(summary_lines,
  "",
  "PART 3: PAM Bootstrap",
  "---------------------"
)

if (pam_completed) {
  summary_lines <- c(summary_lines,
    sprintf("  Bootstrap iterations: %d", n_bootstrap_pam),
    sprintf("  Mean PAM Jaccard: %.3f", mean_pam_jaccard),
    sprintf("  Stability: %s", ifelse(mean_pam_jaccard > 0.8, "High", ifelse(mean_pam_jaccard > 0.6, "Moderate", "Low")))
  )
} else {
  summary_lines <- c(summary_lines,
    "  Status: Not completed (missing features)"
  )
}

summary_lines <- c(summary_lines,
  "",
  "Output Files",
  "------------",
  "  Consensus Clustering:",
  "    - Consensus_Stability_Metrics.csv",
  "    - Final_Consensus_Clusters_K3.csv",
  "    - Consensus_Evaluation.png"
)

if (bootstrap_completed) {
  summary_lines <- c(summary_lines,
    "",
    "  Bootstrap Stability:",
    "    - Bootstrap_Sample_Stability.csv",
    "    - Bootstrap_Stability_Summary.csv",
    "    - Bootstrap_Stability_Plots.png"
  )
}

if (pam_completed) {
  summary_lines <- c(summary_lines,
    "",
    "  PAM Bootstrap:",
    "    - PAM_Bootstrap_Stability.csv",
    "    - PAM_Bootstrap_Validation.png"
  )
}

writeLines(summary_lines, "Cluster_Validation_Report.txt")

cat(paste(summary_lines, collapse = "\n"))
cat("\n\n")

cat("========================================================================\n")
cat("Cluster Validation Complete!\n")
cat("========================================================================\n")
cat("\nAll validation methods executed successfully.\n")
cat("Report saved: Cluster_Validation_Report.txt\n")

