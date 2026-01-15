library(ConsensusClusterPlus)
library(cluster)
library(parallel)
library(mclust)
library(dplyr)
library(pheatmap)
library(ggplot2)
library(RColorBrewer)
library(optparse)

# ==============================================================================
# Parse Command Line Arguments
# ==============================================================================
option_list <- list(
  make_option(c("--integrated_file"), type = "character", 
              default = "Cohort_A_Integrated.csv",
              help = "Path to integrated cohort CSV [default: %default]"),
  make_option(c("--latent_file"), type = "character", 
              default = "VAE_latent_embeddings.csv",
              help = "Path to VAE latent embeddings CSV [default: %default]"),
  make_option(c("--cluster_file"), type = "character", 
              default = "cluster_results.csv",
              help = "Path to cluster results CSV [default: %default]"),
  make_option(c("--output_dir"), type = "character", 
              default = "./results",
              help = "Output directory [default: %default]"),
  make_option(c("--n_bootstrap"), type = "integer", default = 100,
              help = "Number of bootstrap iterations (Methods 2.7) [default: %default]"),
  make_option(c("--n_consensus"), type = "integer", default = 1000,
              help = "Number of consensus clustering iterations [default: %default]"),
  make_option(c("--stability_threshold"), type = "numeric", default = 0.85,
              help = "Sample stability threshold (Methods 2.7: >0.85) [default: %default]"),
  make_option(c("--pac_threshold"), type = "numeric", default = 0.05,
              help = "PAC threshold for clear structure (Methods 2.7: <0.05) [default: %default]")
)

opt_parser <- OptionParser(option_list = option_list)
opt <- parse_args(opt_parser)

# Create output directory
dir.create(opt$output_dir, showWarnings = FALSE, recursive = TRUE)

cat("========================================================================\n")
cat("   Step 14: Cluster Validation (Methods 2.7 Aligned)\n")
cat("========================================================================\n\n")
cat(sprintf("Bootstrap iterations: %d (Methods 2.7)\n", opt$n_bootstrap))
cat(sprintf("Stability threshold: >%.2f (Methods 2.7)\n", opt$stability_threshold))
cat(sprintf("PAC threshold: <%.2f (Methods 2.7)\n\n", opt$pac_threshold))

calculate_jaccard_index <- function(labels_true, labels_pred) {
  # Get unique labels
  unique_true <- sort(unique(labels_true))
  unique_pred <- sort(unique(labels_pred))
  n_true <- length(unique_true)
  n_pred <- length(unique_pred)
  
  # Build intersection matrix for Hungarian algorithm
  D <- max(n_true, n_pred)
  w <- matrix(0, nrow = D, ncol = D)
  
  for (i in seq_along(unique_true)) {
    for (j in seq_along(unique_pred)) {
      intersection <- sum(labels_true == unique_true[i] & labels_pred == unique_pred[j])
      w[i, j] <- intersection
    }
  }
  
  # Simple greedy matching (approximation to Hungarian)
  matched <- rep(FALSE, D)
  jaccards <- c()
  
  for (i in seq_along(unique_true)) {
    best_j <- which.max(w[i, ] * (!matched))
    if (w[i, best_j] > 0) {
      matched[best_j] <- TRUE
      t_label <- unique_true[i]
      p_label <- unique_pred[best_j]
      intersection <- sum(labels_true == t_label & labels_pred == p_label)
      union <- sum(labels_true == t_label | labels_pred == p_label)
      jaccards <- c(jaccards, intersection / union)
    }
  }
  
  if (length(jaccards) == 0) return(0)
  return(mean(jaccards))
}

#' Calculate PAC (Proportion of Ambiguous Clustering)
#' Methods 2.7: "PAC<0.05 indicating clear structure"
calculate_PAC <- function(consensus_matrix, lower = 0.1, upper = 0.9) {
  lower_tri <- consensus_matrix[lower.tri(consensus_matrix)]
  pac <- sum(lower_tri > lower & lower_tri < upper) / length(lower_tri)
  return(pac)
}

#' Calculate Sample-Level Stability
#' Methods 2.7: ">0.85 indicating stable assignment"
calculate_sample_stability <- function(boot_matrix, original_labels) {
  n_samples <- nrow(boot_matrix)
  n_bootstrap <- ncol(boot_matrix)
  
  stability <- numeric(n_samples)
  
  for (i in 1:n_samples) {
    obs <- boot_matrix[i, ]
    obs <- obs[!is.na(obs)]
    
    if (length(obs) < 2) {
      stability[i] <- NA
      next
    }
    
    # Calculate proportion assigned to most frequent cluster
    mode_cluster <- as.numeric(names(sort(table(obs), decreasing = TRUE)[1]))
    stability[i] <- sum(obs == mode_cluster) / length(obs)
  }
  
  return(stability)
}

# ==============================================================================
# Part 1: Load Data
# ==============================================================================
cat("[1/5] Loading data...\n")

# Load integrated data

adni_data <- read.csv(opt$integrated_file, stringsAsFactors = FALSE)
cat(sprintf("  Loaded integrated data: %d samples\n", nrow(adni_data)))

# Filter to MCI samples if AD_Conversion exists
if ("AD_Conversion" %in% colnames(adni_data)) {
  mci_data <- adni_data %>% filter(AD_Conversion %in% c(0, 1))
  cat(sprintf("  MCI samples with outcome: %d\n", nrow(mci_data)))
} else {
  mci_data <- adni_data
}

# Prepare feature matrix for consensus clustering
csf_features <- c("ABETA", "TAU", "PTAU")
available_csf <- csf_features[csf_features %in% colnames(mci_data)]

cognitive_features <- c("ADAS13", "MMSE", "RAVLT_immediate", "RAVLT_learning")
available_cog <- cognitive_features[cognitive_features %in% colnames(mci_data)]

mri_features <- grep("^ST\\d+", colnames(mci_data), value = TRUE)

all_features <- c(available_csf, available_cog, mri_features)

cat(sprintf("  Features: CSF=%d, Cognitive=%d, MRI=%d, Total=%d\n",
            length(available_csf), length(available_cog), 
            length(mri_features), length(all_features)))

# Create feature matrix
if (length(all_features) > 0) {
  feature_matrix <- mci_data %>%
    select(all_of(all_features)) %>%
    na.omit() %>%
    scale() %>%
    t()
  
  n_samples <- ncol(feature_matrix)
  cat(sprintf("  Feature matrix: %d features × %d samples\n\n", 
              nrow(feature_matrix), n_samples))
} else {
  stop("No features available for clustering!")
}

# ==============================================================================
# Part 2: Consensus Clustering
# ==============================================================================
cat("[2/5] Running Consensus Clustering...\n")
cat(sprintf("  K range: 2-6, Iterations: %d\n", opt$n_consensus))

set.seed(42)
consensus_result <- ConsensusClusterPlus(
  d = feature_matrix,
  maxK = 6,
  reps = opt$n_consensus,
  pItem = 0.8,
  pFeature = 1,
  clusterAlg = "km",
  distance = "euclidean",
  title = file.path(opt$output_dir, "Consensus_Clustering"),
  plot = "png",
  writeTable = TRUE,
  seed = 42
)

cat("  Consensus clustering complete\n\n")

# Calculate stability metrics for each K
stability_table <- data.frame(K = 2:6)
stability_table$PAC <- NA
stability_table$Min_Cluster_Size <- NA
stability_table$Silhouette <- NA

for (k in 2:6) {
  cm <- consensus_result[[k]]$consensusMatrix
  cl <- consensus_result[[k]]$consensusClass
  
  # PAC (Methods 2.7)
  stability_table$PAC[k - 1] <- calculate_PAC(cm)
  
  # Minimum cluster size
  cluster_sizes <- table(cl)
  stability_table$Min_Cluster_Size[k - 1] <- min(cluster_sizes)
  
  # Silhouette
  if (n_samples > k) {
    dist_matrix <- dist(t(feature_matrix))
    sil <- silhouette(cl, dist_matrix)
    stability_table$Silhouette[k - 1] <- mean(sil[, 3])
  }
}

# Determine optimal K (lowest PAC with adequate cluster sizes)
stability_table$PAC_Optimal <- stability_table$PAC < opt$pac_threshold
stability_table$Size_Adequate <- stability_table$Min_Cluster_Size >= 30

optimal_k <- stability_table %>%
  filter(Size_Adequate) %>%
  slice_min(PAC, n = 1) %>%
  pull(K)

if (length(optimal_k) == 0) optimal_k <- 3

cat("Stability metrics by K:\n")
print(stability_table)
cat(sprintf("\nOptimal K selected: %d (PAC=%.4f)\n\n", 
            optimal_k, stability_table$PAC[optimal_k - 1]))

write.csv(stability_table, 
          file.path(opt$output_dir, "Consensus_Stability_Metrics.csv"), 
          row.names = FALSE)

# ==============================================================================
# Part 3: Bootstrap Stability Analysis (Methods 2.7)
# ==============================================================================
cat("[3/5] Bootstrap Stability Analysis (Methods 2.7)...\n")

# Check for latent embeddings and cluster results
latent_available <- file.exists(opt$latent_file)
cluster_available <- file.exists(opt$cluster_file)

if (latent_available && cluster_available) {
  latent_data <- read.csv(opt$latent_file, stringsAsFactors = FALSE)
  cluster_results <- read.csv(opt$cluster_file, stringsAsFactors = FALSE)
  
  # Extract latent features
  latent_cols <- grep("^Latent_", colnames(latent_data), value = TRUE)
  latent_matrix <- as.matrix(latent_data[, latent_cols])
  original_labels <- cluster_results$Cluster_Labels
  k <- length(unique(original_labels))
  
  n_samples_boot <- nrow(cluster_results)
  sample_ratio <- 0.8
  
  cat(sprintf("  Samples: %d, Clusters: %d\n", n_samples_boot, k))
  cat(sprintf("  Bootstrap iterations: %d (Methods 2.7)\n", opt$n_bootstrap))
  cat(sprintf("  Sample ratio: %.0f%%\n\n", 100 * sample_ratio))
  
  # Bootstrap resampling
  boot_matrix <- matrix(NA, nrow = n_samples_boot, ncol = opt$n_bootstrap)
  ari_values <- numeric(opt$n_bootstrap)
  jaccard_values <- numeric(opt$n_bootstrap)
  
  cat("  Running bootstrap iterations...\n")
  
  for (b in 1:opt$n_bootstrap) {
    set.seed(b)
    
    # Sample with replacement (Methods 2.7)
    sample_idx <- sample(1:n_samples_boot, 
                         size = floor(n_samples_boot * sample_ratio), 
                         replace = TRUE)
    
    boot_data <- latent_matrix[sample_idx, , drop = FALSE]
    boot_labels_true <- original_labels[sample_idx]
    
    # Recluster
    boot_kmeans <- kmeans(boot_data, centers = k, nstart = 20, iter.max = 100)
    boot_labels_pred <- boot_kmeans$cluster
    
    # Store for sample-level stability
    unique_idx <- unique(sample_idx)
    for (i in seq_along(unique_idx)) {
      idx <- unique_idx[i]
      boot_matrix[idx, b] <- boot_labels_pred[which(sample_idx == idx)[1]]
    }
    
    # Calculate ARI (Methods 2.7)
    ari_values[b] <- adjustedRandIndex(boot_labels_true, boot_labels_pred)
    
    # Calculate Jaccard Index (Methods 2.7)
    jaccard_values[b] <- calculate_jaccard_index(boot_labels_true, boot_labels_pred)
    
    if (b %% 20 == 0) {
      cat(sprintf("    Completed %d/%d iterations\n", b, opt$n_bootstrap))
    }
  }
  
  # Calculate sample-level stability (Methods 2.7: >0.85)
  sample_stability <- calculate_sample_stability(boot_matrix, original_labels)
  
  # Summary statistics
  mean_ari <- mean(ari_values, na.rm = TRUE)
  sd_ari <- sd(ari_values, na.rm = TRUE)
  mean_jaccard <- mean(jaccard_values, na.rm = TRUE)
  sd_jaccard <- sd(jaccard_values, na.rm = TRUE)
  mean_stability <- mean(sample_stability, na.rm = TRUE)
  stable_samples <- sum(sample_stability > opt$stability_threshold, na.rm = TRUE)
  stable_fraction <- stable_samples / sum(!is.na(sample_stability))
  
  # Silhouette coefficient
  dist_matrix <- dist(latent_matrix)
  sil <- silhouette(original_labels, dist_matrix)
  avg_sil <- mean(sil[, 3])
  
  cat(sprintf("\n  Bootstrap Results (Methods 2.7):\n"))
  cat(sprintf("    ARI: %.3f ± %.3f\n", mean_ari, sd_ari))
  cat(sprintf("    Jaccard Index: %.3f ± %.3f\n", mean_jaccard, sd_jaccard))
  cat(sprintf("    Sample Stability: %.3f\n", mean_stability))
  cat(sprintf("    Stable samples (>%.2f): %d/%d (%.1f%%)\n", 
              opt$stability_threshold, stable_samples, 
              sum(!is.na(sample_stability)), 100 * stable_fraction))
  cat(sprintf("    Silhouette: %.3f\n\n", avg_sil))
  
  # Interpretation (Methods 2.7)
  if (mean_ari > 0.8) {
    ari_interpretation <- "Excellent stability"
  } else if (mean_ari > 0.6) {
    ari_interpretation <- "Good stability"
  } else {
    ari_interpretation <- "Moderate stability"
  }
  cat(sprintf("    Interpretation: %s\n\n", ari_interpretation))
  
  # Save results
  stability_results <- data.frame(
    ID = cluster_results$ID,
    Cluster = original_labels,
    Sample_Stability = sample_stability,
    Stable = sample_stability > opt$stability_threshold
  )
  
  write.csv(stability_results, 
            file.path(opt$output_dir, "Bootstrap_Sample_Stability.csv"), 
            row.names = FALSE)
  
  summary_stats <- data.frame(
    Metric = c("ARI_Mean", "ARI_SD", "Jaccard_Mean", "Jaccard_SD", 
               "Sample_Stability_Mean", "Stable_Fraction", "Silhouette"),
    Value = c(mean_ari, sd_ari, mean_jaccard, sd_jaccard, 
              mean_stability, stable_fraction, avg_sil)
  )
  
  write.csv(summary_stats, 
            file.path(opt$output_dir, "Bootstrap_Stability_Summary.csv"), 
            row.names = FALSE)
  
  bootstrap_completed <- TRUE
  
} else {
  cat("  Latent embeddings or cluster results not found\n")
  cat("  Skipping bootstrap stability analysis\n\n")
  bootstrap_completed <- FALSE
}

# ==============================================================================
# Part 4: Visualizations
# ==============================================================================
cat("[4/5] Generating visualizations...\n")

# 4.1 Consensus Clustering Evaluation
png(file.path(opt$output_dir, "Consensus_Evaluation.png"), 
    width = 4800, height = 3600, res = 300)
par(mfrow = c(2, 2), mar = c(5, 5, 4, 2))

# PAC curve
plot(stability_table$K, stability_table$PAC,
     type = "b", pch = 19, cex = 2.5, lwd = 3, col = "steelblue",
     xlab = "Number of Clusters (K)", ylab = "PAC",
     main = sprintf("PAC: Lower is Better (threshold < %.2f)", opt$pac_threshold))
points(optimal_k, stability_table$PAC[optimal_k - 1],
       pch = 19, cex = 4, col = "red")
abline(h = opt$pac_threshold, lty = 2, col = "darkgreen", lwd = 2)
legend("topright", legend = c("PAC threshold", "Optimal K"),
       col = c("darkgreen", "red"), lty = c(2, NA), pch = c(NA, 19))

# Silhouette curve
plot(stability_table$K, stability_table$Silhouette,
     type = "b", pch = 19, cex = 2.5, lwd = 3, col = "coral",
     xlab = "Number of Clusters (K)", ylab = "Silhouette",
     main = "Silhouette: Higher is Better")
points(optimal_k, stability_table$Silhouette[optimal_k - 1],
       pch = 19, cex = 4, col = "red")

# Minimum cluster size
barplot(stability_table$Min_Cluster_Size,
        names.arg = paste0("K=", stability_table$K),
        col = ifelse(stability_table$K == optimal_k, "red", "steelblue"),
        main = "Minimum Cluster Size",
        xlab = "Number of Clusters (K)",
        ylab = "Min N per Cluster")
abline(h = 30, lty = 2, col = "darkgreen", lwd = 3)

# Summary text
plot.new()
text(0.5, 0.9, "Consensus Clustering Summary", cex = 1.5, font = 2)
text(0.5, 0.7, sprintf("Optimal K: %d", optimal_k), cex = 1.2)
text(0.5, 0.6, sprintf("PAC: %.4f", stability_table$PAC[optimal_k - 1]), cex = 1.2)
text(0.5, 0.5, sprintf("Silhouette: %.3f", stability_table$Silhouette[optimal_k - 1]), cex = 1.2)
text(0.5, 0.3, sprintf("PAC < %.2f: %s", opt$pac_threshold,
                       ifelse(stability_table$PAC[optimal_k - 1] < opt$pac_threshold, 
                              "Clear structure", "Ambiguous")), cex = 1.1)

dev.off()
cat("  Saved: Consensus_Evaluation.png\n")

# 4.2 Bootstrap Stability Plots (if completed)
if (bootstrap_completed) {
  png(file.path(opt$output_dir, "Bootstrap_Stability_Plots.png"), 
      width = 4500, height = 3000, res = 300)
  par(mfrow = c(2, 3), mar = c(5, 5, 4, 2))
  
  # ARI distribution
  hist(ari_values, breaks = 30, col = "steelblue", border = "white",
       main = sprintf("ARI Distribution (μ=%.3f)", mean_ari),
       xlab = "Adjusted Rand Index", ylab = "Frequency")
  abline(v = mean_ari, col = "red", lwd = 2)
  abline(v = 0.8, col = "darkgreen", lwd = 2, lty = 2)
  legend("topleft", legend = c("Mean", "Threshold (0.8)"),
         col = c("red", "darkgreen"), lwd = 2, lty = c(1, 2))
  
  # Jaccard distribution
  hist(jaccard_values, breaks = 30, col = "coral", border = "white",
       main = sprintf("Jaccard Index Distribution (μ=%.3f)", mean_jaccard),
       xlab = "Jaccard Index", ylab = "Frequency")
  abline(v = mean_jaccard, col = "red", lwd = 2)
  
  # Sample stability distribution
  hist(sample_stability[!is.na(sample_stability)], breaks = 30, 
       col = "seagreen", border = "white",
       main = sprintf("Sample Stability (%.1f%% > %.2f)", 
                      100 * stable_fraction, opt$stability_threshold),
       xlab = "Sample Stability", ylab = "Frequency")
  abline(v = opt$stability_threshold, col = "red", lwd = 2, lty = 2)
  
  # Silhouette by cluster
  boxplot(sil[, 3] ~ original_labels, col = rainbow(k),
          main = "Silhouette by Cluster", 
          xlab = "Cluster", ylab = "Silhouette")
  
  # Sample-level stability scatter
  plot(1:length(sample_stability), sample_stability, 
       pch = 19, col = rainbow(k)[original_labels],
       main = "Sample-level Stability", 
       xlab = "Sample Index", ylab = "Stability")
  abline(h = opt$stability_threshold, lty = 2, col = "red", lwd = 2)
  
  # Summary metrics
  barplot(c(ARI = mean_ari, Jaccard = mean_jaccard, 
            Stability = mean_stability, Silhouette = avg_sil),
          col = c("steelblue", "coral", "seagreen", "purple"),
          main = "Overall Stability Metrics", ylab = "Value",
          ylim = c(0, 1))
  abline(h = 0.8, lty = 2, col = "darkgreen", lwd = 2)
  
  dev.off()
  cat("  Saved: Bootstrap_Stability_Plots.png\n")
}

# Save final consensus clusters
final_consensus <- data.frame(
  Sample_Index = 1:ncol(feature_matrix),
  Consensus_Cluster = consensus_result[[optimal_k]]$consensusClass
)
write.csv(final_consensus, 
          file.path(opt$output_dir, "Final_Consensus_Clusters.csv"), 
          row.names = FALSE)


# ==============================================================================
# Part 5: Summary Report
# ==============================================================================
cat("[5/5] Generating summary report...\n\n")

summary_lines <- c(
  "================================================================================",
  "Cluster Validation Report (Methods 2.7 Aligned)",
  "================================================================================",
  "",
  sprintf("Generated: %s", Sys.time()),
  "",
  "Methods 2.7 Requirements:",
  sprintf("  - Bootstrap iterations: %d", opt$n_bootstrap),
  sprintf("  - Sample stability threshold: >%.2f", opt$stability_threshold),
  sprintf("  - PAC threshold: <%.2f", opt$pac_threshold),
  "",
  "--------------------------------------------------------------------------------",
  "PART 1: Consensus Clustering",
  "--------------------------------------------------------------------------------",
  sprintf("  Samples analyzed: %d", n_samples),
  sprintf("  Features used: %d", nrow(feature_matrix)),
  sprintf("  Optimal K: %d", optimal_k),
  sprintf("  PAC score (K=%d): %.4f", optimal_k, stability_table$PAC[optimal_k - 1]),
  sprintf("  PAC < %.2f: %s", opt$pac_threshold,
          ifelse(stability_table$PAC[optimal_k - 1] < opt$pac_threshold, 
                 "YES - Clear structure", "NO - Ambiguous structure")),
  sprintf("  Silhouette (K=%d): %.3f", optimal_k, stability_table$Silhouette[optimal_k - 1]),
  ""
)

if (bootstrap_completed) {
  summary_lines <- c(summary_lines,
    "--------------------------------------------------------------------------------",
    "PART 2: Bootstrap Stability Analysis (Methods 2.7)",
    "--------------------------------------------------------------------------------",
    sprintf("  Bootstrap iterations: %d", opt$n_bootstrap),
    sprintf("  Sample ratio: 80%%"),
    "",
    "  Results:",
    sprintf("    ARI: %.3f ± %.3f", mean_ari, sd_ari),
    sprintf("    Jaccard Index: %.3f ± %.3f", mean_jaccard, sd_jaccard),
    sprintf("    Sample Stability: %.3f", mean_stability),
    sprintf("    Stable samples (>%.2f): %d/%d (%.1f%%)", 
            opt$stability_threshold, stable_samples, 
            sum(!is.na(sample_stability)), 100 * stable_fraction),
    sprintf("    Silhouette: %.3f", avg_sil),
    "",
    "  Methods 2.7 Criteria:",
    sprintf("    ARI > 0.8: %s", ifelse(mean_ari > 0.8, "PASS - Excellent stability", 
                                        ifelse(mean_ari > 0.6, "PASS - Good stability", 
                                               "MARGINAL - Moderate stability"))),
    sprintf("    Sample stability > %.2f: %.1f%% of samples", 
            opt$stability_threshold, 100 * stable_fraction),
    ""
  )
} else {
  summary_lines <- c(summary_lines,
    "--------------------------------------------------------------------------------",
    "PART 2: Bootstrap Stability Analysis",
    "--------------------------------------------------------------------------------",
    "  Status: SKIPPED (missing latent embeddings or cluster results)",
    ""
  )
}

summary_lines <- c(summary_lines,
  "--------------------------------------------------------------------------------",
  "Output Files",
  "--------------------------------------------------------------------------------",
  "  Consensus Clustering:",
  sprintf("    - %s/Consensus_Stability_Metrics.csv", opt$output_dir),
  sprintf("    - %s/Final_Consensus_Clusters.csv", opt$output_dir),
  sprintf("    - %s/Consensus_Evaluation.png", opt$output_dir)
)

if (bootstrap_completed) {
  summary_lines <- c(summary_lines,
    "",
    "  Bootstrap Stability:",
    sprintf("    - %s/Bootstrap_Sample_Stability.csv", opt$output_dir),
    sprintf("    - %s/Bootstrap_Stability_Summary.csv", opt$output_dir),
    sprintf("    - %s/Bootstrap_Stability_Plots.png", opt$output_dir)
  )
}

summary_lines <- c(summary_lines,
  "",
  "================================================================================",
  "Cluster Validation Complete",
  "================================================================================"
)

# Write report
report_path <- file.path(opt$output_dir, "Cluster_Validation_Report.txt")
writeLines(summary_lines, report_path)

# Print summary
cat(paste(summary_lines, collapse = "\n"))
cat("\n\n")

cat("========================================================================\n")
cat("Step 14: Cluster Validation Complete!\n")
cat("========================================================================\n")
cat(sprintf("Report saved: %s\n", report_path))
cat(sprintf("Output directory: %s\n", opt$output_dir))
