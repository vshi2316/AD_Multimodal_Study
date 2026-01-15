library(limma)
library(dplyr)
library(tidyr)
library(ggplot2)
library(ggrepel)
library(pheatmap)
library(tableone)
library(RColorBrewer)
library(optparse)

# ==============================================================================
# Parse Command Line Arguments
# ==============================================================================
option_list <- list(
  make_option(c("--cluster_file"), type = "character", default = "cluster_results.csv",
              help = "Path to cluster results CSV [default: %default]"),
  make_option(c("--clinical_file"), type = "character", default = "Clinical_data.csv",
              help = "Path to clinical data CSV [default: %default]"),
  make_option(c("--smri_file"), type = "character", default = "sMRI_data.csv",
              help = "Path to sMRI data CSV [default: %default]"),
  make_option(c("--csf_file"), type = "character", default = "CSF_data.csv",
              help = "Path to CSF data CSV [default: %default]"),
  make_option(c("--output_dir"), type = "character", default = "./results",
              help = "Output directory [default: %default]"),
  make_option(c("--fdr_threshold"), type = "numeric", default = 0.05,
              help = "FDR significance threshold (Methods 2.4: q < 0.05) [default: %default]"),
  make_option(c("--smd_threshold"), type = "numeric", default = 0.5,
              help = "SMD clinical meaningfulness threshold (Methods 2.4: |SMD| > 0.5) [default: %default]")
)

opt_parser <- OptionParser(option_list = option_list)
opt <- parse_args(opt_parser)

# Create output directory
dir.create(opt$output_dir, showWarnings = FALSE, recursive = TRUE)

# ==============================================================================
# Methods 2.8: Effect Size Functions
# ==============================================================================

#' Calculate Cohen's d (Standardized Mean Difference)
#' Methods 2.4: "requiring |SMD| > 0.5 for clinical meaningfulness"
calculate_cohens_d <- function(group1, group2) {
  n1 <- length(group1)
  n2 <- length(group2)
  mean1 <- mean(group1, na.rm = TRUE)
  mean2 <- mean(group2, na.rm = TRUE)
  var1 <- var(group1, na.rm = TRUE)
  var2 <- var(group2, na.rm = TRUE)
  
  # Pooled standard deviation
  pooled_sd <- sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
  
  if (is.na(pooled_sd) || pooled_sd == 0) return(NA)
  
  cohens_d <- (mean1 - mean2) / pooled_sd
  return(cohens_d)
}

#' Calculate Eta-squared effect size
#' Methods 2.8: "Eta-squared (η²) quantifies effect sizes: between-group sum 
#' of squares divided by total sum of squares, with 0.01, 0.06, and 0.14 
#' representing small, medium, and significant effects per Cohen's criteria"
calculate_eta_squared <- function(values, groups) {
  values <- as.numeric(values)
  groups <- as.factor(groups)
  
  # Remove NA
  valid_idx <- !is.na(values) & !is.na(groups)
  values <- values[valid_idx]
  groups <- groups[valid_idx]
  
  if (length(unique(groups)) < 2) return(NA)
  
  grand_mean <- mean(values)
  ss_total <- sum((values - grand_mean)^2)
  
  ss_between <- 0
  for (g in unique(groups)) {
    group_values <- values[groups == g]
    group_mean <- mean(group_values)
    ss_between <- ss_between + length(group_values) * (group_mean - grand_mean)^2
  }
  
  eta_squared <- ss_between / ss_total
  return(eta_squared)
}

#' Interpret effect size per Cohen's criteria
interpret_eta_squared <- function(eta_sq) {
  if (is.na(eta_sq)) return("NA")
  if (eta_sq < 0.01) return("Negligible")
  if (eta_sq < 0.06) return("Small")
  if (eta_sq < 0.14) return("Medium")
  return("Large")
}

interpret_cohens_d <- function(d) {
  if (is.na(d)) return("NA")
  abs_d <- abs(d)
  if (abs_d < 0.2) return("Negligible")
  if (abs_d < 0.5) return("Small")
  if (abs_d < 0.8) return("Medium")
  return("Large")
}

# ==============================================================================
# Load Data
# ==============================================================================
cat("[1/6] Loading data...\n")

cluster_results <- read.csv(opt$cluster_file, stringsAsFactors = FALSE)
cluster_results$ID <- as.character(cluster_results$ID)

# Define risk groups based on cluster labels
n_clusters <- length(unique(cluster_results$Cluster_Labels))
cat(sprintf("  Detected %d clusters\n", n_clusters))

# For 2-cluster case: Cluster 2 = High-Risk, Cluster 1 = Low-Risk
# For 3-cluster case: Use conversion rates to determine risk
if (n_clusters == 2) {
  cluster_results$Risk_Group <- ifelse(cluster_results$Cluster_Labels == 2, "HighRisk", "LowRisk")
} else {
  # Calculate conversion rate per cluster to assign risk
  if ("AD_Conversion" %in% colnames(cluster_results)) {
    conv_rates <- cluster_results %>%
      group_by(Cluster_Labels) %>%
      summarise(conv_rate = mean(AD_Conversion, na.rm = TRUE), .groups = "drop") %>%
      arrange(desc(conv_rate))
    high_risk_cluster <- conv_rates$Cluster_Labels[1]
    cluster_results$Risk_Group <- ifelse(cluster_results$Cluster_Labels == high_risk_cluster, 
                                          "HighRisk", "LowRisk")
  } else {
    cluster_results$Risk_Group <- ifelse(cluster_results$Cluster_Labels == max(cluster_results$Cluster_Labels), 
                                          "HighRisk", "LowRisk")
  }
}

high_risk_ids <- cluster_results %>% filter(Risk_Group == "HighRisk") %>% pull(ID)
low_risk_ids <- cluster_results %>% filter(Risk_Group == "LowRisk") %>% pull(ID)

cat(sprintf("  High-Risk: n=%d, Low-Risk: n=%d\n\n", length(high_risk_ids), length(low_risk_ids)))

# Load modality data
modalities <- list()
if (file.exists(opt$clinical_file)) {
  modalities[["Clinical"]] <- read.csv(opt$clinical_file, stringsAsFactors = FALSE)
  cat(sprintf("  Loaded Clinical: %d samples, %d features\n", nrow(modalities[["Clinical"]]), ncol(modalities[["Clinical"]]) - 1))
}
if (file.exists(opt$smri_file)) {
  modalities[["sMRI"]] <- read.csv(opt$smri_file, stringsAsFactors = FALSE)
  cat(sprintf("  Loaded sMRI: %d samples, %d features\n", nrow(modalities[["sMRI"]]), ncol(modalities[["sMRI"]]) - 1))
}
if (file.exists(opt$csf_file)) {
  modalities[["CSF"]] <- read.csv(opt$csf_file, stringsAsFactors = FALSE)
  cat(sprintf("  Loaded CSF: %d samples, %d features\n", nrow(modalities[["CSF"]]), ncol(modalities[["CSF"]]) - 1))
}

if (length(modalities) == 0) {
  stop("No modality data files found!")
}

# ==============================================================================
# Part 1: limma Differential Analysis with FDR Correction
# ==============================================================================
cat("\n[2/6] limma Differential Expression Analysis...\n")
cat("  Applying Benjamini-Hochberg FDR correction (Methods 2.4)\n\n")

limma_results <- list()

for (modality_name in names(modalities)) {
  cat(sprintf("[%s] Processing...\n", modality_name))
  
  data <- modalities[[modality_name]]
  data$ID <- as.character(data$ID)
  
  # Filter to samples in cluster results
  data <- data %>%
    filter(ID %in% cluster_results$ID) %>%
    left_join(cluster_results %>% select(ID, Risk_Group), by = "ID")
  
  # Get numeric feature columns
  feature_cols <- sapply(data, is.numeric)
  feature_cols["ID"] <- FALSE
  feature_cols["Risk_Group"] <- FALSE
  
  data_matrix <- data[, feature_cols, drop = FALSE]
  
  if (ncol(data_matrix) == 0) {
    cat(sprintf("  No numeric features in %s\n", modality_name))
    next
  }
  
  # Remove features with >50% missing
  missing_rate <- colMeans(is.na(data_matrix))
  data_matrix <- data_matrix[, missing_rate <= 0.5, drop = FALSE]
  
  # Median imputation for remaining missing values
  for (j in 1:ncol(data_matrix)) {
    if (any(is.na(data_matrix[, j]))) {
      data_matrix[is.na(data_matrix[, j]), j] <- median(data_matrix[, j], na.rm = TRUE)
    }
  }
  
  # Design matrix
  group <- factor(data$Risk_Group, levels = c("LowRisk", "HighRisk"))
  design <- model.matrix(~0 + group)
  colnames(design) <- c("LowRisk", "HighRisk")
  
  # limma fit
  fit <- lmFit(t(data_matrix), design)
  contrast_matrix <- makeContrasts(HighVsLow = HighRisk - LowRisk, levels = design)
  fit2 <- contrasts.fit(fit, contrast_matrix)
  fit2 <- eBayes(fit2, trend = TRUE)
  
  # Get results with FDR correction (Benjamini-Hochberg)
  results <- topTable(fit2, coef = "HighVsLow", number = Inf, adjust.method = "BH")
  results$Feature <- rownames(results)
  results$Modality <- modality_name
  
  # Calculate SMD (Cohen's d) for each feature
  results$SMD <- NA
  results$SMD_Interpretation <- NA
  
  high_risk_data <- data_matrix[data$Risk_Group == "HighRisk", , drop = FALSE]
  low_risk_data <- data_matrix[data$Risk_Group == "LowRisk", , drop = FALSE]
  
  for (feat in results$Feature) {
    if (feat %in% colnames(data_matrix)) {
      smd <- calculate_cohens_d(high_risk_data[, feat], low_risk_data[, feat])
      results$SMD[results$Feature == feat] <- smd
      results$SMD_Interpretation[results$Feature == feat] <- interpret_cohens_d(smd)
    }
  }
  
  # Calculate Eta-squared
  results$Eta_Squared <- NA
  results$Eta_Interpretation <- NA
  
  for (feat in results$Feature) {
    if (feat %in% colnames(data_matrix)) {
      eta_sq <- calculate_eta_squared(data_matrix[, feat], data$Risk_Group)
      results$Eta_Squared[results$Feature == feat] <- eta_sq
      results$Eta_Interpretation[results$Feature == feat] <- interpret_eta_squared(eta_sq)
    }
  }
  
  # Methods 2.4: FDR q < 0.05 AND |SMD| > 0.5 for clinical meaningfulness
  results$Clinically_Meaningful <- abs(results$SMD) > opt$smd_threshold
  
  significant <- results %>%
    filter(adj.P.Val < opt$fdr_threshold & abs(SMD) > opt$smd_threshold) %>%
    arrange(adj.P.Val)
  
  cat(sprintf("  Total features: %d\n", nrow(results)))
  cat(sprintf("  FDR significant (q < %.2f): %d\n", opt$fdr_threshold, sum(results$adj.P.Val < opt$fdr_threshold)))
  cat(sprintf("  Clinically meaningful (|SMD| > %.1f): %d\n", opt$smd_threshold, sum(results$Clinically_Meaningful, na.rm = TRUE)))
  cat(sprintf("  Both criteria met: %d\n\n", nrow(significant)))
  
  limma_results[[modality_name]] <- list(
    all_results = results,
    significant = significant,
    data_matrix = data_matrix,
    risk_group = data$Risk_Group
  )
  
  # Save results
  write.csv(results, file.path(opt$output_dir, sprintf("DiffExpr_%s_All_FDR.csv", modality_name)), row.names = FALSE)
  
  if (nrow(significant) > 0) {
    write.csv(significant, file.path(opt$output_dir, sprintf("DiffExpr_%s_Significant_FDR.csv", modality_name)), row.names = FALSE)
  }
}

# ==============================================================================
# Part 2: Comprehensive SMD Analysis with FDR Correction
# ==============================================================================
cat("\n[3/6] Standardized Mean Difference (SMD) Analysis...\n")
cat("  Methods 2.4: Benjamini-Hochberg correction (q < 0.05)\n")
cat("  Methods 2.4: Clinical meaningfulness threshold |SMD| > 0.5\n\n")

# Merge all modality data
all_data <- cluster_results

for (modality_name in names(modalities)) {
  mod_data <- modalities[[modality_name]]
  mod_data$ID <- as.character(mod_data$ID)
  
  # Rename columns to include modality prefix
  feature_cols <- setdiff(colnames(mod_data), "ID")
  colnames(mod_data)[colnames(mod_data) %in% feature_cols] <- paste0(modality_name, "_", feature_cols)
  
  all_data <- all_data %>%
    left_join(mod_data, by = "ID")
}

# Get all feature columns
exclude_cols <- c("ID", "Cluster_Labels", "Risk_Group", "AD_Conversion", "Time_to_Event", "Followup_Years")
feature_cols <- setdiff(colnames(all_data), exclude_cols)
feature_cols <- feature_cols[sapply(all_data[, feature_cols, drop = FALSE], is.numeric)]

cat(sprintf("  Total features across modalities: %d\n", length(feature_cols)))

# Calculate SMD for all features
all_smd_results <- data.frame()

for (feat in feature_cols) {
  values <- all_data[[feat]]
  
  # Skip if too many missing
  if (mean(is.na(values)) > 0.5) next
  
  # Median imputation
  values[is.na(values)] <- median(values, na.rm = TRUE)
  
  high_risk_vals <- values[all_data$Risk_Group == "HighRisk"]
  low_risk_vals <- values[all_data$Risk_Group == "LowRisk"]
  
  # Wilcoxon rank-sum test (Methods 2.4)
  wilcox_test <- tryCatch(
    wilcox.test(high_risk_vals, low_risk_vals),
    error = function(e) list(p.value = NA)
  )
  
  # Calculate SMD
  smd <- calculate_cohens_d(high_risk_vals, low_risk_vals)
  
  # Calculate Eta-squared
  eta_sq <- calculate_eta_squared(values, all_data$Risk_Group)
  
  # Determine modality
  modality <- "Unknown"
  for (mod_name in names(modalities)) {
    if (grepl(paste0("^", mod_name, "_"), feat)) {
      modality <- mod_name
      break
    }
  }
  
  all_smd_results <- rbind(all_smd_results, data.frame(
    Feature = feat,
    Modality = modality,
    Mean_HighRisk = mean(high_risk_vals, na.rm = TRUE),
    Mean_LowRisk = mean(low_risk_vals, na.rm = TRUE),
    SD_HighRisk = sd(high_risk_vals, na.rm = TRUE),
    SD_LowRisk = sd(low_risk_vals, na.rm = TRUE),
    SMD = smd,
    SMD_Interpretation = interpret_cohens_d(smd),
    Eta_Squared = eta_sq,
    Eta_Interpretation = interpret_eta_squared(eta_sq),
    P_Value_Wilcoxon = wilcox_test$p.value,
    stringsAsFactors = FALSE
  ))
}

# Apply Benjamini-Hochberg FDR correction (Methods 2.4)
all_smd_results$P_adj_FDR <- p.adjust(all_smd_results$P_Value_Wilcoxon, method = "BH")

# Determine significance
all_smd_results$FDR_Significant <- all_smd_results$P_adj_FDR < opt$fdr_threshold
all_smd_results$Clinically_Meaningful <- abs(all_smd_results$SMD) > opt$smd_threshold
all_smd_results$Both_Criteria <- all_smd_results$FDR_Significant & all_smd_results$Clinically_Meaningful

# Sort by absolute SMD
all_smd_results <- all_smd_results %>%
  arrange(desc(abs(SMD)))

# Filter significant features
significant_smd <- all_smd_results %>%
  filter(Both_Criteria) %>%
  arrange(P_adj_FDR, desc(abs(SMD)))

cat(sprintf("\n  Total features analyzed: %d\n", nrow(all_smd_results)))
cat(sprintf("  FDR significant (q < %.2f): %d\n", opt$fdr_threshold, sum(all_smd_results$FDR_Significant, na.rm = TRUE)))
cat(sprintf("  Clinically meaningful (|SMD| > %.1f): %d\n", opt$smd_threshold, sum(all_smd_results$Clinically_Meaningful, na.rm = TRUE)))
cat(sprintf("  Both criteria met: %d\n\n", nrow(significant_smd)))

# Save SMD results
write.csv(all_smd_results, file.path(opt$output_dir, "SMD_All_Features_FDR.csv"), row.names = FALSE)
write.csv(significant_smd, file.path(opt$output_dir, "SMD_Significant_Features_FDR.csv"), row.names = FALSE)

# ==============================================================================
# Part 3: Visualizations
# ==============================================================================
cat("[4/6] Generating Visualizations...\n\n")

# 3.1 Volcano plots with FDR threshold
for (modality_name in names(limma_results)) {
  result <- limma_results[[modality_name]]
  if (is.null(result)) next
  
  cat(sprintf("  Creating volcano plot: %s\n", modality_name))
  
  plot_data <- result$all_results
  plot_data$Regulation <- "NS"
  plot_data$Regulation[plot_data$adj.P.Val < opt$fdr_threshold & plot_data$SMD > opt$smd_threshold] <- "Up in High-Risk"
  plot_data$Regulation[plot_data$adj.P.Val < opt$fdr_threshold & plot_data$SMD < -opt$smd_threshold] <- "Down in High-Risk"
  
  # Label top features
  plot_data <- plot_data %>%
    arrange(adj.P.Val) %>%
    mutate(Label = ifelse(row_number() <= 10 & adj.P.Val < opt$fdr_threshold, Feature, ""))
  
  p <- ggplot(plot_data, aes(x = SMD, y = -log10(adj.P.Val), color = Regulation)) +
    geom_point(aes(size = abs(SMD)), alpha = 0.7) +
    scale_size_continuous(range = c(1, 4), name = "|SMD|") +
    scale_color_manual(values = c(
      "Up in High-Risk" = "#d62728",
      "Down in High-Risk" = "#1f77b4",
      "NS" = "grey70"
    )) +
    geom_hline(yintercept = -log10(opt$fdr_threshold), linetype = "dashed", color = "grey30") +
    geom_vline(xintercept = c(-opt$smd_threshold, opt$smd_threshold), linetype = "dashed", color = "grey30") +
    geom_text_repel(aes(label = Label), size = 3.5, max.overlaps = 20) +
    labs(
      title = sprintf("%s: High-Risk vs Low-Risk (FDR-corrected)", modality_name),
      subtitle = sprintf("FDR threshold: q < %.2f, SMD threshold: |SMD| > %.1f", opt$fdr_threshold, opt$smd_threshold),
      x = "Standardized Mean Difference (Cohen's d)",
      y = expression(-log[10]~"(FDR-adjusted P-value)")
    ) +
    theme_bw() +
    theme(
      legend.position = "right",
      plot.title = element_text(hjust = 0.5, face = "bold"),
      plot.subtitle = element_text(hjust = 0.5, size = 10)
    )
  
  ggsave(file.path(opt$output_dir, sprintf("Volcano_%s_FDR.png", modality_name)), 
         plot = p, width = 10, height = 8, dpi = 300)
}

# 3.2 Heatmaps for significant features
for (modality_name in names(limma_results)) {
  result <- limma_results[[modality_name]]
  if (is.null(result) || nrow(result$significant) < 2) next
  
  cat(sprintf("  Creating heatmap: %s\n", modality_name))
  
  sig_features <- result$significant$Feature
  sig_features <- sig_features[sig_features %in% colnames(result$data_matrix)]
  
  if (length(sig_features) < 2) next
  
  sig_features <- head(sig_features, 50)
  
  heatmap_data <- t(result$data_matrix[, sig_features, drop = FALSE])
  heatmap_data_scaled <- t(scale(t(heatmap_data)))
  
  annotation_col <- data.frame(
    Risk_Group = factor(result$risk_group, levels = c("HighRisk", "LowRisk"))
  )
  rownames(annotation_col) <- 1:nrow(annotation_col)
  colnames(heatmap_data_scaled) <- 1:ncol(heatmap_data_scaled)
  
  ann_colors <- list(Risk_Group = c("HighRisk" = "#d62728", "LowRisk" = "#1f77b4"))
  
  png(file.path(opt$output_dir, sprintf("Heatmap_%s_FDR.png", modality_name)), 
      width = 3600, height = 2400, res = 300)
  pheatmap(
    heatmap_data_scaled,
    annotation_col = annotation_col,
    annotation_colors = ann_colors,
    cluster_rows = TRUE,
    cluster_cols = TRUE,
    show_colnames = FALSE,
    show_rownames = TRUE,
    main = sprintf("%s: FDR-Significant Features (q < %.2f, |SMD| > %.1f)", 
                   modality_name, opt$fdr_threshold, opt$smd_threshold),
    color = colorRampPalette(rev(brewer.pal(11, "RdBu")))(100),
    border_color = NA
  )
  dev.off()
}

# 3.3 Top SMD features barplot
if (nrow(all_smd_results) > 0) {
  cat("  Creating Top SMD features plot\n")
  
  top_features <- all_smd_results %>%
    arrange(desc(abs(SMD))) %>%
    head(min(30, nrow(all_smd_results)))
  
  top_features$Direction <- ifelse(top_features$SMD > 0, "Higher in High-Risk", "Lower in High-Risk")
  top_features$Significant <- ifelse(top_features$Both_Criteria, "Significant", "Not Significant")
  
  p <- ggplot(top_features, aes(x = reorder(Feature, abs(SMD)), y = SMD, fill = Modality, alpha = Significant)) +
    geom_bar(stat = "identity") +
    coord_flip() +
    scale_alpha_manual(values = c("Significant" = 1.0, "Not Significant" = 0.5)) +
    geom_hline(yintercept = c(-opt$smd_threshold, opt$smd_threshold), linetype = "dashed", color = "red") +
    labs(
      title = "Top 30 Features by Standardized Mean Difference",
      subtitle = sprintf("Red dashed lines: |SMD| = %.1f (clinical meaningfulness threshold)", opt$smd_threshold),
      x = "Feature",
      y = "SMD (Cohen's d)"
    ) +
    theme_minimal() +
    theme(plot.title = element_text(face = "bold"))
  
  ggsave(file.path(opt$output_dir, "Top30_SMD_Features_FDR.png"), plot = p, width = 12, height = 10, dpi = 300)
}

# 3.4 Effect size distribution
if (nrow(all_smd_results) > 0) {
  cat("  Creating effect size distribution plot\n")
  
  p <- ggplot(all_smd_results, aes(x = abs(SMD), fill = Modality)) +
    geom_histogram(bins = 30, alpha = 0.7, position = "identity") +
    geom_vline(xintercept = opt$smd_threshold, linetype = "dashed", color = "red", size = 1) +
    annotate("text", x = opt$smd_threshold + 0.1, y = Inf, vjust = 2, 
             label = sprintf("|SMD| = %.1f\n(Clinical threshold)", opt$smd_threshold), 
             color = "red", size = 3) +
    labs(
      title = "Distribution of Effect Sizes (|SMD|)",
      subtitle = "Methods 2.4: |SMD| > 0.5 required for clinical meaningfulness",
      x = "|Standardized Mean Difference|",
      y = "Count"
    ) +
    theme_minimal() +
    theme(plot.title = element_text(face = "bold"))
  
  ggsave(file.path(opt$output_dir, "Effect_Size_Distribution.png"), plot = p, width = 10, height = 6, dpi = 300)
}

# ==============================================================================
# Part 4: Summary Report
# ==============================================================================
cat("\n[5/6] Generating Summary Report...\n\n")

summary_report <- data.frame()

for (modality_name in names(limma_results)) {
  result <- limma_results[[modality_name]]
  
  top_features_str <- if (nrow(result$significant) > 0) {
    paste(head(result$significant$Feature, 5), collapse = "; ")
  } else {
    "None"
  }
  
  # Mean effect sizes
  mean_smd <- mean(abs(result$all_results$SMD), na.rm = TRUE)
  mean_eta <- mean(result$all_results$Eta_Squared, na.rm = TRUE)
  
  summary_report <- rbind(summary_report, data.frame(
    Modality = modality_name,
    N_Features = nrow(result$all_results),
    N_FDR_Significant = sum(result$all_results$adj.P.Val < opt$fdr_threshold, na.rm = TRUE),
    N_Clinically_Meaningful = sum(result$all_results$Clinically_Meaningful, na.rm = TRUE),
    N_Both_Criteria = nrow(result$significant),
    Mean_Abs_SMD = round(mean_smd, 3),
    Mean_Eta_Squared = round(mean_eta, 4),
    Top_Features = top_features_str,
    stringsAsFactors = FALSE
  ))
}

print(summary_report)

write.csv(summary_report, file.path(opt$output_dir, "DiffExpr_Summary_FDR.csv"), row.names = FALSE)

# ==============================================================================
# Part 5: Methods Compliance Check
# ==============================================================================
cat("\n[6/6] Methods Compliance Verification...\n\n")

cat("Methods 2.4 Compliance:\n")
cat(sprintf("  ✓ Benjamini-Hochberg FDR correction applied (q < %.2f)\n", opt$fdr_threshold))
cat(sprintf("  ✓ SMD clinical meaningfulness threshold (|SMD| > %.1f)\n", opt$smd_threshold))
cat("  ✓ Wilcoxon rank-sum tests for continuous variables\n")

cat("\nMethods 2.8 Compliance:\n")
cat("  ✓ Eta-squared effect sizes calculated\n")
cat("  ✓ Cohen's criteria applied (0.01 small, 0.06 medium, 0.14 large)\n")
cat("  ✓ Cohen's d (SMD) calculated for all features\n")

cat("\n========================================================================\n")
cat("Differential Analysis Complete!\n")
cat("========================================================================\n\n")

cat("Output files:\n")
for (modality_name in names(limma_results)) {
  cat(sprintf("  - DiffExpr_%s_All_FDR.csv\n", modality_name))
  if (nrow(limma_results[[modality_name]]$significant) > 0) {
    cat(sprintf("  - DiffExpr_%s_Significant_FDR.csv\n", modality_name))
  }
  cat(sprintf("  - Volcano_%s_FDR.png\n", modality_name))
}
cat("  - SMD_All_Features_FDR.csv\n")
cat("  - SMD_Significant_Features_FDR.csv\n")
cat("  - Top30_SMD_Features_FDR.png\n")
cat("  - Effect_Size_Distribution.png\n")
cat("  - DiffExpr_Summary_FDR.csv\n")

cat("\n========================================================================\n")
cat("Methods Alignment Summary:\n")
cat("  - FDR threshold: q < 0.05 (Methods 2.4)\n")
cat("  - SMD threshold: |SMD| > 0.5 (Methods 2.4)\n")
cat("  - Eta-squared with Cohen's criteria (Methods 2.8)\n")
cat("========================================================================\n")

