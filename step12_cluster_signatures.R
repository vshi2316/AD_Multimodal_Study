library(dplyr)
library(tidyr)
library(ggplot2)
library(pheatmap)
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
  make_option(c("--smd_file"), type = "character", default = "SMD_All_Features_FDR.csv",
              help = "Path to SMD analysis results [default: %default]"),
  make_option(c("--output_dir"), type = "character", default = "./results",
              help = "Output directory [default: %default]"),
  make_option(c("--fdr_threshold"), type = "numeric", default = 0.05,
              help = "FDR significance threshold [default: %default]")
)

opt_parser <- OptionParser(option_list = option_list)
opt <- parse_args(opt_parser)

# Create output directory
dir.create(opt$output_dir, showWarnings = FALSE, recursive = TRUE)

#' Calculate Eta-squared effect size
#' of squares divided by total sum of squares, with 0.01, 0.06, and 0.14 
#' representing small, medium, and significant effects per Cohen's criteria"
calculate_eta_squared <- function(values, groups) {
  values <- as.numeric(values)
  groups <- as.factor(groups)
  
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

interpret_eta_squared <- function(eta_sq) {
  if (is.na(eta_sq)) return("NA")
  if (eta_sq < 0.01) return("Negligible")
  if (eta_sq < 0.06) return("Small")
  if (eta_sq < 0.14) return("Medium")
  return("Large")
}

#' Calculate Cohen's d for pairwise comparisons
calculate_cohens_d <- function(group1, group2) {
  n1 <- length(group1)
  n2 <- length(group2)
  mean1 <- mean(group1, na.rm = TRUE)
  mean2 <- mean(group2, na.rm = TRUE)
  var1 <- var(group1, na.rm = TRUE)
  var2 <- var(group2, na.rm = TRUE)
  
  pooled_sd <- sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
  
  if (is.na(pooled_sd) || pooled_sd == 0) return(NA)
  
  cohens_d <- (mean1 - mean2) / pooled_sd
  return(cohens_d)
}

# ==============================================================================
# Load Data
# ==============================================================================
cat("[1/5] Loading data...\n")

cluster_results <- read.csv(opt$cluster_file, stringsAsFactors = FALSE)
cluster_results$ID <- as.character(cluster_results$ID)
cluster_results$Cluster <- factor(cluster_results$Cluster_Labels)

n_clusters <- length(unique(cluster_results$Cluster_Labels))
cat(sprintf("  Detected %d clusters\n", n_clusters))

# Load modality data
file_mapping <- list(
  Clinical = opt$clinical_file,
  sMRI = opt$smri_file,
  CSF = opt$csf_file
)

full_data <- cluster_results

for (modality in names(file_mapping)) {
  data_file <- file_mapping[[modality]]
  if (!file.exists(data_file)) {
    cat(sprintf("  Skipping %s (file not found)\n", modality))
    next
  }
  
  mod_data <- read.csv(data_file, stringsAsFactors = FALSE)
  mod_data$ID <- as.character(mod_data$ID)
  mod_data <- mod_data %>% filter(ID %in% cluster_results$ID)
  
  feature_cols <- sapply(mod_data, is.numeric)
  feature_cols["ID"] <- FALSE
  
  data_features <- mod_data[, feature_cols, drop = FALSE]
  colnames(data_features) <- paste0(modality, "_", colnames(data_features))
  
  mod_data_final <- cbind(mod_data["ID"], data_features)
  full_data <- full_data %>% left_join(mod_data_final, by = "ID")
  
  cat(sprintf("  Loaded %s: %d features\n", modality, ncol(data_features)))
}

# Load SMD results if available
signature_features <- NULL
if (file.exists(opt$smd_file)) {
  smd_results <- read.csv(opt$smd_file, stringsAsFactors = FALSE)
  
  # Get top features per modality
  top_features_per_modality <- smd_results %>%
    group_by(Modality) %>%
    slice_max(order_by = abs(SMD), n = 15) %>%
    ungroup()
  
  signature_features <- top_features_per_modality$Feature
  cat(sprintf("  Loaded %d signature features from SMD analysis\n", length(signature_features)))
}

# ==============================================================================
# Statistical Analysis with FDR Correction
# ==============================================================================
cat("\n[2/5] Statistical analysis (ANOVA/Kruskal-Wallis with FDR)...\n")

# Get feature columns
exclude_cols <- c("ID", "Cluster_Labels", "Cluster", "AD_Conversion", "Risk_Group")
feature_cols <- setdiff(colnames(full_data), exclude_cols)
feature_cols <- feature_cols[sapply(full_data[, feature_cols, drop = FALSE], is.numeric)]

# Filter to signature features if available
if (!is.null(signature_features)) {
  available_features <- feature_cols[feature_cols %in% signature_features]
  if (length(available_features) > 0) {
    feature_cols <- available_features
  }
}

# If still no features, use top variance features
if (length(feature_cols) == 0) {
  all_numeric <- full_data %>% select(where(is.numeric)) %>% select(-any_of(exclude_cols))
  feature_vars <- apply(all_numeric, 2, var, na.rm = TRUE)
  feature_cols <- names(sort(feature_vars, decreasing = TRUE))[1:min(30, length(feature_vars))]
}

cat(sprintf("  Analyzing %d features\n", length(feature_cols)))

# Statistical tests for each feature
stat_results <- data.frame()

for (feat in feature_cols) {
  values <- full_data[[feat]]
  groups <- full_data$Cluster
  
  # Skip if too many missing
  if (mean(is.na(values)) > 0.5) next
  
  # Median imputation
  values[is.na(values)] <- median(values, na.rm = TRUE)
  
  # Kruskal-Wallis test 
  kw_test <- tryCatch(
    kruskal.test(values ~ groups),
    error = function(e) list(statistic = NA, p.value = NA)
  )
  
  # ANOVA for comparison
  anova_test <- tryCatch({
    aov_result <- aov(values ~ groups)
    summary(aov_result)[[1]]
  }, error = function(e) NULL)
  
  anova_p <- if (!is.null(anova_test)) anova_test$`Pr(>F)`[1] else NA
  
  # Eta-squared 
  eta_sq <- calculate_eta_squared(values, groups)
  
  # Calculate cluster means
  cluster_means <- tapply(values, groups, mean, na.rm = TRUE)
  cluster_sds <- tapply(values, groups, sd, na.rm = TRUE)
  
  # Determine modality
  modality <- "Unknown"
  for (mod_name in names(file_mapping)) {
    if (grepl(paste0("^", mod_name, "_"), feat)) {
      modality <- mod_name
      break
    }
  }
  
  result_row <- data.frame(
    Feature = feat,
    Modality = modality,
    KW_Statistic = kw_test$statistic,
    KW_P_Value = kw_test$p.value,
    ANOVA_P_Value = anova_p,
    Eta_Squared = eta_sq,
    Eta_Interpretation = interpret_eta_squared(eta_sq),
    stringsAsFactors = FALSE
  )
  
  # Add cluster means
  for (cl in names(cluster_means)) {
    result_row[[paste0("Mean_Cluster", cl)]] <- cluster_means[cl]
    result_row[[paste0("SD_Cluster", cl)]] <- cluster_sds[cl]
  }
  
  stat_results <- rbind(stat_results, result_row)
}

# Apply Benjamini-Hochberg FDR correction
stat_results$P_adj_FDR <- p.adjust(stat_results$KW_P_Value, method = "BH")
stat_results$FDR_Significant <- stat_results$P_adj_FDR < opt$fdr_threshold

# Sort by effect size
stat_results <- stat_results %>%
  arrange(desc(Eta_Squared))

cat(sprintf("  FDR-significant features (q < %.2f): %d\n", 
            opt$fdr_threshold, sum(stat_results$FDR_Significant, na.rm = TRUE)))
cat(sprintf("  Large effect size (η² ≥ 0.14): %d\n", 
            sum(stat_results$Eta_Squared >= 0.14, na.rm = TRUE)))

write.csv(stat_results, file.path(opt$output_dir, "Cluster_Signature_Stats_FDR.csv"), row.names = FALSE)

# ==============================================================================
# Cluster Profile Summary
# ==============================================================================
cat("\n[3/5] Generating cluster profiles...\n")

# Prepare data for visualization
filtered_data <- full_data[, c("ID", "Cluster", feature_cols), drop = FALSE]

# Standardize features for visualization
for (col in feature_cols) {
  if (col %in% colnames(filtered_data)) {
    vals <- filtered_data[[col]]
    vals[is.na(vals)] <- median(vals, na.rm = TRUE)
    filtered_data[[col]] <- scale(vals)[, 1]
  }
}

# Calculate cluster means
plot_data <- filtered_data %>%
  pivot_longer(cols = -c(ID, Cluster), names_to = "Feature", values_to = "Value") %>%
  group_by(Cluster, Feature) %>%
  summarize(
    Mean_Value = mean(Value, na.rm = TRUE),
    SD_Value = sd(Value, na.rm = TRUE),
    N = n(),
    .groups = "drop"
  )

# Add statistical significance
plot_data <- plot_data %>%
  left_join(stat_results %>% select(Feature, Eta_Squared, P_adj_FDR, FDR_Significant), by = "Feature")

# Create cluster names based on risk
if ("AD_Conversion" %in% colnames(full_data)) {
  conv_rates <- full_data %>%
    group_by(Cluster) %>%
    summarise(conv_rate = mean(AD_Conversion, na.rm = TRUE), .groups = "drop") %>%
    arrange(desc(conv_rate))
  
  cluster_names <- setNames(
    paste0("Cluster ", conv_rates$Cluster, " (", 
           ifelse(conv_rates$conv_rate == max(conv_rates$conv_rate), "High-Risk", "Low-Risk"), ")"),
    conv_rates$Cluster
  )
} else {
  cluster_names <- setNames(paste0("Cluster ", unique(full_data$Cluster)), unique(full_data$Cluster))
}

plot_data$Cluster_Name <- cluster_names[as.character(plot_data$Cluster)]

# Save cluster profile data
cluster_profile_wide <- plot_data %>%
  select(Cluster, Feature, Mean_Value) %>%
  pivot_wider(names_from = Cluster, values_from = Mean_Value, names_prefix = "Cluster_")

write.csv(cluster_profile_wide, file.path(opt$output_dir, "Cluster_Signature_Wide.csv"), row.names = FALSE)

# ==============================================================================
# Visualizations
# ==============================================================================
cat("\n[4/5] Generating visualizations...\n")

# 4.1 Overall Signature Profile
cat("  Creating signature profile plot...\n")

# Order features by effect size
feature_order <- stat_results %>%
  arrange(desc(Eta_Squared)) %>%
  pull(Feature)

plot_data$Feature <- factor(plot_data$Feature, levels = feature_order)

p_profile <- ggplot(plot_data, aes(x = Feature, y = Mean_Value, 
                                    group = Cluster_Name, color = Cluster_Name)) +
  geom_point(size = 2.5, alpha = 0.8) +
  geom_line(linewidth = 1.2) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray40") +
  scale_color_brewer(palette = "Set1") +
  labs(
    title = "Cluster Signature Profiles",
    subtitle = "Features ordered by effect size (η²)",
    x = "Feature", 
    y = "Mean Standardized Value", 
    color = "Cluster"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1, size = 8),
    legend.position = "top",
    plot.title = element_text(face = "bold")
  )

ggsave(file.path(opt$output_dir, "Signature_Profile_Overall.png"), 
       plot = p_profile, width = 16, height = 10, dpi = 300)

# 4.2 Per-modality profiles
for (modality in names(file_mapping)) {
  modality_features <- feature_cols[grepl(paste0("^", modality, "_"), feature_cols)]
  
  if (length(modality_features) == 0) next
  
  modality_data <- plot_data %>% 
    filter(Feature %in% modality_features)
  
  if (nrow(modality_data) == 0) next
  
  # Order by effect size within modality
  modality_order <- stat_results %>%
    filter(Feature %in% modality_features) %>%
    arrange(desc(Eta_Squared)) %>%
    pull(Feature)
  
  modality_data$Feature <- factor(modality_data$Feature, levels = modality_order)
  modality_data$Feature_Clean <- gsub(paste0("^", modality, "_"), "", modality_data$Feature)
  
  p_modality <- ggplot(modality_data, aes(x = Feature_Clean, y = Mean_Value, 
                                           group = Cluster_Name, color = Cluster_Name)) +
    geom_point(size = 3, alpha = 0.8) +
    geom_line(linewidth = 1.3) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "gray40") +
    scale_color_brewer(palette = "Set1") +
    labs(
      title = sprintf("%s Signature Profile", modality),
      x = "Feature", 
      y = "Mean Standardized Value", 
      color = "Cluster"
    ) +
    theme_minimal() +
    theme(
      axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),
      legend.position = "top",
      plot.title = element_text(face = "bold")
    )
  
  ggsave(file.path(opt$output_dir, sprintf("Signature_Profile_%s.png", modality)), 
         plot = p_modality, width = 12, height = 8, dpi = 300)
}

# 4.3 Heatmap with FDR significance
cat("  Creating signature heatmap...\n")

# Select top features by effect size
top_features <- stat_results %>%
  filter(FDR_Significant | Eta_Squared >= 0.06) %>%
  arrange(desc(Eta_Squared)) %>%
  head(50) %>%
  pull(Feature)

if (length(top_features) < 5) {
  top_features <- head(stat_results$Feature, 30)
}

heatmap_data <- cluster_profile_wide %>%
  filter(Feature %in% top_features) %>%
  column_to_rownames("Feature")

# Annotation for significance
row_annotation <- stat_results %>%
  filter(Feature %in% top_features) %>%
  select(Feature, Eta_Interpretation, FDR_Significant) %>%
  column_to_rownames("Feature")

row_annotation$Significance <- ifelse(row_annotation$FDR_Significant, "FDR < 0.05", "NS")

ann_colors <- list(
  Eta_Interpretation = c("Large" = "#d62728", "Medium" = "#ff7f0e", 
                          "Small" = "#2ca02c", "Negligible" = "grey70"),
  Significance = c("FDR < 0.05" = "darkgreen", "NS" = "grey80")
)

png(file.path(opt$output_dir, "Signature_Heatmap_FDR.png"), width = 2400, height = 3200, res = 300)
pheatmap(
  as.matrix(heatmap_data),
  color = colorRampPalette(rev(brewer.pal(11, "RdBu")))(100),
  scale = "row",
  cluster_cols = FALSE,
  cluster_rows = TRUE,
  annotation_row = row_annotation[, c("Eta_Interpretation", "Significance"), drop = FALSE],
  annotation_colors = ann_colors,
  show_rownames = TRUE,
  show_colnames = TRUE,
  fontsize_row = 7,
  main = "Cluster Signature Heatmap (FDR-corrected)",
  breaks = seq(-2, 2, length.out = 101)
)
dev.off()

# 4.4 Effect size distribution
cat("  Creating effect size plot...\n")

p_effect <- ggplot(stat_results, aes(x = Eta_Squared, fill = Eta_Interpretation)) +
  geom_histogram(bins = 30, alpha = 0.8, color = "white") +
  geom_vline(xintercept = c(0.01, 0.06, 0.14), linetype = "dashed", color = "red") +
  scale_fill_manual(values = c("Large" = "#d62728", "Medium" = "#ff7f0e", 
                                "Small" = "#2ca02c", "Negligible" = "grey70")) +
  labs(
    title = "Distribution of Effect Sizes (η²)",
    subtitle = "Cohen's criteria (0.01 small, 0.06 medium, 0.14 large)",
    x = "Eta-squared (η²)",
    y = "Count",
    fill = "Effect Size"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"))

ggsave(file.path(opt$output_dir, "Effect_Size_Distribution_Clusters.png"), 
       plot = p_effect, width = 10, height = 6, dpi = 300)

# ==============================================================================
# Summary Report
# ==============================================================================
cat("\n[5/5] Generating summary report...\n")

# Summary by modality
modality_summary <- stat_results %>%
  group_by(Modality) %>%
  summarise(
    N_Features = n(),
    N_FDR_Significant = sum(FDR_Significant, na.rm = TRUE),
    N_Large_Effect = sum(Eta_Squared >= 0.14, na.rm = TRUE),
    N_Medium_Effect = sum(Eta_Squared >= 0.06 & Eta_Squared < 0.14, na.rm = TRUE),
    Mean_Eta_Squared = mean(Eta_Squared, na.rm = TRUE),
    .groups = "drop"
  )

print(modality_summary)
write.csv(modality_summary, file.path(opt$output_dir, "Cluster_Signature_Summary.csv"), row.names = FALSE)

cat("\n========================================================================\n")
cat("Cluster Signature Analysis Complete!\n")
cat("========================================================================\n\n")

cat("\nKey Findings:\n")
cat(sprintf("  Total features analyzed: %d\n", nrow(stat_results)))
cat(sprintf("  FDR-significant: %d\n", sum(stat_results$FDR_Significant, na.rm = TRUE)))
cat(sprintf("  Large effect (η² ≥ 0.14): %d\n", sum(stat_results$Eta_Squared >= 0.14, na.rm = TRUE)))
cat(sprintf("  Medium effect (η² ≥ 0.06): %d\n", sum(stat_results$Eta_Squared >= 0.06, na.rm = TRUE)))

cat("\nOutput files:\n")
cat("  - Cluster_Signature_Stats_FDR.csv\n")
cat("  - Cluster_Signature_Wide.csv\n")
cat("  - Cluster_Signature_Summary.csv\n")
cat("  - Signature_Profile_Overall.png\n")
cat("  - Signature_Heatmap_FDR.png\n")
cat("  - Effect_Size_Distribution_Clusters.png\n")

cat("\n========================================================================\n")

