library(limma)
library(dplyr)
library(tidyr)
library(ggplot2)
library(ggrepel)
library(pheatmap)
library(RColorBrewer)
library(optparse)

# ==============================================================================
# Parse Command Line Arguments
# ==============================================================================
option_list <- list(
  make_option(c("--outcome_file"), type = "character", default = "outcome_data.csv",
              help = "Path to outcome data CSV with AD_Conversion column [default: %default]"),
  make_option(c("--clinical_file"), type = "character", default = "Clinical_data.csv",
              help = "Path to clinical data CSV [default: %default]"),
  make_option(c("--smri_file"), type = "character", default = "sMRI_data.csv",
              help = "Path to sMRI data CSV [default: %default]"),
  make_option(c("--csf_file"), type = "character", default = "CSF_data.csv",
              help = "Path to CSF data CSV [default: %default]"),
  make_option(c("--output_dir"), type = "character", default = "./results",
              help = "Output directory [default: %default]"),
  make_option(c("--fdr_threshold"), type = "numeric", default = 0.05,
              help = "FDR significance threshold (q < 0.05) [default: %default]"),
  make_option(c("--smd_threshold"), type = "numeric", default = 0.5,
              help = "SMD clinical meaningfulness threshold (|SMD| > 0.5) [default: %default]")
)

opt_parser <- OptionParser(option_list = option_list)
opt <- parse_args(opt_parser)

# Create output directory
dir.create(opt$output_dir, showWarnings = FALSE, recursive = TRUE)

#' Calculate Cohen's d (Standardized Mean Difference)
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

#' Calculate Eta-squared effect size
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
cat("[1/5] Loading data...\n")

# Load outcome data
if (!file.exists(opt$outcome_file)) {
  stop(sprintf("Outcome file not found: %s", opt$outcome_file))
}

outcome <- read.csv(opt$outcome_file, stringsAsFactors = FALSE)
outcome$ID <- as.character(outcome$ID)

if (!"AD_Conversion" %in% colnames(outcome)) {
  stop("AD_Conversion column not found in outcome file!")
}

outcome$AD_Conversion <- as.numeric(outcome$AD_Conversion)

n_converters <- sum(outcome$AD_Conversion == 1, na.rm = TRUE)
n_nonconverters <- sum(outcome$AD_Conversion == 0, na.rm = TRUE)

cat(sprintf("  Converters: %d\n", n_converters))
cat(sprintf("  Non-converters: %d\n", n_nonconverters))

# Load modality data
all_data <- outcome

if (file.exists(opt$clinical_file)) {
  clinical <- read.csv(opt$clinical_file, stringsAsFactors = FALSE)
  clinical$ID <- as.character(clinical$ID)
  all_data <- all_data %>% inner_join(clinical, by = "ID")
  cat(sprintf("  Loaded Clinical: %d features\n", ncol(clinical) - 1))
}

if (file.exists(opt$smri_file)) {
  smri <- read.csv(opt$smri_file, stringsAsFactors = FALSE)
  smri$ID <- as.character(smri$ID)
  all_data <- all_data %>% inner_join(smri, by = "ID")
  cat(sprintf("  Loaded sMRI: %d features\n", ncol(smri) - 1))
}

if (file.exists(opt$csf_file)) {
  csf <- read.csv(opt$csf_file, stringsAsFactors = FALSE)
  csf$ID <- as.character(csf$ID)
  all_data <- all_data %>% inner_join(csf, by = "ID")
  cat(sprintf("  Loaded CSF: %d features\n", ncol(csf) - 1))
}

# Filter to complete cases for AD_Conversion
all_data <- all_data %>% filter(!is.na(AD_Conversion))
cat(sprintf("\n  Final sample size: %d\n", nrow(all_data)))

# ==============================================================================
# Differential Analysis: Converters vs Non-Converters
# ==============================================================================
cat("\n[2/5] Differential analysis with FDR correction...\n")

# Get feature columns
exclude_cols <- c("ID", "AD_Conversion", "Cluster_Labels", "Risk_Group", "Time_to_Event")
feature_cols <- setdiff(colnames(all_data), exclude_cols)
feature_cols <- feature_cols[sapply(all_data[, feature_cols, drop = FALSE], is.numeric)]

cat(sprintf("  Analyzing %d features\n", length(feature_cols)))

# Determine modality for each feature
get_modality <- function(feat) {
  if (grepl("^Clinical_", feat)) return("Clinical")
  if (grepl("^sMRI_", feat)) return("sMRI")
  if (grepl("^CSF_", feat)) return("CSF")
  return("Unknown")
}

# Differential analysis
all_diff_results <- data.frame()

for (feat in feature_cols) {
  values <- all_data[[feat]]
  
  # Skip if too many missing
  if (mean(is.na(values)) > 0.5) next
  
  # Median imputation
  values[is.na(values)] <- median(values, na.rm = TRUE)
  
  converter_vals <- values[all_data$AD_Conversion == 1]
  nonconverter_vals <- values[all_data$AD_Conversion == 0]
  
  # Wilcoxon rank-sum test 
  wilcox_test <- tryCatch(
    wilcox.test(converter_vals, nonconverter_vals),
    error = function(e) list(statistic = NA, p.value = NA)
  )
  
  # Calculate SMD (Cohen's d)
  smd <- calculate_cohens_d(converter_vals, nonconverter_vals)
  
  # Calculate Eta-squared
  eta_sq <- calculate_eta_squared(values, all_data$AD_Conversion)
  
  # Calculate log fold change (approximation)
  mean_conv <- mean(converter_vals, na.rm = TRUE)
  mean_nonconv <- mean(nonconverter_vals, na.rm = TRUE)
  
  # Avoid log of zero/negative
  if (mean_conv > 0 && mean_nonconv > 0) {
    logFC <- log2(mean_conv / mean_nonconv)
  } else {
    logFC <- smd  # Use SMD as proxy
  }
  
  all_diff_results <- rbind(all_diff_results, data.frame(
    Feature = feat,
    Modality = get_modality(feat),
    Mean_Converter = mean_conv,
    Mean_NonConverter = mean_nonconv,
    SD_Converter = sd(converter_vals, na.rm = TRUE),
    SD_NonConverter = sd(nonconverter_vals, na.rm = TRUE),
    logFC = logFC,
    SMD = smd,
    SMD_Interpretation = interpret_cohens_d(smd),
    Eta_Squared = eta_sq,
    Eta_Interpretation = interpret_eta_squared(eta_sq),
    P_Value = wilcox_test$p.value,
    Direction = ifelse(smd > 0, "Up in Converters", "Down in Converters"),
    stringsAsFactors = FALSE
  ))
}

# Apply Benjamini-Hochberg FDR correction ( q < 0.05)
all_diff_results$P_adj_FDR <- p.adjust(all_diff_results$P_Value, method = "BH")

# Determine significance
all_diff_results$FDR_Significant <- all_diff_results$P_adj_FDR < opt$fdr_threshold
all_diff_results$Clinically_Meaningful <- abs(all_diff_results$SMD) > opt$smd_threshold
all_diff_results$Both_Criteria <- all_diff_results$FDR_Significant & all_diff_results$Clinically_Meaningful

# Sort by significance and effect size
all_diff_results <- all_diff_results %>%
  arrange(P_adj_FDR, desc(abs(SMD)))

cat(sprintf("  FDR significant (q < %.2f): %d\n", opt$fdr_threshold, sum(all_diff_results$FDR_Significant, na.rm = TRUE)))
cat(sprintf("  Clinically meaningful (|SMD| > %.1f): %d\n", opt$smd_threshold, sum(all_diff_results$Clinically_Meaningful, na.rm = TRUE)))
cat(sprintf("  Both criteria met: %d\n", sum(all_diff_results$Both_Criteria, na.rm = TRUE)))

# ==============================================================================
# Retain AD Core Biomarkers
# ==============================================================================
cat("\n[3/5] Identifying AD core biomarkers...\n")

# Define AD core biomarker patterns
ad_core_patterns <- c("ABETA", "TAU", "PTAU", "ADAS", "MMSE", "FAQ", "APOE",
                      "Hippocampus", "Entorhinal", "Amygdala", "Ventricle", 
                      "CDR", "Temporal", "Parahippocampal")

# Identify core features
all_diff_results$Is_AD_Core <- sapply(all_diff_results$Feature, function(feat) {
  any(sapply(ad_core_patterns, function(pattern) {
    grepl(pattern, feat, ignore.case = TRUE)
  }))
})

# Filter significant features
significant_features <- all_diff_results %>%
  filter(Both_Criteria) %>%
  arrange(P_adj_FDR)

# Add core features that didn't meet strict criteria but are biologically important
ad_core_features <- all_diff_results %>%
  filter(Is_AD_Core & !Both_Criteria & P_adj_FDR < 0.1)

if (nrow(ad_core_features) > 0) {
  significant_features_enhanced <- rbind(significant_features, ad_core_features) %>%
    arrange(P_adj_FDR)
  cat(sprintf("  Added %d AD core features with relaxed criteria\n", nrow(ad_core_features)))
} else {
  significant_features_enhanced <- significant_features
}

cat(sprintf("  Total significant features: %d\n", nrow(significant_features_enhanced)))

# Save results
write.csv(all_diff_results, file.path(opt$output_dir, "Conversion_Differential_All_FDR.csv"), row.names = FALSE)
write.csv(significant_features_enhanced, file.path(opt$output_dir, "Conversion_Differential_Significant_FDR.csv"), row.names = FALSE)

# ==============================================================================
# Visualizations
# ==============================================================================
cat("\n[4/5] Generating visualizations...\n")

# 4.1 Volcano Plot with FDR threshold
cat("  Creating volcano plot...\n")

all_diff_results$Label <- ""
top_features <- all_diff_results %>%
  arrange(P_adj_FDR) %>%
  head(15)
all_diff_results$Label[match(top_features$Feature, all_diff_results$Feature)] <- top_features$Feature

all_diff_results$Significance <- case_when(
  all_diff_results$Both_Criteria & all_diff_results$SMD > 0 ~ "Up (Significant)",
  all_diff_results$Both_Criteria & all_diff_results$SMD < 0 ~ "Down (Significant)",
  all_diff_results$Is_AD_Core ~ "AD Core Marker",
  TRUE ~ "NS"
)

p_volcano <- ggplot(all_diff_results, aes(x = SMD, y = -log10(P_adj_FDR), 
                                           color = Significance, label = Label)) +
  geom_point(aes(size = abs(SMD)), alpha = 0.7) +
  scale_size_continuous(range = c(1, 4), name = "|SMD|") +
  scale_color_manual(values = c(
    "Up (Significant)" = "#d62728",
    "Down (Significant)" = "#2ca02c",
    "AD Core Marker" = "#ff7f0e",
    "NS" = "grey70"
  )) +
  geom_hline(yintercept = -log10(opt$fdr_threshold), linetype = "dashed", color = "blue") +
  geom_vline(xintercept = c(-opt$smd_threshold, opt$smd_threshold), linetype = "dashed", color = "blue") +
  geom_text_repel(size = 3, max.overlaps = 15) +
  labs(
    title = "Differential Analysis: AD Converters vs Non-Converters",
    subtitle = sprintf("FDR threshold: q < %.2f, SMD threshold: |SMD| > %.1f ", 
                       opt$fdr_threshold, opt$smd_threshold),
    x = "Standardized Mean Difference (Cohen's d)",
    y = expression(-log[10]~"(FDR-adjusted P-value)")
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5)
  )

ggsave(file.path(opt$output_dir, "Volcano_Plot_Conversion_FDR.png"), 
       plot = p_volcano, width = 12, height = 10, dpi = 300)

# 4.2 Feature Count Summary by Modality
cat("  Creating feature count summary...\n")

if (nrow(significant_features_enhanced) > 0) {
  summary_data <- significant_features_enhanced %>%
    group_by(Modality, Direction) %>%
    summarise(Count = n(), .groups = "drop")
  
  p_summary <- ggplot(summary_data, aes(x = Modality, y = Count, fill = Direction)) +
    geom_bar(stat = "identity", position = "dodge") +
    scale_fill_manual(values = c("Up in Converters" = "#d62728", "Down in Converters" = "#2ca02c")) +
    geom_text(aes(label = Count), position = position_dodge(width = 0.9), vjust = -0.5) +
    labs(
      title = "Significant Features by Modality",
      subtitle = sprintf("FDR < %.2f AND |SMD| > %.1f", opt$fdr_threshold, opt$smd_threshold),
      x = "Modality",
      y = "Number of Features"
    ) +
    theme_minimal() +
    theme(plot.title = element_text(face = "bold"))
  
  ggsave(file.path(opt$output_dir, "Feature_Count_Summary_FDR.png"), 
         plot = p_summary, width = 10, height = 6, dpi = 300)
}

# 4.3 Top SMD Features
cat("  Creating top SMD features plot...\n")

top_smd <- all_diff_results %>%
  arrange(desc(abs(SMD))) %>%
  head(30)

top_smd$Significant <- ifelse(top_smd$Both_Criteria, "Significant", "Not Significant")

p_top_smd <- ggplot(top_smd, aes(x = reorder(Feature, abs(SMD)), y = SMD, 
                                  fill = Modality, alpha = Significant)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  scale_alpha_manual(values = c("Significant" = 1.0, "Not Significant" = 0.5)) +
  geom_hline(yintercept = c(-opt$smd_threshold, opt$smd_threshold), 
             linetype = "dashed", color = "red") +
  labs(
    title = "Top 30 Features by Effect Size (SMD)",
    subtitle = sprintf("Red lines: |SMD| = %.1f (clinical meaningfulness threshold)", opt$smd_threshold),
    x = "Feature",
    y = "SMD (Cohen's d)"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"))

ggsave(file.path(opt$output_dir, "Top30_SMD_Conversion_FDR.png"), 
       plot = p_top_smd, width = 12, height = 10, dpi = 300)

# 4.4 Effect Size Distribution
cat("  Creating effect size distribution...\n")

p_effect <- ggplot(all_diff_results, aes(x = Eta_Squared, fill = Eta_Interpretation)) +
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

ggsave(file.path(opt$output_dir, "Effect_Size_Distribution_Conversion.png"), 
       plot = p_effect, width = 10, height = 6, dpi = 300)

# ==============================================================================
# Overlap Analysis with Cluster Differential Features
# ==============================================================================
cat("\n[5/5] Checking overlap with cluster differential features...\n")

cluster_diff_file <- file.path(opt$output_dir, "SMD_Significant_Features_FDR.csv")
if (file.exists(cluster_diff_file)) {
  cluster_diff <- read.csv(cluster_diff_file, stringsAsFactors = FALSE)
  
  overlap <- intersect(significant_features_enhanced$Feature, cluster_diff$Feature)
  
  if (length(overlap) > 0) {
    cat(sprintf("  Found %d overlapping features\n", length(overlap)))
    
    overlap_df <- significant_features_enhanced %>%
      filter(Feature %in% overlap) %>%
      select(Feature, Modality, SMD, P_adj_FDR, Direction)
    
    write.csv(overlap_df, file.path(opt$output_dir, "Overlapping_Features_Cluster_Conversion.csv"), row.names = FALSE)
  } else {
    cat("  No overlapping features found\n")
  }
} else {
  cat("  Cluster differential file not found, skipping overlap analysis\n")
}

# ==============================================================================
# Summary Report
# ==============================================================================
cat("\n========================================================================\n")
cat("Conversion Differential Analysis Complete!\n")
cat("========================================================================\n\n")
cat("\nKey Findings:\n")
cat(sprintf("  Total features analyzed: %d\n", nrow(all_diff_results)))
cat(sprintf("  FDR significant (q < %.2f): %d\n", opt$fdr_threshold, sum(all_diff_results$FDR_Significant, na.rm = TRUE)))
cat(sprintf("  Clinically meaningful (|SMD| > %.1f): %d\n", opt$smd_threshold, sum(all_diff_results$Clinically_Meaningful, na.rm = TRUE)))
cat(sprintf("  Both criteria met: %d\n", sum(all_diff_results$Both_Criteria, na.rm = TRUE)))
cat(sprintf("  AD core biomarkers identified: %d\n", sum(all_diff_results$Is_AD_Core, na.rm = TRUE)))

# Summary by modality
modality_summary <- all_diff_results %>%
  group_by(Modality) %>%
  summarise(
    N_Features = n(),
    N_FDR_Significant = sum(FDR_Significant, na.rm = TRUE),
    N_Clinically_Meaningful = sum(Clinically_Meaningful, na.rm = TRUE),
    N_Both = sum(Both_Criteria, na.rm = TRUE),
    .groups = "drop"
  )

cat("\nSummary by Modality:\n")
print(modality_summary)

write.csv(modality_summary, file.path(opt$output_dir, "Conversion_Differential_Summary.csv"), row.names = FALSE)

cat("\nOutput files:\n")
cat("  - Conversion_Differential_All_FDR.csv\n")
cat("  - Conversion_Differential_Significant_FDR.csv\n")
cat("  - Conversion_Differential_Summary.csv\n")
cat("  - Volcano_Plot_Conversion_FDR.png\n")
cat("  - Feature_Count_Summary_FDR.png\n")
cat("  - Top30_SMD_Conversion_FDR.png\n")
cat("  - Effect_Size_Distribution_Conversion.png\n")

cat("\n========================================================================\n")

