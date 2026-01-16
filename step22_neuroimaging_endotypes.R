library(tidyverse)
library(ggplot2)
library(patchwork)
library(optparse)

# ==============================================================================
# Parse Command Line Arguments
# ==============================================================================
option_list <- list(
  make_option(c("--data_file"), type = "character", 
              default = "ADNI_Labeled_For_Classifier.csv",
              help = "Path to labeled data CSV [default: %default]"),
  make_option(c("--output_dir"), type = "character", 
              default = "./results",
              help = "Output directory [default: %default]"),
  make_option(c("--fdr_threshold"), type = "numeric", default = 0.05,
              help = "FDR significance threshold ")
)

opt_parser <- OptionParser(option_list = option_list)
opt <- parse_args(opt_parser)

# Create output directory
dir.create(opt$output_dir, showWarnings = FALSE, recursive = TRUE)

# ==============================================================================
# Part 1: Data Loading and Preparation
# ==============================================================================
cat("[1/6] Loading and preparing data...\n")

data_raw <- read.csv(opt$data_file, stringsAsFactors = FALSE)
cat(sprintf("  Raw data: %d samples\n", nrow(data_raw)))

# Define feature groups
clinical_features <- c("ADAS13", "CDRSB", "FAQTOTAL", "MMSE_Baseline")
clinical_features <- clinical_features[clinical_features %in% colnames(data_raw)]

# MRI features - map ST codes to anatomical names
mri_mapping <- list(
  "ST102TA" = "RightParacentral_TA",
  "ST102CV" = "RightParacentral_CV",
  "ST103TA" = "RightParahippocampal_TA",
  "ST103CV" = "RightParahippocampal_CV",
  "ST104TA" = "RightParsOpercularis_TA",
  "ST105TA" = "RightParsOrbitalis_TA",
  "ST105CV" = "RightParsOrbitalis_CV"
)

# Rename columns if ST codes exist
data <- data_raw
for (st_code in names(mri_mapping)) {
  if (st_code %in% colnames(data)) {
    data <- data %>% rename(!!mri_mapping[[st_code]] := !!st_code)
  }
}

mri_features <- unlist(mri_mapping)
mri_features <- mri_features[mri_features %in% colnames(data)]

covariates <- c("MMSE_Baseline", "Age")
covariates <- covariates[covariates %in% colnames(data)]

# Select and clean data
all_features <- c("ID", "Subtype", clinical_features, mri_features, covariates, "Gender")
all_features <- all_features[all_features %in% colnames(data)]

data <- data %>%
  select(all_of(all_features)) %>%
  filter(!is.na(Subtype)) %>%
  drop_na(any_of(c(clinical_features, mri_features)))

# Calculate global composite metrics for disproportionate atrophy analysis
ta_features <- mri_features[str_detect(mri_features, "_TA$")]
cv_features <- mri_features[str_detect(mri_features, "_CV$")]

if (length(ta_features) > 0) {
  data$Global_TA_Composite <- rowMeans(data[, ta_features, drop = FALSE], na.rm = TRUE)
}
if (length(cv_features) > 0) {
  data$Global_CV_Composite <- rowMeans(data[, cv_features, drop = FALSE], na.rm = TRUE)
}

cat(sprintf("  Complete data: %d samples\n", nrow(data)))
cat(sprintf("  Clinical features: %d\n", length(clinical_features)))
cat(sprintf("  MRI features: %d\n", length(mri_features)))

# Endotype distribution
cat("\n  Endotype distribution:\n")
subtype_dist <- table(data$Subtype)
for (i in 1:length(subtype_dist)) {
  cat(sprintf("    Subtype %s: %d (%.1f%%)\n",
              names(subtype_dist)[i],
              subtype_dist[i],
              100 * subtype_dist[i] / sum(subtype_dist)))
}
cat("\n")

# ==============================================================================
# Helper Function: Interpret Eta-squared 
# ==============================================================================
interpret_eta_squared <- function(eta2) {
  if (is.na(eta2)) return("NA")
  if (eta2 >= 0.14) return("Large")
  if (eta2 >= 0.06) return("Medium")
  if (eta2 >= 0.01) return("Small")
  return("Negligible")
}

# ==============================================================================
# Part 2: Clinical Homogeneity Test 
# ==============================================================================
cat("[2/6] Clinical homogeneity test ...\n")

clinical_homogeneity <- data.frame()

for (feat in clinical_features) {
  if (!feat %in% colnames(data)) next
  
  formula <- paste0(feat, " ~ factor(Subtype)")
  model <- aov(as.formula(formula), data = data)
  sum_model <- summary(model)
  
  f_val <- sum_model[[1]]$`F value`[1]
  p_val <- sum_model[[1]]$`Pr(>F)`[1]
  ss_between <- sum_model[[1]]$`Sum Sq`[1]
  ss_total <- sum(sum_model[[1]]$`Sum Sq`)
  eta2 <- ss_between / ss_total
  
  clinical_homogeneity <- rbind(clinical_homogeneity, data.frame(
    Feature = feat,
    F_value = f_val,
    P_Raw = p_val,
    Eta_squared = eta2,
    Eta_Interpretation = interpret_eta_squared(eta2),
    stringsAsFactors = FALSE
  ))
}

# FDR correction 
clinical_homogeneity$P_FDR <- p.adjust(clinical_homogeneity$P_Raw, method = "fdr")
clinical_homogeneity$Significant_FDR <- clinical_homogeneity$P_FDR < opt$fdr_threshold

cat("  Clinical homogeneity results:\n")
for (i in 1:nrow(clinical_homogeneity)) {
  sig_marker <- ifelse(clinical_homogeneity$Significant_FDR[i], "*", "")
  cat(sprintf("    %s: F=%.2f, p_FDR=%.4f, η²=%.3f (%s)%s\n",
              clinical_homogeneity$Feature[i],
              clinical_homogeneity$F_value[i],
              clinical_homogeneity$P_FDR[i],
              clinical_homogeneity$Eta_squared[i],
              clinical_homogeneity$Eta_Interpretation[i],
              sig_marker))
}

write.csv(clinical_homogeneity, 
          file.path(opt$output_dir, "Clinical_Homogeneity_Results.csv"), 
          row.names = FALSE)

# Clinical homogeneity plot
p_clinical <- clinical_homogeneity %>%
  ggplot(aes(x = reorder(Feature, Eta_squared), y = Eta_squared, 
             fill = Significant_FDR)) +
  geom_bar(stat = "identity", alpha = 0.8) +
  geom_hline(yintercept = 0.01, linetype = "dashed", color = "orange", size = 0.8) +
  geom_hline(yintercept = 0.06, linetype = "dashed", color = "red", size = 0.8) +
  geom_hline(yintercept = 0.14, linetype = "dashed", color = "darkred", size = 0.8) +
  geom_text(aes(label = sprintf("p=%.3f", P_FDR)), hjust = -0.1, size = 3.5) +
  scale_fill_manual(values = c("FALSE" = "#95B3D7", "TRUE" = "#C0504D"),
                    name = "FDR < 0.05") +
  coord_flip() +
  labs(
    title = "Clinical Homogeneity Across Endotypes ",
    subtitle = "Dashed lines: Cohen's η² thresholds (0.01 small, 0.06 medium, 0.14 large)",
    x = "Clinical Feature", 
    y = "Effect Size (η²)"
  ) +
  theme_classic(base_size = 12) +
  theme(plot.title = element_text(face = "bold", size = 14),
        legend.position = "bottom")

ggsave(file.path(opt$output_dir, "Figure_Clinical_Homogeneity.png"), 
       p_clinical, width = 10, height = 6, dpi = 300)
cat("  Saved: Figure_Clinical_Homogeneity.png\n\n")

# ==============================================================================
# Part 3: MRI Heterogeneity Test 
# ==============================================================================
cat("[3/6] MRI heterogeneity test")

mri_heterogeneity <- data.frame()

for (feat in mri_features) {
  if (!feat %in% colnames(data)) next
  
  formula <- paste0(feat, " ~ factor(Subtype)")
  model <- aov(as.formula(formula), data = data)
  sum_model <- summary(model)
  
  f_val <- sum_model[[1]]$`F value`[1]
  p_val <- sum_model[[1]]$`Pr(>F)`[1]
  ss_between <- sum_model[[1]]$`Sum Sq`[1]
  ss_total <- sum(sum_model[[1]]$`Sum Sq`)
  eta2 <- ss_between / ss_total
  
  # Extract region and measure type
  region_name <- str_extract(feat, "^[A-Za-z]+")
  measure_type <- str_extract(feat, "[A-Z]+$")
  
  mri_heterogeneity <- rbind(mri_heterogeneity, data.frame(
    Feature = feat,
    Region = region_name,
    Measure = measure_type,
    F_value = f_val,
    P_Raw = p_val,
    Eta_squared = eta2,
    Eta_Interpretation = interpret_eta_squared(eta2),
    stringsAsFactors = FALSE
  ))
}

# FDR correction 
mri_heterogeneity$P_FDR <- p.adjust(mri_heterogeneity$P_Raw, method = "fdr")
mri_heterogeneity$Significant_FDR <- mri_heterogeneity$P_FDR < opt$fdr_threshold

cat("  MRI heterogeneity results:\n")
n_sig <- sum(mri_heterogeneity$Significant_FDR)
n_large <- sum(mri_heterogeneity$Eta_Interpretation == "Large")
cat(sprintf("    Significant (FDR < %.2f): %d/%d\n", opt$fdr_threshold, n_sig, nrow(mri_heterogeneity)))
cat(sprintf("    Large effect (η² ≥ 0.14): %d/%d\n", n_large, nrow(mri_heterogeneity)))

write.csv(mri_heterogeneity, 
          file.path(opt$output_dir, "MRI_Heterogeneity_Results.csv"), 
          row.names = FALSE)

# MRI heterogeneity plot
p_mri <- mri_heterogeneity %>%
  ggplot(aes(x = reorder(Feature, Eta_squared), y = Eta_squared, 
             fill = Significant_FDR)) +
  geom_bar(stat = "identity", alpha = 0.8) +
  geom_hline(yintercept = 0.01, linetype = "dashed", color = "orange", size = 0.8) +
  geom_hline(yintercept = 0.06, linetype = "dashed", color = "red", size = 0.8) +
  geom_hline(yintercept = 0.14, linetype = "dashed", color = "darkred", size = 0.8) +
  geom_text(aes(label = sprintf("%.3f", Eta_squared)), hjust = -0.1, size = 3) +
  scale_fill_manual(values = c("FALSE" = "#95B3D7", "TRUE" = "#C0504D"),
                    name = "FDR < 0.05") +
  coord_flip() +
  labs(
    title = "MRI Heterogeneity Across Endotypes ",
    subtitle = "Dashed lines: Cohen's η² thresholds (0.01 small, 0.06 medium, 0.14 large)",
    x = "MRI Feature", 
    y = "Effect Size (η²)"
  ) +
  theme_classic(base_size = 12) +
  theme(plot.title = element_text(face = "bold", size = 14),
        legend.position = "bottom")

ggsave(file.path(opt$output_dir, "Figure_MRI_Heterogeneity.png"), 
       p_mri, width = 10, height = 7, dpi = 300)
cat("  Saved: Figure_MRI_Heterogeneity.png\n\n")

# ==============================================================================
# Part 4: Stage Independence Test 
# ==============================================================================
cat("[4/6] Stage independence test (ANCOVA)...\n")

stage_independence <- data.frame()

for (feat in mri_features) {
  if (!feat %in% colnames(data)) next
  if (!"MMSE_Baseline" %in% colnames(data) || !"Age" %in% colnames(data)) next
  
  # Full model with covariates (ANCOVA)
  formula_full <- paste0(feat, " ~ factor(Subtype) + MMSE_Baseline + Age")
  model_full <- aov(as.formula(formula_full), data = data)
  sum_full <- summary(model_full)
  
  # Unadjusted model
  formula_sub <- paste0(feat, " ~ factor(Subtype)")
  model_sub <- aov(as.formula(formula_sub), data = data)
  sum_sub <- summary(model_sub)
  
  # Adjusted eta-squared (from ANCOVA)
  ss_subtype_adj <- sum_full[[1]]$`Sum Sq`[1]
  ss_total_adj <- sum(sum_full[[1]]$`Sum Sq`)
  eta2_adjusted <- ss_subtype_adj / ss_total_adj
  p_val_adjusted <- sum_full[[1]]$`Pr(>F)`[1]
  
  # Unadjusted eta-squared
  ss_sub_only <- sum_sub[[1]]$`Sum Sq`[1]
  ss_total_sub <- sum(sum_sub[[1]]$`Sum Sq`)
  eta2_unadjusted <- ss_sub_only / ss_total_sub
  
  # Effect size change
  eta2_change <- eta2_adjusted - eta2_unadjusted
  pct_change <- 100 * eta2_change / eta2_unadjusted
  
  stage_independence <- rbind(stage_independence, data.frame(
    Feature = feat,
    Eta2_Unadjusted = eta2_unadjusted,
    Eta2_Adjusted = eta2_adjusted,
    Eta2_Change = eta2_change,
    Pct_Change = pct_change,
    P_Adjusted = p_val_adjusted,
    Eta_Interpretation_Adj = interpret_eta_squared(eta2_adjusted),
    stringsAsFactors = FALSE
  ))
}

# FDR correction for adjusted p-values
stage_independence$P_FDR_Adjusted <- p.adjust(stage_independence$P_Adjusted, method = "fdr")
stage_independence$Stage_Independent <- stage_independence$P_FDR_Adjusted < opt$fdr_threshold

cat("  Stage independence results:\n")
n_independent <- sum(stage_independence$Stage_Independent)
cat(sprintf("    Stage-independent (FDR < %.2f after adjustment): %d/%d\n", 
            opt$fdr_threshold, n_independent, nrow(stage_independence)))

write.csv(stage_independence, 
          file.path(opt$output_dir, "Stage_Independence_Results.csv"), 
          row.names = FALSE)

# Stage independence plot
p_stage <- stage_independence %>%
  pivot_longer(cols = c(Eta2_Unadjusted, Eta2_Adjusted),
               names_to = "Type", values_to = "Eta2") %>%
  mutate(Type = factor(Type, 
                       levels = c("Eta2_Unadjusted", "Eta2_Adjusted"),
                       labels = c("Unadjusted", "Adjusted (MMSE + Age)"))) %>%
  ggplot(aes(x = Feature, y = Eta2, fill = Type)) +
  geom_bar(stat = "identity", position = "dodge", alpha = 0.8) +
  geom_hline(yintercept = 0.06, linetype = "dashed", color = "red", size = 0.8) +
  scale_fill_manual(values = c("Unadjusted" = "#95B3D7",
                               "Adjusted (MMSE + Age)" = "#8064A2")) +
  coord_flip() +
  labs(
    title = "Stage Independence of Endotype Markers ",
    subtitle = "ANCOVA: Effect sizes before and after adjusting for disease stage",
    x = "MRI Feature", 
    y = "Effect Size (η²)", 
    fill = NULL
  ) +
  theme_classic(base_size = 12) +
  theme(plot.title = element_text(face = "bold", size = 14),
        legend.position = "bottom")

ggsave(file.path(opt$output_dir, "Figure_Stage_Independence.png"), 
       p_stage, width = 10, height = 7, dpi = 300)
cat("  Saved: Figure_Stage_Independence.png\n\n")

# ==============================================================================
# Part 5: Disproportionate Atrophy Analysis 
# ==============================================================================
cat("[5/6] Disproportionate atrophy analysis (W-score residuals)...\n")

disprop_stats_list <- list()
disprop_residuals_df <- data.frame()

for (feat in mri_features) {
  if (!feat %in% colnames(data)) next
  
  # Determine global composite measure
  is_thickness <- str_detect(feat, "_TA$")
  global_measure <- if (is_thickness && "Global_TA_Composite" %in% colnames(data)) {
    "Global_TA_Composite"
  } else if ("Global_CV_Composite" %in% colnames(data)) {
    "Global_CV_Composite"
  } else {
    next
  }
  
  # Check if Gender exists
  if ("Gender" %in% colnames(data) && "Age" %in% colnames(data)) {
    formula_str <- paste(feat, "~", global_measure, "+ Age + Gender")
  } else if ("Age" %in% colnames(data)) {
    formula_str <- paste(feat, "~", global_measure, "+ Age")
  } else {
    formula_str <- paste(feat, "~", global_measure)
  }
  
  model_resid <- lm(as.formula(formula_str), data = data)
  residuals_std <- rstandard(model_resid)
  
  # Test residual differences across subtypes
  temp_data <- data
  temp_data$Residual <- residuals_std
  
  anova_res <- aov(Residual ~ factor(Subtype), data = temp_data)
  summ <- summary(anova_res)[[1]]
  
  f_val <- summ["factor(Subtype)", "F value"]
  p_val <- summ["factor(Subtype)", "Pr(>F)"]
  ss_between <- summ["factor(Subtype)", "Sum Sq"]
  ss_total <- sum(summ[, "Sum Sq"])
  eta_sq_resid <- ss_between / ss_total
  
  disprop_stats_list[[feat]] <- data.frame(
    Feature = feat,
    Global_Control = global_measure,
    F_value_Resid = f_val,
    P_Raw_Resid = p_val,
    Eta2_Resid = eta_sq_resid,
    Eta_Interpretation_Resid = interpret_eta_squared(eta_sq_resid),
    stringsAsFactors = FALSE
  )
  
  # Save residual data for visualization
  feat_resids <- data.frame(
    ID = if ("ID" %in% colnames(data)) data$ID else 1:nrow(data),
    Subtype = data$Subtype,
    Feature = feat,
    Residual_W_Score = residuals_std
  )
  disprop_residuals_df <- rbind(disprop_residuals_df, feat_resids)
}

# Combine statistics
disprop_stats <- bind_rows(disprop_stats_list)

# FDR correction
disprop_stats$P_FDR_Resid <- p.adjust(disprop_stats$P_Raw_Resid, method = "fdr")
disprop_stats$Significant_Topology <- disprop_stats$P_FDR_Resid < opt$fdr_threshold

cat("  Disproportionate atrophy results:\n")
n_topo_sig <- sum(disprop_stats$Significant_Topology)
cat(sprintf("    Topologically specific (FDR < %.2f): %d/%d\n", 
            opt$fdr_threshold, n_topo_sig, nrow(disprop_stats)))

write.csv(disprop_stats, 
          file.path(opt$output_dir, "Disproportionate_Atrophy_Stats.csv"), 
          row.names = FALSE)

# Disproportionate atrophy plot
if (nrow(disprop_residuals_df) > 0) {
  plot_data_residuals <- disprop_residuals_df
  plot_data_residuals$Region_Label <- str_remove_all(plot_data_residuals$Feature, "_TA|_CV|Right")
  
  p_residuals <- ggplot(plot_data_residuals, 
                        aes(x = Region_Label, y = Residual_W_Score, fill = factor(Subtype))) +
    geom_boxplot(outlier.shape = NA, alpha = 0.8) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
    scale_fill_manual(values = c("1" = "#95B3D7", "2" = "#8064A2", "3" = "#C0504D"),
                      name = "Subtype") +
    labs(
      title = "Topological Specificity (Disproportionate Atrophy)",
      subtitle = "W-score residuals after controlling for Global Atrophy, Age, and Sex",
      y = "Standardized Residuals (W-score)",
      x = "Brain Region"
    ) +
    theme_classic(base_size = 12) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1),
          plot.title = element_text(face = "bold"),
          legend.position = "bottom")
  
  ggsave(file.path(opt$output_dir, "Figure_Disproportionate_Atrophy.png"), 
         p_residuals, width = 10, height = 6, dpi = 300)
  cat("  Saved: Figure_Disproportionate_Atrophy.png\n\n")
}

# ==============================================================================
# Part 6: Summary Report
# ==============================================================================
cat("[6/6] Generating summary report...\n\n")

summary_lines <- c(
  "================================================================================",
  "Neuroimaging Endotype Characterization Report",
  "================================================================================",
  "",
  sprintf("Generated: %s", Sys.time()),
  "--------------------------------------------------------------------------------",
  "Data Summary",
  "--------------------------------------------------------------------------------",
  sprintf("  Sample size: %d", nrow(data)),
  sprintf("  Clinical features: %d", length(clinical_features)),
  sprintf("  MRI features: %d", length(mri_features)),
  ""
)

# Endotype distribution
for (i in 1:length(subtype_dist)) {
  summary_lines <- c(summary_lines,
    sprintf("  Subtype %s: %d (%.1f%%)",
            names(subtype_dist)[i],
            subtype_dist[i],
            100 * subtype_dist[i] / sum(subtype_dist)))
}

summary_lines <- c(summary_lines,
  "",
  "--------------------------------------------------------------------------------",
  "Clinical Homogeneity Results",
  "--------------------------------------------------------------------------------",
  sprintf("  Significant differences (FDR < %.2f): %d/%d", 
          opt$fdr_threshold, 
          sum(clinical_homogeneity$Significant_FDR), 
          nrow(clinical_homogeneity)),
  sprintf("  Mean effect size: η² = %.3f", mean(clinical_homogeneity$Eta_squared)),
  ""
)

summary_lines <- c(summary_lines,
  "--------------------------------------------------------------------------------",
  "MRI Heterogeneity Results",
  "--------------------------------------------------------------------------------",
  sprintf("  Significant differences (FDR < %.2f): %d/%d", 
          opt$fdr_threshold, n_sig, nrow(mri_heterogeneity)),
  sprintf("  Large effect (η² ≥ 0.14): %d/%d", n_large, nrow(mri_heterogeneity)),
  sprintf("  Mean effect size: η² = %.3f", mean(mri_heterogeneity$Eta_squared)),
  ""
)

summary_lines <- c(summary_lines,
  "--------------------------------------------------------------------------------",
  "Stage Independence Results (ANCOVA)",
  "--------------------------------------------------------------------------------",
  sprintf("  Stage-independent features (FDR < %.2f after adjustment): %d/%d", 
          opt$fdr_threshold, n_independent, nrow(stage_independence)),
  ""
)

summary_lines <- c(summary_lines,
  "--------------------------------------------------------------------------------",
  "Disproportionate Atrophy Results",
  "--------------------------------------------------------------------------------",
  sprintf("  Topologically specific (FDR < %.2f): %d/%d", 
          opt$fdr_threshold, n_topo_sig, nrow(disprop_stats)),
  ""
)

summary_lines <- c(summary_lines,
  "--------------------------------------------------------------------------------",
  "Key Findings",
  "--------------------------------------------------------------------------------",
  "  1. Clinical Homogeneity:",
  sprintf("     - %s significant clinical differences across endotypes",
          ifelse(sum(clinical_homogeneity$Significant_FDR) == 0, "No", "Some")),
  "",
  "  2. MRI Heterogeneity:",
  sprintf("     - %d/%d MRI features show significant endotype differences",
          n_sig, nrow(mri_heterogeneity)),
  "",
  "  3. Stage Independence:",
  sprintf("     - %d/%d features remain significant after MMSE & Age adjustment",
          n_independent, nrow(stage_independence)),
  "",
  "  4. Topological Specificity:",
  sprintf("     - %d/%d features show disproportionate atrophy patterns",
          n_topo_sig, nrow(disprop_stats)),
  "",
  "--------------------------------------------------------------------------------",
  "Output Files",
  "--------------------------------------------------------------------------------",
  sprintf("  - %s/Clinical_Homogeneity_Results.csv", opt$output_dir),
  sprintf("  - %s/MRI_Heterogeneity_Results.csv", opt$output_dir),
  sprintf("  - %s/Stage_Independence_Results.csv", opt$output_dir),
  sprintf("  - %s/Disproportionate_Atrophy_Stats.csv", opt$output_dir),
  sprintf("  - %s/Figure_Clinical_Homogeneity.png", opt$output_dir),
  sprintf("  - %s/Figure_MRI_Heterogeneity.png", opt$output_dir),
  sprintf("  - %s/Figure_Stage_Independence.png", opt$output_dir),
  sprintf("  - %s/Figure_Disproportionate_Atrophy.png", opt$output_dir),
  "",
  "================================================================================",
  "Neuroimaging Endotype Characterization Complete",
  "================================================================================"
)

# Write report
report_path <- file.path(opt$output_dir, "Neuroimaging_Endotype_Report.txt")
writeLines(summary_lines, report_path)

cat(paste(summary_lines, collapse = "\n"))
cat("\n\n")

cat("============================================================\n")
cat("Step 22: Neuroimaging Endotype Characterization Complete!\n")
cat("============================================================\n")
cat(sprintf("Report saved: %s\n", report_path))



