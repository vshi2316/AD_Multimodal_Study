library(dplyr)
library(ggplot2)
library(survival)
library(survminer)
library(pheatmap)
library(effsize)
library(optparse)

# ==============================================================================
# Parse Command Line Arguments
# ==============================================================================
option_list <- list(
  make_option(c("--integrated_file"), type = "character", 
              default = "Cohort_A_Integrated.csv",
              help = "Path to integrated cohort CSV [default: %default]"),
  make_option(c("--cluster_file"), type = "character", 
              default = "Final_Consensus_Clusters_K3.csv",
              help = "Path to cluster results CSV [default: %default]"),
  make_option(c("--output_dir"), type = "character", 
              default = "./results",
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

calculate_eta_squared <- function(values, groups) {
  valid_idx <- !is.na(values) & !is.na(groups)
  values <- values[valid_idx]
  groups <- factor(groups[valid_idx])
  
  if (length(unique(groups)) < 2 || length(values) < 10) {
    return(list(eta_sq = NA, interpretation = NA))
  }
  
  grand_mean <- mean(values)
  group_means <- tapply(values, groups, mean)
  group_ns <- tapply(values, groups, length)
  
  ss_between <- sum(group_ns * (group_means - grand_mean)^2)
  ss_total <- sum((values - grand_mean)^2)
  
  eta_sq <- ss_between / ss_total
  
  # Cohen's criteria (Methods 2.8)
  if (eta_sq >= 0.14) {
    interpretation <- "Large"
  } else if (eta_sq >= 0.06) {
    interpretation <- "Medium"
  } else if (eta_sq >= 0.01) {
    interpretation <- "Small"
  } else {
    interpretation <- "Negligible"
  }
  
  return(list(eta_sq = eta_sq, interpretation = interpretation))
}

#' Calculate maximum pairwise Cohen's d (SMD)
#' Methods 2.4: "|SMD| > 0.5 for clinical meaningfulness"
calculate_max_smd <- function(values, groups) {
  groups <- factor(groups)
  levels_g <- levels(groups)
  n_groups <- length(levels_g)
  
  if (n_groups < 2) return(NA)
  
  max_d <- 0
  
  for (i in 1:(n_groups - 1)) {
    for (j in (i + 1):n_groups) {
      g1 <- values[groups == levels_g[i]]
      g2 <- values[groups == levels_g[j]]
      
      g1 <- g1[!is.na(g1)]
      g2 <- g2[!is.na(g2)]
      
      if (length(g1) >= 3 && length(g2) >= 3) {
        d <- cohen.d(g1, g2)
        if (abs(d$estimate) > abs(max_d)) {
          max_d <- d$estimate
        }
      }
    }
  }
  
  return(max_d)
}

# ==============================================================================
# Part 1: Load and Merge Data
# ==============================================================================
cat("[1/5] Loading and merging data...\n")

if (!file.exists(opt$integrated_file)) {
  stop(sprintf("Integrated file not found: %s", opt$integrated_file))
}

if (!file.exists(opt$cluster_file)) {
  stop(sprintf("Cluster file not found: %s", opt$cluster_file))
}

adni_data <- read.csv(opt$integrated_file, stringsAsFactors = FALSE)
cluster_results <- read.csv(opt$cluster_file, stringsAsFactors = FALSE)

cat(sprintf("  ADNI data: %d samples\n", nrow(adni_data)))
cat(sprintf("  Cluster results: %d samples\n", nrow(cluster_results)))

# Detect cluster column
cluster_col <- NULL
for (col in c("Consensus_Cluster_K3", "Consensus_Cluster", "Cluster_Labels", "Cluster")) {
  if (col %in% colnames(cluster_results)) {
    cluster_col <- col
    break
  }
}

if (is.null(cluster_col)) {
  cluster_col <- colnames(cluster_results)[2]
}

cat(sprintf("  Cluster column: %s\n", cluster_col))

# Merge data
if ("Sample_Index" %in% colnames(cluster_results)) {
  adni_data$Subtype <- cluster_results[[cluster_col]][match(1:nrow(adni_data), 
                                                             cluster_results$Sample_Index)]
} else {
  adni_data$Subtype <- cluster_results[[cluster_col]]
}

adni_labeled <- adni_data %>%
  filter(!is.na(Subtype)) %>%
  mutate(Subtype = factor(Subtype))

n_subtypes <- length(unique(adni_labeled$Subtype))
cat(sprintf("  Labeled samples: %d\n", nrow(adni_labeled)))
cat(sprintf("  Number of subtypes: %d\n\n", n_subtypes))

# Subtype distribution
cat("Subtype Distribution:\n")
subtype_counts <- table(adni_labeled$Subtype)
for (i in 1:length(subtype_counts)) {
  cat(sprintf("  Subtype %s: %d (%.1f%%)\n", 
              names(subtype_counts)[i], 
              subtype_counts[i],
              100 * subtype_counts[i] / sum(subtype_counts)))
}
cat("\n")

# ==============================================================================
# Part 2: Feature Group Comparisons (Methods 2.4, 2.8)
# ==============================================================================
cat("[2/5] Feature group comparisons (Methods 2.4, 2.8)...\n")

# Define feature groups
feature_groups <- list(
  Demographics = c("Age", "Gender", "Education"),
  Cognition = c("MMSE_Baseline", "ADAS13", "CDRSB", "RAVLT_immediate", "RAVLT_learning"),
  Genetics = c("APOE4_Positive", "APOE4_Copies"),
  CSF = c("ABETA42", "ABETA", "TAU_TOTAL", "TAU", "PTAU181", "PTAU"),
  MRI = grep("^ST\\d+", colnames(adni_labeled), value = TRUE)
)

all_results <- data.frame()

for (group_name in names(feature_groups)) {
  features <- feature_groups[[group_name]]
  features <- features[features %in% colnames(adni_labeled)]
  
  if (length(features) == 0) next
  
  cat(sprintf("  Processing %s (%d features)...\n", group_name, length(features)))
  
  for (var in features) {
    # Skip if too much missing data
    if (mean(is.na(adni_labeled[[var]])) > 0.5) next
    
    if (is.numeric(adni_labeled[[var]])) {
      # Continuous variable: ANOVA or Kruskal-Wallis
      test_data <- adni_labeled %>%
        select(Subtype, !!sym(var)) %>%
        filter(!is.na(!!sym(var)))
      
      if (nrow(test_data) < 10) next
      
      # Normality test
      shapiro_pvals <- by(test_data[[var]], test_data$Subtype,
                         function(x) if(length(x) >= 3 && length(x) <= 5000) 
                           shapiro.test(x)$p.value else NA)
      normal_dist <- all(shapiro_pvals > 0.05, na.rm = TRUE)
      
      if (normal_dist) {
        anova_result <- aov(as.formula(paste(var, "~ Subtype")), data = test_data)
        p_value <- summary(anova_result)[[1]][["Pr(>F)"]][1]
        test_type <- "ANOVA"
      } else {
        kw_result <- kruskal.test(as.formula(paste(var, "~ Subtype")), data = test_data)
        p_value <- kw_result$p.value
        test_type <- "Kruskal-Wallis"
      }
      
      # Eta-squared (Methods 2.8)
      eta_result <- calculate_eta_squared(test_data[[var]], test_data$Subtype)
      
      # Max SMD (Methods 2.4)
      max_smd <- calculate_max_smd(test_data[[var]], test_data$Subtype)
      
      all_results <- rbind(all_results, data.frame(
        Feature_Group = group_name,
        Feature = var,
        Test = test_type,
        P_Raw = p_value,
        Eta_Squared = eta_result$eta_sq,
        Eta_Interpretation = eta_result$interpretation,
        Max_SMD = max_smd,
        Clinically_Meaningful = ifelse(!is.na(max_smd), abs(max_smd) > opt$smd_threshold, FALSE),
        stringsAsFactors = FALSE
      ))
      
    } else {
      # Categorical variable: Chi-square or Fisher's exact
      contingency_table <- table(adni_labeled$Subtype, adni_labeled[[var]])
      
      if (min(dim(contingency_table)) < 2) next
      
      if (any(contingency_table < 5)) {
        test_result <- fisher.test(contingency_table, simulate.p.value = TRUE)
        test_type <- "Fisher"
      } else {
        test_result <- chisq.test(contingency_table)
        test_type <- "Chi-square"
      }
      
      p_value <- test_result$p.value
      
      # Cramér's V as effect size for categorical
      n <- sum(contingency_table)
      chi_sq <- chisq.test(contingency_table)$statistic
      k <- min(dim(contingency_table))
      cramers_v <- sqrt(chi_sq / (n * (k - 1)))
      
      all_results <- rbind(all_results, data.frame(
        Feature_Group = group_name,
        Feature = var,
        Test = test_type,
        P_Raw = p_value,
        Eta_Squared = NA,
        Eta_Interpretation = NA,
        Max_SMD = cramers_v,  # Using Cramér's V for categorical
        Clinically_Meaningful = cramers_v > 0.3,  # Medium effect for Cramér's V
        stringsAsFactors = FALSE
      ))
    }
  }
}

# FDR correction (Methods 2.4)
all_results$P_FDR <- p.adjust(all_results$P_Raw, method = "fdr")
all_results$Significant_FDR <- all_results$P_FDR < opt$fdr_threshold

# Summary by group
cat("\nResults Summary by Feature Group:\n")
group_summary <- all_results %>%
  group_by(Feature_Group) %>%
  summarise(
    N_Features = n(),
    N_Significant = sum(Significant_FDR, na.rm = TRUE),
    N_Clinical = sum(Clinically_Meaningful, na.rm = TRUE),
    N_Large_Effect = sum(Eta_Interpretation == "Large", na.rm = TRUE),
    .groups = "drop"
  )

for (i in 1:nrow(group_summary)) {
  cat(sprintf("  %s: %d features, %d significant (FDR), %d clinically meaningful\n",
              group_summary$Feature_Group[i],
              group_summary$N_Features[i],
              group_summary$N_Significant[i],
              group_summary$N_Clinical[i]))
}

write.csv(all_results, 
          file.path(opt$output_dir, "Feature_Differences.csv"), 
          row.names = FALSE)
cat("\n")

# ==============================================================================
# Part 3: Survival Analysis (Methods 2.4)
# ==============================================================================
cat("[3/5] Survival analysis (Methods 2.4)...\n")

survival_completed <- FALSE

if ("AD_Conversion" %in% colnames(adni_labeled)) {
  # Prepare survival data
  adni_survival <- adni_labeled %>%
    select(any_of(c("ID", "RID", "Subtype", "AD_Conversion", "Age", "Gender", 
                    "APOE4_Positive", "MMSE_Baseline", "Followup_Time", "Followup_Years"))) %>%
    filter(!is.na(AD_Conversion))
  
  # Handle follow-up time
  if ("Followup_Years" %in% colnames(adni_survival)) {
    # Already have follow-up years
  } else if ("Followup_Time" %in% colnames(adni_survival)) {
    adni_survival$Followup_Years <- adni_survival$Followup_Time
  } else {
    # Default follow-up time
    adni_survival$Followup_Years <- 3
    cat("  Warning: No follow-up time found, using default 3 years\n")
  }
  
  adni_survival <- adni_survival %>%
    filter(!is.na(Followup_Years), Followup_Years > 0)
  
  cat(sprintf("  Survival analysis samples: %d\n", nrow(adni_survival)))
  cat(sprintf("  Events (AD conversion): %d (%.1f%%)\n", 
              sum(adni_survival$AD_Conversion),
              100 * mean(adni_survival$AD_Conversion)))
  
  if (nrow(adni_survival) >= 30 && sum(adni_survival$AD_Conversion) >= 10) {
    # Kaplan-Meier analysis
    surv_obj <- Surv(time = adni_survival$Followup_Years, 
                     event = adni_survival$AD_Conversion)
    km_fit <- survfit(surv_obj ~ Subtype, data = adni_survival)
    
    # Log-rank test
    logrank_test <- survdiff(surv_obj ~ Subtype, data = adni_survival)
    logrank_p <- 1 - pchisq(logrank_test$chisq, df = n_subtypes - 1)
    
    cat(sprintf("  Log-rank test p-value: %.4e\n", logrank_p))
    
    # Kaplan-Meier plot
    km_plot <- ggsurvplot(
      km_fit,
      data = adni_survival,
      pval = TRUE,
      pval.method = TRUE,
      conf.int = TRUE,
      risk.table = TRUE,
      risk.table.col = "strata",
      palette = c("#E41A1C", "#377EB8", "#4DAF4A")[1:n_subtypes],
      title = "Kaplan-Meier Survival Curves by Subtype",
      subtitle = sprintf("ADNI Discovery Cohort (n=%d, events=%d)", 
                         nrow(adni_survival), sum(adni_survival$AD_Conversion)),
      xlab = "Time (Years)",
      ylab = "Survival Probability",
      legend.title = "Subtype",
      legend.labs = paste0("Subtype ", levels(adni_survival$Subtype)),
      ggtheme = theme_bw(base_size = 14)
    )
    
    ggsave(file.path(opt$output_dir, "KM_Curves.png"), 
           km_plot$plot, width = 12, height = 10, dpi = 300)
    cat("  Saved: KM_Curves.png\n")
    
    # Cox proportional hazards regression
    adni_survival$Subtype_Factor <- relevel(factor(adni_survival$Subtype), ref = "1")
    
    # Build formula based on available covariates
    covariates <- c("Age", "Gender", "APOE4_Positive", "MMSE_Baseline")
    available_covs <- covariates[covariates %in% colnames(adni_survival)]
    available_covs <- available_covs[sapply(available_covs, function(x) 
      sum(!is.na(adni_survival[[x]])) > nrow(adni_survival) * 0.5)]
    
    if (length(available_covs) > 0) {
      cox_formula <- as.formula(paste("surv_obj ~ Subtype_Factor +", 
                                       paste(available_covs, collapse = " + ")))
    } else {
      cox_formula <- as.formula("surv_obj ~ Subtype_Factor")
    }
    
    cox_model <- coxph(cox_formula, data = adni_survival)
    cox_summary <- summary(cox_model)
    
    # Extract Cox results
    cox_results <- data.frame(
      Variable = rownames(cox_summary$conf.int),
      HR = cox_summary$conf.int[, "exp(coef)"],
      HR_Lower = cox_summary$conf.int[, "lower .95"],
      HR_Upper = cox_summary$conf.int[, "upper .95"],
      P_Value = cox_summary$coefficients[, "Pr(>|z|)"],
      stringsAsFactors = FALSE
    )
    
    # FDR correction for Cox results
    cox_results$P_FDR <- p.adjust(cox_results$P_Value, method = "fdr")
    cox_results$Significant_FDR <- cox_results$P_FDR < opt$fdr_threshold
    
    cat("\nCox Regression Results:\n")
    for (i in 1:nrow(cox_results)) {
      sig_marker <- ifelse(cox_results$Significant_FDR[i], "*", "")
      cat(sprintf("  %s: HR=%.2f (95%% CI: %.2f-%.2f), p_FDR=%.4f%s\n",
                  cox_results$Variable[i],
                  cox_results$HR[i],
                  cox_results$HR_Lower[i],
                  cox_results$HR_Upper[i],
                  cox_results$P_FDR[i],
                  sig_marker))
    }
    
    write.csv(cox_results, 
              file.path(opt$output_dir, "Cox_Results.csv"), 
              row.names = FALSE)
    
    survival_completed <- TRUE
  } else {
    cat("  Insufficient data for survival analysis\n")
  }
} else {
  cat("  AD_Conversion column not found, skipping survival analysis\n")
}
cat("\n")

# ==============================================================================
# Part 4: MRI Heterogeneity Heatmap
# ==============================================================================
cat("[4/5] Generating MRI heterogeneity heatmap...\n")

mri_results <- all_results %>%
  filter(Feature_Group == "MRI", Significant_FDR)

if (nrow(mri_results) > 0) {
  # Select top significant MRI features
  top_mri <- mri_results %>%
    arrange(P_FDR) %>%
    head(min(30, nrow(mri_results))) %>%
    pull(Feature)
  
  if (length(top_mri) > 0) {
    mri_matrix <- adni_labeled %>%
      select(Subtype, all_of(top_mri)) %>%
      group_by(Subtype) %>%
      summarise(across(everything(), ~ mean(.x, na.rm = TRUE)), .groups = "drop")
    
    subtype_labels <- mri_matrix$Subtype
    mri_matrix <- mri_matrix %>% select(-Subtype) %>% as.matrix()
    rownames(mri_matrix) <- paste0("Subtype ", subtype_labels)
    
    # Z-score normalize
    mri_matrix_scaled <- scale(t(mri_matrix))
    
    png(file.path(opt$output_dir, "MRI_Heterogeneity_Heatmap.png"), 
        width = 3500, height = 2500, res = 300)
    
    pheatmap(mri_matrix_scaled,
             cluster_rows = TRUE,
             cluster_cols = FALSE,
             color = colorRampPalette(c("blue", "white", "red"))(100),
             main = sprintf("MRI Heterogeneity Across Subtypes (Top %d FDR-significant regions)", 
                           length(top_mri)),
             fontsize = 10,
             fontsize_row = 8)
    
    dev.off()
    cat(sprintf("  Saved: MRI_Heterogeneity_Heatmap.png (%d regions)\n", length(top_mri)))
  }
} else {
  cat("  No significant MRI features for heatmap\n")
}
cat("\n")

# ==============================================================================
# Part 5: Save Labeled Data and Summary Report
# ==============================================================================
cat("[5/5] Saving results and generating report...\n")

# Save labeled data for downstream analysis
write.csv(adni_labeled, 
          file.path(opt$output_dir, "ADNI_Labeled_For_Classifier.csv"), 
          row.names = FALSE)

# Generate summary report
summary_lines <- c(
  "================================================================================",
  "ADNI Discovery Analysis Report (Methods 2.4, 2.8 Aligned)",
  "================================================================================",
  "",
  sprintf("Generated: %s", Sys.time()),
  "",
  "Methods Requirements:",
  sprintf("  Methods 2.4: Benjamini-Hochberg FDR correction (q < %.2f)", opt$fdr_threshold),
  sprintf("  Methods 2.4: |SMD| > %.1f for clinical meaningfulness", opt$smd_threshold),
  "  Methods 2.8: Eta-squared effect sizes",
  "    - η² ≥ 0.01: Small effect",
  "    - η² ≥ 0.06: Medium effect",
  "    - η² ≥ 0.14: Large effect",
  "",
  "--------------------------------------------------------------------------------",
  "Data Summary",
  "--------------------------------------------------------------------------------",
  sprintf("  Total samples: %d", nrow(adni_labeled)),
  sprintf("  Number of subtypes: %d", n_subtypes),
  ""
)

for (i in 1:length(subtype_counts)) {
  summary_lines <- c(summary_lines,
    sprintf("  Subtype %s: %d (%.1f%%)", 
            names(subtype_counts)[i], 
            subtype_counts[i],
            100 * subtype_counts[i] / sum(subtype_counts)))
}

summary_lines <- c(summary_lines,
  "",
  "--------------------------------------------------------------------------------",
  "Feature Analysis Results",
  "--------------------------------------------------------------------------------"
)

for (i in 1:nrow(group_summary)) {
  summary_lines <- c(summary_lines,
    sprintf("  %s: %d features, %d significant (FDR < %.2f), %d clinically meaningful",
            group_summary$Feature_Group[i],
            group_summary$N_Features[i],
            group_summary$N_Significant[i],
            opt$fdr_threshold,
            group_summary$N_Clinical[i]))
}

if (survival_completed) {
  summary_lines <- c(summary_lines,
    "",
    "--------------------------------------------------------------------------------",
    "Survival Analysis Results",
    "--------------------------------------------------------------------------------",
    sprintf("  Samples: %d", nrow(adni_survival)),
    sprintf("  Events: %d (%.1f%%)", sum(adni_survival$AD_Conversion),
            100 * mean(adni_survival$AD_Conversion)),
    sprintf("  Log-rank p-value: %.4e", logrank_p),
    ""
  )
  
  for (i in 1:nrow(cox_results)) {
    sig_marker <- ifelse(cox_results$Significant_FDR[i], " *", "")
    summary_lines <- c(summary_lines,
      sprintf("  %s: HR=%.2f (%.2f-%.2f), p_FDR=%.4f%s",
              cox_results$Variable[i],
              cox_results$HR[i],
              cox_results$HR_Lower[i],
              cox_results$HR_Upper[i],
              cox_results$P_FDR[i],
              sig_marker))
  }
}

summary_lines <- c(summary_lines,
  "",
  "--------------------------------------------------------------------------------",
  "Output Files",
  "--------------------------------------------------------------------------------",
  sprintf("  - %s/Feature_Differences.csv", opt$output_dir),
  sprintf("  - %s/ADNI_Labeled_For_Classifier.csv", opt$output_dir)
)

if (survival_completed) {
  summary_lines <- c(summary_lines,
    sprintf("  - %s/KM_Curves.png", opt$output_dir),
    sprintf("  - %s/Cox_Results.csv", opt$output_dir))
}

if (nrow(mri_results) > 0) {
  summary_lines <- c(summary_lines,
    sprintf("  - %s/MRI_Heterogeneity_Heatmap.png", opt$output_dir))
}

summary_lines <- c(summary_lines,
  "",
  "================================================================================",
  "ADNI Discovery Analysis Complete",
  "================================================================================"
)

# Write report
report_path <- file.path(opt$output_dir, "ADNI_Discovery_Report.txt")
writeLines(summary_lines, report_path)

cat(paste(summary_lines, collapse = "\n"))
cat("\n\n")

cat("============================================================\n")
cat("Step 19: ADNI Discovery Analysis Complete!\n")
cat("============================================================\n")
cat(sprintf("Report saved: %s\n", report_path))
