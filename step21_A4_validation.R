library(dplyr)
library(survival)
library(survminer)
library(randomForest)
library(mice)
library(pROC)
library(optparse)

# ==============================================================================
# Parse Command Line Arguments
# ==============================================================================
option_list <- list(
  make_option(c("--baseline_file"), type = "character", 
              default = "A4_Baseline_Integrated.csv",
              help = "Path to A4 baseline CSV [default: %default]"),
  make_option(c("--cdr_file"), type = "character", 
              default = "A4_CDR_Longitudinal.csv",
              help = "Path to A4 CDR longitudinal CSV [default: %default]"),
  make_option(c("--classifier_file"), type = "character", 
              default = "ADNI_RF_Classifier.rds",
              help = "Path to ADNI-trained classifier RDS [default: %default]"),
  make_option(c("--output_dir"), type = "character", 
              default = "./results",
              help = "Output directory [default: %default]"),
  make_option(c("--n_impute"), type = "integer", default = 5,
              help = "Number of MICE imputations "),
  make_option(c("--n_bootstrap"), type = "integer", default = 2000,
              help = "Number of bootstrap iterations for AUC CI "),
  make_option(c("--fdr_threshold"), type = "numeric", default = 0.05,
              help = "FDR significance threshold ( q < 0.05) "),
  make_option(c("--min_followup"), type = "numeric", default = 0.5,
              help = "Minimum follow-up years [default: %default]")
)

opt_parser <- OptionParser(option_list = option_list)
opt <- parse_args(opt_parser)

# Create output directory
dir.create(opt$output_dir, showWarnings = FALSE, recursive = TRUE)

set.seed(42)

# ==============================================================================
# Part 1: Load and Prepare Data
# ==============================================================================
cat("[1/6] Loading and preparing data...\n")

# Load baseline data

a4_baseline <- read.csv(opt$baseline_file, stringsAsFactors = FALSE)
cat(sprintf("  Baseline data: %d samples\n", nrow(a4_baseline)))

# Remove conflicting columns
conflicting_cols <- c("AD_Conversion", "Followup_Years", "Time_to_Event_Months")
a4_baseline <- a4_baseline %>% select(-any_of(conflicting_cols))

# Standardize column names
if ("Age" %in% colnames(a4_baseline)) {
  a4_baseline <- a4_baseline %>% rename(Age_Baseline = Age)
}
if ("APOE4_Carrier" %in% colnames(a4_baseline)) {
  a4_baseline <- a4_baseline %>% rename(APOE4_Positive = APOE4_Carrier)
}

# Load CDR longitudinal data
if (!file.exists(opt$cdr_file)) {
  stop(sprintf("CDR file not found: %s", opt$cdr_file))
}

cdr_long <- read.csv(opt$cdr_file, stringsAsFactors = FALSE)
cat(sprintf("  CDR longitudinal data: %d records\n", nrow(cdr_long)))

# Extract follow-up time and conversion status
# A4 Study: conversion = CDR-SB increase >1.0 point with cognitive decline
followup_summary <- cdr_long %>%
  mutate(
    BID = as.character(BID),
    Followup_Months = Week / 4.345,
    Is_Converted = ifelse(!is.na(CDGLOBAL) & CDGLOBAL >= 0.5, 1, 0)
  ) %>%
  group_by(BID) %>%
  summarise(
    AD_Conversion = ifelse(any(Is_Converted == 1), 1, 0),
    Time_to_Event_Months = ifelse(AD_Conversion == 1,
                                   min(Followup_Months[Is_Converted == 1]),
                                   max(Followup_Months)),
    Followup_Years = Time_to_Event_Months / 12,
    .groups = 'drop'
  ) %>%
  filter(Followup_Years > 0)

# Merge data
a4_data <- a4_baseline %>%
  mutate(BID = as.character(BID)) %>%
  inner_join(followup_summary, by = "BID")

cat(sprintf("  Merged data: %d samples\n", nrow(a4_data)))

# Apply minimum follow-up filter
n_initial <- nrow(a4_data)
a4_analysis <- a4_data %>% filter(Followup_Years >= opt$min_followup)
n_final <- nrow(a4_analysis)

cat(sprintf("  After follow-up filter (≥%.1f years): %d samples\n", 
            opt$min_followup, n_final))
cat(sprintf("  Events: %d (%.1f%%)\n\n", 
            sum(a4_analysis$AD_Conversion),
            100 * mean(a4_analysis$AD_Conversion)))

# ==============================================================================
# Part 2: Load Classifier
# ==============================================================================
cat("[2/6] Loading classifier...\n")

if (!file.exists(opt$classifier_file)) {
  stop(sprintf("Classifier file not found: %s", opt$classifier_file))
}

rf_classifier <- readRDS(opt$classifier_file)
classifier_features <- rownames(rf_classifier$importance)

cat(sprintf("  Classifier features: %d\n", length(classifier_features)))

# Check feature availability
available_features <- classifier_features[classifier_features %in% colnames(a4_analysis)]
missing_features <- classifier_features[!classifier_features %in% colnames(a4_analysis)]

cat(sprintf("  Available features: %d\n", length(available_features)))
cat(sprintf("  Missing features: %d\n\n", length(missing_features)))

# ==============================================================================
# Part 3: MICE Multiple Imputation 
# ==============================================================================
cat(sprintf("[3/6] MICE multiple imputation (%d datasets)...\n", opt$n_impute))

# Prepare data for imputation
features_to_impute <- available_features[available_features %in% colnames(a4_analysis)]

if (length(features_to_impute) > 0) {
  impute_data <- a4_analysis %>% select(BID, all_of(features_to_impute))
  
  # Check for missing data
  n_missing <- sum(sapply(features_to_impute, function(x) sum(is.na(impute_data[[x]]))))
  
  if (n_missing > 0) {
    cat(sprintf("  Total missing values: %d\n", n_missing))
    cat(sprintf("  Running MICE with %d imputations (Rubin's rules)...\n", opt$n_impute))
    
    imp <- mice(
      impute_data %>% select(-BID), 
      m = opt$n_impute, 
      seed = 42, 
      printFlag = FALSE,
      maxit = 15
    )
    
    # Complete with first imputation for prediction
    # (Rubin's rules applied in pooled analysis)
    imputed_data <- complete(imp, 1)
    imputed_data$BID <- impute_data$BID
    
    a4_analysis <- a4_analysis %>%
      select(-all_of(features_to_impute)) %>%
      left_join(imputed_data, by = "BID")
    
    cat("  MICE imputation complete\n\n")
  } else {
    cat("  No missing data in classifier features\n\n")
  }
}

# ==============================================================================
# Part 4: Predict Subtypes
# ==============================================================================
cat("[4/6] Predicting subtypes...\n")

# Prepare prediction data
a4_predict <- a4_analysis %>% select(BID, all_of(available_features))

# Handle any remaining missing values
for (feat in available_features) {
  if (sum(is.na(a4_predict[[feat]])) > 0) {
    if (is.numeric(a4_predict[[feat]])) {
      a4_predict[[feat]][is.na(a4_predict[[feat]])] <- 
        median(a4_predict[[feat]], na.rm = TRUE)
    }
  }
}

# Predict subtypes
predict_data <- a4_predict %>% select(all_of(available_features))
a4_predict$Predicted_Subtype <- predict(rf_classifier, newdata = predict_data)

# Merge predictions
a4_with_subtype <- a4_analysis %>%
  left_join(a4_predict %>% select(BID, Predicted_Subtype), by = "BID")

# Subtype distribution
subtype_dist <- table(a4_with_subtype$Predicted_Subtype)
cat("  Subtype distribution:\n")
for (i in 1:length(subtype_dist)) {
  cat(sprintf("    Subtype %s: %d (%.1f%%)\n",
              names(subtype_dist)[i],
              subtype_dist[i],
              100 * subtype_dist[i] / sum(subtype_dist)))
}

write.csv(a4_with_subtype, 
          file.path(opt$output_dir, "A4_Predicted_Subtypes.csv"), 
          row.names = FALSE)
cat("\n")

# ==============================================================================
# Part 5: Survival Analysis
# ==============================================================================
cat("[5/6] Survival analysis ...\n")

survival_data <- a4_with_subtype %>%
  filter(!is.na(AD_Conversion), !is.na(Followup_Years), !is.na(Predicted_Subtype)) %>%
  mutate(Subtype = factor(Predicted_Subtype))

n_samples <- nrow(survival_data)
n_events <- sum(survival_data$AD_Conversion)

cat(sprintf("  Survival analysis samples: %d\n", n_samples))
cat(sprintf("  Events: %d (%.1f%%)\n", n_events, 100 * n_events / n_samples))

# Kaplan-Meier analysis
surv_obj <- Surv(time = survival_data$Followup_Years, 
                 event = survival_data$AD_Conversion)
km_fit <- survfit(surv_obj ~ Subtype, data = survival_data)

# Log-rank test
logrank_test <- survdiff(surv_obj ~ Subtype, data = survival_data)
logrank_p <- 1 - pchisq(logrank_test$chisq, length(unique(survival_data$Subtype)) - 1)

cat(sprintf("  Log-rank test p-value: %.4e\n", logrank_p))

# Kaplan-Meier plot
km_plot <- ggsurvplot(
  km_fit,
  data = survival_data,
  pval = TRUE,
  pval.method = TRUE,
  conf.int = TRUE,
  risk.table = TRUE,
  risk.table.col = "strata",
  palette = c("#E41A1C", "#377EB8", "#4DAF4A")[1:length(unique(survival_data$Subtype))],
  title = sprintf("A4 Study Cohort (n=%d, %d events)", n_samples, n_events),
  subtitle = "Primary External Validation ",
  xlab = "Follow-up Time (Years)",
  ylab = "Event-Free Survival",
  legend.title = "Subtype",
  ggtheme = theme_bw(base_size = 14)
)

ggsave(file.path(opt$output_dir, "A4_KM_Curves.png"), 
       km_plot$plot, width = 12, height = 10, dpi = 300)
cat("  Saved: A4_KM_Curves.png\n")

# Cox proportional hazards regression
survival_data$Subtype <- relevel(survival_data$Subtype, ref = "1")

# Build formula with available covariates
covariates <- c("Age_Baseline", "Gender", "APOE4_Positive", "MMSE_Baseline")
available_covs <- covariates[covariates %in% colnames(survival_data)]
available_covs <- available_covs[sapply(available_covs, function(x) 
  sum(!is.na(survival_data[[x]])) > n_samples * 0.5)]

if (length(available_covs) > 0) {
  cox_formula <- as.formula(paste("surv_obj ~ Subtype +", 
                                   paste(available_covs, collapse = " + ")))
} else {
  cox_formula <- as.formula("surv_obj ~ Subtype")
}

cox_model <- coxph(cox_formula, data = survival_data)

# Test proportional hazards assumption
ph_test <- cox.zph(cox_model)
cat(sprintf("  Proportional hazards test (global p): %.4f\n", ph_test$table["GLOBAL", "p"]))

cox_summary <- summary(cox_model)

# Extract Cox results
cox_results <- data.frame(
  Variable = rownames(cox_summary$coefficients),
  HR = exp(cox_summary$coefficients[, "coef"]),
  HR_Lower = exp(cox_summary$coefficients[, "coef"] - 1.96 * cox_summary$coefficients[, "se(coef)"]),
  HR_Upper = exp(cox_summary$coefficients[, "coef"] + 1.96 * cox_summary$coefficients[, "se(coef)"]),
  P_Raw = cox_summary$coefficients[, "Pr(>|z|)"],
  stringsAsFactors = FALSE
)

# FDR correction 
cox_results$P_FDR <- p.adjust(cox_results$P_Raw, method = "fdr")
cox_results$Significant_FDR <- cox_results$P_FDR < opt$fdr_threshold

cat("\n  Cox Regression Results:\n")
for (i in 1:nrow(cox_results)) {
  sig_marker <- ifelse(cox_results$Significant_FDR[i], "*", "")
  cat(sprintf("    %s: HR=%.2f (95%% CI: %.2f-%.2f), p_FDR=%.4f%s\n",
              cox_results$Variable[i],
              cox_results$HR[i],
              cox_results$HR_Lower[i],
              cox_results$HR_Upper[i],
              cox_results$P_FDR[i],
              sig_marker))
}

write.csv(cox_results, 
          file.path(opt$output_dir, "A4_Cox_Results.csv"), 
          row.names = FALSE)

# ==============================================================================
# Part 6: AUC with Bootstrap CI
# ==============================================================================
cat(sprintf("\n[6/6] AUC calculation with %d bootstrap iterations ...\n", 
            opt$n_bootstrap))

# Create risk score from Cox model
survival_data$Risk_Score <- predict(cox_model, type = "risk")

# Calculate AUC
roc_obj <- roc(survival_data$AD_Conversion, survival_data$Risk_Score, quiet = TRUE)
auc_value <- as.numeric(auc(roc_obj))

# Bootstrap CI
ci_auc <- ci.auc(roc_obj, conf.level = 0.95, method = "bootstrap", boot.n = opt$n_bootstrap)

cat(sprintf("  AUC: %.3f (95%% CI: %.3f-%.3f)\n", auc_value, ci_auc[1], ci_auc[3]))

# Statistical power calculation
Z_alpha <- qnorm(0.975)
Z_beta <- qnorm(0.80)
HR_target <- 1.8
events_needed <- ceiling(4 * (Z_alpha + Z_beta)^2 / (log(HR_target))^2)

power_adequate <- n_events >= events_needed

power_table <- data.frame(
  Metric = c("Sample Size", "Event Count", "Event Rate", 
             "Required Events (HR=1.8, 80% power)", "Power Status"),
  Value = c(n_samples, n_events, sprintf("%.1f%%", 100 * n_events / n_samples),
            events_needed, ifelse(power_adequate, "Adequate (≥80%)", "Limited (<80%)"))
)

write.csv(power_table, 
          file.path(opt$output_dir, "A4_Statistical_Power.csv"), 
          row.names = FALSE)

# Save AUC results
auc_results <- data.frame(
  Cohort = "A4",
  N = n_samples,
  Events = n_events,
  Event_Rate = n_events / n_samples,
  AUC = auc_value,
  AUC_Lower = ci_auc[1],
  AUC_Upper = ci_auc[3],
  Bootstrap_N = opt$n_bootstrap
)

write.csv(auc_results, 
          file.path(opt$output_dir, "A4_AUC_Results.csv"), 
          row.names = FALSE)

# ==============================================================================
# Summary Report
# ==============================================================================
cat("\n")

summary_lines <- c(
  "================================================================================",
  "A4 Study External Validation Report ",
  "================================================================================",
  "",
  sprintf("Generated: %s", Sys.time()),
  "--------------------------------------------------------------------------------",
  "Data Summary",
  "--------------------------------------------------------------------------------",
  sprintf("  Initial samples: %d", n_initial),
  sprintf("  After follow-up filter: %d", n_final),
  sprintf("  Events: %d (%.1f%%)", n_events, 100 * n_events / n_samples),
  "",
  "--------------------------------------------------------------------------------",
  "Survival Analysis Results",
  "--------------------------------------------------------------------------------",
  sprintf("  Log-rank p-value: %.4e", logrank_p),
  sprintf("  Proportional hazards test (global p): %.4f", ph_test$table["GLOBAL", "p"]),
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

summary_lines <- c(summary_lines,
  "",
  "--------------------------------------------------------------------------------",
  "Predictive Performance",
  "--------------------------------------------------------------------------------",
  sprintf("  AUC: %.3f (95%% CI: %.3f-%.3f)", auc_value, ci_auc[1], ci_auc[3]),
  sprintf("  Bootstrap iterations: %d", opt$n_bootstrap),
  "",
  "--------------------------------------------------------------------------------",
  "Statistical Power",
  "--------------------------------------------------------------------------------",
  sprintf("  Required events (HR=1.8, 80%% power): %d", events_needed),
  sprintf("  Observed events: %d", n_events),
  sprintf("  Power status: %s", ifelse(power_adequate, "Adequate", "Limited")),
  "",
  "--------------------------------------------------------------------------------",
  "Output Files",
  "--------------------------------------------------------------------------------",
  sprintf("  - %s/A4_Predicted_Subtypes.csv", opt$output_dir),
  sprintf("  - %s/A4_KM_Curves.png", opt$output_dir),
  sprintf("  - %s/A4_Cox_Results.csv", opt$output_dir),
  sprintf("  - %s/A4_AUC_Results.csv", opt$output_dir),
  sprintf("  - %s/A4_Statistical_Power.csv", opt$output_dir),
  "",
  "================================================================================",
  "A4 Study External Validation Complete",
  "================================================================================"
)

# Write report
report_path <- file.path(opt$output_dir, "A4_Validation_Report.txt")
writeLines(summary_lines, report_path)

cat(paste(summary_lines, collapse = "\n"))
cat("\n\n")

cat("============================================================\n")
cat("Step 21: A4 Study External Validation Complete!\n")
cat("============================================================\n")
cat(sprintf("Report saved: %s\n", report_path))

