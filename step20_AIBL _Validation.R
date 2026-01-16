library(dplyr)
library(lubridate)
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
              default = "AIBL_Baseline_Integrated.csv",
              help = "Path to AIBL baseline CSV [default: %default]"),
  make_option(c("--cdr_file"), type = "character", 
              default = "AIBL_CDR_Longitudinal.csv",
              help = "Path to AIBL CDR longitudinal CSV [default: %default]"),
  make_option(c("--classifier_file"), type = "character", 
              default = "ADNI_RF_Classifier.rds",
              help = "Path to ADNI-trained classifier RDS [default: %default]"),
  make_option(c("--output_dir"), type = "character", 
              default = "./results",
              help = "Output directory [default: %default]"),
  make_option(c("--n_impute"), type = "integer", default = 5,
              help = "Number of MICE imputations "),
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
cat("[1/5] Loading and preparing data...\n")

# Load baseline data

aibl_baseline <- read.csv(opt$baseline_file, stringsAsFactors = FALSE)
cat(sprintf("  Baseline data: %d samples\n", nrow(aibl_baseline)))

# Remove conflicting columns
conflicting_cols <- c("AD_Conversion", "Followup_Years", "Time_to_Event_Months")
aibl_baseline <- aibl_baseline %>%
  select(-any_of(conflicting_cols))

# Standardize column names
if ("Age" %in% colnames(aibl_baseline)) {
  aibl_baseline <- aibl_baseline %>% rename(Age_Baseline = Age)
}
if ("APOE4_Carrier" %in% colnames(aibl_baseline)) {
  aibl_baseline <- aibl_baseline %>% rename(APOE4_Positive = APOE4_Carrier)
}

# Load longitudinal CDR data
if (!file.exists(opt$cdr_file)) {
  stop(sprintf("CDR file not found: %s", opt$cdr_file))
}

cdr_long <- read.csv(opt$cdr_file, stringsAsFactors = FALSE)
cat(sprintf("  CDR longitudinal data: %d records\n", nrow(cdr_long)))

# Generate follow-up summary
followup_summary <- cdr_long %>%
  mutate(
    RID = as.character(RID),
    Followup_Months = Week / 4.345,
    Is_Converted = ifelse(!is.na(CDGLOBAL) & CDGLOBAL >= 0.5, 1, 0)
  ) %>%
  group_by(RID) %>%
  summarise(
    AD_Conversion = ifelse(any(Is_Converted == 1), 1, 0),
    Time_to_Event_Months = ifelse(AD_Conversion == 1,
                                   min(Followup_Months[Is_Converted == 1]),
                                   max(Followup_Months)),
    Followup_Years = Time_to_Event_Months / 12,
    .groups = 'drop'
  ) %>%
  filter(Followup_Years > 0)

# Merge baseline + follow-up
aibl_data <- aibl_baseline %>%
  mutate(RID = as.character(RID)) %>%
  inner_join(followup_summary, by = "RID")

cat(sprintf("  Merged data: %d samples\n", nrow(aibl_data)))

# Apply minimum follow-up filter
n_initial <- nrow(aibl_data)
aibl_analysis <- aibl_data %>% 
  filter(Followup_Years >= opt$min_followup)
n_final <- nrow(aibl_analysis)

cat(sprintf("  After follow-up filter (≥%.1f years): %d samples\n", 
            opt$min_followup, n_final))
cat(sprintf("  Events: %d (%.1f%%)\n\n", 
            sum(aibl_analysis$AD_Conversion),
            100 * mean(aibl_analysis$AD_Conversion)))

# ==============================================================================
# Part 2: Load Classifier and Prepare Features
# ==============================================================================
cat("[2/5] Loading classifier and preparing features...\n")

if (!file.exists(opt$classifier_file)) {
  stop(sprintf("Classifier file not found: %s", opt$classifier_file))
}

rf_classifier <- readRDS(opt$classifier_file)
classifier_features <- rownames(rf_classifier$importance)

cat(sprintf("  Classifier features: %d\n", length(classifier_features)))

# Check feature availability
available_features <- classifier_features[classifier_features %in% colnames(aibl_analysis)]
missing_features <- classifier_features[!classifier_features %in% colnames(aibl_analysis)]

cat(sprintf("  Available features: %d\n", length(available_features)))
cat(sprintf("  Missing features: %d\n", length(missing_features)))

if (length(missing_features) > 0 && length(missing_features) <= 10) {
  cat(sprintf("  Missing: %s\n", paste(missing_features, collapse = ", ")))
}

# ==============================================================================
# Part 3: MICE Multiple Imputation 
# ==============================================================================
cat(sprintf("\n[3/5] MICE multiple imputation (%d datasets)...\n", opt$n_impute))

# Prepare data for imputation
aibl_predict <- aibl_analysis %>%
  select(RID, all_of(available_features))

# Check missing data
missing_summary <- data.frame(
  Feature = available_features,
  N_Missing = sapply(available_features, function(x) sum(is.na(aibl_predict[[x]]))),
  Pct_Missing = sapply(available_features, function(x) 100 * mean(is.na(aibl_predict[[x]])))
)

features_with_missing <- missing_summary %>% filter(N_Missing > 0)
if (nrow(features_with_missing) > 0) {
  cat("  Features with missing data:\n")
  for (i in 1:min(5, nrow(features_with_missing))) {
    cat(sprintf("    %s: %d (%.1f%%)\n",
                features_with_missing$Feature[i],
                features_with_missing$N_Missing[i],
                features_with_missing$Pct_Missing[i]))
  }
}

# Perform MICE imputation
if (any(missing_summary$N_Missing > 0)) {
  cat(sprintf("  Running MICE with %d imputations...\n", opt$n_impute))
  
  imp <- mice(
    aibl_predict %>% select(-RID), 
    m = opt$n_impute, 
    seed = 42, 
    printFlag = FALSE,
    maxit = 15
  )
  
  # Use first imputed dataset for prediction (Rubin's rules applied in analysis)
  imputed_data <- complete(imp, 1)
  imputed_data$RID <- aibl_predict$RID
  
  aibl_predict <- imputed_data
  cat("  MICE imputation complete\n")
} else {
  cat("  No missing data, skipping imputation\n")
}

# Handle any remaining missing values with median imputation
for (feat in available_features) {
  if (sum(is.na(aibl_predict[[feat]])) > 0) {
    if (is.numeric(aibl_predict[[feat]])) {
      aibl_predict[[feat]][is.na(aibl_predict[[feat]])] <- 
        median(aibl_predict[[feat]], na.rm = TRUE)
    }
  }
}

# ==============================================================================
# Part 4: Predict Subtypes
# ==============================================================================
cat("\n[4/5] Predicting subtypes...\n")

# Predict subtypes
predict_data <- aibl_predict %>% select(all_of(available_features))
aibl_predict$Predicted_Subtype <- predict(rf_classifier, newdata = predict_data)

# Merge predictions with analysis data
aibl_with_subtype <- aibl_analysis %>%
  left_join(aibl_predict %>% select(RID, Predicted_Subtype), by = "RID")

# Subtype distribution
subtype_dist <- table(aibl_with_subtype$Predicted_Subtype)
cat("  Subtype distribution:\n")
for (i in 1:length(subtype_dist)) {
  cat(sprintf("    Subtype %s: %d (%.1f%%)\n",
              names(subtype_dist)[i],
              subtype_dist[i],
              100 * subtype_dist[i] / sum(subtype_dist)))
}

write.csv(aibl_with_subtype, 
          file.path(opt$output_dir, "AIBL_Predicted_Subtypes.csv"), 
          row.names = FALSE)

# ==============================================================================
# Part 5: Survival Analysis 
# ==============================================================================
cat("\n[5/5] Survival analysis ...\n")

survival_data <- aibl_with_subtype %>%
  filter(!is.na(Followup_Years), !is.na(AD_Conversion), !is.na(Predicted_Subtype)) %>%
  mutate(Subtype = factor(Predicted_Subtype))

n_samples <- nrow(survival_data)
n_events <- sum(survival_data$AD_Conversion)

cat(sprintf("  Survival analysis samples: %d\n", n_samples))
cat(sprintf("  Events: %d (%.1f%%)\n", n_events, 100 * n_events / n_samples))

if (n_samples >= 20 && n_events >= 5) {
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
    title = sprintf("AIBL Cohort (n=%d, %d events)", n_samples, n_events),
    subtitle = "External Validation ",
    xlab = "Time (Years)",
    ylab = "Event-Free Survival",
    legend.title = "Subtype",
    ggtheme = theme_bw(base_size = 14)
  )
  
  ggsave(file.path(opt$output_dir, "AIBL_KM_Curves.png"), 
         km_plot$plot, width = 12, height = 10, dpi = 300)
  cat("  Saved: AIBL_KM_Curves.png\n")
  
  # Cox proportional hazards regression
  survival_data$Subtype <- relevel(survival_data$Subtype, ref = "1")
  
  # Build formula based on available covariates
  covariates <- c("Age_Baseline", "Gender", "APOE4_Positive")
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
  cox_summary <- summary(cox_model)
  
  # Extract Cox results
  cox_results <- data.frame(
    Variable = rownames(cox_summary$conf.int),
    HR = cox_summary$conf.int[, "exp(coef)"],
    HR_Lower = cox_summary$conf.int[, "lower .95"],
    HR_Upper = cox_summary$conf.int[, "upper .95"],
    P_Raw = cox_summary$coefficients[, "Pr(>|z|)"],
    stringsAsFactors = FALSE
  )
  
  # FDR correction 
  cox_results$P_FDR <- p.adjust(cox_results$P_Raw, method = "fdr")
  cox_results$Significant_FDR <- cox_results$P_FDR < opt$fdr_threshold
  
  cat("\n  Cox Regression Results :\n")
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
            file.path(opt$output_dir, "AIBL_Cox_Results.csv"), 
            row.names = FALSE)
  
  # AUC calculation
  if (n_events >= 10) {
    # Create risk score from Cox model
    survival_data$Risk_Score <- predict(cox_model, type = "risk")
    
    roc_obj <- roc(survival_data$AD_Conversion, survival_data$Risk_Score, quiet = TRUE)
    auc_value <- as.numeric(auc(roc_obj))
    ci_auc <- ci.auc(roc_obj)
    
    cat(sprintf("\n  AUC: %.3f (95%% CI: %.3f-%.3f)\n", 
                auc_value, ci_auc[1], ci_auc[3]))
    
    # Save AUC results
    auc_results <- data.frame(
      Cohort = "AIBL",
      N = n_samples,
      Events = n_events,
      Event_Rate = n_events / n_samples,
      AUC = auc_value,
      AUC_Lower = ci_auc[1],
      AUC_Upper = ci_auc[3]
    )
    
    write.csv(auc_results, 
              file.path(opt$output_dir, "AIBL_AUC_Results.csv"), 
              row.names = FALSE)
  }
  
  survival_completed <- TRUE
} else {
  cat("  Insufficient data for survival analysis\n")
  survival_completed <- FALSE
}

# ==============================================================================
# Summary Report
# ==============================================================================
cat("\n")

summary_lines <- c(
  "================================================================================",
  "AIBL External Validation Report ",
  "================================================================================",
  "",
  sprintf("Generated: %s", Sys.time()),
  "--------------------------------------------------------------------------------",
  "Data Summary",
  "--------------------------------------------------------------------------------",
  sprintf("  Initial samples: %d", n_initial),
  sprintf("  After follow-up filter: %d", n_final),
  sprintf("  Events: %d (%.1f%%)", sum(aibl_analysis$AD_Conversion),
          100 * mean(aibl_analysis$AD_Conversion)),
  ""
)

if (survival_completed) {
  summary_lines <- c(summary_lines,
    "--------------------------------------------------------------------------------",
    "Survival Analysis Results",
    "--------------------------------------------------------------------------------",
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
  
  if (exists("auc_value")) {
    summary_lines <- c(summary_lines,
      "",
      sprintf("  AUC: %.3f (95%% CI: %.3f-%.3f)", auc_value, ci_auc[1], ci_auc[3]))
  }
}

summary_lines <- c(summary_lines,
  "",
  "--------------------------------------------------------------------------------",
  "Output Files",
  "--------------------------------------------------------------------------------",
  sprintf("  - %s/AIBL_Predicted_Subtypes.csv", opt$output_dir),
  sprintf("  - %s/AIBL_KM_Curves.png", opt$output_dir),
  sprintf("  - %s/AIBL_Cox_Results.csv", opt$output_dir),
  sprintf("  - %s/AIBL_AUC_Results.csv", opt$output_dir),
  "",
  "================================================================================",
  "AIBL External Validation Complete",
  "================================================================================"
)

# Write report
report_path <- file.path(opt$output_dir, "AIBL_Validation_Report.txt")
writeLines(summary_lines, report_path)

cat(paste(summary_lines, collapse = "\n"))
cat("\n\n")

cat("============================================================\n")
cat("Step 20: AIBL External Validation Complete!\n")
cat("============================================================\n")
cat(sprintf("Report saved: %s\n", report_path))

