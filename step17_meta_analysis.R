library(meta)
library(metafor)
library(mice)
library(dplyr)
library(ggplot2)

cat("========================================================================\n")
cat("   Meta-Analysis and Sensitivity Analysis\n")
cat("========================================================================\n\n")

## Path configuration
if (.Platform$OS.type == "windows") {
  BASE_DIR <- "External_Validation_Results"
  AIBL_FILE <- "AIBL_Baseline_Integrated.csv"
  HABS_FILE <- "HABS_Baseline_Integrated.csv"
  A4_FILE <- "A4_Baseline_Integrated.csv"
} else {
  BASE_DIR <- "External_Validation_Results"
  AIBL_FILE <- "AIBL_Baseline_Integrated.csv"
  HABS_FILE <- "HABS_Baseline_Integrated.csv"
  A4_FILE <- "A4_Baseline_Integrated.csv"
}

OUTPUT_DIR <- file.path(BASE_DIR, "Meta_Analysis")
dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)

## Step 1: Load external validation results
cat("[Step 1/8] Loading validation results...\n")

perf_file <- file.path(BASE_DIR, "External_Validation_Performance.csv")
if (file.exists(perf_file)) {
  perf_table <- read.csv(perf_file, stringsAsFactors = FALSE)
  cat("  Loaded validation results\n")
  
  meta_data <- perf_table %>%
    select(Cohort, N, Conversion_Rate, AUC, AUC_95CI) %>%
    mutate(
      Conversion_Rate_Num = as.numeric(sub("%", "", Conversion_Rate)) / 100,
      Events = round(N * Conversion_Rate_Num),
      AUC_Lower = as.numeric(sapply(strsplit(AUC_95CI, "-"), `[`, 1)),
      AUC_Upper = as.numeric(sapply(strsplit(AUC_95CI, "-"), `[`, 2)),
      AUC_SE = (AUC_Upper - AUC_Lower) / (2 * 1.96),
      Event_Rate = Events / N
    ) %>%
    select(Cohort, N, Events, AUC, AUC_Lower, AUC_Upper, AUC_SE, Event_Rate)
  
} else {
  cat("  Using example data\n")
  
  meta_data <- data.frame(
    Cohort = c("AIBL", "HABS", "A4"),
    N = c(48, 100845, 1147),
    Events = c(18, 4343, 440),
    AUC = c(0.720, 0.513, 0.530),
    AUC_Lower = c(0.593, 0.506, 0.498),
    AUC_Upper = c(0.847, 0.519, 0.563)
  ) %>%
    mutate(
      AUC_SE = (AUC_Upper - AUC_Lower) / (2 * 1.96),
      Event_Rate = Events / N
    )
}

cat("\n  Cohort summary:\n")
print(meta_data)

sample_ratio <- max(meta_data$N) / min(meta_data$N)
cat(sprintf("\n  Sample size ratio: %.1f-fold\n", sample_ratio))
cat("\n")

## Step 2: Model consistency check
cat("[Step 2/8] Checking model consistency...\n")

cohorts_list <- list(AIBL = AIBL_FILE, HABS = HABS_FILE, A4 = A4_FILE)
consistency_check <- data.frame(
  Cohort = character(),
  Has_Age = logical(),
  Has_Gender = logical(),
  Has_APOE4 = logical(),
  Has_MMSE = logical(),
  Model_Consistent = logical(),
  stringsAsFactors = FALSE
)

for (cohort_name in names(cohorts_list)) {
  if (file.exists(cohorts_list[[cohort_name]])) {
    cohort_data <- read.csv(cohorts_list[[cohort_name]], stringsAsFactors = FALSE)
    
    has_age <- "Age" %in% colnames(cohort_data)
    has_gender <- "Gender" %in% colnames(cohort_data)
    has_apoe4 <- "APOE4_Positive" %in% colnames(cohort_data)
    has_mmse <- "MMSE_Baseline" %in% colnames(cohort_data)
    
    model_consistent <- all(c(has_age, has_gender, has_apoe4, has_mmse))
    
    consistency_check <- rbind(consistency_check, data.frame(
      Cohort = cohort_name,
      Has_Age = has_age,
      Has_Gender = has_gender,
      Has_APOE4 = has_apoe4,
      Has_MMSE = has_mmse,
      Model_Consistent = model_consistent
    ))
  }
}

if (nrow(consistency_check) > 0) {
  print(consistency_check)
  model_homogeneous <- all(consistency_check$Model_Consistent)
  if (model_homogeneous) {
    cat("\n  All cohorts use consistent features\n")
  } else {
    cat("\n  Feature inconsistency detected\n")
  }
} else {
  model_homogeneous <- NA
}

cat("\n")

## Step 3: Random-effects meta-analysis
cat("[Step 3/8] Performing random-effects meta-analysis...\n")

meta_result <- metagen(
  TE = meta_data$AUC,
  seTE = meta_data$AUC_SE,
  studlab = meta_data$Cohort,
  n.e = meta_data$N,
  data = meta_data,
  sm = "AUC",
  common = FALSE,
  random = TRUE,
  method.tau = "DL",
  method.random.ci = "HK",
  title = "Multi-Cohort External Validation"
)

pooled_auc <- meta_result$TE.random
pooled_lower <- meta_result$lower.random
pooled_upper <- meta_result$upper.random
i2_value <- meta_result$I2
tau2_value <- meta_result$tau2
q_value <- meta_result$Q
q_pvalue <- meta_result$pval.Q

cat(sprintf("\n  Pooled AUC: %.3f (95%% CI: %.3f-%.3f)\n",
            pooled_auc, pooled_lower, pooled_upper))
cat(sprintf("  I² = %.1f%%\n", 100 * i2_value))
cat(sprintf("  τ² = %.4f\n", tau2_value))
cat(sprintf("  Q = %.2f, p = %.4f\n", q_value, q_pvalue))

weights <- meta_result$w.random
cat("\n  Weights:\n")
for (i in 1:nrow(meta_data)) {
  cat(sprintf("    %s: %.1f%%\n",
              meta_data$Cohort[i],
              100 * weights[i] / sum(weights)))
}

cat("\n")

## Step 4: Heterogeneity analysis
cat("[Step 4/8] Analyzing heterogeneity sources...\n")

event_rate_diff <- max(meta_data$Event_Rate) - min(meta_data$Event_Rate)
auc_range <- max(meta_data$AUC) - min(meta_data$AUC)

cat(sprintf("  Sample size difference: %.1f-fold\n", sample_ratio))
cat(sprintf("  Event rate difference: %.1f%%\n", 100 * event_rate_diff))
cat(sprintf("  AUC range: %.3f\n", auc_range))

meta_data$Event_Rate_Group <- ifelse(meta_data$Event_Rate > 0.10,
                                     "High (>10%)",
                                     "Low (≤10%)")

has_subgroup <- length(unique(meta_data$Event_Rate_Group)) > 1

if (has_subgroup) {
  subgroup_result <- update(meta_result,
                            subgroup = meta_data$Event_Rate_Group,
                            tau.common = FALSE)
}

cat("\n")

## Step 5: Forest plot
cat("[Step 5/8] Generating forest plot...\n")

png(file.path(OUTPUT_DIR, "Fig1_Forest_Plot.png"),
    width = 4000, height = 2800, res = 300)

forest(meta_result,
       sortvar = meta_data$N,
       prediction = TRUE,
       print.tau2 = TRUE,
       col.square = "navy",
       col.diamond = "darkred",
       print.I2 = TRUE,
       digits = 3,
       common = FALSE,
       random = TRUE,
       leftcols = c("studlab", "n.e", "effect", "ci"),
       leftlabs = c("Cohort", "N", "AUC", "95% CI"),
       rightcols = c("w.random"),
       rightlabs = c("Weight"),
       fontsize = 11,
       main = "Random-Effects Meta-Analysis",
       cex.main = 1.6)

dev.off()
cat("  Saved: Fig1_Forest_Plot.png\n")

if (has_subgroup) {
  png(file.path(OUTPUT_DIR, "Fig2_Subgroup_Forest.png"),
      width = 4000, height = 2800, res = 300)
  
  forest(subgroup_result,
         col.square = "navy",
         col.diamond = "darkred",
         leftcols = c("studlab", "n.e", "effect"),
         leftlabs = c("Cohort", "N", "AUC"),
         fontsize = 11,
         main = "Subgroup Analysis by Event Rate")
  
  dev.off()
  cat("  Saved: Fig2_Subgroup_Forest.png\n")
}

cat("\n")

## Step 6: Sensitivity analysis - Multiple imputation
cat("[Step 6/8] Sensitivity analysis - Multiple imputation...\n")

if (file.exists(HABS_FILE)) {
  habs_data <- read.csv(HABS_FILE, stringsAsFactors = FALSE)
  
  key_vars <- c("Age", "Gender", "APOE4_Positive", "MMSE_Baseline", "AD_Conversion")
  available_vars <- key_vars[key_vars %in% colnames(habs_data)]
  
  if (length(available_vars) > 0) {
    habs_subset <- habs_data[, available_vars]
    habs_complete <- na.omit(habs_subset)
    complete_rate <- nrow(habs_complete) / nrow(habs_subset)
    
    cat(sprintf("  Complete case rate: %.1f%%\n", 100 * complete_rate))
    
    perform_imputation <- any(colMeans(is.na(habs_subset)) > 0) &&
      any(colMeans(is.na(habs_subset)) < 0.5)
    
    if (perform_imputation && nrow(habs_complete) > 100) {
      cat("  Performing MICE imputation...\n")
      
      set.seed(42)
      imputed_data <- mice(habs_subset, m = 5, method = "pmm",
                          maxit = 10, printFlag = FALSE)
      
      if ("AD_Conversion" %in% available_vars) {
        complete_model <- glm(AD_Conversion ~ .,
                            data = habs_complete,
                            family = binomial)
        
        imputed_complete <- complete(imputed_data, 1)
        imputed_model <- glm(AD_Conversion ~ .,
                           data = imputed_complete,
                           family = binomial)
        
        complete_coef <- coef(summary(complete_model))
        imputed_coef <- coef(summary(imputed_model))
        
        coef_comparison <- data.frame(
          Variable = rownames(complete_coef),
          Complete_Est = complete_coef[, "Estimate"],
          Imputed_Est = imputed_coef[, "Estimate"],
          Difference = abs(complete_coef[, "Estimate"] - imputed_coef[, "Estimate"])
        )
        
        max_diff <- max(coef_comparison$Difference)
        cat(sprintf("  Max coefficient difference: %.4f\n", max_diff))
        
        write.csv(coef_comparison,
                 file.path(OUTPUT_DIR, "Table1_Sensitivity_Imputation.csv"),
                 row.names = FALSE)
        
        sensitivity_robust <- max_diff < 0.1
      } else {
        sensitivity_robust <- NA
      }
    } else {
      sensitivity_robust <- NA
    }
  } else {
    sensitivity_robust <- NA
  }
} else {
  sensitivity_robust <- NA
}

cat("\n")

## Step 7: VIF analysis
cat("[Step 7/8] Variance Inflation Factor analysis...\n")

if (file.exists(HABS_FILE)) {
  habs_data <- read.csv(HABS_FILE, stringsAsFactors = FALSE)
  
  model_vars <- c("Age", "Gender", "APOE4_Positive", "MMSE_Baseline", "AD_Conversion")
  available_vars <- model_vars[model_vars %in% colnames(habs_data)]
  
  if ("AD_Conversion" %in% available_vars && length(available_vars) > 2) {
    habs_model_data <- habs_data[, available_vars] %>% na.omit()
    
    if (nrow(habs_model_data) >= 50) {
      model <- glm(AD_Conversion ~ ., data = habs_model_data, family = binomial)
      X <- model.matrix(model)[, -1]
      
      vif_values <- numeric(ncol(X))
      names(vif_values) <- colnames(X)
      
      for (i in 1:ncol(X)) {
        y_temp <- X[, i]
        X_temp <- X[, -i, drop = FALSE]
        lm_temp <- lm(y_temp ~ X_temp)
        r_squared <- summary(lm_temp)$r.squared
        vif_values[i] <- ifelse(r_squared >= 1, Inf, 1 / (1 - r_squared))
      }
      
      vif_table <- data.frame(
        Variable = names(vif_values),
        VIF = round(vif_values, 2),
        Interpretation = ifelse(vif_values > 10, "Severe",
                               ifelse(vif_values > 5, "Moderate", "Acceptable"))
      )
      
      print(vif_table)
      
      write.csv(vif_table,
               file.path(OUTPUT_DIR, "Table2_VIF.csv"),
               row.names = FALSE)
      
      max_vif <- max(vif_values)
      cat(sprintf("\n  Max VIF: %.2f\n", max_vif))
    }
  }
}

cat("\n")

## Step 8: Publication bias
cat("[Step 8/8] Publication bias assessment...\n")

n_studies <- nrow(meta_data)
cat(sprintf("  Number of studies: %d\n", n_studies))

if (n_studies < 5) {
  cat("  Warning: < 5 studies, bias tests underpowered\n")
}

png(file.path(OUTPUT_DIR, "Fig3_Funnel_Plot.png"),
    width = 3200, height = 2800, res = 300)

funnel(meta_result, xlab = "AUC", studlab = TRUE,
       cex = 1.3, col = "navy", bg = "lightblue", pch = 21)

title(sprintf("Funnel Plot (n=%d studies)", n_studies),
      cex.main = 1.5)

dev.off()
cat("  Saved: Fig3_Funnel_Plot.png\n")

if (n_studies >= 3) {
  egger_test <- metabias(meta_result, method = "Egger")
  cat(sprintf("  Egger test: t=%.3f, p=%.4f\n",
             egger_test$statistic, egger_test$p.value))
  egger_available <- TRUE
} else {
  egger_available <- FALSE
}

cat("\n")

## Weight distribution visualization
png(file.path(OUTPUT_DIR, "Fig4_Weight_Distribution.png"),
    width = 3200, height = 2400, res = 300)

weight_data <- data.frame(
  Cohort = meta_data$Cohort,
  Weight = 100 * weights / sum(weights),
  N = meta_data$N
)

par(mar = c(5, 8, 4, 2))
barplot(weight_data$Weight,
        names.arg = weight_data$Cohort,
        horiz = TRUE,
        col = c("#E41A1C", "#377EB8", "#4DAF4A"),
        xlab = "Weight (%)",
        main = "Weight Distribution",
        cex.names = 1.3,
        las = 1,
        xlim = c(0, max(weight_data$Weight) * 1.2))

text(weight_data$Weight + max(weight_data$Weight) * 0.05,
     seq(0.7, by = 1.2, length.out = nrow(weight_data)),
     sprintf("%.1f%% (N=%d)", weight_data$Weight, weight_data$N),
     cex = 1.2, adj = 0)

dev.off()
cat("  Saved: Fig4_Weight_Distribution.png\n\n")

## Summary report
report <- c(
  "Meta-Analysis Report",
  "====================",
  "",
  sprintf("Generated: %s", Sys.time()),
  "",
  "Studies Included:",
  paste(sprintf("  %s: N=%d, AUC=%.3f",
               meta_data$Cohort, meta_data$N, meta_data$AUC),
        collapse = "\n"),
  "",
  "Pooled Results:",
  sprintf("  AUC: %.3f (95%% CI: %.3f-%.3f)",
          pooled_auc, pooled_lower, pooled_upper),
  "",
  "Heterogeneity:",
  sprintf("  I²: %.1f%%", 100 * i2_value),
  sprintf("  τ²: %.4f", tau2_value),
  sprintf("  Q: %.2f (p=%.4f)", q_value, q_pvalue),
  "",
  "Interpretation:",
  ifelse(i2_value < 0.25, "  Low heterogeneity",
         ifelse(i2_value < 0.50, "  Moderate heterogeneity",
                ifelse(i2_value < 0.75, "  Substantial heterogeneity",
                       "  High heterogeneity"))),
  "",
  "Output Files:",
  "  - Fig1_Forest_Plot.png",
  "  - Fig3_Funnel_Plot.png",
  "  - Fig4_Weight_Distribution.png"
)

writeLines(report, file.path(OUTPUT_DIR, "Meta_Analysis_Report.txt"))

cat("========================================================================\n")
cat("Meta-analysis complete!\n")
cat("========================================================================\n")
cat(sprintf("\nPooled AUC: %.3f (%.3f-%.3f)\n",
            pooled_auc, pooled_lower, pooled_upper))
cat(sprintf("I² = %.1f%%, indicating %s heterogeneity\n",
            100 * i2_value,
            ifelse(i2_value < 0.50, "low-moderate", "substantial")))
cat(sprintf("\nResults saved to: %s\n", OUTPUT_DIR))
