library(meta)
library(metafor)
library(dplyr)
library(ggplot2)

perf_table <- read.csv("External_Validation_Performance.csv",
                       stringsAsFactors = FALSE)

meta_data <- perf_table %>%
  select(Cohort, N, Events, AUC, AUC_95CI,
         Validation_Type, pTau217_Availability) %>%
  mutate(
    AUC_Lower = as.numeric(sapply(strsplit(AUC_95CI, "-"), `[`, 1)),
    AUC_Upper = as.numeric(sapply(strsplit(AUC_95CI, "-"), `[`, 2)),
    AUC_SE    = (AUC_Upper - AUC_Lower) / (2 * 1.96),
    Event_Rate = Events / N
  ) %>%
  select(Cohort, N, Events, AUC, AUC_Lower, AUC_Upper,
         AUC_SE, Event_Rate, Validation_Type, pTau217_Availability)

cat("Cohort summary:\n")
summary_table <- meta_data %>%
  mutate(
    Event_Pct = sprintf("%.1f%%", 100 * Event_Rate),
    AUC_CI    = sprintf("%.3f (%.3f–%.3f)", AUC, AUC_Lower, AUC_Upper)
  ) %>%
  select(Cohort, N, Events, Event_Pct, AUC_CI,
         Validation_Type, pTau217_Availability)

print(summary_table)

validation_types <- unique(meta_data$Validation_Type)
cat(sprintf("\nValidation types: %s\n",
            paste(validation_types, collapse = ", ")))

## ============================================================
## 2. Random-effects meta-analysis of AUC
## ============================================================

cat("\nRandom-effects meta-analysis of AUC\n")

meta_result <- metagen(
  TE      = meta_data$AUC,
  seTE    = meta_data$AUC_SE,
  studlab = meta_data$Cohort,
  n.e     = meta_data$N,
  data    = meta_data,
  sm      = "AUC",
  common  = FALSE,
  random  = TRUE,
  method.tau      = "DL",
  method.random.ci = "HK",
  title   = "External Validation Meta-analysis (Corrected)"
)

pooled_auc   <- meta_result$TE.random
pooled_lower <- meta_result$lower.random
pooled_upper <- meta_result$upper.random
i2_value     <- meta_result$I2
tau2_value   <- meta_result$tau2
q_value      <- meta_result$Q
q_pvalue     <- meta_result$pval.Q

cat(sprintf("Pooled AUC: %.3f (95%% CI %.3f–%.3f)\n",
            pooled_auc, pooled_lower, pooled_upper))
cat(sprintf("Heterogeneity: I^2 = %.1f%%, tau^2 = %.4f\n",
            i2_value, tau2_value))
cat(sprintf("Cochran Q: %.2f (p=%.4f)\n", q_value, q_pvalue))

## ============================================================
## 3. Forest plot
## ============================================================

png("step18_fig1_meta_forest_corrected.png",
    width = 4000, height = 2800, res = 300)

forest(meta_result,
       sortvar           = meta_data$N,
       prediction        = TRUE,
       print.tau2        = TRUE,
       col.square        = "navy",
       col.square.lines  = "navy",
       col.diamond       = "darkred",
       col.diamond.lines = "darkred",
       col.predict       = "purple",
       print.I2          = TRUE,
       print.I2.ci       = TRUE,
       digits            = 3,
       common            = FALSE,
       random            = TRUE,
       leftcols          = c("studlab", "n.e", "effect", "ci"),
       leftlabs          = c("Cohort", "N", "AUC", "95% CI"),
       smlab             = "",
       rightcols         = c("w.random"),
       rightlabs         = c("Weight"),
       fontsize          = 11,
       fs.heading        = 13,
       squaresize        = 0.6,
       main = "External Validation Meta-analysis (Corrected)")

dev.off()
cat("Saved forest plot: step18_fig1_meta_forest_corrected.png\n")

## ============================================================
## 4. Heterogeneity and weight distribution
## ============================================================

cat("\nHeterogeneity sources and weight distribution\n")

sample_ratio   <- max(meta_data$N) / min(meta_data$N)
event_rate_diff <- max(meta_data$Event_Rate) - min(meta_data$Event_Rate)
auc_range      <- max(meta_data$AUC) - min(meta_data$AUC)

cat(sprintf("Sample size ratio (max/min): %.1f\n", sample_ratio))
cat(sprintf("Event rate difference: %.1f%%\n", 100 * event_rate_diff))
cat(sprintf("AUC range: %.3f (%.3f–%.3f)\n",
            auc_range, min(meta_data$AUC), max(meta_data$AUC)))

weights <- meta_result$w.random
weight_data <- data.frame(
  Cohort = meta_data$Cohort,
  Weight = 100 * weights / sum(weights),
  N      = meta_data$N,
  AUC    = meta_data$AUC
)

cat("\nCohort weights:\n")
for (i in seq_len(nrow(weight_data))) {
  cat(sprintf("  %s: %.1f%% (N=%d, AUC=%.3f)\n",
              weight_data$Cohort[i],
              weight_data$Weight[i],
              weight_data$N[i],
              weight_data$AUC[i]))
}

png("step18_fig2_weight_distribution.png",
    width = 3200, height = 2400, res = 300)

par(mar = c(5, 8, 4, 2))
barplot(weight_data$Weight,
        names.arg = weight_data$Cohort,
        horiz = TRUE,
        col = c("#E41A1C", "#377EB8", "#4DAF4A", "#984EA3")[seq_len(nrow(weight_data))],
        xlab = "Weight (%)",
        main = "Weight Distribution in Corrected Meta-analysis",
        cex.names = 1.3,
        cex.lab = 1.4,
        cex.main = 1.6,
        las = 1,
        xlim = c(0, max(weight_data$Weight) * 1.2))

text(weight_data$Weight + max(weight_data$Weight) * 0.05,
     seq(0.7, by = 1.2, length.out = nrow(weight_data)),
     sprintf("%.1f%% (N=%d, AUC=%.3f)",
             weight_data$Weight, weight_data$N, weight_data$AUC),
     cex = 1.0, adj = 0)

dev.off()
cat("Saved weight distribution: step18_fig2_weight_distribution.png\n")

## ============================================================
## 5. Funnel plot (limited small-study bias assessment)
## ============================================================

png("step18_fig3_funnel_plot_limited.png",
    width = 3200, height = 2800, res = 300)

funnel(meta_result,
       xlab       = "AUC",
       studlab    = TRUE,
       cex        = 1.3,
       cex.studlab = 1.1,
       col        = "navy",
       bg         = "lightblue",
       pch        = 21)

n_studies <- nrow(meta_data)
title(sprintf("Funnel Plot (n=%d studies, corrected data)", n_studies),
      cex.main = 1.5, font.main = 2)

dev.off()
cat("Saved funnel plot: step18_fig3_funnel_plot_limited.png\n")

## ============================================================
## 6. HABS missing data summary (sensitivity)
## ============================================================

  habs_data <- read.csv("HABS_Baseline_Integrated.csv",
                        stringsAsFactors = FALSE)
  key_vars <- c("Age", "Gender", "APOE4_Positive",
                "MMSE_Baseline", "AD_Conversion")
  available_vars <- key_vars[key_vars %in% colnames(habs_data)]
  if (length(available_vars) > 0) {
    habs_subset <- habs_data[, available_vars]
    missing_summary <- data.frame(
      Variable   = available_vars,
      N_Missing  = colSums(is.na(habs_subset)),
      Pct_Missing = 100 * colMeans(is.na(habs_subset))
    )
    print(missing_summary)
    write.csv(missing_summary,
              "step18_habs_missing_analysis.csv",
              row.names = FALSE)
    cat("Saved HABS missing analysis: step18_habs_missing_analysis.csv\n")
  }
} else {
  cat("\nHABS data file not found. Skipping missing data summary.\n")
}

## ============================================================
## 7. Save key meta-analysis objects
## ============================================================

save(meta_data, meta_result, weight_data,
     file = "step18_meta_analysis_results.RData")

cat("\nStep 18 complete. Meta-analysis results and figures saved.\n")

