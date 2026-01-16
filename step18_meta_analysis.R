library(meta)
library(metafor)
library(dplyr)
library(ggplot2)
library(optparse)

# ==============================================================================
# Parse Command Line Arguments
# ==============================================================================
option_list <- list(
  make_option(c("--perf_file"), type = "character", 
              default = "External_Validation_Performance.csv",
              help = "Path to external validation performance CSV [default: %default]"),
  make_option(c("--habs_file"), type = "character", 
              default = "HABS_Baseline_Integrated.csv",
              help = "Path to HABS integrated CSV for missing data analysis [default: %default]"),
  make_option(c("--output_dir"), type = "character", 
              default = "./results",
              help = "Output directory [default: %default]"),
  make_option(c("--i2_moderate"), type = "numeric", default = 50,
              help = "I² threshold for moderate heterogeneity"),
  make_option(c("--i2_high"), type = "numeric", default = 75,
              help = "I² threshold for high heterogeneity ")
)

opt_parser <- OptionParser(option_list = option_list)
opt <- parse_args(opt_parser)

# Create output directory
dir.create(opt$output_dir, showWarnings = FALSE, recursive = TRUE)

# ==============================================================================
# Part 1: Load and Prepare Data
# ==============================================================================
cat("[1/6] Loading and preparing data...\n")

perf_table <- read.csv(opt$perf_file, stringsAsFactors = FALSE)

# Parse AUC and CI
meta_data <- perf_table %>%
  select(Cohort, N, Events, AUC, AUC_95CI, 
         matches("Validation_Type|pTau217_Availability")) %>%
  mutate(
    # Parse CI from string format "0.xxx-0.xxx"
    AUC_Lower = as.numeric(sapply(strsplit(AUC_95CI, "-"), `[`, 1)),
    AUC_Upper = as.numeric(sapply(strsplit(AUC_95CI, "-"), `[`, 2)),
    AUC_SE = (AUC_Upper - AUC_Lower) / (2 * 1.96),
    Event_Rate = Events / N
  )

# Add validation type if not present
if (!"Validation_Type" %in% colnames(meta_data)) {
  meta_data$Validation_Type <- "External"
}

cat(sprintf("  Cohorts included: %d\n", nrow(meta_data)))
cat(sprintf("  Total N: %d\n", sum(meta_data$N)))
cat(sprintf("  Total Events: %d\n\n", sum(meta_data$Events)))

# Display cohort summary
cat("Cohort Summary:\n")
summary_table <- meta_data %>%
  mutate(
    Event_Pct = sprintf("%.1f%%", 100 * Event_Rate),
    AUC_CI = sprintf("%.3f (%.3f–%.3f)", AUC, AUC_Lower, AUC_Upper)
  ) %>%
  select(Cohort, N, Events, Event_Pct, AUC_CI)

print(summary_table)
cat("\n")

# ==============================================================================
# Part 2: Random-Effects Meta-Analysis 
# ==============================================================================
cat("[2/6] Random-effects meta-analysis...\n")

# Perform meta-analysis using DerSimonian-Laird estimator
meta_result <- metagen(
  TE = meta_data$AUC,
  seTE = meta_data$AUC_SE,
  studlab = meta_data$Cohort,
  n.e = meta_data$N,
  data = meta_data,
  sm = "AUC",
  common = FALSE,
  random = TRUE,
  method.tau = "DL",  # DerSimonian-Laird
  method.random.ci = "HK",  # Hartung-Knapp adjustment
  title = "External Validation Meta-analysis"
)

# Extract results
pooled_auc <- meta_result$TE.random
pooled_lower <- meta_result$lower.random
pooled_upper <- meta_result$upper.random
i2_value <- meta_result$I2
tau2_value <- meta_result$tau2
q_value <- meta_result$Q
q_pvalue <- meta_result$pval.Q

# Interpret heterogeneity 
if (i2_value > opt$i2_high) {
  heterogeneity_level <- "High"
} else if (i2_value > opt$i2_moderate) {
  heterogeneity_level <- "Moderate"
} else {
  heterogeneity_level <- "Low"
}

cat(sprintf("\nPooled AUC: %.3f (95%% CI: %.3f–%.3f)\n", 
            pooled_auc, pooled_lower, pooled_upper))
cat(sprintf("Heterogeneity: I² = %.1f%% (%s), τ² = %.4f\n", 
            i2_value, heterogeneity_level, tau2_value))
cat(sprintf("Cochran's Q: %.2f (p = %.4f)\n\n", q_value, q_pvalue))

# ==============================================================================
# Part 3: Forest Plot
# ==============================================================================
cat("[3/6] Generating forest plot...\n")

png(file.path(opt$output_dir, "step18_fig1_meta_forest.png"),
    width = 4000, height = 2800, res = 300)

forest(meta_result,
       sortvar = meta_data$N,
       prediction = TRUE,
       print.tau2 = TRUE,
       col.square = "navy",
       col.square.lines = "navy",
       col.diamond = "darkred",
       col.diamond.lines = "darkred",
       col.predict = "purple",
       print.I2 = TRUE,
       print.I2.ci = TRUE,
       digits = 3,
       common = FALSE,
       random = TRUE,
       leftcols = c("studlab", "n.e", "effect", "ci"),
       leftlabs = c("Cohort", "N", "AUC", "95% CI"),
       smlab = "",
       rightcols = c("w.random"),
       rightlabs = c("Weight"),
       fontsize = 11,
       fs.heading = 13,
       squaresize = 0.6,
       main = sprintf("External Validation Meta-analysis (I² = %.1f%%, %s heterogeneity)",
                      i2_value, heterogeneity_level))

dev.off()
cat("  Saved: step18_fig1_meta_forest.png\n\n")

# ==============================================================================
# Part 4: Sensitivity Analysis - Leave-One-Out 
# ==============================================================================
cat("[4/6] Sensitivity analysis (leave-one-out)...\n")

sensitivity_results <- data.frame()

for (i in 1:nrow(meta_data)) {
  # Exclude cohort i
  meta_subset <- meta_data[-i, ]
  
  if (nrow(meta_subset) >= 2) {
    meta_loo <- metagen(
      TE = meta_subset$AUC,
      seTE = meta_subset$AUC_SE,
      studlab = meta_subset$Cohort,
      n.e = meta_subset$N,
      data = meta_subset,
      sm = "AUC",
      common = FALSE,
      random = TRUE,
      method.tau = "DL"
    )
    
    sensitivity_results <- rbind(sensitivity_results, data.frame(
      Excluded_Cohort = meta_data$Cohort[i],
      Pooled_AUC = meta_loo$TE.random,
      CI_Lower = meta_loo$lower.random,
      CI_Upper = meta_loo$upper.random,
      I2 = meta_loo$I2,
      Change_AUC = meta_loo$TE.random - pooled_auc
    ))
  }
}

cat("\nLeave-One-Out Sensitivity Analysis:\n")
for (i in 1:nrow(sensitivity_results)) {
  cat(sprintf("  Excluding %s: AUC = %.3f (%.3f–%.3f), I² = %.1f%%, ΔAUC = %+.3f\n",
              sensitivity_results$Excluded_Cohort[i],
              sensitivity_results$Pooled_AUC[i],
              sensitivity_results$CI_Lower[i],
              sensitivity_results$CI_Upper[i],
              sensitivity_results$I2[i],
              sensitivity_results$Change_AUC[i]))
}

write.csv(sensitivity_results, 
          file.path(opt$output_dir, "step18_sensitivity_analysis.csv"), 
          row.names = FALSE)
cat("\n")

# ==============================================================================
# Part 5: Weight Distribution and Heterogeneity Analysis
# ==============================================================================
cat("[5/6] Analyzing weight distribution and heterogeneity sources...\n")

# Calculate weights
weights <- meta_result$w.random
weight_data <- data.frame(
  Cohort = meta_data$Cohort,
  Weight_Pct = 100 * weights / sum(weights),
  N = meta_data$N,
  Events = meta_data$Events,
  Event_Rate = meta_data$Event_Rate,
  AUC = meta_data$AUC
) %>%
  arrange(desc(Weight_Pct))

cat("\nCohort Weights:\n")
for (i in 1:nrow(weight_data)) {
  cat(sprintf("  %s: %.1f%% (N=%d, Events=%d, Rate=%.1f%%, AUC=%.3f)\n",
              weight_data$Cohort[i],
              weight_data$Weight_Pct[i],
              weight_data$N[i],
              weight_data$Events[i],
              100 * weight_data$Event_Rate[i],
              weight_data$AUC[i]))
}

# Heterogeneity sources
sample_ratio <- max(meta_data$N) / min(meta_data$N)
event_rate_range <- max(meta_data$Event_Rate) - min(meta_data$Event_Rate)
auc_range <- max(meta_data$AUC) - min(meta_data$AUC)

cat(sprintf("\nHeterogeneity Sources:\n"))
cat(sprintf("  Sample size ratio (max/min): %.1f\n", sample_ratio))
cat(sprintf("  Event rate range: %.1f%% (%.1f%% – %.1f%%)\n", 
            100 * event_rate_range, 
            100 * min(meta_data$Event_Rate), 
            100 * max(meta_data$Event_Rate)))
cat(sprintf("  AUC range: %.3f (%.3f – %.3f)\n\n", 
            auc_range, min(meta_data$AUC), max(meta_data$AUC)))

# Weight distribution plot
png(file.path(opt$output_dir, "step18_fig2_weight_distribution.png"),
    width = 3200, height = 2400, res = 300)

par(mar = c(5, 10, 4, 2))
barplot(weight_data$Weight_Pct,
        names.arg = weight_data$Cohort,
        horiz = TRUE,
        col = c("#E41A1C", "#377EB8", "#4DAF4A", "#984EA3")[1:nrow(weight_data)],
        xlab = "Weight (%)",
        main = "Weight Distribution in Random-Effects Meta-analysis",
        cex.names = 1.2,
        cex.lab = 1.3,
        cex.main = 1.5,
        las = 1,
        xlim = c(0, max(weight_data$Weight_Pct) * 1.3))

# Add labels
text(weight_data$Weight_Pct + max(weight_data$Weight_Pct) * 0.05,
     seq(0.7, by = 1.2, length.out = nrow(weight_data)),
     sprintf("%.1f%% (N=%d)", weight_data$Weight_Pct, weight_data$N),
     cex = 1.0, adj = 0)

dev.off()
cat("  Saved: step18_fig2_weight_distribution.png\n")

# Funnel plot
png(file.path(opt$output_dir, "step18_fig3_funnel_plot.png"),
    width = 3200, height = 2800, res = 300)

funnel(meta_result,
       xlab = "AUC",
       studlab = TRUE,
       cex = 1.3,
       cex.studlab = 1.1,
       col = "navy",
       bg = "lightblue",
       pch = 21)

title(sprintf("Funnel Plot (n=%d cohorts)", nrow(meta_data)),
      cex.main = 1.5, font.main = 2)

dev.off()
cat("  Saved: step18_fig3_funnel_plot.png\n\n")

# ==============================================================================
# Part 6: HABS Missing Data Analysis (if available)
# ==============================================================================
cat("[6/6] HABS missing data analysis...\n")

if (file.exists(opt$habs_file)) {
  habs_data <- read.csv(opt$habs_file, stringsAsFactors = FALSE)
  
  key_vars <- c("Age", "Gender", "APOE4_Positive", "MMSE_Baseline", 
                "AD_Conversion", "pTau217_Primary")
  available_vars <- key_vars[key_vars %in% colnames(habs_data)]
  
  if (length(available_vars) > 0) {
    habs_subset <- habs_data[, available_vars, drop = FALSE]
    
    missing_summary <- data.frame(
      Variable = available_vars,
      N_Total = nrow(habs_subset),
      N_Missing = colSums(is.na(habs_subset)),
      Pct_Missing = 100 * colMeans(is.na(habs_subset)),
      N_Available = colSums(!is.na(habs_subset))
    )
    
    cat("\nHABS Missing Data Summary:\n")
    for (i in 1:nrow(missing_summary)) {
      cat(sprintf("  %s: %d/%d available (%.1f%% missing)\n",
                  missing_summary$Variable[i],
                  missing_summary$N_Available[i],
                  missing_summary$N_Total[i],
                  missing_summary$Pct_Missing[i]))
    }
    
    write.csv(missing_summary, 
              file.path(opt$output_dir, "step18_habs_missing_analysis.csv"), 
              row.names = FALSE)
    cat("  Saved: step18_habs_missing_analysis.csv\n")
  }
} else {
  cat("  HABS file not found, skipping missing data analysis\n")
}

# ==============================================================================
# Save Results and Summary Report
# ==============================================================================
cat("\n")

# Save meta-analysis objects
save(meta_data, meta_result, weight_data, sensitivity_results,
     file = file.path(opt$output_dir, "step18_meta_analysis_results.RData"))

# Generate summary report
summary_lines <- c(
  "================================================================================",
  "Meta-Analysis Report ",
  "================================================================================",
  "",
  "--------------------------------------------------------------------------------",
  "Cohort Summary",
  "--------------------------------------------------------------------------------",
  sprintf("  Number of cohorts: %d", nrow(meta_data)),
  sprintf("  Total sample size: %d", sum(meta_data$N)),
  sprintf("  Total events: %d (%.1f%%)", sum(meta_data$Events), 
          100 * sum(meta_data$Events) / sum(meta_data$N)),
  ""
)

for (i in 1:nrow(meta_data)) {
  summary_lines <- c(summary_lines,
    sprintf("  %s: N=%d, Events=%d (%.1f%%), AUC=%.3f",
            meta_data$Cohort[i], meta_data$N[i], meta_data$Events[i],
            100 * meta_data$Event_Rate[i], meta_data$AUC[i]))
}

summary_lines <- c(summary_lines,
  "",
  "--------------------------------------------------------------------------------",
  "Meta-Analysis Results",
  "--------------------------------------------------------------------------------",
  sprintf("  Pooled AUC: %.3f (95%% CI: %.3f–%.3f)", 
          pooled_auc, pooled_lower, pooled_upper),
  sprintf("  Heterogeneity: I² = %.1f%% (%s)", i2_value, heterogeneity_level),
  sprintf("  Between-study variance: τ² = %.4f", tau2_value),
  sprintf("  Cochran's Q: %.2f (p = %.4f)", q_value, q_pvalue),
  "",
  "--------------------------------------------------------------------------------",
  "Sensitivity Analysis (Leave-One-Out)",
  "--------------------------------------------------------------------------------"
)

for (i in 1:nrow(sensitivity_results)) {
  summary_lines <- c(summary_lines,
    sprintf("  Excluding %s: AUC = %.3f, I² = %.1f%%, ΔAUC = %+.3f",
            sensitivity_results$Excluded_Cohort[i],
            sensitivity_results$Pooled_AUC[i],
            sensitivity_results$I2[i],
            sensitivity_results$Change_AUC[i]))
}

summary_lines <- c(summary_lines,
  "",
  "--------------------------------------------------------------------------------",
  "Interpretation",
  "--------------------------------------------------------------------------------",
  sprintf("  Heterogeneity level: %s (I² = %.1f%%)", heterogeneity_level, i2_value),
  ifelse(i2_value > opt$i2_high,
         "  → High heterogeneity suggests substantial between-study variability",
         ifelse(i2_value > opt$i2_moderate,
                "  → Moderate heterogeneity suggests some between-study variability",
                "  → Low heterogeneity suggests consistent effects across studies")),
  "",
  "--------------------------------------------------------------------------------",
  "Output Files",
  "--------------------------------------------------------------------------------",
  sprintf("  - %s/step18_meta_analysis_results.RData", opt$output_dir),
  sprintf("  - %s/step18_sensitivity_analysis.csv", opt$output_dir),
  sprintf("  - %s/step18_fig1_meta_forest.png", opt$output_dir),
  sprintf("  - %s/step18_fig2_weight_distribution.png", opt$output_dir),
  sprintf("  - %s/step18_fig3_funnel_plot.png", opt$output_dir),
  "",
  "================================================================================",
  "Meta-Analysis Complete",
  "================================================================================"
)

# Write report
report_path <- file.path(opt$output_dir, "Meta_Analysis_Report.txt")
writeLines(summary_lines, report_path)

cat(paste(summary_lines, collapse = "\n"))
cat("\n\n")

cat("============================================================\n")
cat("Step 18: Meta-Analysis Complete!\n")
cat("============================================================\n")
cat(sprintf("Report saved: %s\n", report_path))

