suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(jsonlite)
  library(optparse)
})

option_list <- list(
  make_option(c("--step14_dir"), type = "character", default = "./step14_results",
              help = "Directory containing step14 outputs [default: %default]"),
  make_option(c("--step2_dir"), type = "character", default = "./AI_vs_Clinician_Analysis",
              help = "Directory containing step2 outputs [default: %default]"),
  make_option(c("--step16_dir"), type = "character", default = "./step16_results",
              help = "Directory containing step16 outputs [default: %default]"),
  make_option(c("--step20_dir"), type = "character", default = "./step20_results",
              help = "Directory containing step20 outputs [default: %default]"),
  make_option(c("--step21_dir"), type = "character", default = "./step21_results",
              help = "Directory containing step21 outputs [default: %default]"),
  make_option(c("--step12_dir"), type = "character", default = "./step12_results",
              help = "Directory containing step12 outputs [default: %default]"),
  make_option(c("--step22_dir"), type = "character", default = "./step22_results",
              help = "Directory containing step22 outputs [default: %default]"),
  make_option(c("--external_file"), type = "character", default = "External_Validation_Performance.csv",
              help = "Legacy external summary used only as a fallback [default: %default]"),
  make_option(c("--output_dir"), type = "character", default = "./step18_results",
              help = "Output directory [default: %default]")
)

opt <- parse_args(OptionParser(option_list = option_list))
dir.create(opt$output_dir, showWarnings = FALSE, recursive = TRUE)

safe_csv <- function(path) {
  if (!file.exists(path)) return(NULL)
  read.csv(path, stringsAsFactors = FALSE)
}

safe_json <- function(path) {
  if (!file.exists(path)) return(NULL)
  fromJSON(path)
}

parse_ci_string <- function(x) {
  parts <- strsplit(as.character(x), "-")[[1]]
  if (length(parts) != 2) return(c(NA_real_, NA_real_))
  c(as.numeric(parts[1]), as.numeric(parts[2]))
}

format_result <- function(auc, lower, upper) {
  sprintf("%.3f (%.3f-%.3f)", auc, lower, upper)
}

cat("==============================================================================\n")
cat("Step 18: Structured Evidence Synthesis\n")
cat("==============================================================================\n\n")

evidence <- list()

cat("[1/4] Loading discovery evidence\n")
evidence$discovery_conversion <- safe_csv(file.path(opt$step14_dir, "Conversion_Rates.csv"))
evidence$discovery_or <- safe_csv(file.path(opt$step14_dir, "Logistic_Regression_ORs.csv"))
evidence$cox_time_meta <- safe_csv(file.path(opt$step14_dir, "Cox_Time_Source_Metadata.csv"))
evidence$mri_heterogeneity <- safe_csv(file.path(opt$step22_dir, "MRI_Heterogeneity_Results.csv"))
evidence$network_results <- safe_csv(file.path(opt$step22_dir, "Network_Level_Results.csv"))
evidence$signature_stats <- safe_csv(file.path(opt$step12_dir, "Subtype_Signature_Stats_FDR.csv"))

cat("[2/4] Loading holdout evidence\n")
evidence$test_results <- safe_json(file.path(opt$step2_dir, "AI_test_results.json"))
if (is.null(evidence$test_results)) {
  stop("AI_test_results.json not found. Re-run Step 2 before running Step 18.")
}

cat("[3/4] Loading external cohort evidence\n")
evidence$aibl_summary <- safe_csv(file.path(opt$step20_dir, "step20_aibl_summary.csv"))
evidence$a4_summary <- safe_csv(file.path(opt$step21_dir, "step21_a4_summary.csv"))
evidence$habs_summary <- safe_csv(file.path(opt$step16_dir, "step16_manuscript_summary.csv"))
evidence$legacy_external <- NULL

if (is.null(evidence$aibl_summary) && is.null(evidence$a4_summary) && is.null(evidence$habs_summary)) {
  evidence$legacy_external <- safe_csv(opt$external_file)
  if (!is.null(evidence$legacy_external)) {
    cat("  Warning: falling back to legacy external summary file\n")
  }
}

cat("[4/4] Building structured synthesis table\n")

discovery_row <- data.frame(
  Cohort = "ADNI Discovery",
  N = 157,
  Population = "MCI (CSF-confirmed)",
  Validation_Dimension = "Biological Heterogeneity",
  Primary_Metric = "Conversion rate gradient",
  Result = NA_character_,
  P_Value = NA_character_,
  MRI_Significant = NA_character_,
  MRI_Top_Effect = NA_character_,
  OR_Note = "Logistic OR is the primary discovery effect size",
  stringsAsFactors = FALSE
)

if (!is.null(evidence$discovery_conversion)) {
  cr <- evidence$discovery_conversion %>% arrange(rate)
  discovery_row$Result <- sprintf("%.1f%% vs %.1f%% vs %.1f%%",
                                  cr$rate[1] * 100,
                                  cr$rate[2] * 100,
                                  cr$rate[3] * 100)
  discovery_row$P_Value <- "Chi square and Fisher exact reported in step14"
}

if (!is.null(evidence$mri_heterogeneity)) {
  mri <- evidence$mri_heterogeneity
  discovery_row$MRI_Significant <- sprintf("%d/%d FDR<0.05",
                                           sum(mri$Significant, na.rm = TRUE),
                                           nrow(mri))
  if (all(c("Eta2", "Anatomy") %in% colnames(mri))) {
    top_idx <- which.max(mri$Eta2)
    discovery_row$MRI_Top_Effect <- sprintf("eta2=%.3f (%s)",
                                            mri$Eta2[top_idx],
                                            mri$Anatomy[top_idx])
  }
}

test_results <- evidence$test_results
test_row <- data.frame(
  Cohort = "Independent Test",
  N = test_results$N_test,
  Population = "MCI (ADNI, non-overlapping)",
  Validation_Dimension = "Predictive Generalization",
  Primary_Metric = "AUC (Elastic Net, frozen pipeline)",
  Result = format_result(test_results$AUC,
                         test_results$AUC_95CI_Lower,
                         test_results$AUC_95CI_Upper),
  P_Value = sprintf("Sens=%.3f, Spec=%.3f", test_results$Sensitivity, test_results$Specificity),
  MRI_Significant = "N/A",
  MRI_Top_Effect = "N/A",
  OR_Note = if (!is.null(test_results$Endpoint_Window_Months)) {
    sprintf("Binary endpoint within %s months", paste(test_results$Endpoint_Window_Months, collapse = ", "))
  } else {
    "Binary endpoint"
  },
  stringsAsFactors = FALSE
)

external_rows <- list()

if (!is.null(evidence$aibl_summary)) {
  x <- evidence$aibl_summary[1, ]
  external_rows[[length(external_rows) + 1]] <- data.frame(
    Cohort = "AIBL",
    N = x$N,
    Population = "High-age",
    Validation_Dimension = x$Validation_Dimension,
    Primary_Metric = "AUC (projected subtype score)",
    Result = format_result(x$AUC, x$CI_Lower, x$CI_Upper),
    P_Value = sprintf("Conv=%.1f%%", x$Event_Rate * 100),
    MRI_Significant = "N/A",
    MRI_Top_Effect = "N/A",
    OR_Note = "Direct subtype transportability",
    stringsAsFactors = FALSE
  )
}

if (!is.null(evidence$a4_summary)) {
  x <- evidence$a4_summary[1, ]
  external_rows[[length(external_rows) + 1]] <- data.frame(
    Cohort = "A4",
    N = x$N,
    Population = "Preclinical",
    Validation_Dimension = x$Validation_Dimension,
    Primary_Metric = "AUC (projected subtype score)",
    Result = format_result(x$AUC, x$CI_Lower, x$CI_Upper),
    P_Value = sprintf("Conv=%.1f%%", x$Event_Rate * 100),
    MRI_Significant = "N/A",
    MRI_Top_Effect = "N/A",
    OR_Note = "Direct subtype transportability",
    stringsAsFactors = FALSE
  )
}

if (!is.null(evidence$habs_summary)) {
  matched_complete <- evidence$habs_summary %>%
    filter(Stratum == "HABS_pTau217_Subset_Complete")
  if (nrow(matched_complete) == 1) {
    x <- matched_complete[1, ]
    external_rows[[length(external_rows) + 1]] <- data.frame(
      Cohort = "HABS",
      N = x$N,
      Population = "Community-based pTau217 subset",
      Validation_Dimension = "Framework Adaptation",
      Primary_Metric = "AUC (matched subset complete model)",
      Result = format_result(x$AUC, x$CI_Lower, x$CI_Upper),
      P_Value = sprintf("Conv=%.1f%%", x$Event_Rate * 100),
      MRI_Significant = "N/A",
      MRI_Top_Effect = "N/A",
      OR_Note = "Cohort-specific adaptation with plasma p-tau217",
      stringsAsFactors = FALSE
    )
  }
}

if (length(external_rows) == 0 && !is.null(evidence$legacy_external)) {
  legacy_rows <- lapply(seq_len(nrow(evidence$legacy_external)), function(i) {
    x <- evidence$legacy_external[i, ]
    ci_vals <- parse_ci_string(x$AUC_95CI)
    data.frame(
      Cohort = x$Cohort,
      N = x$N,
      Population = x$Population,
      Validation_Dimension = "Legacy External Summary",
      Primary_Metric = "AUC (legacy summary)",
      Result = format_result(x$AUC, ci_vals[1], ci_vals[2]),
      P_Value = sprintf("Conv=%s", x$Conversion_Rate),
      MRI_Significant = "N/A",
      MRI_Top_Effect = "N/A",
      OR_Note = "Legacy fallback summary",
      stringsAsFactors = FALSE
    )
  })
  external_rows <- legacy_rows
}

master_table <- bind_rows(c(list(discovery_row, test_row), external_rows))
write.csv(master_table,
          file.path(opt$output_dir, "Evidence_Synthesis_Table.csv"),
          row.names = FALSE)

auc_rows <- master_table %>%
  filter(grepl("AUC", Primary_Metric, fixed = TRUE)) %>%
  mutate(
    AUC = as.numeric(sub(" .*", "", Result)),
    CI_Lower = as.numeric(sub(".*\((.*)-.*", "\\1", Result)),
    CI_Upper = as.numeric(sub(".*-(.*)\)", "\\1", Result)),
    Plot_Group = Validation_Dimension
  )

if (nrow(auc_rows) > 0) {
  p_auc <- ggplot(auc_rows, aes(x = reorder(Cohort, AUC), y = AUC, color = Plot_Group, size = N)) +
    geom_hline(yintercept = 0.5, linetype = 2, color = "gray60") +
    geom_point(alpha = 0.85) +
    geom_errorbar(aes(ymin = CI_Lower, ymax = CI_Upper), width = 0.15, show.legend = FALSE) +
    coord_flip() +
    scale_y_continuous(limits = c(0.4, 1.0), breaks = seq(0.4, 1.0, 0.1)) +
    labs(
      title = "AUC Summary Across Holdout and External Cohorts",
      x = NULL,
      y = "AUC",
      color = "Validation dimension",
      size = "N"
    ) +
    theme_minimal(base_size = 12)
  ggsave(file.path(opt$output_dir, "Figure_AUC_Summary.png"), p_auc, width = 10, height = 6, dpi = 300)
}

report_file <- file.path(opt$output_dir, "Evidence_Synthesis_Report.txt")
sink(report_file)
cat("========================================================================\n")
cat("STRUCTURED EVIDENCE SYNTHESIS REPORT\n")
cat(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n")
cat("========================================================================\n\n")

cat("Discovery dimension\n")
cat("  Cohort: ADNI discovery\n")
if (!is.null(evidence$discovery_conversion)) {
  cat("  Conversion gradients:\n")
  print(evidence$discovery_conversion)
}
if (!is.null(evidence$mri_heterogeneity)) {
  cat(sprintf("  Significant MRI features: %d of %d\n",
              sum(evidence$mri_heterogeneity$Significant, na.rm = TRUE),
              nrow(evidence$mri_heterogeneity)))
}
if (!is.null(evidence$cox_time_meta)) {
  cat(sprintf("  Cox time source: %s\n", evidence$cox_time_meta$Time_Source[1]))
}
cat("  Logistic regression remains the primary discovery effect size in this synthesis.\n\n")

cat("Holdout dimension\n")
cat(sprintf("  AUC: %.3f (%.3f-%.3f)\n",
            test_results$AUC,
            test_results$AUC_95CI_Lower,
            test_results$AUC_95CI_Upper))
cat(sprintf("  Sensitivity: %.3f\n", test_results$Sensitivity))
cat(sprintf("  Specificity: %.3f\n", test_results$Specificity))
if (!is.null(test_results$Brier)) {
  cat(sprintf("  Brier: %.4f\n", test_results$Brier))
}
if (!is.null(test_results$Endpoint_Window_Months)) {
  cat(sprintf("  Endpoint window: %s months\n", paste(test_results$Endpoint_Window_Months, collapse = ", ")))
}
cat("\n")

cat("External dimensions\n")
if (length(external_rows) > 0) {
  print(bind_rows(external_rows))
} else {
  cat("  No external cohort rows were built.\n")
}

cat("\nInterpretation\n")
cat("  A4 and AIBL are treated as subtype transportability analyses.\n")
cat("  HABS is treated as framework adaptation rather than direct subtype transport.\n")
cat("  This script now avoids hardcoded holdout metrics and avoids using stale cross cohort summaries when direct step outputs are available.\n")

sink()

cat("Saved Evidence_Synthesis_Table.csv\n")
cat("Saved Evidence_Synthesis_Report.txt\n")
if (file.exists(file.path(opt$output_dir, "Figure_AUC_Summary.png"))) {
  cat("Saved Figure_AUC_Summary.png\n")
}
