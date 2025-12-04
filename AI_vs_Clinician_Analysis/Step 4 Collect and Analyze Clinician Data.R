library(tidyverse)
library(readxl)
library(writexl)

cat(paste(rep("=", 70), collapse = ""), "\n")
cat("Step 4: Collect and Analyze Clinician Data\n")
cat(paste(rep("=", 70), collapse = ""), "\n")

## Setup
output_dir <- "AI_vs_Clinician_Test"
excel_file <- file.path(output_dir, "Clinician_Assessment_Template.xlsx")

## Check file existence
if (!file.exists(excel_file)) {
  cat("\nError: Clinician assessment data file not found!\n")
  cat(sprintf("Expected: %s\n", excel_file))
  cat("\nPlease ensure:\n")
  cat("  1. Assessment forms have been sent to clinicians\n")
  cat("  2. Clinicians have completed assessments\n")
  cat("  3. Excel file is saved to correct location\n")
  stop("Clinician assessment file not found")
}

## Load data
cat("\nFound clinician assessment file\n")

expert_data <- tryCatch({
  read_excel(excel_file)
}, error = function(e) {
  cat("\nError reading Excel file!\n")
  cat(sprintf("Error: %s\n", e$message))
  stop("Cannot read Excel file")
})

cat(sprintf("Loaded: %d rows × %d columns\n", nrow(expert_data), ncol(expert_data)))

## Data validation
cat("\n" , paste(rep("=", 70), collapse = ""), "\n")
cat("Data Completeness Check\n")
cat(paste(rep("=", 70), collapse = ""), "\n")

## Check required columns
required_cols <- c("CaseID", "Stage1_Probability", "Stage2_Probability",
                   "MRI_Impact", "Clinician")

missing_cols <- setdiff(required_cols, names(expert_data))
if (length(missing_cols) > 0) {
  cat("\nError: Missing required columns!\n")
  for (col in missing_cols) {
    cat(sprintf("  - %s\n", col))
  }
  stop("Excel file incomplete")
}

cat("All required columns present\n")

## Check data rows
n_cases <- nrow(expert_data) / 5
if (nrow(expert_data) %% 5 != 0) {
  cat(sprintf("\nWarning: Row count not multiple of 5\n"))
  cat(sprintf("  Current: %d rows\n", nrow(expert_data)))
}

cat(sprintf("Data rows: %d (approximately %.0f cases)\n", nrow(expert_data), n_cases))

## Check clinicians
clinicians <- unique(expert_data$Clinician)
n_clinicians <- length(clinicians)
cat(sprintf("Clinicians: %d\n", n_clinicians))

for (i in seq_along(clinicians)) {
  clinician <- clinicians[i]
  n_eval <- sum(expert_data$Clinician == clinician)
  cat(sprintf("  %s: %d assessments\n", clinician, n_eval))
}

## Check missing values
cat("\nMissing value check:\n")
for (col in required_cols) {
  n_missing <- sum(is.na(expert_data[[col]]))
  if (n_missing > 0) {
    pct_missing <- n_missing / nrow(expert_data) * 100
    cat(sprintf("  %s: %d missing (%.1f%%)\n", col, n_missing, pct_missing))
  } else {
    cat(sprintf("  %s: Complete\n", col))
  }
}

## Data processing
cat("\n" , paste(rep("=", 70), collapse = ""), "\n")
cat("Data Processing\n")
cat(paste(rep("=", 70), collapse = ""), "\n")

## Rename columns
expert_data_clean <- expert_data %>%
  rename(
    Stage1_Prob = Stage1_Probability,
    Stage2_Prob = Stage2_Probability
  )

## Convert percentages if needed
if (max(expert_data_clean$Stage1_Prob, na.rm = TRUE) > 1) {
  expert_data_clean$Stage1_Prob <- expert_data_clean$Stage1_Prob / 100
  expert_data_clean$Stage2_Prob <- expert_data_clean$Stage2_Prob / 100
  cat("Converted probabilities from percentage to decimal\n")
}

## Save long format
long_format_file <- file.path(output_dir, "Clinician_Predictions_Long.csv")
write_csv(expert_data_clean, long_format_file)
cat(sprintf("Saved long format: %s\n", long_format_file))

## Statistical analysis
cat("\n" , paste(rep("=", 70), collapse = ""), "\n")
cat("Statistical Analysis\n")
cat(paste(rep("=", 70), collapse = ""), "\n")

cat("\n1. Assessment probability statistics:\n")
cat(sprintf("  Stage 1 mean: %.1f%% (SD=%.1f%%)\n",
            mean(expert_data_clean$Stage1_Prob, na.rm = TRUE) * 100,
            sd(expert_data_clean$Stage1_Prob, na.rm = TRUE) * 100))
cat(sprintf("  Stage 2 mean: %.1f%% (SD=%.1f%%)\n",
            mean(expert_data_clean$Stage2_Prob, na.rm = TRUE) * 100,
            sd(expert_data_clean$Stage2_Prob, na.rm = TRUE) * 100))

## MRI impact
cat("\n2. MRI impact analysis:\n")
if ("MRI_Impact" %in% names(expert_data_clean)) {
  mri_impact <- table(expert_data_clean$MRI_Impact)
  for (impact in names(mri_impact)) {
    pct <- mri_impact[impact] / sum(mri_impact) * 100
    cat(sprintf("  %s: %d times (%.1f%%)\n", impact, mri_impact[impact], pct))
  }
}

## Probability change
prob_change <- expert_data_clean$Stage2_Prob - expert_data_clean$Stage1_Prob
cat("\n3. Probability change after MRI:\n")
cat(sprintf("  Mean change: %.1f percentage points\n", mean(prob_change, na.rm = TRUE) * 100))
cat(sprintf("  Increased: %d times (%.1f%%)\n",
            sum(prob_change > 0, na.rm = TRUE),
            sum(prob_change > 0, na.rm = TRUE) / length(prob_change) * 100))
cat(sprintf("  Decreased: %d times (%.1f%%)\n",
            sum(prob_change < 0, na.rm = TRUE),
            sum(prob_change < 0, na.rm = TRUE) / length(prob_change) * 100))

## Inter-rater reliability (ICC)
cat("\n" , paste(rep("=", 70), collapse = ""), "\n")
cat("Inter-Rater Reliability (ICC)\n")
cat(paste(rep("=", 70), collapse = ""), "\n")

icc_available <- requireNamespace("irr", quietly = TRUE)

if (icc_available) {
  library(irr)
  
  stage1_wide <- expert_data_clean %>%
    select(CaseID, Clinician, Stage1_Prob) %>%
    pivot_wider(names_from = Clinician, values_from = Stage1_Prob) %>%
    select(-CaseID)
  
  stage2_wide <- expert_data_clean %>%
    select(CaseID, Clinician, Stage2_Prob) %>%
    pivot_wider(names_from = Clinician, values_from = Stage2_Prob) %>%
    select(-CaseID)
  
  tryCatch({
    icc1 <- icc(stage1_wide, model = "twoway", type = "agreement", unit = "single")
    cat(sprintf("Stage 1 ICC: %.3f [%.3f, %.3f]\n", 
                icc1$value, icc1$lbound, icc1$ubound))
    
    if (icc1$value < 0.40) {
      cat("  Interpretation: Poor agreement\n")
    } else if (icc1$value < 0.60) {
      cat("  Interpretation: Fair agreement\n")
    } else if (icc1$value < 0.75) {
      cat("  Interpretation: Good agreement\n")
    } else {
      cat("  Interpretation: Excellent agreement\n")
    }
    
    icc2 <- icc(stage2_wide, model = "twoway", type = "agreement", unit = "single")
    cat(sprintf("\nStage 2 ICC: %.3f [%.3f, %.3f]\n", 
                icc2$value, icc2$lbound, icc2$ubound))
    
    if (icc2$value < 0.40) {
      cat("  Interpretation: Poor agreement\n")
    } else if (icc2$value < 0.60) {
      cat("  Interpretation: Fair agreement\n")
    } else if (icc2$value < 0.75) {
      cat("  Interpretation: Good agreement\n")
    } else {
      cat("  Interpretation: Excellent agreement\n")
    }
    
  }, error = function(e) {
    cat("ICC calculation failed:", e$message, "\n")
  })
  
} else {
  cat("irr package not installed, using correlation instead\n")
  
  stage1_wide <- expert_data_clean %>%
    select(CaseID, Clinician, Stage1_Prob) %>%
    pivot_wider(names_from = Clinician, values_from = Stage1_Prob) %>%
    select(-CaseID)
  
  cor_matrix <- cor(stage1_wide, use = "complete.obs")
  cor_vals <- cor_matrix[upper.tri(cor_matrix)]
  mean_cor <- mean(cor_vals, na.rm = TRUE)
  
  cat(sprintf("  Stage 1 mean correlation: %.3f\n", mean_cor))
  
  stage2_wide <- expert_data_clean %>%
    select(CaseID, Clinician, Stage2_Prob) %>%
    pivot_wider(names_from = Clinician, values_from = Stage2_Prob) %>%
    select(-CaseID)
  
  cor_matrix2 <- cor(stage2_wide, use = "complete.obs")
  cor_vals2 <- cor_matrix2[upper.tri(cor_matrix2)]
  mean_cor2 <- mean(cor_vals2, na.rm = TRUE)
  
  cat(sprintf("  Stage 2 mean correlation: %.3f\n", mean_cor2))
}

## Summary by case
cat("\n" , paste(rep("=", 70), collapse = ""), "\n")
cat("Summary by Case\n")
cat(paste(rep("=", 70), collapse = ""), "\n")

clinician_summary <- expert_data_clean %>%
  group_by(CaseID) %>%
  summarise(
    N_Clinicians = n(),
    Stage1_Mean = mean(Stage1_Prob, na.rm = TRUE),
    Stage1_SD = sd(Stage1_Prob, na.rm = TRUE),
    Stage2_Mean = mean(Stage2_Prob, na.rm = TRUE),
    Stage2_SD = sd(Stage2_Prob, na.rm = TRUE),
    Prob_Change = Stage2_Mean - Stage1_Mean,
    .groups = 'drop'
  )

cat(sprintf("Summarized %d cases\n", nrow(clinician_summary)))
cat(sprintf("Mean assessments per case: %.1f\n", mean(clinician_summary$N_Clinicians)))

## Quality check
cat("\n" , paste(rep("=", 70), collapse = ""), "\n")
cat("Data Quality Check\n")
cat(paste(rep("=", 70), collapse = ""), "\n")

## Incomplete cases
incomplete_cases <- clinician_summary %>% filter(N_Clinicians < 5)
if (nrow(incomplete_cases) > 0) {
  cat(sprintf("Warning: %d cases with <5 clinician assessments\n", nrow(incomplete_cases)))
} else {
  cat("All cases have 5 clinician assessments\n")
}

## High disagreement
high_disagreement <- clinician_summary %>%
  filter(Stage1_SD > 0.2 | Stage2_SD > 0.2) %>%
  arrange(desc(Stage1_SD))

if (nrow(high_disagreement) > 0) {
  cat(sprintf("\nWarning: %d cases with high disagreement (SD>20%%)\n", 
              nrow(high_disagreement)))
} else {
  cat("\nClinician agreement is reasonable\n")
}

## Generate report
report <- sprintf("
%s
Clinician Assessment Data Collection Report
%s

1. Overview
  Cases assessed: %d
  Clinicians: %d
  Total assessments: %d
  Completeness: %.1f%%

2. Assessment Probabilities
  Stage 1 (Clinical + Biomarkers):
    Mean: %.1f%% (SD=%.1f%%)
    Range: [%.1f%%, %.1f%%]
  
  Stage 2 (Add MRI):
    Mean: %.1f%% (SD=%.1f%%)
    Range: [%.1f%%, %.1f%%]
  
  MRI Impact:
    Mean change: %.1f percentage points
    Increased probability: %d times (%.1f%%)
    Decreased probability: %d times (%.1f%%)

3. Data Quality
  Incomplete assessments: %d cases
  High disagreement (SD>20%%): %d cases

%s
",
                  paste(rep("=", 70), collapse = ""),
                  paste(rep("=", 70), collapse = ""),
                  length(unique(expert_data_clean$CaseID)),
                  length(unique(expert_data_clean$Clinician)),
                  nrow(expert_data_clean),
                  (1 - sum(is.na(expert_data_clean$Stage1_Prob)) / nrow(expert_data_clean)) * 100,
                  mean(expert_data_clean$Stage1_Prob, na.rm = TRUE) * 100,
                  sd(expert_data_clean$Stage1_Prob, na.rm = TRUE) * 100,
                  min(expert_data_clean$Stage1_Prob, na.rm = TRUE) * 100,
                  max(expert_data_clean$Stage1_Prob, na.rm = TRUE) * 100,
                  mean(expert_data_clean$Stage2_Prob, na.rm = TRUE) * 100,
                  sd(expert_data_clean$Stage2_Prob, na.rm = TRUE) * 100,
                  min(expert_data_clean$Stage2_Prob, na.rm = TRUE) * 100,
                  max(expert_data_clean$Stage2_Prob, na.rm = TRUE) * 100,
                  mean(prob_change, na.rm = TRUE) * 100,
                  sum(prob_change > 0, na.rm = TRUE),
                  sum(prob_change > 0, na.rm = TRUE) / length(prob_change) * 100,
                  sum(prob_change < 0, na.rm = TRUE),
                  sum(prob_change < 0, na.rm = TRUE) / length(prob_change) * 100,
                  nrow(incomplete_cases),
                  nrow(high_disagreement),
                  paste(rep("=", 70), collapse = "")
)

## Save report
report_file <- file.path(output_dir, "Clinician_Data_Summary.txt")
writeLines(report, report_file)

cat(report)
cat(sprintf("\nReport saved: %s\n", report_file))

cat("\n" , paste(rep("=", 70), collapse = ""), "\n")
cat("Step 4 Complete!\n")
cat(paste(rep("=", 70), collapse = ""), "\n")
cat("\nOutput files:\n")
cat(sprintf("  1. %s\n", long_format_file))
cat(sprintf("  2. %s\n", report_file))
cat(paste(rep("=", 70), collapse = ""), "\n")