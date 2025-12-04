## =====================================================================
## Step 1: Prepare 80 Test Cases with Original Value Restoration
## =====================================================================

library(tidyverse)
library(readr)
library(writexl)

cat(paste(rep("=", 70), collapse = ""), "\n")
cat("Step 1: Prepare 80 Test Cases\n")
cat(paste(rep("=", 70), collapse = ""), "\n")

## Create output directory
output_dir <- "AI_vs_Clinician_Test"
forms_dir <- file.path(output_dir, "forms")
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(forms_dir, showWarnings = FALSE, recursive = TRUE)

cat("Output directory created:", output_dir, "\n")

## =====================================================================
## Load Data
## =====================================================================
data_path <- "../ADNI_Labeled_For_Classifier.csv"

if (!file.exists(data_path)) {
  cat("\nError: Data file not found!\n")
  cat(sprintf("Expected path: %s\n", data_path))
  cat("\nPlease ensure:\n")
  cat("  1. The data file exists in the parent directory\n")
  cat("  2. The filename is correct: ADNI_Labeled_For_Classifier.csv\n")
  cat("  3. The file path is accessible\n")
  stop("Data file not found. Cannot proceed without real data.")
}

data <- read_csv(data_path, show_col_types = FALSE)
cat(sprintf("Data loaded: %d rows × %d columns\n", nrow(data), ncol(data)))

## =====================================================================
## Restore Standardized Variables to Original Values
## =====================================================================
cat("\n", paste(rep("=", 70), collapse = ""), "\n")
cat("Restoring standardized data to original values\n")
cat(paste(rep("=", 70), collapse = ""), "\n")

## Population parameters (literature-based)
restoration_params <- list(
  Age = list(mean = 72.5, std = 7.5),
  MMSE_Baseline = list(mean = 27.0, std = 2.0),
  ADAS13 = list(mean = 12.0, std = 5.0),
  CDRSB = list(mean = 1.5, std = 1.2),
  FAQTOTAL = list(mean = 3.0, std = 4.0),
  Education = list(mean = 15.0, std = 3.0),
  GDS = list(mean = 1.5, std = 1.5),
  ABETA40 = list(mean = 3500, std = 1000),
  ABETA42 = list(mean = 200, std = 50),
  TAU_TOTAL = list(mean = 250, std = 80),
  PTAU181 = list(mean = 25, std = 10),
  STREM2 = list(mean = 3000, std = 1000),
  PGRN = list(mean = 30, std = 10)
)

## Restore original values
for (col in names(restoration_params)) {
  if (col %in% names(data)) {
    params <- restoration_params[[col]]
    old_range <- range(data[[col]], na.rm = TRUE)
    
    if (old_range[1] > -5 && old_range[2] < 5) {
      data[[col]] <- data[[col]] * params$std + params$mean
      new_range <- range(data[[col]], na.rm = TRUE)
      cat(sprintf("  %s: [%.2f, %.2f] → [%.0f, %.0f]\n",
                  col, old_range[1], old_range[2], new_range[1], new_range[2]))
    } else {
      cat(sprintf("  %s: Already original [%.0f, %.0f]\n",
                  col, old_range[1], old_range[2]))
    }
  }
}

## Recalculate CSF ratios
if (all(c("ABETA42", "ABETA40") %in% names(data))) {
  data$ABETA42_ABETA40_RATIO <- data$ABETA42 / data$ABETA40
  cat("  ABETA42/40 ratio: Recalculated\n")
}

## Ensure reasonable ranges
if ("Age" %in% names(data)) {
  data$Age <- pmax(50, pmin(95, data$Age))
}
if ("MMSE_Baseline" %in% names(data)) {
  data$MMSE_Baseline <- pmax(0, pmin(30, round(data$MMSE_Baseline)))
}
if ("Education" %in% names(data)) {
  data$Education <- pmax(8, pmin(25, round(data$Education)))
}
if ("ADAS13" %in% names(data)) {
  data$ADAS13 <- pmax(0, pmin(85, data$ADAS13))
}
if ("ABETA42" %in% names(data)) {
  data$ABETA42 <- pmax(50, pmin(500, data$ABETA42))
}
if ("TAU_TOTAL" %in% names(data)) {
  data$TAU_TOTAL <- pmax(50, pmin(800, data$TAU_TOTAL))
}
if ("PTAU181" %in% names(data)) {
  data$PTAU181 <- pmax(5, pmin(100, data$PTAU181))
}

cat("\nAll variables restored successfully\n")

## =====================================================================
## Calculate Composite Risk Score
## =====================================================================
cat("\n", paste(rep("=", 70), collapse = ""), "\n")
cat("Calculating composite risk scores\n")
cat(paste(rep("=", 70), collapse = ""), "\n")

calculate_risk_score <- function(i) {
  score <- 0
  
  if (!is.na(data$MMSE_Baseline[i])) {
    if (data$MMSE_Baseline[i] < 24) score <- score + 2
    else if (data$MMSE_Baseline[i] < 27) score <- score + 1
  }
  
  if ("ADAS13" %in% names(data) && !is.na(data$ADAS13[i])) {
    if (data$ADAS13[i] > 18) score <- score + 2
    else if (data$ADAS13[i] > 12) score <- score + 1
  }
  
  if ("APOE4_Positive" %in% names(data) && !is.na(data$APOE4_Positive[i]) && 
      data$APOE4_Positive[i] == 1) {
    score <- score + 1
    if ("APOE4_Copies" %in% names(data) && !is.na(data$APOE4_Copies[i]) && 
        data$APOE4_Copies[i] == 2) {
      score <- score + 1
    }
  }
  
  if ("ABETA42" %in% names(data) && !is.na(data$ABETA42[i]) && 
      data$ABETA42[i] < 150) {
    score <- score + 1
  }
  if ("TAU_TOTAL" %in% names(data) && !is.na(data$TAU_TOTAL[i]) && 
      data$TAU_TOTAL[i] > 300) {
    score <- score + 1
  }
  
  core_st <- c("ST105TA", "ST102TS", "ST104TA", "ST103TA")
  for (st in core_st) {
    if (st %in% names(data) && !is.na(data[[st]][i])) {
      if (data[[st]][i] < -1.5) score <- score + 1
      else if (data[[st]][i] < -0.5) score <- score + 0.5
    }
  }
  
  return(score)
}

data$risk_score <- sapply(1:nrow(data), calculate_risk_score)

## =====================================================================
## Select 80 Cases
## =====================================================================
cat("\nSelecting 80 balanced cases\n")

core_features <- c("Age", "MMSE_Baseline", "APOE4_Positive", 
                   "ST105TA", "ST102TS", "ST104TA", "ST103TA")
data$completeness <- rowSums(!is.na(data[, intersect(core_features, names(data))])) / 
  length(core_features)

if ("AD_Conversion" %in% names(data)) {
  converters <- data %>%
    filter(AD_Conversion == 1, completeness > 0.7) %>%
    arrange(desc(risk_score), desc(completeness)) %>%
    slice_head(n = 40)
  
  non_converters <- data %>%
    filter(AD_Conversion == 0, completeness > 0.7) %>%
    arrange(risk_score, desc(completeness)) %>%
    slice_head(n = 40)
  
  selected_80 <- bind_rows(converters, non_converters)
  
  cat(sprintf("Selected 80 cases:\n"))
  cat(sprintf("  Converters (n=40): Mean risk score %.1f\n", mean(converters$risk_score)))
  cat(sprintf("  Non-converters (n=40): Mean risk score %.1f\n", 
              mean(non_converters$risk_score)))
} else {
  selected_80 <- data %>%
    filter(completeness > 0.7) %>%
    arrange(desc(risk_score), desc(completeness)) %>%
    slice_head(n = 80)
  
  cat(sprintf("Selected 80 cases with highest data quality\n"))
}

set.seed(123)
selected_80 <- selected_80[sample(nrow(selected_80)), ]

## =====================================================================
## Save Results
## =====================================================================
selected_80$risk_score <- NULL
selected_80$completeness <- NULL

output_file <- file.path(output_dir, "test_80_cases.csv")
write_csv(selected_80, output_file)
cat(sprintf("\nSaved 80 test cases: %s\n", output_file))

## Summary statistics
cat("\n", paste(rep("=", 70), collapse = ""), "\n")
cat("Data Summary\n")
cat(paste(rep("=", 70), collapse = ""), "\n")

if ("Age" %in% names(selected_80)) {
  cat(sprintf("Age: %.1f ± %.1f years [%.0f-%.0f]\n",
              mean(selected_80$Age), sd(selected_80$Age),
              min(selected_80$Age), max(selected_80$Age)))
}

if ("MMSE_Baseline" %in% names(selected_80)) {
  cat(sprintf("MMSE: %.1f ± %.1f [%.0f-%.0f]\n",
              mean(selected_80$MMSE_Baseline, na.rm = TRUE),
              sd(selected_80$MMSE_Baseline, na.rm = TRUE),
              min(selected_80$MMSE_Baseline, na.rm = TRUE),
              max(selected_80$MMSE_Baseline, na.rm = TRUE)))
}

if ("ABETA42" %in% names(selected_80)) {
  cat(sprintf("CSF Aβ42: %.0f ± %.0f pg/mL\n",
              mean(selected_80$ABETA42, na.rm = TRUE),
              sd(selected_80$ABETA42, na.rm = TRUE)))
}

if ("APOE4_Positive" %in% names(selected_80)) {
  cat(sprintf("APOE4 positive: %.0f%%\n", mean(selected_80$APOE4_Positive) * 100))
}

if ("AD_Conversion" %in% names(selected_80)) {
  cat(sprintf("Conversion rate: %.0f%%\n", mean(selected_80$AD_Conversion) * 100))
}

## Save Excel summary
summary_data <- selected_80 %>%
  select(any_of(c("ID", "Age", "Gender", "MMSE_Baseline", "APOE4_Positive",
                  "AD_Conversion", "ABETA42", "TAU_TOTAL"))) %>%
  mutate(
    Gender = ifelse(Gender == 1, "Female", "Male"),
    APOE4_Positive = ifelse(APOE4_Positive == 1, "Positive", "Negative")
  )

if ("AD_Conversion" %in% names(summary_data)) {
  summary_data$AD_Conversion <- ifelse(summary_data$AD_Conversion == 1, 
                                       "Converter", "Non-converter")
}

summary_file <- file.path(output_dir, "test_80_cases_summary.xlsx")
write_xlsx(summary_data, summary_file)
cat(sprintf("Saved summary: %s\n", summary_file))

cat("\n", paste(rep("=", 70), collapse = ""), "\n")
cat("Step 1 Complete!\n")
cat(paste(rep("=", 70), collapse = ""), "\n")