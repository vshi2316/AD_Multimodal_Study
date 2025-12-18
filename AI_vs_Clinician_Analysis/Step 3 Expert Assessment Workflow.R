## step3:Expert Assessment Workflow: Form Generation & Data Validation
## Core Logic: Generate expert assessment forms from AI predictions and validate collected expert data
## No simulated data - all operations use real clinical/AI data

library(tidyverse)
library(readr)
library(writexl)
library(readxl)

# Basic progress header
cat("Expert Assessment Workflow: Form Generation & Data Validation\n")
cat("==============================================================\n\n")

# =====================================================================
# Unified Configuration
# =====================================================================
output_dir <- "AI_vs_Clinician_Test"
forms_dir <- file.path(output_dir, "forms")
input_file <- file.path(output_dir, "independent_test_set.csv")
excel_template_file <- file.path(output_dir, "Expert_Assessment_Summary.xlsx")

# AI prediction file paths (priority order)
ai_pred_files <- c(
  file.path(output_dir, "AI_Predictions_Final.csv")
)

# Create required directories
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(forms_dir, showWarnings = FALSE, recursive = TRUE)

# =====================================================================
# Load Base Data (Test Set + AI Predictions)
# =====================================================================
cat("Loading base dataset...\n")

# Validate test set existence
if (!file.exists(input_file)) {
  cat("ERROR: Test set file not found!\n")
  cat(sprintf("Expected path: %s\n", input_file))
  cat("Please generate the independent test set first\n")
  stop("Test set missing - process terminated")
}

# Read test set data
test_data <- read_csv(input_file, show_col_types = FALSE)
cat(sprintf("Test set loaded: %d cases\n", nrow(test_data)))

# Load AI predictions (if available)
ai_pred_file <- NULL
for (f in ai_pred_files) {
  if (file.exists(f)) {
    ai_pred_file <- f
    break
  }
}

ai_predictions <- NULL
if (!is.null(ai_pred_file)) {
  ai_predictions <- read_csv(ai_pred_file, show_col_types = FALSE)
  cat(sprintf("AI predictions loaded: %s\n", basename(ai_pred_file)))
  
  # Merge AI predictions with test data
  if ("ID" %in% names(ai_predictions)) {
    test_data <- test_data %>%
      left_join(ai_predictions %>% select(ID, AI_Probability, Risk_Group), 
                by = "ID")
    cat(sprintf("Merged AI predictions: %d cases with AI scores\n", 
                sum(!is.na(test_data$AI_Probability))))
  }
} else {
  cat("WARNING: No AI predictions found - continuing without AI reference data\n")
}

# Basic dataset statistics
cat(sprintf("\nFinal dataset: %d cases\n", nrow(test_data)))
cat(sprintf("AD converters: %d (%.1f%%)\n", 
            sum(test_data$AD_Conversion),
            100 * mean(test_data$AD_Conversion)))

# =====================================================================
# Generate Individual Expert Assessment Forms
# =====================================================================
cat("\nGenerating individual expert assessment forms...\n")

# Function to create standardized assessment form
create_assessment_form <- function(case_data, output_path) {
  
  # Case metadata compilation
  case_info <- list(
    CaseID = case_data$ID,
    RID = case_data$RID,
    Age = round(case_data$Age),
    Gender = ifelse(case_data$Gender == 1, "Female", "Male"),
    Education = round(case_data$Education),
    MMSE = round(case_data$MMSE_Baseline),
    APOE4 = ifelse(case_data$APOE4_Positive == 1, "Positive", "Negative")
  )
  
  # Add cognitive scores (if available)
  if (!is.na(case_data$ADAS13)) case_info$ADAS13 <- round(case_data$ADAS13, 1)
  if (!is.na(case_data$CDRSB)) case_info$CDRSB <- round(case_data$CDRSB, 1)
  if (!is.na(case_data$FAQTOTAL)) case_info$FAQ <- round(case_data$FAQTOTAL, 1)
  
  # Add CSF biomarkers (if available)
  if (!is.na(case_data$ABETA42)) case_info$CSF_Abeta42 <- round(case_data$ABETA42, 1)
  if (!is.na(case_data$TAU_TOTAL)) case_info$CSF_Tau <- round(case_data$TAU_TOTAL, 1)
  if (!is.na(case_data$PTAU181)) case_info$CSF_pTau <- round(case_data$PTAU181, 1)
  
  # Add MRI features (z-scores) with validation
  mri_cols <- grep("^ST", names(case_data), value = TRUE)
  for (feat in mri_cols) {
    if (is.numeric(case_data[[feat]]) && !is.na(case_data[[feat]])) {
      case_info[[feat]] <- round(case_data[[feat]], 2)
    } else {
      case_info[[feat]] <- NA
    }
  }
  
  # Compile form content
  form_lines <- c(
    "EXPERT ASSESSMENT FORM - AD CONVERSION PREDICTION",
    "==================================================",
    "",
    sprintf("Case ID: %s", case_info$CaseID),
    sprintf("Assessment Date: %s", Sys.Date()),
    "",
    "SECTION 1: PATIENT DEMOGRAPHICS",
    "----------------------------------",
    sprintf("Age: %d years", case_info$Age),
    sprintf("Gender: %s", case_info$Gender),
    sprintf("Education: %d years", case_info$Education),
    sprintf("APOE4 Status: %s", case_info$APOE4),
    "",
    "SECTION 2: COGNITIVE ASSESSMENT",
    "----------------------------------",
    sprintf("MMSE Score: %d/30", case_info$MMSE)
  )
  
  # Add additional cognitive scores
  if (!is.null(case_info$ADAS13)) form_lines <- c(form_lines, sprintf("ADAS-Cog 13: %.1f", case_info$ADAS13))
  if (!is.null(case_info$CDRSB)) form_lines <- c(form_lines, sprintf("CDR Sum of Boxes: %.1f", case_info$CDRSB))
  if (!is.null(case_info$FAQ)) form_lines <- c(form_lines, sprintf("FAQ Total: %.1f", case_info$FAQ))
  
  # CSF biomarkers section
  form_lines <- c(form_lines,
                  "",
                  "SECTION 3: CSF BIOMARKERS",
                  "----------------------------------")
  
  if (!is.null(case_info$CSF_Abeta42)) {
    form_lines <- c(form_lines, 
                    sprintf("Abeta42: %.1f pg/mL", case_info$CSF_Abeta42),
                    "  Reference: <192 pg/mL = Positive for amyloid pathology")
  } else {
    form_lines <- c(form_lines, "Abeta42: Not available")
  }
  
  if (!is.null(case_info$CSF_Tau)) {
    form_lines <- c(form_lines, 
                    sprintf("Total Tau: %.1f pg/mL", case_info$CSF_Tau),
                    "  Reference: >300 pg/mL = Elevated")
  } else {
    form_lines <- c(form_lines, "Total Tau: Not available")
  }
  
  if (!is.null(case_info$CSF_pTau)) {
    form_lines <- c(form_lines, 
                    sprintf("p-Tau181: %.1f pg/mL", case_info$CSF_pTau),
                    "  Reference: >27 pg/mL = Elevated")
  } else {
    form_lines <- c(form_lines, "p-Tau181: Not available")
  }
  
  # MRI features section
  form_lines <- c(form_lines,
                  "",
                  "SECTION 4: MRI FEATURES (Z-scores)",
                  "----------------------------------",
                  "",
                  "Note: Z-scores represent standardized brain volumes",
                  "  Z > 0: Above average volume (protective)",
                  "  Z = 0: Average volume",
                  "  Z < -1: Mild atrophy",
                  "  Z < -2: Moderate atrophy (concerning)",
                  "  Z < -3: Severe atrophy (high risk)",
                  "")
  
  # MRI feature labeling (FreeSurfer standardized)
  mri_labels <- list(
    ST102TS = "Hippocampus (bilateral)",
    ST103TA = "Entorhinal Cortex",
    ST104TA = "Fusiform Gyrus",
    ST105TA = "Middle Temporal Gyrus",
    ST106TA = "Inferior Temporal Gyrus",
    ST107TA = "Parahippocampal Gyrus",
    ST108TA = "Temporal Pole",
    ST109TA = "Superior Temporal Gyrus"
  )
  
  # Add available MRI features
  mri_available <- FALSE
  for (feat in names(mri_labels)) {
    if (!is.null(case_info[[feat]])) {
      form_lines <- c(form_lines, 
                      sprintf("%-30s: %6.2f (z-score)", mri_labels[[feat]], case_info[[feat]]))
      mri_available <- TRUE
    }
  }
  
  if (!mri_available) form_lines <- c(form_lines, "MRI data: Not available")
  
  # Assessment sections (two-stage)
  form_lines <- c(form_lines,
                  "",
                  "STAGE 1 ASSESSMENT: CLINICAL DATA + BIOMARKERS (NO MRI)",
                  "=======================================================",
                  "",
                  "Based on demographics, cognitive scores, APOE4, and CSF biomarkers ONLY:",
                  "(DO NOT consider MRI data at this stage)",
                  "",
                  "1. Estimated probability of AD conversion within 3 years (0-100%): _______",
                  "",
                  "2. Risk level (check one):",
                  "   [ ] Low Risk (0-35%)",
                  "   [ ] Medium Risk (35-60%)",
                  "   [ ] High Risk (60-100%)",
                  "",
                  "3. Confidence in assessment (check one):",
                  "   [ ] Low",
                  "   [ ] Medium",
                  "   [ ] High",
                  "",
                  "4. Key factors influencing your Stage 1 assessment:",
                  "   _________________________________________________________________",
                  "",
                  "STAGE 2 ASSESSMENT: CLINICAL DATA + BIOMARKERS + MRI",
                  "=======================================================",
                  "",
                  "Now considering ALL information including MRI features:",
                  "",
                  "1. Revised probability of AD conversion within 3 years (0-100%): _______",
                  "",
                  "2. Risk level (check one):",
                  "   [ ] Low Risk (0-35%)",
                  "   [ ] Medium Risk (35-60%)",
                  "   [ ] High Risk (60-100%)",
                  "",
                  "3. Confidence in assessment (check one):",
                  "   [ ] Low",
                  "   [ ] Medium",
                  "   [ ] High",
                  "",
                  "4. Impact of MRI information (check one):",
                  "   [ ] Significantly increased risk estimate (+15% or more)",
                  "   [ ] Moderately increased risk estimate (+5% to +15%)",
                  "   [ ] No significant change (-5% to +5%)",
                  "   [ ] Moderately decreased risk estimate (-5% to -15%)",
                  "   [ ] Significantly decreased risk estimate (-15% or more)",
                  "",
                  "5. How did MRI features influence your assessment?",
                  "   _________________________________________________________________",
                  "",
                  "ASSESSOR INFORMATION",
                  "=======================================================",
                  "",
                  "Assessor Name/ID: _______________________",
                  "Specialty: _______________________",
                  "Years of Experience: _______",
                  "Assessment Date: _______________________",
                  "Time spent (minutes): _______",
                  "",
                  "END OF ASSESSMENT FORM"
  )
  
  # Write form to file
  writeLines(form_lines, output_path)
}

# Generate forms for all cases
cat(sprintf("Generating %d assessment forms...\n", nrow(test_data)))
for (i in 1:nrow(test_data)) {
  case <- test_data[i, ]
  form_file <- file.path(forms_dir, sprintf("Assessment_Form_%s.txt", case$ID))
  create_assessment_form(case, form_file)
  
  # Progress update every 50 forms
  if (i %% 50 == 0) cat(sprintf("  Generated %d/%d forms\n", i, nrow(test_data)))
}

cat(sprintf("Completed form generation: %d forms saved to %s\n", nrow(test_data), forms_dir))

# =====================================================================
# Create Excel Template for Expert Data Collection
# =====================================================================
cat("\nCreating Excel data collection template...\n")

# Generate standardized template (5 experts per case)
expert_template <- data.frame(
  CaseID = rep(test_data$ID, each = 5),
  RID = rep(test_data$RID, each = 5),
  Age = rep(test_data$Age, each = 5),
  Gender = rep(ifelse(test_data$Gender == 1, "Female", "Male"), each = 5),
  MMSE = rep(test_data$MMSE_Baseline, each = 5),
  APOE4 = rep(ifelse(test_data$APOE4_Positive == 1, "Positive", "Negative"), each = 5),
  Expert = rep(paste0("Expert", 1:5), times = nrow(test_data)),
  Stage1_Conversion_Prob = NA_real_,
  Stage1_Risk_Level = NA_character_,
  Stage1_Confidence = NA_character_,
  Stage2_Conversion_Prob = NA_real_,
  Stage2_Risk_Level = NA_character_,
  Stage2_Confidence = NA_character_,
  MRI_Impact = NA_character_,
  Assessment_Date = NA_character_,
  Time_Minutes = NA_real_,
  stringsAsFactors = FALSE
)

# Add AI reference data (if available)
if (!is.null(ai_predictions) && "AI_Probability" %in% names(test_data)) {
  ai_ref <- test_data %>%
    select(ID, AI_Probability, Risk_Group) %>%
    rename(CaseID = ID,
           AI_Prob_Reference = AI_Probability,
           AI_Risk_Reference = Risk_Group)
  
  expert_template <- expert_template %>%
    left_join(ai_ref, by = "CaseID")
}

# Save Excel template
write_xlsx(expert_template, excel_template_file)
cat(sprintf("Excel template saved: %s\n", excel_template_file))
cat(sprintf("Template contains %d rows (%d cases x 5 experts)\n", 
            nrow(expert_template), nrow(test_data)))

# =====================================================================
# Generate Expert Instructions Document
# =====================================================================
cat("\nGenerating expert assessor instructions...\n")

# Compile standardized instructions
instructions <- c(
  "INSTRUCTIONS FOR EXPERT ASSESSORS",
  "AI vs Clinician Comparison Study",
  "==================================================",
  "",
  "Dear Expert Assessor,",
  "",
  "Thank you for participating in this study comparing AI and clinical",
  "assessment of AD conversion risk in MCI patients.",
  "",
  "STUDY OVERVIEW",
  "----------------------------------",
  sprintf("Total cases to assess: %d", nrow(test_data)),
  "Assessment stages: 2 (Clinical only, then Clinical + MRI)",
  "Number of experts: 5",
  "Estimated time per case: 5-10 minutes",
  sprintf("Total estimated time: %.1f-%.1f hours", 
          nrow(test_data) * 5 / 60, nrow(test_data) * 10 / 60),
  "",
  "ASSESSMENT PROCEDURE",
  "----------------------------------",
  "For each case, complete a TWO-STAGE assessment:",
  "",
  "STAGE 1: Clinical Data + Biomarkers (NO MRI)",
  "  - Review: Demographics, cognitive scores, APOE4, CSF biomarkers",
  "  - Provide: Conversion probability (0-100%), risk level, confidence",
  "  - DO NOT look at MRI data yet!",
  "",
  "STAGE 2: Clinical Data + Biomarkers + MRI",
  "  - Review: All Stage 1 data PLUS MRI features (z-scores)",
  "  - Provide: Revised probability, risk level, confidence, MRI impact",
  "",
  "RISK LEVEL DEFINITIONS",
  "----------------------------------",
  "Low Risk: 0-35% probability of AD conversion within 3 years",
  "Medium Risk: 35-60% probability of AD conversion within 3 years",
  "High Risk: 60-100% probability of AD conversion within 3 years",
  "",
  "CSF BIOMARKER REFERENCE VALUES",
  "----------------------------------",
  "Abeta42: <192 pg/mL = Positive for amyloid pathology",
  "Total Tau: >300 pg/mL = Elevated",
  "p-Tau181: >27 pg/mL = Elevated",
  "",
  "MRI FEATURES INTERPRETATION",
  "----------------------------------",
  "MRI features are provided as Z-scores (standardized values):",
  "  Z > 0: Above average volume (PROTECTIVE)",
  "  Z = 0: Average volume",
  "  Z < -1: Mild atrophy",
  "  Z < -2: Moderate atrophy (CONCERNING)",
  "  Z < -3: Severe atrophy (HIGH RISK)",
  "",
  "Key regions for AD:",
  "  - Hippocampus: Most important predictor",
  "  - Entorhinal Cortex: Early AD pathology",
  "  - Middle Temporal Gyrus: Memory function",
  "",
  "DATA SUBMISSION",
  "----------------------------------",
  "After completing all assessments:",
  "",
  "1. Fill in the Excel summary file: 'Expert_Assessment_Summary.xlsx'",
  "2. For each case, provide:",
  "   - Your expert ID (Expert1-5)",
  "   - Stage 1: Probability (0-100), Risk Level, Confidence",
  "   - Stage 2: Probability (0-100), Risk Level, Confidence",
  "   - MRI Impact category",
  "   - Assessment date and time spent",
  "3. Return the completed Excel file to the study coordinator",
  "",
  "IMPORTANT NOTES",
  "----------------------------------",
  "- Complete Stage 1 BEFORE viewing MRI data",
  "- Base assessments on clinical experience and judgment",
  "- All patient data is de-identified",
  "- Your assessments will be compared with AI predictions",
  "",
  "Thank you for your valuable contribution!",
  sprintf("Generated: %s", Sys.Date())
)

# Save instructions file
instructions_file <- file.path(output_dir, "Instructions_for_Experts.txt")
writeLines(instructions, instructions_file)
cat(sprintf("Expert instructions saved: %s\n", instructions_file))

# =====================================================================
# Expert Data Collection & Validation
# =====================================================================
cat("\n\nExpert Data Validation Workflow\n")
cat("==============================================================\n")

# Validate Excel file existence
if (!file.exists(excel_template_file)) {
  cat("ERROR: Expert assessment Excel file not found!\n")
  cat(sprintf("Expected path: %s\n", excel_template_file))
  cat("Please ensure experts have completed and returned the assessment template\n")
  stop("Expert data file missing - validation terminated")
}

# Read expert assessment data
expert_data <- tryCatch({
  read_excel(excel_template_file)
}, error = function(e) {
  cat("ERROR: Failed to read Excel file!\n")
  cat(sprintf("Error message: %s\n", e$message))
  stop("Unable to process expert assessment data")
})

cat(sprintf("Expert data loaded: %d rows x %d columns\n", nrow(expert_data), ncol(expert_data)))

# Validate required columns
required_cols <- c("CaseID", "Expert", "Stage1_Conversion_Prob", "Stage2_Conversion_Prob",
                   "Stage1_Risk_Level", "Stage2_Risk_Level", "MRI_Impact")

missing_cols <- setdiff(required_cols, names(expert_data))
if (length(missing_cols) > 0) {
  cat("ERROR: Excel file missing required columns:\n")
  for (col in missing_cols) cat(sprintf("  - %s\n", col))
  stop("Incomplete expert data columns")
}

# Check data completeness
n_filled <- sum(!is.na(expert_data$Stage1_Conversion_Prob))
completeness_pct <- 100 * n_filled / nrow(expert_data)
cat(sprintf("Data completeness: %d/%d rows (%.1f%%)\n", n_filled, nrow(expert_data), completeness_pct))

if (n_filled == 0) {
  cat("ERROR: Excel file contains no completed assessments\n")
  stop("Empty expert assessment data")
}

# Validate expert count
experts <- unique(expert_data$Expert[!is.na(expert_data$Expert)])
n_experts <- length(experts)
cat(sprintf("Number of participating experts: %d\n", n_experts))

# Validate probability ranges
cat("\nProbability value validation:\n")
for (stage in c("Stage1_Conversion_Prob", "Stage2_Conversion_Prob")) {
  if (stage %in% names(expert_data)) {
    probs <- expert_data[[stage]][!is.na(expert_data[[stage]])]
    if (length(probs) > 0) {
      # Convert percentage to decimal if needed
      if (max(probs) > 1) {
        expert_data[[stage]] <- expert_data[[stage]] / 100
        cat(sprintf("  %s: Converted from percentage to decimal\n", stage))
        probs <- probs / 100
      }
      cat(sprintf("  %s: Range [%.1f%%, %.1f%%], mean=%.1f%%\n", 
                  stage, min(probs)*100, max(probs)*100, mean(probs)*100))
    }
  }
}

# Data cleaning
expert_data_clean <- expert_data %>%
  rename_with(~"Stage1_Prob", matches("Stage1_Conversion_Prob")) %>%
  rename_with(~"Stage2_Prob", matches("Stage2_Conversion_Prob")) %>%
  filter(!is.na(Stage1_Prob) | !is.na(Stage2_Prob))

# Save cleaned expert data
cleaned_data_file <- file.path(output_dir, "Expert_Predictions_Cleaned.csv")
write_csv(expert_data_clean, cleaned_data_file)
cat(sprintf("\nCleaned expert data saved: %s\n", cleaned_data_file))

# =====================================================================
# Statistical Analysis of Expert Assessments
# =====================================================================
cat("\nExpert Assessment Statistical Analysis\n")
cat("--------------------------------------\n")

# Basic probability statistics
stage1_mean <- mean(expert_data_clean$Stage1_Prob, na.rm = TRUE) * 100
stage1_sd <- sd(expert_data_clean$Stage1_Prob, na.rm = TRUE) * 100
stage2_mean <- mean(expert_data_clean$Stage2_Prob, na.rm = TRUE) * 100
stage2_sd <- sd(expert_data_clean$Stage2_Prob, na.rm = TRUE) * 100

cat(sprintf("Stage 1 mean probability: %.1f%% (SD=%.1f%%)\n", stage1_mean, stage1_sd))
cat(sprintf("Stage 2 mean probability: %.1f%% (SD=%.1f%%)\n", stage2_mean, stage2_sd))

# MRI impact analysis
if ("MRI_Impact" %in% names(expert_data_clean)) {
  mri_impact <- table(expert_data_clean$MRI_Impact, useNA = "ifany")
  cat("\nMRI Impact Distribution:\n")
  for (impact in names(mri_impact)) {
    pct <- 100 * mri_impact[impact] / sum(mri_impact)
    cat(sprintf("  %s: %d (%.1f%%)\n", impact, mri_impact[impact], pct))
  }
}

# Probability change analysis (Stage 2 - Stage 1)
prob_change <- expert_data_clean$Stage2_Prob - expert_data_clean$Stage1_Prob
mean_change <- mean(prob_change, na.rm = TRUE) * 100
increase_count <- sum(prob_change > 0, na.rm = TRUE)
decrease_count <- sum(prob_change < 0, na.rm = TRUE)
no_change_count <- sum(prob_change == 0, na.rm = TRUE)
total_valid <- sum(!is.na(prob_change))

cat(sprintf("\nProbability change after MRI review:\n"))
cat(sprintf("  Average change: %.1f percentage points\n", mean_change))
cat(sprintf("  Increased: %d (%.1f%%)\n", increase_count, 100*increase_count/total_valid))
cat(sprintf("  Decreased: %d (%.1f%%)\n", decrease_count, 100*decrease_count/total_valid))
cat(sprintf("  Unchanged: %d (%.1f%%)\n", no_change_count, 100*no_change_count/total_valid))

# Inter-rater reliability (ICC if package available)
cat("\nInter-rater Reliability Analysis\n")
icc_available <- requireNamespace("irr", quietly = TRUE)
if (icc_available) {
  library(irr)
  
  # Prepare wide format data for ICC
  stage1_wide <- expert_data_clean %>%
    select(CaseID, Expert, Stage1_Prob) %>%
    filter(!is.na(Stage1_Prob)) %>%
    pivot_wider(names_from = Expert, values_from = Stage1_Prob) %>%
    select(-CaseID)
  
  if (ncol(stage1_wide) >= 2) {
    icc1 <- icc(stage1_wide, model = "twoway", type = "agreement", unit = "single")
    cat(sprintf("Stage 1 ICC: %.3f (95%% CI: [%.3f, %.3f])\n", 
                icc1$value, icc1$lbound, icc1$ubound))
  }
  
  stage2_wide <- expert_data_clean %>%
    select(CaseID, Expert, Stage2_Prob) %>%
    filter(!is.na(Stage2_Prob)) %>%
    pivot_wider(names_from = Expert, values_from = Stage2_Prob) %>%
    select(-CaseID)
  
  if (ncol(stage2_wide) >= 2) {
    icc2 <- icc(stage2_wide, model = "twoway", type = "agreement", unit = "single")
    cat(sprintf("Stage 2 ICC: %.3f (95%% CI: [%.3f, %.3f])\n", 
                icc2$value, icc2$lbound, icc2$ubound))
  }
} else {
  cat("Note: irr package not installed - using correlation as alternative measure\n")
  
  # Alternative correlation analysis
  stage1_wide <- expert_data_clean %>%
    select(CaseID, Expert, Stage1_Prob) %>%
    filter(!is.na(Stage1_Prob)) %>%
    pivot_wider(names_from = Expert, values_from = Stage1_Prob) %>%
    select(-CaseID)
  
  if (ncol(stage1_wide) >= 2) {
    cor_matrix <- cor(stage1_wide, use = "complete.obs")
    mean_cor <- mean(cor_matrix[upper.tri(cor_matrix)], na.rm = TRUE)
    cat(sprintf("Stage 1 average expert correlation: %.3f\n", mean_cor))
  }
}

# Case-level summary
expert_summary <- expert_data_clean %>%
  group_by(CaseID) %>%
  summarise(
    N_Experts = sum(!is.na(Stage1_Prob)),
    Stage1_Mean = mean(Stage1_Prob, na.rm = TRUE),
    Stage1_SD = sd(Stage1_Prob, na.rm = TRUE),
    Stage2_Mean = mean(Stage2_Prob, na.rm = TRUE),
    Stage2_SD = sd(Stage2_Prob, na.rm = TRUE),
    Prob_Change = Stage2_Mean - Stage1_Mean,
    .groups = 'drop'
  )

# Identify quality issues
incomplete_cases <- sum(expert_summary$N_Experts < 5)
high_disagreement <- sum(expert_summary$Stage1_SD > 0.2 | expert_summary$Stage2_SD > 0.2)

cat(sprintf("\nData Quality Summary:\n"))
cat(sprintf("  Incomplete cases (<5 experts): %d\n", incomplete_cases))
cat(sprintf("  High disagreement cases (SD>20%%): %d\n", high_disagreement))

# =====================================================================
# Generate Final Summary Report
# =====================================================================
cat("\nGenerating final expert assessment report...\n")

# Compile comprehensive report
summary_report <- c(
  "EXPERT ASSESSMENT SUMMARY REPORT",
  "==================================================",
  "",
  sprintf("Generated: %s", Sys.Date()),
  "",
  "1. Study Overview",
  sprintf("   Total cases: %d", length(unique(expert_data_clean$CaseID))),
  sprintf("   Participating experts: %d", n_experts),
  sprintf("   Total assessments: %d", nrow(expert_data_clean)),
  sprintf("   Data completeness: %.1f%%", completeness_pct),
  "",
  "2. Assessment Statistics",
  sprintf("   Stage 1 (Clinical + Biomarkers):"),
  sprintf("     Mean probability: %.1f%% (SD=%.1f%%)", stage1_mean, stage1_sd),
  sprintf("   Stage 2 (Clinical + Biomarkers + MRI):"),
  sprintf("     Mean probability: %.1f%% (SD=%.1f%%)", stage2_mean, stage2_sd),
  sprintf("   Average probability change: %.1f percentage points", mean_change),
  "",
  "3. Data Quality",
  sprintf("   Incomplete cases: %d", incomplete_cases),
  sprintf("   High disagreement cases: %d", high_disagreement),
  "",
  "4. Next Steps",
  "   - Compare expert assessments with AI predictions",
  "   - Analyze incremental value of MRI data",
  "   - Evaluate AI vs expert performance metrics"
)

# Save summary report
report_file <- file.path(output_dir, "Expert_Assessment_Summary_Report.txt")
writeLines(summary_report, report_file)
cat(sprintf("Final report saved: %s\n", report_file))

# =====================================================================
# Workflow Completion
# =====================================================================
cat("\n\nExpert Assessment Workflow Complete\n")
cat("==============================================================\n")
cat("Generated files:\n")
cat(sprintf("  1. Individual assessment forms: %s\n", forms_dir))
cat(sprintf("  2. Expert instructions: %s\n", instructions_file))
cat(sprintf("  3. Cleaned expert data: %s\n", cleaned_data_file))
cat(sprintf("  4. Summary report: %s\n", report_file))
cat("\nNext steps: Compare expert assessments with AI prediction results\n")

