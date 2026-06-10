library(tidyverse)
library(readr)
library(writexl)
library(readxl)
library(optparse)

# ==============================================================================
# Parse Command Line Arguments
# ==============================================================================
option_list <- list(
  make_option(c("--test_file"), type = "character", 
              default = "./AI_vs_Clinician_Test/independent_test_set.csv",
              help = "Path to independent test set CSV [default: %default]"),
  make_option(c("--ai_file"), type = "character", 
              default = "./AI_vs_Clinician_Test/AI_Predictions_Final.csv",
              help = "Path to AI predictions CSV [default: %default]"),
  make_option(c("--output_dir"), type = "character", 
              default = "./AI_vs_Clinician_Test",
              help = "Output directory [default: %default]"),
  make_option(c("--n_experts"), type = "integer", default = 5,
              help = "Number of expert assessors  [default: %default]"),
  make_option(c("--seed"), type = "integer", default = 2024,
              help = "Random seed [default: %default]")
)

opt_parser <- OptionParser(option_list = option_list)
opt <- parse_args(opt_parser)

set.seed(opt$seed)

# Create output directories
dir.create(opt$output_dir, showWarnings = FALSE, recursive = TRUE)
forms_dir <- file.path(opt$output_dir, "forms")
dir.create(forms_dir, showWarnings = FALSE, recursive = TRUE)

# ==============================================================================
# Load Data
# ==============================================================================
cat("[1/4] Loading data...\n")

test_data <- read_csv(opt$test_file, show_col_types = FALSE)
cat(sprintf("  Test set loaded: %d cases\n", nrow(test_data)))

# Load AI predictions if available
ai_predictions <- NULL
if (file.exists(opt$ai_file)) {
  ai_predictions <- read_csv(opt$ai_file, show_col_types = FALSE)
  cat(sprintf("  AI predictions loaded: %s\n", basename(opt$ai_file)))
  
  if ("CaseID" %in% names(ai_predictions)) {
    test_data <- test_data %>%
      left_join(ai_predictions %>% select(CaseID, AI_Probability, AI_Risk_Level), 
                by = c("ID" = "CaseID"))
  }
}

cat(sprintf("\n  Final dataset: %d cases\n", nrow(test_data)))
cat(sprintf("  AD converters: %d (%.1f%%)\n", 
            sum(test_data$AD_Conversion),
            100 * mean(test_data$AD_Conversion)))

# ==============================================================================
# Generate Individual Assessment Forms
# ==============================================================================
cat("\n[2/4] Generating individual expert assessment forms...\n")

#' Create assessment form for a single case
#' @param case_data Single row of case data
#' @param output_path Path to save the form
create_assessment_form <- function(case_data, output_path) {
  
  # Extract case information
  case_info <- list(
    CaseID = case_data$ID,
    RID = case_data$RID,
    Age = round(case_data$Age),
    Gender = ifelse(case_data$Gender == 1, "Female", "Male"),
    Education = round(case_data$Education),
    MMSE = round(case_data$MMSE_Baseline),
    APOE4 = ifelse(case_data$APOE4_Positive == 1, "Positive", "Negative")
  )
  
  # Optional clinical scores
  if (!is.na(case_data$ADAS13)) case_info$ADAS13 <- round(case_data$ADAS13, 1)
  if (!is.na(case_data$CDRSB)) case_info$CDRSB <- round(case_data$CDRSB, 1)
  if (!is.na(case_data$FAQTOTAL)) case_info$FAQ <- round(case_data$FAQTOTAL, 1)
  
  # CSF biomarkers
  if (!is.na(case_data$ABETA42)) case_info$CSF_Abeta42 <- round(case_data$ABETA42, 1)
  if (!is.na(case_data$TAU_TOTAL)) case_info$CSF_Tau <- round(case_data$TAU_TOTAL, 1)
  if (!is.na(case_data$PTAU181)) case_info$CSF_pTau <- round(case_data$PTAU181, 1)
  
  # MRI features
  mri_cols <- grep("^ST", names(case_data), value = TRUE)
  for (feat in mri_cols) {
    if (is.numeric(case_data[[feat]]) && !is.na(case_data[[feat]])) {
      case_info[[feat]] <- round(case_data[[feat]], 2)
    }
  }
  
  # Build form content
  form_lines <- c(
    "EXPERT ASSESSMENT FORM - AD CONVERSION PREDICTION",
    strrep("=", 60),
    "",
    sprintf("Case ID: %s", case_info$CaseID),
    sprintf("Assessment Date: %s", Sys.Date()),
    "",
    "SECTION 1: PATIENT DEMOGRAPHICS",
    strrep("-", 40),
    sprintf("Age: %d years", case_info$Age),
    sprintf("Gender: %s", case_info$Gender),
    sprintf("Education: %d years", case_info$Education),
    sprintf("APOE4 Status: %s", case_info$APOE4),
    "",
    "SECTION 2: COGNITIVE ASSESSMENT",
    strrep("-", 40),
    sprintf("MMSE Score: %d/30", case_info$MMSE)
  )
  
  if (!is.null(case_info$ADAS13)) 
    form_lines <- c(form_lines, sprintf("ADAS-Cog 13: %.1f", case_info$ADAS13))
  if (!is.null(case_info$CDRSB)) 
    form_lines <- c(form_lines, sprintf("CDR Sum of Boxes: %.1f", case_info$CDRSB))
  if (!is.null(case_info$FAQ)) 
    form_lines <- c(form_lines, sprintf("FAQ Total: %.1f", case_info$FAQ))
  
  # CSF Biomarkers section
  form_lines <- c(form_lines, "", "SECTION 3: CSF BIOMARKERS", strrep("-", 40))
  
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
  
  # MRI Features section
  form_lines <- c(form_lines,
    "",
    "SECTION 4: MRI FEATURES (Z-scores)",
    strrep("-", 40),
    "",
    "Note: Z-scores represent standardized brain volumes",
    "  Z > 0: Above average volume (protective)",
    "  Z = 0: Average volume",
    "  Z < -1: Mild atrophy",
    "  Z < -2: Moderate atrophy (concerning)",
    "  Z < -3: Severe atrophy (high risk)",
    ""
  )
  
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
  
  mri_available <- FALSE
  for (feat in names(mri_labels)) {
    if (!is.null(case_info[[feat]])) {
      form_lines <- c(form_lines, 
        sprintf("%-30s: %6.2f (z-score)", mri_labels[[feat]], case_info[[feat]]))
      mri_available <- TRUE
    }
  }
  
  if (!mri_available) form_lines <- c(form_lines, "MRI data: Not available")

  # Stage 1 Assessment
  form_lines <- c(form_lines,
    "",
    strrep("=", 60),
    "STAGE 1 ASSESSMENT: CLINICAL DATA + BIOMARKERS (NO MRI)",
    strrep("=", 60),
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
    strrep("=", 60),
    "STAGE 2 ASSESSMENT: CLINICAL DATA + BIOMARKERS + MRI",
    strrep("=", 60),
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
    strrep("=", 60),
    "ASSESSOR INFORMATION",
    strrep("=", 60),
    "",
    "Assessor Name/ID: _______________________",
    "Specialty: _______________________",
    "Years of Experience: _______",
    "Assessment Date: _______________________",
    "Time spent (minutes): _______",
    "",
    "END OF ASSESSMENT FORM"
  )
  
  writeLines(form_lines, output_path)
}

# Generate forms for all cases
cat(sprintf("  Generating %d assessment forms...\n", nrow(test_data)))
for (i in 1:nrow(test_data)) {
  case <- test_data[i, ]
  form_file <- file.path(forms_dir, sprintf("Assessment_Form_%s.txt", case$ID))
  create_assessment_form(case, form_file)
  if (i %% 50 == 0) cat(sprintf("    Generated %d/%d forms\n", i, nrow(test_data)))
}
cat(sprintf("  Completed: %d forms saved to %s\n", nrow(test_data), forms_dir))

# ==============================================================================
# Create Excel Data Collection Template
# ==============================================================================
cat("\n[3/4] Creating Excel data collection template...\n")

excel_template_file <- file.path(opt$output_dir, "Expert_Assessment_Summary.xlsx")

expert_template <- data.frame(
  CaseID = rep(test_data$ID, each = opt$n_experts),
  RID = rep(test_data$RID, each = opt$n_experts),
  Age = rep(test_data$Age, each = opt$n_experts),
  Gender = rep(ifelse(test_data$Gender == 1, "Female", "Male"), each = opt$n_experts),
  MMSE = rep(test_data$MMSE_Baseline, each = opt$n_experts),
  APOE4 = rep(ifelse(test_data$APOE4_Positive == 1, "Positive", "Negative"), each = opt$n_experts),
  Expert = rep(paste0("Expert", 1:opt$n_experts), times = nrow(test_data)),
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

# Add AI reference if available
if (!is.null(ai_predictions) && "AI_Probability" %in% names(test_data)) {
  ai_ref <- test_data %>%
    select(ID, AI_Probability, AI_Risk_Level) %>%
    rename(CaseID = ID,
           AI_Prob_Reference = AI_Probability,
           AI_Risk_Reference = AI_Risk_Level)
  
  expert_template <- expert_template %>%
    left_join(ai_ref, by = "CaseID")
}

write_xlsx(expert_template, excel_template_file)
cat(sprintf("  Excel template saved: %s\n", excel_template_file))

# ==============================================================================
# Generate Expert Instructions
# ==============================================================================
cat("\n[4/4] Generating expert assessor instructions...\n")

instructions <- c(
  "INSTRUCTIONS FOR EXPERT ASSESSORS",
  "AI vs Clinician Comparison Study",
  strrep("=", 60),
  "",
  "Dear Expert Assessor,",
  "",
  "Thank you for participating in this study comparing AI and clinical",
  "assessment of AD conversion risk in MCI patients.",
  "",
  "STUDY OVERVIEW",
  strrep("-", 40),
  sprintf("Total cases to assess: %d", nrow(test_data)),
  "Assessment stages: 2 (Clinical only, then Clinical + MRI)",
  sprintf("Number of experts: %d", opt$n_experts),
  "Estimated time per case: 5-10 minutes",
  "",
  strrep("-", 40),
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
  strrep("-", 40),
  "Low Risk: 0-35% probability of AD conversion within 3 years",
  "Medium Risk: 35-60% probability of AD conversion within 3 years",
  "High Risk: 60-100% probability of AD conversion within 3 years",
  "",
  "CSF BIOMARKER REFERENCE VALUES",
  strrep("-", 40),
  "Abeta42: <192 pg/mL = Positive for amyloid pathology",
  "Total Tau: >300 pg/mL = Elevated",
  "p-Tau181: >27 pg/mL = Elevated",
  "",
  "MRI FEATURES INTERPRETATION",
  strrep("-", 40),
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
  strrep("-", 40),
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
  strrep("-", 40),
  "- Complete Stage 1 BEFORE viewing MRI data",
  "- Base assessments on clinical experience and judgment",
  "- All patient data is de-identified",
  "- Your assessments will be compared with AI predictions",
  "",
  "Thank you for your valuable contribution!",
  sprintf("Generated: %s", Sys.Date())
)

instructions_file <- file.path(opt$output_dir, "Instructions_for_Experts.txt")
writeLines(instructions, instructions_file)
cat(sprintf("  Expert instructions saved: %s\n", instructions_file))

# ==============================================================================
# Validate Expert Data (if available)
# ==============================================================================
cat("\n" + strrep("=", 60) + "\n")
cat("Expert Data Validation\n")
cat(strrep("=", 60) + "\n")

if (file.exists(excel_template_file)) {
  expert_data <- tryCatch({
    read_excel(excel_template_file)
  }, error = function(e) {
    cat("  Note: Excel file exists but no data filled yet\n")
    return(NULL)
  })
  
  if (!is.null(expert_data)) {
    cat(sprintf("  Expert data loaded: %d rows x %d columns\n", 
                nrow(expert_data), ncol(expert_data)))
    
    # Check completeness
    n_filled <- sum(!is.na(expert_data$Stage1_Conversion_Prob))
    completeness_pct <- 100 * n_filled / nrow(expert_data)
    cat(sprintf("  Data completeness: %d/%d rows (%.1f%%)\n", 
                n_filled, nrow(expert_data), completeness_pct))
    
    if (n_filled > 0) {
      # Normalize probabilities if needed
      for (stage in c("Stage1_Conversion_Prob", "Stage2_Conversion_Prob")) {
        if (stage %in% names(expert_data)) {
          probs <- expert_data[[stage]][!is.na(expert_data[[stage]])]
          if (length(probs) > 0 && max(probs) > 1) {
            expert_data[[stage]] <- expert_data[[stage]] / 100
          }
        }
      }
      
      # Clean and save
      expert_data_clean <- expert_data %>%
        rename_with(~"Stage1_Prob", matches("Stage1_Conversion_Prob")) %>%
        rename_with(~"Stage2_Prob", matches("Stage2_Conversion_Prob")) %>%
        filter(!is.na(Stage1_Prob) | !is.na(Stage2_Prob))
      
      cleaned_data_file <- file.path(opt$output_dir, "Expert_Predictions_Long.csv")
      write_csv(expert_data_clean, cleaned_data_file)
      cat(sprintf("  Cleaned expert data saved: %s\n", cleaned_data_file))
    }
  }
} else {
  cat("  Expert assessment template not yet created\n")
}

# ==============================================================================
# Summary
# ==============================================================================
cat("\n" + strrep("=", 60) + "\n")
cat("Step 3 Complete\n")
cat(strrep("=", 60) + "\n")
cat("  ✓ Two-stage assessment protocol implemented\n")
cat("  ✓ Stage 1: Clinical + Biomarkers (no MRI)\n")
cat("  ✓ Stage 2: Clinical + Biomarkers + MRI\n")
cat(sprintf("  ✓ %d expert assessors configured\n", opt$n_experts))
cat("  ✓ Risk categories: Low (0-35%), Medium (35-60%), High (60-100%)\n")
cat("\nOutput files:\n")
cat(sprintf("  - %s (assessment forms)\n", forms_dir))
cat(sprintf("  - %s\n", excel_template_file))
cat(sprintf("  - %s\n", instructions_file))

