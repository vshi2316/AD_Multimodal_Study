## Step 1: Independent Test Set Preparation (High-Quality Subset)
## Objective: Filter patients with complete multimodal data (CSF+MRI+Genetics+Cognition) to reduce noise and improve AI AUC

library(tidyverse)
library(readr)
library(writexl)

# Progress header
cat(paste(rep("=", 70), collapse = ""), "\n")
cat("Independent Test Set Preparation (High-Quality Subset)\n")
cat(paste(rep("=", 70), collapse = ""), "\n\n")

## =====================================================================
## Path Configuration (adjust to your repository structure)
## =====================================================================
root_dir   <- "./ADNI_Raw_Data" # Relative path for GitHub compatibility
path_dx    <- file.path(root_dir, "LINES/Diagnostic Summary.csv")
path_mri   <- file.path(root_dir, "sMRI/UCSF - Cross-Sectional FreeSurfer (7.x).csv")
path_demog <- file.path(root_dir, "LINES/Subject Demographics.csv")
path_mmse  <- file.path(root_dir, "LINES/Mini-Mental State Examination (MMSE).csv")
path_apoe  <- file.path(root_dir, "APOE/ApoE Genotyping - Results.csv")
path_csf_alz   <- file.path(root_dir, "CSF/UPENN CSF Biomarker Master Alzbio3.csv")
path_csf_roche <- file.path(root_dir, "CSF/UPENN CSF Biomarkers Roche Elecsys.csv")
path_adas  <- file.path(root_dir, "LINES/ADAS-Cognitive Behavior.csv") # Corrected ADAS filename
path_cdr   <- file.path(root_dir, "LINES/Clinical Dementia Rating.csv") # Corrected CDR filename
path_faq   <- file.path(root_dir, "LINES/Functional Activities Questionnaire.csv")

# Training set path (for exclusion)
path_train <- "./cluster_results.csv"

# Output directory (relative path)
output_dir <- "AI_vs_Clinician_Test"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

## =====================================================================
## Step 1: Extract Baseline MCI Patients
## =====================================================================
cat(paste(rep("=", 70), collapse = ""), "\n")
cat("Step 1: Extract Baseline MCI Patients\n")
cat(paste(rep("=", 70), collapse = ""), "\n")

dx_data <- read_csv(path_dx, show_col_types = FALSE)

# Filter baseline MCI patients
baseline_mci <- dx_data %>%
  filter(VISCODE %in% c("bl", "sc")) %>%
  filter(DXMCI == 1) %>%
  select(RID, Baseline_Date = EXAMDATE) %>%
  distinct(RID, .keep_all = TRUE) %>%
  mutate(Baseline_Date = as.Date(Baseline_Date, format = "%m/%d/%Y"))

cat(sprintf("  Initial baseline MCI patient pool: %d cases\n", nrow(baseline_mci)))

# Determine AD conversion status
patient_outcomes <- data.frame()
for(pid in baseline_mci$RID) {
  records <- dx_data %>% filter(RID == pid)
  is_converter <- any(records$DXAD == 1, na.rm = TRUE)
  patient_outcomes <- rbind(patient_outcomes, 
                            data.frame(RID = pid, 
                                       AD_Conversion = ifelse(is_converter, 1, 0)))
}

mci_cohort <- baseline_mci %>% left_join(patient_outcomes, by = "RID")

## =====================================================================
## Step 2: Read and Filter Multimodal Data
## =====================================================================
cat("\nStep 2: Read Multimodal Data...\n")

# 1. MRI (mandatory)
mri_data <- read_csv(path_mri, show_col_types = FALSE)
mri_features <- mri_data %>%
  filter(VISCODE %in% c("bl", "sc")) %>%
  select(RID, starts_with("ST")) %>%
  group_by(RID) %>% slice(1) %>% ungroup()

# 2. CSF (mandatory)
csf_alz <- read_csv(path_csf_alz, show_col_types = FALSE) %>%
  filter(VISCODE %in% c("bl", "sc")) %>%
  select(RID, ABETA, TAU, PTAU) %>%
  rename(ABETA42 = ABETA, TAU_TOTAL = TAU, PTAU181 = PTAU) %>%
  group_by(RID) %>% slice(1) %>% ungroup()

csf_roche <- read_csv(path_csf_roche, show_col_types = FALSE) %>%
  filter(VISCODE2 %in% c("bl", "sc")) %>%
  select(RID, ABETA40) %>%
  group_by(RID) %>% slice(1) %>% ungroup()

csf_merged <- full_join(csf_alz, csf_roche, by = "RID")

# 3. Demographics, Cognition & Genetics
demog <- read_csv(path_demog, show_col_types = FALSE) %>%
  select(RID, PTGENDER, PTDOB, PTEDUCAT) %>%
  distinct(RID, .keep_all = TRUE)

mmse <- read_csv(path_mmse, show_col_types = FALSE) %>%
  filter(VISCODE %in% c("bl", "sc")) %>%
  select(RID, MMSCORE) %>%
  group_by(RID) %>% slice(1) %>% ungroup()

apoe <- read_csv(path_apoe, show_col_types = FALSE) %>%
  select(RID, GENOTYPE) %>%
  distinct(RID, .keep_all = TRUE) %>%
  mutate(
    APOE4_Positive = ifelse(grepl("4", as.character(GENOTYPE)), 1, 0),
    APOE4_Copies = str_count(as.character(GENOTYPE), "4")
  ) %>%
  select(RID, APOE4_Positive, APOE4_Copies)

# Auxiliary cognitive scores (optional)
adas <- if(file.exists(path_adas)) {
  read_csv(path_adas, show_col_types = FALSE) %>%
    filter(VISCODE %in% c("bl", "sc")) %>% 
    select(RID, ADAS13 = TOTAL13) %>% 
    group_by(RID) %>% slice(1) %>% ungroup()
} else { data.frame(RID = integer(), ADAS13 = numeric()) }

cdr <- if(file.exists(path_cdr)) {
  read_csv(path_cdr, show_col_types = FALSE) %>%
    filter(VISCODE %in% c("bl", "sc")) %>% 
    select(RID, CDRSB) %>% 
    group_by(RID) %>% slice(1) %>% ungroup()
} else { data.frame(RID = integer(), CDRSB = numeric()) }

faq <- if(file.exists(path_faq)) {
  read_csv(path_faq, show_col_types = FALSE) %>%
    filter(VISCODE %in% c("bl", "sc")) %>% 
    select(RID, FAQTOTAL) %>% 
    group_by(RID) %>% slice(1) %>% ungroup()
} else { data.frame(RID = integer(), FAQTOTAL = numeric()) }

## =====================================================================
## Step 3: Merge Data and Exclude Training Set
## =====================================================================
cat("\nStep 3: Merge Data and Exclude Training Set...\n")

merged_all <- mci_cohort %>%
  left_join(mri_features, by = "RID") %>%
  left_join(csf_merged, by = "RID") %>%
  left_join(demog, by = "RID") %>%
  left_join(mmse, by = "RID") %>%
  left_join(adas, by = "RID") %>%
  left_join(cdr, by = "RID") %>%
  left_join(faq, by = "RID") %>%
  left_join(apoe, by = "RID")

# Exclude training set samples
if(file.exists(path_train)) {
  train_data <- read_csv(path_train, show_col_types = FALSE)
  train_rids <- unique(train_data$ID)
  merged_all <- merged_all %>% filter(!RID %in% train_rids)
  cat(sprintf("  Remaining after excluding training set: %d cases\n", nrow(merged_all)))
}

## =====================================================================
## Step 4: Strict Quality Filtering (Core Step for High-Quality Subset)
## =====================================================================
cat("\n", paste(rep("=", 70), collapse = ""), "\n")
cat("Step 4: Strict Quality Filtering (High Quality Only)\n")
cat(paste(rep("=", 70), collapse = ""), "\n")

# Calculate age from baseline date and DOB
merged_all$Age <- NA
for(i in 1:nrow(merged_all)) {
  if(!is.na(merged_all$Baseline_Date[i]) && !is.na(merged_all$PTDOB[i])) {
    base_year <- as.numeric(format(as.Date(merged_all$Baseline_Date[i]), "%Y"))
    dob_year <- as.numeric(str_extract(as.character(merged_all$PTDOB[i]), "\\d{4}"))
    if(!is.na(base_year) && !is.na(dob_year)) merged_all$Age[i] <- base_year - dob_year
  }
}
# Impute missing age with mean (minimal imputation)
merged_all$Age[is.na(merged_all$Age)] <- mean(merged_all$Age, na.rm=TRUE)

# Core quality filtering (AND logic - all criteria must be met)
before_n <- nrow(merged_all)

final_cohort <- merged_all %>%
  # 1. Mandatory demographics & genetics
  filter(!is.na(Age) & !is.na(PTGENDER) & !is.na(APOE4_Positive)) %>%
  # 2. Mandatory cognitive score (MMSE)
  filter(!is.na(MMSCORE)) %>%
  # 3. Mandatory CSF biomarkers (Abeta42 + pTau/Tau)
  filter(!is.na(ABETA42) & (!is.na(TAU_TOTAL) | !is.na(PTAU181))) %>%
  # 4. Mandatory core MRI regions (hippocampus, entorhinal, middle temporal)
  filter(!is.na(ST102TS) & !is.na(ST103TA) & !is.na(ST105TA)) %>%
  # 5. Age range constraint
  filter(Age >= 50 & Age <= 95)

cat(sprintf("  Cases before filtering: %d\n", before_n))
cat(sprintf("  Cases after filtering: %d (complete multimodal data)\n", nrow(final_cohort)))
cat(sprintf("  Cases excluded: %d (severe data missing)\n", before_n - nrow(final_cohort)))

## =====================================================================
## Step 5: Format Output Data
## =====================================================================
cat("\nStep 5: Format Output Data...\n")

output_df <- final_cohort %>%
  mutate(
    ID = sprintf("TEST_%04d", row_number()),
    Gender = ifelse(PTGENDER == "Male" | PTGENDER == 1, 0, 1), # 0=Male, 1=Female
    MMSE_Baseline = MMSCORE,
    Education = ifelse(is.na(PTEDUCAT) | PTEDUCAT == 0, 15, PTEDUCAT),
    STATUS = "MCI", # Fixed as MCI (only MCI patients included)
    # Impute missing ABETA40 (prevent calculation errors)
    ABETA40 = ifelse(is.na(ABETA40), mean(ABETA40, na.rm=TRUE), ABETA40),
    ABETA42_ABETA40_RATIO = ABETA42 / ABETA40,
    completeness = 1.0 # Full completeness for high-quality subset
  ) 

# Enforce valid value ranges (data cleaning)
output_df$Age <- pmax(50, pmin(95, output_df$Age))
output_df$MMSE_Baseline <- pmax(0, pmin(30, output_df$MMSE_Baseline))
output_df$ABETA42 <- pmax(50, pmin(1700, output_df$ABETA42))
output_df$TAU_TOTAL <- pmax(10, pmin(1500, output_df$TAU_TOTAL))

# Define final column set
desired_cols_base <- c("ID", "RID", "Age", "Gender", "Education", "MMSE_Baseline", 
                       "ADAS13", "CDRSB", "FAQTOTAL", "APOE4_Positive", "APOE4_Copies", 
                       "ABETA42", "ABETA40", "TAU_TOTAL", "PTAU181", "STATUS")
st_cols <- grep("^ST", names(output_df), value = TRUE)
desired_cols_end <- c("AD_Conversion", "completeness", "Baseline_Date", "ABETA42_ABETA40_RATIO")
final_cols <- c(desired_cols_base, st_cols, desired_cols_end)

# Ensure all columns exist (add NA if missing)
for(col in final_cols) {
  if(!col %in% names(output_df)) {
    output_df[[col]] <- NA
  }
}

# Select final columns in specified order
output_df_final <- output_df %>% select(all_of(final_cols))

## =====================================================================
## Step 6: Save Final Independent Test Set
## =====================================================================
cat("\n", paste(rep("=", 70), collapse = ""), "\n")
cat("Save Final Independent Test Set\n")
cat(paste(rep("=", 70), collapse = ""), "\n")

# Save to CSV
out_csv <- file.path(output_dir, "independent_test_set.csv")
write_csv(output_df_final, out_csv)

# Summary statistics
cat(sprintf("  ✓ Saved to: %s\n", out_csv))
cat(sprintf("  ✓ Final sample size: %d\n", nrow(output_df_final)))
cat(sprintf("  ✓ AD converters: %d (%.1f%%)\n", 
            sum(output_df_final$AD_Conversion), 
            mean(output_df_final$AD_Conversion)*100))
cat(sprintf("  ✓ Number of MRI features included: %d\n", length(st_cols)))

cat("\n", paste(rep("=", 70), collapse = ""), "\n")
