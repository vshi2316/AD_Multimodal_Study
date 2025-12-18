#!/usr/bin/env Rscript
# Step 1: Independent Test Set Preparation
# Objective: Filter high-quality multimodal MCI cases for independent testing
# CRITICAL: No imputation using test set statistics - preserve NA for frozen pipeline

library(tidyverse)
library(readr)

set.seed(2024)

cat("Step 1: Independent Test Set Preparation\n")
cat(strrep("=", 70), "\n\n")

root_dir <- "./ADNI_Raw_Data"
output_dir <- "AI_vs_Clinician_Test"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

path_dx <- file.path(root_dir, "LINES/Diagnostic Summary.csv")
path_mri <- file.path(root_dir, "sMRI/UCSF - Cross-Sectional FreeSurfer (7.x).csv")
path_demog <- file.path(root_dir, "LINES/Subject Demographics.csv")
path_mmse <- file.path(root_dir, "LINES/Mini-Mental State Examination (MMSE).csv")
path_apoe <- file.path(root_dir, "APOE/ApoE Genotyping - Results.csv")
path_csf_alz <- file.path(root_dir, "CSF/UPENN CSF Biomarker Master Alzbio3.csv")
path_csf_roche <- file.path(root_dir, "CSF/UPENN CSF Biomarkers Roche Elecsys.csv")
path_adas <- file.path(root_dir, "LINES/ADAS-Cognitive Behavior.csv")
path_cdr <- file.path(root_dir, "LINES/Clinical Dementia Rating.csv")
path_faq <- file.path(root_dir, "LINES/Futional Activities Questionnaire.csv")
path_train <- "./cluster_results.csv"

cat("Extracting baseline MCI patients...\n")
dx_data <- read_csv(path_dx, show_col_types = FALSE)

baseline_mci <- dx_data %>%
  filter(VISCODE %in% c("bl", "sc"), DXMCI == 1) %>%
  select(RID, Baseline_Date = EXAMDATE) %>%
  distit(RID, .keep_all = TRUE) %>%
  mutate(Baseline_Date = as.Date(Baseline_Date, format = "%m/%d/%Y"))

cat(sprintf("  Baseline MCI patients: %d\n", nrow(baseline_mci)))

patient_outcomes <- data.frame()
for(pid in baseline_mci$RID) {
  records <- dx_data %>% filter(RID == pid)
  is_converter <- any(records$DXAD == 1, na.rm = TRUE)
  patient_outcomes <- rbind(patient_outcomes, 
                            data.frame(RID = pid, AD_Conversion = ifelse(is_converter, 1, 0)))
}

mci_cohort <- baseline_mci %>% left_join(patient_outcomes, by = "RID")

cat("Loading multimodal data...\n")

mri_data <- read_csv(path_mri, show_col_types = FALSE) %>%
  filter(VISCODE %in% c("bl", "sc")) %>%
  select(RID, starts_with("ST")) %>%
  group_by(RID) %>% slice(1) %>% ungroup()

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

demog <- read_csv(path_demog, show_col_types = FALSE) %>%
  select(RID, PTGENDER, PTDOB, PTEDUCAT) %>%
  distit(RID, .keep_all = TRUE)

mmse <- read_csv(path_mmse, show_col_types = FALSE) %>%
  filter(VISCODE %in% c("bl", "sc")) %>%
  select(RID, MMSCORE) %>%
  group_by(RID) %>% slice(1) %>% ungroup()

apoe <- read_csv(path_apoe, show_col_types = FALSE) %>%
  select(RID, GENOTYPE) %>%
  distit(RID, .keep_all = TRUE) %>%
  mutate(APOE4_Positive = ifelse(grepl("4", as.character(GENOTYPE)), 1, 0),
         APOE4_Copies = str_count(as.character(GENOTYPE), "4")) %>%
  select(RID, APOE4_Positive, APOE4_Copies)

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

merged_all <- mci_cohort %>%
  left_join(mri_data, by = "RID") %>%
  left_join(csf_merged, by = "RID") %>%
  left_join(demog, by = "RID") %>%
  left_join(mmse, by = "RID") %>%
  left_join(adas, by = "RID") %>%
  left_join(cdr, by = "RID") %>%
  left_join(faq, by = "RID") %>%
  left_join(apoe, by = "RID")

if(file.exists(path_train)) {
  train_data <- read_csv(path_train, show_col_types = FALSE)
  train_rids <- unique(train_data$ID)
  merged_all <- merged_all %>% filter(!RID %in% train_rids)
  cat(sprintf("  After excluding training set: %d cases\n", nrow(merged_all)))
}

merged_all$Age <- NA
for(i in 1:nrow(merged_all)) {
  if(!is.na(merged_all$Baseline_Date[i]) && !is.na(merged_all$PTDOB[i])) {
    base_year <- as.numeric(format(as.Date(merged_all$Baseline_Date[i]), "%Y"))
    dob_year <- as.numeric(str_extract(as.character(merged_all$PTDOB[i]), "\\d{4}"))
    if(!is.na(base_year) && !is.na(dob_year)) merged_all$Age[i] <- base_year - dob_year
  }
}

cat("Applying quality filters...\n")
before_n <- nrow(merged_all)

final_cohort <- merged_all %>%
  filter(!is.na(Age) & !is.na(PTGENDER) & !is.na(APOE4_Positive)) %>%
  filter(!is.na(MMSCORE)) %>%
  filter(!is.na(ABETA42) & (!is.na(TAU_TOTAL) | !is.na(PTAU181))) %>%
  filter(!is.na(ST102TS) & !is.na(ST103TA) & !is.na(ST105TA)) %>%
  filter(Age >= 50 & Age <= 95)

cat(sprintf("  Cases before filtering: %d\n", before_n))
cat(sprintf("  Cases after filtering: %d\n", nrow(final_cohort)))

output_df <- final_cohort %>%
  mutate(
    ID = sprintf("TEST_%04d", row_number()),
    Gender = ifelse(PTGENDER == "Male" | PTGENDER == 1, 0, 1),
    MMSE_Baseline = MMSCORE,
    Education = ifelse(is.na(PTEDUCAT) | PTEDUCAT == 0, 15, PTEDUCAT),
    STATUS = "MCI",
    completeness = 1.0
  )

output_df$Age <- pmax(50, pmin(95, output_df$Age))
output_df$MMSE_Baseline <- pmax(0, pmin(30, output_df$MMSE_Baseline))
output_df$ABETA42 <- pmax(50, pmin(1700, output_df$ABETA42))
output_df$TAU_TOTAL <- pmax(10, pmin(1500, output_df$TAU_TOTAL))

desired_cols_base <- c("ID", "RID", "Age", "Gender", "Education", "MMSE_Baseline", 
                       "ADAS13", "CDRSB", "FAQTOTAL", "APOE4_Positive", "APOE4_Copies",
                       "ABETA42", "ABETA40", "TAU_TOTAL", "PTAU181", "STATUS")
st_cols <- grep("^ST", names(output_df), value = TRUE)
desired_cols_end <- c("AD_Conversion", "completeness", "Baseline_Date")
final_cols <- c(desired_cols_base, st_cols, desired_cols_end)

for(col in final_cols) {
  if(!col %in% names(output_df)) output_df[[col]] <- NA
}

output_df_final <- output_df %>% select(all_of(final_cols))

out_csv <- file.path(output_dir, "independent_test_set.csv")
write_csv(output_df_final, out_csv)

cat(sprintf("\nTest set saved: %s\n", out_csv))
cat(sprintf("  Final sample size: %d\n", nrow(output_df_final)))
cat(sprintf("  AD converters: %d (%.1f%%)\n", 
            sum(output_df_final$AD_Conversion), 
            mean(output_df_final$AD_Conversion)*100))
cat(sprintf("  MRI features: %d\n", length(st_cols)))
cat("\nStep 1 complete.\n")
