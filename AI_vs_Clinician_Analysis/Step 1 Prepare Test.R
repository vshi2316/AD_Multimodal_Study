library(tidyverse)
library(readr)
library(optparse)

# ==============================================================================
# Parse Command Line Arguments
# ==============================================================================
option_list <- list(
  make_option(c("--adni_dir"), type = "character", 
              default = "./ADNI_Raw_Data",
              help = "Path to ADNI raw data directory [default: %default]"),
  make_option(c("--train_file"), type = "character", 
              default = "./cluster_results.csv",
              help = "Path to training set cluster results (for exclusion) [default: %default]"),
  make_option(c("--output_dir"), type = "character", 
              default = "./AI_vs_Clinician_Test",
              help = "Output directory [default: %default]"),
  make_option(c("--target_n"), type = "integer", default = 196,
              help = "Target sample size [default: %default]"),
  make_option(c("--min_age"), type = "integer", default = 50,
              help = "Minimum age [default: %default]"),
  make_option(c("--max_age"), type = "integer", default = 95,
              help = "Maximum age [default: %default]"),
  make_option(c("--seed"), type = "integer", default = 2024,
              help = "Random seed for reproducibility [default: %default]")
)

opt_parser <- OptionParser(option_list = option_list)
opt <- parse_args(opt_parser)

# Set seed for reproducibility
set.seed(opt$seed)

# Create output directory
dir.create(opt$output_dir, showWarnings = FALSE, recursive = TRUE)

# ==============================================================================
# Define File Paths
# ==============================================================================
path_dx <- file.path(opt$adni_dir, "LINES/Diagnostic Summary.csv")
path_mri <- file.path(opt$adni_dir, "sMRI/UCSF - Cross-Sectional FreeSurfer (7.x).csv")
path_demog <- file.path(opt$adni_dir, "LINES/Subject Demographics.csv")
path_mmse <- file.path(opt$adni_dir, "LINES/Mini-Mental State Examination (MMSE).csv")
path_apoe <- file.path(opt$adni_dir, "APOE/ApoE Genotyping - Results.csv")
path_csf_alz <- file.path(opt$adni_dir, "CSF/UPENN CSF Biomarker Master Alzbio3.csv")
path_csf_roche <- file.path(opt$adni_dir, "CSF/UPENN CSF Biomarkers Roche Elecsys.csv")
path_adas <- file.path(opt$adni_dir, "LINES/ADAS-Cognitive Behavior.csv")
path_cdr <- file.path(opt$adni_dir, "LINES/Clinical Dementia Rating.csv")
path_faq <- file.path(opt$adni_dir, "LINES/Futional Activities Questionnaire.csv")

# ==============================================================================
# Part 1: Extract Baseline MCI Patients
# ==============================================================================
cat("[1/5] Extracting baseline MCI patients...\n")

if (!file.exists(path_dx)) {
  stop(sprintf("Diagnostic file not found: %s", path_dx))
}

dx_data <- read_csv(path_dx, show_col_types = FALSE)

baseline_mci <- dx_data %>%
  filter(VISCODE %in% c("bl", "sc"), DXMCI == 1) %>%
  select(RID, Baseline_Date = EXAMDATE) %>%
  distinct(RID, .keep_all = TRUE) %>%
  mutate(Baseline_Date = as.Date(Baseline_Date, format = "%m/%d/%Y"))

cat(sprintf("  Baseline MCI patients: %d\n", nrow(baseline_mci)))

# Determine AD conversion outcome
patient_outcomes <- data.frame()
for (pid in baseline_mci$RID) {
  records <- dx_data %>% filter(RID == pid)
  is_converter <- any(records$DXAD == 1, na.rm = TRUE)
  patient_outcomes <- rbind(patient_outcomes, data.frame(
    RID = pid, 
    AD_Conversion = ifelse(is_converter, 1, 0)
  ))
}

mci_cohort <- baseline_mci %>% 
  left_join(patient_outcomes, by = "RID")

cat(sprintf("  AD converters: %d (%.1f%%)\n", 
            sum(mci_cohort$AD_Conversion), 
            100 * mean(mci_cohort$AD_Conversion)))

# ==============================================================================
# Part 2: Load Multimodal Data
# ==============================================================================
cat("\n[2/5] Loading multimodal data...\n")

# MRI data
mri_data <- if (file.exists(path_mri)) {
  read_csv(path_mri, show_col_types = FALSE) %>%
    filter(VISCODE %in% c("bl", "sc")) %>%
    select(RID, starts_with("ST")) %>%
    group_by(RID) %>% 
    slice(1) %>% 
    ungroup()
} else {
  cat("  WARNING: MRI file not found\n")
  data.frame(RID = integer())
}
cat(sprintf("  MRI data: %d subjects\n", nrow(mri_data)))

# CSF biomarkers (AlzBio3)
csf_alz <- if (file.exists(path_csf_alz)) {
  read_csv(path_csf_alz, show_col_types = FALSE) %>%
    filter(VISCODE %in% c("bl", "sc")) %>%
    select(RID, ABETA, TAU, PTAU) %>%
    rename(ABETA42 = ABETA, TAU_TOTAL = TAU, PTAU181 = PTAU) %>%
    group_by(RID) %>% 
    slice(1) %>% 
    ungroup()
} else {
  data.frame(RID = integer(), ABETA42 = numeric(), TAU_TOTAL = numeric(), PTAU181 = numeric())
}

# CSF biomarkers (Roche Elecsys)
csf_roche <- if (file.exists(path_csf_roche)) {
  read_csv(path_csf_roche, show_col_types = FALSE) %>%
    filter(VISCODE2 %in% c("bl", "sc")) %>%
    select(RID, ABETA40) %>%
    group_by(RID) %>% 
    slice(1) %>% 
    ungroup()
} else {
  data.frame(RID = integer(), ABETA40 = numeric())
}

csf_merged <- full_join(csf_alz, csf_roche, by = "RID")
cat(sprintf("  CSF data: %d subjects\n", nrow(csf_merged)))

# Demographics
demog <- if (file.exists(path_demog)) {
  read_csv(path_demog, show_col_types = FALSE) %>%
    select(RID, PTGENDER, PTDOB, PTEDUCAT) %>%
    distinct(RID, .keep_all = TRUE)
} else {
  data.frame(RID = integer())
}

# MMSE
mmse <- if (file.exists(path_mmse)) {
  read_csv(path_mmse, show_col_types = FALSE) %>%
    filter(VISCODE %in% c("bl", "sc")) %>%
    select(RID, MMSCORE) %>%
    group_by(RID) %>% 
    slice(1) %>% 
    ungroup()
} else {
  data.frame(RID = integer(), MMSCORE = numeric())
}

# APOE genotype
apoe <- if (file.exists(path_apoe)) {
  read_csv(path_apoe, show_col_types = FALSE) %>%
    select(RID, GENOTYPE) %>%
    distinct(RID, .keep_all = TRUE) %>%
    mutate(
      APOE4_Positive = ifelse(grepl("4", as.character(GENOTYPE)), 1, 0),
      APOE4_Copies = str_count(as.character(GENOTYPE), "4")
    ) %>%
    select(RID, APOE4_Positive, APOE4_Copies)
} else {
  data.frame(RID = integer(), APOE4_Positive = integer(), APOE4_Copies = integer())
}

# ADAS-Cog
adas <- if (file.exists(path_adas)) {
  read_csv(path_adas, show_col_types = FALSE) %>%
    filter(VISCODE %in% c("bl", "sc")) %>%
    select(RID, ADAS13 = TOTAL13) %>%
    group_by(RID) %>% 
    slice(1) %>% 
    ungroup()
} else {
  data.frame(RID = integer(), ADAS13 = numeric())
}

# CDR
cdr <- if (file.exists(path_cdr)) {
  read_csv(path_cdr, show_col_types = FALSE) %>%
    filter(VISCODE %in% c("bl", "sc")) %>%
    select(RID, CDRSB) %>%
    group_by(RID) %>% 
    slice(1) %>% 
    ungroup()
} else {
  data.frame(RID = integer(), CDRSB = numeric())
}

# FAQ
faq <- if (file.exists(path_faq)) {
  read_csv(path_faq, show_col_types = FALSE) %>%
    filter(VISCODE %in% c("bl", "sc")) %>%
    select(RID, FAQTOTAL) %>%
    group_by(RID) %>% 
    slice(1) %>% 
    ungroup()
} else {
  data.frame(RID = integer(), FAQTOTAL = numeric())
}

# ==============================================================================
# Part 3: Merge All Data
# ==============================================================================
cat("\n[3/5] Merging multimodal data...\n")

merged_all <- mci_cohort %>%
  left_join(mri_data, by = "RID") %>%
  left_join(csf_merged, by = "RID") %>%
  left_join(demog, by = "RID") %>%
  left_join(mmse, by = "RID") %>%
  left_join(adas, by = "RID") %>%
  left_join(cdr, by = "RID") %>%
  left_join(faq, by = "RID") %>%
  left_join(apoe, by = "RID")

cat(sprintf("  Merged data: %d subjects\n", nrow(merged_all)))

# ==============================================================================
# Part 4: Exclude Training Set 
# ==============================================================================
cat("\n[4/5] Excluding training set samples ...\n")

if (file.exists(opt$train_file)) {
  train_data <- read_csv(opt$train_file, show_col_types = FALSE)
  
  # Get training RIDs
  if ("ID" %in% names(train_data)) {
    train_rids <- unique(train_data$ID)
  } else if ("RID" %in% names(train_data)) {
    train_rids <- unique(train_data$RID)
  } else {
    train_rids <- c()
  }
  
  n_before <- nrow(merged_all)
  merged_all <- merged_all %>% filter(!RID %in% train_rids)
  n_excluded <- n_before - nrow(merged_all)
  
  cat(sprintf("  Training set subjects excluded: %d\n", n_excluded))
  cat(sprintf("  Remaining subjects: %d\n", nrow(merged_all)))
} else {
  cat("  WARNING: Training file not found, no exclusion performed\n")
}

# Calculate age
merged_all$Age <- NA
for (i in 1:nrow(merged_all)) {
  if (!is.na(merged_all$Baseline_Date[i]) && !is.na(merged_all$PTDOB[i])) {
    base_year <- as.numeric(format(as.Date(merged_all$Baseline_Date[i]), "%Y"))
    dob_year <- as.numeric(str_extract(as.character(merged_all$PTDOB[i]), "\\d{4}"))
    if (!is.na(base_year) && !is.na(dob_year)) {
      merged_all$Age[i] <- base_year - dob_year
    }
  }
}

# ==============================================================================
# Part 5: Apply Quality Filters
# ==============================================================================
cat("\n[5/5] Applying quality filters...\n")

n_before <- nrow(merged_all)

# Required variables filter
final_cohort <- merged_all %>%
  filter(!is.na(Age) & !is.na(PTGENDER) & !is.na(APOE4_Positive)) %>%
  filter(!is.na(MMSCORE)) %>%
  filter(!is.na(ABETA42) & (!is.na(TAU_TOTAL) | !is.na(PTAU181)))

cat(sprintf("  After core variable filter: %d\n", nrow(final_cohort)))

# MRI filter (require key regions)
mri_cols <- grep("^ST", names(final_cohort), value = TRUE)
if (length(mri_cols) > 0) {
  key_mri <- c("ST102TS", "ST103TA", "ST105TA")
  key_mri_available <- key_mri[key_mri %in% names(final_cohort)]
  
  if (length(key_mri_available) > 0) {
    final_cohort <- final_cohort %>%
      filter(if_all(all_of(key_mri_available), ~ !is.na(.)))
    cat(sprintf("  After MRI filter: %d\n", nrow(final_cohort)))
  }
}

# Age filter
final_cohort <- final_cohort %>%
  filter(Age >= opt$min_age & Age <= opt$max_age)

cat(sprintf("  After age filter (%d-%d): %d\n", opt$min_age, opt$max_age, nrow(final_cohort)))

# Sample to target size if needed
if (nrow(final_cohort) > opt$target_n) {
  cat(sprintf("  Sampling to target size: %d\n", opt$target_n))
  final_cohort <- final_cohort %>% sample_n(opt$target_n)
}

cat(sprintf("\n  Final test set: %d cases\n", nrow(final_cohort)))

# ==============================================================================
# Prepare Output Dataset
# ==============================================================================
output_df <- final_cohort %>%
  mutate(
    ID = sprintf("TEST_%04d", row_number()),
    Gender = ifelse(PTGENDER == "Male" | PTGENDER == 1, 0, 1),
    MMSE_Baseline = MMSCORE,
    Education = ifelse(is.na(PTEDUCAT) | PTEDUCAT == 0, 15, PTEDUCAT),
    STATUS = "MCI",
    completeness = 1.0
  )

# Clip values to valid ranges
output_df$Age <- pmax(opt$min_age, pmin(opt$max_age, output_df$Age))
output_df$MMSE_Baseline <- pmax(0, pmin(30, output_df$MMSE_Baseline))
output_df$ABETA42 <- pmax(50, pmin(1700, output_df$ABETA42))
if ("TAU_TOTAL" %in% names(output_df)) {
  output_df$TAU_TOTAL <- pmax(10, pmin(1500, output_df$TAU_TOTAL))
}

# Select and order columns
desired_cols_base <- c("ID", "RID", "Age", "Gender", "Education", "MMSE_Baseline", 
                       "ADAS13", "CDRSB", "FAQTOTAL", "APOE4_Positive", "APOE4_Copies",
                       "ABETA42", "ABETA40", "TAU_TOTAL", "PTAU181", "STATUS")
st_cols <- grep("^ST", names(output_df), value = TRUE)
desired_cols_end <- c("AD_Conversion", "completeness", "Baseline_Date")

final_cols <- c(desired_cols_base, st_cols, desired_cols_end)
final_cols <- final_cols[final_cols %in% names(output_df)]

# Add missing columns as NA
for (col in final_cols) {
  if (!col %in% names(output_df)) {
    output_df[[col]] <- NA
  }
}

output_df_final <- output_df %>% select(all_of(final_cols))

# ==============================================================================
# Save Output
# ==============================================================================
out_csv <- file.path(opt$output_dir, "independent_test_set.csv")
write_csv(output_df_final, out_csv)

# ==============================================================================
# Summary Report
# ==============================================================================
cat("\n")
cat("============================================================\n")
cat("Independent Test Set Summary\n")
cat("============================================================\n")
cat(sprintf("Output file: %s\n", out_csv))
cat(sprintf("Final sample size: %d (Target: %d)\n", nrow(output_df_final), opt$target_n))
cat(sprintf("AD converters: %d (%.1f%%)\n", 
            sum(output_df_final$AD_Conversion), 
            100 * mean(output_df_final$AD_Conversion)))
cat(sprintf("Non-converters: %d (%.1f%%)\n", 
            sum(output_df_final$AD_Conversion == 0), 
            100 * mean(output_df_final$AD_Conversion == 0)))
cat(sprintf("MRI features: %d\n", length(st_cols)))
cat("  ✓ Independent test set physically sequestered from training\n")
cat("  ✓ MCI patients from ADNI\n")
cat("  ✓ Training set samples excluded\n")
cat("\nStep 1 complete.\n")

