# AIBL feasibility gate for external validation upgrade

library(optparse)

suppressPackageStartupMessages({
  needed <- c("readr", "dplyr", "tidyr", "stringr", "lubridate", "purrr")
  missing_pkgs <- needed[!vapply(needed, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing_pkgs) > 0) {
    stop("Missing required R packages: ", paste(missing_pkgs, collapse = ", "), call. = FALSE)
  }
  library(readr)
  library(dplyr)
  library(tidyr)
  library(stringr)
  library(lubridate)
  library(purrr)
})

set.seed(20260614)

# =========================
# 0. Paths
# =========================
option_list <- list(
  make_option(c("--aibl_dir"), type = "character", default = "./aibl_19Sep2019/Data_extract_3.3.0",
              help = "AIBL raw Data_extract_3.3.0 directory [default: %default]"),
  make_option(c("--adni_holdout_file"), type = "character", default = "./AI_vs_Clinician_Test/independent_test_set.csv",
              help = "ADNI holdout test set CSV [default: %default]"),
  make_option(c("--adni_discovery_file"), type = "character", default = "./Phase1_ADNI_Discovery/ADNI_Labeled_For_Classifier.csv",
              help = "ADNI discovery labeled classifier CSV [default: %default]"),
  make_option(c("--model_config"), type = "character", default = "./step11_results/model_config.rds",
              help = "Optional full-model configuration RDS [default: %default]"),
  make_option(c("--feature_importance"), type = "character", default = "./step11_results/Feature_Importance_RF.csv",
              help = "Optional full-model feature-importance CSV [default: %default]"),
  make_option(c("--output_dir"), type = "character", default = "./4_external_contextualization/AIBL_Feasibility_Gate",
              help = "Output directory [default: %default]"),
  make_option(c("--seed"), type = "integer", default = 20260614,
              help = "Random seed [default: %default]")
)
opt <- parse_args(OptionParser(option_list = option_list))
set.seed(opt$seed)

aibl_dir <- opt$aibl_dir
adni_holdout_path <- opt$adni_holdout_file
adni_discovery_path <- opt$adni_discovery_file
model_config_path <- opt$model_config
feature_importance_path <- opt$feature_importance
out_dir <- opt$output_dir

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

file_required <- function(path) {
  if (!file.exists(path)) stop("Missing file: ", path, call. = FALSE)
  path
}

read_csv_chr <- function(path) {
  file_required(path)
  readr::read_csv(
    path,
    col_types = readr::cols(.default = readr::col_character()),
    show_col_types = FALSE,
    progress = FALSE
  )
}

write_csv_safe <- function(x, path) {
  readr::write_csv(x, path, na = "")
  invisible(path)
}

missing_codes <- c("", "NA", "NaN", "-4", "-1", "-9", "-99", "-999", "-9999", "-777777")

clean_missing <- function(df) {
  df %>%
    mutate(across(everything(), ~ {
      x <- trimws(as.character(.x))
      x[x %in% missing_codes] <- NA_character_
      x
    }))
}

num <- function(x) suppressWarnings(as.numeric(x))

parse_date_safe <- function(x) {
  x <- trimws(as.character(x))
  x[x %in% missing_codes] <- NA_character_
  out <- suppressWarnings(lubridate::mdy(x))
  if (all(is.na(out))) out <- suppressWarnings(lubridate::ymd(x))
  out
}

visit_month_from_viscode <- function(x) {
  x <- tolower(trimws(as.character(x)))
  out <- rep(NA_real_, length(x))
  out[x %in% c("bl", "sc", "scmri", "init", "m0")] <- 0
  m <- stringr::str_match(x, "^m(\\d+)$")[, 2]
  out[!is.na(m)] <- as.numeric(m[!is.na(m)])
  out
}

extract_birth_year <- function(x) {
  y <- stringr::str_extract(as.character(x), "(19|20)\\d{2}")
  suppressWarnings(as.numeric(y))
}

first_baseline <- function(df) {
  df %>%
    mutate(visit_month = visit_month_from_viscode(VISCODE)) %>%
    filter(!is.na(RID), visit_month == 0) %>%
    arrange(RID, VISCODE) %>%
    group_by(RID) %>%
    slice(1) %>%
    ungroup()
}

# =========================
# 1. Load raw AIBL files
# =========================
pdx <- read_csv_chr(file.path(aibl_dir, "aibl_pdxconv_01-Jun-2018.csv")) %>% clean_missing()
mmse <- read_csv_chr(file.path(aibl_dir, "aibl_mmse_01-Jun-2018.csv")) %>% clean_missing()
cdr <- read_csv_chr(file.path(aibl_dir, "aibl_cdr_01-Jun-2018.csv")) %>% clean_missing()
neuro <- read_csv_chr(file.path(aibl_dir, "aibl_neurobat_01-Jun-2018.csv")) %>% clean_missing()
apoe <- read_csv_chr(file.path(aibl_dir, "aibl_apoeres_01-Jun-2018.csv")) %>% clean_missing()
demog <- read_csv_chr(file.path(aibl_dir, "aibl_ptdemog_01-Jun-2018.csv")) %>% clean_missing()
lab <- read_csv_chr(file.path(aibl_dir, "aibl_labdata_01-Jun-2018.csv")) %>% clean_missing()
mri3 <- read_csv_chr(file.path(aibl_dir, "aibl_mri3meta_01-Jun-2018.csv")) %>% clean_missing()
pib <- read_csv_chr(file.path(aibl_dir, "aibl_pibmeta_01-Jun-2018.csv")) %>% clean_missing()
av45 <- read_csv_chr(file.path(aibl_dir, "aibl_av45meta_01-Jun-2018.csv")) %>% clean_missing()

# =========================
# 2. True baseline MCI -> AD conversion endpoint
# =========================
pdx2 <- pdx %>%
  mutate(
    RID = as.character(RID),
    visit_month = visit_month_from_viscode(VISCODE),
    DXCURREN = num(DXCURREN),
    DXMCI = num(DXMCI),
    DXAD = num(DXAD),
    baseline_is_mci = (DXMCI %in% 1) | (DXCURREN %in% 2),
    visit_is_ad = (DXAD %in% 1) | (DXCURREN %in% 3)
  )

baseline_dx <- pdx2 %>%
  filter(visit_month == 0) %>%
  arrange(RID, VISCODE) %>%
  group_by(RID) %>%
  slice(1) %>%
  ungroup() %>%
  transmute(
    RID,
    baseline_visit = VISCODE,
    baseline_visit_month = visit_month,
    baseline_DXCURREN = DXCURREN,
    baseline_DXMCI = DXMCI,
    baseline_DXAD = DXAD,
    baseline_MCI = baseline_is_mci,
    baseline_AD = visit_is_ad
  )

mci_baseline <- baseline_dx %>%
  filter(baseline_MCI, !baseline_AD)

followup_dx <- pdx2 %>%
  inner_join(mci_baseline %>% select(RID, baseline_visit_month), by = "RID") %>%
  filter(!is.na(visit_month), visit_month > baseline_visit_month) %>%
  group_by(RID) %>%
  summarise(
    n_followup_dx_visits = n(),
    last_followup_month = max(visit_month, na.rm = TRUE),
    event_month = if (any(visit_is_ad, na.rm = TRUE)) {
      min(visit_month[visit_is_ad], na.rm = TRUE)
    } else {
      NA_real_
    },
    AD_Conversion = as.integer(!is.na(event_month)),
    Time_to_Event_Months = ifelse(AD_Conversion == 1, event_month, last_followup_month),
    Followup_Years = Time_to_Event_Months / 12,
    .groups = "drop"
  )

# =========================
# 3. Baseline feature assembly
# =========================
base_mmse <- first_baseline(mmse) %>%
  transmute(
    RID,
    Baseline_ExamDate_MMSE = parse_date_safe(EXAMDATE),
    MMSE_Baseline = num(MMSCORE)
  )

base_cdr <- first_baseline(cdr) %>%
  transmute(
    RID,
    Baseline_ExamDate_CDR = parse_date_safe(EXAMDATE),
    CDR_Baseline = num(CDGLOBAL)
  )

base_neuro <- first_baseline(neuro) %>%
  transmute(
    RID,
    Baseline_ExamDate_Neuro = parse_date_safe(EXAMDATE),
    LogicalMemory_Imm = num(LIMMTOTAL),
    LogicalMemory_Del = num(LDELTOTAL)
  )

base_demog <- first_baseline(demog) %>%
  transmute(
    RID,
    Gender = num(PTGENDER),
    Birth_Year = extract_birth_year(PTDOB)
  )

base_apoe <- apoe %>%
  mutate(
    RID = as.character(RID),
    visit_month = visit_month_from_viscode(VISCODE),
    APGEN1 = as.character(APGEN1),
    APGEN2 = as.character(APGEN2)
  ) %>%
  arrange(RID, visit_month) %>%
  group_by(RID) %>%
  slice(1) %>%
  ungroup() %>%
  rowwise() %>%
  mutate(
    APOE4_Count = sum(c(APGEN1, APGEN2) == "4", na.rm = TRUE),
    APOE4_Positive = as.integer(APOE4_Count > 0)
  ) %>%
  ungroup() %>%
  transmute(RID, APOE4_Count, APOE4_Positive)

lab_cols <- setdiff(names(lab), c("RID", "SITEID", "VISCODE"))
base_lab <- first_baseline(lab) %>%
  mutate(across(all_of(lab_cols), num)) %>%
  select(RID, all_of(lab_cols))

base_mri3 <- first_baseline(mri3) %>%
  transmute(
    RID,
    Baseline_ExamDate_MRI = parse_date_safe(EXAMDATE),
    MRI_3T_Available = as.integer(
      num(MMCONDCT) == 1 | num(MMSMPRAGE) == 1 | num(MMRMPRAGE) == 1
    )
  )

base_pib <- first_baseline(pib) %>%
  transmute(
    RID,
    Baseline_ExamDate_PIB = parse_date_safe(EXAMDATE),
    PIB_PET_Available = as.integer(num(PBCONDCT) == 1)
  )

base_av45 <- first_baseline(av45) %>%
  transmute(
    RID,
    Baseline_ExamDate_AV45 = parse_date_safe(EXAMDATE),
    AV45_PET_Available = as.integer(num(PBCONDCT) == 1)
  )

aibl_cohort_all <- mci_baseline %>%
  select(RID, baseline_DXCURREN, baseline_DXMCI, baseline_DXAD) %>%
  left_join(followup_dx, by = "RID") %>%
  left_join(base_mmse, by = "RID") %>%
  left_join(base_cdr, by = "RID") %>%
  left_join(base_neuro, by = "RID") %>%
  left_join(base_demog, by = "RID") %>%
  left_join(base_apoe, by = "RID") %>%
  left_join(base_lab, by = "RID") %>%
  left_join(base_mri3, by = "RID") %>%
  left_join(base_pib, by = "RID") %>%
  left_join(base_av45, by = "RID") %>%
  mutate(
    Baseline_ExamDate = coalesce(
      Baseline_ExamDate_MMSE,
      Baseline_ExamDate_CDR,
      Baseline_ExamDate_Neuro,
      Baseline_ExamDate_MRI,
      Baseline_ExamDate_PIB,
      Baseline_ExamDate_AV45
    ),
    Age = ifelse(!is.na(Birth_Year) & !is.na(Baseline_ExamDate),
                 lubridate::year(Baseline_ExamDate) - Birth_Year,
                 NA_real_),
    AD_Conversion = as.integer(AD_Conversion),
    Followup_Years = as.numeric(Followup_Years)
  ) %>%
  relocate(RID, AD_Conversion, Followup_Years, Time_to_Event_Months,
           Age, Gender, MMSE_Baseline, CDR_Baseline,
           LogicalMemory_Imm, LogicalMemory_Del,
           APOE4_Count, APOE4_Positive)

aibl_eligible <- aibl_cohort_all %>%
  filter(!is.na(Followup_Years), Followup_Years >= 0.5, !is.na(AD_Conversion))

# =========================
# 4. PRISMA-style flow
# =========================
prisma <- tibble::tibble(
  Step = c(
    "Baseline diagnostic records",
    "Baseline non-AD MCI",
    "Baseline non-AD MCI with any post-baseline diagnostic follow-up",
    "Eligible baseline MCI with follow-up >=0.5 years",
    "Eligible AD converters",
    "Eligible non-converters"
  ),
  N = c(
    nrow(baseline_dx),
    nrow(mci_baseline),
    sum(mci_baseline$RID %in% followup_dx$RID),
    nrow(aibl_eligible),
    sum(aibl_eligible$AD_Conversion == 1, na.rm = TRUE),
    sum(aibl_eligible$AD_Conversion == 0, na.rm = TRUE)
  )
)

# =========================
# 5. Feature overlap audit
# =========================
adni_holdout_cols <- names(read_csv_chr(adni_holdout_path))
adni_discovery_cols <- if (file.exists(adni_discovery_path)) names(read_csv_chr(adni_discovery_path)) else character(0)

model_feature_names <- character(0)
feature_source <- "not_found"

if (file.exists(model_config_path)) {
  cfg <- tryCatch(readRDS(model_config_path), error = function(e) NULL)
  if (!is.null(cfg) && !is.null(cfg$feature_names)) {
    model_feature_names <- as.character(cfg$feature_names)
    feature_source <- "model_config.rds$feature_names"
  }
}

if (length(model_feature_names) == 0 && file.exists(feature_importance_path)) {
  fi <- read_csv_chr(feature_importance_path)
  if ("Feature" %in% names(fi)) {
    model_feature_names <- unique(fi$Feature)
    feature_source <- "Feature_Importance_RF.csv$Feature"
  }
}

if (length(model_feature_names) == 0) {
  model_feature_names <- setdiff(adni_holdout_cols, c("ID", "RID", "AD_Conversion", "Conversion_Label", "STATUS"))
  feature_source <- "fallback_independent_test_set_header"
}

aibl_cols <- names(aibl_eligible)

alias_map <- tibble::tribble(
  ~required_feature, ~aibl_candidate, ~alias_note,
  "MMSE", "MMSE_Baseline", "same clinical scale, renamed",
  "MMSE_Baseline", "MMSE_Baseline", "same clinical scale",
  "Age", "Age", "same concept after recalculation from birth year and exam year",
  "AGE", "Age", "same concept after recalculation from birth year and exam year",
  "Gender", "Gender", "same concept; coding must be verified",
  "SEX", "Gender", "same concept; coding must be verified",
  "Education", NA_character_, "not available in current AIBL raw extract used here",
  "GDS", NA_character_, "not available in current AIBL raw extract used here",
  "APOE4_Positive", "APOE4_Positive", "same concept derived from APGEN1/APGEN2",
  "APOE4_Copies", "APOE4_Count", "same concept derived from APGEN1/APGEN2",
  "APOE4_Count", "APOE4_Count", "same concept derived from APGEN1/APGEN2",
  "CDRSB", "CDR_Baseline", "not identical: ADNI CDRSB vs AIBL CDGLOBAL",
  "CDGLOBAL", "CDR_Baseline", "same global CDR concept",
  "LIMMTOTAL", "LogicalMemory_Imm", "same AIBL raw logical memory immediate score",
  "LDELTOTAL", "LogicalMemory_Del", "same AIBL raw logical memory delayed score"
)

feature_group <- function(x) {
  dplyr::case_when(
    str_detect(x, "^ST\\d|^Z\\d|^Latent") ~ "MRI_or_latent",
    str_detect(x, "ABETA|TAU|PTAU|STREM2|PGRN|CSF") ~ "CSF_or_AD_biomarker",
    str_detect(x, "MMSE|CDR|ADAS|FAQ|GDS|Age|AGE|Gender|SEX|Education|APOE") ~ "clinical_or_genetic",
    TRUE ~ "other"
  )
}

feature_audit <- tibble::tibble(required_feature = unique(model_feature_names)) %>%
  left_join(alias_map, by = "required_feature") %>%
  mutate(
    feature_source = feature_source,
    feature_group = feature_group(required_feature),
    exact_available_in_AIBL = required_feature %in% aibl_cols,
    alias_available_in_AIBL = !is.na(aibl_candidate) & aibl_candidate %in% aibl_cols,
    availability_status = case_when(
      exact_available_in_AIBL ~ "exact_available",
      alias_available_in_AIBL ~ "alias_available_not_identical",
      TRUE ~ "missing"
    )
  ) %>%
  arrange(feature_group, availability_status, required_feature)

feature_gate <- feature_audit %>%
  summarise(
    feature_source = first(feature_source),
    required_features = n(),
    exact_available = sum(exact_available_in_AIBL),
    exact_or_alias_available = sum(availability_status != "missing"),
    exact_or_alias_pct = exact_or_alias_available / required_features,
    mri_or_latent_required = sum(feature_group == "MRI_or_latent"),
    mri_or_latent_available = sum(feature_group == "MRI_or_latent" & availability_status != "missing"),
    csf_or_ad_biomarker_required = sum(feature_group == "CSF_or_AD_biomarker"),
    csf_or_ad_biomarker_available = sum(feature_group == "CSF_or_AD_biomarker" & availability_status != "missing"),
    clinical_or_genetic_required = sum(feature_group == "clinical_or_genetic"),
    clinical_or_genetic_available = sum(feature_group == "clinical_or_genetic" & availability_status != "missing"),
    .groups = "drop"
  )

# =========================
# 6. Reduced-feature feasibility
# =========================
candidate_reduced_features <- c(
  "Age", "Gender", "MMSE_Baseline", "CDR_Baseline",
  "LogicalMemory_Imm", "LogicalMemory_Del",
  "APOE4_Count", "APOE4_Positive",
  "MRI_3T_Available", "PIB_PET_Available", "AV45_PET_Available",
  "AXT117", "BAT126", "HMT3", "HMT7", "HMT13", "HMT40",
  "HMT100", "HMT102", "RCT6", "RCT11", "RCT20", "RCT392"
)

missingness_summary <- tibble::tibble(feature = intersect(candidate_reduced_features, names(aibl_eligible))) %>%
  mutate(
    non_missing_n = map_int(feature, ~ sum(!is.na(aibl_eligible[[.x]]))),
    missing_n = nrow(aibl_eligible) - non_missing_n,
    missing_pct = ifelse(nrow(aibl_eligible) > 0, missing_n / nrow(aibl_eligible), NA_real_),
    usable_for_reduced_model = missing_pct <= 0.40
  )

core_reduced <- c("Age", "Gender", "MMSE_Baseline", "CDR_Baseline",
                  "LogicalMemory_Del", "APOE4_Positive")

core_reduced_status <- tibble::tibble(feature = core_reduced) %>%
  mutate(
    available = feature %in% names(aibl_eligible),
    non_missing_n = map_int(feature, ~ if (.x %in% names(aibl_eligible)) sum(!is.na(aibl_eligible[[.x]])) else 0L),
    non_missing_pct = ifelse(nrow(aibl_eligible) > 0, non_missing_n / nrow(aibl_eligible), NA_real_),
    usable = available & non_missing_pct >= 0.60
  )

eligible_n <- nrow(aibl_eligible)
event_n <- sum(aibl_eligible$AD_Conversion == 1, na.rm = TRUE)
nonevent_n <- sum(aibl_eligible$AD_Conversion == 0, na.rm = TRUE)
event_rate <- ifelse(eligible_n > 0, event_n / eligible_n, NA_real_)

full_feature_pass <- with(feature_gate, {
  eligible_n >= 30 &&
    event_n >= 10 &&
    required_features > 0 &&
    exact_or_alias_pct >= 0.80 &&
    (mri_or_latent_required == 0 || mri_or_latent_available / mri_or_latent_required >= 0.80) &&
    (csf_or_ad_biomarker_required == 0 || csf_or_ad_biomarker_available / csf_or_ad_biomarker_required >= 0.50)
})

reduced_feature_pass <- eligible_n >= 30 &&
  event_n >= 10 &&
  sum(core_reduced_status$usable) >= 5

reader_study_feasible <- eligible_n >= 30 &&
  event_n >= 10 &&
  all(c("Age", "MMSE_Baseline", "CDR_Baseline") %in% names(aibl_eligible)) &&
  all(core_reduced_status$non_missing_pct[match(c("Age", "MMSE_Baseline", "CDR_Baseline"),
                                                core_reduced_status$feature)] >= 0.60)

final_decision <- dplyr::case_when(
  full_feature_pass ~ "PASS_full_feature_direct_external_validation",
  reduced_feature_pass ~ "PASS_reduced_feature_external_validation",
  reader_study_feasible ~ "PASS_reader_study_feasible_but_model_validation_limited",
  TRUE ~ "FAIL_contextualization_only"
)

decision_summary <- tibble::tibble(
  decision = final_decision,
  eligible_n = eligible_n,
  event_n = event_n,
  nonevent_n = nonevent_n,
  event_rate = event_rate,
  full_feature_pass = full_feature_pass,
  reduced_feature_pass = reduced_feature_pass,
  reader_study_feasible = reader_study_feasible,
  required_model_feature_source = feature_gate$feature_source,
  required_features = feature_gate$required_features,
  exact_available = feature_gate$exact_available,
  exact_or_alias_available = feature_gate$exact_or_alias_available,
  exact_or_alias_pct = feature_gate$exact_or_alias_pct,
  mri_or_latent_required = feature_gate$mri_or_latent_required,
  mri_or_latent_available = feature_gate$mri_or_latent_available,
  csf_or_ad_biomarker_required = feature_gate$csf_or_ad_biomarker_required,
  csf_or_ad_biomarker_available = feature_gate$csf_or_ad_biomarker_available,
  reduced_core_usable_n = sum(core_reduced_status$usable),
  recommended_manuscript_position = dplyr::case_when(
    final_decision == "PASS_full_feature_direct_external_validation" ~
      "Use AIBL as direct external validation of the frozen full-feature model.",
    final_decision == "PASS_reduced_feature_external_validation" ~
      "Use AIBL as external validation of a prespecified ADNI-trained shared-feature reduced model; do not claim full-feature frozen-model validation.",
    final_decision == "PASS_reader_study_feasible_but_model_validation_limited" ~
      "Use AIBL for a blinded retrospective reader study or contextual external cohort; model validation remains limited.",
    TRUE ~
      "Keep AIBL as contextualization/supplement only unless additional same-pipeline features are generated."
  )
)

# =========================
# 7. Blinded case packet for possible external reader study
# =========================
reader_base <- aibl_eligible %>%
  arrange(num(RID)) %>%
  mutate(Case_ID = sprintf("AIBL_%03d", row_number()))

reader_packet <- reader_base %>%
  transmute(
    Case_ID,
    Age,
    Gender,
    MMSE_Baseline,
    CDR_Baseline,
    LogicalMemory_Imm,
    LogicalMemory_Del,
    APOE4_Positive,
    APOE4_Count,
    MRI_3T_Available,
    PIB_PET_Available,
    AV45_PET_Available,
    AXT117, BAT126, HMT3, HMT7, HMT13, HMT40,
    HMT100, HMT102, RCT6, RCT11, RCT20, RCT392
  )

reader_key <- reader_base %>%
  transmute(
    Case_ID,
    RID,
    AD_Conversion,
    Followup_Years,
    Time_to_Event_Months,
    baseline_DXCURREN,
    baseline_DXMCI,
    n_followup_dx_visits
  )

# =========================
# 8. Outputs
# =========================
write_csv_safe(aibl_cohort_all, file.path(out_dir, "01_aibl_all_baseline_mci_rebuilt.csv"))
write_csv_safe(aibl_eligible, file.path(out_dir, "02_aibl_eligible_mci_to_ad_conversion_cohort.csv"))
write_csv_safe(prisma, file.path(out_dir, "03_aibl_prisma_sample_flow.csv"))
write_csv_safe(feature_audit, file.path(out_dir, "04_aibl_vs_adni_feature_overlap_audit.csv"))
write_csv_safe(feature_gate, file.path(out_dir, "05_aibl_feature_gate_summary.csv"))
write_csv_safe(missingness_summary, file.path(out_dir, "06_aibl_reduced_feature_missingness.csv"))
write_csv_safe(core_reduced_status, file.path(out_dir, "07_aibl_reduced_core_feature_status.csv"))
write_csv_safe(decision_summary, file.path(out_dir, "08_aibl_gate_decision_summary.csv"))
write_csv_safe(reader_packet, file.path(out_dir, "09_aibl_reader_study_blinded_case_packet.csv"))
write_csv_safe(reader_key, file.path(out_dir, "10_aibl_reader_study_outcome_key_do_not_share.csv"))

readme <- c(
  "AIBL FEASIBILITY GATE",
  paste0("Generated: ", format(Sys.time(), "%Y-%m-%d %H:%M:%S")),
  "",
  "Inputs:",
  paste0("  AIBL raw directory: ", aibl_dir),
  paste0("  ADNI holdout: ", adni_holdout_path),
  paste0("  ADNI model config: ", model_config_path),
  paste0("  Feature source used: ", feature_gate$feature_source),
  "",
  "Decision:",
  paste0("  ", decision_summary$decision),
  "",
  "Key counts:",
  paste0("  Eligible baseline MCI with follow-up >=0.5 years: ", eligible_n),
  paste0("  AD converters: ", event_n),
  paste0("  Non-converters: ", nonevent_n),
  paste0("  Event rate: ", round(event_rate, 3)),
  "",
  "Interpretation:",
  paste0("  ", decision_summary$recommended_manuscript_position),
  "",
  "Critical notes:",
  "  1. Full-feature direct external validation requires same-pipeline ADNI-equivalent features.",
  "  2. Alias matches such as CDRSB -> CDGLOBAL are not identical and should not be treated as exact feature identity.",
  "  3. Reader packet is blinded; outcome key must not be shared with clinicians.",
  "",
  "Main outputs:",
  "  02_aibl_eligible_mci_to_ad_conversion_cohort.csv",
  "  04_aibl_vs_adni_feature_overlap_audit.csv",
  "  08_aibl_gate_decision_summary.csv",
  "  09_aibl_reader_study_blinded_case_packet.csv",
  "  10_aibl_reader_study_outcome_key_do_not_share.csv"
)
writeLines(readme, file.path(out_dir, "README_AIBL_feasibility_gate.txt"), useBytes = TRUE)

cat("\nAIBL feasibility gate complete.\n")
cat("Output directory:\n  ", out_dir, "\n", sep = "")
cat("\nDecision:\n")
print(decision_summary)
cat("\nPRISMA sample flow:\n")
print(prisma)
cat("\nCore reduced-feature status:\n")
print(core_reduced_status)
