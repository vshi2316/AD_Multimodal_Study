options(stringsAsFactors = FALSE)

suppressPackageStartupMessages({
  library(optparse)
  library(readr)
  library(dplyr)
  library(tidyr)
  library(purrr)
  library(ggplot2)
  library(broom)
  library(stringr)
  library(patchwork)
})

option_list <- list(
  make_option(c("--rulec_dir"), type = "character", default = "./3_AI_vs_Clinician_Analysis/Q1_Human_AI_Extension",
              help = "Directory from Human_AI_RuleC_Workflow_Extension.R [default: %default]"),
  make_option(c("--no_vae_dir"), type = "character", default = "./3_AI_vs_Clinician_Analysis/NoVAE_Sensitivity",
              help = "Directory from AI_Prediction_NoVAE.py [default: %default]"),
  make_option(c("--data_root"), type = "character", default = ".",
              help = "Repository/data root [default: %default]"),
  make_option(c("--output_dir"), type = "character", default = "./3_AI_vs_Clinician_Analysis/Longitudinal_Outcome_Sensitivity",
              help = "Output directory [default: %default]"),
  make_option(c("--seed"), type = "integer", default = 20260614,
              help = "Random seed [default: %default]")
)

opt <- parse_args(OptionParser(option_list = option_list))
set.seed(opt$seed)

base_dir <- normalizePath(opt$data_root, winslash = "/", mustWork = FALSE)
rulec_dir <- normalizePath(opt$rulec_dir, winslash = "/", mustWork = FALSE)
no_vae_dir <- normalizePath(opt$no_vae_dir, winslash = "/", mustWork = FALSE)
out_dir <- normalizePath(opt$output_dir, winslash = "/", mustWork = FALSE)
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
fig_dir <- file.path(out_dir, "figures")
dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)

read_csv_safe <- function(path) {
  if (!file.exists(path)) stop("Missing required file: ", path, call. = FALSE)
  readr::read_csv(path, show_col_types = FALSE, progress = FALSE)
}

write_csv_safe <- function(x, path) readr::write_csv(x, path, na = "")

first_existing <- function(...) {
  paths <- c(...)
  for (p in paths) if (file.exists(p)) return(p)
  paths[[1]]
}

as_binary <- function(x) {
  if (is.factor(x)) x <- as.character(x)
  if (is.character(x)) {
    x_low <- tolower(trimws(x))
    return(as.integer(x_low %in% c("1", "yes", "true", "ad", "converter", "positive", "high")))
  }
  as.integer(x)
}

zscore_safe <- function(x) {
  x <- as.numeric(x)
  if (all(is.na(x))) return(x)
  s <- stats::sd(x, na.rm = TRUE)
  if (!is.finite(s) || s == 0) return(rep(0, length(x)))
  as.numeric(scale(x))
}

parse_date_safe <- function(x) {
  if (inherits(x, "Date")) return(x)
  x <- as.character(x)
  out <- suppressWarnings(as.Date(x))
  if (all(is.na(out))) {
    out <- suppressWarnings(as.Date(x, format = "%m/%d/%Y"))
  }
  if (all(is.na(out))) {
    out <- suppressWarnings(as.Date(x, format = "%Y-%m-%d"))
  }
  out
}

safe_num <- function(x) suppressWarnings(as.numeric(x))

get_existing_col <- function(df, candidates) {
  for (cand in candidates) {
    if (cand %in% names(df)) return(cand)
    hit <- names(df)[tolower(names(df)) == tolower(cand)]
    if (length(hit) > 0) return(hit[[1]])
  }
  NA_character_
}

extract_subject_slopes <- function(df, rid_col = "RID", date_candidates = c("EXAMDATE", "VISDATE"), value_col) {
  rid_col <- get_existing_col(df, c(rid_col))
  if (is.na(rid_col) || !(value_col %in% names(df))) return(NULL)
  date_col <- get_existing_col(df, date_candidates)
  if (is.na(date_col)) return(NULL)

  dat <- df %>%
    transmute(
      RID = safe_num(.data[[rid_col]]),
      visit_date = parse_date_safe(.data[[date_col]]),
      value = safe_num(.data[[value_col]])
    ) %>%
    filter(!is.na(RID), !is.na(visit_date), is.finite(value)) %>%
    arrange(RID, visit_date) %>%
    group_by(RID) %>%
    mutate(
      baseline_date = first(visit_date),
      years_from_baseline = as.numeric(difftime(visit_date, baseline_date, units = "days")) / 365.25
    ) %>%
    ungroup()

  slope_df <- dat %>%
    group_by(RID) %>%
    filter(n() >= 2, max(years_from_baseline, na.rm = TRUE) > 0) %>%
    summarise(
      n_visits = n(),
      followup_years = max(years_from_baseline, na.rm = TRUE),
      baseline_value = first(value),
      last_value = last(value),
      slope_per_year = coef(lm(value ~ years_from_baseline))[["years_from_baseline"]],
      .groups = "drop"
    )
  slope_df
}

model_table <- function(dat, outcome, predictor, baseline_col = NULL, adjust_bsi = FALSE) {
  d <- dat %>% filter(is.finite(.data[[outcome]]), is.finite(.data[[predictor]]))
  if (!is.null(baseline_col) && baseline_col %in% names(d)) {
    d <- d %>% filter(is.finite(.data[[baseline_col]]))
  }
  covars <- c("Age", "Gender", "Education", "APOE4_Copies", "MMSE_Baseline")
  covars <- intersect(covars, names(d))
  if (!adjust_bsi && !is.null(baseline_col) && baseline_col %in% names(d)) covars <- c(covars, baseline_col)
  covars <- unique(covars)
  rhs <- c(predictor, covars)
  form <- as.formula(paste(outcome, "~", paste(rhs, collapse = " + ")))
  fit <- lm(form, data = d)
  td <- broom::tidy(fit, conf.int = TRUE) %>% filter(term == predictor)
  tib <- tibble(
    term = predictor,
    estimate = td$estimate,
    std_error = td$std.error,
    t_value = td$statistic,
    p_value = td$p.value,
    n = nrow(d),
    df = fit$df.residual,
    r2 = summary(fit)$r.squared,
    outcome = outcome,
    slope_direction = ifelse(outcome == "MMSE", "higher_worse", "higher_worse"),
    primary_predictor = predictor,
    primary_predictor_label = predictor,
    model_n = nrow(d),
    primary_term = TRUE
  )
  tib
}

spearman_table <- function(dat, outcome, probability_cols) {
  dat <- dat %>% filter(.data$outcome == outcome)
  rows <- list()
  y <- dat$slope_per_year
  for (pcol in probability_cols) {
    x <- dat[[pcol]]
    keep <- is.finite(y) & is.finite(x)
    if (sum(keep) < 5) next
    ct <- suppressWarnings(cor.test(x[keep], y[keep], method = "spearman", exact = FALSE))
    rows[[length(rows) + 1]] <- tibble(
      outcome = outcome,
      probability = pcol,
      n = sum(keep),
      spearman_rho = unname(ct$estimate),
      p_value = ct$p.value
    )
  }
  bind_rows(rows)
}

group_summary <- function(dat, outcome, label_col, label_name) {
  d <- dat %>%
    filter(.data$outcome == outcome, is.finite(slope_per_year), !is.na(.data[[label_col]])) %>%
    mutate(group = as.integer(.data[[label_col]]))
  grp <- d %>%
    group_by(group) %>%
    summarise(
      outcome = outcome,
      label = label_name,
      n = n(),
      mean_worse_slope = mean(slope_per_year, na.rm = TRUE),
      sd_worse_slope = sd(slope_per_year, na.rm = TRUE),
      median_worse_slope = median(slope_per_year, na.rm = TRUE),
      q1 = quantile(slope_per_year, 0.25, na.rm = TRUE),
      q3 = quantile(slope_per_year, 0.75, na.rm = TRUE),
      .groups = "drop"
    )
  hi <- d %>% filter(group == 1) %>% pull(slope_per_year)
  lo <- d %>% filter(group == 0) %>% pull(slope_per_year)
  test <- if (length(hi) > 1 && length(lo) > 1) stats::wilcox.test(hi, lo, exact = FALSE)$p.value else NA_real_
  diff <- mean(hi, na.rm = TRUE) - mean(lo, na.rm = TRUE)
  bind_rows(
    grp,
    tibble(
      outcome = outcome,
      label = label_name,
      group = "high_vs_low_test",
      n = nrow(d),
      mean_worse_slope = diff,
      sd_worse_slope = NA_real_,
      median_worse_slope = median(hi, na.rm = TRUE) - median(lo, na.rm = TRUE),
      q1 = NA_real_,
      q3 = NA_real_,
      p_value = test
    )
  )
}

load_master <- function() {
  master_path <- first_existing(
    file.path(rulec_dir, "00_case_level_master_with_combined_predictions.csv"),
    file.path(rulec_dir, "00_case_level_master.csv")
  )
  m <- read_csv_safe(master_path)
  if (!"ID" %in% names(m)) {
    stop("Master Rule C file must contain ID")
  }
  m
}

load_no_vae <- function() {
  no_vae_path <- first_existing(
    file.path(no_vae_dir, "00_no_vae_case_level_master.csv"),
    file.path(no_vae_dir, "AI_Predictions_Final_no_vae.csv"),
    file.path(no_vae_dir, "AI_per_patient_predictions_no_vae.csv")
  )
  read_csv_safe(no_vae_path)
}

load_raw_longitudinal <- function() {
  mmse_path <- first_existing(
    file.path(base_dir, "ADNI_Raw_Data", "LINES", "Mini-Mental State Examination (MMSE).csv"),
    file.path(base_dir, "LINES", "Mini-Mental State Examination (MMSE).csv")
  )
  adas_path <- first_existing(
    file.path(base_dir, "ADNI_Raw_Data", "LINES", "ADAS-Cognitive Behavior.csv"),
    file.path(base_dir, "LINES", "ADAS-Cognitive Behavior.csv")
  )
  cdr_path <- first_existing(
    file.path(base_dir, "ADNI_Raw_Data", "LINES", "Clinical Dementia Rating.csv"),
    file.path(base_dir, "LINES", "Clinical Dementia Rating.csv")
  )
  faq_path <- first_existing(
    file.path(base_dir, "ADNI_Raw_Data", "LINES", "Futional Activities Questionnaire.csv"),
    file.path(base_dir, "LINES", "Futional Activities Questionnaire.csv")
  )
  list(
    mmse = read_csv_safe(mmse_path),
    adas = read_csv_safe(adas_path),
    cdr = read_csv_safe(cdr_path),
    faq = read_csv_safe(faq_path)
  )
}

build_longitudinal_slopes <- function(raw_list) {
  mmse <- extract_subject_slopes(raw_list$mmse, value_col = "MMSCORE")
  adas <- extract_subject_slopes(raw_list$adas, value_col = "TOTAL13")
  cdr <- extract_subject_slopes(raw_list$cdr, value_col = "CDRSB")
  faq <- extract_subject_slopes(raw_list$faq, value_col = "FAQTOTAL")

  bind_rows(
    mmse %>% mutate(outcome = "MMSE", slope_direction = "higher_worse", slope_per_year = -slope_per_year),
    adas %>% mutate(outcome = "ADAS13", slope_direction = "higher_worse"),
    cdr %>% mutate(outcome = "CDRSB", slope_direction = "higher_worse"),
    faq %>% mutate(outcome = "FAQTOTAL", slope_direction = "higher_worse")
  )
}

make_bsi <- function(base_dir) {
  paths <- c(
    file.path(base_dir, "longitudinal_bsi_validation", "individual_bsi_slopes.csv"),
    file.path(base_dir, "longitudinal_bsi_validation", "bsi_longitudinal_merged.csv")
  )
  path <- paths[file.exists(paths)][1]
  if (is.na(path) || !file.exists(path)) return(NULL)
  dat <- read_csv_safe(path)
  if ("RID" %in% names(dat) && "slope_per_year" %in% names(dat)) {
    return(dat %>%
      transmute(
        RID = safe_num(RID),
        slope_per_year = safe_num(slope_per_year),
        n_visits = if ("n_visits" %in% names(dat)) safe_num(n_visits) else NA_real_,
        outcome = "BSI",
        slope_direction = "higher_worse"
      ))
  }
  if ("RID" %in% names(dat) && "DBCBBSI" %in% names(dat)) {
    return(dat %>%
      transmute(
        RID = safe_num(RID),
        slope_per_year = safe_num(DBCBBSI) / if ("Years" %in% names(dat)) pmax(safe_num(Years), 1e-6) else 1,
        n_visits = if ("n_visits" %in% names(dat)) safe_num(n_visits) else NA_real_,
        outcome = "BSI",
        slope_direction = "higher_worse"
      ) %>%
      group_by(RID, outcome, slope_direction) %>%
      summarise(slope_per_year = mean(slope_per_year, na.rm = TRUE), n_visits = first(n_visits), .groups = "drop"))
  }
  NULL
}

master <- load_master()
no_vae <- load_no_vae()
raw <- load_raw_longitudinal()
slopes <- build_longitudinal_slopes(raw)
bsi <- make_bsi(base_dir)
if (!is.null(bsi)) {
  slopes <- bind_rows(slopes, bsi)
}

master <- master %>%
  mutate(
    ID = as.character(ID),
    RID = if ("RID" %in% names(.)) safe_num(RID) else NA_real_,
    ai_prob = if ("ai_prob" %in% names(.)) safe_num(ai_prob) else safe_num(AI_Probability),
    expert_stage2_prob = if ("expert_stage2_prob" %in% names(.)) safe_num(expert_stage2_prob) else safe_num(Stage2_Prob),
    RuleC_Label = if ("second_reader_40_60_pred" %in% names(.)) as_binary(second_reader_40_60_pred) else as_binary(rulec_pred),
    Expert_Stage2_Label = if ("expert_stage2_pred" %in% names(.)) as_binary(expert_stage2_pred) else as_binary(expert_stage2_prob >= 0.5),
    AI_Prob_z = zscore_safe(ai_prob)
  )

no_vae <- no_vae %>%
  mutate(
    ID = if ("CaseID" %in% names(.)) as.character(CaseID) else as.character(ID),
    NoVAE_AI_Probability = if ("NoVAE_AI_Probability" %in% names(.)) safe_num(NoVAE_AI_Probability) else safe_num(Predicted_Prob),
    RuleC_NoVAE_Label = if ("rulec_no_vae_pred" %in% names(.)) as_binary(rulec_no_vae_pred) else NA_integer_
  ) %>%
  select(ID, NoVAE_AI_Probability, RuleC_NoVAE_Label)

master <- master %>%
  left_join(no_vae, by = "ID") %>%
  mutate(NoVAE_AI_Prob_z = zscore_safe(NoVAE_AI_Probability))

if (!"RID" %in% names(master)) stop("Rule C master file must contain RID for longitudinal linking")

master_map <- master %>%
  select(
    ID, RID,
    ai_prob, AI_Prob_z,
    expert_stage2_prob, Expert_Stage2_Label,
    RuleC_Label,
    NoVAE_AI_Probability, NoVAE_AI_Prob_z,
    RuleC_NoVAE_Label,
    any_of(c("Age", "Gender", "Education", "MMSE_Baseline", "ADAS13", "CDRSB", "FAQTOTAL", "APOE4_Copies"))
  ) %>%
  mutate(
    RuleC_Label = ifelse(is.na(RuleC_Label), as.integer(expert_stage2_prob >= 0.5), RuleC_Label),
    Expert_Stage2_Label = ifelse(is.na(Expert_Stage2_Label), as.integer(expert_stage2_prob >= 0.5), Expert_Stage2_Label),
    RuleC_NoVAE_Label = ifelse(is.na(RuleC_NoVAE_Label), as.integer(expert_stage2_prob >= 0.5), RuleC_NoVAE_Label)
  )

long_dat <- slopes %>%
  inner_join(master_map, by = "RID")

availability <- long_dat %>%
  group_by(outcome) %>%
  summarise(
    raw_records_in_holdout = n(),
    subjects_with_slope = n_distinct(RID[is.finite(slope_per_year)]),
    .groups = "drop"
  )
write_csv_safe(availability, file.path(out_dir, "47_slope_availability.csv"))

outcome_baseline_map <- c(
  "ADAS13" = "ADAS13",
  "CDRSB" = "CDRSB",
  "FAQTOTAL" = "FAQTOTAL",
  "MMSE" = "MMSE_Baseline"
)

predictor_specs <- tribble(
  ~primary_predictor, ~primary_predictor_label,
  "AI_Prob_z", "AI probability per SD",
  "RuleC_Label", "Rule C high-risk label",
  "Expert_Stage2_Label", "Expert Stage 2 high-risk label",
  "NoVAE_AI_Prob_z", "No-VAE AI probability per SD",
  "RuleC_NoVAE_Label", "Rule C no-VAE high-risk label"
)

model_rows <- list()
for (outcome in c("ADAS13", "CDRSB", "FAQTOTAL", "MMSE", "BSI")) {
  base_col <- outcome_baseline_map[[outcome]]
  dat0 <- long_dat %>% filter(.data$outcome == outcome, is.finite(slope_per_year), !is.na(RID))
  if (outcome != "BSI") {
    dat0 <- dat0 %>%
      filter(!is.na(.data[[base_col]]))
  }
  for (i in seq_len(nrow(predictor_specs))) {
    predictor <- predictor_specs$primary_predictor[[i]]
    label <- predictor_specs$primary_predictor_label[[i]]
    if (!predictor %in% names(dat0)) next
    d <- dat0 %>% filter(is.finite(.data[[predictor]]))
    if (nrow(d) < 10) next
    covars <- c("Age", "Gender", "Education", "APOE4_Copies", "MMSE_Baseline")
    covars <- intersect(covars, names(d))
    if (outcome != "BSI" && !is.na(base_col) && base_col %in% names(d)) {
      covars <- c(covars, base_col)
    }
    covars <- unique(covars)
    rhs <- c(predictor, covars)
    form <- as.formula(paste("slope_per_year ~", paste(rhs, collapse = " + ")))
    fit <- lm(form, data = d)
    td <- broom::tidy(fit, conf.int = TRUE) %>% filter(term == predictor)
    model_rows[[length(model_rows) + 1]] <- tibble(
      term = predictor,
      estimate = td$estimate,
      std_error = td$std.error,
      t_value = td$statistic,
      p_value = td$p.value,
      n = nrow(d),
      df = fit$df.residual,
      r2 = summary(fit)$r.squared,
      outcome = outcome,
      slope_direction = "higher_worse",
      primary_predictor = predictor,
      primary_predictor_label = label,
      model_n = nrow(d),
      primary_term = TRUE
    )
  }
}

trajectory_models <- bind_rows(model_rows) %>%
  mutate(fdr_q = p.adjust(p_value, method = "BH"))
write_csv_safe(trajectory_models, file.path(out_dir, "47_adjusted_trajectory_models.csv"))

prob_cols <- intersect(c("ai_prob", "NoVAE_AI_Probability", "expert_stage2_prob"), names(long_dat))
cor_rows <- map_dfr(c("ADAS13", "CDRSB", "FAQTOTAL", "MMSE", "BSI"), ~ spearman_table(long_dat, .x, prob_cols))
cor_rows <- cor_rows %>% mutate(fdr_q = p.adjust(p_value, method = "BH"))
write_csv_safe(cor_rows, file.path(out_dir, "48_probability_slope_correlations.csv"))

group_rows <- bind_rows(
  group_summary(long_dat, "ADAS13", "RuleC_Label", "RuleC_Label"),
  group_summary(long_dat, "CDRSB", "RuleC_Label", "RuleC_Label"),
  group_summary(long_dat, "FAQTOTAL", "RuleC_Label", "RuleC_Label"),
  group_summary(long_dat, "MMSE", "RuleC_Label", "RuleC_Label"),
  group_summary(long_dat, "ADAS13", "Expert_Stage2_Label", "Expert_Stage2_Label"),
  group_summary(long_dat, "CDRSB", "Expert_Stage2_Label", "Expert_Stage2_Label"),
  group_summary(long_dat, "FAQTOTAL", "Expert_Stage2_Label", "Expert_Stage2_Label"),
  group_summary(long_dat, "MMSE", "Expert_Stage2_Label", "Expert_Stage2_Label"),
  group_summary(long_dat, "ADAS13", "RuleC_NoVAE_Label", "RuleC_NoVAE_Label"),
  group_summary(long_dat, "CDRSB", "RuleC_NoVAE_Label", "RuleC_NoVAE_Label"),
  group_summary(long_dat, "FAQTOTAL", "RuleC_NoVAE_Label", "RuleC_NoVAE_Label"),
  group_summary(long_dat, "MMSE", "RuleC_NoVAE_Label", "RuleC_NoVAE_Label")
)
write_csv_safe(group_rows, file.path(out_dir, "48_rulec_group_slope_summaries.csv"))

forest_dat <- trajectory_models %>%
  mutate(
    label = paste0(primary_predictor_label, " | ", outcome),
    lo = estimate - 1.96 * std_error,
    hi = estimate + 1.96 * std_error
  ) %>%
  arrange(outcome, primary_predictor)

fig_a <- ggplot(forest_dat, aes(x = estimate, y = reorder(label, estimate))) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "gray50") +
  geom_errorbarh(aes(xmin = lo, xmax = hi), height = 0.12, color = "#4C78A8") +
  geom_point(size = 2, color = "#4C78A8") +
  labs(x = "Adjusted association with annualized worsening slope", y = NULL, title = "Longitudinal trajectory validation") +
  theme_minimal(base_size = 11)

box_plot <- function(outcome_name) {
  d <- long_dat %>%
    filter(outcome == outcome_name, is.finite(slope_per_year), !is.na(RuleC_Label)) %>%
    mutate(rulec = factor(RuleC_Label, levels = c(0, 1), labels = c("Low", "High")))
  ggplot(d, aes(x = rulec, y = slope_per_year, fill = rulec)) +
    geom_boxplot(width = 0.7, alpha = 0.65, outlier.shape = NA) +
    geom_jitter(width = 0.15, height = 0, size = 1, alpha = 0.55) +
    labs(x = "Rule C high-risk label", y = paste0(outcome_name, " annualized worsening slope"), title = paste0(outcome_name, " trajectory by Rule C")) +
    theme_minimal(base_size = 11) +
    theme(legend.position = "none")
}

fig_b <- box_plot("ADAS13")
fig_c <- box_plot("CDRSB")
fig_d <- box_plot("FAQTOTAL")
fig_e <- box_plot("MMSE")

fig_31 <- fig_a / (fig_b + fig_c) / (fig_d + fig_e) + plot_annotation(tag_levels = "A")
ggsave(file.path(fig_dir, "Supplementary_Figure_31_Longitudinal_Outcome.png"), fig_31, width = 14, height = 12, dpi = 300)
ggsave(file.path(fig_dir, "Supplementary_Figure_31_Longitudinal_Outcome.pdf"), fig_31, width = 14, height = 12)

summary_lines <- c(
  "Longitudinal holdout sensitivity analysis",
  sprintf("Master file: %s", file.path(rulec_dir, "00_case_level_master_with_combined_predictions.csv")),
  sprintf("No-VAE file: %s", first_existing(file.path(no_vae_dir, "AI_Predictions_Final_no_vae.csv"), file.path(no_vae_dir, "AI_per_patient_predictions_no_vae.csv"))),
  sprintf("Outcomes available: %s", paste(unique(long_dat$outcome), collapse = ", ")),
  sprintf("BSI linked: %s", ifelse(any(long_dat$outcome == "BSI", na.rm = TRUE), "yes", "no")),
  "",
  "This script computes subject-level annualized slopes from raw longitudinal ADNI records,",
  "then fits adjusted linear models and Spearman correlations against AI, Rule C, expert,",
  "and no-VAE Rule C predictors."
)
writeLines(summary_lines, con = file.path(out_dir, "README_longitudinal_sensitivity.txt"))

cat("Wrote outputs to ", out_dir, "\n", sep = "")
cat("Figures in ", fig_dir, "\n", sep = "")
