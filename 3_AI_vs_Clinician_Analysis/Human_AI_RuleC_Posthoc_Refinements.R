options(stringsAsFactors = FALSE)

library(optparse)

option_list <- list(
  make_option(c("--rulec_dir"), type = "character", default = "./3_AI_vs_Clinician_Analysis/Q1_Human_AI_Extension",
              help = "Output directory from Human_AI_RuleC_Workflow_Extension.R [default: %default]"),
  make_option(c("--output_dir"), type = "character", default = NULL,
              help = "Output directory [default: <rulec_dir>/posthoc_analyses]"),
  make_option(c("--seed"), type = "integer", default = 20260614,
              help = "Random seed [default: %default]")
)
opt <- parse_args(OptionParser(option_list = option_list))
set.seed(opt$seed)

required_pkgs <- c("readr", "dplyr", "tibble", "tidyr", "purrr", "broom", "stringr")
missing_pkgs <- required_pkgs[!vapply(required_pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_pkgs) > 0) install.packages(missing_pkgs, dependencies = TRUE)

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(tibble)
  library(tidyr)
  library(purrr)
  library(broom)
  library(stringr)
})

base_dir <- normalizePath(opt$rulec_dir, winslash = "/", mustWork = FALSE)
master_path <- file.path(base_dir, "00_case_level_master_with_combined_predictions.csv")
if (!file.exists(master_path)) {
  stop("Missing master file: ", master_path, call. = FALSE)
}

out_dir <- if (is.null(opt$output_dir)) file.path(base_dir, "posthoc_analyses") else opt$output_dir
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

write_csv_safe <- function(x, path) readr::write_csv(x, path, na = "")
read_csv_safe <- function(path, ...) readr::read_csv(path, ...)

zscore_safe <- function(x) {
  x <- as.numeric(x)
  if (all(is.na(x))) return(x)
  s <- stats::sd(x, na.rm = TRUE)
  if (!is.finite(s) || s == 0) return(rep(0, length(x)))
  as.numeric(scale(x))
}

as_binary <- function(x) {
  if (is.factor(x)) x <- as.character(x)
  if (is.character(x)) {
    x_low <- tolower(trimws(x))
    return(as.integer(x_low %in% c("1", "yes", "true", "ad", "converter", "positive", "high")))
  }
  as.integer(x)
}

bootstrap_ci <- function(x, probs = c(0.025, 0.975)) {
  stats::quantile(x, probs = probs, na.rm = TRUE, names = FALSE)
}

master <- read_csv_safe(master_path, show_col_types = FALSE, progress = FALSE)

# -------------------------------------------------------------------------
# 1) Z-scored adjusted discordance
# -------------------------------------------------------------------------

adjustment_covariates <- c("Age", "Education", "MMSE_Baseline", "ADAS13", "CDRSB", "APOE4_Copies")
adjustment_covariates <- adjustment_covariates[adjustment_covariates %in% names(master)]

candidate_predictors <- c(
  "ai_minus_expert2_prob",
  "expert_stage2_prob_sd",
  "expert_mri_delta",
  "ABETA42_ABETA40_RATIO",
  "PTAU181",
  "TAU_TOTAL",
  "ST105CV",
  "ST102TS",
  "ST108TS",
  "ST109CV",
  "ST103TA",
  "ST108TA",
  "ST106TA",
  "ST102TA",
  "ST107CV",
  "ST111CV",
  "ST104CV"
)
candidate_predictors <- candidate_predictors[candidate_predictors %in% names(master)]

z_cols <- unique(c(adjustment_covariates, candidate_predictors))
z_cols <- z_cols[vapply(master[z_cols], is.numeric, logical(1))]

std_master <- master %>%
  mutate(across(all_of(z_cols), zscore_safe)) %>%
  mutate(Gender = if ("Gender" %in% names(.)) factor(Gender) else Gender)

fit_adjusted_discordance_z <- function(dat, target_group, predictor, covariates) {
  tmp <- dat %>%
    mutate(target = as.integer(discordance_group == target_group)) %>%
    select(target, all_of(c(predictor, covariates, if ("Gender" %in% names(dat)) "Gender" else NULL))) %>%
    tidyr::drop_na()

  if (nrow(tmp) < 30 || length(unique(tmp$target)) < 2) {
    return(tibble(
      target_group = target_group,
      predictor = predictor,
      n = nrow(tmp),
      estimate = NA_real_,
      std.error = NA_real_,
      statistic = NA_real_,
      p.value = NA_real_,
      odds_ratio = NA_real_,
      odds_ratio_ci_low = NA_real_,
      odds_ratio_ci_high = NA_real_
    ))
  }

  rhs <- c(predictor, covariates)
  if ("Gender" %in% names(tmp)) rhs <- c(rhs, "Gender")
  form <- as.formula(paste("target ~", paste(rhs, collapse = " + ")))
  fit <- tryCatch(glm(form, data = tmp, family = binomial()), error = function(e) NULL)
  if (is.null(fit)) {
    return(tibble(
      target_group = target_group,
      predictor = predictor,
      n = nrow(tmp),
      estimate = NA_real_,
      std.error = NA_real_,
      statistic = NA_real_,
      p.value = NA_real_,
      odds_ratio = NA_real_,
      odds_ratio_ci_low = NA_real_,
      odds_ratio_ci_high = NA_real_
    ))
  }

  row <- broom::tidy(fit, conf.int = TRUE) %>%
    filter(term == predictor) %>%
    mutate(
      target_group = target_group,
      predictor = predictor,
      n = nrow(tmp),
      odds_ratio = exp(estimate),
      odds_ratio_ci_low = exp(conf.low),
      odds_ratio_ci_high = exp(conf.high)
    ) %>%
    select(target_group, predictor, n, estimate, std.error, statistic, p.value,
           odds_ratio, odds_ratio_ci_low, odds_ratio_ci_high)

  if (nrow(row) == 0) {
    row <- tibble(
      target_group = target_group,
      predictor = predictor,
      n = nrow(tmp),
      estimate = NA_real_,
      std.error = NA_real_,
      statistic = NA_real_,
      p.value = NA_real_,
      odds_ratio = NA_real_,
      odds_ratio_ci_low = NA_real_,
      odds_ratio_ci_high = NA_real_
    )
  }
  row
}

zscore_discordance_results <- crossing(
  target_group = c("AI correct / Expert wrong", "Expert correct / AI wrong"),
  predictor = candidate_predictors
) %>%
  mutate(result = map2(target_group, predictor, ~ fit_adjusted_discordance_z(std_master, .x, .y, adjustment_covariates))) %>%
  select(result) %>%
  unnest(result) %>%
  mutate(p_fdr = p.adjust(p.value, method = "BH")) %>%
  arrange(target_group, p.value)

write_csv_safe(zscore_discordance_results, file.path(out_dir, "10_adjusted_discordance_zscore_results.csv"))

# -------------------------------------------------------------------------
# 2) Paired FP/FN comparison: Rule C vs Expert Stage 2
# -------------------------------------------------------------------------

paired_rulec_error_change <- function(dat, ref_pred = "expert_stage2_pred", new_pred = "second_reader_40_60_pred",
                                      truth_col = "outcome", reps = 5000, seed = 20260614) {
  y <- as_binary(dat[[truth_col]])
  ref <- as_binary(dat[[ref_pred]])
  new <- as_binary(dat[[new_pred]])
  keep <- complete.cases(y, ref, new)
  y <- y[keep]
  ref <- ref[keep]
  new <- new[keep]

  non <- y == 0
  ev <- y == 1

  fp_ref <- sum(ref[non] == 1)
  fp_new <- sum(new[non] == 1)
  fn_ref <- sum(ref[ev] == 0)
  fn_new <- sum(new[ev] == 0)

  tab_fp <- table(factor(ref[non], levels = c(0, 1)), factor(new[non], levels = c(0, 1)))
  tab_fn <- table(factor(ref[ev] == 0, levels = c(FALSE, TRUE)), factor(new[ev] == 0, levels = c(FALSE, TRUE)))

  fp_p <- tryCatch(if (all(dim(tab_fp) == c(2, 2))) stats::mcnemar.test(tab_fp)$p.value else NA_real_,
                   error = function(e) NA_real_)
  fn_p <- tryCatch(if (all(dim(tab_fn) == c(2, 2))) stats::mcnemar.test(tab_fn)$p.value else NA_real_,
                   error = function(e) NA_real_)

  set.seed(seed)
  boot <- replicate(reps, {
    idx <- sample(seq_along(y), replace = TRUE)
    yy <- y[idx]
    rr <- ref[idx]
    nn <- new[idx]
    nonb <- yy == 0
    evb <- yy == 1
    c(
      delta_fp = sum(nn[nonb] == 1) - sum(rr[nonb] == 1),
      delta_fn = sum(nn[evb] == 0) - sum(rr[evb] == 0)
    )
  })
  boot <- t(boot)
  ci_fp <- bootstrap_ci(boot[, "delta_fp"])
  ci_fn <- bootstrap_ci(boot[, "delta_fn"])

  tibble(
    comparison = "Rule C primary 40-60 vs Expert Stage 2",
    n = length(y),
    n_events = sum(ev),
    n_nonevents = sum(non),
    fp_ref = fp_ref,
    fp_rulec = fp_new,
    delta_fp = fp_new - fp_ref,
    avoided_fp = fp_ref - fp_new,
    fp_mcnemar_p = fp_p,
    fp_boot_ci_low = ci_fp[1],
    fp_boot_ci_high = ci_fp[2],
    fn_ref = fn_ref,
    fn_rulec = fn_new,
    delta_fn = fn_new - fn_ref,
    extra_fn = fn_new - fn_ref,
    fn_mcnemar_p = fn_p,
    fn_boot_ci_low = ci_fn[1],
    fn_boot_ci_high = ci_fn[2]
  )
}

rulec_error_results <- paired_rulec_error_change(master, reps = 5000)
write_csv_safe(rulec_error_results, file.path(out_dir, "11_ruleC_fp_fn_paired_comparison.csv"))

# -------------------------------------------------------------------------
# 3) Bootstrap CI for DCA net benefit
# -------------------------------------------------------------------------

net_benefit_vec <- function(y, p, thresholds) {
  y <- as_binary(y)
  p <- as.numeric(p)
  keep <- !is.na(y) & is.finite(p)
  y <- y[keep]
  p <- p[keep]
  n <- length(y)
  vapply(thresholds, function(pt) {
    pred <- as.integer(p >= pt)
    tp <- sum(pred == 1 & y == 1)
    fp <- sum(pred == 1 & y == 0)
    (tp / n) - (fp / n) * (pt / (1 - pt))
  }, numeric(1))
}

dca_bootstrap_ci <- function(dat, model_cols, thresholds = seq(0.05, 0.80, by = 0.01),
                             reps = 1000, seed = 20260614) {
  set.seed(seed)
  point <- map_dfr(model_cols, function(m) {
    tibble(
      model = m,
      threshold = thresholds,
      net_benefit = net_benefit_vec(dat$outcome, dat[[m]], thresholds)
    )
  })

  boot_summary <- map_dfr(model_cols, function(m) {
    boot_mat <- replicate(reps, {
      idx <- sample(seq_len(nrow(dat)), replace = TRUE)
      net_benefit_vec(dat$outcome[idx], dat[[m]][idx], thresholds)
    })
    if (is.null(dim(boot_mat))) boot_mat <- matrix(boot_mat, nrow = length(thresholds))
    ci_low <- apply(boot_mat, 1, stats::quantile, probs = 0.025, na.rm = TRUE)
    ci_high <- apply(boot_mat, 1, stats::quantile, probs = 0.975, na.rm = TRUE)
    tibble(
      model = m,
      threshold = thresholds,
      nb_ci_low = as.numeric(ci_low),
      nb_ci_high = as.numeric(ci_high)
    )
  })

  left_join(point, boot_summary, by = c("model", "threshold"))
}

dca_models <- c(
  "ai_prob",
  "expert_stage2_prob",
  "second_reader_40_60_prob",
  "mean_ai_expert2_prob",
  "rankmean_ai_expert2_prob",
  "expert_uncertain_ai_specificity_prune_prob"
)
dca_models <- dca_models[dca_models %in% names(master)]

dca_curve_ci <- dca_bootstrap_ci(master, dca_models, reps = 1000)
write_csv_safe(dca_curve_ci, file.path(out_dir, "12_dca_net_benefit_bootstrap_ci_curve.csv"))

key_thresholds <- c(0.20, 0.30, 0.50)
dca_key_threshold_ci <- dca_curve_ci %>%
  filter(round(threshold, 2) %in% key_thresholds) %>%
  arrange(model, threshold)
write_csv_safe(dca_key_threshold_ci, file.path(out_dir, "12_dca_net_benefit_bootstrap_ci_key_thresholds.csv"))

summary_lines <- c(
  "Post-hoc analyses completed.",
  paste0("Output dir: ", out_dir),
  "Files:",
  "10_adjusted_discordance_zscore_results.csv",
  "11_ruleC_fp_fn_paired_comparison.csv",
  "12_dca_net_benefit_bootstrap_ci_curve.csv",
  "12_dca_net_benefit_bootstrap_ci_key_thresholds.csv"
)
writeLines(summary_lines, con = file.path(out_dir, "README_posthoc_analyses.txt"))

cat("Done.\n")
cat("Outputs written to: ", out_dir, "\n", sep = "")
