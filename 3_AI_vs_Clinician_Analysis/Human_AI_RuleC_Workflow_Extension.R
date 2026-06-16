# Human-AI Rule C workflow extension for AD multimodal study

library(optparse)

options(stringsAsFactors = FALSE)

option_list <- list(
  make_option(c("--data_root"), type = "character", default = ".",
              help = "Repository root or data root [default: %default]"),
  make_option(c("--output_dir"), type = "character", default = "./3_AI_vs_Clinician_Analysis/Q1_Human_AI_Extension",
              help = "Output directory [default: %default]"),
  make_option(c("--ai_file"), type = "character", default = "./AI_vs_Clinician_Test/AI_Predictions_Final.csv",
              help = "AI predictions CSV [default: %default]"),
  make_option(c("--expert_file"), type = "character", default = "./AI_vs_Clinician_Test/Expert_Predictions_Long.csv",
              help = "Expert predictions CSV [default: %default]"),
  make_option(c("--test_file"), type = "character", default = "./AI_vs_Clinician_Test/independent_test_set.csv",
              help = "Independent test set CSV [default: %default]"),
  make_option(c("--n_bootstrap"), type = "integer", default = 2000,
              help = "Bootstrap repetitions [default: %default]"),
  make_option(c("--cv_repeats"), type = "integer", default = 200,
              help = "Repeated CV repetitions for exploratory combinations [default: %default]"),
  make_option(c("--seed"), type = "integer", default = 20260614,
              help = "Random seed [default: %default]")
)

opt <- parse_args(OptionParser(option_list = option_list))
set.seed(opt$seed)

required_pkgs <- c("readr", "dplyr", "tidyr", "purrr", "tibble", "stringr", "ggplot2", "scales", "pROC", "broom")
missing_pkgs <- required_pkgs[!vapply(required_pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_pkgs) > 0) {
  install.packages(missing_pkgs, dependencies = TRUE)
}

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(tidyr)
  library(purrr)
  library(tibble)
  library(stringr)
  library(ggplot2)
  library(scales)
  library(pROC)
  library(broom)
})

base_dir <- normalizePath(opt$data_root, winslash = "/", mustWork = FALSE)
out_dir <- normalizePath(opt$output_dir, winslash = "/", mustWork = FALSE)
fig_dir <- file.path(out_dir, "figures")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)

ai_threshold_primary <- 0.5142
expert_threshold_primary <- 0.50
combined_threshold_primary <- 0.50
cv_repeats <- opt$cv_repeats
cv_folds <- 5
bootstrap_reps <- opt$n_bootstrap

# -------------------------------------------------------------------------
# 1. Utility functions
# -------------------------------------------------------------------------

read_csv_safe <- function(path) {
  if (!file.exists(path)) stop("Missing required file: ", path, call. = FALSE)
  readr::read_csv(path, show_col_types = FALSE, progress = FALSE)
}

write_csv_safe <- function(x, path) {
  readr::write_csv(x, path, na = "")
}

clip_prob <- function(x, eps = 1e-6) {
  pmin(pmax(as.numeric(x), eps), 1 - eps)
}

logit_clip <- function(x) {
  p <- clip_prob(x)
  log(p / (1 - p))
}

rank01 <- function(x) {
  x <- as.numeric(x)
  out <- rep(NA_real_, length(x))
  keep <- is.finite(x)
  if (sum(keep) == 0) return(out)
  out[keep] <- (rank(x[keep], ties.method = "average") - 0.5) / sum(keep)
  out
}

as_binary <- function(x) {
  if (is.factor(x)) x <- as.character(x)
  if (is.character(x)) {
    x_low <- tolower(trimws(x))
    return(as.integer(x_low %in% c("1", "yes", "true", "ad", "converter", "positive", "high")))
  }
  as.integer(x)
}

confidence_score <- function(x) {
  x <- tolower(trimws(as.character(x)))
  dplyr::case_when(
    x %in% c("high", "h", "3") ~ 3,
    x %in% c("medium", "moderate", "m", "2") ~ 2,
    x %in% c("low", "l", "1") ~ 1,
    TRUE ~ NA_real_
  )
}

first_nonmissing_chr <- function(x) {
  x <- as.character(x)
  x <- x[!is.na(x) & x != ""]
  if (length(x) == 0) NA_character_ else x[[1]]
}

safe_auc_obj <- function(truth, prob) {
  keep <- is.finite(prob) & !is.na(truth)
  truth <- as_binary(truth[keep])
  prob <- as.numeric(prob[keep])
  if (length(unique(truth)) < 2 || length(prob) < 3) return(NULL)
  pROC::roc(response = truth, predictor = prob, levels = c(0, 1),
            direction = "<", quiet = TRUE)
}

safe_auc <- function(truth, prob) {
  roc_obj <- safe_auc_obj(truth, prob)
  if (is.null(roc_obj)) return(NA_real_)
  as.numeric(pROC::auc(roc_obj))
}

safe_auc_ci <- function(truth, prob) {
  roc_obj <- safe_auc_obj(truth, prob)
  if (is.null(roc_obj)) return(c(NA_real_, NA_real_))
  ci <- tryCatch(as.numeric(pROC::ci.auc(roc_obj)), error = function(e) rep(NA_real_, 3))
  c(ci[1], ci[3])
}

safe_youden_threshold <- function(truth, prob) {
  roc_obj <- safe_auc_obj(truth, prob)
  if (is.null(roc_obj)) return(NA_real_)
  coords <- tryCatch(
    pROC::coords(roc_obj, "best", best.method = "youden",
                 ret = "threshold", transpose = FALSE),
    error = function(e) NA_real_
  )
  as.numeric(coords[1])
}

calc_metrics <- function(truth, prob, threshold = 0.5, label = "model") {
  truth <- as_binary(truth)
  prob <- as.numeric(prob)
  keep <- !is.na(truth) & is.finite(prob)
  truth <- truth[keep]
  prob <- prob[keep]
  pred <- as.integer(prob >= threshold)

  tp <- sum(pred == 1 & truth == 1)
  tn <- sum(pred == 0 & truth == 0)
  fp <- sum(pred == 1 & truth == 0)
  fn <- sum(pred == 0 & truth == 1)
  n <- length(truth)

  sens <- ifelse((tp + fn) > 0, tp / (tp + fn), NA_real_)
  spec <- ifelse((tn + fp) > 0, tn / (tn + fp), NA_real_)
  ppv <- ifelse((tp + fp) > 0, tp / (tp + fp), NA_real_)
  npv <- ifelse((tn + fn) > 0, tn / (tn + fn), NA_real_)
  acc <- ifelse(n > 0, (tp + tn) / n, NA_real_)
  f1 <- ifelse((2 * tp + fp + fn) > 0, 2 * tp / (2 * tp + fp + fn), NA_real_)
  brier <- ifelse(n > 0, mean((prob - truth)^2), NA_real_)
  auc_ci <- safe_auc_ci(truth, prob)

  tibble(
    model = label,
    threshold = threshold,
    n = n,
    events = sum(truth == 1),
    event_rate = mean(truth == 1),
    auc = safe_auc(truth, prob),
    auc_ci_low = auc_ci[1],
    auc_ci_high = auc_ci[2],
    sensitivity = sens,
    specificity = spec,
    ppv = ppv,
    npv = npv,
    accuracy = acc,
    f1 = f1,
    brier = brier,
    tp = tp,
    tn = tn,
    fp = fp,
    fn = fn
  )
}

calc_metrics_from_pred <- function(truth, prob, pred, label = "model", threshold = NA_real_) {
  truth <- as_binary(truth)
  prob <- as.numeric(prob)
  pred <- as_binary(pred)
  keep <- !is.na(truth) & is.finite(prob) & !is.na(pred)
  truth <- truth[keep]
  prob <- prob[keep]
  pred <- pred[keep]

  tp <- sum(pred == 1 & truth == 1)
  tn <- sum(pred == 0 & truth == 0)
  fp <- sum(pred == 1 & truth == 0)
  fn <- sum(pred == 0 & truth == 1)
  n <- length(truth)

  sens <- ifelse((tp + fn) > 0, tp / (tp + fn), NA_real_)
  spec <- ifelse((tn + fp) > 0, tn / (tn + fp), NA_real_)
  ppv <- ifelse((tp + fp) > 0, tp / (tp + fp), NA_real_)
  npv <- ifelse((tn + fn) > 0, tn / (tn + fn), NA_real_)
  acc <- ifelse(n > 0, (tp + tn) / n, NA_real_)
  f1 <- ifelse((2 * tp + fp + fn) > 0, 2 * tp / (2 * tp + fp + fn), NA_real_)
  brier <- ifelse(n > 0 && all(prob >= 0 & prob <= 1, na.rm = TRUE),
                  mean((prob - truth)^2), NA_real_)
  auc_ci <- safe_auc_ci(truth, prob)

  tibble(
    model = label,
    threshold = threshold,
    n = n,
    events = sum(truth == 1),
    event_rate = mean(truth == 1),
    auc = safe_auc(truth, prob),
    auc_ci_low = auc_ci[1],
    auc_ci_high = auc_ci[2],
    sensitivity = sens,
    specificity = spec,
    ppv = ppv,
    npv = npv,
    accuracy = acc,
    f1 = f1,
    brier = brier,
    tp = tp,
    tn = tn,
    fp = fp,
    fn = fn
  )
}

delong_pair <- function(dat, prob_a, prob_b, label_a, label_b, truth_col = "outcome") {
  roc_a <- safe_auc_obj(dat[[truth_col]], dat[[prob_a]])
  roc_b <- safe_auc_obj(dat[[truth_col]], dat[[prob_b]])
  if (is.null(roc_a) || is.null(roc_b)) {
    return(tibble(
      model_a = label_a, model_b = label_b,
      auc_a = NA_real_, auc_b = NA_real_, delta_auc = NA_real_, p_value = NA_real_
    ))
  }
  test <- tryCatch(
    pROC::roc.test(roc_a, roc_b, paired = TRUE, method = "delong"),
    error = function(e) NULL
  )
  tibble(
    model_a = label_a,
    model_b = label_b,
    auc_a = as.numeric(pROC::auc(roc_a)),
    auc_b = as.numeric(pROC::auc(roc_b)),
    delta_auc = as.numeric(pROC::auc(roc_a)) - as.numeric(pROC::auc(roc_b)),
    p_value = ifelse(is.null(test), NA_real_, as.numeric(test$p.value))
  )
}

make_stratified_folds <- function(y, k = 5) {
  y <- as_binary(y)
  folds <- vector("list", k)
  for (cls in sort(unique(y))) {
    idx <- sample(which(y == cls))
    split_idx <- split(idx, rep(seq_len(k), length.out = length(idx)))
    for (i in seq_len(k)) {
      folds[[i]] <- c(folds[[i]], split_idx[[as.character(i)]])
    }
  }
  lapply(folds, sort)
}

fit_predict_glm <- function(train, test, formula_obj) {
  fit <- tryCatch(
    glm(formula_obj, data = train, family = binomial()),
    warning = function(w) suppressWarnings(glm(formula_obj, data = train, family = binomial())),
    error = function(e) NULL
  )
  if (is.null(fit)) return(rep(NA_real_, nrow(test)))
  pred <- tryCatch(
    as.numeric(predict(fit, newdata = test, type = "response")),
    error = function(e) rep(NA_real_, nrow(test))
  )
  clip_prob(pred)
}

combined_repeated_cv <- function(dat, formula_obj, repeats = 200, k = 5, seed = 20260614) {
  set.seed(seed)
  pred_sum <- rep(0, nrow(dat))
  pred_n <- rep(0, nrow(dat))

  for (r in seq_len(repeats)) {
    folds <- make_stratified_folds(dat$outcome, k = k)
    for (fold_id in seq_len(k)) {
      test_idx <- folds[[fold_id]]
      train_idx <- setdiff(seq_len(nrow(dat)), test_idx)
      pred <- fit_predict_glm(dat[train_idx, , drop = FALSE],
                              dat[test_idx, , drop = FALSE],
                              formula_obj)
      ok <- is.finite(pred)
      pred_sum[test_idx[ok]] <- pred_sum[test_idx[ok]] + pred[ok]
      pred_n[test_idx[ok]] <- pred_n[test_idx[ok]] + 1
    }
  }

  out <- pred_sum / pred_n
  out[!is.finite(out)] <- NA_real_
  out
}

continuous_nri_idi <- function(old_prob, new_prob, y) {
  y <- as_binary(y)
  old_prob <- as.numeric(old_prob)
  new_prob <- as.numeric(new_prob)
  keep <- !is.na(y) & is.finite(old_prob) & is.finite(new_prob)
  y <- y[keep]
  old_prob <- old_prob[keep]
  new_prob <- new_prob[keep]

  event <- y == 1
  nonevent <- y == 0
  event_up <- mean(new_prob[event] > old_prob[event])
  event_down <- mean(new_prob[event] < old_prob[event])
  nonevent_down <- mean(new_prob[nonevent] < old_prob[nonevent])
  nonevent_up <- mean(new_prob[nonevent] > old_prob[nonevent])

  nri_event <- event_up - event_down
  nri_nonevent <- nonevent_down - nonevent_up
  nri <- nri_event + nri_nonevent

  old_slope <- mean(old_prob[event]) - mean(old_prob[nonevent])
  new_slope <- mean(new_prob[event]) - mean(new_prob[nonevent])
  idi <- new_slope - old_slope

  tibble(
    nri = nri,
    nri_event = nri_event,
    nri_nonevent = nri_nonevent,
    idi = idi,
    old_discrimination_slope = old_slope,
    new_discrimination_slope = new_slope
  )
}

bootstrap_nri_idi <- function(dat, old_col, new_col, label, reps = 2000) {
  point <- continuous_nri_idi(dat[[old_col]], dat[[new_col]], dat$outcome)
  boot <- replicate(reps, {
    idx <- sample(seq_len(nrow(dat)), replace = TRUE)
    tmp <- continuous_nri_idi(dat[[old_col]][idx], dat[[new_col]][idx], dat$outcome[idx])
    as.numeric(tmp[1, c("nri", "nri_event", "nri_nonevent", "idi")])
  })
  boot <- t(boot)
  colnames(boot) <- c("nri", "nri_event", "nri_nonevent", "idi")
  ci <- apply(boot, 2, quantile, probs = c(0.025, 0.975), na.rm = TRUE)
  tibble(
    comparison = label,
    nri = point$nri,
    nri_ci_low = ci[1, "nri"],
    nri_ci_high = ci[2, "nri"],
    nri_event = point$nri_event,
    nri_event_ci_low = ci[1, "nri_event"],
    nri_event_ci_high = ci[2, "nri_event"],
    nri_nonevent = point$nri_nonevent,
    nri_nonevent_ci_low = ci[1, "nri_nonevent"],
    nri_nonevent_ci_high = ci[2, "nri_nonevent"],
    idi = point$idi,
    idi_ci_low = ci[1, "idi"],
    idi_ci_high = ci[2, "idi"],
    old_discrimination_slope = point$old_discrimination_slope,
    new_discrimination_slope = point$new_discrimination_slope
  )
}

categorical_nri <- function(old_pred, new_pred, y) {
  y <- as_binary(y)
  old_pred <- as_binary(old_pred)
  new_pred <- as_binary(new_pred)
  keep <- !is.na(y) & !is.na(old_pred) & !is.na(new_pred)
  y <- y[keep]
  old_pred <- old_pred[keep]
  new_pred <- new_pred[keep]

  event <- y == 1
  nonevent <- y == 0

  event_up_n <- sum(event & old_pred == 0 & new_pred == 1)
  event_down_n <- sum(event & old_pred == 1 & new_pred == 0)
  nonevent_down_n <- sum(nonevent & old_pred == 1 & new_pred == 0)
  nonevent_up_n <- sum(nonevent & old_pred == 0 & new_pred == 1)

  n_event <- sum(event)
  n_nonevent <- sum(nonevent)
  event_nri <- ifelse(n_event > 0, (event_up_n - event_down_n) / n_event, NA_real_)
  nonevent_nri <- ifelse(n_nonevent > 0, (nonevent_down_n - nonevent_up_n) / n_nonevent, NA_real_)

  tibble(
    n = length(y),
    n_event = n_event,
    n_nonevent = n_nonevent,
    event_up_n = event_up_n,
    event_down_n = event_down_n,
    nonevent_down_n = nonevent_down_n,
    nonevent_up_n = nonevent_up_n,
    event_nri = event_nri,
    nonevent_nri = nonevent_nri,
    categorical_nri = event_nri + nonevent_nri
  )
}

bootstrap_categorical_nri <- function(dat, old_pred_col, new_pred_col, label, reps = 2000) {
  point <- categorical_nri(dat[[old_pred_col]], dat[[new_pred_col]], dat$outcome)
  boot <- replicate(reps, {
    idx <- sample(seq_len(nrow(dat)), replace = TRUE)
    tmp <- categorical_nri(dat[[old_pred_col]][idx], dat[[new_pred_col]][idx], dat$outcome[idx])
    as.numeric(tmp[1, c("event_nri", "nonevent_nri", "categorical_nri")])
  })
  boot <- t(boot)
  colnames(boot) <- c("event_nri", "nonevent_nri", "categorical_nri")
  ci <- apply(boot, 2, quantile, probs = c(0.025, 0.975), na.rm = TRUE)
  p_two_sided <- function(x) {
    x <- x[is.finite(x)]
    if (length(x) == 0) return(NA_real_)
    min(1, 2 * min(mean(x <= 0), mean(x >= 0)))
  }

  point %>%
    mutate(
      comparison = label,
      event_nri_ci_low = ci[1, "event_nri"],
      event_nri_ci_high = ci[2, "event_nri"],
      nonevent_nri_ci_low = ci[1, "nonevent_nri"],
      nonevent_nri_ci_high = ci[2, "nonevent_nri"],
      categorical_nri_ci_low = ci[1, "categorical_nri"],
      categorical_nri_ci_high = ci[2, "categorical_nri"],
      categorical_nri_p_boot = p_two_sided(boot[, "categorical_nri"])
    ) %>%
    select(comparison, everything())
}

icc_two_way_absolute <- function(wide_numeric) {
  mat <- as.matrix(wide_numeric)
  storage.mode(mat) <- "numeric"
  mat <- mat[complete.cases(mat), , drop = FALSE]
  n <- nrow(mat)
  k <- ncol(mat)
  if (n < 2 || k < 2) {
    return(tibble(
      n_cases_complete = n, n_raters = k,
      icc_absolute_single = NA_real_, icc_absolute_average = NA_real_,
      ms_case = NA_real_, ms_rater = NA_real_, ms_error = NA_real_
    ))
  }

  grand <- mean(mat)
  row_means <- rowMeans(mat)
  col_means <- colMeans(mat)
  ss_case <- k * sum((row_means - grand)^2)
  ss_rater <- n * sum((col_means - grand)^2)
  residual <- sweep(sweep(mat, 1, row_means, "-"), 2, col_means, "-") + grand
  ss_error <- sum(residual^2)

  ms_case <- ss_case / (n - 1)
  ms_rater <- ss_rater / (k - 1)
  ms_error <- ss_error / ((n - 1) * (k - 1))

  icc_single <- (ms_case - ms_error) /
    (ms_case + (k - 1) * ms_error + k * (ms_rater - ms_error) / n)
  icc_average <- (ms_case - ms_error) /
    (ms_case + (ms_rater - ms_error) / n)

  tibble(
    n_cases_complete = n,
    n_raters = k,
    icc_absolute_single = icc_single,
    icc_absolute_average = icc_average,
    ms_case = ms_case,
    ms_rater = ms_rater,
    ms_error = ms_error
  )
}

net_benefit_curve <- function(dat, model_col, label, thresholds = seq(0.05, 0.80, by = 0.01)) {
  y <- as_binary(dat$outcome)
  p <- as.numeric(dat[[model_col]])
  keep <- !is.na(y) & is.finite(p)
  y <- y[keep]
  p <- p[keep]
  n <- length(y)

  map_dfr(thresholds, function(pt) {
    pred <- as.integer(p >= pt)
    tp <- sum(pred == 1 & y == 1)
    fp <- sum(pred == 1 & y == 0)
    nb <- (tp / n) - (fp / n) * (pt / (1 - pt))
    tibble(model = label, threshold = pt, net_benefit = nb)
  })
}

net_benefit_all_none <- function(dat, thresholds = seq(0.05, 0.80, by = 0.01)) {
  y <- as_binary(dat$outcome)
  prevalence <- mean(y == 1)
  bind_rows(
    tibble(model = "Treat none", threshold = thresholds, net_benefit = 0),
    tibble(
      model = "Treat all",
      threshold = thresholds,
      net_benefit = prevalence - (1 - prevalence) * thresholds / (1 - thresholds)
    )
  )
}

standardized_mean_by_group <- function(dat, group_col, vars) {
  vars <- vars[vars %in% names(dat)]
  zdat <- dat
  for (v in vars) {
    x <- as.numeric(zdat[[v]])
    zdat[[v]] <- as.numeric(scale(x))
  }
  zdat %>%
    select(all_of(group_col), all_of(vars)) %>%
    pivot_longer(cols = all_of(vars), names_to = "feature", values_to = "z_value") %>%
    group_by(.data[[group_col]], feature) %>%
    summarise(mean_z = mean(z_value, na.rm = TRUE), n = sum(!is.na(z_value)), .groups = "drop") %>%
    rename(group = all_of(group_col))
}

compare_feature_by_group <- function(dat, var, group_col = "discordance_group") {
  x <- dat[[var]]
  g <- factor(dat[[group_col]])
  keep <- !is.na(g) & !is.na(x)
  x <- x[keep]
  g <- droplevels(g[keep])
  if (length(unique(g)) < 2 || length(x) < 5) {
    return(tibble(feature = var, type = NA_character_, p_value = NA_real_, summary = NA_character_))
  }

  if (is.numeric(x) || is.integer(x)) {
    p <- tryCatch(kruskal.test(x ~ g)$p.value, error = function(e) NA_real_)
    summary_tbl <- tibble(value = x, group = g) %>%
      group_by(group) %>%
      summarise(
        n = sum(!is.na(value)),
        median = median(value, na.rm = TRUE),
        q1 = quantile(value, 0.25, na.rm = TRUE),
        q3 = quantile(value, 0.75, na.rm = TRUE),
        .groups = "drop"
      ) %>%
      mutate(txt = paste0(group, ": ", n, ", ", signif(median, 4),
                          " [", signif(q1, 4), ", ", signif(q3, 4), "]")) %>%
      pull(txt) %>%
      paste(collapse = "; ")
    tibble(feature = var, type = "continuous_kruskal", p_value = p, summary = summary_tbl)
  } else {
    tab <- table(g, x)
    p <- tryCatch({
      if (any(tab < 5)) fisher.test(tab)$p.value else chisq.test(tab)$p.value
    }, error = function(e) NA_real_)
    summary_tbl <- as.data.frame(tab) %>%
      group_by(g) %>%
      mutate(prop = Freq / sum(Freq)) %>%
      ungroup() %>%
      mutate(txt = paste0(g, "/", x, ": ", Freq, " (", round(100 * prop, 1), "%)")) %>%
      pull(txt) %>%
      paste(collapse = "; ")
    tibble(feature = var, type = "categorical_fisher_or_chisq", p_value = p, summary = summary_tbl)
  }
}

save_plot_both <- function(plot_obj, filename, width = 8, height = 5) {
  ggsave(file.path(fig_dir, paste0(filename, ".pdf")), plot_obj, width = width, height = height)
  ggsave(file.path(fig_dir, paste0(filename, ".png")), plot_obj, width = width, height = height, dpi = 300)
}

# -------------------------------------------------------------------------
# 2. Read and harmonize core data
# -------------------------------------------------------------------------

test_set <- read_csv_safe(file.path(base_dir, opt$test_file))
ai_pred <- read_csv_safe(file.path(base_dir, opt$ai_file))
expert_long <- read_csv_safe(file.path(base_dir, opt$expert_file))

required_test_cols <- c("ID", "AD_Conversion")
required_ai_cols <- c("ID", "Actual", "Predicted_Prob", "Predicted_Class")
required_exp_cols <- c("CaseID", "Expert", "Stage1_Prob", "Stage2_Prob")

missing_test <- setdiff(required_test_cols, names(test_set))
missing_ai <- setdiff(required_ai_cols, names(ai_pred))
missing_exp <- setdiff(required_exp_cols, names(expert_long))

if (length(missing_test) > 0) stop("Missing columns in independent_test_set.csv: ", paste(missing_test, collapse = ", "))
if (length(missing_ai) > 0) stop("Missing columns in AI prediction file: ", paste(missing_ai, collapse = ", "))
if (length(missing_exp) > 0) stop("Missing columns in Expert_Predictions_Long.csv: ", paste(missing_exp, collapse = ", "))

expert_case <- expert_long %>%
  mutate(
    Stage1_Prob = as.numeric(Stage1_Prob),
    Stage2_Prob = as.numeric(Stage2_Prob),
    Stage1_Confidence_Score = if ("Stage1_Confidence" %in% names(.)) confidence_score(Stage1_Confidence) else NA_real_,
    Stage2_Confidence_Score = if ("Stage2_Confidence" %in% names(.)) confidence_score(Stage2_Confidence) else NA_real_
  ) %>%
  group_by(CaseID) %>%
  summarise(
    expert_n = n_distinct(Expert),
    expert_stage1_prob = mean(Stage1_Prob, na.rm = TRUE),
    expert_stage1_prob_sd = sd(Stage1_Prob, na.rm = TRUE),
    expert_stage2_prob = mean(Stage2_Prob, na.rm = TRUE),
    expert_stage2_prob_sd = sd(Stage2_Prob, na.rm = TRUE),
    expert_mri_delta = expert_stage2_prob - expert_stage1_prob,
    expert_stage1_vote_high = mean(Stage1_Prob >= expert_threshold_primary, na.rm = TRUE),
    expert_stage2_vote_high = mean(Stage2_Prob >= expert_threshold_primary, na.rm = TRUE),
    expert_stage1_vote_pred = as.integer(expert_stage1_vote_high >= 0.5),
    expert_stage2_vote_pred = as.integer(expert_stage2_vote_high >= 0.5),
    expert_stage1_mean_confidence = mean(Stage1_Confidence_Score, na.rm = TRUE),
    expert_stage2_mean_confidence = mean(Stage2_Confidence_Score, na.rm = TRUE),
    .groups = "drop"
  )

expert_stage2_wide <- expert_long %>%
  mutate(Stage2_Prob = as.numeric(Stage2_Prob)) %>%
  select(CaseID, Expert, Stage2_Prob) %>%
  tidyr::pivot_wider(names_from = Expert, values_from = Stage2_Prob, names_prefix = "expert_stage2_")

expert_stage2_rating_cols <- setdiff(names(expert_stage2_wide), "CaseID")

expert_agreement_case <- expert_stage2_wide %>%
  rowwise() %>%
  mutate(
    expert_stage2_min = min(c_across(all_of(expert_stage2_rating_cols)), na.rm = TRUE),
    expert_stage2_max = max(c_across(all_of(expert_stage2_rating_cols)), na.rm = TRUE),
    expert_stage2_range = expert_stage2_max - expert_stage2_min,
    expert_stage2_unanimous_high = all(c_across(all_of(expert_stage2_rating_cols)) >= expert_threshold_primary, na.rm = TRUE),
    expert_stage2_unanimous_low = all(c_across(all_of(expert_stage2_rating_cols)) < expert_threshold_primary, na.rm = TRUE),
    expert_stage2_majority_margin = abs(mean(c_across(all_of(expert_stage2_rating_cols)) >= expert_threshold_primary, na.rm = TRUE) - 0.5) * 2,
    expert_stage2_collective_confident = expert_stage2_range <= 0.20 | expert_stage2_unanimous_high | expert_stage2_unanimous_low
  ) %>%
  ungroup() %>%
  select(CaseID, expert_stage2_min, expert_stage2_max, expert_stage2_range,
         expert_stage2_unanimous_high, expert_stage2_unanimous_low,
         expert_stage2_majority_margin, expert_stage2_collective_confident)

expert_stage2_icc <- icc_two_way_absolute(expert_stage2_wide %>% select(all_of(expert_stage2_rating_cols))) %>%
  mutate(rating = "Expert Stage 2 probability")

write_csv_safe(expert_stage2_icc, file.path(out_dir, "00_expert_stage2_interrater_icc.csv"))

clinical_candidates <- c(
  "ID", "RID", "Age", "Gender", "Education", "MMSE_Baseline",
  "ADAS13", "CDRSB", "FAQTOTAL", "APOE4_Positive", "APOE4_Copies",
  "ABETA42", "ABETA40", "ABETA42_ABETA40_RATIO", "TAU_TOTAL", "PTAU181",
  "completeness", "Baseline_Date", "STATUS", "AD_Conversion"
)

mri_candidates <- names(test_set)[str_detect(names(test_set), "^ST")]
keep_cols <- intersect(c(clinical_candidates, mri_candidates), names(test_set))

master <- test_set %>%
  select(all_of(keep_cols)) %>%
  mutate(outcome = as_binary(AD_Conversion)) %>%
  left_join(
    ai_pred %>%
      transmute(
        ID,
        ai_actual = as_binary(Actual),
        ai_prob = as.numeric(Predicted_Prob),
        ai_original_class = as_binary(Predicted_Class)
      ),
    by = "ID"
  ) %>%
  left_join(expert_case, by = c("ID" = "CaseID")) %>%
  left_join(expert_agreement_case, by = c("ID" = "CaseID")) %>%
  mutate(
    ai_pred = as.integer(ai_prob >= ai_threshold_primary),
    expert_stage1_pred = as.integer(expert_stage1_prob >= expert_threshold_primary),
    expert_stage2_pred = as.integer(expert_stage2_prob >= expert_threshold_primary),
    ai_correct = ai_pred == outcome,
    expert_stage1_correct = expert_stage1_pred == outcome,
    expert_stage2_correct = expert_stage2_pred == outcome,
    discordance_group = case_when(
      ai_correct & expert_stage2_correct ~ "Both correct",
      ai_correct & !expert_stage2_correct ~ "AI correct / Expert wrong",
      !ai_correct & expert_stage2_correct ~ "Expert correct / AI wrong",
      TRUE ~ "Both wrong"
    ),
    discordance_group = factor(
      discordance_group,
      levels = c(
        "Both correct",
        "AI correct / Expert wrong",
        "Expert correct / AI wrong",
        "Both wrong"
      )
    ),
    ai_minus_expert2_prob = ai_prob - expert_stage2_prob,
    expert_uncertain_40_60 = expert_stage2_prob >= 0.40 & expert_stage2_prob <= 0.60,
    expert_uncertain_35_65 = expert_stage2_prob >= 0.35 & expert_stage2_prob <= 0.65,
    expert_uncertain_30_70 = expert_stage2_prob >= 0.30 & expert_stage2_prob <= 0.70
  )

if (any(master$outcome != master$ai_actual, na.rm = TRUE)) {
  warning("AD_Conversion and AI Actual differ for some cases. Please check case IDs.")
}

gray_zone_check <- tibble(
  gray_zone = c("40-60 primary prespecified", "35-65 sensitivity", "30-70 sensitivity"),
  lower = c(0.40, 0.35, 0.30),
  upper = c(0.60, 0.65, 0.70)
) %>%
  rowwise() %>%
  mutate(
    n_cases = sum(master$expert_stage2_prob >= lower & master$expert_stage2_prob <= upper, na.rm = TRUE),
    percent_cases = 100 * n_cases / nrow(master),
    events = sum(master$outcome == 1 & master$expert_stage2_prob >= lower & master$expert_stage2_prob <= upper, na.rm = TRUE),
    nonevents = sum(master$outcome == 0 & master$expert_stage2_prob >= lower & master$expert_stage2_prob <= upper, na.rm = TRUE),
    note = ifelse(
      gray_zone == "40-60 primary prespecified",
      "Primary gray zone; do not change after seeing outcomes.",
      "Prespecified sensitivity gray zone; not the primary claim."
    )
  ) %>%
  ungroup()

write_csv_safe(gray_zone_check, file.path(out_dir, "00_gray_zone_distribution_check.csv"))

gray_zone_histogram <- master %>%
  ggplot(aes(x = expert_stage2_prob)) +
  geom_histogram(binwidth = 0.05, boundary = 0, fill = "#4C78A8", color = "white") +
  geom_vline(xintercept = c(0.40, 0.60), color = "#D62728", linetype = "solid", size = 0.8) +
  geom_vline(xintercept = c(0.35, 0.65), color = "#FF7F0E", linetype = "dashed", size = 0.65) +
  geom_vline(xintercept = c(0.30, 0.70), color = "#2CA02C", linetype = "dotted", size = 0.65) +
  scale_x_continuous(limits = c(0, 1), labels = scales::percent_format(accuracy = 1)) +
  labs(x = "Mean expert Stage 2 probability", y = "Number of cases",
       title = "Prespecified expert uncertainty zones") +
  theme_minimal(base_size = 11)

save_plot_both(gray_zone_histogram, "Figure_Q1_Gray_Zone_Distribution_Check", width = 7.2, height = 4.8)

write_csv_safe(master, file.path(out_dir, "00_case_level_master.csv"))

# -------------------------------------------------------------------------
# 3. Human-AI integration
#    Primary leakage-safe combinations use no fitted weights:
#      - simple average of AI and expert probabilities
#      - rank average of AI and expert probabilities
#      - prespecified workflow rules
#    Logistic stacking is retained only as post-hoc exploratory analysis,
#    because fitting weights on the 196-case holdout cohort would otherwise
#    turn the independent benchmark into a training set.
# -------------------------------------------------------------------------

model_dat <- master %>%
  transmute(
    ID,
    outcome,
    ai_prob = clip_prob(ai_prob),
    expert_stage1_prob = clip_prob(expert_stage1_prob),
    expert_stage2_prob = clip_prob(expert_stage2_prob),
    ai_logit = logit_clip(ai_prob),
    expert1_logit = logit_clip(expert_stage1_prob),
    expert2_logit = logit_clip(expert_stage2_prob)
  ) %>%
  filter(!is.na(outcome), is.finite(ai_logit), is.finite(expert1_logit), is.finite(expert2_logit))

combined_stage2_cv <- combined_repeated_cv(
  model_dat,
  outcome ~ ai_logit + expert2_logit,
  repeats = cv_repeats,
  k = cv_folds
)

combined_stage1_cv <- combined_repeated_cv(
  model_dat,
  outcome ~ ai_logit + expert1_logit,
  repeats = cv_repeats,
  k = cv_folds
)

apparent_stage2_fit <- glm(outcome ~ ai_logit + expert2_logit, data = model_dat, family = binomial())
apparent_stage1_fit <- glm(outcome ~ ai_logit + expert1_logit, data = model_dat, family = binomial())

combined_predictions <- model_dat %>%
  mutate(
    combined_stage2_cv_prob = clip_prob(combined_stage2_cv),
    combined_stage1_cv_prob = clip_prob(combined_stage1_cv),
    combined_stage2_apparent_prob = clip_prob(predict(apparent_stage2_fit, newdata = model_dat, type = "response")),
    combined_stage1_apparent_prob = clip_prob(predict(apparent_stage1_fit, newdata = model_dat, type = "response"))
  ) %>%
  select(ID, combined_stage2_cv_prob, combined_stage1_cv_prob,
         combined_stage2_apparent_prob, combined_stage1_apparent_prob)

master <- master %>%
  left_join(combined_predictions, by = "ID") %>%
  mutate(
    combined_stage2_cv_pred = as.integer(combined_stage2_cv_prob >= combined_threshold_primary),
    combined_stage1_cv_pred = as.integer(combined_stage1_cv_prob >= combined_threshold_primary),
    mean_ai_expert2_prob = (ai_prob + expert_stage2_prob) / 2,
    mean_ai_expert2_pred = as.integer(mean_ai_expert2_prob >= 0.50),
    rankmean_ai_expert2_prob = (rank01(ai_prob) + rank01(expert_stage2_prob)) / 2,
    rankmean_ai_expert2_pred = as.integer(rankmean_ai_expert2_prob >= 0.50),
    expert_uncertain_ai_specificity_prune_prob = ifelse(
      expert_uncertain_40_60 & ai_pred == 0,
      ai_prob,
      expert_stage2_prob
    ),
    expert_uncertain_ai_specificity_prune_pred = ifelse(
      expert_uncertain_40_60 & ai_pred == 0,
      0L,
      expert_stage2_pred
    ),
    second_reader_40_60_prob = ifelse(expert_uncertain_40_60, ai_prob, expert_stage2_prob),
    second_reader_40_60_pred = ifelse(expert_uncertain_40_60, ai_pred, expert_stage2_pred),
    second_reader_35_65_prob = ifelse(expert_uncertain_35_65, ai_prob, expert_stage2_prob),
    second_reader_35_65_pred = ifelse(expert_uncertain_35_65, ai_pred, expert_stage2_pred),
    second_reader_30_70_prob = ifelse(expert_uncertain_30_70, ai_prob, expert_stage2_prob),
    second_reader_30_70_pred = ifelse(expert_uncertain_30_70, ai_pred, expert_stage2_pred),
    ai_confirmed_high_specificity_prob = pmin(ai_prob, expert_stage2_prob),
    ai_or_expert_high_sensitivity_prob = pmax(ai_prob, expert_stage2_prob),
    ai_confirmed_high_specificity_pred = as.integer(ai_pred == 1 & expert_stage2_pred == 1),
    ai_or_expert_high_sensitivity_pred = as.integer(ai_pred == 1 | expert_stage2_pred == 1),
    ai_correct_expert_collective_blindspot = ai_correct & !expert_stage2_correct & expert_stage2_collective_confident,
    ai_correct_expert_split_decision = ai_correct & !expert_stage2_correct & !expert_stage2_collective_confident
  )

write_csv_safe(master, file.path(out_dir, "00_case_level_master_with_combined_predictions.csv"))

coef_summary <- bind_rows(
  broom::tidy(apparent_stage2_fit, conf.int = TRUE) %>% mutate(model = "AI + expert Stage 2 apparent"),
  broom::tidy(apparent_stage1_fit, conf.int = TRUE) %>% mutate(model = "AI + expert Stage 1 apparent")
) %>%
  mutate(
    odds_ratio = exp(estimate),
    odds_ratio_ci_low = exp(conf.low),
    odds_ratio_ci_high = exp(conf.high)
  ) %>%
  select(model, term, estimate, std.error, statistic, p.value,
         odds_ratio, odds_ratio_ci_low, odds_ratio_ci_high)
write_csv_safe(coef_summary, file.path(out_dir, "00_combined_model_coefficients_apparent.csv"))

# -------------------------------------------------------------------------
# 4. Performance summary and pairwise AUC tests
# -------------------------------------------------------------------------

youden_thresholds <- tibble(
  model = c("AI", "Expert Stage 1", "Expert Stage 2",
            "Simple mean AI + Expert Stage 2", "Rank mean AI + Expert Stage 2",
            "Exploratory CV AI + Expert Stage 1", "Exploratory CV AI + Expert Stage 2"),
  prob_col = c("ai_prob", "expert_stage1_prob", "expert_stage2_prob",
               "mean_ai_expert2_prob", "rankmean_ai_expert2_prob",
               "combined_stage1_cv_prob", "combined_stage2_cv_prob")
) %>%
  rowwise() %>%
  mutate(youden_threshold = safe_youden_threshold(master$outcome, master[[prob_col]])) %>%
  ungroup()

write_csv_safe(youden_thresholds, file.path(out_dir, "01_youden_thresholds_on_current_test_set_sensitivity_only.csv"))

performance_primary <- bind_rows(
  calc_metrics_from_pred(master$outcome, master$ai_prob, master$ai_pred,
                         "AI alone, frozen threshold", ai_threshold_primary),
  calc_metrics_from_pred(master$outcome, master$expert_stage1_prob, master$expert_stage1_pred,
                         "Expert Stage 1 mean probability", expert_threshold_primary),
  calc_metrics_from_pred(master$outcome, master$expert_stage2_prob, master$expert_stage2_pred,
                         "Expert Stage 2 mean probability", expert_threshold_primary),
  calc_metrics_from_pred(master$outcome, master$mean_ai_expert2_prob, master$mean_ai_expert2_pred,
                         "Leakage-safe simple mean: AI + Expert Stage 2", 0.50),
  calc_metrics_from_pred(master$outcome, master$rankmean_ai_expert2_prob, master$rankmean_ai_expert2_pred,
                         "Leakage-safe rank mean: AI + Expert Stage 2", 0.50),
  calc_metrics_from_pred(master$outcome, master$second_reader_40_60_prob, master$second_reader_40_60_pred,
                         "AI second reader if expert Stage 2 0.40-0.60", NA_real_),
  calc_metrics_from_pred(master$outcome, master$second_reader_35_65_prob, master$second_reader_35_65_pred,
                         "AI second reader if expert Stage 2 0.35-0.65", NA_real_),
  calc_metrics_from_pred(master$outcome, master$expert_uncertain_ai_specificity_prune_prob,
                         master$expert_uncertain_ai_specificity_prune_pred,
                         "AI specificity-prune when expert Stage 2 0.40-0.60", NA_real_),
  calc_metrics_from_pred(master$outcome, master$ai_confirmed_high_specificity_prob, master$ai_confirmed_high_specificity_pred,
                         "AI AND Expert Stage 2 high-specificity rule", NA_real_),
  calc_metrics_from_pred(master$outcome, master$ai_or_expert_high_sensitivity_prob, master$ai_or_expert_high_sensitivity_pred,
                         "AI OR Expert Stage 2 high-sensitivity rule", NA_real_)
)

performance_exploratory_cv <- bind_rows(
  calc_metrics_from_pred(master$outcome, master$combined_stage1_cv_prob, master$combined_stage1_cv_pred,
                         "Post-hoc exploratory CV logistic: AI + Expert Stage 1", combined_threshold_primary),
  calc_metrics_from_pred(master$outcome, master$combined_stage2_cv_prob, master$combined_stage2_cv_pred,
                         "Post-hoc exploratory CV logistic: AI + Expert Stage 2", combined_threshold_primary)
) %>%
  mutate(
    caveat = paste(
      "Exploratory only: this estimates a new integration rule within the 196-case holdout cohort.",
      "Do not describe as an externally validated or frozen-model benchmark."
    )
  )

performance_youden <- youden_thresholds %>%
  mutate(metrics = map2(prob_col, youden_threshold, ~ calc_metrics(master$outcome, master[[.x]], .y,
                                                                   paste0(model, " Youden threshold")))) %>%
  select(metrics) %>%
  unnest(metrics)

write_csv_safe(performance_primary, file.path(out_dir, "01_performance_summary_primary_thresholds.csv"))
write_csv_safe(performance_exploratory_cv, file.path(out_dir, "01_performance_summary_exploratory_cv_logistic.csv"))
write_csv_safe(performance_youden, file.path(out_dir, "01_performance_summary_youden_sensitivity_only.csv"))

delong_tests_primary <- bind_rows(
  delong_pair(master, "ai_prob", "expert_stage2_prob", "AI", "Expert Stage 2"),
  delong_pair(master, "ai_prob", "expert_stage1_prob", "AI", "Expert Stage 1"),
  delong_pair(master, "mean_ai_expert2_prob", "expert_stage2_prob", "Leakage-safe mean AI + Expert Stage 2", "Expert Stage 2"),
  delong_pair(master, "rankmean_ai_expert2_prob", "expert_stage2_prob", "Leakage-safe rank mean AI + Expert Stage 2", "Expert Stage 2"),
  delong_pair(master, "second_reader_40_60_prob", "expert_stage2_prob", "AI second reader 0.40-0.60", "Expert Stage 2"),
  delong_pair(master, "expert_uncertain_ai_specificity_prune_prob", "expert_stage2_prob",
              "AI specificity-prune in expert-uncertain cases", "Expert Stage 2")
)

delong_tests_exploratory_cv <- bind_rows(
  delong_pair(master, "combined_stage2_cv_prob", "expert_stage2_prob",
              "Post-hoc exploratory CV AI + Expert Stage 2", "Expert Stage 2"),
  delong_pair(master, "combined_stage2_cv_prob", "ai_prob",
              "Post-hoc exploratory CV AI + Expert Stage 2", "AI"),
  delong_pair(master, "combined_stage1_cv_prob", "expert_stage1_prob",
              "Post-hoc exploratory CV AI + Expert Stage 1", "Expert Stage 1")
) %>%
  mutate(
    caveat = paste(
      "Exploratory only: cross-validated stacking inside the holdout cohort does not equal",
      "a frozen externally validated integration model."
    )
  )

write_csv_safe(delong_tests_primary, file.path(out_dir, "02_delong_auc_tests_primary_leakage_safe.csv"))
write_csv_safe(delong_tests_exploratory_cv, file.path(out_dir, "02_delong_auc_tests_exploratory_cv_logistic.csv"))

performance_plot <- performance_primary %>%
  select(model, auc, sensitivity, specificity, ppv, npv, brier) %>%
  pivot_longer(cols = c(auc, sensitivity, specificity, ppv, npv),
               names_to = "metric", values_to = "value") %>%
  mutate(model = factor(model, levels = rev(performance_primary$model))) %>%
  ggplot(aes(x = value, y = model, fill = metric)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.72) +
  scale_x_continuous(limits = c(0, 1), labels = scales::percent_format(accuracy = 1)) +
  labs(x = NULL, y = NULL, fill = NULL,
       title = "AI, expert, and human-AI workflow performance") +
  theme_minimal(base_size = 11) +
  theme(panel.grid.major.y = element_blank())

save_plot_both(performance_plot, "Figure_Q1_Performance_Human_AI_Workflows", width = 9.5, height = 5.5)

# -------------------------------------------------------------------------
# 5. Case-level discordance analysis
# -------------------------------------------------------------------------

case_discordance <- master %>%
  select(
    ID, RID, outcome,
    ai_prob, ai_pred, ai_correct,
    expert_stage1_prob, expert_stage1_pred, expert_stage1_correct,
    expert_stage2_prob, expert_stage2_pred, expert_stage2_correct,
    expert_stage2_prob_sd, expert_mri_delta,
    expert_stage2_min, expert_stage2_max, expert_stage2_range,
    expert_stage2_unanimous_high, expert_stage2_unanimous_low,
    expert_stage2_majority_margin, expert_stage2_collective_confident,
    combined_stage2_cv_prob, combined_stage2_cv_pred,
    discordance_group,
    ai_correct_expert_collective_blindspot,
    ai_correct_expert_split_decision,
    any_of(c("Age", "Gender", "Education", "MMSE_Baseline", "ADAS13", "CDRSB",
             "FAQTOTAL", "APOE4_Positive", "APOE4_Copies", "ABETA42", "ABETA40",
             "ABETA42_ABETA40_RATIO", "TAU_TOTAL", "PTAU181"))
  ) %>%
  arrange(discordance_group, desc(outcome), desc(abs(ai_prob - expert_stage2_prob)))

write_csv_safe(case_discordance, file.path(out_dir, "03_case_level_discordance.csv"))

discordance_counts <- master %>%
  count(discordance_group, outcome, name = "n") %>%
  group_by(discordance_group) %>%
  mutate(group_total = sum(n), within_group_percent = 100 * n / group_total) %>%
  ungroup()
write_csv_safe(discordance_counts, file.path(out_dir, "03_discordance_group_counts_by_outcome.csv"))

discordance_subclassification <- master %>%
  filter(discordance_group == "AI correct / Expert wrong") %>%
  summarise(
    n_ai_correct_expert_wrong = n(),
    n_collective_blindspot = sum(ai_correct_expert_collective_blindspot, na.rm = TRUE),
    n_split_decision = sum(ai_correct_expert_split_decision, na.rm = TRUE),
    mean_expert_stage2_range = mean(expert_stage2_range, na.rm = TRUE),
    median_expert_stage2_range = median(expert_stage2_range, na.rm = TRUE),
    mean_expert_stage2_prob_sd = mean(expert_stage2_prob_sd, na.rm = TRUE),
    median_expert_stage2_prob_sd = median(expert_stage2_prob_sd, na.rm = TRUE)
  )
write_csv_safe(discordance_subclassification, file.path(out_dir, "03_ai_correct_expert_wrong_subclassification.csv"))

feature_candidates <- c(
  "Age", "Gender", "Education", "MMSE_Baseline", "ADAS13", "CDRSB", "FAQTOTAL",
  "APOE4_Positive", "APOE4_Copies", "ABETA42", "ABETA40", "ABETA42_ABETA40_RATIO",
  "TAU_TOTAL", "PTAU181",
  "ST105CV", "ST102TS", "ST108TS", "ST109CV", "ST103TA", "ST108TA", "ST106TA",
  "ST102TA", "ST107CV", "ST111CV", "ST104CV"
)
feature_candidates <- feature_candidates[feature_candidates %in% names(master)]

discordance_feature_tests <- map_dfr(feature_candidates, ~ compare_feature_by_group(master, .x))
discordance_feature_tests <- discordance_feature_tests %>%
  mutate(p_fdr = p.adjust(p_value, method = "BH")) %>%
  arrange(p_value)
write_csv_safe(discordance_feature_tests, file.path(out_dir, "03_discordance_feature_comparison.csv"))

adjustment_vars <- c("Age", "Gender", "Education", "MMSE_Baseline", "ADAS13", "CDRSB", "APOE4_Copies")
adjustment_vars <- adjustment_vars[adjustment_vars %in% names(master)]

candidate_mechanism_vars <- c(
  "ai_minus_expert2_prob", "expert_stage2_prob_sd", "expert_mri_delta",
  "ABETA42_ABETA40_RATIO", "PTAU181", "TAU_TOTAL",
  "ST105CV", "ST102TS", "ST108TS", "ST109CV", "ST103TA", "ST108TA", "ST106TA",
  "ST102TA", "ST107CV", "ST111CV", "ST104CV"
)
candidate_mechanism_vars <- candidate_mechanism_vars[candidate_mechanism_vars %in% names(master)]

fit_adjusted_discordance <- function(dat, target_group, predictor, adjusters) {
  tmp <- dat %>%
    mutate(target = as.integer(discordance_group == target_group)) %>%
    select(target, all_of(c(predictor, adjusters))) %>%
    mutate(across(where(is.character), as.factor)) %>%
    tidyr::drop_na()
  if (nrow(tmp) < 30 || length(unique(tmp$target)) < 2) {
    return(tibble(
      target_group = target_group, predictor = predictor,
      n = nrow(tmp), odds_ratio = NA_real_, ci_low = NA_real_,
      ci_high = NA_real_, p_value = NA_real_
    ))
  }
  formula_obj <- as.formula(paste("target ~", paste(c(predictor, adjusters), collapse = " + ")))
  fit <- tryCatch(glm(formula_obj, data = tmp, family = binomial()), error = function(e) NULL)
  if (is.null(fit)) {
    return(tibble(
      target_group = target_group, predictor = predictor,
      n = nrow(tmp), odds_ratio = NA_real_, ci_low = NA_real_,
      ci_high = NA_real_, p_value = NA_real_
    ))
  }
  broom::tidy(fit, conf.int = TRUE) %>%
    filter(term == predictor) %>%
    transmute(
      target_group = target_group,
      predictor = predictor,
      n = nrow(tmp),
      odds_ratio = exp(estimate),
      ci_low = exp(conf.low),
      ci_high = exp(conf.high),
      p_value = p.value
    )
}

adjusted_discordance_models <- crossing(
  target_group = c("AI correct / Expert wrong", "Expert correct / AI wrong"),
  predictor = candidate_mechanism_vars
) %>%
  mutate(result = map2(target_group, predictor,
                       ~ fit_adjusted_discordance(master, .x, .y, adjustment_vars))) %>%
  select(result) %>%
  unnest(result) %>%
  mutate(
    p_fdr = p.adjust(p_value, method = "BH"),
    adjustment_set = paste(adjustment_vars, collapse = " + ")
  ) %>%
  arrange(target_group, p_value)

write_csv_safe(adjusted_discordance_models, file.path(out_dir, "03_adjusted_discordance_models_mmse_adjusted.csv"))

heatmap_features <- c(
  "Age", "Education", "MMSE_Baseline", "ADAS13", "CDRSB", "FAQTOTAL",
  "APOE4_Copies", "ABETA42_ABETA40_RATIO", "PTAU181",
  "ST105CV", "ST102TS", "ST108TS", "ST109CV", "ST103TA", "ST108TA", "ST106TA"
)
heatmap_features <- heatmap_features[heatmap_features %in% names(master)]

if (length(heatmap_features) >= 3) {
  heatmap_dat <- standardized_mean_by_group(master, "discordance_group", heatmap_features)
  write_csv_safe(heatmap_dat, file.path(out_dir, "03_discordance_feature_standardized_means.csv"))

  heatmap_plot <- heatmap_dat %>%
    mutate(
      group = factor(group, levels = levels(master$discordance_group)),
      feature = factor(feature, levels = rev(unique(feature)))
    ) %>%
    ggplot(aes(x = group, y = feature, fill = mean_z)) +
    geom_tile(color = "white", size = 0.3) +
    scale_fill_gradient2(low = "#2C7BB6", mid = "white", high = "#D7191C",
                         midpoint = 0, name = "Mean z") +
    labs(x = NULL, y = NULL,
         title = "Clinical and biomarker profile of AI-expert discordant cases") +
    theme_minimal(base_size = 10) +
    theme(axis.text.x = element_text(angle = 30, hjust = 1),
          panel.grid = element_blank())

  save_plot_both(heatmap_plot, "Figure_Q1_Discordance_Feature_Heatmap", width = 8.5, height = 5.5)
}

waterfall_plot <- master %>%
  mutate(
    rank_id = row_number(ai_minus_expert2_prob),
    outcome_label = ifelse(outcome == 1, "Converter", "Non-converter")
  ) %>%
  arrange(ai_minus_expert2_prob) %>%
  mutate(rank_id = row_number()) %>%
  ggplot(aes(x = rank_id, y = ai_minus_expert2_prob, fill = discordance_group)) +
  geom_col(width = 0.85) +
  geom_hline(yintercept = 0, color = "gray30", size = 0.4) +
  labs(x = "Cases ranked by AI minus expert Stage 2 probability",
       y = "AI probability - expert Stage 2 probability",
       fill = NULL,
       title = "Case-level probability discordance between AI and expert assessment") +
  theme_minimal(base_size = 11) +
  theme(panel.grid.major.x = element_blank())

save_plot_both(waterfall_plot, "Figure_Q1_Case_Level_AI_Expert_Probability_Discordance", width = 9, height = 4.8)

# -------------------------------------------------------------------------
# 6. Workflow simulation: what changes clinically?
# -------------------------------------------------------------------------

workflow_rules <- tibble(
  workflow = c(
    "Expert Stage 2 alone",
    "AI alone, frozen threshold",
    "AI second reader if expert Stage 2 0.40-0.60",
    "AI second reader if expert Stage 2 0.35-0.65",
    "AI second reader if expert Stage 2 0.30-0.70",
    "AI specificity-prune when expert Stage 2 0.40-0.60",
    "Leakage-safe simple mean: AI + Expert Stage 2",
    "Leakage-safe rank mean: AI + Expert Stage 2",
    "AI AND Expert Stage 2 high-specificity rule",
    "AI OR Expert Stage 2 high-sensitivity rule"
  ),
  prob_col = c(
    "expert_stage2_prob",
    "ai_prob",
    "second_reader_40_60_prob",
    "second_reader_35_65_prob",
    "second_reader_30_70_prob",
    "expert_uncertain_ai_specificity_prune_prob",
    "mean_ai_expert2_prob",
    "rankmean_ai_expert2_prob",
    "ai_confirmed_high_specificity_prob",
    "ai_or_expert_high_sensitivity_prob"
  ),
  pred_col = c(
    "expert_stage2_pred",
    "ai_pred",
    "second_reader_40_60_pred",
    "second_reader_35_65_pred",
    "second_reader_30_70_pred",
    "expert_uncertain_ai_specificity_prune_pred",
    "mean_ai_expert2_pred",
    "rankmean_ai_expert2_pred",
    "ai_confirmed_high_specificity_pred",
    "ai_or_expert_high_sensitivity_pred"
  ),
  ai_review_count = c(
    0L,
    nrow(master),
    sum(master$expert_uncertain_40_60, na.rm = TRUE),
    sum(master$expert_uncertain_35_65, na.rm = TRUE),
    sum(master$expert_uncertain_30_70, na.rm = TRUE),
    sum(master$expert_uncertain_40_60, na.rm = TRUE),
    nrow(master),
    nrow(master),
    nrow(master),
    nrow(master)
  )
)

workflow_metrics <- workflow_rules %>%
  mutate(metrics = pmap(list(prob_col, pred_col, workflow, ai_review_count), function(prob_col, pred_col, workflow, ai_review_count) {
    # Threshold is not used for rule-based prediction columns, so calculate manually.
    y <- as_binary(master$outcome)
    p <- as.numeric(master[[prob_col]])
    pred <- as_binary(master[[pred_col]])
    keep <- !is.na(y) & is.finite(p) & !is.na(pred)
    y <- y[keep]
    p <- p[keep]
    pred <- pred[keep]
    tp <- sum(pred == 1 & y == 1)
    tn <- sum(pred == 0 & y == 0)
    fp <- sum(pred == 1 & y == 0)
    fn <- sum(pred == 0 & y == 1)
    n <- length(y)
    tibble(
      model = workflow,
      n = n,
      events = sum(y == 1),
      event_rate = mean(y == 1),
      auc = safe_auc(y, p),
      sensitivity = ifelse(tp + fn > 0, tp / (tp + fn), NA_real_),
      specificity = ifelse(tn + fp > 0, tn / (tn + fp), NA_real_),
      ppv = ifelse(tp + fp > 0, tp / (tp + fp), NA_real_),
      npv = ifelse(tn + fn > 0, tn / (tn + fn), NA_real_),
      accuracy = ifelse(n > 0, (tp + tn) / n, NA_real_),
      tp = tp, tn = tn, fp = fp, fn = fn,
      missed_converters = fn,
      unnecessary_high_risk_nonconverters = fp,
      cases_requiring_ai_review = ai_review_count
    )
  })) %>%
  select(metrics) %>%
  unnest(metrics)

reference_workflow <- workflow_metrics %>%
  filter(model == "Expert Stage 2 alone") %>%
  select(ref_fn = fn, ref_fp = fp, ref_sensitivity = sensitivity, ref_specificity = specificity)

workflow_metrics <- workflow_metrics %>%
  mutate(
    avoided_missed_converters_vs_expert = reference_workflow$ref_fn[1] - fn,
    avoided_unnecessary_high_risk_flags_vs_expert = reference_workflow$ref_fp[1] - fp,
    delta_sensitivity_vs_expert = sensitivity - reference_workflow$ref_sensitivity[1],
    delta_specificity_vs_expert = specificity - reference_workflow$ref_specificity[1],
    nns_per_avoided_missed_converter = ifelse(
      avoided_missed_converters_vs_expert > 0,
      cases_requiring_ai_review / avoided_missed_converters_vs_expert,
      NA_real_
    ),
    nns_per_avoided_unnecessary_high_risk_flag = ifelse(
      avoided_unnecessary_high_risk_flags_vs_expert > 0,
      cases_requiring_ai_review / avoided_unnecessary_high_risk_flags_vs_expert,
      NA_real_
    )
  )

write_csv_safe(workflow_metrics, file.path(out_dir, "04_workflow_metrics_vs_expert_stage2.csv"))

workflow_metrics_exploratory_cv <- tibble(
  workflow = "Post-hoc exploratory CV logistic: AI + Expert Stage 2",
  prob_col = "combined_stage2_cv_prob",
  pred_col = "combined_stage2_cv_pred",
  ai_review_count = nrow(master)
) %>%
  mutate(metrics = pmap(list(prob_col, pred_col, workflow, ai_review_count), function(prob_col, pred_col, workflow, ai_review_count) {
    y <- as_binary(master$outcome)
    p <- as.numeric(master[[prob_col]])
    pred <- as_binary(master[[pred_col]])
    keep <- !is.na(y) & is.finite(p) & !is.na(pred)
    y <- y[keep]
    p <- p[keep]
    pred <- pred[keep]
    tp <- sum(pred == 1 & y == 1)
    tn <- sum(pred == 0 & y == 0)
    fp <- sum(pred == 1 & y == 0)
    fn <- sum(pred == 0 & y == 1)
    n <- length(y)
    tibble(
      model = workflow,
      n = n,
      events = sum(y == 1),
      event_rate = mean(y == 1),
      auc = safe_auc(y, p),
      sensitivity = ifelse(tp + fn > 0, tp / (tp + fn), NA_real_),
      specificity = ifelse(tn + fp > 0, tn / (tn + fp), NA_real_),
      ppv = ifelse(tp + fp > 0, tp / (tp + fp), NA_real_),
      npv = ifelse(tn + fn > 0, tn / (tn + fn), NA_real_),
      accuracy = ifelse(n > 0, (tp + tn) / n, NA_real_),
      tp = tp, tn = tn, fp = fp, fn = fn,
      missed_converters = fn,
      unnecessary_high_risk_nonconverters = fp,
      cases_requiring_ai_review = ai_review_count,
      caveat = "Exploratory only; not a frozen validated human-AI integration rule."
    )
  })) %>%
  select(metrics) %>%
  unnest(metrics)

write_csv_safe(workflow_metrics_exploratory_cv, file.path(out_dir, "04_workflow_metrics_exploratory_cv_logistic.csv"))

workflow_plot <- workflow_metrics %>%
  select(model, missed_converters, unnecessary_high_risk_nonconverters,
         avoided_missed_converters_vs_expert, avoided_unnecessary_high_risk_flags_vs_expert) %>%
  pivot_longer(cols = c(missed_converters, unnecessary_high_risk_nonconverters),
               names_to = "error_type", values_to = "count") %>%
  mutate(
    error_type = recode(error_type,
                        missed_converters = "Missed converters",
                        unnecessary_high_risk_nonconverters = "Unnecessary high-risk non-converters"),
    model = factor(model, levels = rev(workflow_metrics$model))
  ) %>%
  ggplot(aes(x = count, y = model, fill = error_type)) +
  geom_col(position = position_dodge(width = 0.76), width = 0.68) +
  labs(x = "Number of cases", y = NULL, fill = NULL,
       title = "Clinical workflow error profile") +
  theme_minimal(base_size = 11) +
  theme(panel.grid.major.y = element_blank())

save_plot_both(workflow_plot, "Figure_Q1_Workflow_Error_Profile", width = 9.5, height = 5.2)

# -------------------------------------------------------------------------
# 7. NRI, IDI, and decision-curve analysis
# -------------------------------------------------------------------------

nri_idi_summary <- bind_rows(
  bootstrap_nri_idi(master, "expert_stage2_prob", "ai_prob",
                    "AI vs expert Stage 2", reps = bootstrap_reps),
  bootstrap_nri_idi(master, "expert_stage2_prob", "mean_ai_expert2_prob",
                    "Leakage-safe mean AI + expert Stage 2 vs expert Stage 2", reps = bootstrap_reps),
  bootstrap_nri_idi(master, "expert_stage2_prob", "rankmean_ai_expert2_prob",
                    "Leakage-safe rank mean AI + expert Stage 2 vs expert Stage 2", reps = bootstrap_reps),
  bootstrap_nri_idi(master, "expert_stage2_prob", "second_reader_40_60_prob",
                    "AI second reader 0.40-0.60 vs expert Stage 2", reps = bootstrap_reps),
  bootstrap_nri_idi(master, "expert_stage2_prob", "expert_uncertain_ai_specificity_prune_prob",
                    "AI specificity-prune in expert-uncertain cases vs expert Stage 2", reps = bootstrap_reps)
)

nri_idi_exploratory_cv <- bind_rows(
  bootstrap_nri_idi(master, "expert_stage2_prob", "combined_stage2_cv_prob",
                    "Post-hoc exploratory CV AI + expert Stage 2 vs expert Stage 2", reps = bootstrap_reps),
  bootstrap_nri_idi(master, "ai_prob", "combined_stage2_cv_prob",
                    "Post-hoc exploratory CV AI + expert Stage 2 vs AI", reps = bootstrap_reps),
  bootstrap_nri_idi(master, "expert_stage1_prob", "combined_stage1_cv_prob",
                    "Post-hoc exploratory CV AI + expert Stage 1 vs expert Stage 1", reps = bootstrap_reps)
) %>%
  mutate(caveat = "Exploratory only; do not report as a frozen externally validated integration benchmark.")

write_csv_safe(nri_idi_summary, file.path(out_dir, "05_nri_idi_summary.csv"))
write_csv_safe(nri_idi_exploratory_cv, file.path(out_dir, "05_nri_idi_exploratory_cv_logistic.csv"))

categorical_nri_primary <- bind_rows(
  bootstrap_categorical_nri(master, "expert_stage2_pred", "second_reader_40_60_pred",
                            "Rule C primary 40-60 vs expert Stage 2", reps = bootstrap_reps),
  bootstrap_categorical_nri(master, "expert_stage2_pred", "ai_confirmed_high_specificity_pred",
                            "Rule B AND high-specificity vs expert Stage 2", reps = bootstrap_reps),
  bootstrap_categorical_nri(master, "expert_stage2_pred", "ai_or_expert_high_sensitivity_pred",
                            "Rule A OR high-sensitivity vs expert Stage 2", reps = bootstrap_reps),
  bootstrap_categorical_nri(master, "expert_stage2_pred", "mean_ai_expert2_pred",
                            "Leakage-safe mean AI + expert Stage 2 vs expert Stage 2", reps = bootstrap_reps),
  bootstrap_categorical_nri(master, "expert_stage2_pred", "rankmean_ai_expert2_pred",
                            "Leakage-safe rank mean AI + expert Stage 2 vs expert Stage 2", reps = bootstrap_reps)
) %>%
  mutate(analysis_tier = "primary_or_prespecified_no_fitted_weights")

categorical_nri_sensitivity_gray_zone <- bind_rows(
  bootstrap_categorical_nri(master, "expert_stage2_pred", "second_reader_35_65_pred",
                            "Rule C sensitivity 35-65 vs expert Stage 2", reps = bootstrap_reps),
  bootstrap_categorical_nri(master, "expert_stage2_pred", "second_reader_30_70_pred",
                            "Rule C sensitivity 30-70 vs expert Stage 2", reps = bootstrap_reps)
) %>%
  mutate(analysis_tier = "prespecified_gray_zone_sensitivity")

write_csv_safe(categorical_nri_primary, file.path(out_dir, "05_categorical_nri_primary.csv"))
write_csv_safe(categorical_nri_sensitivity_gray_zone, file.path(out_dir, "05_categorical_nri_gray_zone_sensitivity.csv"))

dca_thresholds <- seq(0.05, 0.80, by = 0.01)
decision_curve <- bind_rows(
  net_benefit_all_none(master, thresholds = dca_thresholds),
  net_benefit_curve(master, "expert_stage2_prob", "Expert Stage 2", dca_thresholds),
  net_benefit_curve(master, "ai_prob", "AI", dca_thresholds),
  net_benefit_curve(master, "second_reader_40_60_prob", "AI second reader 0.40-0.60", dca_thresholds),
  net_benefit_curve(master, "expert_uncertain_ai_specificity_prune_prob",
                    "AI specificity-prune in expert-uncertain cases", dca_thresholds),
  net_benefit_curve(master, "mean_ai_expert2_prob", "Leakage-safe mean AI + Expert Stage 2", dca_thresholds),
  net_benefit_curve(master, "rankmean_ai_expert2_prob", "Leakage-safe rank mean AI + Expert Stage 2", dca_thresholds)
)

write_csv_safe(decision_curve, file.path(out_dir, "06_decision_curve_net_benefit.csv"))

decision_curve_exploratory_cv <- bind_rows(
  net_benefit_all_none(master, thresholds = dca_thresholds),
  net_benefit_curve(master, "expert_stage2_prob", "Expert Stage 2", dca_thresholds),
  net_benefit_curve(master, "ai_prob", "AI", dca_thresholds),
  net_benefit_curve(master, "combined_stage2_cv_prob",
                    "Post-hoc exploratory CV AI + Expert Stage 2", dca_thresholds)
) %>%
  mutate(caveat = "Exploratory only; fitted integration weights are estimated within the holdout cohort.")

write_csv_safe(decision_curve_exploratory_cv, file.path(out_dir, "06_decision_curve_exploratory_cv_logistic.csv"))

dca_resource_translation <- decision_curve %>%
  filter(round(threshold, 2) %in% c(0.20, 0.30, 0.50)) %>%
  select(model, threshold, net_benefit) %>%
  left_join(
    decision_curve %>%
      filter(model == "Treat all", round(threshold, 2) %in% c(0.20, 0.30, 0.50)) %>%
      select(threshold, treat_all_net_benefit = net_benefit),
    by = "threshold"
  ) %>%
  mutate(
    unnecessary_interventions_avoided_per_100_vs_treat_all =
      (net_benefit - treat_all_net_benefit) / (threshold / (1 - threshold)) * 100
  ) %>%
  arrange(threshold, desc(unnecessary_interventions_avoided_per_100_vs_treat_all))

write_csv_safe(dca_resource_translation, file.path(out_dir, "06_dca_resource_translation_per_100.csv"))

dca_plot <- decision_curve %>%
  filter(threshold <= 0.70) %>%
  ggplot(aes(x = threshold, y = net_benefit, color = model, linewidth = model)) +
  geom_line() +
  scale_linewidth_manual(values = c(
    "Treat none" = 0.55,
    "Treat all" = 0.55,
    "Expert Stage 2" = 0.9,
    "AI" = 0.9,
    "AI second reader 0.40-0.60" = 0.9,
    "AI specificity-prune in expert-uncertain cases" = 0.9,
    "Leakage-safe mean AI + Expert Stage 2" = 1.0,
    "Leakage-safe rank mean AI + Expert Stage 2" = 1.0
  ), guide = "none") +
  labs(x = "Risk threshold", y = "Net benefit", color = NULL,
       title = "Decision-curve analysis for AI, experts, and human-AI workflows") +
  theme_minimal(base_size = 11) +
  theme(legend.position = "bottom")

save_plot_both(dca_plot, "Figure_Q1_Decision_Curve_Human_AI", width = 8.8, height = 5.5)

# -------------------------------------------------------------------------
# 8. Threshold sweep for clinical operating points
# -------------------------------------------------------------------------

threshold_grid <- seq(0.05, 0.95, by = 0.01)

threshold_sweep <- crossing(
  model = c(
    "AI",
    "Expert Stage 2",
    "Leakage-safe mean AI + Expert Stage 2",
    "Leakage-safe rank mean AI + Expert Stage 2",
    "AI second reader 0.40-0.60",
    "AI specificity-prune in expert-uncertain cases"
  ),
  threshold = threshold_grid
) %>%
  mutate(
    prob_col = case_when(
      model == "AI" ~ "ai_prob",
      model == "Expert Stage 2" ~ "expert_stage2_prob",
      model == "Leakage-safe mean AI + Expert Stage 2" ~ "mean_ai_expert2_prob",
      model == "Leakage-safe rank mean AI + Expert Stage 2" ~ "rankmean_ai_expert2_prob",
      model == "AI second reader 0.40-0.60" ~ "second_reader_40_60_prob",
      model == "AI specificity-prune in expert-uncertain cases" ~ "expert_uncertain_ai_specificity_prune_prob",
      TRUE ~ NA_character_
    ),
    metrics = map2(
      prob_col,
      threshold,
      ~ calc_metrics(master$outcome, master[[.x]], .y, .x) %>%
        select(-model, -threshold)
    )
  ) %>%
  select(model, threshold, metrics) %>%
  unnest(metrics) %>%
  select(model, threshold, n, events, event_rate, sensitivity, specificity, ppv, npv,
         accuracy, f1, tp, tn, fp, fn) %>%
  mutate(
    caveat = paste(
      "Descriptive sensitivity analysis only.",
      "Do not select new primary thresholds from the 196-case holdout test set."
    )
  )

write_csv_safe(threshold_sweep, file.path(out_dir, "07_threshold_sweep_metrics_descriptive_sensitivity_only.csv"))

operating_points <- bind_rows(
  threshold_sweep %>%
    filter(specificity >= 0.90) %>%
    group_by(model) %>%
    slice_max(sensitivity, n = 1, with_ties = FALSE) %>%
    ungroup() %>%
    mutate(strategy = "Max sensitivity with specificity >= 0.90"),
  threshold_sweep %>%
    filter(sensitivity >= 0.85) %>%
    group_by(model) %>%
    slice_max(specificity, n = 1, with_ties = FALSE) %>%
    ungroup() %>%
    mutate(strategy = "Max specificity with sensitivity >= 0.85"),
  threshold_sweep %>%
    group_by(model) %>%
    slice_max(f1, n = 1, with_ties = FALSE) %>%
    ungroup() %>%
    mutate(strategy = "Max F1")
) %>%
  mutate(
    caveat = paste(
      "Descriptive operating points identified on the holdout test set.",
      "Use only for sensitivity/discussion unless the same thresholds are prespecified from discovery data."
    )
  ) %>%
  arrange(strategy, model)

write_csv_safe(operating_points, file.path(out_dir, "07_clinical_operating_points_descriptive_not_primary.csv"))

threshold_plot <- threshold_sweep %>%
  filter(model %in% c("AI", "Expert Stage 2", "Leakage-safe mean AI + Expert Stage 2",
                      "AI second reader 0.40-0.60")) %>%
  select(model, threshold, sensitivity, specificity, ppv, npv) %>%
  pivot_longer(cols = c(sensitivity, specificity, ppv, npv),
               names_to = "metric", values_to = "value") %>%
  ggplot(aes(x = threshold, y = value, color = model)) +
  geom_line(linewidth = 0.85) +
  facet_wrap(~ metric, ncol = 2) +
  scale_y_continuous(limits = c(0, 1), labels = scales::percent_format(accuracy = 1)) +
  labs(x = "Probability threshold", y = NULL, color = NULL,
       title = "Threshold-dependent clinical performance") +
  theme_minimal(base_size = 11) +
  theme(legend.position = "bottom")

save_plot_both(threshold_plot, "Figure_Q1_Threshold_Sweep", width = 8.5, height = 6)

# -------------------------------------------------------------------------
# 9. Optional BSI longitudinal mechanism/context layer
# -------------------------------------------------------------------------

extract_rid_from_ptid <- function(x) {
  x <- as.character(x)
  out <- suppressWarnings(as.integer(str_extract(x, "\\d+$")))
  out
}

standardize_bsi_candidate <- function(path) {
  if (!file.exists(path)) return(NULL)
  dat <- tryCatch(readr::read_csv(path, show_col_types = FALSE, progress = FALSE), error = function(e) NULL)
  if (is.null(dat) || nrow(dat) == 0) return(NULL)
  if (!("slope_per_year" %in% names(dat)) && !("DBCBBSI" %in% names(dat))) return(NULL)

  if ("RID" %in% names(dat)) {
    rid <- suppressWarnings(as.integer(dat$RID))
  } else if ("PTID" %in% names(dat)) {
    rid <- extract_rid_from_ptid(dat$PTID)
  } else {
    return(NULL)
  }

  if ("slope_per_year" %in% names(dat)) {
    out <- dat %>%
      mutate(
        RID = rid,
        bsi_slope_per_year = as.numeric(slope_per_year),
        bsi_metric_safe = if ("metric_name" %in% names(dat)) as.character(metric_name) else "Whole Brain BSI"
      ) %>%
      filter(!is.na(RID), is.finite(bsi_slope_per_year)) %>%
      group_by(RID) %>%
      summarise(
        bsi_slope_per_year = mean(bsi_slope_per_year, na.rm = TRUE),
        bsi_n_records = n(),
        bsi_metric = first_nonmissing_chr(bsi_metric_safe),
        bsi_source_file = path,
        .groups = "drop"
      )
  } else {
    out <- dat %>%
      mutate(
        RID = rid,
        Years = if ("Years" %in% names(dat)) as.numeric(Years) else NA_real_,
        DBCBBSI = as.numeric(DBCBBSI)
      ) %>%
      filter(!is.na(RID), is.finite(DBCBBSI), is.finite(Years), Years > 0) %>%
      group_by(RID) %>%
      summarise(
        bsi_slope_per_year = mean(DBCBBSI / Years, na.rm = TRUE),
        bsi_n_records = n(),
        bsi_metric = "Whole Brain BSI estimated from DBCBBSI / Years",
        bsi_source_file = path,
        .groups = "drop"
      )
  }

  out
}

bsi_candidate_paths <- file.path(base_dir, c(
  "longitudinal_bsi_validation/individual_bsi_slopes.csv",
  "longitudinal_bsi_validation/bsi_longitudinal_merged.csv",
  "FOXLABBSI_02Mar2026.csv"
))

bsi_candidates <- purrr::map(bsi_candidate_paths, standardize_bsi_candidate) %>%
  purrr::compact()

master_bsi <- NULL
if (length(bsi_candidates) > 0 && "RID" %in% names(master)) {
  bsi_overlap_report <- map_dfr(bsi_candidates, function(bdat) {
    tibble(
      source_file = unique(bdat$bsi_source_file)[1],
      candidate_rows = nrow(bdat),
      overlap_n = sum(as.integer(master$RID) %in% as.integer(bdat$RID))
    )
  }) %>%
    arrange(desc(overlap_n))

  write_csv_safe(bsi_overlap_report, file.path(out_dir, "08_bsi_candidate_overlap_report.csv"))

  best_bsi_path <- bsi_overlap_report$source_file[which.max(bsi_overlap_report$overlap_n)]
  best_bsi <- bsi_candidates[[which(map_chr(bsi_candidates, ~ unique(.x$bsi_source_file)[1]) == best_bsi_path)[1]]]

  if (max(bsi_overlap_report$overlap_n, na.rm = TRUE) > 0) {
    master_bsi <- master %>%
      mutate(RID = as.integer(RID)) %>%
      left_join(best_bsi, by = "RID")

    write_csv_safe(master_bsi, file.path(out_dir, "08_case_level_master_with_bsi.csv"))

    bsi_group_summary <- master_bsi %>%
      filter(is.finite(bsi_slope_per_year)) %>%
      group_by(discordance_group) %>%
      summarise(
        n = n(),
        median_bsi_slope_per_year = median(bsi_slope_per_year, na.rm = TRUE),
        q1_bsi_slope_per_year = quantile(bsi_slope_per_year, 0.25, na.rm = TRUE),
        q3_bsi_slope_per_year = quantile(bsi_slope_per_year, 0.75, na.rm = TRUE),
        mean_bsi_slope_per_year = mean(bsi_slope_per_year, na.rm = TRUE),
        .groups = "drop"
      )

    bsi_group_test <- compare_feature_by_group(master_bsi, "bsi_slope_per_year") %>%
      mutate(source_file = best_bsi_path)

    bsi_adjusted <- fit_adjusted_discordance(
      master_bsi,
      target_group = "AI correct / Expert wrong",
      predictor = "bsi_slope_per_year",
      adjusters = adjustment_vars
    ) %>%
      mutate(source_file = best_bsi_path)

    write_csv_safe(bsi_group_summary, file.path(out_dir, "08_bsi_discordance_group_summary.csv"))
    write_csv_safe(bind_rows(bsi_group_test, bsi_adjusted), file.path(out_dir, "08_bsi_discordance_tests.csv"))

    bsi_plot <- master_bsi %>%
      filter(is.finite(bsi_slope_per_year)) %>%
      ggplot(aes(x = discordance_group, y = bsi_slope_per_year, fill = discordance_group)) +
      geom_boxplot(width = 0.62, alpha = 0.82, outlier.shape = NA) +
      geom_jitter(width = 0.12, height = 0, size = 1.8, alpha = 0.75) +
      labs(x = NULL, y = "Annualized whole-brain BSI slope",
           title = "Longitudinal atrophy rate across AI-expert discordance groups") +
      theme_minimal(base_size = 11) +
      theme(axis.text.x = element_text(angle = 25, hjust = 1),
            legend.position = "none")

    save_plot_both(bsi_plot, "Figure_Q1_BSI_Discordance", width = 7.5, height = 5)
  }
}

if (is.null(master_bsi)) {
  note <- paste0(
    "No usable BSI file could be linked to the 196-case AI-vs-expert test set by RID.\n",
    "This does not block the primary human-AI analyses. If a holdout-level BSI/atrophy file\n",
    "is available, add it to the candidate list with RID and an annualized slope column.\n\n",
    "Candidate files checked:\n",
    paste(bsi_candidate_paths, collapse = "\n")
  )
  writeLines(note, con = file.path(out_dir, "08_bsi_NOTE_no_linkable_holdout_slopes.txt"))
}

# -------------------------------------------------------------------------
# 10. Optional VAE subgroup mechanism/context layer
# -------------------------------------------------------------------------

standardize_vae_candidate <- function(path) {
  if (!file.exists(path)) return(NULL)
  dat <- tryCatch(readr::read_csv(path, show_col_types = FALSE, progress = FALSE), error = function(e) NULL)
  if (is.null(dat) || nrow(dat) == 0) return(NULL)

  names_original <- names(dat)
  subgroup_col <- intersect(c("VAE_Subgroup", "VAE_Subtype", "Subtype", "subtype", "Subtype_F"), names(dat))[1]
  if (is.na(subgroup_col)) subgroup_col <- NA_character_

  z_cols <- intersect(c("Z1", "Z2", "Z3", "z1", "z2", "z3"), names(dat))
  id_key <- intersect(c("ID", "CaseID", "PTID"), names(dat))[1]
  rid_key <- intersect(c("RID", "rid"), names(dat))[1]

  if (!is.na(id_key)) {
    out <- dat %>%
      transmute(
        join_key = "ID",
        join_value = as.character(.data[[id_key]]),
        VAE_Subgroup = if (!is.na(subgroup_col)) as.character(.data[[subgroup_col]]) else NA_character_,
        Z1 = if ("Z1" %in% names(dat)) as.numeric(.data[["Z1"]]) else NA_real_,
        Z2 = if ("Z2" %in% names(dat)) as.numeric(.data[["Z2"]]) else NA_real_,
        Z3 = if ("Z3" %in% names(dat)) as.numeric(.data[["Z3"]]) else NA_real_
      ) %>%
      group_by(join_key, join_value) %>%
      summarise(
        VAE_Subgroup = first_nonmissing_chr(VAE_Subgroup),
        Z1 = mean(Z1, na.rm = TRUE),
        Z2 = mean(Z2, na.rm = TRUE),
        Z3 = mean(Z3, na.rm = TRUE),
        .groups = "drop"
      )
  } else if (!is.na(rid_key)) {
    out <- dat %>%
      transmute(
        join_key = "RID",
        join_value = as.character(.data[[rid_key]]),
        VAE_Subgroup = if (!is.na(subgroup_col)) as.character(.data[[subgroup_col]]) else NA_character_,
        Z1 = if ("Z1" %in% names(dat)) as.numeric(.data[["Z1"]]) else NA_real_,
        Z2 = if ("Z2" %in% names(dat)) as.numeric(.data[["Z2"]]) else NA_real_,
        Z3 = if ("Z3" %in% names(dat)) as.numeric(.data[["Z3"]]) else NA_real_
      ) %>%
      group_by(join_key, join_value) %>%
      summarise(
        VAE_Subgroup = first_nonmissing_chr(VAE_Subgroup),
        Z1 = mean(Z1, na.rm = TRUE),
        Z2 = mean(Z2, na.rm = TRUE),
        Z3 = mean(Z3, na.rm = TRUE),
        .groups = "drop"
      )
  } else {
    return(NULL)
  }

  out %>%
    mutate(source_file = path, source_columns = paste(names_original, collapse = "|"))
}

vae_candidate_paths <- c(
  file.path(base_dir, "holdout_vae_assignments.csv"),
  file.path(base_dir, "vae_holdout_predictions.csv"),
  file.path(base_dir, "VAE_holdout_assignments.csv"),
  file.path(base_dir, "longitudinal_bsi_validation/bsi_longitudinal_merged.csv"),
  file.path(base_dir, "longitudinal_bsi_validation/individual_bsi_slopes.csv"),
  file.path(base_dir, "PET_cohort_analysis/PET_latent_representations.csv"),
  file.path(base_dir, "vae_revised_output/latent_representations.csv"),
  file.path(base_dir, "vae_revised_output/subtype_assignments.csv")
)

vae_candidates <- purrr::map(vae_candidate_paths, standardize_vae_candidate) %>%
  purrr::compact()

vae_overlap_report <- tibble()
master_vae <- NULL

if (length(vae_candidates) > 0) {
  master_keys <- master %>%
    transmute(ID = as.character(ID), RID = if ("RID" %in% names(.)) as.character(RID) else NA_character_)

  vae_overlap_report <- map_dfr(vae_candidates, function(vdat) {
    if (unique(vdat$join_key)[1] == "ID") {
      overlap_n <- sum(master_keys$ID %in% vdat$join_value)
    } else {
      overlap_n <- sum(master_keys$RID %in% vdat$join_value)
    }
    tibble(
      source_file = unique(vdat$source_file)[1],
      join_key = unique(vdat$join_key)[1],
      candidate_rows = nrow(vdat),
      overlap_n = overlap_n
    )
  }) %>%
    arrange(desc(overlap_n))

  write_csv_safe(vae_overlap_report, file.path(out_dir, "09_vae_candidate_overlap_report.csv"))

  best_idx <- which.max(vae_overlap_report$overlap_n)
  best_path <- vae_overlap_report$source_file[best_idx]
  best_vae <- vae_candidates[[which(map_chr(vae_candidates, ~ unique(.x$source_file)[1]) == best_path)[1]]]

  if (vae_overlap_report$overlap_n[best_idx] > 0) {
    if (unique(best_vae$join_key)[1] == "ID") {
      master_vae <- master %>%
        mutate(ID_join = as.character(ID)) %>%
        left_join(best_vae %>% select(join_value, VAE_Subgroup, Z1, Z2, Z3),
                  by = c("ID_join" = "join_value"))
    } else {
      master_vae <- master %>%
        mutate(RID_join = as.character(RID)) %>%
        left_join(best_vae %>% select(join_value, VAE_Subgroup, Z1, Z2, Z3),
                  by = c("RID_join" = "join_value"))
    }

    master_vae <- master_vae %>%
      mutate(VAE_Subgroup = ifelse(is.na(VAE_Subgroup) | VAE_Subgroup == "", NA, VAE_Subgroup))

    write_csv_safe(master_vae, file.path(out_dir, "09_case_level_master_with_optional_vae.csv"))

    vae_summary <- master_vae %>%
      filter(!is.na(VAE_Subgroup)) %>%
      count(VAE_Subgroup, discordance_group, outcome, name = "n") %>%
      group_by(VAE_Subgroup) %>%
      mutate(percent_within_subgroup = 100 * n / sum(n)) %>%
      ungroup()

    write_csv_safe(vae_summary, file.path(out_dir, "09_vae_discordance_summary.csv"))

    vae_test <- tryCatch({
      tab <- table(master_vae$VAE_Subgroup, master_vae$discordance_group)
      tibble(
        test = "Fisher exact test: VAE_Subgroup by AI-expert discordance group",
        p_value = fisher.test(tab)$p.value,
        source_file = best_path,
        overlap_n = vae_overlap_report$overlap_n[best_idx]
      )
    }, error = function(e) {
      tibble(
        test = "Fisher exact test: VAE_Subgroup by AI-expert discordance group",
        p_value = NA_real_,
        source_file = best_path,
        overlap_n = vae_overlap_report$overlap_n[best_idx]
      )
    })

    z_tests <- c("Z1", "Z2", "Z3") %>%
      keep(~ .x %in% names(master_vae)) %>%
      map_dfr(~ compare_feature_by_group(master_vae, .x))

    write_csv_safe(bind_rows(vae_test, z_tests), file.path(out_dir, "09_vae_discordance_tests.csv"))

    if (nrow(vae_summary) > 0) {
      vae_plot <- master_vae %>%
        filter(!is.na(VAE_Subgroup)) %>%
        count(VAE_Subgroup, discordance_group, name = "n") %>%
        group_by(VAE_Subgroup) %>%
        mutate(percent = 100 * n / sum(n)) %>%
        ungroup() %>%
        ggplot(aes(x = VAE_Subgroup, y = percent, fill = discordance_group)) +
        geom_col(width = 0.72) +
        labs(x = "Exploratory VAE structural subgroup", y = "Cases (%)", fill = NULL,
             title = "Exploratory VAE subgroup distribution across AI-expert discordance") +
        theme_minimal(base_size = 11) +
        theme(panel.grid.major.x = element_blank())

      save_plot_both(vae_plot, "Figure_Q1_Exploratory_VAE_Discordance", width = 7.2, height = 4.8)
    }
  }
}

if (is.null(master_vae)) {
  note <- paste0(
    "No usable VAE subgroup file could be linked to the 196-case AI-vs-expert test set.\n",
    "This does not block the Q1 human-AI analyses. It means the VAE component should remain\n",
    "an exploratory/contextual layer unless a holdout-level VAE assignment file with ID or RID\n",
    "matching independent_test_set.csv is provided.\n\n",
    "Candidate files checked:\n",
    paste(vae_candidate_paths, collapse = "\n")
  )
  writeLines(note, con = file.path(out_dir, "09_vae_discordance_NOTE_no_linkable_holdout_assignments.txt"))
}

# -------------------------------------------------------------------------
# 11. Plain-language summary for manuscript updating
# -------------------------------------------------------------------------

summary_lines <- c(
  "Q1 Human-AI Extension Analysis",
  "================================",
  paste0("Run date: ", Sys.time()),
  paste0("Input folder: ", base_dir),
  paste0("Output folder: ", out_dir),
  "",
  "Primary files to inspect:",
  "00_gray_zone_distribution_check.csv",
  "00_expert_stage2_interrater_icc.csv",
  "00_case_level_master_with_combined_predictions.csv",
  "01_performance_summary_primary_thresholds.csv",
  "01_performance_summary_exploratory_cv_logistic.csv",
  "02_delong_auc_tests_primary_leakage_safe.csv",
  "02_delong_auc_tests_exploratory_cv_logistic.csv",
  "03_case_level_discordance.csv",
  "03_discordance_feature_comparison.csv",
  "03_adjusted_discordance_models_mmse_adjusted.csv",
  "03_ai_correct_expert_wrong_subclassification.csv",
  "04_workflow_metrics_vs_expert_stage2.csv",
  "04_workflow_metrics_exploratory_cv_logistic.csv",
  "05_categorical_nri_primary.csv",
  "05_categorical_nri_gray_zone_sensitivity.csv",
  "05_nri_idi_summary.csv",
  "06_decision_curve_net_benefit.csv",
  "06_dca_resource_translation_per_100.csv",
  "07_threshold_sweep_metrics_descriptive_sensitivity_only.csv",
  "07_clinical_operating_points_descriptive_not_primary.csv",
  "08_bsi_candidate_overlap_report.csv / 08_bsi_discordance_group_summary.csv if linkable",
  "09_vae_candidate_overlap_report.csv / 09_vae_discordance_summary.csv if linkable",
  "",
  "Interpretation rules before manuscript use:",
  "1. Primary human-AI claims must use frozen AI threshold 0.5142, expert threshold 0.50, and prespecified Rule A/B/C workflows.",
  "2. Rule C 40-60 is the primary clinical gray zone. 35-65 and 30-70 are sensitivity analyses only.",
  "3. Logistic stacking is post-hoc exploratory only; do not report as an independent validated integration benchmark.",
  "4. Categorical NRI is the primary reclassification statistic for Rule C. Continuous NRI/IDI is supportive.",
  "5. BSI and VAE are mechanism/context layers; do not make them primary validation claims.",
  "6. If Youden thresholds are reported, label them as sensitivity analyses because they are selected on this test set."
)

writeLines(summary_lines, con = file.path(out_dir, "README_Q1_extension_outputs.txt"))

cat("\nDone.\n")
cat("Outputs written to:\n", out_dir, "\n", sep = "")
cat("Key next files for Codex to read after you run this script:\n")
cat("  00_gray_zone_distribution_check.csv\n")
cat("  00_expert_stage2_interrater_icc.csv\n")
cat("  01_performance_summary_primary_thresholds.csv\n")
cat("  02_delong_auc_tests_primary_leakage_safe.csv\n")
cat("  03_discordance_feature_comparison.csv\n")
cat("  03_adjusted_discordance_models_mmse_adjusted.csv\n")
cat("  04_workflow_metrics_vs_expert_stage2.csv\n")
cat("  05_categorical_nri_primary.csv\n")
cat("  05_nri_idi_summary.csv\n")
cat("  06_decision_curve_net_benefit.csv\n")
cat("  06_dca_resource_translation_per_100.csv\n")
cat("  07_threshold_sweep_metrics_descriptive_sensitivity_only.csv\n")
cat("  07_clinical_operating_points_descriptive_not_primary.csv\n")
