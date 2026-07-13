# AIBL reduced-feature external validation

library(optparse)

options(stringsAsFactors = FALSE)

option_list <- list(
  make_option(c("--data_root"), type = "character", default = ".",
              help = "Repository root or data root [default: %default]"),
  make_option(c("--out_dir"), type = "character", default = "./4_external_contextualization/AIBL_Reduced_Feature_External_Validation",
              help = "Output directory [default: %default]"),
  make_option(c("--seed"), type = "integer", default = 20260615,
              help = "Random seed [default: %default]")
)
opt <- parse_args(OptionParser(option_list = option_list))
set.seed(opt$seed)

base_dir <- normalizePath(opt$data_root, winslash = "/", mustWork = FALSE)
out_dir <- normalizePath(opt$out_dir, winslash = "/", mustWork = FALSE)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

adni_path <- file.path(base_dir, "ADNI_Labeled_For_Classifier.csv")
adni_demog_path <- file.path(base_dir,"LINES", "Subject Demographics.csv")
adni_mmse_path <- file.path(base_dir, "LINES", "Mini-Mental State Examination (MMSE).csv")
adni_cdr_path <- file.path(base_dir, "LINES", "Clinical Dementia Rating.csv")
adni_apoe_path <- file.path(base_dir,"APOE", "ApoE Genotyping - Results.csv")
aibl_raw_dir <- file.path(base_dir,  "Data_extract_3.3.0")
aibl_gate_dir <- file.path(base_dir, "AIBL_MRI_Download_Targets")

pdx_path <- file.path(aibl_raw_dir, "aibl_pdxconv_01-Jun-2018.csv")
mmse_path <- file.path(aibl_raw_dir, "aibl_mmse_01-Jun-2018.csv")
cdr_path <- file.path(aibl_raw_dir, "aibl_cdr_01-Jun-2018.csv")
apoe_path <- file.path(aibl_raw_dir, "aibl_apoeres_01-Jun-2018.csv")
ptdemog_path <- file.path(aibl_raw_dir, "aibl_ptdemog_01-Jun-2018.csv")

needed <- c(adni_path, adni_demog_path, adni_mmse_path, adni_cdr_path, adni_apoe_path,
            pdx_path, mmse_path, cdr_path, apoe_path, ptdemog_path)
missing <- needed[!file.exists(needed)]
if (length(missing) > 0) stop("Missing required files:\n", paste(missing, collapse = "\n"))

read_csv_base <- function(path) {
  read.csv(path, check.names = FALSE, stringsAsFactors = FALSE)
}

visit_month <- function(x) {
  x <- tolower(trimws(as.character(x)))
  out <- rep(NA_real_, length(x))
  out[x == "bl"] <- 0
  is_m <- grepl("^m[0-9]+$", x)
  out[is_m] <- as.numeric(sub("^m", "", x[is_m]))
  out
}

safe_num <- function(x) suppressWarnings(as.numeric(x))
visit_rank <- function(x) {
  x <- tolower(trimws(as.character(x)))
  out <- rep(99L, length(x))
  out[x == "bl"] <- 1L
  out[x == "sc"] <- 2L
  out[x == "scmri"] <- 3L
  out[x == "f"] <- 4L
  out
}
take_baseline_like <- function(df) {
  df$.visit_rank <- visit_rank(df$VISCODE)
  df <- df[order(df$RID, df$.visit_rank), ]
  df <- df[!duplicated(df$RID), ]
  df$.visit_rank <- NULL
  df
}
mode_value <- function(x) {
  x <- x[!is.na(x)]
  if (length(x) == 0) return(NA)
  names(sort(table(x), decreasing = TRUE))[1]
}
bind_rows_base <- function(xs) {
  all_names <- unique(unlist(lapply(xs, names)))
  xs2 <- lapply(xs, function(x) {
    missing_names <- setdiff(all_names, names(x))
    for (nm in missing_names) x[[nm]] <- NA
    x[, all_names, drop = FALSE]
  })
  do.call(rbind, xs2)
}

roc_auc <- function(y, p) {
  y <- as.integer(y)
  ok <- is.finite(p) & !is.na(y)
  y <- y[ok]; p <- p[ok]
  n1 <- sum(y == 1); n0 <- sum(y == 0)
  if (n1 == 0 || n0 == 0) return(NA_real_)
  r <- rank(p, ties.method = "average")
  (sum(r[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}

metric_at_threshold <- function(y, p, thr) {
  y <- as.integer(y)
  pred <- as.integer(p >= thr)
  tp <- sum(pred == 1 & y == 1, na.rm = TRUE)
  tn <- sum(pred == 0 & y == 0, na.rm = TRUE)
  fp <- sum(pred == 1 & y == 0, na.rm = TRUE)
  fn <- sum(pred == 0 & y == 1, na.rm = TRUE)
  sens <- ifelse(tp + fn > 0, tp / (tp + fn), NA_real_)
  spec <- ifelse(tn + fp > 0, tn / (tn + fp), NA_real_)
  ppv <- ifelse(tp + fp > 0, tp / (tp + fp), NA_real_)
  npv <- ifelse(tn + fn > 0, tn / (tn + fn), NA_real_)
  acc <- (tp + tn) / (tp + tn + fp + fn)
  bal_acc <- mean(c(sens, spec), na.rm = TRUE)
  data.frame(threshold = thr, TP = tp, TN = tn, FP = fp, FN = fn,
             sensitivity = sens, specificity = spec, PPV = ppv, NPV = npv,
             accuracy = acc, balanced_accuracy = bal_acc)
}

best_youden_threshold <- function(y, p) {
  cand <- sort(unique(p[is.finite(p)]))
  if (length(cand) == 0) return(0.5)
  vals <- do.call(rbind, lapply(cand, function(th) metric_at_threshold(y, p, th)))
  vals$youden <- vals$sensitivity + vals$specificity - 1
  vals <- vals[order(-vals$youden, -vals$balanced_accuracy, vals$threshold), ]
  vals$threshold[1]
}

bootstrap_auc_ci <- function(y, p, B = 2000, seed = 20260615) {
  set.seed(seed)
  y <- as.integer(y)
  n <- length(y)
  vals <- rep(NA_real_, B)
  for (b in seq_len(B)) {
    idx <- sample(seq_len(n), n, replace = TRUE)
    if (length(unique(y[idx])) < 2) next
    vals[b] <- roc_auc(y[idx], p[idx])
  }
  quantile(vals, c(0.025, 0.5, 0.975), na.rm = TRUE, names = FALSE)
}

bootstrap_metric_ci <- function(y, p, thr, B = 2000, seed = 20260616) {
  set.seed(seed)
  n <- length(y)
  vals <- matrix(NA_real_, nrow = B, ncol = 6)
  colnames(vals) <- c("sensitivity", "specificity", "PPV", "NPV", "accuracy", "balanced_accuracy")
  for (b in seq_len(B)) {
    idx <- sample(seq_len(n), n, replace = TRUE)
    if (length(unique(y[idx])) < 2) next
    m <- metric_at_threshold(y[idx], p[idx], thr)
    vals[b, ] <- as.numeric(m[1, colnames(vals)])
  }
  out <- t(apply(vals, 2, function(z) quantile(z, c(0.025, 0.5, 0.975), na.rm = TRUE, names = FALSE)))
  data.frame(metric = rownames(out), lower = out[,1], median = out[,2], upper = out[,3], row.names = NULL)
}

brier_score <- function(y, p) mean((as.integer(y) - p)^2, na.rm = TRUE)

calibration_intercept_slope <- function(y, p) {
  eps <- 1e-6
  lp <- qlogis(pmin(pmax(p, eps), 1 - eps))
  d <- data.frame(y = as.integer(y), lp = lp)
  fit <- try(glm(y ~ lp, data = d, family = binomial()), silent = TRUE)
  if (inherits(fit, "try-error")) return(data.frame(calibration_intercept = NA_real_, calibration_slope = NA_real_))
  cf <- coef(fit)
  data.frame(calibration_intercept = unname(cf[1]), calibration_slope = unname(cf[2]))
}

net_benefit <- function(y, p, thresholds) {
  y <- as.integer(y)
  n <- length(y)
  prevalence <- mean(y == 1)
  do.call(rbind, lapply(thresholds, function(pt) {
    pred <- p >= pt
    tp <- sum(pred & y == 1)
    fp <- sum(pred & y == 0)
    nb <- (tp / n) - (fp / n) * (pt / (1 - pt))
    nb_all <- prevalence - (1 - prevalence) * (pt / (1 - pt))
    nb_none <- 0
    data.frame(threshold = pt, net_benefit = nb, treat_all = nb_all, treat_none = nb_none,
               avoided_unnecessary_per_100_vs_all = (nb - nb_all) / (pt / (1 - pt)) * 100)
  }))
}

prepare_design <- function(train, test, features) {
  train2 <- train[, c("AD_Conversion", features), drop = FALSE]
  test2 <- test[, c("AD_Conversion", features), drop = FALSE]
  means <- list(); sds <- list(); meds <- list(); modes <- list()
  for (f in features) {
    if (is.numeric(train2[[f]])) {
      meds[[f]] <- median(train2[[f]], na.rm = TRUE)
      train2[[f]][is.na(train2[[f]])] <- meds[[f]]
      test2[[f]][is.na(test2[[f]])] <- meds[[f]]
      means[[f]] <- mean(train2[[f]], na.rm = TRUE)
      sds[[f]] <- sd(train2[[f]], na.rm = TRUE)
      if (is.na(sds[[f]]) || sds[[f]] == 0) sds[[f]] <- 1
      train2[[paste0(f, "_z")]] <- (train2[[f]] - means[[f]]) / sds[[f]]
      test2[[paste0(f, "_z")]] <- (test2[[f]] - means[[f]]) / sds[[f]]
    } else {
      modes[[f]] <- mode_value(train2[[f]])
      train2[[f]][is.na(train2[[f]]) | train2[[f]] == ""] <- modes[[f]]
      test2[[f]][is.na(test2[[f]]) | test2[[f]] == ""] <- modes[[f]]
      train2[[f]] <- factor(train2[[f]])
      test2[[f]] <- factor(test2[[f]], levels = levels(train2[[f]]))
    }
  }
  list(train = train2, test = test2, means = means, sds = sds, medians = meds, modes = modes)
}

fit_and_eval <- function(adni, aibl, features, model_name, B = 2000) {
  ds <- prepare_design(adni, aibl, features)
  z_features <- unlist(lapply(features, function(f) if (is.numeric(ds$train[[f]])) paste0(f, "_z") else f))
  formula_txt <- paste("AD_Conversion ~", paste(z_features, collapse = " + "))
  fit <- glm(as.formula(formula_txt), data = ds$train, family = binomial())
  adni_prob <- as.numeric(predict(fit, newdata = ds$train, type = "response"))
  aibl_prob <- as.numeric(predict(fit, newdata = ds$test, type = "response"))
  frozen_thr <- best_youden_threshold(ds$train$AD_Conversion, adni_prob)
  adni_metrics <- cbind(data.frame(model = model_name, cohort = "ADNI_discovery"),
                        data.frame(AUC = roc_auc(ds$train$AD_Conversion, adni_prob),
                                   Brier = brier_score(ds$train$AD_Conversion, adni_prob)),
                        metric_at_threshold(ds$train$AD_Conversion, adni_prob, frozen_thr),
                        calibration_intercept_slope(ds$train$AD_Conversion, adni_prob))
  aibl_auc_ci <- bootstrap_auc_ci(ds$test$AD_Conversion, aibl_prob, B = B)
  aibl_metrics <- cbind(data.frame(model = model_name, cohort = "AIBL_external"),
                        data.frame(AUC = roc_auc(ds$test$AD_Conversion, aibl_prob),
                                   AUC_CI_low = aibl_auc_ci[1], AUC_CI_median = aibl_auc_ci[2], AUC_CI_high = aibl_auc_ci[3],
                                   Brier = brier_score(ds$test$AD_Conversion, aibl_prob)),
                        metric_at_threshold(ds$test$AD_Conversion, aibl_prob, frozen_thr),
                        calibration_intercept_slope(ds$test$AD_Conversion, aibl_prob))
  pred <- cbind(aibl[, setdiff(names(aibl), features), drop = FALSE], aibl[, features, drop = FALSE])
  pred$model <- model_name
  pred$AIBL_reduced_model_prob <- aibl_prob
  pred$AIBL_reduced_model_pred <- as.integer(aibl_prob >= frozen_thr)
  pred$frozen_threshold_from_ADNI <- frozen_thr
  coef_df <- data.frame(model = model_name, term = names(coef(fit)), coefficient = as.numeric(coef(fit)), row.names = NULL)
  metric_ci <- bootstrap_metric_ci(ds$test$AD_Conversion, aibl_prob, frozen_thr, B = B)
  metric_ci$model <- model_name
  dca <- net_benefit(ds$test$AD_Conversion, aibl_prob, thresholds = seq(0.05, 0.80, by = 0.05))
  dca$model <- model_name
  list(fit = fit, threshold = frozen_thr, adni_metrics = adni_metrics, aibl_metrics = aibl_metrics,
       predictions = pred, coefficients = coef_df, bootstrap_metric_ci = metric_ci, dca = dca,
       preprocessing = data.frame(model = model_name,
                                  feature = features,
                                  train_median_or_mode = sapply(features, function(f) if (!is.null(ds$medians[[f]])) ds$medians[[f]] else ds$modes[[f]]),
                                  train_mean = sapply(features, function(f) if (!is.null(ds$means[[f]])) ds$means[[f]] else NA),
                                  train_sd = sapply(features, function(f) if (!is.null(ds$sds[[f]])) ds$sds[[f]] else NA),
                                  row.names = NULL))
}

# ADNI discovery: use the discovery ID/outcome list, but rebuild shared
# clinical variables from raw ADNI tables to keep the same scale as AIBL.
adni_label <- read_csv_base(adni_path)
adni_label$AD_Conversion <- as.integer(adni_label$AD_Conversion)
adni_label$RID <- safe_num(sub(".*_S_", "", adni_label$ID))
adni_label <- adni_label[, c("RID", "ID", "AD_Conversion")]

adni_demog <- read_csv_base(adni_demog_path)
adni_demog$RID <- safe_num(adni_demog$RID)
adni_demog <- take_baseline_like(adni_demog[, c("RID", "VISCODE", "VISDATE", "PTGENDER", "PTDOBYY")])
adni_demog$Gender_Male <- ifelse(adni_demog$PTGENDER == 1, 1,
                                 ifelse(adni_demog$PTGENDER == 2, 0, NA))
adni_demog$birth_year <- safe_num(adni_demog$PTDOBYY)
adni_demog$visit_year <- safe_num(format(as.Date(adni_demog$VISDATE), "%Y"))
adni_demog$Age <- adni_demog$visit_year - adni_demog$birth_year

adni_mmse <- read_csv_base(adni_mmse_path)
adni_mmse$RID <- safe_num(adni_mmse$RID)
adni_mmse <- take_baseline_like(adni_mmse[, c("RID", "VISCODE", "VISDATE", "MMSCORE")])
names(adni_mmse)[names(adni_mmse) == "MMSCORE"] <- "MMSE_Baseline"
adni_mmse$MMSE_Baseline <- safe_num(adni_mmse$MMSE_Baseline)

adni_cdr <- read_csv_base(adni_cdr_path)
adni_cdr$RID <- safe_num(adni_cdr$RID)
adni_cdr <- take_baseline_like(adni_cdr[, c("RID", "VISCODE", "VISDATE", "CDGLOBAL", "CDRSB")])
adni_cdr$CDR_proxy <- safe_num(adni_cdr$CDGLOBAL)

adni_apoe <- read_csv_base(adni_apoe_path)
adni_apoe$RID <- safe_num(adni_apoe$RID)
adni_apoe <- take_baseline_like(adni_apoe[, c("RID", "VISCODE", "GENOTYPE")])
adni_apoe$APOE4_Positive <- as.integer(grepl("4", adni_apoe$GENOTYPE))

adni <- merge(adni_label, adni_demog[, c("RID", "Age", "Gender_Male")], by = "RID", all.x = TRUE)
adni <- merge(adni, adni_mmse[, c("RID", "MMSE_Baseline")], by = "RID", all.x = TRUE)
adni <- merge(adni, adni_apoe[, c("RID", "APOE4_Positive")], by = "RID", all.x = TRUE)
adni <- merge(adni, adni_cdr[, c("RID", "CDR_proxy")], by = "RID", all.x = TRUE)

# AIBL clinical cohort reconstruction
pdx <- read_csv_base(pdx_path)
pdx$visit_month <- visit_month(pdx$VISCODE)
pdx$is_mci <- pdx$DXCURREN == 2 | pdx$DXMCI == 1
pdx$is_ad <- pdx$DXCURREN == 3 | pdx$DXAD == 1
baseline <- pdx[tolower(pdx$VISCODE) == "bl" & pdx$is_mci, ]
follow <- pdx[!is.na(pdx$visit_month) & pdx$visit_month > 0, ]
follow_any <- aggregate(VISCODE ~ RID, data = follow, FUN = length)
names(follow_any)[2] <- "followup_n"
max_follow <- aggregate(visit_month ~ RID, data = follow, FUN = max)
names(max_follow)[2] <- "max_followup_month"
conv <- aggregate(is_ad ~ RID, data = follow, FUN = any)
names(conv)[2] <- "converted_to_ad"
first_ad <- aggregate(visit_month ~ RID, data = follow[follow$is_ad, ], FUN = min)
names(first_ad)[2] <- "first_ad_month"

aibl <- baseline[, c("RID", "SITEID", "VISCODE", "DXCURREN", "DXMCI", "DXAD")]
aibl <- merge(aibl, follow_any, by = "RID", all.x = TRUE)
aibl <- merge(aibl, max_follow, by = "RID", all.x = TRUE)
aibl <- merge(aibl, conv, by = "RID", all.x = TRUE)
aibl <- merge(aibl, first_ad, by = "RID", all.x = TRUE)
aibl <- aibl[!is.na(aibl$followup_n) & aibl$followup_n > 0, ]
aibl$AD_Conversion <- as.integer(aibl$converted_to_ad %in% TRUE)

mmse <- read_csv_base(mmse_path)
mmse <- mmse[tolower(mmse$VISCODE) == "bl", c("RID", "EXAMDATE", "MMSCORE")]
mmse <- mmse[!duplicated(mmse$RID), ]
names(mmse)[names(mmse) == "MMSCORE"] <- "MMSE_Baseline"

cdr <- read_csv_base(cdr_path)
cdr <- cdr[tolower(cdr$VISCODE) == "bl", c("RID", "CDGLOBAL")]
cdr <- cdr[!duplicated(cdr$RID), ]
names(cdr)[names(cdr) == "CDGLOBAL"] <- "CDR_proxy"

apoe <- read_csv_base(apoe_path)
apoe <- apoe[tolower(apoe$VISCODE) == "bl", c("RID", "APGEN1", "APGEN2")]
apoe <- apoe[!duplicated(apoe$RID), ]
apoe$APOE4_Positive <- as.integer(apoe$APGEN1 == 4 | apoe$APGEN2 == 4)

demog <- read_csv_base(ptdemog_path)
demog <- demog[tolower(demog$VISCODE) == "bl", c("RID", "PTGENDER", "PTDOB")]
demog <- demog[!duplicated(demog$RID), ]
demog$Gender_Male <- ifelse(demog$PTGENDER == 1, 1, ifelse(demog$PTGENDER == 2, 0, NA))

aibl <- merge(aibl, mmse, by = "RID", all.x = TRUE)
aibl <- merge(aibl, cdr, by = "RID", all.x = TRUE)
aibl <- merge(aibl, apoe[, c("RID", "APOE4_Positive")], by = "RID", all.x = TRUE)
aibl <- merge(aibl, demog[, c("RID", "Gender_Male", "PTDOB")], by = "RID", all.x = TRUE)
aibl$birth_year <- safe_num(gsub("[^0-9]", "", aibl$PTDOB))
aibl$exam_date <- as.Date(aibl$EXAMDATE, format = "%m/%d/%Y")
aibl$exam_year <- as.numeric(format(aibl$exam_date, "%Y"))
aibl$Age <- aibl$exam_year - aibl$birth_year
aibl$Age <- safe_num(aibl$Age)
aibl$MMSE_Baseline <- safe_num(aibl$MMSE_Baseline)
aibl$CDR_proxy <- safe_num(aibl$CDR_proxy)
aibl$APOE4_Positive <- safe_num(aibl$APOE4_Positive)
aibl$Gender_Male <- safe_num(aibl$Gender_Male)

# Keep complete cases for the main model. This should remain n=48 if the gate data are intact.
main_features <- c("Age", "Gender_Male", "MMSE_Baseline", "APOE4_Positive")
main_adni <- adni[complete.cases(adni[, c("AD_Conversion", main_features)]), c("AD_Conversion", main_features)]
main_aibl <- aibl[complete.cases(aibl[, c("AD_Conversion", main_features)]), c("RID", "SITEID", "followup_n", "max_followup_month", "first_ad_month", "AD_Conversion", main_features)]

sens_features <- c("Age", "Gender_Male", "MMSE_Baseline", "APOE4_Positive", "CDR_proxy")
sens_adni <- adni[complete.cases(adni[, c("AD_Conversion", sens_features)]), c("AD_Conversion", sens_features)]
sens_aibl <- aibl[complete.cases(aibl[, c("AD_Conversion", sens_features)]), c("RID", "SITEID", "followup_n", "max_followup_month", "first_ad_month", "AD_Conversion", sens_features)]

B <- 2000
main_res <- fit_and_eval(main_adni, main_aibl, main_features, "Main_shared_clinical_4_feature", B = B)
res_list <- list(main_res)
if (nrow(sens_adni) > 20 && nrow(sens_aibl) > 20) {
  sens_res <- fit_and_eval(sens_adni, sens_aibl, sens_features, "Sensitivity_plus_CDR_proxy", B = B)
  res_list <- c(res_list, list(sens_res))
}

summary_cohort <- data.frame(
  cohort = c("ADNI_discovery_main_model", "AIBL_external_main_model", "ADNI_discovery_sensitivity", "AIBL_external_sensitivity"),
  n = c(nrow(main_adni), nrow(main_aibl), nrow(sens_adni), nrow(sens_aibl)),
  events = c(sum(main_adni$AD_Conversion == 1), sum(main_aibl$AD_Conversion == 1), sum(sens_adni$AD_Conversion == 1), sum(sens_aibl$AD_Conversion == 1)),
  non_events = c(sum(main_adni$AD_Conversion == 0), sum(main_aibl$AD_Conversion == 0), sum(sens_adni$AD_Conversion == 0), sum(sens_aibl$AD_Conversion == 0))
)

all_adni_metrics <- do.call(rbind, lapply(res_list, `[[`, "adni_metrics"))
all_aibl_metrics <- do.call(rbind, lapply(res_list, `[[`, "aibl_metrics"))
all_coef <- do.call(rbind, lapply(res_list, `[[`, "coefficients"))
all_pred <- bind_rows_base(lapply(res_list, `[[`, "predictions"))
all_ci <- do.call(rbind, lapply(res_list, `[[`, "bootstrap_metric_ci"))
all_dca <- do.call(rbind, lapply(res_list, `[[`, "dca"))
all_prep <- do.call(rbind, lapply(res_list, `[[`, "preprocessing"))

write.csv(summary_cohort, file.path(out_dir, "01_cohort_summary.csv"), row.names = FALSE)
write.csv(all_adni_metrics, file.path(out_dir, "02_adni_discovery_training_performance.csv"), row.names = FALSE)
write.csv(all_aibl_metrics, file.path(out_dir, "03_aibl_external_validation_performance.csv"), row.names = FALSE)
write.csv(all_coef, file.path(out_dir, "04_frozen_model_coefficients.csv"), row.names = FALSE)
write.csv(all_pred, file.path(out_dir, "05_aibl_external_predictions.csv"), row.names = FALSE)
write.csv(all_ci, file.path(out_dir, "06_aibl_bootstrap_metric_ci.csv"), row.names = FALSE)
write.csv(all_dca, file.path(out_dir, "07_aibl_decision_curve.csv"), row.names = FALSE)
write.csv(all_prep, file.path(out_dir, "08_frozen_preprocessing_parameters.csv"), row.names = FALSE)

# Simple plots if ggplot2 is available.
if (requireNamespace("ggplot2", quietly = TRUE)) {
  library(ggplot2)
  pred_main <- all_pred[all_pred$model == "Main_shared_clinical_4_feature", ]
  p1 <- ggplot(pred_main, aes(x = AIBL_reduced_model_prob, fill = factor(AD_Conversion))) +
    geom_histogram(position = "identity", alpha = 0.55, bins = 12) +
    geom_vline(xintercept = unique(pred_main$frozen_threshold_from_ADNI), linetype = 2) +
    scale_fill_manual(values = c("0" = "#5B677A", "1" = "#C84B31"), name = "AD conversion") +
    labs(x = "Frozen ADNI-trained reduced-model probability", y = "AIBL participants") +
    theme_classic(base_size = 12)
  ggsave(file.path(out_dir, "09_aibl_probability_distribution.png"), p1, width = 6, height = 4, dpi = 300)

  dca_main <- all_dca[all_dca$model == "Main_shared_clinical_4_feature", ]
  p2 <- ggplot(dca_main, aes(x = threshold)) +
    geom_line(aes(y = net_benefit, color = "Reduced model"), linewidth = 0.9) +
    geom_line(aes(y = treat_all, color = "Treat all"), linewidth = 0.7, linetype = 2) +
    geom_hline(yintercept = 0, color = "grey40") +
    scale_color_manual(values = c("Reduced model" = "#0F6B6E", "Treat all" = "#9A6A22"), name = NULL) +
    labs(x = "Risk threshold", y = "Net benefit") +
    theme_classic(base_size = 12)
  ggsave(file.path(out_dir, "10_aibl_decision_curve.png"), p2, width = 6, height = 4, dpi = 300)
}

sink(file.path(out_dir, "00_README_results_summary.txt"))
cat("AIBL reduced-feature external validation\n")
cat("Output directory:", out_dir, "\n\n")
cat("Design: ADNI discovery-trained reduced clinical model; threshold derived in ADNI discovery; frozen application to AIBL.\n")
cat("Main model features:", paste(main_features, collapse = ", "), "\n")
cat("Sensitivity model features:", paste(sens_features, collapse = ", "), "\n\n")
cat("Cohort summary:\n")
print(summary_cohort)
cat("\nAIBL external performance:\n")
print(all_aibl_metrics)
cat("\nInterpretation guardrail:\n")
cat("This is external validation of a prespecified reduced-feature clinical model, not direct external validation of the full multimodal AI model.\n")
sink()

cat("\nCompleted AIBL reduced-feature external validation.\n")
cat("Output directory: ", out_dir, "\n", sep = "")
cat("Main AIBL performance file: ", file.path(out_dir, "03_aibl_external_validation_performance.csv"), "\n", sep = "")
print(summary_cohort)
print(all_aibl_metrics)
