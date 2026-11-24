options(stringsAsFactors = FALSE)
set.seed(42)

pkg <- c("dplyr", "lubridate", "glmnet", "pROC")
need <- pkg[!vapply(pkg, requireNamespace, logical(1), quietly = TRUE)]
if (length(need)) stop("Please install: ", paste(need, collapse = ", "))
library(dplyr)
library(lubridate)
library(glmnet)
library(pROC)

# ---- paths ----
data_root <- Sys.getenv("AD_MULTIMODAL_DATA_ROOT")
AIBL_RAW_DIR <- file.path(data_root, "AIBL_Raw_Data/Data_extract_3.3.0")
ADNI_LABEL_FILE <- file.path(data_root, "Analysis_Inputs/Clinical_Classifier/ADNI_Labeled_For_Classifier.csv")
ADNI_DEMOG_FILE <- file.path(data_root, "ADNI_Raw_Data/LINES/Subject Demographics.csv")
ADNI_DIAG_FILE <- file.path(data_root, "ADNI_Raw_Data/LINES/Diagnostic Summary.csv")
OUTPUT_DIR <- file.path(Sys.getenv("AD_MULTIMODAL_RESULTS_DIR"), "aibl_three_feature")
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

pdx_file <- file.path(AIBL_RAW_DIR, "aibl_pdxconv_01-Jun-2018.csv")
cdr_file <- file.path(AIBL_RAW_DIR, "aibl_cdr_01-Jun-2018.csv")
demog_file <- file.path(AIBL_RAW_DIR, "aibl_ptdemog_01-Jun-2018.csv")
apoe_file <- file.path(AIBL_RAW_DIR, "aibl_apoeres_01-Jun-2018.csv")
must_exist <- c(pdx_file, cdr_file, demog_file, apoe_file, ADNI_LABEL_FILE,
                ADNI_DEMOG_FILE, ADNI_DIAG_FILE)
if (any(!file.exists(must_exist))) stop("Missing file(s): ", paste(must_exist[!file.exists(must_exist)], collapse = "; "))

num <- function(x) suppressWarnings(as.numeric(as.character(x)))
as_date_safe <- function(x) {
  x <- as.character(x)
  out <- suppressWarnings(as.Date(x, format = "%m/%d/%Y"))
  bad <- is.na(out)
  if (any(bad)) out[bad] <- suppressWarnings(as.Date(x[bad], format = "%Y-%m-%d"))
  out
}
safe_auc <- function(y, p) {
  if (length(unique(y)) < 2) return(NA_real_)
  as.numeric(pROC::auc(pROC::roc(y, p, quiet = TRUE, direction = "<")))
}
bootstrap_auc_ci <- function(y, p, B = 2000, seed = 42) {
  set.seed(seed)
  vals <- replicate(B, {
    i <- sample(seq_along(y), replace = TRUE)
    if (length(unique(y[i])) < 2) return(NA_real_)
    safe_auc(y[i], p[i])
  })
  vals <- vals[is.finite(vals)]
  c(AUC_CI_Lower = unname(quantile(vals, 0.025)),
    AUC_CI_Upper = unname(quantile(vals, 0.975)))
}
choose_youden <- function(y, p) {
  roc_obj <- pROC::roc(y, p, quiet = TRUE, direction = "<")
  as.numeric(pROC::coords(roc_obj, "best", best.method = "youden",
                           ret = "threshold", transpose = FALSE)[1])
}
metric_row <- function(y, p, threshold) {
  cls <- as.integer(p >= threshold)
  tp <- sum(cls == 1 & y == 1); tn <- sum(cls == 0 & y == 0)
  fp <- sum(cls == 1 & y == 0); fn <- sum(cls == 0 & y == 1)
  z <- qlogis(pmin(pmax(p, 1e-6), 1 - 1e-6))
  cal <- suppressWarnings(glm(y ~ z, family = binomial()))
  data.frame(
    N = length(y), Events = sum(y == 1), AUC = safe_auc(y, p),
    Brier = mean((p - y)^2), Threshold = threshold,
    Sensitivity = ifelse(tp + fn > 0, tp / (tp + fn), NA_real_),
    Specificity = ifelse(tn + fp > 0, tn / (tn + fp), NA_real_),
    PPV = ifelse(tp + fp > 0, tp / (tp + fp), NA_real_),
    NPV = ifelse(tn + fn > 0, tn / (tn + fn), NA_real_),
    Accuracy = mean(cls == y), False_Positive = fp, False_Negative = fn,
    Calibration_Intercept = unname(coef(cal)[1]),
    Calibration_Slope = unname(coef(cal)[2])
  )
}
stratified_folds <- function(y, k = 5) {
  fold <- integer(length(y))
  for (class_value in sort(unique(y))) {
    i <- which(y == class_value)
    fold[i] <- sample(rep(seq_len(min(k, length(i))), length.out = length(i)))
  }
  fold
}
fit_preprocess <- function(x) {
  med <- apply(x, 2, median, na.rm = TRUE)
  for (j in seq_len(ncol(x))) x[is.na(x[, j]), j] <- med[j]
  ctr <- colMeans(x); scl <- apply(x, 2, sd)
  scl[!is.finite(scl) | scl == 0] <- 1
  list(median = med, center = ctr, scale = scl)
}
apply_preprocess <- function(x, prep) {
  for (j in seq_len(ncol(x))) x[is.na(x[, j]), j] <- prep$median[j]
  sweep(sweep(x, 2, prep$center, "-"), 2, prep$scale, "/")
}
fit_model <- function(x, y, foldid = NULL) {
  alphas <- c(0, 0.5, 1)
  cvfit <- lapply(alphas, function(alpha) {
    cv.glmnet(x, y, family = "binomial", alpha = alpha,
              foldid = foldid, type.measure = "auc", standardize = FALSE,
              keep = TRUE)
  })
  score <- vapply(cvfit, function(z) max(z$cvm, na.rm = TRUE), numeric(1))
  best <- which.max(score)
  list(model = cvfit[[best]], alpha = alphas[best], cv_auc = score[best],
       all_auc = data.frame(alpha = alphas, CV_AUC = score))
}

cat("[1/4] Build AIBL strict 36-month cohort without MMSE ...\n")
pdx <- read.csv(pdx_file) %>%
  transmute(RID = as.character(RID), VISCODE = toupper(trimws(VISCODE)),
            DXCURREN = num(DXCURREN), DXMCI = num(DXMCI), DXAD = num(DXAD)) %>%
  mutate(Is_AD = as.integer(DXAD == 1 | DXCURREN == 3))
cdr <- read.csv(cdr_file) %>%
  transmute(RID = as.character(RID), VISCODE = toupper(trimws(VISCODE)),
            EXAMDATE = as_date_safe(EXAMDATE)) %>%
  filter(!is.na(EXAMDATE)) %>% arrange(RID, EXAMDATE) %>% distinct(RID, VISCODE, .keep_all = TRUE)
demog <- read.csv(demog_file) %>%
  transmute(RID = as.character(RID), VISCODE = toupper(trimws(VISCODE)),
            PTGENDER = num(PTGENDER), PTDOB = as.character(PTDOB)) %>%
  filter(VISCODE == "BL") %>% distinct(RID, .keep_all = TRUE)
apoe <- read.csv(apoe_file) %>%
  transmute(RID = as.character(RID), VISCODE = toupper(trimws(VISCODE)),
            APGEN1 = num(APGEN1), APGEN2 = num(APGEN2)) %>%
  filter(VISCODE == "BL") %>% distinct(RID, .keep_all = TRUE) %>%
  transmute(RID, APOE4_Positive = as.integer(APGEN1 == 4 | APGEN2 == 4))

month_map <- c(BL = 0, M18 = 18, M36 = 36, M54 = 54, M72 = 72, M90 = 90, M108 = 108, M126 = 126)
base_dates <- cdr %>% filter(VISCODE == "BL") %>% select(RID, Baseline_Date = EXAMDATE)
mci_ids <- pdx %>% filter(VISCODE == "BL", DXMCI == 1 | DXCURREN == 2) %>% distinct(RID)
visits <- cdr %>%
  inner_join(mci_ids, by = "RID") %>%
  left_join(base_dates, by = "RID") %>%
  mutate(Date_Months = as.numeric(difftime(EXAMDATE, Baseline_Date, units = "days")) / 30.44,
         VIS_Months = unname(month_map[VISCODE]),
         Followup_Months = ifelse(!is.na(Date_Months) & Date_Months >= 0, Date_Months, VIS_Months)) %>%
  left_join(pdx %>% select(RID, VISCODE, Is_AD), by = c("RID", "VISCODE")) %>%
  mutate(Is_AD = ifelse(is.na(Is_AD), 0L, Is_AD)) %>% filter(!is.na(Followup_Months))

endpoint <- visits %>% group_by(RID) %>%
  summarise(
    First_AD_Month = ifelse(any(Is_AD == 1), min(Followup_Months[Is_AD == 1]), Inf),
    Max_Followup_Month = max(Followup_Months),
    Strict36_Outcome = case_when(
      is.finite(First_AD_Month) & First_AD_Month <= 36 ~ 1L,
      !is.finite(First_AD_Month) & Max_Followup_Month >= 36 ~ 0L,
      TRUE ~ NA_integer_
    ),
    .groups = "drop"
  )

aibl <- base_dates %>% inner_join(mci_ids, by = "RID") %>%
  left_join(demog, by = "RID") %>% left_join(apoe, by = "RID") %>%
  left_join(endpoint, by = "RID") %>%
  mutate(
    Birth_Year = num(gsub("[^0-9]", "", PTDOB)),
    Age = year(Baseline_Date) - Birth_Year,
    Gender_Female = case_when(PTGENDER == 2 ~ 1, PTGENDER == 1 ~ 0, TRUE ~ NA_real_)
  ) %>%
  select(RID, Age, Gender_Female, APOE4_Positive, First_AD_Month,
         Max_Followup_Month, Strict36_Outcome) %>%
  filter(!is.na(Strict36_Outcome), complete.cases(Age, Gender_Female, APOE4_Positive))

if (nrow(aibl) < 10 || length(unique(aibl$Strict36_Outcome)) < 2) stop("AIBL cohort is not analysable after strict endpoint filtering.")
write.csv(aibl, file.path(OUTPUT_DIR, "AIBL_no_MMSE_analysis_dataset.csv"), row.names = FALSE)

cat("[2/4] Build the fixed ADNI three-feature proxy training cohort ...\n")
label <- read.csv(ADNI_LABEL_FILE) %>%
  mutate(RID = sub("^0+", "", sub(".*_S_", "", ID)),
         APOE4_Positive = num(APOE4_Positive)) %>%
  select(RID, APOE4_Positive)

# The three-feature proxy used the fixed 157-person, 84-event discovery labels.
adni_demog <- read.csv(ADNI_DEMOG_FILE) %>%
  transmute(RID = as.character(RID), VISDATE = as_date_safe(VISDATE),
            PTDOBYY = num(PTDOBYY), PTGENDER = num(PTGENDER)) %>%
  filter(!is.na(VISDATE)) %>% arrange(RID, VISDATE) %>% distinct(RID, .keep_all = TRUE) %>%
  transmute(RID, ADNI_Age_Raw = year(VISDATE) - PTDOBYY,
            ADNI_Gender_Female_Raw = case_when(PTGENDER == 2 ~ 1,
                                               PTGENDER == 1 ~ 0,
                                               TRUE ~ NA_real_))
label$AD_Conversion <- read.csv(ADNI_LABEL_FILE)$AD_Conversion
adni <- label %>%
  left_join(adni_demog, by = "RID") %>%
  transmute(RID, AD_Conversion = num(AD_Conversion),
            Age = ADNI_Age_Raw,
            Gender_Female = ADNI_Gender_Female_Raw,
            APOE4_Positive = num(APOE4_Positive)) %>%
  filter(!is.na(AD_Conversion), complete.cases(Age, Gender_Female, APOE4_Positive))
if (nrow(adni) < 30 || length(unique(adni$AD_Conversion)) < 2) stop("ADNI training data are not analysable.")
if (nrow(adni) != 157 || sum(adni$AD_Conversion == 1) != 84) {
  stop("Three-feature ADNI cohort mismatch: obtained N=", nrow(adni), ", events=", sum(adni$AD_Conversion == 1),
       ". Expected N=157 and events=84; do not interpret validation until this is resolved.")
}

features <- c("Age", "Gender_Female", "APOE4_Positive")
x <- as.matrix(adni[, features]); y <- adni$AD_Conversion
x_aibl <- as.matrix(aibl[, features])

cat("[3/4] Cross-validated ADNI fitting and frozen AIBL prediction ...\n")
foldid <- stratified_folds(y, 5)
prep <- fit_preprocess(x)
xz <- apply_preprocess(x, prep)
fit <- fit_model(xz, y, foldid)
oof <- as.numeric(fit$model$fit.preval[, which.min(abs(fit$model$lambda - fit$model$lambda.min))])
threshold <- choose_youden(y, oof)
aibl_p <- as.numeric(predict(fit$model, newx = apply_preprocess(x_aibl, prep),
                             s = "lambda.min", type = "response"))

performance <- bind_rows(
  cbind(Dataset = "ADNI_discovery_OOF_three_feature_proxy", metric_row(y, oof, threshold)),
  cbind(Dataset = "AIBL_strict36_three_feature_proxy", metric_row(aibl$Strict36_Outcome, aibl_p, threshold))
)
adni_ci <- bootstrap_auc_ci(y, oof, seed = 42)
aibl_ci <- bootstrap_auc_ci(aibl$Strict36_Outcome, aibl_p, seed = 43)
performance$AUC_CI_Lower <- c(adni_ci[1], aibl_ci[1])
performance$AUC_CI_Upper <- c(adni_ci[2], aibl_ci[2])
predictions <- aibl %>%
  mutate(Frozen_Proxy_Probability = aibl_p,
         Frozen_Proxy_Prediction = as.integer(aibl_p >= threshold),
         Frozen_Threshold_From_ADNI = threshold)
write.csv(performance, file.path(OUTPUT_DIR, "AIBL_no_MMSE_performance.csv"), row.names = FALSE)
write.csv(predictions, file.path(OUTPUT_DIR, "AIBL_no_MMSE_predictions.csv"), row.names = FALSE)
write.csv(fit$all_auc, file.path(OUTPUT_DIR, "AIBL_no_MMSE_alpha_selection.csv"), row.names = FALSE)
write.csv(data.frame(Feature = features, Training_Median = prep$median,
                     Training_Mean = prep$center, Training_SD = prep$scale),
          file.path(OUTPUT_DIR, "AIBL_no_MMSE_preprocessing.csv"), row.names = FALSE)
coef_df <- as.matrix(coef(fit$model, s = "lambda.min"))
write.csv(data.frame(Term = rownames(coef_df), Coefficient = as.numeric(coef_df)),
          file.path(OUTPUT_DIR, "AIBL_no_MMSE_coefficients.csv"), row.names = FALSE)
write.csv(data.frame(
  AIBL_MMSE_read = FALSE, AIBL_MMSE_used = FALSE,
  Model_Features = paste(features, collapse = "; "),
  AIBL_N = nrow(aibl), AIBL_Events = sum(aibl$Strict36_Outcome == 1),
  AIBL_NonEvents = sum(aibl$Strict36_Outcome == 0),
  Frozen_Threshold_From_ADNI = threshold, Selected_Alpha = fit$alpha,
  Selected_Lambda = fit$model$lambda.min
), file.path(OUTPUT_DIR, "AIBL_no_MMSE_audit.csv"), row.names = FALSE)

cat("[4/4] Finished. Output: ", OUTPUT_DIR, "\n", sep = "")
print(performance)
