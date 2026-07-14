# ==============================================================================
# MCI-to-AD conversion prediction with latent and multimodal features
# ==============================================================================
library(randomForest)
library(pROC)
library(ggplot2)
library(dplyr)
library(caret)
library(mice)
library(glmnet)
library(xgboost)
library(corrplot)
library(ResourceSelection)
library(optparse)

option_list <- list(
  make_option(c("--vae_dir"), type = "character", default = ".",
              help = "Directory containing subtype_assignments.csv and latent_representations.csv [default: %default]"),
  make_option(c("--data_dir"), type = "character", default = ".",
              help = "Directory containing Clinical_data.csv, RNA_plasma.csv, and metabolites.csv [default: %default]"),
  make_option(c("--output_dir"), type = "character", default = "./step11_results",
              help = "Output directory [default: %default]"),
  make_option(c("--cv_folds"), type = "integer", default = 10,
              help = "Number of cross-validation folds [default: %default]"),
  make_option(c("--n_bootstrap"), type = "integer", default = 2000,
              help = "Number of bootstrap resamples [default: %default]"),
  make_option(c("--n_mice"), type = "integer", default = 5,
              help = "Number of MICE imputations [default: %default]")
)

opt <- parse_args(OptionParser(option_list = option_list))

cat("==============================================================================\\n")
cat("Step 11: MCI-to-AD Conversion Prediction\\n")
cat("==============================================================================\\n\\n")

vae_dir <- opt$vae_dir
data_dir <- opt$data_dir
output_dir <- opt$output_dir
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

N_CV_FOLDS <- opt$cv_folds
N_BOOTSTRAP <- opt$n_bootstrap
N_MICE <- opt$n_mice
set.seed(42)

# ==============================================================================
# 1. Load Data
# ==============================================================================
cat("[1/9] Loading data...\n")
# --- VAE outputs ---
subtypes <- read.csv(file.path(vae_dir, "subtype_assignments.csv"),
                     stringsAsFactors = FALSE)
latent   <- read.csv(file.path(vae_dir, "latent_representations.csv"),
                     stringsAsFactors = FALSE)
subtypes$ID <- as.character(subtypes$ID)
latent$ID   <- as.character(latent$ID)
cat(sprintf("  VAE subtypes: %d participants\n", nrow(subtypes)))
cat(sprintf("  VAE latent:   %d participants x %d dims\n",
            nrow(latent), ncol(latent) - 1))
# Verify AD_Conversion exists
if (!"AD_Conversion" %in% colnames(subtypes)) {
  stop("AD_Conversion column not found in subtype_assignments.csv!")
}
# --- Original multimodal data ---
clinical <- read.csv(file.path(data_dir, "Clinical_data.csv"),
                     stringsAsFactors = FALSE)
smri     <- read.csv(file.path(data_dir, "RNA_plasma.csv"),
                     stringsAsFactors = FALSE)
csf      <- read.csv(file.path(data_dir, "metabolites.csv"),
                     stringsAsFactors = FALSE)
clinical$ID <- as.character(clinical$ID)
smri$ID     <- as.character(smri$ID)
csf$ID      <- as.character(csf$ID)
cat(sprintf("  Clinical: %d x %d\n", nrow(clinical), ncol(clinical)))
cat(sprintf("  sMRI:     %d x %d\n", nrow(smri), ncol(smri)))
cat(sprintf("  CSF:      %d x %d\n", nrow(csf), ncol(csf)))
# ==============================================================================
# 2. Merge and Define Feature Set
# ==============================================================================
cat("\n[2/9] Merging data and defining features...\n")
# Start with outcome
master <- subtypes %>% select(ID, AD_Conversion)
master$AD_Conversion <- factor(master$AD_Conversion,
                               levels = c(0, 1),
                               labels = c("NonConverter", "Converter"))
# Merge latent variables (Z1, Z2, Z3)
master <- master %>% left_join(latent, by = "ID")
# Merge clinical (keep only legal variables)
clinical_keep <- clinical %>%
  select(ID, any_of(c("MMSE", "EDUCATION", "GDS", "APOE4_DOSAGE")))
master <- master %>% left_join(clinical_keep, by = "ID")
# Merge CSF (keep only retained biomarkers)
csf_keep <- csf %>%
  select(ID, any_of(c("PTAU181", "ABETA42_ABETA40_RATIO", "ABETA40")))
master <- master %>% left_join(csf_keep, by = "ID")
# Merge MRI (all ST*** columns)
mri_cols <- grep("^ST", colnames(smri), value = TRUE)
smri_keep <- smri %>% select(ID, all_of(mri_cols))
master <- master %>% left_join(smri_keep, by = "ID")
# --- Define EXCLUDED variables (circularity / confounding) ---
EXCLUDED_PATTERNS <- c("FAQ", "FAQTOTAL", "ADAS13", "ADAS", "CDRSB", "CDR",
                       "^SEX$", "^AGE$", "^GENDER$", "APOE4_STATUS",
                       "AD_Conversion", "^ID$", "Womac", "VAE_Subtype",
                       "Direct_KMeans")
# Build regex
exclude_regex <- paste(EXCLUDED_PATTERNS, collapse = "|")
# Get all feature columns (everything except ID and outcome)
all_cols <- colnames(master)
feature_cols <- all_cols[!grepl(exclude_regex, all_cols, ignore.case = TRUE)]
feature_cols <- feature_cols[feature_cols != "AD_Conversion"]
feature_cols <- feature_cols[feature_cols != "ID"]
# Keep only numeric columns
feature_cols <- feature_cols[sapply(master[, feature_cols, drop = FALSE], is.numeric)]
cat(sprintf("  Total features after exclusion: %d\n", length(feature_cols)))
cat(sprintf("  Breakdown: Z1-Z3 (3) + Clinical (%d) + CSF (%d) + MRI (%d)\n",
            sum(feature_cols %in% c("MMSE", "EDUCATION", "GDS", "APOE4_DOSAGE")),
            sum(feature_cols %in% c("PTAU181", "ABETA42_ABETA40_RATIO", "ABETA40")),
            sum(grepl("^ST", feature_cols))))
# Verify no circular variables leaked in
circular_check <- c("FAQ", "FAQTOTAL", "ADAS13", "CDRSB")
leaked <- intersect(toupper(feature_cols), toupper(circular_check))
if (length(leaked) > 0) {
  stop(paste("CIRCULARITY LEAK DETECTED:", paste(leaked, collapse = ", ")))
} else {
  cat("  Circularity check PASSED: no FAQ/ADAS13/CDRSB in features\n")
}
cat(sprintf("\n  Features used:\n"))
cat(sprintf("    %s\n", paste(feature_cols, collapse = ", ")))
# ==============================================================================
# 3. Missing Data Handling
# ==============================================================================
cat("\n[3/9] MICE multiple imputation...\n")
feature_matrix <- master[, feature_cols, drop = FALSE]
# Report missingness
missing_rate <- colMeans(is.na(feature_matrix))
if (any(missing_rate > 0)) {
  cat("  Missing data:\n")
  for (v in names(missing_rate)[missing_rate > 0]) {
    cat(sprintf("    %s: %.1f%%\n", v, missing_rate[v] * 100))
  }
}
# Remove features with >50% missing
high_miss <- names(missing_rate)[missing_rate > 0.5]
if (length(high_miss) > 0) {
  feature_matrix <- feature_matrix[, !colnames(feature_matrix) %in% high_miss,
                                   drop = FALSE]
  feature_cols <- setdiff(feature_cols, high_miss)
  cat(sprintf("  Removed %d features with >50%% missing: %s\n",
              length(high_miss), paste(high_miss, collapse = ", ")))
}
# MICE imputation
if (any(is.na(feature_matrix))) {
  cat(sprintf("  Running MICE with %d datasets...\n", N_MICE))
  mice_obj <- mice(feature_matrix, m = N_MICE, method = "pmm",
                   maxit = 15, seed = 42, printFlag = FALSE)
  feature_matrix_imputed <- complete(mice_obj, 1)
  cat("  MICE imputation complete\n")
} else {
  feature_matrix_imputed <- feature_matrix
  cat("  No missing values detected\n")
}
cat(sprintf("  Final feature matrix: %d x %d\n",
            nrow(feature_matrix_imputed), ncol(feature_matrix_imputed)))
# ==============================================================================
# 4. Remove Highly Correlated Features
# ==============================================================================
cat("\n[4/9] Removing redundant features (|r| > 0.85)...\n")
cor_matrix <- cor(feature_matrix_imputed, use = "pairwise.complete.obs")
# Save correlation plot
png(file.path(output_dir, "Feature_Correlation_Matrix.png"),
    width = 3000, height = 3000, res = 300)
corrplot(cor_matrix, method = "color", type = "upper", tl.cex = 0.4,
         tl.col = "black", title = "Feature Correlation Matrix",
         mar = c(0, 0, 2, 0))
dev.off()
# AD core biomarkers to prioritize keeping
ad_core_patterns <- c("ABETA", "PTAU", "TAU", "MMSE", "APOE",
                       "Hippocampus", "Entorhinal", "Amygdala",
                       "Z1", "Z2", "Z3", "ST103", "ST108")
is_core_feature <- function(feature_name) {
  any(sapply(ad_core_patterns, function(p) grepl(p, feature_name, ignore.case = TRUE)))
}
# Find highly correlated pairs (threshold 0.85)
high_cor_pairs <- which(abs(cor_matrix) > 0.85 & abs(cor_matrix) < 1, arr.ind = TRUE)
high_cor_pairs <- high_cor_pairs[high_cor_pairs[, 1] < high_cor_pairs[, 2], , drop = FALSE]
features_to_remove <- c()
if (nrow(high_cor_pairs) > 0) {
  for (i in 1:nrow(high_cor_pairs)) {
    feat1 <- rownames(cor_matrix)[high_cor_pairs[i, 1]]
    feat2 <- rownames(cor_matrix)[high_cor_pairs[i, 2]]
    is_f1_core <- is_core_feature(feat1)
    is_f2_core <- is_core_feature(feat2)
    if (is_f1_core && !is_f2_core) {
      features_to_remove <- c(features_to_remove, feat2)
    } else if (!is_f1_core && is_f2_core) {
      features_to_remove <- c(features_to_remove, feat1)
    } else {
      # Both or neither core: remove lower variance
      if (var(feature_matrix_imputed[, feat1]) <
          var(feature_matrix_imputed[, feat2])) {
        features_to_remove <- c(features_to_remove, feat1)
      } else {
        features_to_remove <- c(features_to_remove, feat2)
      }
    }
  }
  features_to_remove <- unique(features_to_remove)
  if (length(features_to_remove) > 0) {
    feature_matrix_imputed <- feature_matrix_imputed[,
      !colnames(feature_matrix_imputed) %in% features_to_remove, drop = FALSE]
    cat(sprintf("  Removed %d correlated features\n", length(features_to_remove)))
  }
}
cat(sprintf("  Features after deduplication: %d\n", ncol(feature_matrix_imputed)))
# ==============================================================================
# 5. Feature Importance (Random Forest)
# ==============================================================================
cat("\n[5/9] Feature importance analysis...\n")
rf_temp <- randomForest(x = feature_matrix_imputed,
                        y = master$AD_Conversion,
                        ntree = 500, importance = TRUE)
importance_scores <- importance(rf_temp)[, "MeanDecreaseGini"]
importance_df <- data.frame(
  Feature    = names(importance_scores),
  Importance = importance_scores,
  IsCore     = sapply(names(importance_scores), is_core_feature)
) %>% arrange(desc(Importance))
write.csv(importance_df,
          file.path(output_dir, "Feature_Importance_RF.csv"),
          row.names = FALSE)
# Select top features + all core features
n_keep <- min(50, max(15, floor(ncol(feature_matrix_imputed) * 0.6)))
top_features <- head(importance_df$Feature, n_keep)
core_in_data <- importance_df$Feature[importance_df$IsCore]
final_features <- unique(c(top_features, core_in_data))
feature_matrix_final <- feature_matrix_imputed[, final_features, drop = FALSE]
cat(sprintf("  Selected %d features (including %d core markers)\n",
            length(final_features),
            sum(sapply(final_features, is_core_feature))))
# Top 10 features
cat("  Top 10 features:\n")
for (i in 1:min(10, nrow(importance_df))) {
  cat(sprintf("    %2d. %s (%.3f)%s\n", i,
              importance_df$Feature[i], importance_df$Importance[i],
              ifelse(importance_df$IsCore[i], " *CORE*", "")))
}
# ==============================================================================
# 6. Model Training with Cross-Validation
# ==============================================================================
cat("\n[6/9] Training models with 10-fold CV...\n")
modeling_data <- data.frame(Outcome = master$AD_Conversion, feature_matrix_final)
# Custom summary function: AUC + Precision + Recall + F1
custom_summary <- function(data, lev = NULL, model = NULL) {
  # Standard twoClassSummary metrics
  tcs <- twoClassSummary(data, lev, model)
  # Precision, Recall, F1
  pred_class <- data$pred
  obs_class  <- data$obs
  pos_level  <- lev[2]  # "Converter"
  tp <- sum(pred_class == pos_level & obs_class == pos_level)
  fp <- sum(pred_class == pos_level & obs_class != pos_level)
  fn <- sum(pred_class != pos_level & obs_class == pos_level)
  precision <- ifelse((tp + fp) > 0, tp / (tp + fp), 0)
  recall    <- ifelse((tp + fn) > 0, tp / (tp + fn), 0)
  f1        <- ifelse((precision + recall) > 0,
                      2 * precision * recall / (precision + recall), 0)
  c(tcs, Precision = precision, Recall = recall, F1 = f1)
}
train_control <- trainControl(
  method = "cv",
  number = N_CV_FOLDS,
  summaryFunction = custom_summary,
  classProbs = TRUE,
  savePredictions = "final"
)
# --- Random Forest ---
cat("  Training Random Forest...\n")
set.seed(42)
model_rf <- train(Outcome ~ ., data = modeling_data, method = "rf",
                  trControl = train_control,
                  tuneGrid = expand.grid(mtry = c(2, 4, 6, 8, 10)),
                  metric = "ROC", ntree = 500)
# --- Elastic Net ---
cat("  Training Elastic Net...\n")
set.seed(42)
model_enet <- train(Outcome ~ ., data = modeling_data, method = "glmnet",
                    trControl = train_control,
                    tuneGrid = expand.grid(
                      alpha = seq(0, 1, by = 0.2),
                      lambda = 10^seq(-4, 0, length = 20)),
                    metric = "ROC")
# --- SVM-RBF ---
cat("  Training SVM-RBF...\n")
set.seed(42)
model_svm <- train(Outcome ~ ., data = modeling_data, method = "svmRadial",
                   trControl = train_control,
                   tuneGrid = expand.grid(
                     sigma = c(0.01, 0.05, 0.1),
                     C = c(0.5, 1, 2, 5)),
                   metric = "ROC")
# --- XGBoost ---
cat("  Training XGBoost...\n")
set.seed(42)
suppressWarnings({
  model_xgb <- train(Outcome ~ ., data = modeling_data, method = "xgbTree",
                     trControl = train_control,
                     tuneGrid = expand.grid(
                       nrounds = c(50, 100, 150),
                       max_depth = c(3, 5, 7),
                       eta = c(0.01, 0.05, 0.1),
                       gamma = 0, colsample_bytree = 0.8,
                       min_child_weight = 1, subsample = 0.8),
                     metric = "ROC", verbose = 0)
})
models_list <- list(
  "Random_Forest" = model_rf,
  "Elastic_Net"   = model_enet,
  "SVM_RBF"       = model_svm,
  "XGBoost"       = model_xgb
)
# ==============================================================================
# 7. Model Evaluation — OUT-OF-FOLD CV Predictions
# ==============================================================================
# CRITICAL FIX: All metrics (AUC CI, Precision, Recall, F1, Brier, Youden
# threshold) are computed from OUT-OF-FOLD CV predictions (model$pred),
# NOT from training set predictions. Using training set predictions would
# produce inflated metrics (e.g., AUC CI = 1.000-1.000 for RF/SVM/XGB).
# ==============================================================================
cat("\n[7/9] Model evaluation (out-of-fold CV predictions)...\n")
cat(sprintf("  Bootstrap: %d iterations for 95%% CI\n", N_BOOTSTRAP))
# Helper: Brier score
calc_brier <- function(actual, pred_prob) mean((pred_prob - actual)^2)
# Helper: Hosmer-Lemeshow
calc_hl <- function(actual, pred_prob, g = 10) {
  tryCatch({
    hl <- hoslem.test(actual, pred_prob, g = g)
    list(stat = hl$statistic, p = hl$p.value)
  }, error = function(e) list(stat = NA, p = NA))
}
model_comparison <- data.frame()
for (model_name in names(models_list)) {
  model <- models_list[[model_name]]
  best_result <- model$results[which.max(model$results$ROC), ]
  # ---- Extract OUT-OF-FOLD CV predictions ----
  # model$pred contains predictions from each fold where the sample
  # was in the held-out set. Filter to the best tuning parameters.
  cv_preds <- model$pred
  # Identify best tuning parameters
  best_tune <- model$bestTune
  for (param_name in colnames(best_tune)) {
    cv_preds <- cv_preds[cv_preds[[param_name]] == best_tune[[param_name]], ]
  }
  # Out-of-fold predicted probabilities and actual labels
  oof_probs  <- cv_preds$Converter
  oof_actual <- as.numeric(cv_preds$obs == "Converter")
  oof_pred_class <- cv_preds$pred
  cat(sprintf("  %s: %d OOF predictions extracted\n", model_name, length(oof_probs)))
  # ---- Precision, Recall, F1 from OOF predictions ----
  tp <- sum(oof_pred_class == "Converter" & cv_preds$obs == "Converter")
  fp <- sum(oof_pred_class == "Converter" & cv_preds$obs == "NonConverter")
  fn <- sum(oof_pred_class == "NonConverter" & cv_preds$obs == "Converter")
  tn <- sum(oof_pred_class == "NonConverter" & cv_preds$obs == "NonConverter")
  precision <- ifelse((tp + fp) > 0, tp / (tp + fp), 0)
  recall    <- ifelse((tp + fn) > 0, tp / (tp + fn), 0)
  f1        <- ifelse((precision + recall) > 0,
                      2 * precision * recall / (precision + recall), 0)
  # ---- Bootstrap AUC CI from OOF predictions ----
  set.seed(42)
  boot_aucs <- numeric(N_BOOTSTRAP)
  for (b in 1:N_BOOTSTRAP) {
    idx <- sample(1:length(oof_actual), replace = TRUE)
    if (length(unique(oof_actual[idx])) < 2) { boot_aucs[b] <- NA; next }
    roc_b <- tryCatch(roc(oof_actual[idx], oof_probs[idx], quiet = TRUE),
                      error = function(e) NULL)
    boot_aucs[b] <- if (!is.null(roc_b)) auc(roc_b) else NA
  }
  boot_aucs <- boot_aucs[!is.na(boot_aucs)]
  auc_ci_lo <- quantile(boot_aucs, 0.025)
  auc_ci_hi <- quantile(boot_aucs, 0.975)
  # ---- Brier + HL from OOF predictions ----
  brier <- calc_brier(oof_actual, oof_probs)
  hl    <- calc_hl(oof_actual, oof_probs)
  # ---- Youden's J threshold from OOF predictions ----
  roc_obj <- roc(oof_actual, oof_probs, quiet = TRUE)
  youden_coords <- coords(roc_obj, "best", ret = c("threshold", "sensitivity",
                                                     "specificity"),
                          best.method = "youden")
  optimal_threshold <- youden_coords$threshold[1]
  model_comparison <- rbind(model_comparison, data.frame(
    Model           = model_name,
    CV_AUC          = round(best_result$ROC, 4),
    AUC_95CI_Lower  = round(auc_ci_lo, 4),
    AUC_95CI_Upper  = round(auc_ci_hi, 4),
    CV_Sensitivity  = round(best_result$Sens, 4),
    CV_Specificity  = round(best_result$Spec, 4),
    Precision       = round(precision, 4),
    Recall          = round(recall, 4),
    F1_Score        = round(f1, 4),
    Brier_Score     = round(brier, 4),
    HL_P_Value      = round(hl$p, 4),
    Youden_Threshold = round(optimal_threshold, 4),
    stringsAsFactors = FALSE
  ))
  cat(sprintf("  %s: AUC=%.3f (%.3f-%.3f), Prec=%.3f, Rec=%.3f, F1=%.3f, Brier=%.4f\n",
              model_name, best_result$ROC, auc_ci_lo, auc_ci_hi,
              precision, recall, f1, brier))
}
# Select best model
best_model_name <- model_comparison$Model[which.max(model_comparison$CV_AUC)]
best_model <- models_list[[best_model_name]]
best_threshold <- model_comparison$Youden_Threshold[
  model_comparison$Model == best_model_name]
cat(sprintf("\n  BEST MODEL: %s (AUC = %.3f, Threshold = %.3f)\n",
            best_model_name, max(model_comparison$CV_AUC), best_threshold))
write.csv(model_comparison,
          file.path(output_dir, "Model_Comparison_Full.csv"),
          row.names = FALSE)
# ==============================================================================
# 8. FAQ Sensitivity Analysis (Reviewer #23)
# ==============================================================================
cat("\n[8/9] FAQ sensitivity analysis...\n")
cat("  Comparing: current model (no FAQ) vs hypothetical model (with FAQ)\n")
# Check if FAQ is available in original clinical data
faq_col <- NULL
if ("FAQTOTAL" %in% colnames(clinical)) {
  faq_col <- "FAQTOTAL"
} else if ("FAQ" %in% colnames(clinical)) {
  faq_col <- "FAQ"
}
if (!is.null(faq_col)) {
  # Build dataset WITH FAQ
  faq_values <- clinical[, c("ID", faq_col)]
  colnames(faq_values) <- c("ID", "FAQ_SCORE")
  faq_merged <- master %>%
    select(ID) %>%
    left_join(faq_values, by = "ID")
  faq_feature_matrix <- cbind(feature_matrix_final, FAQ_SCORE = faq_merged$FAQ_SCORE)
  # Impute FAQ if needed
  if (any(is.na(faq_feature_matrix$FAQ_SCORE))) {
    mice_faq <- mice(faq_feature_matrix, m = 1, method = "pmm",
                     maxit = 10, seed = 42, printFlag = FALSE)
    faq_feature_matrix <- complete(mice_faq, 1)
  }
  modeling_data_faq <- data.frame(Outcome = master$AD_Conversion,
                                  faq_feature_matrix)
  # Train best model type with FAQ
  cat(sprintf("  Training %s WITH FAQ...\n", best_model_name))
  set.seed(42)
  if (best_model_name == "SVM_RBF") {
    model_with_faq <- train(Outcome ~ ., data = modeling_data_faq,
                            method = "svmRadial", trControl = train_control,
                            tuneGrid = expand.grid(
                              sigma = c(0.01, 0.05, 0.1),
                              C = c(0.5, 1, 2, 5)),
                            metric = "ROC")
  } else if (best_model_name == "Random_Forest") {
    model_with_faq <- train(Outcome ~ ., data = modeling_data_faq,
                            method = "rf", trControl = train_control,
                            tuneGrid = expand.grid(mtry = c(2, 4, 6, 8, 10)),
                            metric = "ROC", ntree = 500)
  } else if (best_model_name == "Elastic_Net") {
    model_with_faq <- train(Outcome ~ ., data = modeling_data_faq,
                            method = "glmnet", trControl = train_control,
                            tuneGrid = expand.grid(
                              alpha = seq(0, 1, by = 0.2),
                              lambda = 10^seq(-4, 0, length = 20)),
                            metric = "ROC")
  } else {
    suppressWarnings({
      model_with_faq <- train(Outcome ~ ., data = modeling_data_faq,
                              method = "xgbTree", trControl = train_control,
                              tuneGrid = expand.grid(
                                nrounds = c(50, 100, 150),
                                max_depth = c(3, 5, 7),
                                eta = c(0.01, 0.05, 0.1),
                                gamma = 0, colsample_bytree = 0.8,
                                min_child_weight = 1, subsample = 0.8),
                              metric = "ROC", verbose = 0)
    })
  }
  faq_best <- model_with_faq$results[which.max(model_with_faq$results$ROC), ]
  no_faq_auc <- max(model_comparison$CV_AUC)
  faq_sensitivity <- data.frame(
    Condition     = c("Without FAQ (primary)", "With FAQ (sensitivity)"),
    AUC           = c(no_faq_auc, faq_best$ROC),
    Sensitivity   = c(model_comparison$CV_Sensitivity[
                        model_comparison$Model == best_model_name],
                      faq_best$Sens),
    Specificity   = c(model_comparison$CV_Specificity[
                        model_comparison$Model == best_model_name],
                      faq_best$Spec),
    Delta_AUC     = c(NA, faq_best$ROC - no_faq_auc),
    stringsAsFactors = FALSE
  )
  cat("\n  FAQ Sensitivity Analysis Results:\n")
  cat(sprintf("    Without FAQ: AUC = %.3f\n", no_faq_auc))
  cat(sprintf("    With FAQ:    AUC = %.3f (delta = %+.3f)\n",
              faq_best$ROC, faq_best$ROC - no_faq_auc))
  write.csv(faq_sensitivity,
            file.path(output_dir, "FAQ_Sensitivity_Analysis.csv"),
            row.names = FALSE)
} else {
  cat("  FAQ column not found in clinical data, skipping sensitivity analysis\n")
}
# ==============================================================================
# 9. Export Model Artifacts for Step 2
# ==============================================================================
cat("\n[9/9] Exporting model artifacts for step2...\n")
# Save best model object
saveRDS(best_model,
        file.path(output_dir, "best_model.rds"))
# Save feature list and threshold for step2
model_config <- list(
  best_model_name    = best_model_name,
  optimal_threshold  = best_threshold,
  feature_names      = colnames(feature_matrix_final),
  n_features         = ncol(feature_matrix_final),
  cv_auc             = max(model_comparison$CV_AUC),
  excluded_variables = c("FAQ", "FAQTOTAL", "ADAS13", "CDRSB", "CDR",
                         "SEX", "AGE", "APOE4_STATUS", "AD_Conversion"),
  latent_variables   = c("Z1", "Z2", "Z3"),
  data_note          = "Features include VAE latent Z1-Z3 from 37-var modality-weighted VAE"
)
saveRDS(model_config,
        file.path(output_dir, "model_config.rds"))
cat(sprintf("  Saved: best_model.rds (%s)\n", best_model_name))
cat(sprintf("  Saved: model_config.rds (threshold=%.3f, %d features)\n",
            best_threshold, ncol(feature_matrix_final)))
# ==============================================================================
# 10. Visualizations
# ==============================================================================
cat("\n[10] Generating visualizations...\n")
# Feature Importance Plot (Top 20)
top20 <- head(importance_df, 20)
top20$Label <- ifelse(top20$IsCore,
                      paste0(top20$Feature, " *"),
                      as.character(top20$Feature))
p_imp <- ggplot(top20, aes(x = reorder(Label, Importance),
                           y = Importance, fill = IsCore)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  scale_fill_manual(values = c("FALSE" = "steelblue", "TRUE" = "darkred"),
                    labels = c("Other", "Core Marker")) +
  labs(title = "Top 20 Feature Importance (Random Forest)",
       subtitle = "* = AD core biomarker or VAE latent variable",
       x = "Feature", y = "Mean Decrease Gini", fill = "Type") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"))
ggsave(file.path(output_dir, "Feature_Importance_Top20.png"),
       plot = p_imp, width = 10, height = 8, dpi = 300)
# ROC Curves — using OUT-OF-FOLD CV predictions
roc_plot_data <- data.frame()
for (mn in names(models_list)) {
  m <- models_list[[mn]]
  # Extract OOF predictions for best tune
  cv_p <- m$pred
  bt <- m$bestTune
  for (pn in colnames(bt)) {
    cv_p <- cv_p[cv_p[[pn]] == bt[[pn]], ]
  }
  oof_pp <- cv_p$Converter
  oof_ac <- as.numeric(cv_p$obs == "Converter")
  ro <- roc(oof_ac, oof_pp, quiet = TRUE)
  roc_plot_data <- rbind(roc_plot_data, data.frame(
    Model = rep(mn, length(ro$sensitivities)),
    Sensitivity = ro$sensitivities,
    FPR = 1 - ro$specificities,
    AUC = rep(auc(ro), length(ro$sensitivities))
  ))
}
p_roc <- ggplot(roc_plot_data, aes(x = FPR, y = Sensitivity, color = Model)) +
  geom_line(linewidth = 1.2) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "grey50") +
  labs(title = "ROC Curves: Model Comparison (Out-of-Fold CV)",
       subtitle = sprintf("Best: %s", best_model_name),
       x = "1 - Specificity", y = "Sensitivity") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"))
ggsave(file.path(output_dir, "ROC_Curves_Comparison.png"),
       plot = p_roc, width = 10, height = 8, dpi = 300)
# Model Comparison Barplot
p_comp <- ggplot(model_comparison,
                 aes(x = reorder(Model, CV_AUC), y = CV_AUC)) +
  geom_bar(stat = "identity", fill = "steelblue", alpha = 0.8) +
  geom_errorbar(aes(ymin = AUC_95CI_Lower, ymax = AUC_95CI_Upper), width = 0.2) +
  geom_text(aes(label = sprintf("%.3f", CV_AUC)), vjust = -0.5, size = 4) +
  coord_flip() +
  labs(title = "Model Performance Comparison",
       subtitle = sprintf("95%% CI from %d bootstrap iterations", N_BOOTSTRAP),
       x = "Model", y = "AUC") +
  ylim(0, 1) +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"))
ggsave(file.path(output_dir, "Model_Comparison_AUC.png"),
       plot = p_comp, width = 10, height = 6, dpi = 300)
# Calibration Plot for best model — using OUT-OF-FOLD CV predictions
best_cv_preds <- best_model$pred
best_bt <- best_model$bestTune
for (pn in colnames(best_bt)) {
  best_cv_preds <- best_cv_preds[best_cv_preds[[pn]] == best_bt[[pn]], ]
}
pred_best <- best_cv_preds$Converter
actual_best <- as.numeric(best_cv_preds$obs == "Converter")
cal_bins <- cut(pred_best, breaks = seq(0, 1, length.out = 11), include.lowest = TRUE)
cal_data <- data.frame(predicted = pred_best, actual = actual_best, bin = cal_bins) %>%
  group_by(bin) %>%
  summarise(mean_pred = mean(predicted), mean_obs = mean(actual),
            n = n(), .groups = "drop")
p_cal <- ggplot(cal_data, aes(x = mean_pred, y = mean_obs)) +
  geom_point(aes(size = n), color = "steelblue") +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red") +
  geom_smooth(method = "loess", se = TRUE, color = "blue", alpha = 0.2) +
  scale_size_continuous(name = "N") +
  labs(title = sprintf("Calibration Plot: %s", best_model_name),
       x = "Mean Predicted Probability", y = "Observed Proportion") +
  coord_equal(xlim = c(0, 1), ylim = c(0, 1)) +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"))
ggsave(file.path(output_dir, "Calibration_Plot.png"),
       plot = p_cal, width = 8, height = 8, dpi = 300)
# ==============================================================================
# Summary Report
# ==============================================================================
cat("\n======================================================================\n")
cat("Step 11 Complete: Predictive Modeling\n")
cat("======================================================================\n\n")
cat("Pipeline:\n")
cat(sprintf("  Input: 37-var VAE outputs + raw multimodal data\n"))
cat(sprintf("  Features: %d (after selection)\n", ncol(feature_matrix_final)))
cat(sprintf("  EXCLUDED: FAQ, ADAS13, CDRSB (circularity)\n"))
cat(sprintf("  EXCLUDED: Sex, Age (confounding)\n"))
cat(sprintf("  ADDED: Z1, Z2, Z3 (VAE latent variables)\n"))
cat(sprintf("  MICE: %d imputed datasets\n", N_MICE))
cat(sprintf("  CV: %d-fold\n", N_CV_FOLDS))
cat(sprintf("  Bootstrap: %d iterations\n\n", N_BOOTSTRAP))
cat("Results:\n")
for (i in 1:nrow(model_comparison)) {
  row <- model_comparison[i, ]
  marker <- ifelse(row$Model == best_model_name, " <-- BEST", "")
  cat(sprintf("  %s: AUC=%.3f (%.3f-%.3f), Prec=%.3f, Rec=%.3f, F1=%.3f%s\n",
              row$Model, row$CV_AUC, row$AUC_95CI_Lower, row$AUC_95CI_Upper,
              row$Precision, row$Recall, row$F1_Score, marker))
}
cat(sprintf("\nBest model: %s\n", best_model_name))
cat(sprintf("  AUC: %.3f (95%% CI: %.3f-%.3f)\n",
            model_comparison$CV_AUC[model_comparison$Model == best_model_name],
            model_comparison$AUC_95CI_Lower[model_comparison$Model == best_model_name],
            model_comparison$AUC_95CI_Upper[model_comparison$Model == best_model_name]))
cat(sprintf("  Youden threshold: %.3f\n", best_threshold))
cat(sprintf("\nOutput: %s\n", output_dir))
cat("  Model_Comparison_Full.csv\n")
cat("  FAQ_Sensitivity_Analysis.csv\n")
cat("  Feature_Importance_RF.csv\n")
cat("  best_model.rds (for step2)\n")
cat("  model_config.rds (for step2)\n")
cat("  Feature_Importance_Top20.png\n")
cat("  ROC_Curves_Comparison.png\n")
cat("  Model_Comparison_AUC.png\n")
cat("  Calibration_Plot.png\n")
cat("  Feature_Correlation_Matrix.png\n")
cat("\n======================================================================\n")

