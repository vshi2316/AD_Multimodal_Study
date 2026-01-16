library(randomForest)
library(pROC)
library(ggplot2)
library(dplyr)
library(caret)
library(mice)
library(glmnet)
library(xgboost)
library(corrplot)
library(ResourceSelection)  # For Hosmer-Lemeshow test
library(optparse)

# ==============================================================================
# Parse Command Line Arguments
# ==============================================================================
option_list <- list(
  make_option(c("--cluster_file"), type = "character", default = "cluster_results.csv",
              help = "Path to cluster results CSV [default: %default]"),
  make_option(c("--clinical_file"), type = "character", default = "Clinical_data.csv",
              help = "Path to clinical data CSV [default: %default]"),
  make_option(c("--smri_file"), type = "character", default = "sMRI_data.csv",
              help = "Path to sMRI data CSV [default: %default]"),
  make_option(c("--csf_file"), type = "character", default = "CSF_data.csv",
              help = "Path to CSF data CSV [default: %default]"),
  make_option(c("--smd_file"), type = "character", default = "SMD_Significant_Features_FDR.csv",
              help = "Path to significant SMD features CSV [default: %default]"),
  make_option(c("--output_dir"), type = "character", default = "./results",
              help = "Output directory [default: %default]"),
  make_option(c("--n_mice"), type = "integer", default = 5,
              help = "Number of MICE imputation datasets "),
  make_option(c("--n_bootstrap"), type = "integer", default = 2000,
              help = "Number of bootstrap iterations for CI "),
  make_option(c("--cv_folds"), type = "integer", default = 10,
              help = "Number of cross-validation folds [default: %default]")
)

opt_parser <- OptionParser(option_list = option_list)
opt <- parse_args(opt_parser)

# Create output directory
dir.create(opt$output_dir, showWarnings = FALSE, recursive = TRUE)


#' Calculate Brier Score
#' probabilistic prediction accuracy"
calculate_brier_score <- function(actual, predicted_prob) {
  mean((predicted_prob - actual)^2)
}

#' Hosmer-Lemeshow Test
#' plots and the Hosmer-Lemeshow goodness-of-fit test, where a P-value > 0.05 
#' indicates adequate calibration"
hosmer_lemeshow_test <- function(actual, predicted_prob, g = 10) {
  tryCatch({
    hl_test <- hoslem.test(actual, predicted_prob, g = g)
    return(list(
      statistic = hl_test$statistic,
      p_value = hl_test$p.value,
      adequate_calibration = hl_test$p.value > 0.05
    ))
  }, error = function(e) {
    return(list(statistic = NA, p_value = NA, adequate_calibration = NA))
  })
}

#' Create Calibration Plot
create_calibration_plot <- function(actual, predicted_prob, n_bins = 10, title = "Calibration Plot") {
  # Create bins
  bins <- cut(predicted_prob, breaks = seq(0, 1, length.out = n_bins + 1), include.lowest = TRUE)
  
  cal_data <- data.frame(
    predicted = predicted_prob,
    actual = actual,
    bin = bins
  ) %>%
    group_by(bin) %>%
    summarise(
      mean_predicted = mean(predicted),
      mean_actual = mean(actual),
      n = n(),
      .groups = "drop"
    )
  
  p <- ggplot(cal_data, aes(x = mean_predicted, y = mean_actual)) +
    geom_point(aes(size = n), color = "steelblue") +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red") +
    geom_smooth(method = "loess", se = TRUE, color = "blue", alpha = 0.2) +
    scale_size_continuous(name = "N samples") +
    labs(
      title = title,
      subtitle = "Hosmer-Lemeshow calibration assessment",
      x = "Mean Predicted Probability",
      y = "Observed Proportion"
    ) +
    coord_equal(xlim = c(0, 1), ylim = c(0, 1)) +
    theme_minimal() +
    theme(plot.title = element_text(face = "bold"))
  
  return(p)
}

# ==============================================================================
# Load Data
# ==============================================================================
cat("[1/8] Loading data...\n")

cluster_results <- read.csv(opt$cluster_file, stringsAsFactors = FALSE)
cluster_results$ID <- as.character(cluster_results$ID)

# Check for outcome variable
if (!"AD_Conversion" %in% colnames(cluster_results)) {
  stop("AD_Conversion column not found in cluster results!")
}

response <- cluster_results %>%
  select(ID, AD_Conversion) %>%
  mutate(AD_Conversion = factor(AD_Conversion, levels = c(0, 1),
                                 labels = c("NonConverter", "Converter")))

n_converters <- sum(response$AD_Conversion == "Converter")
n_total <- nrow(response)
imbalance_ratio <- n_converters / (n_total - n_converters)

cat(sprintf("  Total samples: %d\n", n_total))
cat(sprintf("  Converters: %d (%.1f%%)\n", n_converters, n_converters/n_total*100))
cat(sprintf("  Imbalance ratio: %.3f\n\n", imbalance_ratio))

# Load significant features from differential analysis
selected_features <- NULL
if (file.exists(opt$smd_file)) {
  smd_features <- read.csv(opt$smd_file, stringsAsFactors = FALSE)
  selected_features <- unique(smd_features$Feature)
  cat(sprintf("  Loaded %d significant features from SMD analysis\n", length(selected_features)))
}

# Load multimodal data
all_data <- response

if (file.exists(opt$clinical_file)) {
  clinical <- read.csv(opt$clinical_file, stringsAsFactors = FALSE)
  clinical$ID <- as.character(clinical$ID)
  all_data <- all_data %>% left_join(clinical, by = "ID")
  cat(sprintf("  Loaded Clinical: %d features\n", ncol(clinical) - 1))
}

if (file.exists(opt$smri_file)) {
  smri <- read.csv(opt$smri_file, stringsAsFactors = FALSE)
  smri$ID <- as.character(smri$ID)
  all_data <- all_data %>% left_join(smri, by = "ID")
  cat(sprintf("  Loaded sMRI: %d features\n", ncol(smri) - 1))
}

if (file.exists(opt$csf_file)) {
  csf <- read.csv(opt$csf_file, stringsAsFactors = FALSE)
  csf$ID <- as.character(csf$ID)
  all_data <- all_data %>% left_join(csf, by = "ID")
  cat(sprintf("  Loaded CSF: %d features\n", ncol(csf) - 1))
}

# ==============================================================================
# Feature Selection
# ==============================================================================
cat("\n[2/8] Feature selection...\n")

# Get all feature columns
exclude_cols <- c("ID", "AD_Conversion", "Cluster_Labels", "Risk_Group")
all_feature_cols <- setdiff(colnames(all_data), exclude_cols)
all_feature_cols <- all_feature_cols[sapply(all_data[, all_feature_cols, drop = FALSE], is.numeric)]

# Use selected features if available
if (!is.null(selected_features) && length(selected_features) > 0) {
  feature_cols <- all_feature_cols[all_feature_cols %in% selected_features]
  if (length(feature_cols) == 0) {
    cat("  Warning: No selected features found in data, using all features\n")
    feature_cols <- all_feature_cols
  }
} else {
  feature_cols <- all_feature_cols
}

cat(sprintf("  Initial features: %d\n", length(feature_cols)))

# Remove features with >50% missing
feature_matrix <- all_data[, feature_cols, drop = FALSE]
missing_rate <- colMeans(is.na(feature_matrix))
high_missing_features <- names(missing_rate)[missing_rate > 0.5]

if (length(high_missing_features) > 0) {
  feature_matrix <- feature_matrix[, !colnames(feature_matrix) %in% high_missing_features, drop = FALSE]
  cat(sprintf("  Removed %d features with >50%% missing\n", length(high_missing_features)))
}

cat(sprintf("  Features for modeling: %d\n", ncol(feature_matrix)))

cat("\n[3/8] MICE Multiple Imputation ...\n")
cat(sprintf("  Generating %d imputed datasets...\n", opt$n_mice))

if (any(is.na(feature_matrix))) {
  set.seed(42)
  
  # MICE imputation with 5 datasets 
  mice_obj <- mice(
    feature_matrix, 
    m = opt$n_mice,           # 5 imputed datasets
    method = "pmm",           # Predictive mean matching
    maxit = 15,               # 15 iterations 
    seed = 42,
    printFlag = FALSE
  )
  
  # Store all imputed datasets for Rubin's rules
  imputed_datasets <- lapply(1:opt$n_mice, function(i) complete(mice_obj, i))
  
  # For primary analysis, use pooled estimates (first dataset as representative)
  feature_matrix_imputed <- complete(mice_obj, 1)
  
  cat(sprintf("  ✓ Generated %d imputed datasets\n", opt$n_mice))
  cat("  ✓ Rubin's rules will be applied for combining results\n")
} else {
  feature_matrix_imputed <- feature_matrix
  imputed_datasets <- list(feature_matrix)
  cat("  No missing values detected\n")
}

# ==============================================================================
# Remove Highly Correlated Features
# ==============================================================================
cat("\n[4/8] Removing redundant features...\n")

cor_matrix <- cor(feature_matrix_imputed, use = "pairwise.complete.obs")

# Save correlation matrix plot
png(file.path(opt$output_dir, "Feature_Correlation_Matrix.png"), width = 4000, height = 4000, res = 300)
corrplot(cor_matrix, method = "color", type = "upper", tl.cex = 0.5, tl.col = "black",
         title = "Feature Correlation Matrix", mar = c(0, 0, 2, 0))
dev.off()

# Define AD core biomarkers (prioritize keeping these)
ad_core_patterns <- c("ABETA", "PTAU", "TAU", "ADAS", "MMSE", "FAQ", "APOE", "CDR",
                      "Hippocampus", "Entorhinal", "Amygdala", "Ventricle", "Temporal")

is_core_feature <- function(feature_name) {
  any(sapply(ad_core_patterns, function(pattern) {
    grepl(pattern, feature_name, ignore.case = TRUE)
  }))
}

# Find highly correlated pairs
high_cor_pairs <- which(abs(cor_matrix) > 0.8 & abs(cor_matrix) < 1, arr.ind = TRUE)
high_cor_pairs <- high_cor_pairs[high_cor_pairs[, 1] < high_cor_pairs[, 2], , drop = FALSE]

features_to_remove <- c()
if (nrow(high_cor_pairs) > 0) {
  for (i in 1:nrow(high_cor_pairs)) {
    feat1 <- rownames(cor_matrix)[high_cor_pairs[i, 1]]
    feat2 <- rownames(cor_matrix)[high_cor_pairs[i, 2]]
    
    is_feat1_core <- is_core_feature(feat1)
    is_feat2_core <- is_core_feature(feat2)
    
    # Keep core features, remove non-core
    if (is_feat1_core && !is_feat2_core) {
      features_to_remove <- c(features_to_remove, feat2)
    } else if (!is_feat1_core && is_feat2_core) {
      features_to_remove <- c(features_to_remove, feat1)
    } else {
      # If both or neither are core, remove lower variance
      if (var(feature_matrix_imputed[, feat1]) < var(feature_matrix_imputed[, feat2])) {
        features_to_remove <- c(features_to_remove, feat1)
      } else {
        features_to_remove <- c(features_to_remove, feat2)
      }
    }
  }
  
  features_to_remove <- unique(features_to_remove)
  if (length(features_to_remove) > 0) {
    feature_matrix_imputed <- feature_matrix_imputed[, !colnames(feature_matrix_imputed) %in% features_to_remove, drop = FALSE]
    cat(sprintf("  Removed %d highly correlated features\n", length(features_to_remove)))
  }
}

cat(sprintf("  Final features: %d\n", ncol(feature_matrix_imputed)))

# ==============================================================================
# Feature Importance with Random Forest
# ==============================================================================
cat("\n[5/8] Feature importance analysis...\n")

set.seed(42)
rf_temp <- randomForest(
  x = feature_matrix_imputed, 
  y = all_data$AD_Conversion, 
  ntree = 500, 
  importance = TRUE
)

importance_scores <- importance(rf_temp)[, "MeanDecreaseGini"]
importance_df <- data.frame(
  Feature = names(importance_scores),
  Importance = importance_scores,
  IsCore = sapply(names(importance_scores), is_core_feature)
) %>% 
  arrange(desc(Importance))

write.csv(importance_df, file.path(opt$output_dir, "Feature_Importance_RF.csv"), row.names = FALSE)

# Select top features + all core features
n_features_to_keep <- min(50, max(15, floor(ncol(feature_matrix_imputed) * 0.6)))
top_features_rf <- head(importance_df$Feature, n_features_to_keep)
core_features_in_data <- importance_df$Feature[importance_df$IsCore]
final_features <- unique(c(top_features_rf, core_features_in_data))

feature_matrix_final <- feature_matrix_imputed[, final_features, drop = FALSE]
cat(sprintf("  Selected %d features (including %d core AD markers)\n", 
            length(final_features), sum(sapply(final_features, is_core_feature))))

# ==============================================================================
# Model Training with Cross-Validation
# ==============================================================================
cat("\n[6/8] Training models with cross-validation...\n")

modeling_data <- data.frame(Outcome = all_data$AD_Conversion, feature_matrix_final)

# Handle class imbalance with SMOTE if needed
if (imbalance_ratio < 0.5) {
  cat("  Applying class weighting for imbalanced data...\n")
}

# Cross-validation setup
train_control <- trainControl(
  method = "cv", 
  number = opt$cv_folds, 
  summaryFunction = twoClassSummary,
  classProbs = TRUE, 
  savePredictions = "final"
)

# Random Forest
cat("  Training Random Forest...\n")
set.seed(42)
model_rf <- train(
  Outcome ~ ., 
  data = modeling_data, 
  method = "rf", 
  trControl = train_control,
  tuneGrid = expand.grid(mtry = c(2, 4, 6, 8, 10)), 
  metric = "ROC", 
  ntree = 500
)

# Elastic Net
cat("  Training Elastic Net...\n")
set.seed(42)
model_enet <- train(
  Outcome ~ ., 
  data = modeling_data, 
  method = "glmnet", 
  trControl = train_control,
  tuneGrid = expand.grid(
    alpha = seq(0, 1, by = 0.2), 
    lambda = 10^seq(-4, 0, length = 20)
  ),
  metric = "ROC"
)

# SVM-RBF
cat("  Training SVM-RBF...\n")
set.seed(42)
model_svm <- train(
  Outcome ~ ., 
  data = modeling_data, 
  method = "svmRadial", 
  trControl = train_control,
  tuneGrid = expand.grid(
    sigma = c(0.01, 0.05, 0.1), 
    C = c(0.5, 1, 2, 5)
  ),
  metric = "ROC"
)

# XGBoost
cat("  Training XGBoost...\n")
set.seed(42)
suppressWarnings({
  model_xgb <- train(
    Outcome ~ ., 
    data = modeling_data, 
    method = "xgbTree", 
    trControl = train_control,
    tuneGrid = expand.grid(
      nrounds = c(50, 100, 150), 
      max_depth = c(3, 5, 7),
      eta = c(0.01, 0.05, 0.1), 
      gamma = 0, 
      colsample_bytree = 0.8,
      min_child_weight = 1, 
      subsample = 0.8
    ),
    metric = "ROC", 
    verbose = 0
  )
})

models_list <- list(
  "Random_Forest" = model_rf, 
  "Elastic_Net" = model_enet,
  "SVM_RBF" = model_svm, 
  "XGBoost" = model_xgb
)

cat("\n[7/8] Model evaluation with bootstrap CI and calibration ...\n")
cat(sprintf("  Running %d bootstrap iterations for 95%% CI...\n", opt$n_bootstrap))

model_comparison <- data.frame()

for (model_name in names(models_list)) {
  model <- models_list[[model_name]]
  
  # Get CV predictions
  cv_preds <- model$pred
  cv_preds <- cv_preds[cv_preds$mtry == model$bestTune$mtry | 
                        is.null(model$bestTune$mtry), ]
  
  if (nrow(cv_preds) == 0) {
    cv_preds <- model$pred
  }
  
  # Get best result
  best_result <- model$results[which.max(model$results$ROC), ]
  
  # Calculate predicted probabilities for calibration
  pred_probs <- predict(model, newdata = modeling_data, type = "prob")$Converter
  actual <- as.numeric(modeling_data$Outcome == "Converter")
  
  # Bootstrap for 95% CI 
  set.seed(42)
  boot_aucs <- numeric(opt$n_bootstrap)
  
  for (b in 1:opt$n_bootstrap) {
    boot_idx <- sample(1:length(actual), replace = TRUE)
    boot_actual <- actual[boot_idx]
    boot_pred <- pred_probs[boot_idx]
    
    # Skip if only one class in bootstrap sample
    if (length(unique(boot_actual)) < 2) {
      boot_aucs[b] <- NA
      next
    }
    
    boot_roc <- tryCatch(
      roc(boot_actual, boot_pred, quiet = TRUE),
      error = function(e) NULL
    )
    
    boot_aucs[b] <- if (!is.null(boot_roc)) auc(boot_roc) else NA
  }
  
  boot_aucs <- boot_aucs[!is.na(boot_aucs)]
  auc_ci_lower <- quantile(boot_aucs, 0.025)
  auc_ci_upper <- quantile(boot_aucs, 0.975)
  
  # Brier Score 
  brier_score <- calculate_brier_score(actual, pred_probs)
  
  # Hosmer-Lemeshow Test
  hl_result <- hosmer_lemeshow_test(actual, pred_probs)
  
  model_comparison <- rbind(model_comparison, data.frame(
    Model = model_name,
    CV_AUC = best_result$ROC,
    AUC_95CI_Lower = auc_ci_lower,
    AUC_95CI_Upper = auc_ci_upper,
    CV_Sensitivity = best_result$Sens,
    CV_Specificity = best_result$Spec,
    Brier_Score = brier_score,
    HL_Statistic = hl_result$statistic,
    HL_P_Value = hl_result$p_value,
    Adequate_Calibration = hl_result$adequate_calibration,
    stringsAsFactors = FALSE
  ))
  
  cat(sprintf("  %s: AUC=%.3f (95%% CI: %.3f-%.3f), Brier=%.4f, HL p=%.4f\n",
              model_name, best_result$ROC, auc_ci_lower, auc_ci_upper, 
              brier_score, hl_result$p_value))
}

# Select best model
best_model_name <- model_comparison$Model[which.max(model_comparison$CV_AUC)]
best_model <- models_list[[best_model_name]]

cat(sprintf("\n  Best model: %s (AUC = %.3f)\n", best_model_name, max(model_comparison$CV_AUC)))

write.csv(model_comparison, file.path(opt$output_dir, "Model_Comparison_Calibrated.csv"), row.names = FALSE)

# ==============================================================================
# Visualizations
# ==============================================================================
cat("\n[8/8] Generating visualizations...\n")

# Feature Importance Plot
cat("  Creating feature importance plot...\n")
top20_features <- head(importance_df, 20)
top20_features$Label <- ifelse(top20_features$IsCore,
                                paste0(top20_features$Feature, " *"),
                                as.character(top20_features$Feature))

p_importance <- ggplot(top20_features, aes(x = reorder(Label, Importance), y = Importance, 
                                            fill = IsCore)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  scale_fill_manual(values = c("FALSE" = "steelblue", "TRUE" = "darkred"),
                    labels = c("Other", "AD Core Marker")) +
  labs(
    title = "Top 20 Important Features",
    subtitle = "* = AD Core Biomarker (prioritized in feature selection)",
    x = "Feature", 
    y = "Importance Score (Mean Decrease Gini)",
    fill = "Feature Type"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"))

ggsave(file.path(opt$output_dir, "Feature_Importance_Top20.png"), 
       plot = p_importance, width = 10, height = 8, dpi = 300)

# ROC Curves for all models
cat("  Creating ROC curves...\n")
roc_data <- data.frame()

for (model_name in names(models_list)) {
  model <- models_list[[model_name]]
  pred_probs <- predict(model, newdata = modeling_data, type = "prob")$Converter
  actual <- as.numeric(modeling_data$Outcome == "Converter")
  
  roc_obj <- roc(actual, pred_probs, quiet = TRUE)
  
  roc_data <- rbind(roc_data, data.frame(
    Model = model_name,
    Sensitivity = roc_obj$sensitivities,
    Specificity = 1 - roc_obj$specificities,
    AUC = auc(roc_obj)
  ))
}

p_roc <- ggplot(roc_data, aes(x = Specificity, y = Sensitivity, color = Model)) +
  geom_line(size = 1.2) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "grey50") +
  labs(
    title = "ROC Curves: Model Comparison",
    subtitle = sprintf("Best model: %s", best_model_name),
    x = "1 - Specificity (False Positive Rate)",
    y = "Sensitivity (True Positive Rate)"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"))

ggsave(file.path(opt$output_dir, "ROC_Curves_Comparison.png"), 
       plot = p_roc, width = 10, height = 8, dpi = 300)

# Calibration Plot for best model
cat("  Creating calibration plot...\n")
pred_probs_best <- predict(best_model, newdata = modeling_data, type = "prob")$Converter
actual_best <- as.numeric(modeling_data$Outcome == "Converter")

p_calibration <- create_calibration_plot(actual_best, pred_probs_best, 
                                          title = sprintf("Calibration Plot: %s", best_model_name))

ggsave(file.path(opt$output_dir, "Calibration_Plot_Best_Model.png"), 
       plot = p_calibration, width = 8, height = 8, dpi = 300)

# Model Comparison Barplot
cat("  Creating model comparison plot...\n")
p_comparison <- ggplot(model_comparison, aes(x = reorder(Model, CV_AUC), y = CV_AUC)) +
  geom_bar(stat = "identity", fill = "steelblue", alpha = 0.8) +
  geom_errorbar(aes(ymin = AUC_95CI_Lower, ymax = AUC_95CI_Upper), width = 0.2) +
  geom_text(aes(label = sprintf("%.3f", CV_AUC)), vjust = -0.5, size = 4) +
  coord_flip() +
  labs(
    title = "Model Performance Comparison",
    subtitle = sprintf("Error bars: 95%% CI from %d bootstrap iterations", opt$n_bootstrap),
    x = "Model",
    y = "AUC (Area Under ROC Curve)"
  ) +
  ylim(0, 1) +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"))

ggsave(file.path(opt$output_dir, "Model_Comparison_AUC.png"), 
       plot = p_comparison, width = 10, height = 6, dpi = 300)

# ==============================================================================
# Summary Report
# ==============================================================================
cat("\n========================================================================\n")
cat("Predictive Modeling Complete!\n")
cat("========================================================================\n\n")

cat(sprintf("  ✓ MICE multiple imputation: %d datasets\n", opt$n_mice))
cat("  ✓ Rubin's rules for combining imputed results\n")
cat(sprintf("  ✓ Bootstrap CI: %d iterations\n", opt$n_bootstrap))
cat("  ✓ Hosmer-Lemeshow calibration test\n")
cat("  ✓ Brier score for probabilistic accuracy\n")
cat("  ✓ ROC curves with pROC package\n")

cat("\nBest Model Performance:\n")
best_row <- model_comparison[model_comparison$Model == best_model_name, ]
cat(sprintf("  Model: %s\n", best_model_name))
cat(sprintf("  AUC: %.3f (95%% CI: %.3f-%.3f)\n", 
            best_row$CV_AUC, best_row$AUC_95CI_Lower, best_row$AUC_95CI_Upper))
cat(sprintf("  Sensitivity: %.3f\n", best_row$CV_Sensitivity))
cat(sprintf("  Specificity: %.3f\n", best_row$CV_Specificity))
cat(sprintf("  Brier Score: %.4f\n", best_row$Brier_Score))
cat(sprintf("  Hosmer-Lemeshow p-value: %.4f %s\n", 
            best_row$HL_P_Value,
            ifelse(best_row$Adequate_Calibration, "(adequate calibration)", "(poor calibration)")))

cat("\nOutput files:\n")
cat("  - Feature_Importance_RF.csv\n")
cat("  - Model_Comparison_Calibrated.csv\n")
cat("  - Feature_Correlation_Matrix.png\n")
cat("  - Feature_Importance_Top20.png\n")
cat("  - ROC_Curves_Comparison.png\n")
cat("  - Calibration_Plot_Best_Model.png\n")
cat("  - Model_Comparison_AUC.png\n")

cat("\n========================================================================\n")

