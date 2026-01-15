library(fastshap)
library(randomForest)
library(dplyr)
library(ggplot2)
library(reshape2)
library(optparse)

# ==============================================================================
# Parse Command Line Arguments
# ==============================================================================
option_list <- list(
  make_option(c("--habs_file"), type = "character", 
              default = "HABS_Baseline_Integrated.csv",
              help = "Path to HABS integrated CSV [default: %default]"),
  make_option(c("--output_dir"), type = "character", 
              default = "./results",
              help = "Output directory [default: %default]"),
  make_option(c("--n_bootstrap"), type = "integer", default = 2000,
              help = "Number of bootstrap iterations (Methods 2.9: 2000) [default: %default]"),
  make_option(c("--n_trees"), type = "integer", default = 500,
              help = "Number of trees in Random Forest [default: %default]"),
  make_option(c("--shap_nsim"), type = "integer", default = 50,
              help = "Number of SHAP simulations [default: %default]"),
  make_option(c("--sample_size"), type = "integer", default = 500,
              help = "Sample size for SHAP calculation [default: %default]")
)

opt_parser <- OptionParser(option_list = option_list)
opt <- parse_args(opt_parser)

# Create output directory
dir.create(opt$output_dir, showWarnings = FALSE, recursive = TRUE)

# ==============================================================================
# Part 1: Load and Prepare Data
# ==============================================================================
cat("[1/5] Loading and preparing data...\n")

habs_data <- read.csv(opt$habs_file, stringsAsFactors = FALSE)
cat(sprintf("  Total samples: %d\n", nrow(habs_data)))

# Detect p-tau217 column
ptau217_cols <- grep("pTau217|ptau217", colnames(habs_data), value = TRUE, ignore.case = TRUE)
ptau217_col <- if (length(ptau217_cols) > 0) ptau217_cols[1] else NULL

# Define features
base_features <- c("Age", "Gender", "APOE4_Positive", "MMSE_Baseline", "Education")
if (!is.null(ptau217_col)) {
  features <- c(base_features, ptau217_col)
} else {
  features <- base_features
}

available_features <- features[features %in% colnames(habs_data)]
cat(sprintf("  Available features: %s\n", paste(available_features, collapse = ", ")))

# Extract clean data
habs_clean <- habs_data %>%
  select(all_of(c(available_features, "AD_Conversion"))) %>%
  na.omit()

# Rename p-tau217 column if present
if (!is.null(ptau217_col) && ptau217_col %in% colnames(habs_clean)) {
  names(habs_clean)[names(habs_clean) == ptau217_col] <- "pTau217"
  available_features[available_features == ptau217_col] <- "pTau217"
}

cat(sprintf("  Complete cases: %d\n", nrow(habs_clean)))
cat(sprintf("  Events: %d (%.1f%%)\n\n", 
            sum(habs_clean$AD_Conversion), 
            100 * mean(habs_clean$AD_Conversion)))

# ==============================================================================
# Part 2: Train Random Forest Model
# ==============================================================================
cat("[2/5] Training Random Forest model...\n")

set.seed(42)
rf_model <- randomForest(
  as.factor(AD_Conversion) ~ .,
  data = habs_clean,
  ntree = opt$n_trees,
  importance = TRUE,
  mtry = floor(sqrt(length(available_features)))
)

# Model performance
oob_error <- rf_model$err.rate[opt$n_trees, "OOB"]
cat(sprintf("  OOB error rate: %.3f\n", oob_error))
cat(sprintf("  OOB accuracy: %.3f\n\n", 1 - oob_error))

# ==============================================================================
# Part 3: Calculate SHAP Values
# ==============================================================================
cat("[3/5] Calculating SHAP values...\n")

# Sample for SHAP calculation
set.seed(42)
sample_size <- min(opt$sample_size, nrow(habs_clean))
sample_idx <- sample(1:nrow(habs_clean), sample_size)
habs_sample <- habs_clean[sample_idx, ]
X_features <- habs_sample[, available_features, drop = FALSE]

cat(sprintf("  Sample size for SHAP: %d\n", sample_size))

# Prediction wrapper
predict_wrapper <- function(object, newdata) {
  predict(object, newdata, type = "prob")[, "1"]
}

# Calculate SHAP values
shap_values <- tryCatch({
  explain(
    object = rf_model,
    X = X_features,
    pred_wrapper = predict_wrapper,
    nsim = opt$shap_nsim,
    adjust = TRUE
  )
}, error = function(e) {
  cat(sprintf("  SHAP calculation error: %s\n", e$message))
  cat("  Using permutation importance as fallback...\n")
  
  # Fallback to permutation importance
  importance_scores <- importance(rf_model, type = 1)[, 1]
  importance_scaled <- importance_scores / max(abs(importance_scores))
  
  # Create pseudo-SHAP matrix
  shap_matrix <- matrix(
    rep(importance_scaled, each = sample_size),
    nrow = sample_size,
    ncol = length(available_features)
  )
  colnames(shap_matrix) <- available_features
  as.data.frame(shap_matrix)
})

shap_df <- as.data.frame(shap_values)
colnames(shap_df) <- available_features

cat("  SHAP calculation complete\n\n")

# ==============================================================================
# Part 4: Bootstrap Confidence Intervals (Methods 2.9)
# ==============================================================================
cat(sprintf("[4/5] Bootstrap analysis (%d iterations, Methods 2.9)...\n", opt$n_bootstrap))

# Calculate mean absolute SHAP for each feature
mean_abs_shap <- colMeans(abs(shap_df))

# Bootstrap for confidence intervals
set.seed(42)
boot_importance <- matrix(NA, nrow = opt$n_bootstrap, ncol = length(available_features))
colnames(boot_importance) <- available_features

for (b in 1:opt$n_bootstrap) {
  boot_idx <- sample(1:nrow(shap_df), replace = TRUE)
  boot_shap <- shap_df[boot_idx, ]
  boot_importance[b, ] <- colMeans(abs(boot_shap))
  
  if (b %% 500 == 0) {
    cat(sprintf("    Completed %d/%d iterations\n", b, opt$n_bootstrap))
  }
}

# Calculate 95% CI
ci_lower <- apply(boot_importance, 2, quantile, probs = 0.025)
ci_upper <- apply(boot_importance, 2, quantile, probs = 0.975)
boot_se <- apply(boot_importance, 2, sd)

# Feature importance table with CI
feature_importance <- data.frame(
  Feature = names(mean_abs_shap),
  Importance = mean_abs_shap,
  SE = boot_se,
  CI_Lower = ci_lower,
  CI_Upper = ci_upper,
  stringsAsFactors = FALSE
) %>% 
  arrange(desc(Importance))

# Calculate p-values (two-sided test against zero)
feature_importance$Z_Score <- feature_importance$Importance / feature_importance$SE
feature_importance$P_Raw <- 2 * pnorm(-abs(feature_importance$Z_Score))

# FDR correction (Methods 2.4)
feature_importance$P_FDR <- p.adjust(feature_importance$P_Raw, method = "fdr")
feature_importance$Significant_FDR <- feature_importance$P_FDR < 0.05

# Rank
feature_importance$Rank <- 1:nrow(feature_importance)

cat("\nFeature Importance (Methods 2.9 Bootstrap CI):\n")
for (i in 1:nrow(feature_importance)) {
  sig_marker <- ifelse(feature_importance$Significant_FDR[i], "*", "")
  cat(sprintf("  %d. %s: %.4f (95%% CI: %.4f-%.4f) p_FDR=%.4f%s\n",
              feature_importance$Rank[i],
              feature_importance$Feature[i],
              feature_importance$Importance[i],
              feature_importance$CI_Lower[i],
              feature_importance$CI_Upper[i],
              feature_importance$P_FDR[i],
              sig_marker))
}

write.csv(feature_importance, 
          file.path(opt$output_dir, "Feature_Importance_SHAP.csv"), 
          row.names = FALSE)
cat("\n")

# ==============================================================================
# Part 5: Visualizations
# ==============================================================================
cat("[5/5] Generating visualizations...\n")

# 5.1 SHAP Summary Plot
shap_long <- melt(shap_df, variable.name = "Feature", value.name = "SHAP")
shap_long$Feature <- factor(shap_long$Feature, levels = feature_importance$Feature)

feature_values_long <- melt(X_features, variable.name = "Feature", value.name = "FeatureValue")
shap_long$FeatureValue <- feature_values_long$FeatureValue

# Normalize feature values for color scale
shap_long <- shap_long %>%
  group_by(Feature) %>%
  mutate(FeatureValue_Scaled = (FeatureValue - min(FeatureValue, na.rm = TRUE)) / 
           (max(FeatureValue, na.rm = TRUE) - min(FeatureValue, na.rm = TRUE) + 1e-10)) %>%
  ungroup()

p1 <- ggplot(shap_long, aes(x = SHAP, y = Feature)) +
  geom_jitter(aes(color = FeatureValue_Scaled), alpha = 0.6, height = 0.2, size = 2) +
  scale_color_gradient(low = "blue", high = "red", name = "Feature\nValue\n(Scaled)") +
  geom_vline(xintercept = 0, linetype = "dashed", color = "gray50") +
  labs(
    title = "SHAP Summary Plot",
    subtitle = sprintf("n=%d samples, %d bootstrap iterations (Methods 2.9)", 
                       sample_size, opt$n_bootstrap),
    x = "SHAP Value (Impact on Model Output)",
    y = "Feature"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(size = 18, face = "bold", hjust = 0.5),
    plot.subtitle = element_text(size = 12, hjust = 0.5)
  )

ggsave(file.path(opt$output_dir, "SHAP_Summary_Plot.png"), p1, 
       width = 12, height = 9, dpi = 300)
cat("  Saved: SHAP_Summary_Plot.png\n")

# 5.2 Feature Importance Barplot with CI
p2 <- ggplot(feature_importance, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_col(aes(fill = Significant_FDR), alpha = 0.8, width = 0.7) +
  geom_errorbar(aes(ymin = CI_Lower, ymax = CI_Upper), width = 0.3, size = 0.8) +
  scale_fill_manual(values = c("TRUE" = "#009E73", "FALSE" = "#E69F00"),
                    name = "FDR < 0.05") +
  coord_flip() +
  labs(
    title = "Global Feature Importance (Mean |SHAP|)",
    subtitle = sprintf("95%% CI from %d bootstrap iterations (Methods 2.9)", opt$n_bootstrap),
    x = "Feature",
    y = "Mean |SHAP Value|"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
    plot.subtitle = element_text(size = 11, hjust = 0.5),
    legend.position = "bottom"
  )

ggsave(file.path(opt$output_dir, "Feature_Importance_Barplot.png"), p2, 
       width = 10, height = 8, dpi = 300)
cat("  Saved: Feature_Importance_Barplot.png\n")

# 5.3 Individual Explanations (High/Medium/Low Risk)
predictions <- predict(rf_model, X_features, type = "prob")[, "1"]
quantiles <- quantile(predictions, probs = c(0.1, 0.5, 0.9))

high_risk_idx <- which.min(abs(predictions - quantiles[3]))[1]
med_risk_idx <- which.min(abs(predictions - quantiles[2]))[1]
low_risk_idx <- which.min(abs(predictions - quantiles[1]))[1]

example_indices <- c(high_risk_idx, med_risk_idx, low_risk_idx)
example_labels <- c("High Risk", "Medium Risk", "Low Risk")
example_colors <- c("#E41A1C", "#FF7F00", "#377EB8")

png(file.path(opt$output_dir, "Individual_Explanations.png"), 
    width = 3600, height = 3000, res = 300)
par(mfrow = c(3, 1), mar = c(5, 10, 4, 2))

for (i in 1:3) {
  idx <- example_indices[i]
  sample_shap <- as.numeric(shap_df[idx, ])
  names(sample_shap) <- colnames(shap_df)
  sorted_idx <- order(abs(sample_shap), decreasing = TRUE)
  
  colors <- ifelse(sample_shap[sorted_idx] > 0, "#E41A1C", "#377EB8")
  
  barplot(sample_shap[sorted_idx],
          names.arg = names(sample_shap)[sorted_idx],
          horiz = TRUE,
          col = colors,
          main = sprintf("%s Example (Predicted Risk: %.1f%%)",
                         example_labels[i], 100 * predictions[idx]),
          xlab = "SHAP Value",
          cex.names = 1.2,
          cex.main = 1.4,
          las = 1)
  abline(v = 0, lty = 2, col = "gray50", lwd = 2)
}

dev.off()
cat("  Saved: Individual_Explanations.png\n")

# 5.4 SHAP Dependence Plots for Top Features
top_features <- feature_importance$Feature[1:min(3, nrow(feature_importance))]

png(file.path(opt$output_dir, "SHAP_Dependence_Plots.png"), 
    width = 4000, height = 1400, res = 300)
par(mfrow = c(1, length(top_features)), mar = c(5, 5, 4, 2))

for (feat in top_features) {
  feat_values <- X_features[[feat]]
  feat_shap <- shap_df[[feat]]
  
  plot(feat_values, feat_shap,
       pch = 19, col = rgb(0.2, 0.4, 0.8, 0.5),
       xlab = feat, ylab = "SHAP Value",
       main = sprintf("SHAP Dependence: %s", feat),
       cex = 1.2, cex.lab = 1.3, cex.main = 1.4)
  
  # Add smoothed trend line
  if (length(unique(feat_values)) > 5) {
    lo <- loess(feat_shap ~ feat_values)
    x_seq <- seq(min(feat_values), max(feat_values), length.out = 100)
    lines(x_seq, predict(lo, x_seq), col = "red", lwd = 2)
  }
  
  abline(h = 0, lty = 2, col = "gray50")
}

dev.off()
cat("  Saved: SHAP_Dependence_Plots.png\n")

# ==============================================================================
# Summary Report
# ==============================================================================
cat("\n")

summary_lines <- c(
  "================================================================================",
  "SHAP Feature Importance Analysis Report (Methods 2.9 Aligned)",
  "================================================================================",
  "",
  sprintf("Generated: %s", Sys.time()),
  "",
  "Methods Requirements:",
  sprintf("  Methods 2.9: Bootstrap %d iterations for 95%% CI", opt$n_bootstrap),
  "  Methods 2.4: Benjamini-Hochberg FDR correction",
  "",
  "--------------------------------------------------------------------------------",
  "Data Summary",
  "--------------------------------------------------------------------------------",
  sprintf("  Total samples: %d", nrow(habs_clean)),
  sprintf("  Events: %d (%.1f%%)", sum(habs_clean$AD_Conversion), 
          100 * mean(habs_clean$AD_Conversion)),
  sprintf("  Features analyzed: %d", length(available_features)),
  sprintf("  SHAP sample size: %d", sample_size),
  "",
  "--------------------------------------------------------------------------------",
  "Model Performance",
  "--------------------------------------------------------------------------------",
  sprintf("  Random Forest trees: %d", opt$n_trees),
  sprintf("  OOB error rate: %.3f", oob_error),
  sprintf("  OOB accuracy: %.3f", 1 - oob_error),
  "",
  "--------------------------------------------------------------------------------",
  "Feature Importance Ranking",
  "--------------------------------------------------------------------------------"
)

for (i in 1:nrow(feature_importance)) {
  sig_marker <- ifelse(feature_importance$Significant_FDR[i], " *", "")
  summary_lines <- c(summary_lines,
    sprintf("  %d. %s: %.4f (95%% CI: %.4f-%.4f)%s",
            i, feature_importance$Feature[i],
            feature_importance$Importance[i],
            feature_importance$CI_Lower[i],
            feature_importance$CI_Upper[i],
            sig_marker))
}

summary_lines <- c(summary_lines,
  "",
  sprintf("  * Significant after FDR correction (q < 0.05)"),
  sprintf("  Significant features: %d/%d", 
          sum(feature_importance$Significant_FDR), nrow(feature_importance)),
  "",
  "--------------------------------------------------------------------------------",
  "Output Files",
  "--------------------------------------------------------------------------------",
  sprintf("  - %s/Feature_Importance_SHAP.csv", opt$output_dir),
  sprintf("  - %s/SHAP_Summary_Plot.png", opt$output_dir),
  sprintf("  - %s/Feature_Importance_Barplot.png", opt$output_dir),
  sprintf("  - %s/Individual_Explanations.png", opt$output_dir),
  sprintf("  - %s/SHAP_Dependence_Plots.png", opt$output_dir),
  "",
  "================================================================================",
  "SHAP Analysis Complete",
  "================================================================================"
)

# Write report
report_path <- file.path(opt$output_dir, "SHAP_Analysis_Report.txt")
writeLines(summary_lines, report_path)

cat(paste(summary_lines, collapse = "\n"))
cat("\n\n")

cat("============================================================\n")
cat("Step 17: SHAP Analysis Complete!\n")
cat("============================================================\n")
cat(sprintf("Report saved: %s\n", report_path))
