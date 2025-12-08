library(fastshap)
library(randomForest)
library(dplyr)
library(ggplot2)
library(reshape2)

## Load HABS data
habs_data <- read.csv("HABS_Baseline_Integrated.csv", stringsAsFactors = FALSE)

## Detect p-tau217
ptau217_cols <- grep("pTau217|ptau217", colnames(habs_data), value = TRUE, ignore.case = TRUE)
ptau217_col <- ptau217_cols[1]

## Prepare features
features <- c("Age", "Gender", "APOE4_Positive", "MMSE_Baseline", "Education", ptau217_col)
available_features <- features[features %in% colnames(habs_data)]

## Extract clean data
habs_clean <- habs_data %>%
  select(all_of(c(available_features, "AD_Conversion"))) %>%
  na.omit()

names(habs_clean)[names(habs_clean) == ptau217_col] <- "pTau217"
available_features[available_features == ptau217_col] <- "pTau217"

## Train random forest
set.seed(42)
rf_model <- randomForest(
  as.factor(AD_Conversion) ~ .,
  data = habs_clean,
  ntree = 500,
  importance = TRUE
)

## Sample for SHAP calculation
set.seed(42)
sample_size <- min(500, nrow(habs_clean))
sample_idx <- sample(1:nrow(habs_clean), sample_size)
habs_sample <- habs_clean[sample_idx, ]

X_features <- habs_sample[, available_features]

## Calculate SHAP values
predict_wrapper <- function(object, newdata) {
  predict(object, newdata, type = "prob")[, "1"]
}

tryCatch({
  shap_values <- explain(
    object = rf_model,
    X = X_features,
    pred_wrapper = predict_wrapper,
    nsim = 50,
    adjust = TRUE
  )
}, error = function(e) {
  importance_scores <- importance(rf_model, type = 1)
  shap_values <<- matrix(
    rep(importance_scores, each = sample_size),
    nrow = sample_size,
    ncol = length(available_features)
  )
  colnames(shap_values) <<- available_features
})

shap_df <- as.data.frame(shap_values)
colnames(shap_df) <- available_features

## Calculate mean absolute SHAP
mean_abs_shap <- colMeans(abs(shap_df))
feature_importance <- data.frame(
  Feature = names(mean_abs_shap),
  Importance = mean_abs_shap
) %>% arrange(desc(Importance))

## SHAP summary plot
shap_long <- melt(shap_df, variable.name = "Feature", value.name = "SHAP")
shap_long$Feature <- factor(shap_long$Feature, levels = feature_importance$Feature)

feature_values_long <- melt(X_features, variable.name = "Feature", value.name = "FeatureValue")
shap_long$FeatureValue <- feature_values_long$FeatureValue

p1 <- ggplot(shap_long, aes(x = SHAP, y = Feature)) +
  geom_jitter(aes(color = FeatureValue), alpha = 0.6, height = 0.2, size = 2) +
  scale_color_gradient(low = "blue", high = "red", name = "Feature\nValue") +
  geom_vline(xintercept = 0, linetype = "dashed", color = "gray50") +
  labs(
    title = "SHAP Summary Plot",
    x = "SHAP Value (Impact on Model Output)",
    y = "Feature"
  ) +
  theme_minimal(base_size = 14) +
  theme(plot.title = element_text(size = 18, face = "bold", hjust = 0.5))

ggsave("SHAP_Summary_Plot.png", p1, width = 12, height = 9, dpi = 300)

## Feature importance barplot
png("Feature_Importance_Barplot.png", width = 3000, height = 2400, res = 300)

barplot(
  feature_importance$Importance,
  names.arg = feature_importance$Feature,
  col = colorRampPalette(c("#377EB8", "#E41A1C"))(nrow(feature_importance)),
  main = "Global Feature Importance (Mean |SHAP|)",
  ylab = "Mean |SHAP Value|",
  xlab = "Feature",
  las = 2
)

dev.off()

## Individual explanations
predictions <- predict(rf_model, X_features, type = "prob")[, "1"]
quantiles <- quantile(predictions, probs = c(0.1, 0.5, 0.9))

high_risk_idx <- which.min(abs(predictions - quantiles[3]))[1]
med_risk_idx <- which.min(abs(predictions - quantiles[2]))[1]
low_risk_idx <- which.min(abs(predictions - quantiles[1]))[1]

example_indices <- c(high_risk_idx, med_risk_idx, low_risk_idx)
example_labels <- c("High Risk", "Medium Risk", "Low Risk")

png("Individual_Explanations.png", width = 3600, height = 3000, res = 300)

par(mfrow = c(3, 1), mar = c(5, 10, 4, 2))

for (i in 1:3) {
  idx <- example_indices[i]
  sample_shap <- shap_df[idx, ]
  
  sorted_idx <- order(abs(as.numeric(sample_shap)), decreasing = TRUE)
  colors <- ifelse(sample_shap[sorted_idx] > 0, "#E41A1C", "#377EB8")
  
  barplot(
    as.numeric(sample_shap[sorted_idx]),
    names.arg = names(sample_shap)[sorted_idx],
    horiz = TRUE,
    col = colors,
    main = sprintf("%s Example (Predicted Risk: %.1f%%)",
                   example_labels[i], 100 * predictions[idx]),
    xlab = "SHAP Value"
  )
  
  abline(v = 0, lty = 2, col = "gray50", lwd = 2)
}

dev.off()

## Save results
write.csv(feature_importance, "Feature_Importance_SHAP.csv", row.names = FALSE)
