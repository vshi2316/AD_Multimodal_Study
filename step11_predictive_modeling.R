library(randomForest)
library(pROC)
library(ggplot2)
library(dplyr)
library(caret)
library(mice)
library(glmnet)
library(xgboost)
library(DMwR)
library(corrplot)

## Load outcome data
cluster_results <- read.csv("cluster_results.csv", stringsAsFactors = FALSE)

response <- cluster_results %>%
  select(ID, AD_Conversion) %>%
  mutate(AD_Conversion = factor(AD_Conversion, levels = c(0, 1),
                                labels = c("NonConverter", "Converter")))

imbalance_ratio <- sum(response$AD_Conversion == "Converter") / sum(response$AD_Conversion == "NonConverter")

## Load significant features
smd_features <- data.frame()
conv_features <- data.frame()

if (file.exists("SMD_Significant_Features.csv")) {
  smd_features <- read.csv("SMD_Significant_Features.csv", stringsAsFactors = FALSE)
}

if (file.exists("Conversion_Differential_Significant_Enhanced.csv")) {
  conv_features <- read.csv("Conversion_Differential_Significant_Enhanced.csv", stringsAsFactors = FALSE)
} else if (file.exists("Conversion_Differential_Significant.csv")) {
  conv_features <- read.csv("Conversion_Differential_Significant.csv", stringsAsFactors = FALSE)
}

## Merge features
if (nrow(smd_features) > 0 && nrow(conv_features) > 0) {
  selected_features <- unique(c(smd_features$Feature, conv_features$Feature))
} else if (nrow(conv_features) > 0) {
  selected_features <- unique(conv_features$Feature)
} else if (nrow(smd_features) > 0) {
  selected_features <- unique(smd_features$Feature)
} else {
  selected_features <- NULL
}

## Load multimodal data
clinical <- read.csv("Clinical_data.csv", stringsAsFactors = FALSE)
smri <- read.csv("RNA_plasma.csv", stringsAsFactors = FALSE)
csf <- read.csv("metabolites.csv", stringsAsFactors = FALSE)

clinical_cols_all <- setdiff(colnames(clinical), "ID")
smri_cols_all <- setdiff(colnames(smri), "ID")
csf_cols_all <- setdiff(colnames(csf), "ID")

## Merge all data
all_data <- response %>%
  left_join(clinical, by = "ID") %>%
  left_join(smri, by = "ID") %>%
  left_join(csf, by = "ID")

all_feature_cols <- c(clinical_cols_all, smri_cols_all, csf_cols_all)
all_feature_cols <- all_feature_cols[all_feature_cols %in% colnames(all_data)]

if (!is.null(selected_features) && length(selected_features) > 0) {
  feature_cols <- all_feature_cols[all_feature_cols %in% selected_features]
} else {
  feature_cols <- all_feature_cols
}

feature_matrix <- all_data[, feature_cols, drop = FALSE]

## MICE imputation
missing_rate <- colMeans(is.na(feature_matrix))
high_missing_features <- names(missing_rate)[missing_rate > 0.5]

if (length(high_missing_features) > 0) {
  feature_matrix <- feature_matrix[, !colnames(feature_matrix) %in% high_missing_features, drop = FALSE]
}

if (any(is.na(feature_matrix))) {
  imputed_data <- mice(feature_matrix, m = 5, method = "pmm", seed = 42, printFlag = FALSE)
  feature_matrix <- complete(imputed_data)
}

## Remove redundant features
cor_matrix <- cor(feature_matrix, use = "pairwise.complete.obs")

png("Feature_Correlation_Matrix.png", width = 4000, height = 4000, res = 300)
corrplot(cor_matrix, method = "color", type = "upper", tl.cex = 0.5, tl.col = "black")
dev.off()

high_cor_pairs <- which(abs(cor_matrix) > 0.8 & abs(cor_matrix) < 1, arr.ind = TRUE)
high_cor_pairs <- high_cor_pairs[high_cor_pairs[,1] < high_cor_pairs[,2], , drop = FALSE]

## Define AD core biomarkers
ad_core_patterns <- c("ABETA", "PTAU", "TAU", "ADAS", "MMSE", "FAQ", "APOE", "CDR",
                       "Hippocampus", "Entorhinal", "Amygdala", "Ventricle", "Temporal")

is_core_feature <- function(feature_name) {
  any(sapply(ad_core_patterns, function(pattern) {
    grepl(pattern, feature_name, ignore.case = TRUE)
  }))
}

## Remove redundant features (keep core markers)
features_to_remove <- c()
if (nrow(high_cor_pairs) > 0) {
  for (i in 1:nrow(high_cor_pairs)) {
    feat1 <- rownames(cor_matrix)[high_cor_pairs[i, 1]]
    feat2 <- rownames(cor_matrix)[high_cor_pairs[i, 2]]
    
    is_feat1_core <- is_core_feature(feat1)
    is_feat2_core <- is_core_feature(feat2)
    
    if (is_feat1_core && !is_feat2_core) {
      features_to_remove <- c(features_to_remove, feat2)
    } else if (!is_feat1_core && is_feat2_core) {
      features_to_remove <- c(features_to_remove, feat1)
    } else {
      if (var(feature_matrix[, feat1]) < var(feature_matrix[, feat2])) {
        features_to_remove <- c(features_to_remove, feat1)
      } else {
        features_to_remove <- c(features_to_remove, feat2)
      }
    }
  }
  
  features_to_remove <- unique(features_to_remove)
  if (length(features_to_remove) > 0) {
    feature_matrix <- feature_matrix[, !colnames(feature_matrix) %in% features_to_remove, drop = FALSE]
  }
}

## Algorithmic feature selection
set.seed(42)
rf_temp <- randomForest(x = feature_matrix, y = all_data$AD_Conversion, ntree = 500, importance = TRUE)

importance_scores <- importance(rf_temp)[, "MeanDecreaseGini"]
importance_df <- data.frame(
  Feature = names(importance_scores),
  Importance = importance_scores,
  IsCore = sapply(names(importance_scores), is_core_feature)
) %>% arrange(desc(Importance))

write.csv(importance_df, "Feature_Importance_RF_Full.csv", row.names = FALSE)

n_features_to_keep <- min(50, max(15, floor(ncol(feature_matrix) * 0.6)))
top_features_rf <- head(importance_df$Feature, n_features_to_keep)

core_features_in_data <- importance_df$Feature[importance_df$IsCore]
top_features_rf <- unique(c(top_features_rf, core_features_in_data))

## Lasso feature selection
x_matrix <- as.matrix(feature_matrix)
y_vector <- as.numeric(all_data$AD_Conversion) - 1

set.seed(42)
cv_lasso <- cv.glmnet(x_matrix, y_vector, family = "binomial", alpha = 1, nfolds = 10)

lasso_coef <- coef(cv_lasso, s = "lambda.1se")
selected_by_lasso <- rownames(lasso_coef)[lasso_coef[, 1] != 0]
selected_by_lasso <- selected_by_lasso[selected_by_lasso != "(Intercept)"]

final_features <- unique(c(top_features_rf, selected_by_lasso, core_features_in_data))
feature_matrix_final <- feature_matrix[, final_features, drop = FALSE]

## Prepare modeling data with SMOTE
modeling_data <- data.frame(Outcome = all_data$AD_Conversion, feature_matrix_final)

if (imbalance_ratio < 0.5) {
  set.seed(42)
  tryCatch({
    modeling_data <- SMOTE(Outcome ~ ., data = modeling_data, perc.over = 200, perc.under = 150)
  }, error = function(e) {})
}

## Train models
train_control <- trainControl(method = "cv", number = 10, summaryFunction = twoClassSummary,
                              classProbs = TRUE, savePredictions = "final")

## Random Forest
set.seed(42)
model_rf <- train(Outcome ~ ., data = modeling_data, method = "rf", trControl = train_control,
                  tuneGrid = expand.grid(mtry = c(2, 4, 6, 8, 10)), metric = "ROC", ntree = 500)

## Elastic Net
set.seed(42)
model_enet <- train(Outcome ~ ., data = modeling_data, method = "glmnet", trControl = train_control,
                    tuneGrid = expand.grid(alpha = seq(0, 1, by = 0.1), lambda = 10^seq(-4, 0, length = 30)),
                    metric = "ROC")

## SVM
set.seed(42)
model_svm <- train(Outcome ~ ., data = modeling_data, method = "svmRadial", trControl = train_control,
                   tuneGrid = expand.grid(sigma = c(0.005, 0.01, 0.05, 0.1, 0.2), C = c(0.5, 1, 2, 5, 10)),
                   metric = "ROC")

## XGBoost
set.seed(42)
suppressWarnings({
  model_xgb <- train(Outcome ~ ., data = modeling_data, method = "xgbTree", trControl = train_control,
                     tuneGrid = expand.grid(nrounds = c(50, 100, 150, 200), max_depth = c(3, 5, 7),
                                           eta = c(0.01, 0.05, 0.1), gamma = 0, colsample_bytree = 0.8,
                                           min_child_weight = 1, subsample = 0.8),
                     metric = "ROC", verbose = 0)
})

## Model comparison
models_list <- list("Random Forest" = model_rf, "Elastic Net" = model_enet,
                    "SVM-RBF" = model_svm, "XGBoost" = model_xgb)

model_comparison <- data.frame()

for (model_name in names(models_list)) {
  model <- models_list[[model_name]]
  best_result <- model$results[which.max(model$results$ROC), ]
  
  model_comparison <- rbind(model_comparison, data.frame(
    Model = model_name,
    CV_AUC = best_result$ROC,
    CV_Sensitivity = best_result$Sens,
    CV_Specificity = best_result$Spec
  ))
}

write.csv(model_comparison, "Model_Comparison.csv", row.names = FALSE)

## Select best model
best_model_name <- model_comparison$Model[which.max(model_comparison$CV_AUC)]

## Visualizations
png("Feature_Importance_Top20.png", width = 4000, height = 3200, res = 300)

top20_features <- head(importance_df, 20)
top20_features$Label <- ifelse(top20_features$IsCore,
                               paste0(top20_features$Feature, " *"),
                               as.character(top20_features$Feature))

ggplot(top20_features, aes(x = reorder(Label, Importance), y = Importance, alpha = IsCore)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  scale_alpha_manual(values = c("FALSE" = 0.7, "TRUE" = 1.0)) +
  labs(title = "Top 20 Important Features (* = AD Core Marker)",
       x = "Feature", y = "Importance Score") +
  theme_minimal()

dev.off()
