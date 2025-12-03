library(pROC)
library(dplyr)
library(ggplot2)
library(logistf)
library(PRROC)

## Load HABS data
habs_data <- read.csv("HABS_Baseline_Integrated.csv", stringsAsFactors = FALSE)

## Detect p-tau217
ptau217_cols <- grep("pTau217|ptau217", colnames(habs_data), value = TRUE, ignore.case = TRUE)
ptau217_col <- ptau217_cols[1]

## Extract key variables
key_vars <- c("Age", "Gender", "APOE4_Positive", "MMSE_Baseline", ptau217_col, "AD_Conversion")
habs_subset <- habs_data %>% select(all_of(key_vars))
names(habs_subset)[names(habs_subset) == ptau217_col] <- "pTau217"

habs_complete <- na.omit(habs_subset)

## Split data
set.seed(42)
train_idx <- sample(1:nrow(habs_complete), size = 0.7 * nrow(habs_complete))
habs_train <- habs_complete[train_idx, ]
habs_test <- habs_complete[-train_idx, ]

## Fit models (Firth correction)
model_base <- logistf(AD_Conversion ~ Age + Gender + APOE4_Positive + MMSE_Baseline,
                      data = habs_train)
model_plasma <- logistf(AD_Conversion ~ Age + Gender + APOE4_Positive + MMSE_Baseline + pTau217,
                        data = habs_train)

pred_base <- predict(model_base, newdata = habs_test, type = "response")
pred_plasma <- predict(model_plasma, newdata = habs_test, type = "response")

## AUC analysis
roc_base <- roc(habs_test$AD_Conversion, pred_base, quiet = TRUE)
roc_plasma <- roc(habs_test$AD_Conversion, pred_plasma, quiet = TRUE)

auc_base <- auc(roc_base)
auc_plasma <- auc(roc_plasma)
ci_plasma <- ci.auc(roc_plasma)

roc_test <- roc.test(roc_base, roc_plasma, method = "delong")

## Define high-risk subgroups
habs_test$pred_plasma <- pred_plasma

habs_test <- habs_test %>%
  mutate(
    MCI_like = ifelse(MMSE_Baseline < 27, 1, 0),
    APOE4_Group = ifelse(APOE4_Positive == 1, 1, 0),
    Elderly = ifelse(Age > 70, 1, 0),
    High_Risk_Combined = ifelse(Elderly == 1 & APOE4_Group == 1, 1, 0)
  )

## Subgroup AUC calculation
calc_auc_subgroup <- function(data, outcome_col = "AD_Conversion", pred_col = "pred_plasma") {
  if (nrow(data) < 10 || sum(data[[outcome_col]]) < 5) {
    return(list(AUC = NA, CI_Lower = NA, CI_Upper = NA))
  }
  roc_obj <- roc(data[[outcome_col]], data[[pred_col]], quiet = TRUE)
  ci_obj <- ci.auc(roc_obj)
  return(list(AUC = as.numeric(roc_obj$auc), CI_Lower = ci_obj[1], CI_Upper = ci_obj[3]))
}

auc_subgroups <- data.frame(
  Subgroup = "Overall",
  AUC = auc_plasma,
  CI_Lower = ci_plasma[1],
  CI_Upper = ci_plasma[3]
)

## APOE4+ subgroup
apoe4_data <- habs_test %>% filter(APOE4_Group == 1)
if (nrow(apoe4_data) >= 10) {
  apoe4_auc <- calc_auc_subgroup(apoe4_data)
  auc_subgroups <- rbind(auc_subgroups, data.frame(
    Subgroup = "APOE4+",
    AUC = apoe4_auc$AUC,
    CI_Lower = apoe4_auc$CI_Lower,
    CI_Upper = apoe4_auc$CI_Upper
  ))
}

write.csv(auc_subgroups, "Subgroup_AUC_Results.csv", row.names = FALSE)

## AUPRC analysis
calc_auprc <- function(y_true, y_pred) {
  pr_obj <- pr.curve(scores.class0 = y_pred[y_true == 1],
                     scores.class1 = y_pred[y_true == 0],
                     curve = TRUE)
  return(list(AUPRC = pr_obj$auc.integral, PR_Curve = pr_obj$curve))
}

auprc_plasma <- calc_auprc(habs_test$AD_Conversion, pred_plasma)
baseline_auprc <- sum(habs_test$AD_Conversion) / nrow(habs_test)

## NRI analysis
threshold_sets <- c(0, 0.10, 0.20, 1)

calc_nri <- function(pred_old, pred_new, outcome, thresholds) {
  cat_old <- cut(pred_old, breaks = thresholds, include.lowest = TRUE)
  cat_new <- cut(pred_new, breaks = thresholds, include.lowest = TRUE)
  
  events_idx <- which(outcome == 1)
  nonevents_idx <- which(outcome == 0)
  
  if (length(events_idx) == 0 || length(nonevents_idx) == 0) {
    return(list(total_nri = NA))
  }
  
  events_table <- table(Old = cat_old[events_idx], New = cat_new[events_idx])
  nonevents_table <- table(Old = cat_old[nonevents_idx], New = cat_new[nonevents_idx])
  
  events_nri <- (sum(events_table[lower.tri(events_table)]) - 
                 sum(events_table[upper.tri(events_table)])) / length(events_idx)
  nonevents_nri <- (sum(nonevents_table[upper.tri(nonevents_table)]) - 
                    sum(nonevents_table[lower.tri(nonevents_table)])) / length(nonevents_idx)
  
  return(list(total_nri = events_nri + nonevents_nri))
}

nri_result <- calc_nri(pred_base, pred_plasma, habs_test$AD_Conversion, threshold_sets)

## Visualizations
p1 <- ggplot() +
  geom_line(aes(x = 1 - roc_base$specificities, y = roc_base$sensitivities, color = "Base"), size = 1) +
  geom_line(aes(x = 1 - roc_plasma$specificities, y = roc_plasma$sensitivities, color = "Plasma"), size = 1) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
  labs(title = "ROC Curve Comparison", x = "1 - Specificity", y = "Sensitivity") +
  scale_color_manual(values = c("Base" = "#E69F00", "Plasma" = "#009E73")) +
  theme_bw()

ggsave("ROC_Comparison.pdf", p1, width = 6, height = 6)

## Subgroup forest plot
p2 <- ggplot(auc_subgroups, aes(x = AUC, y = Subgroup)) +
  geom_point(size = 3, color = "#0072B2") +
  geom_errorbarh(aes(xmin = CI_Lower, xmax = CI_Upper), height = 0.2) +
  geom_vline(xintercept = 0.5, linetype = "dashed") +
  labs(title = "Subgroup AUC Analysis", x = "AUC (95% CI)") +
  theme_bw()

ggsave("Subgroup_AUC_Forest.pdf", p2, width = 8, height = 5)
