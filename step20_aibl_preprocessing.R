library(dplyr)
library(lubridate)
library(survival)
library(survminer)
library(randomForest)

## Load AIBL baseline data
aibl_baseline <- read.csv("AIBL_Baseline_Integrated.csv", stringsAsFactors = FALSE)

## Remove conflicting columns
conflicting_cols <- c("AD_Conversion", "Followup_Years", "Time_to_Event_Months")
aibl_baseline <- aibl_baseline %>%
  select(-any_of(conflicting_cols))

## Standardize column names
if ("Age" %in% colnames(aibl_baseline)) {
  aibl_baseline <- aibl_baseline %>% rename(Age_Baseline = Age)
}
if ("APOE4_Carrier" %in% colnames(aibl_baseline)) {
  aibl_baseline <- aibl_baseline %>% rename(APOE4_Positive = APOE4_Carrier)
}

## Load longitudinal CDR data
cdr_long <- read.csv("AIBL_CDR_Longitudinal.csv", stringsAsFactors = FALSE)

## Generate followup summary
followup_summary <- cdr_long %>%
  mutate(
    RID = as.character(RID),
    Followup_Months = Week / 4.345,
    Is_Converted = ifelse(!is.na(CDGLOBAL) & CDGLOBAL >= 0.5, 1, 0)
  ) %>%
  group_by(RID) %>%
  summarise(
    AD_Conversion = ifelse(any(Is_Converted == 1), 1, 0),
    Time_to_Event_Months = ifelse(AD_Conversion == 1,
                                  min(Followup_Months[Is_Converted == 1]),
                                  max(Followup_Months)),
    Followup_Years = Time_to_Event_Months / 12,
    .groups = 'drop'
  ) %>%
  filter(Followup_Years > 0)

## Merge baseline + followup
aibl_data <- aibl_baseline %>%
  mutate(RID = as.character(RID)) %>%
  inner_join(followup_summary, by = "RID")

## PRISMA sample flow
n_initial <- nrow(aibl_data)
aibl_analysis <- aibl_data %>% filter(Followup_Years >= 0.5)
n_final <- nrow(aibl_analysis)

## Load ADNI classifier
rf_classifier <- readRDS("ADNI_RF_Classifier.rds")
classifier_features <- rownames(rf_classifier$importance)

## Predict subtypes
aibl_predict <- aibl_analysis %>%
  select(RID, all_of(classifier_features))

## Handle missing values
for (feat in classifier_features) {
  if (sum(is.na(aibl_predict[[feat]])) > 0) {
    if (is.numeric(aibl_predict[[feat]])) {
      aibl_predict[[feat]][is.na(aibl_predict[[feat]])] <- median(aibl_predict[[feat]], na.rm = TRUE)
    }
  }
}

aibl_predict$Predicted_Subtype <- predict(rf_classifier, newdata = aibl_predict)

aibl_with_subtype <- aibl_analysis %>%
  left_join(aibl_predict %>% select(RID, Predicted_Subtype), by = "RID")

write.csv(aibl_with_subtype, "AIBL_Predicted_Subtypes.csv", row.names = FALSE)

## Survival analysis
survival_data <- aibl_with_subtype %>%
  filter(!is.na(Followup_Years), !is.na(AD_Conversion), !is.na(Predicted_Subtype)) %>%
  mutate(Subtype = factor(Predicted_Subtype))

## Kaplan-Meier
km_fit <- survfit(Surv(Followup_Years, AD_Conversion) ~ Subtype, data = survival_data)

logrank_test <- survdiff(Surv(Followup_Years, AD_Conversion) ~ Subtype, data = survival_data)
p_logrank <- 1 - pchisq(logrank_test$chisq, length(unique(survival_data$Subtype)) - 1)

## KM plot
km_plot <- ggsurvplot(
  km_fit,
  data = survival_data,
  pval = TRUE,
  conf.int = TRUE,
  risk.table = TRUE,
  palette = c("#E41A1C", "#377EB8", "#4DAF4A"),
  title = sprintf("AIBL Cohort (n=%d, %d events)", nrow(survival_data), sum(survival_data$AD_Conversion)),
  xlab = "Time (Years)",
  ylab = "Event-Free Survival"
)

ggsave("AIBL_KM_Curves.png", km_plot$plot, width = 12, height = 10, dpi = 300)

## Cox regression
survival_data$Subtype <- relevel(survival_data$Subtype, ref = "1")

cox_model <- coxph(Surv(Followup_Years, AD_Conversion) ~ Subtype + Age_Baseline + Gender + APOE4_Positive,
                   data = survival_data)

cox_summary <- summary(cox_model)
cox_results <- data.frame(
  Variable = rownames(cox_summary$conf.int),
  HR = cox_summary$conf.int[, "exp(coef)"],
  CI_Lower = cox_summary$conf.int[, "lower .95"],
  CI_Upper = cox_summary$conf.int[, "upper .95"],
  P_value = cox_summary$coefficients[, "Pr(>|z|)"]
)

write.csv(cox_results, "AIBL_Cox_Results.csv", row.names = FALSE)
