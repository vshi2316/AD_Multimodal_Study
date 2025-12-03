library(dplyr)
library(survival)
library(survminer)
library(randomForest)
library(mice)

set.seed(42)

## Load A4 baseline data
a4_baseline <- read.csv("A4_Baseline_Integrated.csv", stringsAsFactors = FALSE)

## Remove conflicting columns
conflicting_cols <- c("AD_Conversion", "Followup_Years", "Time_to_Event_Months")
a4_baseline <- a4_baseline %>% select(-any_of(conflicting_cols))

## Standardize column names
if ("Age" %in% colnames(a4_baseline)) {
  a4_baseline <- a4_baseline %>% rename(Age_Baseline = Age)
}
if ("APOE4_Carrier" %in% colnames(a4_baseline)) {
  a4_baseline <- a4_baseline %>% rename(APOE4_Positive = APOE4_Carrier)
}

## Load CDR longitudinal data
cdr_long <- read.csv("A4_CDR_Longitudinal.csv", stringsAsFactors = FALSE)

## Extract followup time
followup_summary <- cdr_long %>%
  mutate(
    BID = as.character(BID),
    Followup_Months = Week / 4.345,
    Is_Converted = ifelse(!is.na(CDGLOBAL) & CDGLOBAL >= 0.5, 1, 0)
  ) %>%
  group_by(BID) %>%
  summarise(
    AD_Conversion = ifelse(any(Is_Converted == 1), 1, 0),
    Time_to_Event_Months = ifelse(AD_Conversion == 1,
                                  min(Followup_Months[Is_Converted == 1]),
                                  max(Followup_Months)),
    Followup_Years = Time_to_Event_Months / 12,
    .groups = 'drop'
  ) %>%
  filter(Followup_Years > 0)

## Merge data
a4_data <- a4_baseline %>%
  mutate(BID = as.character(BID)) %>%
  inner_join(followup_summary, by = "BID")

## PRISMA sample flow
n_initial <- nrow(a4_data)
a4_analysis <- a4_data %>% filter(Followup_Years >= 0.5)
n_final <- nrow(a4_analysis)

## Load classifier
rf_classifier <- readRDS("ADNI_RF_Classifier.rds")
classifier_features <- rownames(rf_classifier$importance)

## Missing value imputation
features_to_impute <- classifier_features[classifier_features %in% colnames(a4_analysis)]

if (length(features_to_impute) > 0) {
  impute_data <- a4_analysis %>% select(BID, all_of(features_to_impute))
  
  imp <- mice(impute_data %>% select(-BID), m = 5, seed = 42, printFlag = FALSE)
  imputed_data <- complete(imp, 1)
  imputed_data$BID <- impute_data$BID
  
  a4_analysis <- a4_analysis %>%
    select(-all_of(features_to_impute)) %>%
    left_join(imputed_data, by = "BID")
}

## Predict subtypes
a4_predict <- a4_analysis %>% select(BID, all_of(classifier_features))

for (feat in classifier_features) {
  if (sum(is.na(a4_predict[[feat]])) > 0) {
    if (is.numeric(a4_predict[[feat]])) {
      a4_predict[[feat]][is.na(a4_predict[[feat]])] <- median(a4_predict[[feat]], na.rm = TRUE)
    }
  }
}

a4_predict$Predicted_Subtype <- predict(rf_classifier, newdata = a4_predict)

a4_with_subtype <- a4_analysis %>%
  left_join(a4_predict %>% select(BID, Predicted_Subtype), by = "BID")

write.csv(a4_with_subtype, "A4_Predicted_Subtypes.csv", row.names = FALSE)

## Survival analysis
survival_data <- a4_with_subtype %>%
  filter(!is.na(AD_Conversion), !is.na(Followup_Years), !is.na(Predicted_Subtype)) %>%
  mutate(Subtype = factor(Predicted_Subtype))

## Kaplan-Meier
km_fit <- survfit(Surv(Followup_Years, AD_Conversion) ~ Subtype, data = survival_data)

logrank_test <- survdiff(Surv(Followup_Years, AD_Conversion) ~ Subtype, data = survival_data)
logrank_p <- 1 - pchisq(logrank_test$chisq, length(unique(survival_data$Subtype)) - 1)

## KM plot
km_plot <- ggsurvplot(
  km_fit,
  data = survival_data,
  pval = TRUE,
  conf.int = TRUE,
  risk.table = TRUE,
  palette = c("#E41A1C", "#377EB8", "#4DAF4A"),
  title = sprintf("A4 Cohort (n=%d, %d events)", nrow(survival_data), sum(survival_data$AD_Conversion)),
  xlab = "Follow-up Time (Years)",
  ylab = "Event-Free Survival"
)

ggsave("A4_KM_Curves.png", km_plot$plot, width = 12, height = 10, dpi = 300)

## Cox regression
survival_data$Subtype <- relevel(survival_data$Subtype, ref = "1")

cox_multi <- coxph(
  Surv(Followup_Years, AD_Conversion) ~ Subtype + Age_Baseline + Gender + APOE4_Positive + MMSE_Baseline,
  data = survival_data
)

## Cox proportional hazards assumption test
ph_test <- cox.zph(cox_multi)

cox_summary <- summary(cox_multi)
cox_results <- data.frame(
  Variable = rownames(cox_summary$coefficients),
  HR = exp(cox_summary$coefficients[, "coef"]),
  HR_Lower = exp(cox_summary$coefficients[, "coef"] - 1.96 * cox_summary$coefficients[, "se(coef)"]),
  HR_Upper = exp(cox_summary$coefficients[, "coef"] + 1.96 * cox_summary$coefficients[, "se(coef)"]),
  P_Value = cox_summary$coefficients[, "Pr(>|z|)"]
)

write.csv(cox_results, "A4_Cox_Results.csv", row.names = FALSE)

## Statistical power
Z_alpha <- qnorm(0.975)
Z_beta <- qnorm(0.80)
HR_target <- 1.8

events_needed <- ceiling(4 * (Z_alpha + Z_beta)^2 / (log(HR_target))^2)
observed_events <- sum(survival_data$AD_Conversion)

power_table <- data.frame(
  Metric = c("Sample Size", "Event Count", "Required Events", "Observed Power"),
  Value = c(
    nrow(survival_data),
    observed_events,
    events_needed,
    ifelse(observed_events >= events_needed, "Adequate (>80%)", "Limited (<80%)")
  )
)

write.csv(power_table, "A4_Statistical_Power.csv", row.names = FALSE)
