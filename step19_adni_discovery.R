library(dplyr)
library(ggplot2)
library(survival)
library(survminer)
library(pheatmap)

## Load data
adni_data <- read.csv("Cohort_A_Integrated.csv", stringsAsFactors = FALSE)
cluster_results <- read.csv("Final_Consensus_Clusters_K3.csv", stringsAsFactors = FALSE)

## Detect cluster column
cluster_col <- NULL
for (col in c("Consensus_Cluster_K3", "Cluster_Labels", "Cluster")) {
  if (col %in% colnames(cluster_results)) {
    cluster_col <- col
    break
  }
}

## Merge data
if ("Sample_Index" %in% colnames(cluster_results)) {
  adni_data$Subtype <- cluster_results[[cluster_col]][match(1:nrow(adni_data),
                                                            cluster_results$Sample_Index)]
} else {
  adni_data$Subtype <- cluster_results[[cluster_col]]
}

adni_labeled <- adni_data %>%
  filter(!is.na(Subtype)) %>%
  mutate(Subtype = factor(Subtype))

## Feature groups
feature_groups <- list(
  Demographics = c("Age", "Gender", "Education"),
  Cognition = c("MMSE_Baseline", "ADAS13", "CDRSB"),
  Genetics = c("APOE4_Positive", "APOE4_Copies"),
  CSF = c("ABETA42", "TAU_TOTAL", "PTAU181"),
  MRI = grep("^ST\\d+", colnames(adni_labeled), value = TRUE)
)

## Statistical tests
all_results <- data.frame()

for (group_name in names(feature_groups)) {
  features <- feature_groups[[group_name]][feature_groups[[group_name]] %in% colnames(adni_labeled)]
  
  for (var in features) {
    if (mean(is.na(adni_labeled[[var]])) > 0.5) next
    
    if (is.numeric(adni_labeled[[var]])) {
      anova_result <- aov(as.formula(paste(var, "~ Subtype")), data = adni_labeled)
      p_value <- summary(anova_result)[[1]][["Pr(>F)"]][1]
      test_type <- "ANOVA"
    } else {
      contingency_table <- table(adni_labeled$Subtype, adni_labeled[[var]])
      chi_result <- chisq.test(contingency_table)
      p_value <- chi_result$p.value
      test_type <- "Chi-square"
    }
    
    all_results <- rbind(all_results, data.frame(
      Feature_Group = group_name,
      Feature = var,
      Test = test_type,
      P_Value = p_value,
      Significant = ifelse(p_value < 0.05, "Yes", "No")
    ))
  }
}

write.csv(all_results, "Feature_Differences.csv", row.names = FALSE)

## Survival analysis
if ("AD_Conversion" %in% colnames(adni_labeled)) {
  
  adni_survival <- adni_labeled %>%
    select(ID, Subtype, AD_Conversion, Age, Gender, APOE4_Positive, MMSE_Baseline) %>%
    filter(!is.na(AD_Conversion))
  
  if ("Followup_Time" %in% colnames(adni_labeled)) {
    adni_survival$Followup_Years <- adni_labeled$Followup_Time[match(adni_survival$ID, adni_labeled$ID)]
  } else {
    adni_survival$Followup_Years <- 1
  }
  
  adni_survival <- adni_survival %>% filter(!is.na(Followup_Years), Followup_Years > 0)
  
  ## Kaplan-Meier
  surv_obj <- Surv(time = adni_survival$Followup_Years, event = adni_survival$AD_Conversion)
  km_fit <- survfit(surv_obj ~ Subtype, data = adni_survival)
  
  logrank_test <- survdiff(surv_obj ~ Subtype, data = adni_survival)
  logrank_p <- 1 - pchisq(logrank_test$chisq, df = length(unique(adni_survival$Subtype)) - 1)
  
  ## KM plot
  km_plot <- ggsurvplot(
    km_fit,
    data = adni_survival,
    pval = TRUE,
    conf.int = TRUE,
    risk.table = TRUE,
    palette = c("#E41A1C", "#377EB8", "#4DAF4A"),
    title = "Kaplan-Meier Survival Curves",
    xlab = "Time (Years)",
    ylab = "Survival Probability"
  )
  
  ggsave("KM_Curves.png", km_plot$plot, width = 12, height = 10, dpi = 300)
  
  ## Cox regression
  adni_survival$Subtype_Factor <- relevel(factor(adni_survival$Subtype), ref = "1")
  
  cox_model <- coxph(
    surv_obj ~ Subtype_Factor + Age + Gender + APOE4_Positive + MMSE_Baseline,
    data = adni_survival
  )
  
  cox_summary <- summary(cox_model)
  cox_results <- data.frame(
    Variable = rownames(cox_summary$conf.int),
    HR = cox_summary$conf.int[, "exp(coef)"],
    HR_Lower = cox_summary$conf.int[, "lower .95"],
    HR_Upper = cox_summary$conf.int[, "upper .95"],
    P_Value = cox_summary$coefficients[, "Pr(>|z|)"]
  )
  
  write.csv(cox_results, "Cox_Results.csv", row.names = FALSE)
}

## Save labeled data
write.csv(adni_labeled, "ADNI_Labeled_For_Classifier.csv", row.names = FALSE)
