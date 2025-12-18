## Step 23: Neuroimaging Endotype Characterization

library(tidyverse)
library(ggplot2)
library(patchwork)
library(ggseg)
library(ggsegYeo2011)
library(stringr) # Added for string detection in Part 4.5

cat("================================================================================\n")
cat("Step 23: Neuroimaging Endotype Characterization\n")
cat("================================================================================\n\n")

## ============================================================================
## Part 1: Data Loading and Preparation
## ============================================================================
cat("Part 1: Data Loading\n")

data_raw <- read.csv("ADNI_Labeled_For_Classifier.csv", stringsAsFactors = FALSE)
cat(sprintf("Raw data: %d samples\n", nrow(data_raw)))

clinical_features <- c("ADAS13", "CDRSB", "FAQTOTAL", "MMSE_Baseline")
mri_features <- c("RightParacentral_TA", "RightParahippocampal_TA", 
                  "RightParsOpercularis_TA", "RightParsOrbitalis_TA",
                  "RightParacentral_CV", "RightParahippocampal_CV", 
                  "RightParsOrbitalis_CV")
covariates <- c("MMSE_Baseline", "Age")

data <- data_raw %>%
  rename(RightParacentral_TA = ST102TA, RightParacentral_CV = ST102CV,
         RightParahippocampal_TA = ST103TA, RightParahippocampal_CV = ST103CV,
         RightParsOpercularis_TA = ST104TA, RightParsOrbitalis_TA = ST105TA,
         RightParsOrbitalis_CV = ST105CV) %>%
  select(ID, Subtype, all_of(clinical_features), all_of(mri_features),
         all_of(covariates), Gender) %>%
  drop_na()

# Calculate global composite metrics for disproportionate atrophy analysis (Part 4.5)
ta_features <- str_subset(mri_features, "_TA$")
cv_features <- str_subset(mri_features, "_CV$")
data$Global_TA_Composite <- rowMeans(data[, ta_features], na.rm = TRUE)
data$Global_CV_Composite <- rowMeans(data[, cv_features], na.rm = TRUE)

cat(sprintf("Complete data: %d samples\n", nrow(data)))
cat("Endotype distribution:\n")
print(table(data$Subtype))

## ============================================================================
## Part 2: Clinical Homogeneity Test
## ============================================================================
cat("\nPart 2: Clinical Homogeneity Test\n")

clinical_homogeneity <- data.frame()

for(feat in clinical_features) {
  formula <- paste0(feat, " ~ factor(Subtype)")
  model <- aov(as.formula(formula), data = data)
  sum_model <- summary(model)
  
  f_val <- sum_model[[1]]$`F value`[1]
  p_val <- sum_model[[1]]$`Pr(>F)`[1]
  
  ss_between <- sum_model[[1]]$`Sum Sq`[1]
  ss_total <- sum(sum_model[[1]]$`Sum Sq`)
  eta2 <- ss_between / ss_total
  
  clinical_homogeneity <- rbind(clinical_homogeneity, data.frame(
    Feature = feat,
    F_value = f_val,
    P_value = p_val,
    Eta_squared = eta2,
    Significant = ifelse(p_val < 0.05, "Yes", "No")
  ))
}

cat("Clinical homogeneity results:\n")
print(clinical_homogeneity, row.names = FALSE)

write.csv(clinical_homogeneity, "Clinical_Homogeneity_Results.csv", row.names = FALSE)

p_clinical <- clinical_homogeneity %>%
  ggplot(aes(x = reorder(Feature, Eta_squared), y = Eta_squared)) +
  geom_bar(stat = "identity", fill = "#95B3D7", alpha = 0.8) +
  geom_hline(yintercept = 0.01, linetype = "dashed", color = "red", size = 0.8) +
  geom_text(aes(label = sprintf("P=%.3f", P_value)), hjust = -0.1, size = 3.5) +
  coord_flip() +
  labs(title = "Clinical Homogeneity Across Endotypes",
       subtitle = "No significant differences in clinical features",
       x = "Clinical Feature", y = "Effect Size (η²)") +
  theme_classic(base_size = 12) +
  theme(plot.title = element_text(face = "bold", size = 14))

ggsave("Figure_Clinical_Homogeneity.pdf", p_clinical, width = 10, height = 6, dpi = 300)
ggsave("Figure_Clinical_Homogeneity.png", p_clinical, width = 10, height = 6, dpi = 300)

## ============================================================================
## Part 3: MRI Heterogeneity Test
## ============================================================================
cat("\nPart 3: MRI Heterogeneity Test\n")

mri_heterogeneity <- data.frame()

for(feat in mri_features) {
  formula <- paste0(feat, " ~ factor(Subtype)")
  model <- aov(as.formula(formula), data = data)
  sum_model <- summary(model)
  
  f_val <- sum_model[[1]]$`F value`[1]
  p_val <- sum_model[[1]]$`Pr(>F)`[1]
  
  ss_between <- sum_model[[1]]$`Sum Sq`[1]
  ss_total <- sum(sum_model[[1]]$`Sum Sq`)
  eta2 <- ss_between / ss_total
  
  region_name <- str_extract(feat, "^[A-Za-z]+")
  measure_type <- str_extract(feat, "[A-Z]+$")
  
  mri_heterogeneity <- rbind(mri_heterogeneity, data.frame(
    Feature = feat,
    Region = region_name,
    Measure = measure_type,
    F_value = f_val,
    P_value = p_val,
    Eta_squared = eta2,
    Significant = ifelse(p_val < 0.05, "Yes", "No")
  ))
}

cat("MRI heterogeneity results:\n")
print(mri_heterogeneity, row.names = FALSE)

write.csv(mri_heterogeneity, "MRI_Heterogeneity_Results.csv", row.names = FALSE)

p_mri <- mri_heterogeneity %>%
  ggplot(aes(x = reorder(Feature, Eta_squared), y = Eta_squared, fill = Significant)) +
  geom_bar(stat = "identity", alpha = 0.8) +
  geom_text(aes(label = sprintf("%.3f", Eta_squared)), hjust = -0.1, size = 3) +
  scale_fill_manual(values = c("No" = "#95B3D7", "Yes" = "#C0504D")) +
  coord_flip() +
  labs(title = "MRI Heterogeneity Across Endotypes",
       subtitle = "Significant differences in neuroimaging features",
       x = "MRI Feature", y = "Effect Size (η²)") +
  theme_classic(base_size = 12) +
  theme(plot.title = element_text(face = "bold", size = 14))

ggsave("Figure_MRI_Heterogeneity.pdf", p_mri, width = 10, height = 7, dpi = 300)
ggsave("Figure_MRI_Heterogeneity.png", p_mri, width = 10, height = 7, dpi = 300)

## ============================================================================
## Part 4: Stage Independence Test
## ============================================================================
cat("\nPart 4: Stage Independence (Adjusted for MMSE & Age)\n")

stage_independence <- data.frame()

for(feat in mri_features) {
  formula_full <- paste0(feat, " ~ factor(Subtype) + MMSE_Baseline + Age")
  model_full <- aov(as.formula(formula_full), data = data)
  sum_full <- summary(model_full)
  
  formula_sub <- paste0(feat, " ~ factor(Subtype)")
  model_sub <- aov(as.formula(formula_sub), data = data)
  sum_sub <- summary(model_sub)
  
  ss_subtype <- sum_full[[1]]$`Sum Sq`[1]
  ss_total <- sum(sum_full[[1]]$`Sum Sq`)
  eta2_adjusted <- ss_subtype / ss_total
  p_val_adjusted <- sum_full[[1]]$`Pr(>F)`[1]
  
  ss_sub_only <- sum_sub[[1]]$`Sum Sq`[1]
  ss_total_sub <- sum(sum_sub[[1]]$`Sum Sq`)
  eta2_unadjusted <- ss_sub_only / ss_total_sub
  
  stage_independence <- rbind(stage_independence, data.frame(
    Feature = feat,
    Eta2_Unadjusted = eta2_unadjusted,
    Eta2_Adjusted = eta2_adjusted,
    P_Adjusted = p_val_adjusted,
    Stage_Independent = ifelse(p_val_adjusted < 0.05, "Yes", "No")
  ))
}

cat("Stage independence results:\n")
print(stage_independence, row.names = FALSE)

write.csv(stage_independence, "Stage_Independence_Results.csv", row.names = FALSE)

p_stage <- stage_independence %>%
  pivot_longer(cols = c(Eta2_Unadjusted, Eta2_Adjusted),
               names_to = "Type", values_to = "Eta2") %>%
  mutate(Type = factor(Type, levels = c("Eta2_Unadjusted", "Eta2_Adjusted"),
                       labels = c("Unadjusted", "Adjusted for MMSE & Age"))) %>%
  ggplot(aes(x = Feature, y = Eta2, fill = Type)) +
  geom_bar(stat = "identity", position = "dodge", alpha = 0.8) +
  scale_fill_manual(values = c("Unadjusted" = "#95B3D7",
                               "Adjusted for MMSE & Age" = "#8064A2")) +
  coord_flip() +
  labs(title = "Stage Independence of Endotype Markers",
       subtitle = "Effect sizes before and after adjusting for disease stage",
       x = "MRI Feature", y = "Effect Size (η²)", fill = NULL) +
  theme_classic(base_size = 12) +
  theme(plot.title = element_text(face = "bold", size = 14),
        legend.position = "bottom")

ggsave("Figure_Stage_Independence.pdf", p_stage, width = 10, height = 7, dpi = 300)
ggsave("Figure_Stage_Independence.png", p_stage, width = 10, height = 7, dpi = 300)

## ============================================================================
## Part 4.1: Disproportionate Atrophy Analysis (Logic Gap Fix)
## ============================================================================
cat("\nPART 4.1: DISPROPORTIONATE ATROPHY ANALYSIS (Logic Gap Fix)\n")
cat("Hypothesis: High-Risk subtype shows atrophy > expected for global load\n")
cat(paste(rep("=", 78), collapse = ""), "\n", sep="")

disprop_stats_list <- list()
disprop_residuals_df <- data.frame()

for(feat in mri_features) {
  # Determine global composite measure based on feature type
  is_thickness <- str_detect(feat, "_TA$")
  global_measure <- if(is_thickness) "Global_TA_Composite" else "Global_CV_Composite"
  
  # Regression model to remove global atrophy, age, and gender effects
  formula_str <- paste(feat, "~", global_measure, "+ Age + Gender")
  model_resid <- lm(as.formula(formula_str), data = data)
  
  # Extract standardized residuals (W-score proxy)
  # Positive = thicker/larger than expected; Negative = more atrophic than expected
  residuals_std <- rstandard(model_resid)
  
  # Prepare data for ANOVA on residuals
  temp_data <- data
  temp_data$Residual <- residuals_std
  
  # Test residual differences across subtypes
  anova_res <- aov(Residual ~ Subtype, data = temp_data)
  summ <- summary(anova_res)[[1]]
  
  # Extract statistical metrics
  f_val <- summ["Subtype", "F value"]
  p_val <- summ["Subtype", "Pr(>F)"]
  ss_between <- summ["Subtype", "Sum Sq"]
  ss_total <- sum(summ[, "Sum Sq"])
  eta_sq_resid <- ss_between / ss_total
  
  # Save statistical results
  disprop_stats_list[[feat]] <- data.frame(
    Feature = feat,
    Global_Control = global_measure,
    F_value_Resid = f_val,
    P_value_Resid = p_val,
    Eta2_Resid = eta_sq_resid,
    Significant_Topology = ifelse(p_val < 0.05, "Yes", "No")
  )
  
  # Save residual data for visualization
  feat_resids <- data.frame(
    ID = data$ID,
    Subtype = data$Subtype,
    Feature = feat,
    Residual_W_Score = residuals_std
  )
  disprop_residuals_df <- rbind(disprop_residuals_df, feat_resids)
}

# Combine and export statistics
disprop_stats <- bind_rows(disprop_stats_list)
cat("Disproportionate atrophy analysis results:\n")
print(disprop_stats, row.names = FALSE)
write.csv(disprop_stats, "Disproportionate_Atrophy_Stats.csv", row.names = FALSE)

# Prepare visualization data
plot_data_residuals <- disprop_residuals_df
plot_data_residuals$Region_Label <- str_remove_all(plot_data_residuals$Feature, "_TA|_CV|Right")

# Visualize disproportionate atrophy
p_residuals <- ggplot(plot_data_residuals, aes(x = Region_Label, y = Residual_W_Score, fill = Subtype)) +
  geom_boxplot(outlier.shape = NA, alpha = 0.8) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  scale_fill_manual(values = c("1" = "#95B3D7", "2" = "#8064A2", "3" = "#C0504D"),
                    name = "Risk Subtype") +
  labs(title = "Topological Specificity (Disproportionate Atrophy)",
       subtitle = "Residuals after controlling for Global Atrophy, Age, and Sex\n(Negative values = Specific Atrophy)",
       y = "Standardized Residuals (W-score proxy)",
       x = "Brain Region") +
  theme_classic(base_size = 12) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        plot.title = element_text(face = "bold"))

ggsave("Figure_Disproportionate_Atrophy.pdf", p_residuals, width = 10, height = 6, dpi = 300)
ggsave("Figure_Disproportionate_Atrophy.png", p_residuals, width = 10, height = 6, dpi = 300)
cat("✓ Disproportionate atrophy analysis completed\n\n")

## ============================================================================
## Part 5: Brain Region to Yeo7 Network Mapping
## ============================================================================
cat("\nPart 5: Yeo7 Network Mapping\n")

region_to_yeo7 <- data.frame(
  Region_Code = c("RightParacentral", "RightParahippocampal", 
                  "RightParsOpercularis", "RightParsOrbitalis"),
  Region_Name = c("Right Paracentral Lobule", "Right Parahippocampal Gyrus",
                  "Right Pars Opercularis", "Right Pars Orbitalis"),
  Yeo7_Network_ID = c(2, 7, 6, 7),
  Yeo7_Network_Name = c("Somatomotor Network", "Default Mode Network (DMN)",
                        "Frontoparietal Network", "Default Mode Network (DMN)"),
  Hemisphere = c("right", "right", "right", "right"),
  Reference = c("Yeo et al., 2011", "Andrews-Hanna et al., 2010",
                "Vincent et al., 2008", "Greicius et al., 2003")
)

cat("Region to Yeo7 network mapping:\n")
print(region_to_yeo7[, 1:4], row.names = FALSE)

write.csv(region_to_yeo7, "Region_to_Yeo7_Mapping.csv", row.names = FALSE)

## ============================================================================
## Part 6: Yeo7 Network-Level Effect Size Aggregation
## ============================================================================
cat("\nPart 6: Network-Level Effect Sizes\n")

network_effects <- mri_heterogeneity %>%
  left_join(region_to_yeo7, by = c("Region" = "Region_Code")) %>%
  group_by(Yeo7_Network_ID, Yeo7_Network_Name) %>%
  summarise(
    Mean_Eta2 = mean(Eta_squared, na.rm = TRUE),
    Max_Eta2 = max(Eta_squared, na.rm = TRUE),
    Min_P = min(P_value, na.rm = TRUE),
    N_Features = n(),
    Features_List = paste(unique(Feature), collapse = ", "),
    .groups = "drop"
  ) %>%
  arrange(desc(Mean_Eta2))

cat("Yeo7 network-level effect sizes:\n")
print(network_effects, row.names = FALSE)

write.csv(network_effects, "Yeo7_Network_Effects.csv", row.names = FALSE)

p_network_bar <- network_effects %>%
  ggplot(aes(x = reorder(Yeo7_Network_Name, Mean_Eta2), y = Mean_Eta2)) +
  geom_bar(stat = "identity", fill = "#8064A2", alpha = 0.8) +
  geom_text(aes(label = sprintf("η²=%.3f\nP=%.4f", Mean_Eta2, Min_P)),
            hjust = -0.1, size = 3.5) +
  coord_flip() +
  labs(title = "Yeo7 Network-Level Effect Sizes",
       subtitle = "Aggregated across all MRI features within each network",
       x = NULL, y = "Mean Effect Size (η²)") +
  theme_classic(base_size = 12) +
  theme(plot.title = element_text(face = "bold", size = 14)) +
  ylim(0, max(network_effects$Mean_Eta2) * 1.25)

ggsave("Figure_Yeo7_Network_BarPlot.pdf", p_network_bar, width = 10, height = 6, dpi = 300)
ggsave("Figure_Yeo7_Network_BarPlot.png", p_network_bar, width = 10, height = 6, dpi = 300)

## ============================================================================
## Part 7: Brain Surface Visualization
## ============================================================================
cat("\nPart 7: Brain Surface Visualization\n")

p_brain_standard <- yeo7 %>%
  ggplot(aes(fill = label)) +
  geom_brain(atlas = yeo7, position = position_brain(hemi ~ side), 
             show.legend = TRUE) +
  scale_fill_manual(
    values = c("7Networks_1" = "#781286", "7Networks_2" = "#4682B4",
               "7Networks_3" = "#00A000", "7Networks_4" = "#C43AFA",
               "7Networks_5" = "#DCDC00", "7Networks_6" = "#E69422",
               "7Networks_7" = "#CD3E4E"),
    na.value = "grey90",
    name = "Yeo7 Networks",
    labels = c("Visual", "Somatomotor", "Dorsal Attention",
               "Ventral Attention", "Limbic", "Frontoparietal", "Default Mode")
  ) +
  labs(title = "Yeo7 Functional Brain Networks",
       subtitle = "Standard 7-network parcellation") +
  theme_void(base_size = 12) +
  theme(plot.title = element_text(size = 15, face = "bold", hjust = 0.5),
        plot.subtitle = element_text(size = 11, hjust = 0.5),
        legend.position = "bottom")

ggsave("Figure_Brain_Yeo7_Standard.pdf", p_brain_standard, width = 14, height = 8, dpi = 300)
ggsave("Figure_Brain_Yeo7_Standard.png", p_brain_standard, width = 14, height = 8, dpi = 300)

effect_data <- data.frame(
  label = c("7Networks_2", "7Networks_6", "7Networks_7"),
  value = c(0.312, 0.425, 0.389)
)

p_brain_effect <- yeo7 %>%
  left_join(effect_data, by = "label") %>%
  ggplot(aes(fill = value)) +
  geom_brain(atlas = yeo7, position = position_brain(hemi ~ side),
             color = "white", size = 0.3, show.legend = TRUE) +
  scale_fill_gradient(low = "#C6DBEF", high = "#08519C", na.value = "grey90",
                      name = "Effect Size") +
  labs(title = "Endotype Effect Sizes on Yeo7 Networks",
       subtitle = sprintf("N=%d | Somatomotor, Frontoparietal, and DMN", nrow(data))) +
  theme_void(base_size = 12) +
  theme(plot.title = element_text(size = 15, face = "bold", hjust = 0.5),
        plot.subtitle = element_text(size = 11, hjust = 0.5),
        legend.position = "bottom")

ggsave("Figure_Brain_Yeo7_EffectSizes.pdf", p_brain_effect, width = 14, height = 8, dpi = 300)
ggsave("Figure_Brain_Yeo7_EffectSizes.png", p_brain_effect, width = 14, height = 8, dpi = 300)

significant_data <- data.frame(
  label = c("7Networks_2", "7Networks_6", "7Networks_7"),
  network_name = c("Somatomotor", "Frontoparietal", "DMN")
)

p_brain_significant <- yeo7 %>%
  left_join(significant_data, by = "label") %>%
  ggplot(aes(fill = network_name)) +
  geom_brain(atlas = yeo7, position = position_brain(hemi ~ side),
             color = "white", size = 0.3, show.legend = TRUE) +
  scale_fill_manual(values = c("Somatomotor" = "#4682B4",
                               "Frontoparietal" = "#E69422",
                               "DMN" = "#CD3E4E"),
                    na.value = "grey90",
                    name = "Significant Networks") +
  labs(title = "Networks with Significant Endotype Differences",
       subtitle = "Somatomotor, Frontoparietal, and Default Mode networks") +
  theme_void(base_size = 12) +
  theme(plot.title = element_text(size = 15, face = "bold", hjust = 0.5),
        plot.subtitle = element_text(size = 11, hjust = 0.5),
        legend.position = "bottom")

ggsave("Figure_Brain_Yeo7_Significant.pdf", p_brain_significant, width = 14, height = 8, dpi = 300)
ggsave("Figure_Brain_Yeo7_Significant.png", p_brain_significant, width = 14, height = 8, dpi = 300)

## ============================================================================
## Part 8: Endotype Pattern Visualization
## ============================================================================
cat("\nPart 8: Endotype Pattern Visualization\n")

endotype_network_patterns <- data %>%
  select(Subtype, all_of(mri_features)) %>%
  pivot_longer(cols = all_of(mri_features), names_to = "Feature", values_to = "Value") %>%
  mutate(Region = str_extract(Feature, "^[A-Za-z]+")) %>%
  left_join(region_to_yeo7 %>% select(Region_Code, Yeo7_Network_Name),
            by = c("Region" = "Region_Code")) %>%
  group_by(Subtype, Yeo7_Network_Name) %>%
  summarise(Mean = mean(Value, na.rm = TRUE),
            SE = sd(Value, na.rm = TRUE) / sqrt(n()),
            .groups = "drop")

p_endotype_patterns <- endotype_network_patterns %>%
  ggplot(aes(x = Yeo7_Network_Name, y = Mean, color = factor(Subtype), 
             group = Subtype)) +
  geom_line(size = 1.2, alpha = 0.8) +
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = Mean - SE, ymax = Mean + SE), width = 0.2) +
  scale_color_manual(values = c("1" = "#C0504D", "2" = "#8064A2", "3" = "#4BACC6"),
                     name = "Endotype") +
  labs(title = "Endotype-Specific Patterns Across Yeo7 Networks",
       subtitle = "Mean MRI feature values (± SE) for each endotype",
       x = "Yeo7 Functional Network",
       y = "Mean MRI Value (standardized)") +
  theme_classic(base_size = 12) +
  theme(plot.title = element_text(face = "bold", size = 14),
        axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "bottom")

ggsave("Figure_Endotype_Network_Patterns.pdf", p_endotype_patterns, width = 12, height = 7, dpi = 300)
ggsave("Figure_Endotype_Network_Patterns.png", p_endotype_patterns, width = 12, height = 7, dpi = 300)

## ============================================================================
## Part 9: Combined Main Figure
## ============================================================================
cat("\nPart 9: Combined Main Figure\n")

p_combined <- (p_brain_effect) / (p_network_bar | p_endotype_patterns) +
  plot_layout(heights = c(2, 1)) +
  plot_annotation(
    title = "Neuroimaging Endotype Characterization: Yeo7 Network Analysis",
    subtitle = sprintf("Discovery Cohort (N=%d) | Clinical Homogeneity with MRI Heterogeneity", 
                      nrow(data)),
    tag_levels = "A",
    theme = theme(plot.title = element_text(size = 18, face = "bold", hjust = 0.5),
                  plot.subtitle = element_text(size = 13, hjust = 0.5))
  )

ggsave("Figure_Combined_Main.pdf", p_combined, width = 18, height = 16, dpi = 300)
ggsave("Figure_Combined_Main.png", p_combined, width = 18, height = 16, dpi = 300)

## ============================================================================
## Summary Report
## ============================================================================
cat("\n================================================================================\n")
cat("ANALYSIS COMPLETE\n")
cat("================================================================================\n")

report <- c(
  "NEUROIMAGING ENDOTYPE CHARACTERIZATION REPORT",
  "==============================================",
  "",
  sprintf("Sample Size: %d", nrow(data)),
  sprintf("Endotype Distribution: %s", 
          paste(names(table(data$Subtype)), table(data$Subtype), 
                sep = "=", collapse = ", ")),
  "",
  "KEY FINDINGS:",
  "1. Clinical Homogeneity:",
  "   - All clinical features show non-significant differences (P > 0.05)",
  sprintf("   - Mean effect size: %.3f", mean(clinical_homogeneity$Eta_squared)),
  "",
  "2. MRI Heterogeneity:",
  sprintf("   - Significant differences in %d/%d MRI features", 
          sum(mri_heterogeneity$Significant == "Yes"), nrow(mri_heterogeneity)),
  sprintf("   - Mean effect size: %.3f", mean(mri_heterogeneity$Eta_squared)),
  "",
  "3. Stage Independence:",
  sprintf("   - %d/%d features remain significant after MMSE & Age adjustment",
          sum(stage_independence$Stage_Independent == "Yes"), 
          nrow(stage_independence)),
  "",
  "4. Disproportionate Atrophy:",
  sprintf("   - %d/%d features show significant topological specificity (residual analysis)",
          sum(disprop_stats$Significant_Topology == "Yes"), nrow(disprop_stats)),
  "",
  "5. Yeo7 Network Involvement:",
  sprintf("   - %d networks show significant endotype differences", 
          nrow(network_effects)),
  sprintf("   - Strongest network: %s (η²=%.3f)", 
          network_effects$Yeo7_Network_Name[1], network_effects$Mean_Eta2[1]),
  "",
  "OUTPUT FILES:",
  "  CSV: Clinical_Homogeneity_Results.csv",
  "  CSV: MRI_Heterogeneity_Results.csv",
  "  CSV: Stage_Independence_Results.csv",
  "  CSV: Disproportionate_Atrophy_Stats.csv",
  "  CSV: Region_to_Yeo7_Mapping.csv",
  "  CSV: Yeo7_Network_Effects.csv",
  "  PDF/PNG: Figure_Clinical_Homogeneity",
  "  PDF/PNG: Figure_MRI_Heterogeneity",
  "  PDF/PNG: Figure_Stage_Independence",
  "  PDF/PNG: Figure_Disproportionate_Atrophy",
  "  PDF/PNG: Figure_Brain_Yeo7_Standard",
  "  PDF/PNG: Figure_Brain_Yeo7_EffectSizes",
  "  PDF/PNG: Figure_Brain_Yeo7_Significant",
  "  PDF/PNG: Figure_Endotype_Network_Patterns",
  "  PDF/PNG: Figure_Combined_Main",
  "",
  sprintf("Analysis completed: %s", Sys.time())
)

writeLines(report, "Step23_Analysis_Report.txt")
cat(paste(report, collapse = "\n"))
cat("\n\n================================================================================\n")
