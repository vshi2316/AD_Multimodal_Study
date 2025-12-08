## Step 16: HABS External Validation (Parts 1–3)

library(pROC)
library(dplyr)
library(ggplot2)
library(gridExtra)
library(logistf)
library(PRROC)

cat("============================================================\n")
cat("Step 16: HABS External Validation\n")
cat("============================================================\n\n")

## ============================================================
## Part 1: Data preparation and Firth logistic models
## ============================================================

cat("Part 1: Data preparation and model fitting\n")

habs_data <- read.csv("HABS_Baseline_Integrated.csv", stringsAsFactors = FALSE)
cat(sprintf("Total HABS sample: %d\n", nrow(habs_data)))

ptau217_col <- "pTau217_Primary"
if (!ptau217_col %in% colnames(habs_data)) {
  stop(sprintf("Missing column: %s", ptau217_col))
}

ptau_available <- sum(!is.na(habs_data[[ptau217_col]]))
ptau_rate <- 100 * ptau_available / nrow(habs_data)
cat(sprintf("p-tau217 available: %d/%d (%.1f%%)\n", ptau_available, nrow(habs_data), ptau_rate))

key_vars <- c("Age", "Gender", "APOE4_Positive", "MMSE_Baseline",
              ptau217_col, "AD_Conversion")

missing_vars <- setdiff(key_vars, colnames(habs_data))
if (length(missing_vars) > 0) {
  stop(sprintf("Missing key variables: %s", paste(missing_vars, collapse = ", ")))
}

habs_subset <- habs_data %>%
  select(all_of(key_vars)) %>%
  rename(pTau217 = !!ptau217_col)

n_total <- nrow(habs_subset)
n_ptau_available <- sum(!is.na(habs_subset$pTau217))
ptau_rate_subset <- 100 * n_ptau_available / n_total

cat(sprintf("Key variables: %d\n", length(key_vars)))
cat(sprintf("p-tau217 available in subset: %d/%d (%.1f%%)\n",
            n_ptau_available, n_total, ptau_rate_subset))

habs_complete <- na.omit(habs_subset)
complete_rate <- nrow(habs_complete) / nrow(habs_subset)

cat(sprintf("Complete cases: %d (%.1f%%)\n",
            nrow(habs_complete), 100 * complete_rate))
cat(sprintf("Events in complete cases: %d (%.1f%%)\n",
            sum(habs_complete$AD_Conversion),
            100 * mean(habs_complete$AD_Conversion)))

set.seed(42)
train_idx <- sample(seq_len(nrow(habs_complete)),
                    size = floor(0.7 * nrow(habs_complete)))
habs_train <- habs_complete[train_idx, ]
habs_test  <- habs_complete[-train_idx, ]

cat(sprintf("Training set: %d (events=%d, %.1f%%)\n",
            nrow(habs_train),
            sum(habs_train$AD_Conversion == 1),
            100 * mean(habs_train$AD_Conversion == 1)))
cat(sprintf("Test set: %d (events=%d, %.1f%%)\n",
            nrow(habs_test),
            sum(habs_test$AD_Conversion == 1),
            100 * mean(habs_test$AD_Conversion == 1)))

cat("Fitting Firth logistic regression models\n")

model_base <- logistf(
  AD_Conversion ~ Age + Gender + APOE4_Positive + MMSE_Baseline,
  data = habs_train
)

model_complete <- logistf(
  AD_Conversion ~ Age + Gender + APOE4_Positive + MMSE_Baseline + pTau217,
  data = habs_train
)

pred_base <- predict(model_base, newdata = habs_test, type = "response")
pred_complete <- predict(model_complete, newdata = habs_test, type = "response")

roc_base <- roc(habs_test$AD_Conversion, pred_base, quiet = TRUE)
roc_complete <- roc(habs_test$AD_Conversion, pred_complete, quiet = TRUE)

auc_base <- as.numeric(auc(roc_base))
auc_complete <- as.numeric(auc(roc_complete))
ci_base <- ci.auc(roc_base)
ci_complete <- ci.auc(roc_complete)

cat(sprintf("Baseline model AUC: %.3f (95%% CI %.3f–%.3f)\n",
            auc_base, ci_base[1], ci_base[3]))
cat(sprintf("Complete model AUC: %.3f (95%% CI %.3f–%.3f)\n",
            auc_complete, ci_complete[1], ci_complete[3]))
cat(sprintf("Delta AUC: %.3f\n", auc_complete - auc_base))

roc_test <- roc.test(roc_base, roc_complete, method = "delong")
cat(sprintf("DeLong p-value: %.4e\n", roc_test$p.value))

habs_test$pred_base <- pred_base
habs_test$pred_complete <- pred_complete

habs_test <- habs_test %>%
  mutate(
    MCI_like        = ifelse(MMSE_Baseline < 27, 1, 0),
    APOE4_Group     = ifelse(APOE4_Positive == 1, 1, 0),
    Elderly         = ifelse(Age > 70, 1, 0),
    High_Risk_Combined = ifelse(Elderly == 1 & APOE4_Group == 1, 1, 0),
    Low_Cognition   = ifelse(MMSE_Baseline <
                               median(MMSE_Baseline, na.rm = TRUE), 1, 0)
  )

subgroup_summary <- data.frame(
  Subgroup = c("All", "MCI-like (MMSE<27)", "APOE4+",
               "Elderly (>70)", "Elderly + APOE4", "Low cognition"),
  N = c(
    nrow(habs_test),
    sum(habs_test$MCI_like),
    sum(habs_test$APOE4_Group),
    sum(habs_test$Elderly),
    sum(habs_test$High_Risk_Combined),
    sum(habs_test$Low_Cognition)
  ),
  Events = c(
    sum(habs_test$AD_Conversion),
    sum(habs_test$AD_Conversion[habs_test$MCI_like == 1]),
    sum(habs_test$AD_Conversion[habs_test$APOE4_Group == 1]),
    sum(habs_test$AD_Conversion[habs_test$Elderly == 1]),
    sum(habs_test$AD_Conversion[habs_test$High_Risk_Combined == 1]),
    sum(habs_test$AD_Conversion[habs_test$Low_Cognition == 1])
  )
)

subgroup_summary$Event_Rate_Pct <- 100 * subgroup_summary$Events / subgroup_summary$N

cat("\nSubgroup summary:\n")
print(subgroup_summary)

save(habs_data, habs_complete, habs_train, habs_test,
     model_base, model_complete,
     pred_base, pred_complete,
     roc_base, roc_complete,
     auc_base, auc_complete,
     ci_base, ci_complete, roc_test,
     subgroup_summary, ptau_rate, ptau217_col,
     file = "step16_habs_part1_results.RData")

cat("\nPart 1 complete. Results saved to step16_habs_part1_results.RData\n\n")

## ============================================================
## Part 2: AUC by subgroup, AUPRC, NRI, IDI
## ============================================================

cat("Part 2: AUC by subgroup, AUPRC, NRI, IDI\n")

if (!file.exists("step16_habs_part1_results.RData")) {
  stop("Run Part 1 before Part 2.")
}
load("step16_habs_part1_results.RData")

calc_auc_subgroup <- function(data, outcome_col = "AD_Conversion",
                              pred_col = "pred_complete") {
  if (nrow(data) < 10 || sum(data[[outcome_col]]) < 5) {
    return(list(AUC = NA, CI_Lower = NA, CI_Upper = NA))
  }
  roc_obj <- roc(data[[outcome_col]], data[[pred_col]], quiet = TRUE)
  ci_obj <- ci.auc(roc_obj, conf.level = 0.95)
  list(
    AUC = as.numeric(roc_obj$auc),
    CI_Lower = ci_obj[1],
    CI_Upper = ci_obj[3]
  )
}

auc_subgroups <- data.frame(
  Subgroup   = "All",
  N          = nrow(habs_test),
  Events     = sum(habs_test$AD_Conversion),
  Event_Rate = 100 * mean(habs_test$AD_Conversion),
  AUC        = auc_complete,
  CI_Lower   = ci_complete[1],
  CI_Upper   = ci_complete[3]
)

mci_data <- habs_test %>% filter(MCI_like == 1)
if (nrow(mci_data) >= 10) {
  mci_auc <- calc_auc_subgroup(mci_data)
  auc_subgroups <- rbind(auc_subgroups, data.frame(
    Subgroup   = "MCI-like",
    N          = nrow(mci_data),
    Events     = sum(mci_data$AD_Conversion),
    Event_Rate = 100 * mean(mci_data$AD_Conversion),
    AUC        = mci_auc$AUC,
    CI_Lower   = mci_auc$CI_Lower,
    CI_Upper   = mci_auc$CI_Upper
  ))
}

apoe4_data <- habs_test %>% filter(APOE4_Group == 1)
if (nrow(apoe4_data) >= 10) {
  apoe4_auc <- calc_auc_subgroup(apoe4_data)
  auc_subgroups <- rbind(auc_subgroups, data.frame(
    Subgroup   = "APOE4+",
    N          = nrow(apoe4_data),
    Events     = sum(apoe4_data$AD_Conversion),
    Event_Rate = 100 * mean(apoe4_data$AD_Conversion),
    AUC        = apoe4_auc$AUC,
    CI_Lower   = apoe4_auc$CI_Lower,
    CI_Upper   = apoe4_auc$CI_Upper
  ))
}

elderly_data <- habs_test %>% filter(Elderly == 1)
if (nrow(elderly_data) >= 10) {
  elderly_auc <- calc_auc_subgroup(elderly_data)
  auc_subgroups <- rbind(auc_subgroups, data.frame(
    Subgroup   = "Elderly",
    N          = nrow(elderly_data),
    Events     = sum(elderly_data$AD_Conversion),
    Event_Rate = 100 * mean(elderly_data$AD_Conversion),
    AUC        = elderly_auc$AUC,
    CI_Lower   = elderly_auc$CI_Lower,
    CI_Upper   = elderly_auc$CI_Upper
  ))
}

combined_data <- habs_test %>% filter(High_Risk_Combined == 1)
if (nrow(combined_data) >= 10) {
  combined_auc <- calc_auc_subgroup(combined_data)
  auc_subgroups <- rbind(auc_subgroups, data.frame(
    Subgroup   = "Elderly + APOE4",
    N          = nrow(combined_data),
    Events     = sum(combined_data$AD_Conversion),
    Event_Rate = 100 * mean(combined_data$AD_Conversion),
    AUC        = combined_auc$AUC,
    CI_Lower   = combined_auc$CI_Lower,
    CI_Upper   = combined_auc$CI_Upper
  ))
}

cat("\nSubgroup AUC results:\n")
print(auc_subgroups)

auc_improvements <- auc_subgroups$AUC[-1] - auc_subgroups$AUC[1]
best_subgroup_idx <- which.max(auc_improvements) + 1

cat(sprintf("\nMaximum AUC improvement: %.3f in subgroup %s\n",
            max(auc_improvements, na.rm = TRUE),
            auc_subgroups$Subgroup[best_subgroup_idx]))

calc_auprc <- function(y_true, y_pred) {
  pr_obj <- pr.curve(
    scores.class0 = y_pred[y_true == 1],
    scores.class1 = y_pred[y_true == 0],
    curve = TRUE
  )
  list(
    AUPRC   = pr_obj$auc.integral,
    PR_Curve = pr_obj$curve
  )
}

auprc_base     <- calc_auprc(habs_test$AD_Conversion, pred_base)
auprc_complete <- calc_auprc(habs_test$AD_Conversion, pred_complete)
baseline_auprc <- mean(habs_test$AD_Conversion)

cat(sprintf("\nBaseline model AUPRC: %.3f\n", auprc_base$AUPRC))
cat(sprintf("Complete model AUPRC: %.3f\n", auprc_complete$AUPRC))
cat(sprintf("Random baseline AUPRC: %.3f\n", baseline_auprc))
cat(sprintf("Delta AUPRC: %.3f\n",
            auprc_complete$AUPRC - auprc_base$AUPRC))

fold_improvement <- auprc_complete$AUPRC / baseline_auprc
cat(sprintf("Complete model AUPRC is %.2f times random\n", fold_improvement))

threshold_sets <- list(
  High_Event_Rate = c(0, 0.20, 0.40, 0.60, 1),
  Standard        = c(0, 0.10, 0.20, 1),
  Fine_Grained    = c(0, 0.15, 0.25, 0.35, 0.50, 1)
)

calc_nri <- function(pred_old, pred_new, outcome, thresholds) {
  cat_old <- cut(pred_old, breaks = thresholds, include.lowest = TRUE)
  cat_new <- cut(pred_new, breaks = thresholds, include.lowest = TRUE)
  
  events_idx    <- which(outcome == 1)
  nonevents_idx <- which(outcome == 0)
  
  if (length(events_idx) == 0 || length(nonevents_idx) == 0) {
    return(list(events_nri = NA, nonevents_nri = NA, total_nri = NA))
  }
  
  events_table    <- table(Old = cat_old[events_idx],    New = cat_new[events_idx])
  nonevents_table <- table(Old = cat_old[nonevents_idx], New = cat_new[nonevents_idx])
  
  events_up   <- sum(events_table[lower.tri(events_table)])
  events_down <- sum(events_table[upper.tri(events_table)])
  events_nri  <- (events_up - events_down) / length(events_idx)
  
  nonevents_down <- sum(nonevents_table[upper.tri(nonevents_table)])
  nonevents_up   <- sum(nonevents_table[lower.tri(nonevents_table)])
  nonevents_nri  <- (nonevents_down - nonevents_up) / length(nonevents_idx)
  
  total_nri <- events_nri + nonevents_nri
  
  list(events_nri = events_nri,
       nonevents_nri = nonevents_nri,
       total_nri = total_nri)
}

nri_results <- list()
for (name in names(threshold_sets)) {
  nri_results[[name]] <- calc_nri(pred_base, pred_complete,
                                  habs_test$AD_Conversion,
                                  threshold_sets[[name]])
  cat(sprintf("\nNRI with threshold set %s (%s):\n",
              name, paste(threshold_sets[[name]], collapse = ", ")))
  cat(sprintf("  Total NRI: %.3f (%.1f%%)\n",
              nri_results[[name]]$total_nri,
              100 * nri_results[[name]]$total_nri))
  cat(sprintf("  Events NRI: %.3f\n",    nri_results[[name]]$events_nri))
  cat(sprintf("  Non-events NRI: %.3f\n", nri_results[[name]]$nonevents_nri))
}

nri_result <- nri_results$High_Event_Rate

events_idx    <- habs_test$AD_Conversion == 1
nonevents_idx <- habs_test$AD_Conversion == 0

events_improvement    <- mean(pred_complete[events_idx])    - mean(pred_base[events_idx])
nonevents_improvement <- mean(pred_complete[nonevents_idx]) - mean(pred_base[nonevents_idx])
idi_value             <- events_improvement - nonevents_improvement

cat(sprintf("\nEvents probability improvement: %.4f\n", events_improvement))
cat(sprintf("Non-events probability improvement: %.4f\n", nonevents_improvement))
cat(sprintf("IDI: %.4f (%.2f%%)\n", idi_value, 100 * idi_value))

set.seed(42)
n_bootstrap   <- 1000
idi_bootstrap <- numeric(n_bootstrap)

for (i in seq_len(n_bootstrap)) {
  boot_idx  <- sample(seq_len(nrow(habs_test)), replace = TRUE)
  boot_data <- habs_test[boot_idx, ]
  
  boot_events_idx    <- boot_data$AD_Conversion == 1
  boot_nonevents_idx <- boot_data$AD_Conversion == 0
  
  boot_events_improvement <- mean(boot_data$pred_complete[boot_events_idx]) -
    mean(boot_data$pred_base[boot_events_idx])
  boot_nonevents_improvement <- mean(boot_data$pred_complete[boot_nonevents_idx]) -
    mean(boot_data$pred_base[boot_nonevents_idx])
  
  idi_bootstrap[i] <- boot_events_improvement - boot_nonevents_improvement
}

idi_ci      <- as.numeric(quantile(idi_bootstrap, c(0.025, 0.975)))
idi_p_value <- 2 * min(mean(idi_bootstrap > 0), mean(idi_bootstrap < 0))

cat(sprintf("IDI 95%% CI: %.4f–%.4f\n", idi_ci[1], idi_ci[2]))
cat(sprintf("IDI p-value: %.4f\n", idi_p_value))

save(auc_subgroups, auprc_base, auprc_complete, baseline_auprc,
     fold_improvement, nri_results, nri_result,
     idi_value, idi_ci, idi_p_value, auc_improvements,
     file = "step16_habs_part2_results.RData")

cat("\nPart 2 complete. Results saved to step16_habs_part2_results.RData\n\n")

## ============================================================
## Part 3: Decision curve analysis and figures
## ============================================================

cat("Part 3: Decision curve analysis and visualizations\n")

if (!file.exists("step16_habs_part1_results.RData") ||
    !file.exists("step16_habs_part2_results.RData")) {
  stop("Run Part 1 and Part 2 before Part 3.")
}
load("step16_habs_part1_results.RData")
load("step16_habs_part2_results.RData")

calc_net_benefit <- function(pred, outcome, threshold) {
  tp <- sum(pred >= threshold & outcome == 1)
  fp <- sum(pred >= threshold & outcome == 0)
  n  <- length(outcome)
  (tp / n) - (fp / n) * (threshold / (1 - threshold))
}

thresholds_dca <- seq(0, 0.8, by = 0.01)

nb_base     <- sapply(thresholds_dca, function(t) calc_net_benefit(pred_base,
                                                                   habs_test$AD_Conversion, t))
nb_complete <- sapply(thresholds_dca, function(t) calc_net_benefit(pred_complete,
                                                                   habs_test$AD_Conversion, t))

nb_all <- sapply(thresholds_dca, function(t) {
  event_rate <- mean(habs_test$AD_Conversion)
  event_rate - (1 - event_rate) * (t / (1 - t))
})
nb_none <- rep(0, length(thresholds_dca))

max_benefit_idx <- which.max(nb_complete)
optimal_threshold <- thresholds_dca[max_benefit_idx]
max_net_benefit  <- nb_complete[max_benefit_idx]

cat(sprintf("Optimal decision threshold: %.3f\n", optimal_threshold))
cat(sprintf("Maximum net benefit: %.4f\n", max_net_benefit))

p1 <- ggplot() +
  geom_line(aes(x = 1 - roc_base$specificities,
                y = roc_base$sensitivities,
                color = "Baseline model"),
            size = 1.2) +
  geom_line(aes(x = 1 - roc_complete$specificities,
                y = roc_complete$sensitivities,
                color = "Complete model (with p-tau217)"),
            size = 1.2) +
  geom_abline(intercept = 0, slope = 1,
              linetype = "dashed", color = "gray", alpha = 0.7) +
  labs(
    title = "HABS Validation: ROC Curves",
    subtitle = sprintf("Baseline AUC=%.3f, Complete AUC=%.3f, Delta=%.3f (p=%.4e)",
                       auc_base, auc_complete,
                       auc_complete - auc_base, roc_test$p.value),
    x = "1 - Specificity",
    y = "Sensitivity",
    color = NULL
  ) +
  scale_color_manual(values = c("Baseline model" = "#E69F00",
                                "Complete model (with p-tau217)" = "#009E73")) +
  theme_bw(base_size = 12) +
  theme(
    legend.position  = c(0.7, 0.2),
    plot.title       = element_text(hjust = 0.5, size = 14, face = "bold"),
    plot.subtitle    = element_text(hjust = 0.5, size = 11)
  )

ggsave("step16_fig1_roc_comparison.pdf", p1,
       width = 8, height = 6, dpi = 300)

auc_subgroups_plot <- auc_subgroups %>%
  mutate(Subgroup = factor(Subgroup, levels = rev(Subgroup)))

p2 <- ggplot(auc_subgroups_plot, aes(x = AUC, y = Subgroup)) +
  geom_point(size = 4, color = "#0072B2") +
  geom_errorbarh(aes(xmin = CI_Lower, xmax = CI_Upper),
                 height = 0.3, size = 1) +
  geom_vline(xintercept = 0.5, linetype = "dashed",
             color = "gray", alpha = 0.7) +
  geom_vline(xintercept = auc_complete, linetype = "dotted",
             color = "red", size = 1) +
  geom_text(aes(label = sprintf("%.3f", AUC)),
            hjust = -0.3, size = 3.5) +
  labs(
    title = "HABS Subgroup AUC (Complete Model)",
    subtitle = "High-risk subgroups",
    x = "AUC (95% CI)",
    y = NULL
  ) +
  scale_x_continuous(limits = c(0.4, 1.0), breaks = seq(0.4, 1.0, 0.1)) +
  theme_bw(base_size = 12) +
  theme(
    plot.title    = element_text(hjust = 0.5, size = 14, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5, size = 11),
    axis.text.y   = element_text(size = 11)
  )

ggsave("step16_fig2_subgroup_auc_forest.pdf", p2,
       width = 10, height = 6, dpi = 300)

pr_data_complete <- data.frame(
  Recall    = auprc_complete$PR_Curve[, 1],
  Precision = auprc_complete$PR_Curve[, 2],
  Model     = "Complete model"
)

pr_data_base <- data.frame(
  Recall    = auprc_base$PR_Curve[, 1],
  Precision = auprc_base$PR_Curve[, 2],
  Model     = "Baseline model"
)

pr_combined <- rbind(pr_data_base, pr_data_complete)

p3 <- ggplot(pr_combined, aes(x = Recall, y = Precision, color = Model)) +
  geom_line(size = 1.2) +
  geom_hline(yintercept = baseline_auprc,
             linetype = "dashed", color = "gray", alpha = 0.7) +
  geom_text(x = 0.7, y = baseline_auprc + 0.05,
            label = sprintf("Random classifier (%.3f)", baseline_auprc),
            color = "gray", size = 3.5) +
  labs(
    title = "Precision–Recall Curves",
    subtitle = sprintf("Complete model AUPRC=%.3f (%.2fx random)",
                       auprc_complete$AUPRC, fold_improvement),
    x = "Recall",
    y = "Precision",
    color = NULL
  ) +
  scale_color_manual(values = c("Baseline model" = "#E69F00",
                                "Complete model" = "#009E73")) +
  theme_bw(base_size = 12) +
  theme(
    legend.position = c(0.7, 0.8),
    plot.title      = element_text(hjust = 0.5, size = 14, face = "bold"),
    plot.subtitle   = element_text(hjust = 0.5, size = 11)
  )

ggsave("step16_fig3_precision_recall.pdf", p3,
       width = 8, height = 6, dpi = 300)

dca_data <- data.frame(
  Threshold   = rep(thresholds_dca, 4),
  Net_Benefit = c(nb_base, nb_complete, nb_all, nb_none),
  Strategy    = factor(rep(c("Baseline model", "Complete model",
                             "Treat all", "Treat none"),
                           each = length(thresholds_dca)))
)

p4 <- ggplot(dca_data, aes(x = Threshold, y = Net_Benefit, color = Strategy)) +
  geom_line(size = 1.2) +
  geom_vline(xintercept = optimal_threshold,
             linetype = "dotted", color = "red", alpha = 0.7) +
  geom_text(x = optimal_threshold + 0.05, y = max_net_benefit,
            label = sprintf("Optimal threshold: %.3f", optimal_threshold),
            color = "red", size = 3.5, angle = 90) +
  labs(
    title = "Decision Curve Analysis (HABS)",
    subtitle = "Net benefit vs. risk threshold",
    x = "Risk threshold",
    y = "Net benefit",
    color = NULL
  ) +
  scale_color_manual(values = c(
    "Baseline model" = "#E69F00",
    "Complete model" = "#009E73",
    "Treat all"      = "#999999",
    "Treat none"     = "#000000"
  )) +
  theme_bw(base_size = 12) +
  theme(
    legend.position = c(0.7, 0.8),
    plot.title      = element_text(hjust = 0.5, size = 14, face = "bold"),
    plot.subtitle   = element_text(hjust = 0.5, size = 11)
  )

ggsave("step16_fig4_decision_curve.pdf", p4,
       width = 10, height = 6, dpi = 300)

nri_data <- data.frame(
  Metric     = c("NRI (High_Event_Rate)", "NRI (Standard)",
                 "NRI (Fine_Grained)", "IDI"),
  Value      = c(nri_results$High_Event_Rate$total_nri,
                 nri_results$Standard$total_nri,
                 nri_results$Fine_Grained$total_nri,
                 idi_value),
  CI_Lower   = c(NA, NA, NA, idi_ci[1]),
  CI_Upper   = c(NA, NA, NA, idi_ci[2]),
  Significant = c(TRUE, TRUE, TRUE, idi_p_value < 0.05)
)

p5 <- ggplot(nri_data, aes(x = reorder(Metric, Value), y = Value,
                           fill = Significant)) +
  geom_col(alpha = 0.8, width = 0.6) +
  geom_errorbar(aes(ymin = CI_Lower, ymax = CI_Upper),
                width = 0.2, na.rm = TRUE) +
  geom_text(aes(label = sprintf("%.3f", Value)),
            hjust = -0.1, size = 4) +
  scale_fill_manual(values = c("TRUE" = "#009E73", "FALSE" = "#E69F00")) +
  labs(
    title = "NRI and IDI Improvements",
    x = NULL,
    y = "Value"
  ) +
  coord_flip() +
  theme_bw(base_size = 12) +
  theme(
    legend.position = "none",
    plot.title      = element_text(hjust = 0.5, size = 14, face = "bold")
  )

ggsave("step16_fig5_nri_idi.pdf", p5,
       width = 8, height = 6, dpi = 300)

p_combined <- grid.arrange(
  p1 + theme(plot.title = element_text(size = 10),
             plot.subtitle = element_text(size = 8)),
  p3 + theme(plot.title = element_text(size = 10),
             plot.subtitle = element_text(size = 8)),
  p4 + theme(plot.title = element_text(size = 10),
             plot.subtitle = element_text(size = 8)),
  p5 + theme(plot.title = element_text(size = 10),
             plot.subtitle = element_text(size = 8)),
  ncol = 2, nrow = 2,
  top = "HABS External Validation (Complete Model with p-tau217)"
)

ggsave("step16_fig6_combined_analysis.pdf", p_combined,
       width = 16, height = 12, dpi = 300)

dca_results <- list(
  thresholds_dca   = thresholds_dca,
  nb_base          = nb_base,
  nb_complete      = nb_complete,
  nb_all           = nb_all,
  nb_none          = nb_none,
  optimal_threshold = optimal_threshold,
  max_net_benefit   = max_net_benefit
)

save(dca_results, file = "step16_habs_part3_results.RData")

cat("\nPart 3 complete. Figures and results saved.\n")
cat("Step 16 finished.\n")
