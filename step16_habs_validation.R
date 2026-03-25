library(optparse)
library(dplyr)
library(pROC)
library(ggplot2)
library(logistf)
library(PRROC)

option_list <- list(
  make_option(c("--habs_file"), type = "character", default = "HABS_Baseline_Integrated.csv",
              help = "Integrated HABS baseline file [default: %default]"),
  make_option(c("--output_dir"), type = "character", default = "./step16_results",
              help = "Output directory [default: %default]"),
  make_option(c("--ptau_col"), type = "character", default = "pTau217_Primary",
              help = "Plasma p-tau217 column [default: %default]"),
  make_option(c("--n_bootstrap"), type = "integer", default = 2000,
              help = "Bootstrap iterations for confidence intervals [default: %default]"),
  make_option(c("--seed"), type = "integer", default = 42,
              help = "Random seed [default: %default]")
)
opt <- parse_args(OptionParser(option_list = option_list))
dir.create(opt$output_dir, showWarnings = FALSE, recursive = TRUE)
set.seed(opt$seed)

cat("======================================================================\n")
cat("Step 16: HABS Validation with Firth Logistic Regression\n")
cat("======================================================================\n\n")

habs <- read.csv(opt$habs_file, stringsAsFactors = FALSE)
ptau_col <- opt$ptau_col
if (!ptau_col %in% names(habs)) {
  alt <- c("pTau217", "ptau217", "PTAU217", "pTau217_Primary")
  hit <- alt[alt %in% names(habs)]
  if (length(hit) == 0) stop("Could not find plasma p-tau217 column")
  ptau_col <- hit[1]
}

required_common <- c("Age", "Gender", "APOE4_Positive", "MMSE_Baseline", "AD_Conversion")
missing_common <- setdiff(required_common, names(habs))
if (length(missing_common) > 0) stop(sprintf("Missing variables: %s", paste(missing_common, collapse = ", ")))

habs <- habs %>%
  mutate(
    Gender = as.factor(Gender),
    APOE4_Positive = as.numeric(APOE4_Positive),
    MMSE_Baseline = as.numeric(MMSE_Baseline),
    Age = as.numeric(Age),
    AD_Conversion = as.numeric(AD_Conversion),
    pTau217 = as.numeric(.data[[ptau_col]])
  )

calc_auc_ci <- function(y, p, n_boot = 2000) {
  roc_obj <- roc(y, p, quiet = TRUE)
  auc_val <- as.numeric(auc(roc_obj))
  boots <- replicate(n_boot, {
    idx <- sample(seq_along(y), replace = TRUE)
    if (length(unique(y[idx])) < 2) return(NA_real_)
    as.numeric(auc(roc(y[idx], p[idx], quiet = TRUE)))
  })
  boots <- boots[!is.na(boots)]
  c(AUC = auc_val, CI_Lower = quantile(boots, 0.025), CI_Upper = quantile(boots, 0.975))
}

calc_auprc <- function(y, p) {
  pr <- pr.curve(scores.class0 = p[y == 1], scores.class1 = p[y == 0], curve = TRUE)
  list(AUPRC = pr$auc.integral, Curve = as.data.frame(pr$curve))
}

calc_nri <- function(y, old_p, new_p, cutpoints = c(0.05, 0.10, 0.20)) {
  old_cat <- cut(old_p, breaks = c(-Inf, cutpoints, Inf), labels = FALSE)
  new_cat <- cut(new_p, breaks = c(-Inf, cutpoints, Inf), labels = FALSE)
  events <- which(y == 1)
  nonevents <- which(y == 0)
  events_nri <- mean(new_cat[events] > old_cat[events]) - mean(new_cat[events] < old_cat[events])
  nonevents_nri <- mean(new_cat[nonevents] < old_cat[nonevents]) - mean(new_cat[nonevents] > old_cat[nonevents])
  c(NRI_Total = events_nri + nonevents_nri, NRI_Events = events_nri, NRI_NonEvents = nonevents_nri)
}

calc_idi <- function(y, old_p, new_p) {
  events <- y == 1
  nonevents <- y == 0
  (mean(new_p[events]) - mean(old_p[events])) - (mean(new_p[nonevents]) - mean(old_p[nonevents]))
}

calc_net_benefit <- function(y, p, thresholds) {
  sapply(thresholds, function(t) {
    pred <- ifelse(p >= t, 1, 0)
    tp <- sum(pred == 1 & y == 1)
    fp <- sum(pred == 1 & y == 0)
    n <- length(y)
    (tp / n) - (fp / n) * (t / (1 - t))
  })
}

subset_complete <- habs %>% filter(!is.na(pTau217)) %>% select(all_of(required_common), pTau217)
subset_complete <- na.omit(subset_complete)
full_cohort <- habs %>% select(all_of(required_common)) %>% na.omit()

cat(sprintf("Full cohort: n=%d, events=%d\n", nrow(full_cohort), sum(full_cohort$AD_Conversion)))
cat(sprintf("p-tau217 subset: n=%d, events=%d\n\n", nrow(subset_complete), sum(subset_complete$AD_Conversion)))

model_full <- logistf(AD_Conversion ~ Age + Gender + APOE4_Positive + MMSE_Baseline, data = full_cohort)
prob_full <- predict(model_full, newdata = full_cohort, type = "response")
auc_full <- calc_auc_ci(full_cohort$AD_Conversion, prob_full, opt$n_bootstrap)

model_base <- logistf(AD_Conversion ~ Age + Gender + APOE4_Positive + MMSE_Baseline, data = subset_complete)
model_complete <- logistf(AD_Conversion ~ Age + Gender + APOE4_Positive + MMSE_Baseline + pTau217, data = subset_complete)
prob_base <- predict(model_base, newdata = subset_complete, type = "response")
prob_complete <- predict(model_complete, newdata = subset_complete, type = "response")

auc_base <- calc_auc_ci(subset_complete$AD_Conversion, prob_base, opt$n_bootstrap)
auc_complete <- calc_auc_ci(subset_complete$AD_Conversion, prob_complete, opt$n_bootstrap)
delong <- roc.test(roc(subset_complete$AD_Conversion, prob_base, quiet = TRUE), roc(subset_complete$AD_Conversion, prob_complete, quiet = TRUE), method = "delong")

pr_base <- calc_auprc(subset_complete$AD_Conversion, prob_base)
pr_complete <- calc_auprc(subset_complete$AD_Conversion, prob_complete)
event_rate <- mean(subset_complete$AD_Conversion)

nri <- calc_nri(subset_complete$AD_Conversion, prob_base, prob_complete)
idi <- calc_idi(subset_complete$AD_Conversion, prob_base, prob_complete)

thresholds <- seq(0.01, 0.60, by = 0.01)
dca <- data.frame(
  Threshold = thresholds,
  Baseline = calc_net_benefit(subset_complete$AD_Conversion, prob_base, thresholds),
  Complete = calc_net_benefit(subset_complete$AD_Conversion, prob_complete, thresholds)
)

subgroup_defs <- list(
  "MMSE_lt_27" = subset_complete$MMSE_Baseline < 27,
  "APOE4_Positive" = subset_complete$APOE4_Positive == 1,
  "Age_gt_70" = subset_complete$Age > 70,
  "Age_gt_70_APOE4_Positive" = subset_complete$Age > 70 & subset_complete$APOE4_Positive == 1
)
subgroup_rows <- list()
for (nm in names(subgroup_defs)) {
  sub <- subset_complete[subgroup_defs[[nm]], , drop = FALSE]
  if (nrow(sub) < 20 || length(unique(sub$AD_Conversion)) < 2) next
  mod <- logistf(AD_Conversion ~ Age + Gender + APOE4_Positive + MMSE_Baseline + pTau217, data = sub)
  pr <- predict(mod, newdata = sub, type = "response")
  auc_sub <- calc_auc_ci(sub$AD_Conversion, pr, min(opt$n_bootstrap, 500))
  subgroup_rows[[nm]] <- data.frame(
    Subgroup = nm,
    N = nrow(sub),
    Events = sum(sub$AD_Conversion),
    AUC = auc_sub["AUC"],
    CI_Lower = auc_sub["CI_Lower"],
    CI_Upper = auc_sub["CI_Upper"]
  )
}
subgroup_df <- bind_rows(subgroup_rows)

perf <- bind_rows(
  data.frame(
    Stratum = "HABS_Full_No_pTau217",
    Analysis_Set = "Full cohort clinical only",
    N = nrow(full_cohort),
    Events = sum(full_cohort$AD_Conversion),
    auc_full,
    stringsAsFactors = FALSE
  ),
  data.frame(
    Stratum = "HABS_pTau217_Subset_Baseline",
    Analysis_Set = "Matched pTau217 subset baseline",
    N = nrow(subset_complete),
    Events = sum(subset_complete$AD_Conversion),
    auc_base,
    stringsAsFactors = FALSE
  ),
  data.frame(
    Stratum = "HABS_pTau217_Subset_Complete",
    Analysis_Set = "Matched pTau217 subset complete",
    N = nrow(subset_complete),
    Events = sum(subset_complete$AD_Conversion),
    auc_complete,
    stringsAsFactors = FALSE
  )
)
perf$AUPRC <- c(NA, pr_base$AUPRC, pr_complete$AUPRC)
perf$Event_Rate <- perf$Events / perf$N
perf$DeLong_P_Value <- c(NA, delong$p.value, delong$p.value)

write.csv(perf, file.path(opt$output_dir, "step16_performance_summary.csv"), row.names = FALSE)
write.csv(
  perf %>% select(Stratum, Analysis_Set, N, Events, Event_Rate, AUC, CI_Lower, CI_Upper, AUPRC, DeLong_P_Value),
  file.path(opt$output_dir, "step16_manuscript_summary.csv"),
  row.names = FALSE
)
write.csv(subgroup_df, file.path(opt$output_dir, "step16_subgroup_summary.csv"), row.names = FALSE)
write.csv(dca, file.path(opt$output_dir, "step16_decision_curve.csv"), row.names = FALSE)
write.csv(data.frame(Metric = names(nri), Value = as.numeric(nri)), file.path(opt$output_dir, "step16_nri.csv"), row.names = FALSE)
write.csv(data.frame(Metric = "IDI", Value = idi), file.path(opt$output_dir, "step16_idi.csv"), row.names = FALSE)
write.csv(as.data.frame(summary(model_complete)$coefficients), file.path(opt$output_dir, "step16_complete_model_coefficients.csv"))
write.csv(as.data.frame(summary(model_full)$coefficients), file.path(opt$output_dir, "step16_full_cohort_model_coefficients.csv"))

roc_df <- bind_rows(
  data.frame(FPR = 1 - roc(subset_complete$AD_Conversion, prob_base, quiet = TRUE)$specificities, TPR = roc(subset_complete$AD_Conversion, prob_base, quiet = TRUE)$sensitivities, Model = "Baseline"),
  data.frame(FPR = 1 - roc(subset_complete$AD_Conversion, prob_complete, quiet = TRUE)$specificities, TPR = roc(subset_complete$AD_Conversion, prob_complete, quiet = TRUE)$sensitivities, Model = "Complete")
)
p1 <- ggplot(roc_df, aes(FPR, TPR, color = Model)) + geom_line(size = 1) + geom_abline(linetype = 2) + theme_minimal()
ggsave(file.path(opt$output_dir, "step16_roc_curves.pdf"), p1, width = 6, height = 5)

pr_df <- bind_rows(
  data.frame(Recall = pr_base$Curve[, 1], Precision = pr_base$Curve[, 2], Model = "Baseline"),
  data.frame(Recall = pr_complete$Curve[, 1], Precision = pr_complete$Curve[, 2], Model = "Complete")
)
p2 <- ggplot(pr_df, aes(Recall, Precision, color = Model)) + geom_line(size = 1) + geom_hline(yintercept = event_rate, linetype = 2) + theme_minimal()
ggsave(file.path(opt$output_dir, "step16_pr_curves.pdf"), p2, width = 6, height = 5)

p3 <- ggplot(dca, aes(Threshold)) +
  geom_line(aes(y = Baseline, color = "Baseline"), size = 1) +
  geom_line(aes(y = Complete, color = "Complete"), size = 1) +
  theme_minimal() + ylab("Net benefit")
ggsave(file.path(opt$output_dir, "step16_decision_curve.pdf"), p3, width = 6, height = 5)

summary_lines <- c(
  sprintf("Full cohort AUC (without p-tau217): %.3f", perf$AUC[perf$Stratum == "HABS_Full_No_pTau217"]),
  sprintf("p-tau217 subset baseline AUC: %.3f", perf$AUC[perf$Stratum == "HABS_pTau217_Subset_Baseline"]),
  sprintf("p-tau217 subset complete AUC: %.3f", perf$AUC[perf$Stratum == "HABS_pTau217_Subset_Complete"]),
  sprintf("DeLong p-value: %.4f", delong$p.value),
  sprintf("Baseline AUPRC: %.3f", pr_base$AUPRC),
  sprintf("Complete AUPRC: %.3f", pr_complete$AUPRC),
  sprintf("Event-rate baseline AUPRC: %.3f", event_rate),
  sprintf("NRI total: %.3f", nri["NRI_Total"]),
  sprintf("IDI: %.4f", idi)
)
writeLines(summary_lines, file.path(opt$output_dir, "step16_summary.txt"))

cat("Saved HABS validation outputs\n")
