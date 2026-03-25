suppressPackageStartupMessages({
  library(optparse)
  library(survival)
  library(ConsensusClusterPlus)
  library(cluster)
  library(mclust)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(RColorBrewer)
})

option_list <- list(
  make_option(c("--data_dir"), type = "character", default = ".",
              help = "Directory containing the discovery cohort input files [default: %default]"),
  make_option(c("--vae_dir"), type = "character", default = ".",
              help = "Directory containing subtype_assignments.csv and latent_representations.csv [default: %default]"),
  make_option(c("--clinical_file"), type = "character", default = "Clinical_data.csv",
              help = "Clinical input file [default: %default]"),
  make_option(c("--mri_file"), type = "character", default = "RNA_plasma.csv",
              help = "Structural MRI input file [default: %default]"),
  make_option(c("--csf_file"), type = "character", default = "metabolites.csv",
              help = "CSF input file [default: %default]"),
  make_option(c("--outcome_file"), type = "character", default = "Womac_score_pain_function.csv",
              help = "Outcome file containing AD conversion and archived follow up time columns [default: %default]"),
  make_option(c("--output_dir"), type = "character", default = "./step14_results",
              help = "Output directory [default: %default]"),
  make_option(c("--n_bootstrap"), type = "integer", default = 2000,
              help = "Number of bootstrap iterations [default: %default]"),
  make_option(c("--n_consensus"), type = "integer", default = 1000,
              help = "Number of consensus clustering iterations [default: %default]"),
  make_option(c("--stability_threshold"), type = "double", default = 0.85,
              help = "Sample-level stability threshold [default: %default]")
)

opt <- parse_args(OptionParser(option_list = option_list))

data_dir <- opt$data_dir
vae_dir <- opt$vae_dir
output_dir <- opt$output_dir
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

n_bootstrap <- opt$n_bootstrap
n_consensus <- opt$n_consensus
stability_threshold <- opt$stability_threshold
set.seed(42)

cat("========================================================================\\n")
cat("  Step 14: Cluster Validation (37-variable latent solution)\\n")
cat("========================================================================\\n\\n")

# ============================================================================
# Helper functions
# ============================================================================
calculate_jaccard_index <- function(labels_true, labels_pred) {
  unique_true <- sort(unique(labels_true))
  unique_pred <- sort(unique(labels_pred))
  D <- max(length(unique_true), length(unique_pred))
  w <- matrix(0, nrow = D, ncol = D)
  for (i in seq_along(unique_true)) {
    for (j in seq_along(unique_pred)) {
      w[i, j] <- sum(labels_true == unique_true[i] &
                       labels_pred == unique_pred[j])
    }
  }
  matched <- rep(FALSE, D)
  jaccards <- c()
  for (i in seq_along(unique_true)) {
    best_j <- which.max(w[i, ] * (!matched))
    if (w[i, best_j] > 0) {
      matched[best_j] <- TRUE
      inter <- sum(labels_true == unique_true[i] &
                     labels_pred == unique_pred[best_j])
      uni   <- sum(labels_true == unique_true[i] |
                     labels_pred == unique_pred[best_j])
      jaccards <- c(jaccards, inter / uni)
    }
  }
  if (length(jaccards) == 0) return(0)
  mean(jaccards)
}
calculate_PAC <- function(cm, lower = 0.1, upper = 0.9) {
  lt <- cm[lower.tri(cm)]
  sum(lt > lower & lt < upper) / length(lt)
}
calculate_sample_stability <- function(boot_matrix, original_labels) {
  n <- nrow(boot_matrix)
  stability <- numeric(n)
  for (i in 1:n) {
    obs <- boot_matrix[i, ]
    obs <- obs[!is.na(obs)]
    if (length(obs) < 2) { stability[i] <- NA; next }
    mode_cl <- as.numeric(names(sort(table(obs), decreasing = TRUE)[1]))
    stability[i] <- sum(obs == mode_cl) / length(obs)
  }
  stability
}
wilson_ci <- function(x, n, alpha = 0.05) {
  z <- qnorm(1 - alpha / 2)
  p <- x / n
  denom <- 1 + z^2 / n
  center <- (p + z^2 / (2 * n)) / denom
  margin <- z * sqrt((p * (1 - p) + z^2 / (4 * n)) / n) / denom
  c(lower = center - margin, upper = center + margin)
}
# ============================================================================
# [1/7] Load Data
# ============================================================================
cat("[1/7] Loading data...\n")
# VAE outputs
assignments <- read.csv(file.path(vae_dir, "subtype_assignments.csv"),
                        stringsAsFactors = FALSE)
cat(sprintf("  Subtype assignments: %d rows, cols: %s\n",
            nrow(assignments), paste(colnames(assignments), collapse = ", ")))
latent_file <- file.path(vae_dir, "latent_representations.csv")
latent_available <- file.exists(latent_file)
if (latent_available) {
  latent_data <- read.csv(latent_file, stringsAsFactors = FALSE)
  cat(sprintf("  Latent: %d x %d\n", nrow(latent_data), ncol(latent_data) - 1))
}
# Original data from 7A_CSF_Cohort
clinical <- read.csv(file.path(data_dir, opt$clinical_file),
                     stringsAsFactors = FALSE)
cat(sprintf("  Clinical: %d x %d\n", nrow(clinical), ncol(clinical)))
mri_data <- read.csv(file.path(data_dir, opt$mri_file),
                     stringsAsFactors = FALSE)
mri_cols <- grep("^ST\\d+", colnames(mri_data), value = TRUE)
cat(sprintf("  MRI: %d features\n", length(mri_cols)))
csf_data <- read.csv(file.path(data_dir, opt$csf_file),
                     stringsAsFactors = FALSE)
cat(sprintf("  CSF: %d biomarkers\n", ncol(csf_data) - 1))
outcome <- read.csv(file.path(data_dir, opt$outcome_file),
                    stringsAsFactors = FALSE)
cat(sprintf("  Outcome: %d rows\n", nrow(outcome)))
# Merge all — drop overlapping columns from clinical to avoid .x/.y duplicates
# assignments already has: ID, VAE_Subtype, Direct_KMeans_Subtype, SEX, AGE, AD_Conversion
clinical_cols_to_add <- setdiff(colnames(clinical), colnames(assignments))
clinical_slim <- clinical[, c("ID", clinical_cols_to_add), drop = FALSE]
outcome_cols_to_add <- setdiff(colnames(outcome), colnames(assignments))
outcome_slim <- outcome[, c("ID", intersect(outcome_cols_to_add, colnames(outcome))), drop = FALSE]
df <- assignments %>%
  left_join(clinical_slim, by = "ID") %>%
  left_join(outcome_slim, by = "ID") %>%
  left_join(mri_data, by = "ID") %>%
  left_join(csf_data, by = "ID")
# Ensure AD_Conversion exists (from assignments)
if (!"AD_Conversion" %in% colnames(df) && "AD_Conversion" %in% colnames(outcome)) {
  df <- df %>% left_join(outcome %>% select(ID, AD_Conversion), by = "ID")
}
cat(sprintf("  Merged: %d rows, %d cols\n", nrow(df), ncol(df)))
cat(sprintf("  Columns check: AGE=%s, SEX=%s, EDUCATION=%s, AD_Conversion=%s\n",
            "AGE" %in% colnames(df), "SEX" %in% colnames(df),
            "EDUCATION" %in% colnames(df), "AD_Conversion" %in% colnames(df)))

time_candidates <- c("Followup_Years", "Followup_Months", "Time_to_Event")
available_time_cols <- intersect(time_candidates, colnames(df))
cat(sprintf("  Time columns available: %s\n",
            ifelse(length(available_time_cols) > 0, paste(available_time_cols, collapse = ", "), "none")))
# Fix SEX encoding: round to 0/1 (data has one value=0.88 from encoding error)
if ("SEX" %in% colnames(df)) {
  n_odd <- sum(df$SEX != 0 & df$SEX != 1, na.rm = TRUE)
  if (n_odd > 0) {
    cat(sprintf("  WARNING: %d non-binary SEX values detected, rounding to 0/1\n", n_odd))
    df$SEX <- round(df$SEX)
  }
}
# Remap subtypes: lowest conversion = 1 (Low-risk), highest = 3 (High-risk)
conv_rates_raw <- df %>%
  group_by(VAE_Subtype) %>%
  summarise(rate = mean(AD_Conversion, na.rm = TRUE), .groups = "drop") %>%
  arrange(rate)
remap <- setNames(1:nrow(conv_rates_raw), conv_rates_raw$VAE_Subtype)
df$Subtype <- remap[as.character(df$VAE_Subtype)]
df$Subtype_F <- factor(df$Subtype, levels = 1:3,
                       labels = c("Low-risk", "Intermediate", "High-risk"))
cat("\n  Subtype remapping (ascending conversion rate):\n")
for (i in 1:nrow(conv_rates_raw)) {
  cat(sprintf("    VAE_Subtype %d -> Subtype %d (rate=%.1f%%)\n",
              conv_rates_raw$VAE_Subtype[i], i, conv_rates_raw$rate[i] * 100))
}
cat(sprintf("  Total: %d participants, %d converters (%.1f%%)\n\n",
            nrow(df), sum(df$AD_Conversion, na.rm = TRUE),
            mean(df$AD_Conversion, na.rm = TRUE) * 100))
# ============================================================================
# [2/7] Conversion Rate Analysis
# ============================================================================
cat("[2/7] Conversion rate analysis...\n")
conv_table <- df %>%
  group_by(Subtype, Subtype_F) %>%
  summarise(n = n(), converters = sum(AD_Conversion, na.rm = TRUE),
            rate = mean(AD_Conversion, na.rm = TRUE), .groups = "drop")
conv_table$ci_lower <- NA
conv_table$ci_upper <- NA
for (i in 1:nrow(conv_table)) {
  ci <- wilson_ci(conv_table$converters[i], conv_table$n[i])
  conv_table$ci_lower[i] <- ci["lower"]
  conv_table$ci_upper[i] <- ci["upper"]
}
cat("\n  Conversion rates:\n")
for (i in 1:nrow(conv_table)) {
  cat(sprintf("    Subtype %d (%s): n=%d, conv=%d, rate=%.1f%% (95%%CI: %.1f%%-%.1f%%)\n",
              conv_table$Subtype[i], conv_table$Subtype_F[i],
              conv_table$n[i], conv_table$converters[i],
              conv_table$rate[i] * 100,
              conv_table$ci_lower[i] * 100, conv_table$ci_upper[i] * 100))
}
# Chi-square
ct <- table(df$Subtype, df$AD_Conversion)
chi_test <- chisq.test(ct)
cat(sprintf("\n  Chi-square: X2=%.3f, df=%d, P=%.4f\n",
            chi_test$statistic, chi_test$parameter, chi_test$p.value))
# Fisher exact
fisher_test <- fisher.test(ct)
cat(sprintf("  Fisher exact: P=%.4f\n", fisher_test$p.value))
# Pairwise Fisher
cat("\n  Pairwise (Fisher exact):\n")
pairs <- combn(1:3, 2)
pairwise_p <- numeric(ncol(pairs))
for (j in 1:ncol(pairs)) {
  s1 <- pairs[1, j]; s2 <- pairs[2, j]
  sub <- df %>% filter(Subtype %in% c(s1, s2))
  ft <- fisher.test(table(sub$Subtype, sub$AD_Conversion))
  pairwise_p[j] <- ft$p.value
  cat(sprintf("    Subtype %d vs %d: P=%.4f\n", s1, s2, ft$p.value))
}
adj_p <- p.adjust(pairwise_p, "bonferroni")
cat(sprintf("    Bonferroni: %s\n",
            paste(sprintf("%.4f", adj_p), collapse = ", ")))
write.csv(conv_table, file.path(output_dir, "Conversion_Rates.csv"),
          row.names = FALSE)
# ============================================================================
# [3/7] Cox Regression + Logistic Regression
# ============================================================================
cat("\n[3/7] Cox / Logistic regression...\n")
has_age <- "AGE" %in% colnames(df)
has_sex <- "SEX" %in% colnames(df)
has_edu <- "EDUCATION" %in% colnames(df)
cat(sprintf("  Covariates: AGE=%s, SEX=%s, EDUCATION=%s\n",
            has_age, has_sex, has_edu))
df$Subtype_cox <- factor(df$Subtype, levels = c(1, 2, 3))
# --- Logistic regression (primary, no time assumption needed) ---
cat("\n  === Logistic Regression (primary analysis) ===\n")
# Model 1: Unadjusted
logit1 <- glm(AD_Conversion ~ Subtype_cox, data = df, family = binomial)
cat("\n  --- Unadjusted Logistic ---\n")
or1 <- exp(cbind(OR = coef(logit1), confint.default(logit1)))
p1  <- summary(logit1)$coefficients[, 4]
print(round(cbind(or1, P = p1), 4))
# Model 2: Adjusted (age + sex + education)
adj_terms <- "Subtype_cox"
if (has_age) adj_terms <- c(adj_terms, "AGE")
if (has_sex) adj_terms <- c(adj_terms, "SEX")
if (has_edu) adj_terms <- c(adj_terms, "EDUCATION")
fml_adj <- as.formula(paste("AD_Conversion ~", paste(adj_terms, collapse = " + ")))
logit2 <- glm(fml_adj, data = df, family = binomial)
cat("\n  --- Adjusted Logistic (age + sex + education) ---\n")
or2 <- exp(cbind(OR = coef(logit2), confint.default(logit2)))
p2  <- summary(logit2)$coefficients[, 4]
print(round(cbind(or2, P = p2), 4))
# Model 3: + APOE4
if ("APOE4_DOSAGE" %in% colnames(df)) {
  adj_terms3 <- c(adj_terms, "APOE4_DOSAGE")
  fml3 <- as.formula(paste("AD_Conversion ~", paste(adj_terms3, collapse = " + ")))
  logit3 <- glm(fml3, data = df, family = binomial)
  cat("\n  --- Adjusted Logistic (+ APOE4) ---\n")
  or3 <- exp(cbind(OR = coef(logit3), confint.default(logit3)))
  p3  <- summary(logit3)$coefficients[, 4]
  print(round(cbind(or3, P = p3), 4))
}
# --- Cox regression ---
time_col_used <- NULL
time_multiplier <- 1
if ("Followup_Years" %in% colnames(df) && sum(!is.na(df$Followup_Years)) > 20) {
  time_col_used <- "Followup_Years"
  time_multiplier <- 1
} else if ("Followup_Months" %in% colnames(df) && sum(!is.na(df$Followup_Months)) > 20) {
  time_col_used <- "Followup_Months"
  time_multiplier <- 1 / 12
} else if ("Time_to_Event" %in% colnames(df) && sum(!is.na(df$Time_to_Event)) > 20) {
  time_col_used <- "Time_to_Event"
  time_multiplier <- 1 / 12
}

if (!is.null(time_col_used)) {
  cat(sprintf("\n  === Cox Regression (using %s) ===\n", time_col_used))
  df$surv_time <- as.numeric(df[[time_col_used]]) * time_multiplier
} else {
  cat("\n  === Cox Regression (sensitivity analysis, proxy time) ===\n")
  df$surv_time <- ifelse(df$AD_Conversion == 1, 1, 2)
}

time_metadata <- data.frame(
  Time_Source = ifelse(is.null(time_col_used), "proxy_time", time_col_used),
  Used_Proxy_Time = is.null(time_col_used),
  Available_Time_Columns = ifelse(length(available_time_cols) > 0,
                                  paste(available_time_cols, collapse = ", "),
                                  "none"),
  Nonmissing_Time_Count = ifelse(is.null(time_col_used),
                                 NA_integer_,
                                 sum(!is.na(df[[time_col_used]]))),
  Median_Survival_Time_Years = median(df$surv_time, na.rm = TRUE),
  stringsAsFactors = FALSE
)
write.csv(time_metadata,
          file.path(output_dir, "Cox_Time_Source_Metadata.csv"),
          row.names = FALSE)

surv_obj <- Surv(df$surv_time, df$AD_Conversion)
cox1 <- coxph(surv_obj ~ Subtype_cox, data = df)
cat("\n  --- Unadjusted Cox ---\n")
print(summary(cox1))
fml_cox <- as.formula(paste("surv_obj ~", paste(adj_terms, collapse = " + ")))
cox2 <- coxph(fml_cox, data = df)
cat("\n  --- Adjusted Cox ---\n")
print(summary(cox2))
if ("APOE4_DOSAGE" %in% colnames(df)) {
  fml_cox3 <- as.formula(paste("surv_obj ~", paste(adj_terms3, collapse = " + ")))
  cox3 <- coxph(fml_cox3, data = df)
  cat("\n  --- Adjusted Cox (+ APOE4) ---\n")
  print(summary(cox3))
}
# Extract HRs
extract_hr <- function(model, label) {
  s <- summary(model)
  ci <- s$conf.int
  st_rows <- grep("Subtype_cox", rownames(ci))
  if (length(st_rows) == 0) return(NULL)
  data.frame(
    Model = label,
    Comparison = rownames(ci)[st_rows],
    HR = ci[st_rows, 1],
    Lower95 = ci[st_rows, 3],
    Upper95 = ci[st_rows, 4],
    P = s$coefficients[st_rows, 5],
    stringsAsFactors = FALSE
  )
}
hr_all <- rbind(
  extract_hr(cox1, "Unadjusted"),
  extract_hr(cox2, "Adj_Age_Sex_Edu")
)
if (exists("cox3")) {
  hr_all <- rbind(hr_all, extract_hr(cox3, "Adj_Age_Sex_Edu_APOE4"))
}
cat("\n  === Hazard Ratios Summary ===\n")
for (i in 1:nrow(hr_all)) {
  cat(sprintf("    %s | %s: HR=%.3f (%.3f-%.3f), P=%.4f\n",
              hr_all$Model[i], hr_all$Comparison[i],
              hr_all$HR[i], hr_all$Lower95[i], hr_all$Upper95[i],
              hr_all$P[i]))
}
write.csv(hr_all, file.path(output_dir, "Cox_Hazard_Ratios.csv"),
          row.names = FALSE)
# PH assumption
tryCatch({
  ph_test <- cox.zph(cox2)
  cat("\n  Proportional hazards test:\n")
  print(ph_test)
}, error = function(e) {
  cat(sprintf("\n  PH test skipped: %s\n", e$message))
})
# Logistic regression as primary (no time assumption)
cat("\n  --- Logistic regression OR summary ---\n")
fml_logit <- as.formula(paste("AD_Conversion ~", paste(adj_terms, collapse = " + ")))
logit_final <- glm(fml_logit, data = df, family = binomial)
or_final <- exp(cbind(OR = coef(logit_final), confint.default(logit_final)))
p_final  <- summary(logit_final)$coefficients[, 4]
logit_table <- round(cbind(or_final, P = p_final), 4)
print(logit_table)
write.csv(as.data.frame(logit_table),
          file.path(output_dir, "Logistic_Regression_ORs.csv"))
# ============================================================================
# [4/7] MRI ANOVA + ANCOVA
# ============================================================================
cat("\n[4/7] MRI ANOVA across subtypes (30 features)...\n")
mri_results <- data.frame(Feature = character(), F_stat = numeric(),
                          P_value = numeric(), Eta_sq = numeric(),
                          stringsAsFactors = FALSE)
for (feat in mri_cols) {
  if (!feat %in% colnames(df)) next
  vals <- df[[feat]]
  if (all(is.na(vals))) next
  aov_fit <- aov(vals ~ Subtype_F, data = df)
  s <- summary(aov_fit)[[1]]
  rn <- trimws(rownames(s))
  sub_idx <- which(rn == "Subtype_F")
  ss_b <- s[sub_idx, "Sum Sq"]
  ss_t <- sum(s[, "Sum Sq"])
  mri_results <- rbind(mri_results, data.frame(
    Feature = feat, F_stat = s[sub_idx, "F value"],
    P_value = s[sub_idx, "Pr(>F)"], Eta_sq = ss_b / ss_t,
    stringsAsFactors = FALSE))
}
mri_results$P_FDR <- p.adjust(mri_results$P_value, "BH")
mri_results <- mri_results %>% arrange(desc(Eta_sq))
cat(sprintf("  Tested: %d features\n", nrow(mri_results)))
cat(sprintf("  Significant (P_FDR<0.05): %d\n", sum(mri_results$P_FDR < 0.05)))
cat(sprintf("  Significant (P_FDR<0.01): %d\n", sum(mri_results$P_FDR < 0.01)))
cat("\n  Top 10 by eta-squared:\n")
top10 <- head(mri_results, 10)
for (i in 1:nrow(top10)) {
  cat(sprintf("    %s: F=%.2f, P=%.2e, eta2=%.3f, FDR=%.4f\n",
              top10$Feature[i], top10$F_stat[i], top10$P_value[i],
              top10$Eta_sq[i], top10$P_FDR[i]))
}
write.csv(mri_results, file.path(output_dir, "MRI_ANOVA_Results.csv"),
          row.names = FALSE)
# ANCOVA: adjust for MMSE + AGE + SEX
cat("\n  ANCOVA (MMSE + AGE + SEX)...\n")
ancova_results <- data.frame(Feature = character(), F_stat = numeric(),
                             P_value = numeric(), Eta_sq_partial = numeric(),
                             stringsAsFactors = FALSE)
for (feat in mri_cols) {
  if (!feat %in% colnames(df)) next
  vals <- df[[feat]]
  if (all(is.na(vals))) next
  fml_anc <- as.formula(paste(feat, "~ Subtype_F + MMSE + AGE + SEX"))
  tryCatch({
    aov_fit <- aov(fml_anc, data = df)
    s <- summary(aov_fit)[[1]]
    # Fix: rownames have trailing whitespace — use grepl to match
    rn <- trimws(rownames(s))
    sub_row <- which(rn == "Subtype_F")
    res_row <- which(rn == "Residuals")
    if (length(sub_row) == 1 && length(res_row) == 1) {
      ss_sub <- s[sub_row, "Sum Sq"]
      ss_res <- s[res_row, "Sum Sq"]
      ancova_results <- rbind(ancova_results, data.frame(
        Feature = feat, F_stat = s[sub_row, "F value"],
        P_value = s[sub_row, "Pr(>F)"],
        Eta_sq_partial = ss_sub / (ss_sub + ss_res),
        stringsAsFactors = FALSE))
    }
  }, error = function(e) NULL)
}
ancova_results$P_FDR <- p.adjust(ancova_results$P_value, "BH")
ancova_results <- ancova_results %>% arrange(desc(Eta_sq_partial))
cat(sprintf("  ANCOVA significant (FDR<0.05): %d / %d\n",
            sum(ancova_results$P_FDR < 0.05), nrow(ancova_results)))
cat("\n  Top 10 ANCOVA:\n")
top10a <- head(ancova_results, 10)
for (i in 1:nrow(top10a)) {
  cat(sprintf("    %s: F=%.2f, P=%.2e, partial_eta2=%.3f, FDR=%.4f\n",
              top10a$Feature[i], top10a$F_stat[i], top10a$P_value[i],
              top10a$Eta_sq_partial[i], top10a$P_FDR[i]))
}
write.csv(ancova_results, file.path(output_dir, "MRI_ANCOVA_Results.csv"),
          row.names = FALSE)
# ============================================================================
# [5/7] CSF + Clinical comparisons
# ============================================================================
cat("\n[5/7] CSF and clinical comparisons...\n")
csf_vars  <- c("PTAU181", "ABETA42_ABETA40_RATIO", "ABETA42", "ABETA40",
               "TAU_TOTAL", "STREM2", "PGRN")
clin_vars <- c("MMSE", "ADAS13", "CDRSB", "FAQTOTAL", "EDUCATION", "GDS")
all_test_vars <- c(csf_vars, clin_vars)
comp_results <- data.frame(Variable = character(), Test = character(),
                           Statistic = numeric(), P_value = numeric(),
                           Eta_sq = numeric(), Mean_S1 = numeric(),
                           Mean_S2 = numeric(), Mean_S3 = numeric(),
                           stringsAsFactors = FALSE)
for (v in all_test_vars) {
  if (!v %in% colnames(df)) next
  vals <- df[[v]]
  if (all(is.na(vals))) next

  kw <- kruskal.test(vals ~ Subtype_F, data = df)
  aov_fit <- aov(vals ~ Subtype_F, data = df)
  s <- summary(aov_fit)[[1]]
  rn <- trimws(rownames(s))
  sub_idx <- which(rn == "Subtype_F")
  eta_sq <- s[sub_idx, "Sum Sq"] / sum(s[, "Sum Sq"])

  means <- df %>% group_by(Subtype) %>%
    summarise(m = mean(.data[[v]], na.rm = TRUE), .groups = "drop") %>%
    arrange(Subtype)

  comp_results <- rbind(comp_results, data.frame(
    Variable = v, Test = "Kruskal-Wallis",
    Statistic = kw$statistic, P_value = kw$p.value, Eta_sq = eta_sq,
    Mean_S1 = means$m[1], Mean_S2 = means$m[2], Mean_S3 = means$m[3],
    stringsAsFactors = FALSE))
}
cat("\n  Results:\n")
for (i in 1:nrow(comp_results)) {
  sig <- ifelse(comp_results$P_value[i] < 0.05, " *", "")
  cat(sprintf("    %s: H=%.2f, P=%.4f%s, eta2=%.3f  [S1=%.2f S2=%.2f S3=%.2f]\n",
              comp_results$Variable[i], comp_results$Statistic[i],
              comp_results$P_value[i], sig, comp_results$Eta_sq[i],
              comp_results$Mean_S1[i], comp_results$Mean_S2[i],
              comp_results$Mean_S3[i]))
}
write.csv(comp_results, file.path(output_dir, "CSF_Clinical_Comparisons.csv"),
          row.names = FALSE)
# Sex distribution
cat("\n  Sex distribution:\n")
sex_tab <- table(df$Subtype_F, df$SEX)
print(sex_tab)
sex_chi <- chisq.test(sex_tab)
cat(sprintf("  Chi-square: X2=%.3f, P=%.2e\n\n",
            sex_chi$statistic, sex_chi$p.value))
# Age distribution
if (has_age) {
  cat("  Age by subtype:\n")
  age_summary <- df %>% group_by(Subtype_F) %>%
    summarise(mean_age = mean(AGE, na.rm = TRUE),
              sd_age = sd(AGE, na.rm = TRUE), .groups = "drop")
  print(as.data.frame(age_summary))
  age_aov <- summary(aov(AGE ~ Subtype_F, data = df))[[1]]
  rn_age <- trimws(rownames(age_aov))
  age_sub_idx <- which(rn_age == "Subtype_F")
  cat(sprintf("  Age ANOVA: F=%.2f, P=%.4f\n\n",
              age_aov[age_sub_idx, "F value"],
              age_aov[age_sub_idx, "Pr(>F)"]))
}
# ============================================================================
# [6/7] Consensus Clustering + Bootstrap Stability
# ============================================================================
cat("[6/7] Consensus clustering + bootstrap...\n")
# Feature matrix: same 37 variables as VAE input
csf_input  <- csf_vars[csf_vars %in% colnames(df)]
clin_input <- c("APOE4_DOSAGE", "MMSE", "EDUCATION", "GDS")
clin_input <- clin_input[clin_input %in% colnames(df)]
mri_input  <- mri_cols[mri_cols %in% colnames(df)]
all_features <- c(csf_input, clin_input, mri_input)
cat(sprintf("  Features: CSF=%d, Clin=%d, MRI=%d, Total=%d\n",
            length(csf_input), length(clin_input),
            length(mri_input), length(all_features)))
feature_matrix <- df %>%
  select(all_of(all_features)) %>%
  mutate(across(everything(), ~ ifelse(is.na(.), median(., na.rm = TRUE), .))) %>%
  scale() %>%
  t()
n_samples <- ncol(feature_matrix)
# Consensus clustering — use relative path to avoid double-concatenation bug
old_wd <- getwd()
setwd(output_dir)
set.seed(42)
cc <- ConsensusClusterPlus(
  d = feature_matrix, maxK = 6, reps = n_consensus,
  pItem = 0.8, pFeature = 1, clusterAlg = "km", distance = "euclidean",
  title = "Consensus", plot = "png",
  writeTable = TRUE, seed = 42)
setwd(old_wd)
stab_tab <- data.frame(K = 2:6, PAC = NA, Min_Size = NA, Silhouette = NA)
for (k in 2:6) {
  cm <- cc[[k]]$consensusMatrix
  cl <- cc[[k]]$consensusClass
  stab_tab$PAC[k - 1] <- calculate_PAC(cm)
  stab_tab$Min_Size[k - 1] <- min(table(cl))
  if (n_samples > k) {
    stab_tab$Silhouette[k - 1] <- mean(silhouette(cl, dist(t(feature_matrix)))[, 3])
  }
}
cat("\n  Consensus stability:\n")
print(stab_tab)
opt_k <- stab_tab %>% filter(Min_Size >= 30) %>%
  slice_min(PAC, n = 1) %>% pull(K)
if (length(opt_k) == 0) opt_k <- 3
cat(sprintf("  Optimal K: %d\n", opt_k))
write.csv(stab_tab, file.path(output_dir, "Consensus_Stability.csv"),
          row.names = FALSE)
# Bootstrap on latent space
bootstrap_completed <- FALSE
if (latent_available) {
  cat("\n  Bootstrap on VAE latent space...\n")
  lat_cols <- grep("^Z\\d+", colnames(latent_data), value = TRUE)
  lat_mat  <- as.matrix(latent_data[, lat_cols])
  orig_lab <- df$Subtype
  k <- length(unique(orig_lab))
  n_boot <- nrow(lat_mat)

  boot_mat <- matrix(NA, nrow = n_boot, ncol = n_bootstrap)
  ari_vals <- numeric(n_bootstrap)
  jac_vals <- numeric(n_bootstrap)

  for (b in 1:n_bootstrap) {
    set.seed(b)
    idx <- sample(1:n_boot, floor(n_boot * 0.8), replace = TRUE)
    bkm <- kmeans(lat_mat[idx, ], centers = k, nstart = 20, iter.max = 100)
    uid <- unique(idx)
    for (i in seq_along(uid)) {
      boot_mat[uid[i], b] <- bkm$cluster[which(idx == uid[i])[1]]
    }
    ari_vals[b] <- adjustedRandIndex(orig_lab[idx], bkm$cluster)
    jac_vals[b] <- calculate_jaccard_index(orig_lab[idx], bkm$cluster)
    if (b %% 25 == 0) cat(sprintf("    %d/%d\n", b, n_bootstrap))
  }

  samp_stab <- calculate_sample_stability(boot_mat, orig_lab)
  m_ari  <- mean(ari_vals, na.rm = TRUE)
  s_ari  <- sd(ari_vals, na.rm = TRUE)
  m_jac  <- mean(jac_vals, na.rm = TRUE)
  s_jac  <- sd(jac_vals, na.rm = TRUE)
  m_stab <- mean(samp_stab, na.rm = TRUE)
  n_stab <- sum(samp_stab > stability_threshold, na.rm = TRUE)
  n_val  <- sum(!is.na(samp_stab))

  sil_lat <- silhouette(orig_lab, dist(lat_mat))
  avg_sil <- mean(sil_lat[, 3])

  cat(sprintf("\n  Bootstrap results:\n"))
  cat(sprintf("    ARI: %.3f +/- %.3f\n", m_ari, s_ari))
  cat(sprintf("    Jaccard: %.3f +/- %.3f\n", m_jac, s_jac))
  cat(sprintf("    Sample stability: %.3f\n", m_stab))
  cat(sprintf("    Stable (>%.2f): %d/%d (%.1f%%)\n",
              stability_threshold, n_stab, n_val, 100 * n_stab / n_val))
  cat(sprintf("    Silhouette: %.3f\n", avg_sil))

  boot_sum <- data.frame(
    Metric = c("ARI_Mean", "ARI_SD", "Jaccard_Mean", "Jaccard_SD",
               "Sample_Stability", "Stable_Fraction", "Silhouette"),
    Value = c(m_ari, s_ari, m_jac, s_jac, m_stab, n_stab / n_val, avg_sil))
  write.csv(boot_sum, file.path(output_dir, "Bootstrap_Summary.csv"),
            row.names = FALSE)

  stab_df <- data.frame(ID = df$ID, Subtype = df$Subtype,
                        Stability = samp_stab,
                        Stable = samp_stab > stability_threshold)
  write.csv(stab_df, file.path(output_dir, "Bootstrap_Sample_Stability.csv"),
            row.names = FALSE)

  bootstrap_completed <- TRUE
}
# ============================================================================
# [7/7] Visualizations + Summary
# ============================================================================
cat("\n[7/7] Visualizations...\n")
# 7.1 Consensus evaluation
png(file.path(output_dir, "Consensus_Evaluation.png"),
    width = 4800, height = 3600, res = 300)
par(mfrow = c(2, 2), mar = c(5, 5, 4, 2))
plot(stab_tab$K, stab_tab$PAC, type = "b", pch = 19, cex = 2.5, lwd = 3,
     col = "steelblue", xlab = "K", ylab = "PAC", main = "PAC (lower=better)")
points(opt_k, stab_tab$PAC[opt_k - 1], pch = 19, cex = 4, col = "red")
plot(stab_tab$K, stab_tab$Silhouette, type = "b", pch = 19, cex = 2.5,
     lwd = 3, col = "coral", xlab = "K", ylab = "Silhouette",
     main = "Silhouette (higher=better)")
points(opt_k, stab_tab$Silhouette[opt_k - 1], pch = 19, cex = 4, col = "red")
barplot(stab_tab$Min_Size, names.arg = paste0("K=", stab_tab$K),
        col = ifelse(stab_tab$K == opt_k, "red", "steelblue"),
        main = "Min Cluster Size", ylab = "N")
abline(h = 30, lty = 2, col = "darkgreen", lwd = 2)
plot.new()
text(0.5, 0.85, "Summary", cex = 1.5, font = 2)
text(0.5, 0.7, sprintf("Optimal K: %d", opt_k), cex = 1.2)
text(0.5, 0.6, sprintf("PAC: %.4f", stab_tab$PAC[opt_k - 1]), cex = 1.2)
text(0.5, 0.5, sprintf("Silhouette: %.3f", stab_tab$Silhouette[opt_k - 1]),
     cex = 1.2)
if (bootstrap_completed) {
  text(0.5, 0.35, sprintf("ARI: %.3f +/- %.3f", m_ari, s_ari), cex = 1.1)
  text(0.5, 0.25, sprintf("Jaccard: %.3f +/- %.3f", m_jac, s_jac), cex = 1.1)
}
dev.off()
cat("  Saved: Consensus_Evaluation.png\n")
# 7.2 Bootstrap plots
if (bootstrap_completed) {
  png(file.path(output_dir, "Bootstrap_Plots.png"),
      width = 4500, height = 3000, res = 300)
  par(mfrow = c(2, 3), mar = c(5, 5, 4, 2))

  hist(ari_vals, breaks = 30, col = "steelblue", border = "white",
       main = sprintf("ARI (mean=%.3f)", m_ari), xlab = "ARI")
  abline(v = m_ari, col = "red", lwd = 2)

  hist(jac_vals, breaks = 30, col = "coral", border = "white",
       main = sprintf("Jaccard (mean=%.3f)", m_jac), xlab = "Jaccard")
  abline(v = m_jac, col = "red", lwd = 2)

  hist(samp_stab[!is.na(samp_stab)], breaks = 30, col = "seagreen",
       border = "white",
       main = sprintf("Sample Stability (%.1f%% stable)",
                      100 * n_stab / n_val),
       xlab = "Stability")
  abline(v = stability_threshold, col = "red", lwd = 2, lty = 2)

  boxplot(sil_lat[, 3] ~ orig_lab, col = rainbow(k),
          main = "Silhouette by Subtype", xlab = "Subtype", ylab = "Silhouette")

  plot(1:length(samp_stab), samp_stab, pch = 19,
       col = rainbow(k)[orig_lab], main = "Sample Stability",
       xlab = "Index", ylab = "Stability")
  abline(h = stability_threshold, lty = 2, col = "red", lwd = 2)

  barplot(c(ARI = m_ari, Jaccard = m_jac, Stability = m_stab,
            Silhouette = avg_sil),
          col = c("steelblue", "coral", "seagreen", "purple"),
          main = "Overall Metrics", ylab = "Value", ylim = c(0, 1))
  abline(h = 0.8, lty = 2, col = "darkgreen", lwd = 2)

  dev.off()
  cat("  Saved: Bootstrap_Plots.png\n")
}
# 7.3 Conversion rate bar plot
png(file.path(output_dir, "Conversion_Rates_Barplot.png"),
    width = 2400, height = 1800, res = 300)
bp <- barplot(conv_table$rate * 100, names.arg = conv_table$Subtype_F,
              col = c("steelblue", "goldenrod", "firebrick"),
              ylim = c(0, 80), ylab = "Conversion Rate (%)",
              main = sprintf("AD Conversion by Subtype (Chi2 P=%.4f)",
                             chi_test$p.value))
# Error bars
arrows(bp, conv_table$ci_lower * 100, bp, conv_table$ci_upper * 100,
       angle = 90, code = 3, length = 0.1, lwd = 2)
# N labels
text(bp, conv_table$rate * 100 + 3,
     sprintf("n=%d\n%.1f%%", conv_table$n, conv_table$rate * 100),
     cex = 0.8)
dev.off()
cat("  Saved: Conversion_Rates_Barplot.png\n")
# 7.4 MRI eta-squared bar plot (top 15)
png(file.path(output_dir, "MRI_Eta_Squared_Top15.png"),
    width = 3000, height = 2000, res = 300)
top15 <- head(mri_results, 15)
par(mar = c(8, 5, 4, 2))
bp2 <- barplot(top15$Eta_sq, names.arg = top15$Feature,
               col = ifelse(top15$P_FDR < 0.05, "firebrick", "grey70"),
               las = 2, ylab = expression(eta^2),
               main = "MRI Features: Between-Subtype Effect Sizes")
abline(h = 0.06, lty = 2, col = "blue")  # medium effect
abline(h = 0.14, lty = 2, col = "red")   # large effect
legend("topright", legend = c("FDR<0.05", "n.s.", "medium", "large"),
       fill = c("firebrick", "grey70", NA, NA),
       border = c("black", "black", NA, NA),
       lty = c(NA, NA, 2, 2), col = c(NA, NA, "blue", "red"), cex = 0.8)
dev.off()
cat("  Saved: MRI_Eta_Squared_Top15.png\n")
# ============================================================================
# Summary Report
# ============================================================================
cat("\n========================================================================\n")
cat("  SUMMARY REPORT\n")
cat("========================================================================\n\n")
cat("--- Conversion Rates ---\n")
for (i in 1:nrow(conv_table)) {
  cat(sprintf("  Subtype %d (%s): %.1f%% (95%%CI: %.1f%%-%.1f%%), n=%d\n",
              conv_table$Subtype[i], conv_table$Subtype_F[i],
              conv_table$rate[i] * 100,
              conv_table$ci_lower[i] * 100, conv_table$ci_upper[i] * 100,
              conv_table$n[i]))
}
cat(sprintf("  Chi-square P=%.4f, Fisher P=%.4f\n\n",
            chi_test$p.value, fisher_test$p.value))
cat("--- Cox Regression (adjusted) ---\n")
adj_rows <- hr_all[hr_all$Model == "Adj_Age_Sex_Edu", ]
for (i in 1:nrow(adj_rows)) {
  cat(sprintf("  %s: HR=%.3f (%.3f-%.3f), P=%.4f\n",
              adj_rows$Comparison[i], adj_rows$HR[i],
              adj_rows$Lower95[i], adj_rows$Upper95[i], adj_rows$P[i]))
}
cat(sprintf("\n--- MRI ANOVA ---\n"))
cat(sprintf("  Significant (FDR<0.05): %d / %d\n",
            sum(mri_results$P_FDR < 0.05), nrow(mri_results)))
cat(sprintf("  Top feature: %s (eta2=%.3f)\n",
            mri_results$Feature[1], mri_results$Eta_sq[1]))
cat(sprintf("\n--- ANCOVA (adjusted MMSE+AGE+SEX) ---\n"))
cat(sprintf("  Significant (FDR<0.05): %d / %d\n",
            sum(ancova_results$P_FDR < 0.05), nrow(ancova_results)))
cat(sprintf("\n--- Sex Distribution ---\n"))
cat(sprintf("  Chi-square P=%.2e\n", sex_chi$p.value))
if (bootstrap_completed) {
  cat(sprintf("\n--- Bootstrap Stability ---\n"))
  cat(sprintf("  ARI: %.3f +/- %.3f\n", m_ari, s_ari))
  cat(sprintf("  Jaccard: %.3f +/- %.3f\n", m_jac, s_jac))
  cat(sprintf("  Silhouette: %.3f\n", avg_sil))
}
cat(sprintf("\n--- Consensus Clustering ---\n"))
cat(sprintf("  Optimal K: %d (PAC=%.4f)\n", opt_k, stab_tab$PAC[opt_k - 1]))
cat("\n\nAll output files in:", output_dir, "\n")
cat("========================================================================\n")
cat("Step 14 (K=3) Complete.\n")
cat("========================================================================\n")

