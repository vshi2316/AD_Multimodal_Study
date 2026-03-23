################################################################################
# Longitudinal Fox Lab BSI validation of VAE-derived subtypes
################################################################################
suppressPackageStartupMessages({
  library(optparse)
  library(data.table)
  library(dplyr)
  library(tidyr)
  library(survival)
  library(survminer)
  library(ggplot2)
  library(lme4)
  library(lmerTest)
  library(emmeans)
})

option_list <- list(
  make_option(c("--bsi_file"), type = "character", default = "FOXLABBSI_02Mar2026.csv",
              help = "Fox Lab BSI longitudinal file [default: %default]"),
  make_option(c("--vae_dir"), type = "character", default = ".",
              help = "Directory containing subtype_assignments.csv and latent_representations.csv [default: %default]"),
  make_option(c("--clinical_file"), type = "character", default = "Clinical_data.csv",
              help = "Discovery cohort clinical file [default: %default]"),
  make_option(c("--output_dir"), type = "character", default = "./step15_results",
              help = "Output directory [default: %default]")
)

opt <- parse_args(OptionParser(option_list = option_list))

set.seed(42)
cat("================================================================\\n")
cat("  Step 15: Longitudinal Fox Lab BSI Atrophy Validation\\n")
cat("================================================================\\n\\n")

bsi_file <- opt$bsi_file
vae_dir <- opt$vae_dir
cohort_file <- opt$clinical_file
output_dir <- opt$output_dir
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

bsi_metrics <- list(
  list(name = "Whole Brain BSI", col = "DBCBBSI", direction = "negative_is_worse"),
  list(name = "KN-corrected BSI", col = "KMNDBCBBSI", direction = "negative_is_worse"),
  list(name = "Ventricular BSI", col = "VBSI", direction = "positive_is_worse"),
  list(name = "R Hippocampal BSI", col = "HBSI_R", direction = "negative_is_worse"),
  list(name = "L Hippocampal BSI", col = "HBSI_L", direction = "negative_is_worse")
)

# ============================================================================
# 1. Load discovery cohort and VAE subtype assignments
# ============================================================================
cat("[1/8] 加载discovery cohort和VAE亚型...\n")
cohort <- fread(cohort_file)
cat("  Discovery cohort: N =", nrow(cohort), "\n")
cat("  ID前5个:", paste(head(cohort$ID, 5), collapse = ", "), "\n")
assignments <- fread(file.path(vae_dir, "subtype_assignments.csv"))
latent      <- fread(file.path(vae_dir, "latent_representations.csv"))
cat("  VAE亚型分布:\n")
print(table(assignments$VAE_Subtype))
cohort_vae <- merge(cohort, assignments[, .(ID, VAE_Subtype)], by = "ID")
z_cols <- character(0)
if ("Z1" %in% colnames(latent)) {
  z_cols <- grep("^Z\\d+$", colnames(latent), value = TRUE)
  cohort_vae <- merge(cohort_vae, latent[, c("ID", z_cols), with = FALSE], by = "ID")
}
cohort_vae$PTID <- as.character(cohort_vae$ID)
cat("  合并后 N =", nrow(cohort_vae), "\n")
# ============================================================================
# ============================================================================
cat("\n[2/8] 加载Fox Lab BSI纵向数据...\n")
bsi <- fread(bsi_file)
cat("  BSI数据: N =", nrow(bsi), "行,", length(unique(bsi$PTID)), "人\n")
cat("  PHASE分布:\n")
print(table(bsi$PHASE, useNA = "ifany"))
discovery_ptids <- unique(cohort_vae$PTID)
bsi_matched <- bsi[PTID %in% discovery_ptids]
n_matched <- length(unique(bsi_matched$PTID))
cat("\n  ★ PTID匹配结果:", n_matched, "/", length(discovery_ptids), 
    sprintf("(%.1f%%)\n", n_matched / length(discovery_ptids) * 100))
if (n_matched == 0) {
  stop("无法匹配任何受试者到BSI数据！请检查PTID格式。")
}
unmatched <- setdiff(discovery_ptids, unique(bsi_matched$PTID))
if (length(unmatched) > 0) {
  cat("  未匹配受试者:", length(unmatched), "人\n")
  cat("  未匹配PTID前10个:", paste(head(unmatched, 10), collapse = ", "), "\n")
}
if ("QC_PASS" %in% colnames(bsi_matched)) {
  n_before <- nrow(bsi_matched)
  bsi_matched <- bsi_matched[QC_PASS == 1 | is.na(QC_PASS)]
  cat("  QC过滤: ", n_before, "→", nrow(bsi_matched), "行\n")
}
# ============================================================================
# ============================================================================
cat("\n[3/8] 计算纵向时间...\n")
visit_to_months <- function(viscode) {
  viscode <- tolower(trimws(viscode))
  dplyr::case_when(
    viscode %in% c("bl", "sc", "scmri") ~ 0,
    grepl("^m(\\d+)$", viscode) ~ as.numeric(gsub("^m", "", viscode)),
    grepl("^y(\\d+)$", viscode) ~ as.numeric(gsub("^y", "", viscode)) * 12,
    TRUE ~ NA_real_
  )
}
bsi_matched$Months <- visit_to_months(bsi_matched$VISCODE2)
viscode_coverage <- sum(!is.na(bsi_matched$Months)) / nrow(bsi_matched)
cat("  VISCODE2覆盖率:", round(viscode_coverage * 100, 1), "%\n")
if (viscode_coverage < 0.5) {
  cat("  VISCODE2覆盖率低，用EXAMDATE补充...\n")
  bsi_matched$EXAMDATE_d <- as.Date(bsi_matched$EXAMDATE, format = "%Y-%m-%d")
  if (sum(is.na(bsi_matched$EXAMDATE_d)) > nrow(bsi_matched) * 0.5) {
    bsi_matched$EXAMDATE_d <- as.Date(bsi_matched$EXAMDATE, format = "%m/%d/%Y")
  }

  bl_dates <- bsi_matched %>%
    group_by(PTID) %>%
    summarise(bl_date = min(EXAMDATE_d, na.rm = TRUE), .groups = "drop")
  bsi_matched <- merge(bsi_matched, bl_dates, by = "PTID", all.x = TRUE)
  bsi_matched$Months_date <- as.numeric(difftime(bsi_matched$EXAMDATE_d,
                                                 bsi_matched$bl_date,
                                                 units = "days")) / 30.44
  bsi_matched$Months <- ifelse(is.na(bsi_matched$Months),
                               bsi_matched$Months_date,
                               bsi_matched$Months)
}
bsi_matched$Years <- bsi_matched$Months / 12
bsi_clean <- bsi_matched %>%
  filter(!is.na(Months) & !is.na(DBCBBSI)) %>%
  as.data.frame()
cat("  有效纵向数据点:", nrow(bsi_clean), "\n")
cat("  涉及受试者:", length(unique(bsi_clean$PTID)), "\n")
cat("  每人访视数: 中位数=", median(table(bsi_clean$PTID)),
    ", 范围=", paste(range(table(bsi_clean$PTID)), collapse = "-"), "\n")
cat("  随访时间范围:", round(range(bsi_clean$Months, na.rm = TRUE), 1), "月\n")
# ============================================================================
# ============================================================================
cat("\n[4/8] 合并VAE亚型...\n")
merge_cols <- c("PTID", "VAE_Subtype",
                intersect(c("AGE", "SEX", "EDUCATION", "APOE4_DOSAGE", "MMSE", z_cols),
                          colnames(cohort_vae)))
bsi_analysis <- merge(bsi_clean, cohort_vae[, ..merge_cols], by = "PTID")
cat("  合并后数据点:", nrow(bsi_analysis), "\n")
cat("  涉及受试者:", length(unique(bsi_analysis$PTID)), "\n")
bsi_analysis$Subtype_F <- factor(bsi_analysis$VAE_Subtype,
                                 levels = sort(unique(bsi_analysis$VAE_Subtype)),
                                 labels = paste0("Subtype ", sort(unique(bsi_analysis$VAE_Subtype))))
cat("\n  亚型分布（纵向数据中）:\n")
subtype_dist <- bsi_analysis %>%
  group_by(Subtype_F) %>%
  summarise(
    n_subjects = length(unique(PTID)),
    n_visits = n(),
    median_visits = median(table(PTID)),
    max_years = round(max(Years, na.rm = TRUE), 1),
    .groups = "drop"
  )
print(as.data.frame(subtype_dist))
n_subjects <- length(unique(bsi_analysis$PTID))
if (n_subjects < 15) {
  cat(sprintf("\n  ⚠ 仅%d人有纵向BSI数据，统计功效有限\n", n_subjects))
}
# ============================================================================
# ============================================================================
cat("\n[5/8] 计算个体萎缩斜率（多指标）...\n")
all_slopes <- list()
for (metric in bsi_metrics) {
  mcol <- metric$col
  mname <- metric$name
  mdir <- metric$direction

  if (!mcol %in% colnames(bsi_analysis)) {
    cat(sprintf("  %s (%s): 列不存在，跳过\n", mname, mcol))
    next
  }

  slopes_this <- bsi_analysis %>%
    filter(!is.na(.data[[mcol]])) %>%
    group_by(PTID, VAE_Subtype) %>%
    filter(n() >= 2) %>%
    summarise(
      n_visits = n(),
      followup_months = max(Months) - min(Months),
      baseline_val = .data[[mcol]][which.min(Months)],
      last_val = .data[[mcol]][which.max(Months)],
      slope_per_year = ifelse(
        followup_months > 0,
        coef(lm(.data[[mcol]] ~ I(Months / 12)))[2],
        NA_real_
      ),
      .groups = "drop"
    ) %>%
    filter(!is.na(slope_per_year) & followup_months >= 6)

  slopes_this$metric <- mcol
  slopes_this$metric_name <- mname
  slopes_this$direction <- mdir

  cat(sprintf("  %s: %d人有纵向数据\n", mname, nrow(slopes_this)))
  all_slopes[[mcol]] <- slopes_this
}
slope_all <- bind_rows(all_slopes)
cat("  总计:", nrow(slope_all), "个人-指标组合\n")
slope_brain <- all_slopes[["DBCBBSI"]]
if (is.null(slope_brain) || nrow(slope_brain) < 5) {
  cat("  ⚠ DBCBBSI数据不足，尝试KMNDBCBBSI...\n")
  slope_brain <- all_slopes[["KMNDBCBBSI"]]
}
if (is.null(slope_brain) || nrow(slope_brain) < 5) {
  stop("BSI数据不足以进行分析。请检查数据完整性。")
}
covar_cols <- intersect(c("AGE", "SEX", "EDUCATION", "APOE4_DOSAGE", "MMSE", z_cols),
                        colnames(cohort_vae))
slope_brain <- merge(slope_brain, cohort_vae[, c("PTID", covar_cols), with = FALSE],
                     by = "PTID", all.x = TRUE)
slope_brain$Subtype_F <- factor(slope_brain$VAE_Subtype,
                                levels = sort(unique(slope_brain$VAE_Subtype)),
                                labels = paste0("S", sort(unique(slope_brain$VAE_Subtype))))
cat("\n  全脑BSI萎缩斜率描述统计（ml/year）:\n")
slope_summary <- slope_brain %>%
  group_by(Subtype_F) %>%
  summarise(
    n = n(),
    mean_slope = mean(slope_per_year, na.rm = TRUE),
    sd_slope = sd(slope_per_year, na.rm = TRUE),
    median_slope = median(slope_per_year, na.rm = TRUE),
    mean_fu_months = mean(followup_months, na.rm = TRUE),
    .groups = "drop"
  )
print(as.data.frame(slope_summary))
# ============================================================================
# ============================================================================
cat("\n[6/8] 核心统计分析...\n")
available_covars <- c()
for (cv in c("AGE", "SEX", "EDUCATION")) {
  if (cv %in% colnames(slope_brain) && sum(!is.na(slope_brain[[cv]])) > nrow(slope_brain) * 0.7) {
    available_covars <- c(available_covars, cv)
  }
}
cat("  可用协变量:", paste(available_covars, collapse = ", "), "\n")
# ─────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────
cat("\n  === 6a. 多指标ANOVA汇总 ===\n")
anova_results <- list()
for (mcol in names(all_slopes)) {
  sdata <- all_slopes[[mcol]]
  if (nrow(sdata) < 10 || length(unique(sdata$VAE_Subtype)) < 2) next

  aov_res <- aov(slope_per_year ~ factor(VAE_Subtype), data = sdata)
  aov_s <- summary(aov_res)
  ss_b <- aov_s[[1]][1, "Sum Sq"]
  ss_t <- sum(aov_s[[1]][, "Sum Sq"])
  eta2 <- ss_b / ss_t
  p_val <- aov_s[[1]][1, "Pr(>F)"]

  kw <- kruskal.test(slope_per_year ~ factor(VAE_Subtype), data = sdata)

  anova_results[[mcol]] <- data.frame(
    Metric = mcol,
    N = nrow(sdata),
    ANOVA_F = round(aov_s[[1]][1, "F value"], 3),
    ANOVA_P = round(p_val, 4),
    Eta_sq = round(eta2, 4),
    KW_P = round(kw$p.value, 4),
    Significant = p_val < 0.05,
    stringsAsFactors = FALSE
  )

  sig_mark <- ifelse(p_val < 0.05, "★★★", ifelse(p_val < 0.10, "△", ""))
  cat(sprintf("  %-15s: F=%.3f, P=%.4f, η²=%.4f, KW P=%.4f %s (N=%d)\n",
              mcol, aov_s[[1]][1, "F value"], p_val, eta2, kw$p.value, sig_mark, nrow(sdata)))
}
anova_df <- bind_rows(anova_results)
# ─────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────
cat("\n  === 6b. 全脑BSI (DBCBBSI) 详细分析 ===\n")
aov_main <- aov(slope_per_year ~ factor(VAE_Subtype), data = slope_brain)
cat("  未校正ANOVA:\n")
print(summary(aov_main))
aov_s <- summary(aov_main)
eta_sq_main <- aov_s[[1]][1, "Sum Sq"] / sum(aov_s[[1]][, "Sum Sq"])
cat(sprintf("  η² = %.4f\n", eta_sq_main))
kw_main <- kruskal.test(slope_per_year ~ factor(VAE_Subtype), data = slope_brain)
cat(sprintf("  Kruskal-Wallis: χ²=%.3f, P=%.4f\n", kw_main$statistic, kw_main$p.value))
# ANCOVA
if (length(available_covars) > 0) {
  ancova_f <- as.formula(paste("slope_per_year ~ factor(VAE_Subtype) +",
                               paste(available_covars, collapse = " + ")))
  ancova_res <- aov(ancova_f, data = slope_brain)
  cat("\n  ANCOVA (校正", paste(available_covars, collapse = "+"), "):\n")
  print(summary(ancova_res))
}
# Tukey HSD
if (length(unique(slope_brain$VAE_Subtype)) >= 3) {
  cat("\n  Tukey HSD 事后比较:\n")
  print(TukeyHSD(aov_main))
}
# ─────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────
cat("\n  === 6c. 线性混合模型: 纵向BSI轨迹 (修复版) ===\n")
bsi_lmm <- bsi_analysis %>%
  group_by(PTID) %>%
  filter(n() >= 2) %>%
  ungroup() %>%
  as.data.frame()
cat(sprintf("  LMM数据: %d人, %d观测 (过滤≥2访视/人)\n",
            length(unique(bsi_lmm$PTID)), nrow(bsi_lmm)))
bsi_lmm$DBCBBSI_sc <- bsi_lmm$DBCBBSI / 10
bsi_lmm$Subtype_LMM <- factor(bsi_lmm$VAE_Subtype)
visits_per_subj <- table(bsi_lmm$PTID)
cat(sprintf("  每人访视数: 中位=%d, 范围=%d-%d\n",
            median(visits_per_subj), min(visits_per_subj), max(visits_per_subj)))
bsi_lmm <- bsi_lmm %>%
  group_by(PTID) %>%
  filter(sd(Years, na.rm = TRUE) > 0.01) %>%
  ungroup() %>%
  as.data.frame()
cat(sprintf("  去除零方差后: %d人, %d观测\n",
            length(unique(bsi_lmm$PTID)), nrow(bsi_lmm)))
lmm_interaction_p <- NA
lmm_model_type <- "none"
cat("\n  尝试1: DBCBBSI_sc ~ Years * Subtype + (1 + Years | PTID), bobyqa\n")
tryCatch({
  lmm1 <- lmer(DBCBBSI_sc ~ Years * Subtype_LMM + (1 + Years | PTID),
               data = bsi_lmm, REML = FALSE,
               control = lmerControl(optimizer = "bobyqa",
                                     optCtrl = list(maxfun = 50000)))
  anova_lmm <- anova(lmm1)
  cat("  ✓ 成功! ANOVA (Type III):\n")
  print(anova_lmm)

  lmm_interaction_p <- anova_lmm["Years:Subtype_LMM", "Pr(>F)"]
  lmm_model_type <- "random_slope_bobyqa"

  cat("\n  每个亚型的估计萎缩斜率 (emtrends, 单位=10ml/year):\n")
  emm_slopes <- emtrends(lmm1, ~ Subtype_LMM, var = "Years")
  print(summary(emm_slopes))
  cat("\n  斜率两两对比:\n")
  print(summary(pairs(emm_slopes)))

}, error = function(e) {
  cat("  ✗ 失败:", e$message, "\n")
})
if (is.na(lmm_interaction_p)) {
  cat("\n  尝试2: 同模型, nlminb优化器\n")
  tryCatch({
    lmm1b <- lmer(DBCBBSI_sc ~ Years * Subtype_LMM + (1 + Years | PTID),
                  data = bsi_lmm, REML = FALSE,
                  control = lmerControl(optimizer = "nlminbw",
                                        optCtrl = list(maxiter = 50000)))
    anova_lmm <- anova(lmm1b)
    cat("  ✓ 成功! ANOVA (Type III):\n")
    print(anova_lmm)

    lmm_interaction_p <- anova_lmm["Years:Subtype_LMM", "Pr(>F)"]
    lmm_model_type <- "random_slope_nlminb"

    emm_slopes <- emtrends(lmm1b, ~ Subtype_LMM, var = "Years")
    cat("\n  emtrends:\n")
    print(summary(emm_slopes))
    cat("\n  斜率对比:\n")
    print(summary(pairs(emm_slopes)))

  }, error = function(e) {
    cat("  ✗ 失败:", e$message, "\n")
  })
}
if (is.na(lmm_interaction_p)) {
  cat("\n  尝试3: DBCBBSI_sc ~ Years * Subtype + (1 | PTID)\n")
  tryCatch({
    lmm1c <- lmer(DBCBBSI_sc ~ Years * Subtype_LMM + (1 | PTID),
                  data = bsi_lmm, REML = FALSE,
                  control = lmerControl(optimizer = "bobyqa",
                                        optCtrl = list(maxfun = 50000)))
    anova_lmm <- anova(lmm1c)
    cat("  ✓ 成功! ANOVA (Type III):\n")
    print(anova_lmm)

    lmm_interaction_p <- anova_lmm["Years:Subtype_LMM", "Pr(>F)"]
    lmm_model_type <- "random_intercept"

    emm_slopes <- emtrends(lmm1c, ~ Subtype_LMM, var = "Years")
    cat("\n  emtrends:\n")
    print(summary(emm_slopes))
    cat("\n  斜率对比:\n")
    print(summary(pairs(emm_slopes)))

  }, error = function(e) {
    cat("  ✗ 失败:", e$message, "\n")
  })
}
if (is.na(lmm_interaction_p)) {
  cat("\n  尝试4: 随机截距, Nelder_Mead优化器\n")
  tryCatch({
    lmm1d <- lmer(DBCBBSI_sc ~ Years * Subtype_LMM + (1 | PTID),
                  data = bsi_lmm, REML = FALSE,
                  control = lmerControl(optimizer = "Nelder_Mead",
                                        optCtrl = list(maxfun = 50000)))
    anova_lmm <- anova(lmm1d)
    cat("  ✓ 成功! ANOVA (Type III):\n")
    print(anova_lmm)

    lmm_interaction_p <- anova_lmm["Years:Subtype_LMM", "Pr(>F)"]
    lmm_model_type <- "random_intercept_NM"

    emm_slopes <- emtrends(lmm1d, ~ Subtype_LMM, var = "Years")
    cat("\n  emtrends:\n")
    print(summary(emm_slopes))
    cat("\n  斜率对比:\n")
    print(summary(pairs(emm_slopes)))

  }, error = function(e) {
    cat("  ✗ 所有LMM尝试均失败:", e$message, "\n")
  })
}
if (!is.na(lmm_interaction_p)) {
  cat(sprintf("\n  ★ LMM Time × Subtype 交互 P = %.4f (模型: %s) %s\n",
              lmm_interaction_p, lmm_model_type,
              ifelse(lmm_interaction_p < 0.05, "★★★ 显著！",
                     ifelse(lmm_interaction_p < 0.10, "△ 边缘显著", "→ 不显著"))))
} else {
  cat("\n  ⚠ 所有LMM模型均失败，无法获得交互项P值\n")
}
if (!is.na(lmm_interaction_p) && length(available_covars) > 0) {
  cat("\n  === LMM + 协变量 ===\n")

  covars_in_lmm <- intersect(available_covars, colnames(bsi_lmm))

  if (length(covars_in_lmm) > 0) {
    if (lmm_model_type %in% c("random_slope_bobyqa", "random_slope_nlminb")) {
      lmm2_f <- as.formula(paste("DBCBBSI_sc ~ Years * Subtype_LMM +",
                                 paste(covars_in_lmm, collapse = " + "),
                                 "+ (1 + Years | PTID)"))
    } else {
      lmm2_f <- as.formula(paste("DBCBBSI_sc ~ Years * Subtype_LMM +",
                                 paste(covars_in_lmm, collapse = " + "),
                                 "+ (1 | PTID)"))
    }
    tryCatch({
      lmm2 <- lmer(lmm2_f, data = bsi_lmm, REML = FALSE,
                   control = lmerControl(optimizer = "bobyqa",
                                         optCtrl = list(maxfun = 50000)))
      anova_lmm2 <- anova(lmm2)
      cat("  校正后 ANOVA:\n")
      print(anova_lmm2)
      adj_interaction_p <- anova_lmm2["Years:Subtype_LMM", "Pr(>F)"]
      cat(sprintf("  校正后 Time × Subtype P = %.4f\n", adj_interaction_p))
    }, error = function(e) cat("  校正LMM失败:", e$message, "\n"))
  }
}
# ─────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────
cat("\n  === 6d. 连续Z维度 × 萎缩斜率 ===\n")
if (length(z_cols) > 0) {
  for (zc in z_cols) {
    if (zc %in% colnames(slope_brain) && sum(!is.na(slope_brain[[zc]])) > 10) {
      ct <- cor.test(slope_brain[[zc]], slope_brain$slope_per_year, method = "pearson")
      cat(sprintf("  %s vs 全脑BSI斜率: r=%.3f, P=%.4f %s\n",
                  zc, ct$estimate, ct$p.value,
                  ifelse(ct$p.value < 0.05, "★", "")))
    }
  }
}
# ─────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────
cat("\n  === 6e. Cox回归: 萎缩斜率 → AD转化 ===\n")
surv_file <- file.path(vae_dir, "real_followup_analysis/merged_survival_data.csv")
if (file.exists(surv_file)) {
  surv_data <- fread(surv_file)

  if ("PTID" %in% colnames(surv_data)) {
    surv_slope <- merge(surv_data, slope_brain[, c("PTID", "slope_per_year", "baseline_val")],
                        by = "PTID")
  } else if ("ID" %in% colnames(surv_data)) {
    # ID = PTID in this dataset
    slope_brain$ID <- slope_brain$PTID
    surv_slope <- merge(surv_data, slope_brain[, c("ID", "slope_per_year", "baseline_val")],
                        by = "ID")
  } else {
    surv_slope <- NULL
  }

  if (!is.null(surv_slope) && nrow(surv_slope) > 10) {
    cat("  Cox回归样本量:", nrow(surv_slope), "\n")
    surv_slope$slope_std <- scale(surv_slope$slope_per_year)[, 1]

    cox1 <- coxph(Surv(followup_years, event) ~ slope_std, data = surv_slope)
    s1 <- summary(cox1)
    cat(sprintf("\n  全脑BSI斜率 Cox (未校正): HR=%.3f (%.3f-%.3f), P=%.4f, C=%.3f\n",
                s1$conf.int[1,1], s1$conf.int[1,3], s1$conf.int[1,4],
                s1$coefficients[1,5], s1$concordance[1]))

    covars_avail <- intersect(available_covars, colnames(surv_slope))
    if (length(covars_avail) > 0) {
      cox2_f <- as.formula(paste("Surv(followup_years, event) ~ slope_std +",
                                 paste(covars_avail, collapse = " + ")))
      cox2 <- coxph(cox2_f, data = surv_slope)
      s2 <- summary(cox2)
      cat(sprintf("  校正后: HR=%.3f (%.3f-%.3f), P=%.4f, C=%.3f\n",
                  s2$conf.int[1,1], s2$conf.int[1,3], s2$conf.int[1,4],
                  s2$coefficients[1,5], s2$concordance[1]))
    }

    st_col <- intersect(c("Subtype", "VAE_Subtype"), colnames(surv_slope))[1]
    if (!is.na(st_col)) {
      surv_slope$ST <- factor(surv_slope[[st_col]])
      cox3 <- coxph(Surv(followup_years, event) ~ slope_std + ST, data = surv_slope)
      cox_st <- coxph(Surv(followup_years, event) ~ ST, data = surv_slope)
      lr <- anova(cox_st, cox3)
      cat(sprintf("  萎缩斜率增量贡献 (控制亚型后): LR P = %.4f\n",
                  lr$`Pr(>|Chi|)`[2]))
    }
  } else {
    cat("  匹配到生存数据的样本量不足，跳过Cox回归\n")
  }
} else {
  cat("  未找到step14b生存数据，跳过Cox回归\n")
}
# ─────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────
cat("\n  === 6f. K=2 BSI分析 ===\n")
# K-means K=2 on Z1-Z3 latent space
if (length(z_cols) > 0 && nrow(slope_brain) >= 10) {

  z_for_k2 <- as.matrix(slope_brain[, z_cols, drop = FALSE])

  if (all(complete.cases(z_for_k2))) {
    km2 <- kmeans(z_for_k2, centers = 2, nstart = 25)
    slope_brain$K2_cluster <- km2$cluster

    if (file.exists(surv_file)) {
      surv_for_k2 <- fread(surv_file)

      if ("PTID" %in% colnames(surv_for_k2)) {
        k2_surv <- merge(slope_brain[, c("PTID", "K2_cluster", "slope_per_year",
                                         "VAE_Subtype", "baseline_val")],
                         surv_for_k2, by = "PTID")
      } else if ("ID" %in% colnames(surv_for_k2)) {
        slope_brain$ID <- slope_brain$PTID
        k2_surv <- merge(slope_brain[, c("ID", "PTID", "K2_cluster", "slope_per_year",
                                         "VAE_Subtype", "baseline_val")],
                         surv_for_k2, by = "ID")
      } else {
        k2_surv <- NULL
      }

      if (!is.null(k2_surv) && nrow(k2_surv) > 10) {
        conv_by_k2 <- tapply(k2_surv$event, k2_surv$K2_cluster, mean, na.rm = TRUE)
        k2_remap <- setNames(rank(conv_by_k2), names(conv_by_k2))
        k2_surv$K2_Group <- k2_remap[as.character(k2_surv$K2_cluster)]
        k2_surv$K2_F <- factor(k2_surv$K2_Group, levels = 1:2,
                               labels = c("Preserved", "Vulnerable"))

        slope_brain$K2_Group <- k2_remap[as.character(slope_brain$K2_cluster)]
        slope_brain$K2_F <- factor(slope_brain$K2_Group, levels = 1:2,
                                   labels = c("Preserved", "Vulnerable"))

        cat("  K=2分布:\n")
        for (k in 1:2) {
          n_k <- sum(k2_surv$K2_Group == k, na.rm = TRUE)
          n_e <- sum(k2_surv$K2_Group == k & k2_surv$event == 1, na.rm = TRUE)
          mean_slope <- mean(slope_brain$slope_per_year[slope_brain$K2_Group == k], na.rm = TRUE)
          cat(sprintf("    %s: n=%d, events=%d (%.1f%%), mean BSI slope=%.2f ml/yr\n",
                      c("Preserved", "Vulnerable")[k], n_k, n_e,
                      ifelse(n_k > 0, n_e / n_k * 100, 0), mean_slope))
        }

        # --- 6f-1. K=2 ANOVA / t-test ---
        cat("\n  --- K=2 萎缩率比较 ---\n")

        tt <- t.test(slope_per_year ~ K2_F, data = slope_brain)
        cat(sprintf("  Welch t-test: t=%.3f, P=%.4f\n", tt$statistic, tt$p.value))
        cat(sprintf("    Preserved mean=%.2f, Vulnerable mean=%.2f\n",
                    tt$estimate[1], tt$estimate[2]))

        # Wilcoxon
        wt <- wilcox.test(slope_per_year ~ K2_F, data = slope_brain)
        cat(sprintf("  Wilcoxon rank-sum: W=%.0f, P=%.4f\n", wt$statistic, wt$p.value))

        # ANOVA (for consistency)
        aov_k2 <- aov(slope_per_year ~ K2_F, data = slope_brain)
        aov_k2_s <- summary(aov_k2)
        eta2_k2 <- aov_k2_s[[1]][1, "Sum Sq"] / sum(aov_k2_s[[1]][, "Sum Sq"])
        cat(sprintf("  ANOVA: F=%.3f, P=%.4f, η²=%.4f\n",
                    aov_k2_s[[1]][1, "F value"],
                    aov_k2_s[[1]][1, "Pr(>F)"],
                    eta2_k2))

        # Cohen's d
        g1 <- slope_brain$slope_per_year[slope_brain$K2_Group == 1]
        g2 <- slope_brain$slope_per_year[slope_brain$K2_Group == 2]
        pooled_sd <- sqrt(((length(g1) - 1) * sd(g1)^2 + (length(g2) - 1) * sd(g2)^2) /
                            (length(g1) + length(g2) - 2))
        cohens_d <- (mean(g2) - mean(g1)) / pooled_sd
        cat(sprintf("  Cohen's d = %.3f\n", cohens_d))

        k2_ttest_p <- tt$p.value
        k2_anova_p <- aov_k2_s[[1]][1, "Pr(>F)"]

        # --- 6f-2. K=2 LMM ---
        cat("\n  --- K=2 LMM ---\n")

        k2_map <- slope_brain[, c("PTID", "K2_Group", "K2_F")]
        k2_map <- k2_map[!duplicated(k2_map$PTID), ]
        bsi_lmm_k2 <- merge(bsi_lmm, k2_map, by = "PTID")

        cat(sprintf("  K=2 LMM数据: %d人, %d观测\n",
                    length(unique(bsi_lmm_k2$PTID)), nrow(bsi_lmm_k2)))

        k2_lmm_p <- NA

        tryCatch({
          lmm_k2 <- lmer(DBCBBSI_sc ~ Years * K2_F + (1 + Years | PTID),
                         data = bsi_lmm_k2, REML = FALSE,
                         control = lmerControl(optimizer = "bobyqa",
                                               optCtrl = list(maxfun = 50000)))
          anova_k2_lmm <- anova(lmm_k2)
          cat("  K=2 LMM (随机斜率) ANOVA:\n")
          print(anova_k2_lmm)
          k2_lmm_p <- anova_k2_lmm["Years:K2_F", "Pr(>F)"]

          emm_k2 <- emtrends(lmm_k2, ~ K2_F, var = "Years")
          cat("\n  K=2 emtrends:\n")
          print(summary(emm_k2))
          cat("\n  K=2 斜率对比:\n")
          print(summary(pairs(emm_k2)))

        }, error = function(e) {
          cat("  K=2 随机斜率LMM失败:", e$message, "\n")
          cat("  尝试随机截距...\n")
          tryCatch({
            lmm_k2b <- lmer(DBCBBSI_sc ~ Years * K2_F + (1 | PTID),
                            data = bsi_lmm_k2, REML = FALSE,
                            control = lmerControl(optimizer = "bobyqa",
                                                  optCtrl = list(maxfun = 50000)))
            anova_k2_lmm <- anova(lmm_k2b)
            cat("  K=2 LMM (随机截距) ANOVA:\n")
            print(anova_k2_lmm)
            k2_lmm_p <<- anova_k2_lmm["Years:K2_F", "Pr(>F)"]

            emm_k2 <- emtrends(lmm_k2b, ~ K2_F, var = "Years")
            cat("\n  K=2 emtrends:\n")
            print(summary(emm_k2))
          }, error = function(e2) {
            cat("  K=2 随机截距LMM也失败:", e2$message, "\n")
          })
        })

        if (!is.na(k2_lmm_p)) {
          cat(sprintf("\n  ★ K=2 LMM Time × Group P = %.4f %s\n",
                      k2_lmm_p,
                      ifelse(k2_lmm_p < 0.05, "★★★ 显著！",
                             ifelse(k2_lmm_p < 0.10, "△ 边缘显著", "→ 不显著"))))
        }

        cat("\n  --- K=2 Cox回归 ---\n")

        k2_surv$K2_cox <- relevel(factor(k2_surv$K2_Group), ref = "1")

        cox_k2_uni <- coxph(Surv(followup_years, event) ~ K2_cox, data = k2_surv)
        s_k2u <- summary(cox_k2_uni)
        cat(sprintf("  K=2 未校正 Cox: HR=%.3f (%.3f-%.3f), P=%.4f\n",
                    s_k2u$conf.int[1,1], s_k2u$conf.int[1,3], s_k2u$conf.int[1,4],
                    s_k2u$coefficients[1,5]))

        covars_in_surv <- intersect(available_covars, colnames(k2_surv))
        if (length(covars_in_surv) > 0) {
          cox_k2_f <- as.formula(paste("Surv(followup_years, event) ~ K2_cox +",
                                       paste(covars_in_surv, collapse = " + ")))
          cox_k2_adj <- coxph(cox_k2_f, data = k2_surv)
          s_k2a <- summary(cox_k2_adj)
          cat(sprintf("  K=2 校正 Cox: HR=%.3f (%.3f-%.3f), P=%.4f\n",
                      s_k2a$conf.int[1,1], s_k2a$conf.int[1,3], s_k2a$conf.int[1,4],
                      s_k2a$coefficients[1,5]))
        }

        # K=2 Log-rank
        lr_k2 <- survdiff(Surv(followup_years, event) ~ K2_F, data = k2_surv)
        lr_k2_p <- 1 - pchisq(lr_k2$chisq, df = 1)
        cat(sprintf("  K=2 Log-rank: χ²=%.3f, P=%.4f\n", lr_k2$chisq, lr_k2_p))

        cat("\n  --- K=2 KM曲线 ---\n")
        km_k2_bsi <- survfit(Surv(followup_years, event) ~ K2_F, data = k2_surv)

        p_km_k2 <- ggsurvplot(
          km_k2_bsi, data = k2_surv,
          pval = TRUE, pval.method = TRUE,
          risk.table = TRUE, risk.table.height = 0.25,
          palette = c("#2166AC", "#D73027"),
          legend.title = "BSI-based K=2 Group",
          xlab = "Follow-up Time (Years)",
          ylab = "AD Conversion-Free Probability",
          title = "MCI-to-AD Conversion: K=2 Vulnerability Groups (BSI Validation)",
          subtitle = sprintf("N=%d, Log-rank P=%.4f", nrow(k2_surv), lr_k2_p),
          ggtheme = theme_bw()
        )

        pdf(file.path(output_dir, "KM_K2_BSI_validation.pdf"), width = 9, height = 7)
        print(p_km_k2)
        dev.off()
        png(file.path(output_dir, "KM_K2_BSI_validation.png"), width = 9, height = 7,
            units = "in", res = 300)
        print(p_km_k2)
        dev.off()
        cat("  保存: KM_K2_BSI_validation.png/pdf\n")

        p_k2_box <- ggplot(slope_brain, aes(x = K2_F, y = slope_per_year, fill = K2_F)) +
          geom_boxplot(alpha = 0.7, outlier.shape = 21) +
          geom_jitter(width = 0.15, alpha = 0.5, size = 2) +
          scale_fill_manual(values = c("#2166AC", "#D73027")) +
          geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
          labs(
            title = "Annual Brain Atrophy Rate (BSI) by K=2 Vulnerability Group",
            subtitle = sprintf("Welch t P=%.4f, Cohen's d=%.3f (N=%d)",
                               k2_ttest_p, cohens_d, nrow(slope_brain)),
            x = "Vulnerability Group", y = "Whole Brain BSI Slope (ml/year)",
            fill = "Group"
          ) +
          theme_bw(base_size = 12) + theme(legend.position = "none")

        ggsave(file.path(output_dir, "bsi_K2_atrophy_boxplot.png"), p_k2_box,
               width = 7, height = 6, dpi = 300)
        cat("  保存: bsi_K2_atrophy_boxplot.png\n")

        bsi_traj_k2 <- merge(bsi_analysis, k2_map, by = "PTID", all.x = TRUE)
        bsi_traj_k2 <- bsi_traj_k2[!is.na(bsi_traj_k2$K2_F), ]

        p_traj_k2 <- ggplot(bsi_traj_k2, aes(x = Years, y = DBCBBSI,
                                             color = K2_F, group = PTID)) +
          geom_line(alpha = 0.15, linewidth = 0.3) +
          geom_smooth(aes(group = K2_F), method = "lm", se = TRUE,
                      linewidth = 1.5, alpha = 0.3) +
          scale_color_manual(values = c("#2166AC", "#D73027")) +
          labs(
            title = "Longitudinal Brain Atrophy (BSI) by K=2 Vulnerability Group",
            subtitle = sprintf("N=%d subjects", length(unique(bsi_traj_k2$PTID))),
            x = "Time from Baseline (Years)",
            y = "Whole Brain BSI (ml)",
            color = "Group"
          ) +
          theme_bw(base_size = 12) + theme(legend.position = "bottom")

        ggsave(file.path(output_dir, "bsi_K2_longitudinal_trajectories.png"), p_traj_k2,
               width = 10, height = 7, dpi = 300)
        ggsave(file.path(output_dir, "bsi_K2_longitudinal_trajectories.pdf"), p_traj_k2,
               width = 10, height = 7)
        cat("  保存: bsi_K2_longitudinal_trajectories.png/pdf\n")

        cat("\n  ┌─────────────────────────────────────────────────────────┐\n")
        cat("  │              K=2 BSI分析汇总                            │\n")
        cat("  └─────────────────────────────────────────────────────────┘\n")
        cat(sprintf("  Welch t-test P = %.4f\n", k2_ttest_p))
        cat(sprintf("  Wilcoxon P = %.4f\n", wt$p.value))
        cat(sprintf("  ANOVA P = %.4f, η² = %.4f\n", k2_anova_p, eta2_k2))
        cat(sprintf("  Cohen's d = %.3f\n", cohens_d))
        if (!is.na(k2_lmm_p)) cat(sprintf("  LMM交互 P = %.4f\n", k2_lmm_p))
        cat(sprintf("  Cox HR = %.3f, P = %.4f\n",
                    s_k2u$conf.int[1,1], s_k2u$coefficients[1,5]))
        cat(sprintf("  Log-rank P = %.4f\n", lr_k2_p))

        k2_results <- data.frame(
          Test = c("Welch_t", "Wilcoxon", "ANOVA", "Cohens_d",
                   "LMM_interaction", "Cox_HR", "Cox_P", "Logrank_P"),
          Value = c(k2_ttest_p, wt$p.value, k2_anova_p, cohens_d,
                    ifelse(is.na(k2_lmm_p), NA, k2_lmm_p),
                    s_k2u$conf.int[1,1], s_k2u$coefficients[1,5], lr_k2_p)
        )
        fwrite(k2_results, file.path(output_dir, "K2_BSI_analysis_results.csv"))
        fwrite(slope_brain[, c("PTID", "VAE_Subtype", "K2_Group", "K2_F",
                               "slope_per_year", "n_visits", "followup_months")],
               file.path(output_dir, "K2_BSI_individual_slopes.csv"))

      } else {
        cat("  ⚠ 无法匹配K=2标签到生存数据，跳过K=2 Cox分析\n")
      }
    } else {
      cat("  ⚠ 生存数据文件不存在，跳过K=2 Cox分析\n")

      slope_brain$K2_Group <- k2_remap[as.character(slope_brain$K2_cluster)]
      slope_brain$K2_F <- factor(slope_brain$K2_Group, levels = 1:2,
                                 labels = c("Group A", "Group B"))
      tt <- t.test(slope_per_year ~ K2_F, data = slope_brain)
      cat(sprintf("  K=2 Welch t-test: P=%.4f\n", tt$p.value))
    }
  } else {
    cat("  ⚠ Z维度数据不完整，无法进行K=2聚类\n")
  }
} else {
  cat("  ⚠ 无Z维度或样本量不足，跳过K=2分析\n")
}
# ============================================================================
# ============================================================================
cat("\n[7/8] 生成可视化图表...\n")
p_traj <- ggplot(bsi_analysis, aes(x = Years, y = DBCBBSI,
                                   color = Subtype_F, group = PTID)) +
  geom_line(alpha = 0.2, linewidth = 0.3) +
  geom_smooth(aes(group = Subtype_F), method = "lm", se = TRUE,
              linewidth = 1.5, alpha = 0.3) +
  scale_color_manual(values = c("#2166AC", "#FDAE61", "#D73027")) +
  labs(
    title = "Longitudinal Brain Atrophy (BSI) by VAE Subtype",
    subtitle = sprintf("N=%d subjects, %d observations (Fox Lab BSI)",
                       length(unique(bsi_analysis$PTID)), nrow(bsi_analysis)),
    x = "Time from Baseline (Years)",
    y = "Whole Brain BSI (ml, negative = atrophy)",
    color = "VAE Subtype"
  ) +
  theme_bw(base_size = 12) +
  theme(legend.position = "bottom")
ggsave(file.path(output_dir, "bsi_longitudinal_trajectories.png"), p_traj,
       width = 10, height = 7, dpi = 300)
ggsave(file.path(output_dir, "bsi_longitudinal_trajectories.pdf"), p_traj,
       width = 10, height = 7)
cat("  保存: bsi_longitudinal_trajectories.png/pdf\n")
p_slope <- ggplot(slope_brain, aes(x = Subtype_F, y = slope_per_year, fill = Subtype_F)) +
  geom_boxplot(alpha = 0.7, outlier.shape = 21) +
  geom_jitter(width = 0.15, alpha = 0.5, size = 2) +
  scale_fill_manual(values = c("#2166AC", "#FDAE61", "#D73027")) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
  labs(
    title = "Annual Brain Atrophy Rate (BSI) by VAE Subtype",
    subtitle = sprintf("ANOVA P=%.4f, η²=%.4f (N=%d)",
                       summary(aov_main)[[1]][1, "Pr(>F)"], eta_sq_main, nrow(slope_brain)),
    x = "VAE Subtype", y = "Whole Brain BSI Slope (ml/year)", fill = "Subtype"
  ) +
  theme_bw(base_size = 12) + theme(legend.position = "none")
ggsave(file.path(output_dir, "bsi_atrophy_rate_boxplot.png"), p_slope,
       width = 7, height = 6, dpi = 300)
cat("  保存: bsi_atrophy_rate_boxplot.png\n")
if (nrow(slope_all) > 0) {
  slope_all_plot <- slope_all %>%
    group_by(metric) %>%
    mutate(slope_z = scale(slope_per_year)[, 1]) %>%
    ungroup() %>%
    mutate(Subtype_F = factor(VAE_Subtype,
                              levels = sort(unique(VAE_Subtype)),
                              labels = paste0("S", sort(unique(VAE_Subtype)))))

  p_multi <- ggplot(slope_all_plot, aes(x = metric_name, y = slope_z, fill = Subtype_F)) +
    geom_boxplot(alpha = 0.7, position = position_dodge(0.8)) +
    scale_fill_manual(values = c("#2166AC", "#FDAE61", "#D73027")) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    labs(title = "Standardized Atrophy Rate Across BSI Metrics",
         x = "", y = "Standardized Slope (z-score)", fill = "Subtype") +
    theme_bw(base_size = 11) +
    theme(axis.text.x = element_text(angle = 30, hjust = 1))

  ggsave(file.path(output_dir, "bsi_multi_metric_comparison.png"), p_multi,
         width = 10, height = 6, dpi = 300)
  cat("  保存: bsi_multi_metric_comparison.png\n")
}
if (length(z_cols) > 0) {
  for (zc in z_cols) {
    if (zc %in% colnames(slope_brain)) {
      p_z <- ggplot(slope_brain, aes(x = .data[[zc]], y = slope_per_year,
                                     color = Subtype_F)) +
        geom_point(alpha = 0.6, size = 2.5) +
        geom_smooth(method = "lm", se = TRUE, color = "black", linewidth = 1) +
        scale_color_manual(values = c("#2166AC", "#FDAE61", "#D73027")) +
        labs(title = paste0("VAE ", zc, " vs Annual Brain Atrophy (BSI)"),
             x = paste0(zc, " (VAE latent)"), y = "Brain BSI Slope (ml/year)",
             color = "Subtype") +
        theme_bw(base_size = 12)
      ggsave(file.path(output_dir, paste0(zc, "_vs_bsi_atrophy.png")), p_z,
             width = 8, height = 6, dpi = 300)
    }
  }
  cat("  保存: Z*_vs_bsi_atrophy.png\n")
}
# ============================================================================
# ============================================================================
cat("\n[8/8] 汇总报告...\n")
cat("\n")
cat("████████████████████████████████████████████████████████████████\n")
cat("  Step 15 纵向Fox Lab BSI验证 — 完整结果汇总\n")
cat("████████████████████████████████████████████████████████████████\n\n")
cat("  数据概况:\n")
cat(sprintf("    Discovery cohort: N=%d\n", nrow(cohort_vae)))
cat(sprintf("    匹配到BSI: N=%d (%.1f%%)\n", n_matched,
            n_matched / nrow(cohort_vae) * 100))
cat(sprintf("    有≥2时间点: N=%d\n", nrow(slope_brain)))
cat(sprintf("    纵向数据点总数: %d\n", nrow(bsi_analysis)))
cat("\n  全脑BSI萎缩斜率 (ml/year):\n")
for (i in 1:nrow(slope_summary)) {
  cat(sprintf("    %s: mean=%.4f ± %.4f, median=%.4f (n=%d)\n",
              slope_summary$Subtype_F[i],
              slope_summary$mean_slope[i], slope_summary$sd_slope[i],
              slope_summary$median_slope[i], slope_summary$n[i]))
}
cat("\n  多指标ANOVA汇总:\n")
if (nrow(anova_df) > 0) {
  for (i in 1:nrow(anova_df)) {
    sig <- ifelse(anova_df$ANOVA_P[i] < 0.05, "★★★",
                  ifelse(anova_df$ANOVA_P[i] < 0.10, "△", ""))
    cat(sprintf("    %-15s: F=%.3f, P=%.4f, η²=%.4f %s\n",
                anova_df$Metric[i], anova_df$ANOVA_F[i],
                anova_df$ANOVA_P[i], anova_df$Eta_sq[i], sig))
  }
}
cat("\n")
cat("  ┌─────────────────────────────────────────────────────────┐\n")
cat("  │                    ★ 关键判定 ★                         │\n")
cat("  └─────────────────────────────────────────────────────────┘\n")
main_p <- summary(aov_main)[[1]][1, "Pr(>F)"]
any_sig <- any(anova_df$ANOVA_P < 0.05)
any_trend <- any(anova_df$ANOVA_P < 0.10)
cat("\n  --- K=3 ANOVA ---\n")
if (any_sig) {
  sig_metrics <- anova_df$Metric[anova_df$ANOVA_P < 0.05]
  cat("  ★★★ 至少一个BSI指标显示亚型间萎缩速率显著不同！\n")
  cat("  显著指标:", paste(sig_metrics, collapse = ", "), "\n")
} else if (any_trend) {
  trend_metrics <- anova_df$Metric[anova_df$ANOVA_P < 0.10]
  cat("  ★★ 边缘显著趋势存在\n")
  cat("  趋势指标:", paste(trend_metrics, collapse = ", "), "\n")
} else {
  cat("  ★ 所有BSI指标均未显示亚型间萎缩速率显著差异\n")
}
cat("\n  --- LMM ---\n")
if (exists("lmm_interaction_p") && !is.na(lmm_interaction_p)) {
  cat(sprintf("  LMM Time × Subtype P = %.4f (模型: %s)\n",
              lmm_interaction_p, lmm_model_type))
  if (lmm_interaction_p < 0.05) {
    cat("  ★★★ LMM交互项显著！亚型间萎缩轨迹确实不同！\n")
  } else if (lmm_interaction_p < 0.10) {
    cat("  △ LMM交互项边缘显著\n")
  } else {
    cat("  → LMM交互项不显著\n")
  }
} else {
  cat("  ⚠ LMM未成功拟合\n")
}
cat("\n  --- K=2 分析 ---\n")
if (exists("k2_ttest_p")) {
  cat(sprintf("  K=2 Welch t P = %.4f\n", k2_ttest_p))
  if (exists("cohens_d")) cat(sprintf("  Cohen's d = %.3f\n", cohens_d))
  if (exists("k2_lmm_p") && !is.na(k2_lmm_p)) {
    cat(sprintf("  K=2 LMM交互 P = %.4f\n", k2_lmm_p))
  }
  if (exists("lr_k2_p")) cat(sprintf("  K=2 Log-rank P = %.4f\n", lr_k2_p))
} else {
  cat("  K=2分析未执行\n")
}
cat("\n  --- 综合判定 ---\n")
evidence_count <- 0
if (any_sig) evidence_count <- evidence_count + 2
if (any_trend) evidence_count <- evidence_count + 1
if (exists("lmm_interaction_p") && !is.na(lmm_interaction_p) && lmm_interaction_p < 0.05) evidence_count <- evidence_count + 2
if (exists("lmm_interaction_p") && !is.na(lmm_interaction_p) && lmm_interaction_p < 0.10) evidence_count <- evidence_count + 1
if (exists("k2_ttest_p") && k2_ttest_p < 0.05) evidence_count <- evidence_count + 2
if (exists("k2_ttest_p") && k2_ttest_p < 0.10 && k2_ttest_p >= 0.05) evidence_count <- evidence_count + 1
if (evidence_count >= 4) {
  cat("  ★★★ 多条证据线汇聚：VAE亚型预测纵向萎缩轨迹\n")
  cat("  → 'Structural-Clinical Dissociation'叙事得到强力支撑\n")
} else if (evidence_count >= 2) {
  cat("  ★★ 支持性证据存在，但未达到强显著水平\n")
  cat("  → 可报告为supportive evidence\n")
} else {
  cat("  ★ 证据不足以支持亚型预测纵向萎缩\n")
  cat("  → 叙事策略需调整：强调横截面异质性\n")
}
results_list <- list(
  slope_summary = slope_summary,
  anova_df = anova_df,
  main_anova_p = main_p,
  main_eta_sq = eta_sq_main,
  n_matched = n_matched,
  n_longitudinal = nrow(slope_brain),
  lmm_interaction_p = if (exists("lmm_interaction_p")) lmm_interaction_p else NA,
  lmm_model_type = if (exists("lmm_model_type")) lmm_model_type else "none",
  k2_ttest_p = if (exists("k2_ttest_p")) k2_ttest_p else NA,
  k2_cohens_d = if (exists("cohens_d")) cohens_d else NA,
  k2_lmm_p = if (exists("k2_lmm_p")) k2_lmm_p else NA,
  k2_logrank_p = if (exists("lr_k2_p")) lr_k2_p else NA
)
saveRDS(results_list, file.path(output_dir, "bsi_validation_results.rds"))
fwrite(slope_brain, file.path(output_dir, "individual_bsi_slopes.csv"))
fwrite(anova_df, file.path(output_dir, "multi_metric_anova_summary.csv"))
fwrite(as.data.frame(bsi_analysis), file.path(output_dir, "bsi_longitudinal_merged.csv"))
cat("\n  所有结果已保存到:", output_dir, "\n")
cat("================================================================\n")
cat("  Step 15 完成\n")
cat("================================================================\n")

