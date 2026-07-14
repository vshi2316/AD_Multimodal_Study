# ==============================================================================
# MRI subtype characterization for the 30-region structural signature set
# ==============================================================================
library(tidyverse)
library(ggplot2)
library(patchwork)
library(jsonlite)
library(multcomp)
library(optparse)

option_list <- list(
  make_option(c("--vae_dir"), type = "character", default = ".",
              help = "Directory containing subtype_assignments.csv and vae_summary.json [default: %default]"),
  make_option(c("--data_dir"), type = "character", default = ".",
              help = "Directory containing Clinical_data.csv, RNA_plasma.csv, and metabolites.csv [default: %default]"),
  make_option(c("--output_dir"), type = "character", default = "./step22_results",
              help = "Output directory [default: %default]"),
  make_option(c("--fdr_threshold"), type = "double", default = 0.05,
              help = "False discovery rate threshold [default: %default]")
)

opt <- parse_args(OptionParser(option_list = option_list))

cat("==============================================================================\\n")
cat("Step 22: MRI Subtype Characterization\\n")
cat("==============================================================================\\n\\n")

vae_dir <- opt$vae_dir
data_dir <- opt$data_dir
output_dir <- opt$output_dir
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

FDR_THRESHOLD <- opt$fdr_threshold
set.seed(42)

# ==============================================================================
# 1. ST Code -> Anatomical Name Mapping (FreeSurfer convention)
# ==============================================================================
# Suffix: CV = cortical volume, TA = cortical thickness (average),
#          TS = cortical thickness (std), SA = surface area, SV = subcortical vol
st_to_anatomy <- c(
  "ST101SV" = "Left Hippocampus (Vol)",
  "ST102CV" = "Right Paracentral (Vol)",
  "ST102TA" = "Right Paracentral (Thk)",
  "ST102TS" = "Right Paracentral (ThkStd)",
  "ST103CV" = "Right Parahippocampal (Vol)",
  "ST103TA" = "Right Parahippocampal (Thk)",
  "ST103TS" = "Right Parahippocampal (ThkStd)",
  "ST104CV" = "Right Pars Opercularis (Vol)",
  "ST104TA" = "Right Pars Opercularis (Thk)",
  "ST104TS" = "Right Pars Opercularis (ThkStd)",
  "ST105CV" = "Right Pars Orbitalis (Vol)",
  "ST105TA" = "Right Pars Orbitalis (Thk)",
  "ST105TS" = "Right Pars Orbitalis (ThkStd)",
  "ST106CV" = "Right Pars Triangularis (Vol)",
  "ST106TA" = "Right Pars Triangularis (Thk)",
  "ST106TS" = "Right Pars Triangularis (ThkStd)",
  "ST107CV" = "Right Pericalcarine (Vol)",
  "ST107TA" = "Right Pericalcarine (Thk)",
  "ST107TS" = "Right Pericalcarine (ThkStd)",
  "ST108CV" = "Right Postcentral (Vol)",
  "ST108TA" = "Right Postcentral (Thk)",
  "ST108TS" = "Right Postcentral (ThkStd)",
  "ST109CV" = "Right Posterior Cingulate (Vol)",
  "ST109TA" = "Right Posterior Cingulate (Thk)",
  "ST109TS" = "Right Posterior Cingulate (ThkStd)",
  "ST110CV" = "Right Precentral (Vol)",
  "ST110TA" = "Right Precentral (Thk)",
  "ST110TS" = "Right Precentral (ThkStd)",
  "ST111CV" = "Right Precuneus (Vol)",
  "ST111TA" = "Right Precuneus (Thk)",
  "ST111TS" = "Right Precuneus (ThkStd)",
  "ST112CV" = "Right Rostral Anterior Cingulate (Vol)",
  "ST112TA" = "Right Rostral Anterior Cingulate (Thk)",
  "ST113CV" = "Right Rostral Middle Frontal (Vol)",
  "ST113TA" = "Right Rostral Middle Frontal (Thk)",
  "ST114CV" = "Right Superior Frontal (Vol)",
  "ST114TA" = "Right Superior Frontal (Thk)",
  "ST115CV" = "Right Superior Parietal (Vol)",
  "ST115TA" = "Right Superior Parietal (Thk)",
  "ST116CV" = "Right Superior Temporal (Vol)",
  "ST116TA" = "Right Superior Temporal (Thk)",
  "ST117CV" = "Right Supramarginal (Vol)",
  "ST117TA" = "Right Supramarginal (Thk)",
  "ST118SV" = "Right Hippocampus (Vol)",
  "ST119SV" = "Right Amygdala (Vol)",
  "ST24CV"  = "Left Entorhinal (Vol)",
  "ST24TA"  = "Left Entorhinal (Thk)",
  "ST29CV"  = "Left Fusiform (Vol)",
  "ST29TA"  = "Left Fusiform (Thk)",
  "ST31CV"  = "Left Inferior Parietal (Vol)",
  "ST31TA"  = "Left Inferior Parietal (Thk)",
  "ST32CV"  = "Left Inferior Temporal (Vol)",
  "ST32TA"  = "Left Inferior Temporal (Thk)",
  "ST40CV"  = "Left Middle Temporal (Vol)",
  "ST40TA"  = "Left Middle Temporal (Thk)",
  "ST44CV"  = "Left Precuneus (Vol)",
  "ST44TA"  = "Left Precuneus (Thk)",
  "ST50CV"  = "Left Superior Temporal (Vol)",
  "ST50TA"  = "Left Superior Temporal (Thk)",
  "ST52SV"  = "Left Amygdala (Vol)"
)
# Network assignment for aggregation
st_to_network <- c(
  "ST101SV" = "Limbic", "ST118SV" = "Limbic", "ST119SV" = "Limbic",
  "ST52SV"  = "Limbic",
  "ST103CV" = "Limbic", "ST103TA" = "Limbic", "ST103TS" = "Limbic",
  "ST24CV"  = "Limbic", "ST24TA"  = "Limbic",
  "ST29CV"  = "Limbic", "ST29TA"  = "Limbic",
  "ST109CV" = "DMN", "ST109TA" = "DMN", "ST109TS" = "DMN",
  "ST111CV" = "DMN", "ST111TA" = "DMN", "ST111TS" = "DMN",
  "ST44CV"  = "DMN", "ST44TA"  = "DMN",
  "ST40CV"  = "DMN", "ST40TA"  = "DMN",
  "ST113CV" = "DMN", "ST113TA" = "DMN",
  "ST31CV"  = "DMN", "ST31TA"  = "DMN",
  "ST117CV" = "FPN", "ST117TA" = "FPN",
  "ST115CV" = "FPN", "ST115TA" = "FPN",
  "ST104CV" = "FPN", "ST104TA" = "FPN", "ST104TS" = "FPN",
  "ST106CV" = "FPN", "ST106TA" = "FPN", "ST106TS" = "FPN",
  "ST105CV" = "FPN", "ST105TA" = "FPN", "ST105TS" = "FPN",
  "ST114CV" = "FPN", "ST114TA" = "FPN",
  "ST102CV" = "Sensorimotor", "ST102TA" = "Sensorimotor", "ST102TS" = "Sensorimotor",
  "ST108CV" = "Sensorimotor", "ST108TA" = "Sensorimotor", "ST108TS" = "Sensorimotor",
  "ST110CV" = "Sensorimotor", "ST110TA" = "Sensorimotor", "ST110TS" = "Sensorimotor",
  "ST107CV" = "Visual", "ST107TA" = "Visual", "ST107TS" = "Visual",
  "ST112CV" = "Salience", "ST112TA" = "Salience",
  "ST116CV" = "Auditory", "ST116TA" = "Auditory",
  "ST32CV"  = "Temporal", "ST32TA"  = "Temporal",
  "ST50CV"  = "Auditory", "ST50TA"  = "Auditory"
)
# ==============================================================================
# 2. Load and Merge Data
# ==============================================================================
cat("[1/8] Loading and merging data...\n")
# VAE outputs
subtypes <- read.csv(file.path(vae_dir, "subtype_assignments.csv"),
                     stringsAsFactors = FALSE)
subtypes$ID <- as.character(subtypes$ID)
# Read vae_summary.json for feature names
vae_json <- fromJSON(file.path(vae_dir, "vae_summary.json"))
mri_features_st <- vae_json$mri_features  # 30 ST codes
cat(sprintf("  VAE subtypes: %d participants, %d MRI features\n",
            nrow(subtypes), length(mri_features_st)))
# Original data
clinical <- read.csv(file.path(data_dir, "Clinical_data.csv"),
                     stringsAsFactors = FALSE)
smri     <- read.csv(file.path(data_dir, "RNA_plasma.csv"),
                     stringsAsFactors = FALSE)
csf      <- read.csv(file.path(data_dir, "metabolites.csv"),
                     stringsAsFactors = FALSE)
clinical$ID <- as.character(clinical$ID)
smri$ID     <- as.character(smri$ID)
csf$ID      <- as.character(csf$ID)
# Determine subtype column name
cat("  subtype_assignments.csv columns:", paste(colnames(subtypes), collapse = ", "), "\n")
if ("VAE_Subtype" %in% colnames(subtypes)) {
  subtypes$Subtype <- subtypes$VAE_Subtype
} else if (!"Subtype" %in% colnames(subtypes)) {
  stop("Cannot find VAE_Subtype or Subtype column in subtype_assignments.csv")
}
# Build master from subtypes (keep ID, Subtype, AD_Conversion)
keep_cols <- intersect(c("ID", "Subtype", "AD_Conversion"), colnames(subtypes))
master <- subtypes[, keep_cols, drop = FALSE]
# Merge clinical (AGE, SEX, MMSE, EDUCATION, GDS, APOE4_DOSAGE, ADAS13, CDRSB, FAQTOTAL)
master <- merge(master, clinical, by = "ID", all.x = TRUE)
# Merge MRI
mri_cols_available <- intersect(mri_features_st, colnames(smri))
master <- merge(master, smri[, c("ID", mri_cols_available)], by = "ID", all.x = TRUE)
# Merge CSF
csf_cols <- c("PTAU181", "ABETA42_ABETA40_RATIO", "ABETA40")
csf_cols_available <- intersect(csf_cols, colnames(csf))
if (length(csf_cols_available) > 0) {
  master <- merge(master, csf[, c("ID", csf_cols_available)], by = "ID", all.x = TRUE)
}
# Ensure Subtype is factor
master$Subtype <- factor(master$Subtype)
cat(sprintf("  Merged data: %d participants x %d columns\n",
            nrow(master), ncol(master)))
cat(sprintf("  MRI features matched: %d/%d\n",
            length(mri_cols_available), length(mri_features_st)))
# Subtype distribution
cat("\n  Subtype distribution:\n")
st_dist <- table(master$Subtype)
for (i in seq_along(st_dist)) {
  cat(sprintf("    Subtype %s: %d (%.1f%%)\n",
              names(st_dist)[i], st_dist[i],
              100 * st_dist[i] / sum(st_dist)))
}
# Map ST codes to anatomical names for display
get_anatomy <- function(st_code) {
  if (st_code %in% names(st_to_anatomy)) {
    return(st_to_anatomy[st_code])
  }
  return(st_code)  # fallback: keep original
}
get_network <- function(st_code) {
  if (st_code %in% names(st_to_network)) {
    return(st_to_network[st_code])
  }
  return("Other")
}
# Helper: interpret eta-squared
interpret_eta2 <- function(eta2) {
  if (is.na(eta2)) return("NA")
  if (eta2 >= 0.14) return("Large")
  if (eta2 >= 0.06) return("Medium")
  if (eta2 >= 0.01) return("Small")
  return("Negligible")
}
# ==============================================================================
# 3. Demographics and Clinical Summary by Subtype
# ==============================================================================
cat("\n[2/8] Demographics and clinical summary by subtype...\n")
demo_vars <- c("AGE", "SEX", "EDUCATION", "MMSE", "GDS", "APOE4_DOSAGE")
demo_vars <- demo_vars[demo_vars %in% colnames(master)]
# Also include cognitive/functional measures for descriptive comparison
# (ADAS13, CDRSB, FAQTOTAL are NOT used in prediction, but valid for
#  describing subtype clinical profiles)
clinical_desc_vars <- c("ADAS13", "CDRSB", "FAQTOTAL")
clinical_desc_vars <- clinical_desc_vars[clinical_desc_vars %in% colnames(master)]
all_desc_vars <- c(demo_vars, clinical_desc_vars)
demo_summary <- data.frame()
for (v in all_desc_vars) {
  vals <- master[[v]]
  if (!is.numeric(vals)) next

  # Overall
  overall_mean <- mean(vals, na.rm = TRUE)
  overall_sd   <- sd(vals, na.rm = TRUE)

  # Per subtype
  for (st in levels(master$Subtype)) {
    sub_vals <- vals[master$Subtype == st]
    demo_summary <- rbind(demo_summary, data.frame(
      Variable = v,
      Subtype  = st,
      N        = sum(!is.na(sub_vals)),
      Mean     = mean(sub_vals, na.rm = TRUE),
      SD       = sd(sub_vals, na.rm = TRUE),
      Median   = median(sub_vals, na.rm = TRUE),
      stringsAsFactors = FALSE
    ))
  }

  # ANOVA
  fit <- aov(as.formula(paste(v, "~ Subtype")), data = master)
  s <- summary(fit)
  f_val <- s[[1]]$`F value`[1]
  p_val <- s[[1]]$`Pr(>F)`[1]
  ss_b  <- s[[1]]$`Sum Sq`[1]
  ss_t  <- sum(s[[1]]$`Sum Sq`)
  eta2  <- ss_b / ss_t

  cat(sprintf("  %s: F=%.2f, p=%.4f, eta2=%.3f (%s)\n",
              v, f_val, p_val, eta2, interpret_eta2(eta2)))
}
write.csv(demo_summary,
          file.path(output_dir, "Demographics_Clinical_By_Subtype.csv"),
          row.names = FALSE)
# Sex distribution (chi-squared)
if ("SEX" %in% colnames(master)) {
  sex_tab <- table(master$Subtype, round(master$SEX))
  sex_test <- chisq.test(sex_tab)
  cat(sprintf("  Sex distribution: chi2=%.2f, p=%.4f\n",
              sex_test$statistic, sex_test$p.value))
}
# ==============================================================================
# 4. MRI Heterogeneity Test (ANOVA + FDR + Tukey HSD)
# ==============================================================================
cat("\n[3/8] MRI heterogeneity analysis (ANOVA + pairwise Tukey)...\n")
mri_results <- data.frame()
tukey_results <- data.frame()
for (feat in mri_cols_available) {
  if (!is.numeric(master[[feat]])) next

  anatomy <- get_anatomy(feat)
  network <- get_network(feat)

  # ANOVA
  fit <- aov(as.formula(paste0("`", feat, "` ~ Subtype")), data = master)
  s <- summary(fit)
  f_val <- s[[1]]$`F value`[1]
  p_val <- s[[1]]$`Pr(>F)`[1]
  ss_b  <- s[[1]]$`Sum Sq`[1]
  ss_t  <- sum(s[[1]]$`Sum Sq`)
  eta2  <- ss_b / ss_t

  # Per-subtype means
  means_by_st <- tapply(master[[feat]], master$Subtype, mean, na.rm = TRUE)

  mri_results <- rbind(mri_results, data.frame(
    ST_Code    = feat,
    Anatomy    = anatomy,
    Network    = network,
    F_value    = f_val,
    P_Raw      = p_val,
    Eta2       = eta2,
    Eta2_Interp = interpret_eta2(eta2),
    Mean_S1    = means_by_st[1],
    Mean_S2    = means_by_st[2],
    Mean_S3    = means_by_st[3],
    stringsAsFactors = FALSE
  ))

  # Tukey HSD pairwise comparisons
  tukey <- TukeyHSD(fit)
  tukey_df <- as.data.frame(tukey$Subtype)
  tukey_df$Comparison <- rownames(tukey_df)
  tukey_df$ST_Code <- feat
  tukey_df$Anatomy <- anatomy
  tukey_results <- rbind(tukey_results, tukey_df)
}
# FDR correction
mri_results$P_FDR <- p.adjust(mri_results$P_Raw, method = "fdr")
mri_results$Significant <- mri_results$P_FDR < FDR_THRESHOLD
# Sort by effect size
mri_results <- mri_results %>% arrange(desc(Eta2))
n_sig <- sum(mri_results$Significant)
n_large <- sum(mri_results$Eta2 >= 0.14, na.rm = TRUE)
n_medium <- sum(mri_results$Eta2 >= 0.06 & mri_results$Eta2 < 0.14, na.rm = TRUE)
cat(sprintf("  Significant (FDR < %.2f): %d/%d\n", FDR_THRESHOLD, n_sig, nrow(mri_results)))
cat(sprintf("  Large effect (eta2 >= 0.14): %d\n", n_large))
cat(sprintf("  Medium effect (eta2 >= 0.06): %d\n", n_medium))
# Top 10 features
cat("\n  Top 10 MRI features by effect size:\n")
for (i in 1:min(10, nrow(mri_results))) {
  r <- mri_results[i, ]
  sig <- ifelse(r$Significant, " ***", "")
  cat(sprintf("    %2d. %s (eta2=%.3f, F=%.1f, p_FDR=%.4f)%s\n",
              i, r$Anatomy, r$Eta2, r$F_value, r$P_FDR, sig))
}
write.csv(mri_results,
          file.path(output_dir, "MRI_Heterogeneity_Results.csv"),
          row.names = FALSE)
write.csv(tukey_results,
          file.path(output_dir, "MRI_Tukey_Pairwise.csv"),
          row.names = FALSE)
# ==============================================================================
# 5. CSF Biomarker Profiles by Subtype
# ==============================================================================
cat("\n[4/8] CSF biomarker profiles by subtype...\n")
csf_results <- data.frame()
for (feat in csf_cols_available) {
  if (!is.numeric(master[[feat]])) next

  fit <- aov(as.formula(paste0(feat, " ~ Subtype")), data = master)
  s <- summary(fit)
  f_val <- s[[1]]$`F value`[1]
  p_val <- s[[1]]$`Pr(>F)`[1]
  ss_b  <- s[[1]]$`Sum Sq`[1]
  ss_t  <- sum(s[[1]]$`Sum Sq`)
  eta2  <- ss_b / ss_t

  means_by_st <- tapply(master[[feat]], master$Subtype, mean, na.rm = TRUE)

  csf_results <- rbind(csf_results, data.frame(
    Feature    = feat,
    F_value    = f_val,
    P_Raw      = p_val,
    Eta2       = eta2,
    Eta2_Interp = interpret_eta2(eta2),
    Mean_S1    = means_by_st[1],
    Mean_S2    = means_by_st[2],
    Mean_S3    = means_by_st[3],
    stringsAsFactors = FALSE
  ))

  # Tukey
  tukey <- TukeyHSD(fit)
  cat(sprintf("  %s: F=%.2f, p=%.4f, eta2=%.3f\n", feat, f_val, p_val, eta2))
  tukey_df <- as.data.frame(tukey$Subtype)
  for (comp in rownames(tukey_df)) {
    cat(sprintf("    %s: diff=%.3f, p=%.4f\n",
                comp, tukey_df[comp, "diff"], tukey_df[comp, "p adj"]))
  }
}
if (nrow(csf_results) > 0) {
  csf_results$P_FDR <- p.adjust(csf_results$P_Raw, method = "fdr")
  write.csv(csf_results,
            file.path(output_dir, "CSF_Subtype_Profiles.csv"),
            row.names = FALSE)
}
# ==============================================================================
# 6. Stage Independence (ANCOVA: Subtype + MMSE + Age + Sex)
# ==============================================================================
cat("\n[5/8] Stage independence test (ANCOVA)...\n")
has_covariates <- all(c("MMSE", "AGE", "SEX") %in% colnames(master))
if (!has_covariates) {
  cat("  WARNING: Missing covariates for ANCOVA, checking alternatives...\n")
  # Try alternative column names
  if ("MMSE_Baseline" %in% colnames(master) && !"MMSE" %in% colnames(master)) {
    master$MMSE <- master$MMSE_Baseline
  }
  has_covariates <- all(c("MMSE", "AGE", "SEX") %in% colnames(master))
}
stage_results <- data.frame()
if (has_covariates) {
  for (feat in mri_cols_available) {
    if (!is.numeric(master[[feat]])) next

    anatomy <- get_anatomy(feat)

    # Unadjusted
    fit_unadj <- aov(as.formula(paste0("`", feat, "` ~ Subtype")), data = master)
    s_unadj <- summary(fit_unadj)
    ss_b_unadj <- s_unadj[[1]]$`Sum Sq`[1]
    ss_t_unadj <- sum(s_unadj[[1]]$`Sum Sq`)
    eta2_unadj <- ss_b_unadj / ss_t_unadj

    # Adjusted (ANCOVA: Subtype + MMSE + Age + Sex)
    fit_adj <- aov(as.formula(paste0("`", feat, "` ~ Subtype + MMSE + AGE + SEX")),
                   data = master)
    s_adj <- summary(fit_adj)
    # Subtype is first term
    ss_b_adj <- s_adj[[1]]$`Sum Sq`[1]
    ss_t_adj <- sum(s_adj[[1]]$`Sum Sq`)
    eta2_adj <- ss_b_adj / ss_t_adj
    p_adj    <- s_adj[[1]]$`Pr(>F)`[1]

    eta2_change <- eta2_adj - eta2_unadj
    pct_change  <- 100 * eta2_change / (eta2_unadj + 1e-10)

    stage_results <- rbind(stage_results, data.frame(
      ST_Code        = feat,
      Anatomy        = anatomy,
      Eta2_Unadjusted = eta2_unadj,
      Eta2_Adjusted   = eta2_adj,
      Eta2_Change     = eta2_change,
      Pct_Change      = pct_change,
      P_Adjusted      = p_adj,
      Eta2_Interp_Adj = interpret_eta2(eta2_adj),
      stringsAsFactors = FALSE
    ))
  }

  stage_results$P_FDR <- p.adjust(stage_results$P_Adjusted, method = "fdr")
  stage_results$Stage_Independent <- stage_results$P_FDR < FDR_THRESHOLD

  n_indep <- sum(stage_results$Stage_Independent)
  cat(sprintf("  Stage-independent (FDR < %.2f after MMSE+Age+Sex adjustment): %d/%d\n",
              FDR_THRESHOLD, n_indep, nrow(stage_results)))

  # Median effect size change
  med_pct <- median(stage_results$Pct_Change, na.rm = TRUE)
  cat(sprintf("  Median effect size change after adjustment: %.1f%%\n", med_pct))

  write.csv(stage_results,
            file.path(output_dir, "Stage_Independence_ANCOVA.csv"),
            row.names = FALSE)
} else {
  cat("  SKIPPED: MMSE, AGE, or SEX not available\n")
}
# ==============================================================================
# 7. Disproportionate Atrophy (W-score Residuals)
# ==============================================================================
cat("\n[6/8] Disproportionate atrophy analysis (W-score)...\n")
# Separate volume and thickness features
vol_features <- mri_cols_available[grepl("CV$|SV$", mri_cols_available)]
thk_features <- mri_cols_available[grepl("TA$|TS$", mri_cols_available)]
# Global composites
if (length(vol_features) > 1) {
  master$Global_Vol_Composite <- rowMeans(master[, vol_features, drop = FALSE],
                                          na.rm = TRUE)
}
if (length(thk_features) > 1) {
  master$Global_Thk_Composite <- rowMeans(master[, thk_features, drop = FALSE],
                                          na.rm = TRUE)
}
disprop_stats <- data.frame()
disprop_residuals <- data.frame()
for (feat in mri_cols_available) {
  if (!is.numeric(master[[feat]])) next

  anatomy <- get_anatomy(feat)
  is_vol <- grepl("CV$|SV$", feat)
  global_var <- if (is_vol) "Global_Vol_Composite" else "Global_Thk_Composite"

  if (!global_var %in% colnames(master)) next

  # Build regression formula
  formula_parts <- c(global_var)
  if ("AGE" %in% colnames(master)) formula_parts <- c(formula_parts, "AGE")
  if ("SEX" %in% colnames(master)) formula_parts <- c(formula_parts, "SEX")
  formula_str <- paste0("`", feat, "` ~ ", paste(formula_parts, collapse = " + "))

  fit_resid <- lm(as.formula(formula_str), data = master)
  master$temp_resid <- rstandard(fit_resid)

  # ANOVA on residuals
  fit_anova <- aov(temp_resid ~ Subtype, data = master)
  s <- summary(fit_anova)
  f_val <- s[[1]]$`F value`[1]
  p_val <- s[[1]]$`Pr(>F)`[1]
  ss_b  <- s[[1]]$`Sum Sq`[1]
  ss_t  <- sum(s[[1]]$`Sum Sq`)
  eta2  <- ss_b / ss_t

  disprop_stats <- rbind(disprop_stats, data.frame(
    ST_Code    = feat,
    Anatomy    = anatomy,
    Network    = get_network(feat),
    Global_Control = global_var,
    F_value    = f_val,
    P_Raw      = p_val,
    Eta2_Resid = eta2,
    Eta2_Interp = interpret_eta2(eta2),
    stringsAsFactors = FALSE
  ))

  # Save residuals for plotting
  disprop_residuals <- rbind(disprop_residuals, data.frame(
    Subtype  = master$Subtype,
    ST_Code  = feat,
    Anatomy  = anatomy,
    Network  = get_network(feat),
    Residual = master$temp_resid
  ))
}
master$temp_resid <- NULL
# FDR correction
disprop_stats$P_FDR <- p.adjust(disprop_stats$P_Raw, method = "fdr")
disprop_stats$Significant <- disprop_stats$P_FDR < FDR_THRESHOLD
disprop_stats <- disprop_stats %>% arrange(desc(Eta2_Resid))
n_topo <- sum(disprop_stats$Significant)
cat(sprintf("  Topologically specific (FDR < %.2f): %d/%d\n",
            FDR_THRESHOLD, n_topo, nrow(disprop_stats)))
write.csv(disprop_stats,
          file.path(output_dir, "Disproportionate_Atrophy_Stats.csv"),
          row.names = FALSE)
# ==============================================================================
# 8. Network-Level Aggregation
# ==============================================================================
cat("\n[7/8] Network-level aggregation...\n")
network_results <- data.frame()
networks <- unique(na.omit(sapply(mri_cols_available, get_network)))
networks <- networks[networks != "Other"]
for (net in networks) {
  net_feats <- mri_cols_available[sapply(mri_cols_available, get_network) == net]
  if (length(net_feats) < 2) next

  # Composite z-score for this network
  master[[paste0("Net_", net)]] <- rowMeans(master[, net_feats, drop = FALSE],
                                            na.rm = TRUE)
  net_col <- paste0("Net_", net)

  fit <- aov(as.formula(paste0(net_col, " ~ Subtype")), data = master)
  s <- summary(fit)
  f_val <- s[[1]]$`F value`[1]
  p_val <- s[[1]]$`Pr(>F)`[1]
  ss_b  <- s[[1]]$`Sum Sq`[1]
  ss_t  <- sum(s[[1]]$`Sum Sq`)
  eta2  <- ss_b / ss_t

  means_by_st <- tapply(master[[net_col]], master$Subtype, mean, na.rm = TRUE)

  network_results <- rbind(network_results, data.frame(
    Network    = net,
    N_Features = length(net_feats),
    F_value    = f_val,
    P_Raw      = p_val,
    Eta2       = eta2,
    Eta2_Interp = interpret_eta2(eta2),
    Mean_S1    = means_by_st[1],
    Mean_S2    = means_by_st[2],
    Mean_S3    = means_by_st[3],
    stringsAsFactors = FALSE
  ))

  cat(sprintf("  %s (%d features): F=%.2f, p=%.4f, eta2=%.3f (%s)\n",
              net, length(net_feats), f_val, p_val, eta2, interpret_eta2(eta2)))
}
if (nrow(network_results) > 0) {
  network_results$P_FDR <- p.adjust(network_results$P_Raw, method = "fdr")
  network_results <- network_results %>% arrange(desc(Eta2))
  write.csv(network_results,
            file.path(output_dir, "Network_Level_Results.csv"),
            row.names = FALSE)
}
# ==============================================================================
# 9. Visualizations
# ==============================================================================
cat("\n[8/8] Generating visualizations...\n")
# --- Figure A: MRI Heterogeneity (effect sizes, all 30 features) ---
plot_mri <- mri_results %>%
  mutate(Label = Anatomy,
         Label = str_wrap(Label, width = 35)) %>%
  ggplot(aes(x = reorder(Label, Eta2), y = Eta2, fill = Significant)) +
  geom_bar(stat = "identity", alpha = 0.85) +
  geom_hline(yintercept = 0.01, linetype = "dashed", color = "orange", linewidth = 0.6) +
  geom_hline(yintercept = 0.06, linetype = "dashed", color = "red", linewidth = 0.6) +
  geom_hline(yintercept = 0.14, linetype = "dashed", color = "darkred", linewidth = 0.6) +
  scale_fill_manual(values = c("FALSE" = "#95B3D7", "TRUE" = "#C0504D"),
                    name = paste0("FDR < ", FDR_THRESHOLD)) +
  coord_flip() +
  labs(title = "MRI Feature Heterogeneity Across Subtypes",
       subtitle = "Dashed lines: small (0.01), medium (0.06), large (0.14) effect thresholds",
       x = NULL, y = expression(eta^2)) +
  theme_classic(base_size = 11) +
  theme(plot.title = element_text(face = "bold", size = 13),
        axis.text.y = element_text(size = 7),
        legend.position = "bottom")
ggsave(file.path(output_dir, "Figure_MRI_Heterogeneity.png"),
       plot_mri, width = 11, height = 10, dpi = 300)
cat("  Saved: Figure_MRI_Heterogeneity.png\n")
# --- Figure B: Stage Independence (unadjusted vs adjusted) ---
if (nrow(stage_results) > 0) {
  plot_stage <- stage_results %>%
    dplyr::select(Anatomy, Eta2_Unadjusted, Eta2_Adjusted) %>%
    pivot_longer(cols = c(Eta2_Unadjusted, Eta2_Adjusted),
                 names_to = "Type", values_to = "Eta2") %>%
    mutate(Type = factor(Type,
                         levels = c("Eta2_Unadjusted", "Eta2_Adjusted"),
                         labels = c("Unadjusted", "Adjusted (MMSE+Age+Sex)")),
           Anatomy = str_wrap(Anatomy, width = 35)) %>%
    ggplot(aes(x = reorder(Anatomy, Eta2), y = Eta2, fill = Type)) +
    geom_bar(stat = "identity", position = "dodge", alpha = 0.8) +
    geom_hline(yintercept = 0.06, linetype = "dashed", color = "red", linewidth = 0.6) +
    scale_fill_manual(values = c("Unadjusted" = "#95B3D7",
                                 "Adjusted (MMSE+Age+Sex)" = "#8064A2")) +
    coord_flip() +
    labs(title = "Stage Independence of Subtype MRI Markers",
         subtitle = "ANCOVA: effect sizes before and after adjusting for disease stage and demographics",
         x = NULL, y = expression(eta^2), fill = NULL) +
    theme_classic(base_size = 11) +
    theme(plot.title = element_text(face = "bold", size = 13),
          axis.text.y = element_text(size = 7),
          legend.position = "bottom")

  ggsave(file.path(output_dir, "Figure_Stage_Independence.png"),
         plot_stage, width = 11, height = 10, dpi = 300)
  cat("  Saved: Figure_Stage_Independence.png\n")
}
# --- Figure C: Disproportionate Atrophy (W-score boxplots, top 15) ---
if (nrow(disprop_residuals) > 0) {
  top_disprop <- head(disprop_stats$ST_Code, 15)
  plot_resid <- disprop_residuals %>%
    filter(ST_Code %in% top_disprop) %>%
    mutate(Anatomy = str_wrap(Anatomy, width = 30)) %>%
    ggplot(aes(x = Anatomy, y = Residual, fill = Subtype)) +
    geom_boxplot(outlier.shape = NA, alpha = 0.8) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
    scale_fill_manual(values = c("1" = "#4472C4", "2" = "#8064A2", "3" = "#C0504D"),
                      name = "Subtype") +
    labs(title = "Disproportionate Atrophy Patterns (Top 15 Regions)",
         subtitle = "W-score residuals after controlling for global atrophy, age, and sex",
         y = "Standardized Residual (W-score)", x = NULL) +
    theme_classic(base_size = 11) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 7),
          plot.title = element_text(face = "bold", size = 13),
          legend.position = "bottom")

  ggsave(file.path(output_dir, "Figure_Disproportionate_Atrophy.png"),
         plot_resid, width = 12, height = 7, dpi = 300)
  cat("  Saved: Figure_Disproportionate_Atrophy.png\n")
}
# --- Figure D: Network-Level Comparison ---
if (nrow(network_results) > 0) {
  net_long <- network_results %>%
    dplyr::select(Network, Mean_S1, Mean_S2, Mean_S3) %>%
    pivot_longer(cols = starts_with("Mean_S"),
                 names_to = "Subtype", values_to = "Mean_Z") %>%
    mutate(Subtype = gsub("Mean_S", "Subtype ", Subtype))

  plot_network <- ggplot(net_long, aes(x = Network, y = Mean_Z, fill = Subtype)) +
    geom_bar(stat = "identity", position = "dodge", alpha = 0.85) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
    scale_fill_manual(values = c("Subtype 1" = "#4472C4",
                                 "Subtype 2" = "#8064A2",
                                 "Subtype 3" = "#C0504D")) +
    labs(title = "Network-Level MRI Profiles by Subtype",
         subtitle = "Mean z-score composites across brain networks",
         y = "Mean z-score", x = NULL, fill = NULL) +
    theme_classic(base_size = 12) +
    theme(axis.text.x = element_text(angle = 30, hjust = 1),
          plot.title = element_text(face = "bold", size = 13),
          legend.position = "bottom")

  ggsave(file.path(output_dir, "Figure_Network_Profiles.png"),
         plot_network, width = 10, height = 6, dpi = 300)
  cat("  Saved: Figure_Network_Profiles.png\n")
}
# --- Figure E: Heatmap (subtype x region mean z-scores) ---
heatmap_data <- mri_results %>%
  dplyr::select(Anatomy, Mean_S1, Mean_S2, Mean_S3) %>%
  pivot_longer(cols = starts_with("Mean_S"),
               names_to = "Subtype", values_to = "Mean_Z") %>%
  mutate(Subtype = gsub("Mean_S", "S", Subtype),
         Anatomy = str_wrap(Anatomy, width = 30))
plot_heatmap <- ggplot(heatmap_data,
                       aes(x = Subtype, y = reorder(Anatomy, Mean_Z), fill = Mean_Z)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(low = "#2166AC", mid = "white", high = "#B2182B",
                       midpoint = 0, name = "Mean z-score") +
  labs(title = "MRI Subtype Profiles: Region-Level Heatmap",
       x = "Subtype", y = NULL) +
  theme_minimal(base_size = 11) +
  theme(plot.title = element_text(face = "bold", size = 13),
        axis.text.y = element_text(size = 6))
ggsave(file.path(output_dir, "Figure_MRI_Heatmap.png"),
       plot_heatmap, width = 8, height = 12, dpi = 300)
cat("  Saved: Figure_MRI_Heatmap.png\n")
# ==============================================================================
# 10. Summary Report
# ==============================================================================
cat("\n  Generating summary report...\n")
summary_lines <- c(
  "================================================================================",
  "Step 22: MRI Subtype Characterization Report",
  "================================================================================",
  "",
  sprintf("Generated: %s", Sys.time()),
  sprintf("Sample size: %d", nrow(master)),
  sprintf("MRI features: %d (all mapped to anatomical names)", length(mri_cols_available)),
  sprintf("CSF features: %d", length(csf_cols_available)),
  "",
  "--------------------------------------------------------------------------------",
  "Subtype Distribution",
  "--------------------------------------------------------------------------------"
)
for (i in seq_along(st_dist)) {
  summary_lines <- c(summary_lines,
                     sprintf("  Subtype %s: %d (%.1f%%)",
                             names(st_dist)[i], st_dist[i], 100 * st_dist[i] / sum(st_dist)))
}
summary_lines <- c(summary_lines, "",
                   "--------------------------------------------------------------------------------",
                   "MRI Heterogeneity (ANOVA + FDR)",
                   "--------------------------------------------------------------------------------",
                   sprintf("  Significant (FDR < %.2f): %d/%d", FDR_THRESHOLD, n_sig, nrow(mri_results)),
                   sprintf("  Large effect (eta2 >= 0.14): %d", n_large),
                   sprintf("  Medium effect (eta2 >= 0.06): %d", n_medium),
                   ""
)
if (nrow(stage_results) > 0) {
  summary_lines <- c(summary_lines,
                     "--------------------------------------------------------------------------------",
                     "Stage Independence (ANCOVA: Subtype + MMSE + Age + Sex)",
                     "--------------------------------------------------------------------------------",
                     sprintf("  Stage-independent (FDR < %.2f): %d/%d",
                             FDR_THRESHOLD, sum(stage_results$Stage_Independent), nrow(stage_results)),
                     sprintf("  Median effect size change: %.1f%%",
                             median(stage_results$Pct_Change, na.rm = TRUE)),
                     ""
  )
}
summary_lines <- c(summary_lines,
                   "--------------------------------------------------------------------------------",
                   "Disproportionate Atrophy (W-score)",
                   "--------------------------------------------------------------------------------",
                   sprintf("  Topologically specific (FDR < %.2f): %d/%d",
                           FDR_THRESHOLD, n_topo, nrow(disprop_stats)),
                   ""
)
if (nrow(network_results) > 0) {
  summary_lines <- c(summary_lines,
                     "--------------------------------------------------------------------------------",
                     "Network-Level Results",
                     "--------------------------------------------------------------------------------"
  )
  for (i in 1:nrow(network_results)) {
    r <- network_results[i, ]
    summary_lines <- c(summary_lines,
                       sprintf("  %s (%d features): eta2=%.3f, p_FDR=%.4f (%s)",
                               r$Network, r$N_Features, r$Eta2, r$P_FDR, r$Eta2_Interp))
  }
  summary_lines <- c(summary_lines, "")
}
summary_lines <- c(summary_lines,
                   "--------------------------------------------------------------------------------",
                   "Output Files",
                   "--------------------------------------------------------------------------------",
                   sprintf("  %s/MRI_Heterogeneity_Results.csv", output_dir),
                   sprintf("  %s/MRI_Tukey_Pairwise.csv", output_dir),
                   sprintf("  %s/CSF_Subtype_Profiles.csv", output_dir),
                   sprintf("  %s/Stage_Independence_ANCOVA.csv", output_dir),
                   sprintf("  %s/Disproportionate_Atrophy_Stats.csv", output_dir),
                   sprintf("  %s/Network_Level_Results.csv", output_dir),
                   sprintf("  %s/Demographics_Clinical_By_Subtype.csv", output_dir),
                   "",
                   "================================================================================",
                   "Step 22 Complete",
                   "================================================================================"
)
report_path <- file.path(output_dir, "MRI_Subtype_Report.txt")
writeLines(summary_lines, report_path)
cat(paste(summary_lines, collapse = "\n"))
cat("\n\n============================================================\n")
cat("Step 22: MRI Subtype Characterization Complete!\n")
cat("============================================================\n")
cat(sprintf("Report: %s\n", report_path))
cat(sprintf("Output: %s\n", output_dir))

