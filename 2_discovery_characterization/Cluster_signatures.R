# ==============================================================================
# Multimodal subtype signature profiling and visualization
# ==============================================================================
library(dplyr)
library(tidyr)
library(ggplot2)
library(pheatmap)
library(RColorBrewer)
library(jsonlite)
library(stringr)
library(optparse)

option_list <- list(
  make_option(c("--vae_dir"), type = "character", default = ".",
              help = "Directory containing subtype_assignments.csv, vae_summary.json, and latent_representations.csv [default: %default]"),
  make_option(c("--data_dir"), type = "character", default = ".",
              help = "Directory containing Clinical_data.csv, RNA_plasma.csv, and metabolites.csv [default: %default]"),
  make_option(c("--output_dir"), type = "character", default = "./step12_results",
              help = "Output directory [default: %default]"),
  make_option(c("--fdr_threshold"), type = "double", default = 0.05,
              help = "False discovery rate threshold [default: %default]")
)

opt <- parse_args(OptionParser(option_list = option_list))

cat("==============================================================================\\n")
cat("Step 12: Cluster Signature Profiles\\n")
cat("==============================================================================\\n\\n")

vae_dir <- opt$vae_dir
data_dir <- opt$data_dir
output_dir <- opt$output_dir
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

FDR_THRESHOLD <- opt$fdr_threshold
set.seed(42)

# ==============================================================================
# 1. ST Code -> Anatomical Name Mapping (consistent with step22)
# ==============================================================================
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
get_anatomy <- function(st_code) {
  if (st_code %in% names(st_to_anatomy)) return(st_to_anatomy[st_code])
  return(st_code)
}
# ==============================================================================
# 2. Helper Functions
# ==============================================================================
calculate_eta_squared <- function(values, groups) {
  values <- as.numeric(values)
  groups <- as.factor(groups)
  valid_idx <- !is.na(values) & !is.na(groups)
  values <- values[valid_idx]
  groups <- groups[valid_idx]
  if (length(unique(groups)) < 2) return(NA)
  grand_mean <- mean(values)
  ss_total <- sum((values - grand_mean)^2)
  ss_between <- 0
  for (g in unique(groups)) {
    gv <- values[groups == g]
    ss_between <- ss_between + length(gv) * (mean(gv) - grand_mean)^2
  }
  return(ss_between / ss_total)
}
interpret_eta2 <- function(eta2) {
  if (is.na(eta2)) return("NA")
  if (eta2 >= 0.14) return("Large")
  if (eta2 >= 0.06) return("Medium")
  if (eta2 >= 0.01) return("Small")
  return("Negligible")
}
calculate_cohens_d <- function(group1, group2) {
  g1 <- as.numeric(group1[!is.na(group1)])
  g2 <- as.numeric(group2[!is.na(group2)])
  n1 <- length(g1); n2 <- length(g2)
  if (n1 < 2 || n2 < 2) return(NA)
  pooled_sd <- sqrt(((n1 - 1) * var(g1) + (n2 - 1) * var(g2)) / (n1 + n2 - 2))
  if (is.na(pooled_sd) || pooled_sd < 1e-12) return(NA)
  return((mean(g1) - mean(g2)) / pooled_sd)
}
# ==============================================================================
# 3. Load and Merge Data
# ==============================================================================
cat("[1/6] Loading and merging data...\n")
# --- VAE outputs ---
subtypes <- read.csv(file.path(vae_dir, "subtype_assignments.csv"),
                     stringsAsFactors = FALSE)
latent   <- read.csv(file.path(vae_dir, "latent_representations.csv"),
                     stringsAsFactors = FALSE)
vae_json <- fromJSON(file.path(vae_dir, "vae_summary.json"))
subtypes$ID <- as.character(subtypes$ID)
latent$ID   <- as.character(latent$ID)
# Feature lists from VAE
csf_features  <- vae_json$csf_features     # e.g. PTAU181, AB42_40, ABETA40
clin_features <- vae_json$clinical_features # e.g. APOE4, MMSE, EDUCATION, GDS
mri_features  <- vae_json$mri_features      # 30 ST codes
cat(sprintf("  VAE: %d participants, %d CSF + %d Clin + %d MRI = %d features\n",
            nrow(subtypes), length(csf_features), length(clin_features),
            length(mri_features),
            length(csf_features) + length(clin_features) + length(mri_features)))
# Determine subtype column
if ("VAE_Subtype" %in% colnames(subtypes)) {
  subtypes$Subtype <- subtypes$VAE_Subtype
} else if (!"Subtype" %in% colnames(subtypes)) {
  stop("Cannot find VAE_Subtype or Subtype column")
}
subtypes$Subtype <- factor(subtypes$Subtype)
# --- Original data ---
clinical <- read.csv(file.path(data_dir, "Clinical_data.csv"),
                     stringsAsFactors = FALSE)
smri     <- read.csv(file.path(data_dir, "RNA_plasma.csv"),
                     stringsAsFactors = FALSE)
csf      <- read.csv(file.path(data_dir, "metabolites.csv"),
                     stringsAsFactors = FALSE)
clinical$ID <- as.character(clinical$ID)
smri$ID     <- as.character(smri$ID)
csf$ID      <- as.character(csf$ID)
# --- Build master data frame ---
# Start with subtypes (ID, Subtype, AD_Conversion)
keep_cols <- intersect(c("ID", "Subtype", "AD_Conversion"), colnames(subtypes))
master <- subtypes[, keep_cols, drop = FALSE]
# Merge latent variables Z1-Z3
master <- merge(master, latent, by = "ID", all.x = TRUE)
# Merge clinical
clin_cols_want <- c("MMSE", "EDUCATION", "GDS", "APOE4_DOSAGE",
                    "ADAS13", "CDRSB", "FAQTOTAL", "AGE", "SEX")
clin_cols_avail <- intersect(clin_cols_want, colnames(clinical))
master <- merge(master, clinical[, c("ID", clin_cols_avail)],
                by = "ID", all.x = TRUE)
# Merge MRI (30 ST features)
mri_cols_avail <- intersect(mri_features, colnames(smri))
master <- merge(master, smri[, c("ID", mri_cols_avail)],
                by = "ID", all.x = TRUE)
# Merge CSF
csf_cols_want <- c("PTAU181", "ABETA42_ABETA40_RATIO", "ABETA40")
csf_cols_avail <- intersect(csf_cols_want, colnames(csf))
if (length(csf_cols_avail) > 0) {
  master <- merge(master, csf[, c("ID", csf_cols_avail)],
                  by = "ID", all.x = TRUE)
}
cat(sprintf("  Merged: %d participants x %d columns\n",
            nrow(master), ncol(master)))
# --- Subtype distribution ---
n_subtypes <- length(levels(master$Subtype))
cat("\n  Subtype distribution:\n")
for (st in levels(master$Subtype)) {
  n_st <- sum(master$Subtype == st)
  cat(sprintf("    Subtype %s: n=%d (%.1f%%)\n",
              st, n_st, 100 * n_st / nrow(master)))
}
# --- Conversion rates (for ordering subtypes) ---
conv_rates <- NULL
if ("AD_Conversion" %in% colnames(master)) {
  conv_rates <- tapply(master$AD_Conversion, master$Subtype, mean, na.rm = TRUE)
  cat("\n  AD Conversion rates:\n")
  for (st in names(sort(conv_rates, decreasing = TRUE))) {
    cat(sprintf("    Subtype %s: %.1f%%\n", st, conv_rates[st] * 100))
  }
  # Order: High-risk first
  subtype_order <- names(sort(conv_rates, decreasing = TRUE))
} else {
  subtype_order <- levels(master$Subtype)
}
# Subtype labels with risk annotation
subtype_labels <- setNames(
  paste0("Subtype ", levels(master$Subtype)),
  levels(master$Subtype)
)
if (!is.null(conv_rates)) {
  risk_rank <- rank(-conv_rates)
  risk_names <- c("High-Risk", "Intermediate-Risk", "Low-Risk")
  for (st in names(conv_rates)) {
    r <- risk_rank[st]
    if (r <= length(risk_names)) {
      subtype_labels[st] <- sprintf("Subtype %s (%s, %.0f%%)",
                                    st, risk_names[r],
                                    conv_rates[st] * 100)
    }
  }
}
# ==============================================================================
# 4. Define Feature Sets for Signature Analysis
# ==============================================================================
cat("\n[2/6] Defining feature sets...\n")
# VAE input features (37) + latent Z1-Z3 = 40 features for signature
# Exclude: FAQ, ADAS13, CDRSB (circularity), SEX, AGE (confounding)
# But INCLUDE ADAS13/CDRSB/FAQ for DESCRIPTIVE profiling (not prediction)
# Core VAE input features
vae_input_features <- c(csf_cols_avail, 
                        intersect(c("APOE4_DOSAGE", "MMSE", "EDUCATION", "GDS"),
                                  colnames(master)),
                        mri_cols_avail)
# Latent variables
z_cols <- grep("^Z\\d+$", colnames(master), value = TRUE)
# Descriptive clinical features (for profiling, NOT prediction)
desc_features <- intersect(c("ADAS13", "CDRSB", "FAQTOTAL", "AGE"),
                           colnames(master))
# All signature features
signature_features <- c(vae_input_features, z_cols)
# Keep only numeric and available
signature_features <- signature_features[
  signature_features %in% colnames(master) &
    sapply(signature_features, function(f) is.numeric(master[[f]]))
]
# Full feature set (signature + descriptive)
all_analysis_features <- unique(c(signature_features, desc_features))
all_analysis_features <- all_analysis_features[
  all_analysis_features %in% colnames(master) &
    sapply(all_analysis_features, function(f) is.numeric(master[[f]]))
]
# Assign modality labels
get_modality <- function(feat) {
  if (feat %in% csf_cols_avail) return("CSF")
  if (feat %in% c("APOE4_DOSAGE", "MMSE", "EDUCATION", "GDS")) return("Clinical")
  if (feat %in% mri_cols_avail) return("MRI")
  if (grepl("^Z\\d+$", feat)) return("VAE_Latent")
  if (feat %in% c("ADAS13", "CDRSB", "FAQTOTAL")) return("Cognitive")
  if (feat == "AGE") return("Demographic")
  return("Other")
}
cat(sprintf("  VAE input features: %d\n", length(vae_input_features)))
cat(sprintf("  Latent variables: %d\n", length(z_cols)))
cat(sprintf("  Descriptive features: %d\n", length(desc_features)))
cat(sprintf("  Total for analysis: %d\n", length(all_analysis_features)))
# ==============================================================================
# 5. Statistical Analysis (ANOVA + FDR + Pairwise Cohen's d)
# ==============================================================================
cat("\n[3/6] Statistical analysis (ANOVA + FDR + pairwise Cohen's d)...\n")
stat_results <- data.frame()
pairwise_results <- data.frame()
for (feat in all_analysis_features) {
  values <- master[[feat]]
  groups <- master$Subtype

  # Skip if >50% missing
  if (mean(is.na(values)) > 0.5) next

  # Median imputation for analysis
  values[is.na(values)] <- median(values, na.rm = TRUE)

  # Modality
  modality <- get_modality(feat)

  # Display name
  display_name <- if (feat %in% names(st_to_anatomy)) get_anatomy(feat) else feat

  # Kruskal-Wallis
  kw <- tryCatch(kruskal.test(values ~ groups),
                 error = function(e) list(statistic = NA, p.value = NA))

  # ANOVA
  aov_fit <- tryCatch(aov(values ~ groups), error = function(e) NULL)
  anova_p <- NA
  if (!is.null(aov_fit)) {
    s <- summary(aov_fit)
    anova_p <- s[[1]][["Pr(>F)"]][1]
  }

  # Eta-squared
  eta2 <- calculate_eta_squared(values, groups)

  # Per-subtype means and SDs
  means <- tapply(values, groups, mean, na.rm = TRUE)
  sds   <- tapply(values, groups, sd, na.rm = TRUE)

  row <- data.frame(
    Feature       = feat,
    Display_Name  = display_name,
    Modality      = modality,
    KW_Statistic  = as.numeric(kw$statistic),
    KW_P_Value    = kw$p.value,
    ANOVA_P_Value = anova_p,
    Eta2          = eta2,
    Eta2_Interp   = interpret_eta2(eta2),
    stringsAsFactors = FALSE
  )

  # Add per-subtype means/SDs
  for (st in levels(groups)) {
    row[[paste0("Mean_S", st)]] <- means[st]
    row[[paste0("SD_S", st)]]   <- sds[st]
  }

  stat_results <- rbind(stat_results, row)

  # Pairwise Cohen's d
  st_levels <- levels(groups)
  for (i in 1:(length(st_levels) - 1)) {
    for (j in (i + 1):length(st_levels)) {
      g1 <- values[groups == st_levels[i]]
      g2 <- values[groups == st_levels[j]]
      d  <- calculate_cohens_d(g1, g2)
      pairwise_results <- rbind(pairwise_results, data.frame(
        Feature      = feat,
        Display_Name = display_name,
        Modality     = modality,
        Comparison   = paste0("S", st_levels[i], "_vs_S", st_levels[j]),
        Cohens_d     = d,
        Abs_d        = abs(d),
        stringsAsFactors = FALSE
      ))
    }
  }
}
# FDR correction
stat_results$P_FDR <- p.adjust(stat_results$KW_P_Value, method = "BH")
stat_results$FDR_Significant <- stat_results$P_FDR < FDR_THRESHOLD
stat_results <- stat_results %>% arrange(desc(Eta2))
n_sig   <- sum(stat_results$FDR_Significant, na.rm = TRUE)
n_large <- sum(stat_results$Eta2 >= 0.14, na.rm = TRUE)
n_med   <- sum(stat_results$Eta2 >= 0.06 & stat_results$Eta2 < 0.14, na.rm = TRUE)
cat(sprintf("  FDR-significant (q < %.2f): %d/%d\n",
            FDR_THRESHOLD, n_sig, nrow(stat_results)))
cat(sprintf("  Large effect (eta2 >= 0.14): %d\n", n_large))
cat(sprintf("  Medium effect (eta2 >= 0.06): %d\n", n_med))
write.csv(stat_results,
          file.path(output_dir, "Subtype_Signature_Stats_FDR.csv"),
          row.names = FALSE)
write.csv(pairwise_results,
          file.path(output_dir, "Subtype_Pairwise_Cohens_d.csv"),
          row.names = FALSE)
# ==============================================================================
# 6. Prepare Standardized Profile Data
# ==============================================================================
cat("\n[4/6] Preparing standardized profiles...\n")
# Standardize all features for visualization (z-score within sample)
profile_matrix <- master[, c("ID", "Subtype", signature_features), drop = FALSE]
for (col in signature_features) {
  vals <- profile_matrix[[col]]
  vals[is.na(vals)] <- median(vals, na.rm = TRUE)
  profile_matrix[[col]] <- as.numeric(scale(vals))
}
# Calculate subtype mean profiles
profile_long <- profile_matrix %>%
  pivot_longer(cols = all_of(signature_features),
               names_to = "Feature", values_to = "Value") %>%
  group_by(Subtype, Feature) %>%
  summarise(Mean = mean(Value, na.rm = TRUE),
            SD   = sd(Value, na.rm = TRUE),
            N    = n(),
            SE   = SD / sqrt(N),
            .groups = "drop")
# Add metadata
profile_long$Modality     <- sapply(profile_long$Feature, get_modality)
profile_long$Display_Name <- sapply(profile_long$Feature, function(f) {
  if (f %in% names(st_to_anatomy)) get_anatomy(f) else f
})
profile_long$Subtype_Label <- subtype_labels[as.character(profile_long$Subtype)]
# Merge effect sizes
profile_long <- profile_long %>%
  left_join(stat_results %>% dplyr::select(Feature, Eta2, P_FDR, FDR_Significant),
            by = "Feature")
# Wide format for heatmap
profile_wide <- profile_long %>%
  dplyr::select(Feature, Display_Name, Modality, Subtype, Mean) %>%
  pivot_wider(names_from = Subtype, values_from = Mean,
              names_prefix = "Subtype_")
write.csv(profile_wide,
          file.path(output_dir, "Subtype_Signature_Profiles_Wide.csv"),
          row.names = FALSE)
# ==============================================================================
# 7. Figure 4A: Overall Signature Profile (line plot)
# ==============================================================================
cat("\n[5/6] Generating Figure 4 panels...\n")
# --- 4A: Overall signature profile ---
# Grey dashed line at y=0 = population mean (z-score reference)
# Features ordered by eta-squared (largest effect first)
# Subtypes ordered by conversion rate
cat("  Figure 4A: Overall signature profile...\n")
# Feature order by effect size
feat_order_eta2 <- stat_results %>%
  filter(Feature %in% signature_features) %>%
  arrange(desc(Eta2)) %>%
  pull(Feature)
plot_4a_data <- profile_long %>%
  filter(Feature %in% signature_features) %>%
  mutate(Feature = factor(Feature, levels = feat_order_eta2),
         Subtype_Label = factor(Subtype_Label,
                                levels = subtype_labels[subtype_order]))
# Color palette: ordered by risk
subtype_colors <- c("#C0504D", "#8064A2", "#4472C4")  # Red, Purple, Blue
names(subtype_colors) <- subtype_labels[subtype_order]
p_4a <- ggplot(plot_4a_data,
               aes(x = Feature, y = Mean,
                   group = Subtype_Label, color = Subtype_Label)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50", linewidth = 0.8) +
  geom_line(linewidth = 1.1, alpha = 0.9) +
  geom_point(size = 2.2, alpha = 0.9) +
  scale_color_manual(values = subtype_colors) +
  labs(title = "Figure 4A: Multimodal Subtype Signature Profiles",
       subtitle = paste0("Features ordered by effect size (eta-squared). ",
                         "Grey dashed line = population mean (z = 0)."),
       x = NULL, y = "Mean Standardized Value (z-score)",
       color = "Subtype") +
  theme_minimal(base_size = 11) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1, size = 6),
        legend.position = "top",
        plot.title = element_text(face = "bold", size = 13))
ggsave(file.path(output_dir, "Figure4A_Signature_Profile_Overall.png"),
       plot = p_4a, width = 18, height = 8, dpi = 300)
cat("  Saved: Figure4A_Signature_Profile_Overall.png\n")
# --- 4B: Heatmap (subtype x feature, grouped by modality) ---
cat("  Figure 4B: Signature heatmap...\n")
# Select features: FDR-significant OR medium+ effect
heatmap_features <- stat_results %>%
  filter(Feature %in% signature_features,
         FDR_Significant | Eta2 >= 0.06) %>%
  arrange(Modality, desc(Eta2)) %>%
  pull(Feature)
if (length(heatmap_features) < 5) {
  heatmap_features <- head(stat_results$Feature[
    stat_results$Feature %in% signature_features], 30)
}
# Build heatmap matrix
hm_wide <- profile_wide %>%
  filter(Feature %in% heatmap_features)
# Order rows by modality then effect size
hm_order <- stat_results %>%
  filter(Feature %in% heatmap_features) %>%
  arrange(Modality, desc(Eta2))
hm_wide <- hm_wide[match(hm_order$Feature, hm_wide$Feature), ]
hm_mat <- as.matrix(hm_wide[, grep("^Subtype_", colnames(hm_wide))])
rownames(hm_mat) <- hm_wide$Display_Name
# Row annotation
row_ann <- data.frame(
  Modality = hm_wide$Modality,
  row.names = hm_wide$Display_Name
)
# Add effect size annotation
eff_sizes <- stat_results$Eta2_Interp[match(hm_wide$Feature, stat_results$Feature)]
row_ann$Effect <- eff_sizes
ann_colors <- list(
  Modality = c("CSF" = "#E41A1C", "Clinical" = "#377EB8",
               "MRI" = "#4DAF4A", "VAE_Latent" = "#984EA3"),
  Effect = c("Large" = "#d62728", "Medium" = "#ff7f0e",
             "Small" = "#2ca02c", "Negligible" = "grey70")
)
# Column names: Subtype labels
colnames(hm_mat) <- paste0("Subtype ", gsub("Subtype_", "", colnames(hm_mat)))
png(file.path(output_dir, "Figure4B_Signature_Heatmap.png"),
    width = 2000, height = max(1800, nrow(hm_mat) * 45), res = 300)
pheatmap(hm_mat,
         color = colorRampPalette(rev(brewer.pal(11, "RdBu")))(100),
         scale = "none",  # already z-scored
         cluster_cols = FALSE,
         cluster_rows = FALSE,  # keep modality grouping
         annotation_row = row_ann,
         annotation_colors = ann_colors,
         show_rownames = TRUE,
         show_colnames = TRUE,
         fontsize_row = 7,
         fontsize_col = 10,
         main = "Figure 4B: Subtype Signature Heatmap",
         breaks = seq(-1.5, 1.5, length.out = 101),
         gaps_row = cumsum(table(hm_wide$Modality)[
           unique(hm_wide$Modality)]))
dev.off()
cat("  Saved: Figure4B_Signature_Heatmap.png\n")
# --- 4C-E: Per-modality profiles (ordered by conversion rate) ---
cat("  Figure 4C-E: Per-modality profiles...\n")
modality_panels <- list(
  "C" = list(name = "CSF", features = csf_cols_avail),
  "D" = list(name = "Clinical", features = intersect(
    c("APOE4_DOSAGE", "MMSE", "EDUCATION", "GDS"), colnames(master))),
  "E" = list(name = "MRI", features = mri_cols_avail)
)
for (panel_id in names(modality_panels)) {
  panel <- modality_panels[[panel_id]]
  feats <- panel$features
  feats <- feats[feats %in% signature_features]
  if (length(feats) == 0) next

  # Order features by effect size within modality
  feat_order <- stat_results %>%
    filter(Feature %in% feats) %>%
    arrange(desc(Eta2)) %>%
    pull(Feature)

  panel_data <- profile_long %>%
    filter(Feature %in% feats) %>%
    mutate(
      Display_Name = factor(Display_Name,
                            levels = sapply(feat_order, function(f) {
                              if (f %in% names(st_to_anatomy)) get_anatomy(f) else f
                            })),
      Subtype_Label = factor(Subtype_Label,
                             levels = subtype_labels[subtype_order])
    )

  p_panel <- ggplot(panel_data,
                    aes(x = Display_Name, y = Mean,
                        group = Subtype_Label, color = Subtype_Label)) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "gray50",
               linewidth = 0.7) +
    geom_line(linewidth = 1.2, alpha = 0.9) +
    geom_point(size = 3, alpha = 0.9) +
    geom_errorbar(aes(ymin = Mean - SE, ymax = Mean + SE),
                  width = 0.2, alpha = 0.5) +
    scale_color_manual(values = subtype_colors) +
    labs(title = sprintf("Figure 4%s: %s Subtype Profiles", panel_id, panel$name),
         subtitle = "Subtypes ordered by conversion rate. Error bars = SE.",
         x = NULL, y = "Mean Standardized Value",
         color = "Subtype") +
    theme_minimal(base_size = 11) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
          legend.position = "top",
          plot.title = element_text(face = "bold", size = 13))

  fname <- sprintf("Figure4%s_%s_Profile.png", panel_id, panel$name)
  ggsave(file.path(output_dir, fname),
         plot = p_panel,
         width = max(8, length(feats) * 0.8),
         height = 7, dpi = 300)
  cat(sprintf("  Saved: %s\n", fname))
}
# --- VAE Latent Space Profile ---
cat("  Figure 4F: VAE latent variable profiles...\n")
if (length(z_cols) > 0) {
  z_data <- profile_long %>%
    filter(Feature %in% z_cols) %>%
    mutate(Subtype_Label = factor(Subtype_Label,
                                  levels = subtype_labels[subtype_order]))

  p_latent <- ggplot(z_data,
                     aes(x = Feature, y = Mean, fill = Subtype_Label)) +
    geom_bar(stat = "identity", position = position_dodge(width = 0.8),
             alpha = 0.85, width = 0.7) +
    geom_errorbar(aes(ymin = Mean - SE, ymax = Mean + SE),
                  position = position_dodge(width = 0.8),
                  width = 0.2, alpha = 0.6) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
    scale_fill_manual(values = subtype_colors) +
    labs(title = "Figure 4F: VAE Latent Variable Profiles",
         subtitle = "Mean z-score per subtype for each latent dimension",
         x = "Latent Dimension", y = "Mean Standardized Value",
         fill = "Subtype") +
    theme_minimal(base_size = 12) +
    theme(legend.position = "top",
          plot.title = element_text(face = "bold", size = 13))

  ggsave(file.path(output_dir, "Figure4F_VAE_Latent_Profile.png"),
         plot = p_latent, width = 8, height = 6, dpi = 300)
  cat("  Saved: Figure4F_VAE_Latent_Profile.png\n")
}
# --- Effect Size Distribution ---
cat("  Effect size distribution plot...\n")
p_effect <- ggplot(stat_results %>% filter(Feature %in% signature_features),
                   aes(x = Eta2, fill = Eta2_Interp)) +
  geom_histogram(bins = 25, alpha = 0.8, color = "white") +
  geom_vline(xintercept = c(0.01, 0.06, 0.14),
             linetype = "dashed", color = "red", linewidth = 0.6) +
  scale_fill_manual(values = c("Large" = "#d62728", "Medium" = "#ff7f0e",
                               "Small" = "#2ca02c", "Negligible" = "grey70")) +
  labs(title = "Distribution of Effect Sizes Across Signature Features",
       subtitle = "Cohen's thresholds: 0.01 (small), 0.06 (medium), 0.14 (large)",
       x = expression(eta^2), y = "Count", fill = "Effect Size") +
  theme_minimal(base_size = 11) +
  theme(plot.title = element_text(face = "bold"))
ggsave(file.path(output_dir, "Effect_Size_Distribution.png"),
       plot = p_effect, width = 10, height = 6, dpi = 300)
cat("  Saved: Effect_Size_Distribution.png\n")
# ==============================================================================
# 8. Summary Report
# ==============================================================================
cat("\n[6/6] Generating summary report...\n")
# Modality summary
modality_summary <- stat_results %>%
  filter(Feature %in% signature_features) %>%
  group_by(Modality) %>%
  summarise(
    N_Features       = n(),
    N_FDR_Sig        = sum(FDR_Significant, na.rm = TRUE),
    N_Large_Effect   = sum(Eta2 >= 0.14, na.rm = TRUE),
    N_Medium_Effect  = sum(Eta2 >= 0.06 & Eta2 < 0.14, na.rm = TRUE),
    Mean_Eta2        = mean(Eta2, na.rm = TRUE),
    Max_Eta2         = max(Eta2, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(desc(Mean_Eta2))
cat("\n  Modality Summary:\n")
print(as.data.frame(modality_summary))
write.csv(modality_summary,
          file.path(output_dir, "Subtype_Signature_Modality_Summary.csv"),
          row.names = FALSE)
# Top discriminating features
cat("\n  Top 15 discriminating features:\n")
top15 <- head(stat_results %>% filter(Feature %in% signature_features), 15)
for (i in 1:nrow(top15)) {
  r <- top15[i, ]
  sig <- ifelse(r$FDR_Significant, "***", "")
  cat(sprintf("    %2d. %s [%s] eta2=%.3f (%s) p_FDR=%.4f%s\n",
              i, r$Display_Name, r$Modality, r$Eta2, r$Eta2_Interp,
              r$P_FDR, sig))
}
# Summary report text file
summary_lines <- c(
  "================================================================================",
  "Step 12: Cluster Signature Analysis — Summary Report",
  "================================================================================",
  "",
  sprintf("Participants: %d", nrow(master)),
  sprintf("Subtypes: %d", n_subtypes),
  sprintf("Signature features analyzed: %d", length(signature_features)),
  sprintf("FDR threshold: %.2f", FDR_THRESHOLD),
  "",
  "--------------------------------------------------------------------------------",
  "Subtype Distribution",
  "--------------------------------------------------------------------------------"
)
for (st in subtype_order) {
  n_st <- sum(master$Subtype == st)
  cr <- if (!is.null(conv_rates)) sprintf(", conversion=%.1f%%", conv_rates[st] * 100) else ""
  summary_lines <- c(summary_lines,
                     sprintf("  Subtype %s: n=%d (%.1f%%)%s", st, n_st, 100 * n_st / nrow(master), cr))
}
summary_lines <- c(summary_lines, "",
                   "--------------------------------------------------------------------------------",
                   "Statistical Results",
                   "--------------------------------------------------------------------------------",
                   sprintf("  FDR-significant features: %d/%d", n_sig, nrow(stat_results)),
                   sprintf("  Large effect (eta2 >= 0.14): %d", n_large),
                   sprintf("  Medium effect (eta2 >= 0.06): %d", n_med),
                   "",
                   "--------------------------------------------------------------------------------",
                   "Figure 4 Panels",
                   "--------------------------------------------------------------------------------",
                   "  4A: Overall signature profile (all features, ordered by eta-squared)",
                   "      Grey dashed line = population mean (z = 0)",
                   "  4B: Signature heatmap (grouped by modality, FDR-significant features)",
                   "  4C: CSF biomarker profiles by subtype",
                   "  4D: Clinical feature profiles by subtype",
                   "  4E: MRI regional profiles by subtype (anatomical names)",
                   "  4F: VAE latent variable profiles by subtype",
                   "  All panels: subtypes ordered by conversion rate (high-risk first)",
                   "",
                   "--------------------------------------------------------------------------------",
                   "Output Files",
                   "--------------------------------------------------------------------------------",
                   sprintf("  %s/Subtype_Signature_Stats_FDR.csv", output_dir),
                   sprintf("  %s/Subtype_Pairwise_Cohens_d.csv", output_dir),
                   sprintf("  %s/Subtype_Signature_Profiles_Wide.csv", output_dir),
                   sprintf("  %s/Subtype_Signature_Modality_Summary.csv", output_dir),
                   sprintf("  %s/Figure4A-F (6 PNG files)", output_dir),
                   "",
                   "================================================================================",
                   "Step 12 Complete",
                   "================================================================================"
)
report_path <- file.path(output_dir, "Signature_Report.txt")
writeLines(summary_lines, report_path)
cat(paste(summary_lines, collapse = "\n"))
cat("\n\n============================================================\n")
cat("Step 12: Cluster Signature Analysis Complete!\n")
cat("============================================================\n")
cat(sprintf("Report: %s\n", report_path))
cat(sprintf("Output: %s\n", output_dir))

