library(limma)
library(dplyr)
library(ggplot2)
library(ggrepel)
library(pheatmap)
library(RColorBrewer)

## Load outcome data
outcome <- read.csv("Womac_score_pain_function.csv", stringsAsFactors = FALSE)
outcome$AD_Conversion <- as.numeric(outcome$AD_Conversion)

## Load multimodal data
clinical <- read.csv("Clinical_data.csv", stringsAsFactors = FALSE)
smri <- read.csv("RNA_plasma.csv", stringsAsFactors = FALSE)
csf <- read.csv("metabolites.csv", stringsAsFactors = FALSE)

clinical_cols <- setdiff(colnames(clinical), "ID")
smri_cols <- setdiff(colnames(smri), "ID")
csf_cols <- setdiff(colnames(csf), "ID")

## Merge data
all_data <- outcome %>%
  inner_join(clinical, by = "ID") %>%
  inner_join(smri, by = "ID") %>%
  inner_join(csf, by = "ID") %>%
  filter(!is.na(AD_Conversion))

## Validate features
clinical_cols_valid <- clinical_cols[clinical_cols %in% colnames(all_data)]
smri_cols_valid <- smri_cols[smri_cols %in% colnames(all_data)]
csf_cols_valid <- csf_cols[csf_cols %in% colnames(all_data)]

modalities <- list(
  Clinical = clinical_cols_valid,
  sMRI = smri_cols_valid,
  CSF = csf_cols_valid
)

## Limma differential analysis
all_diff_results <- data.frame()

for (modality_name in names(modalities)) {
  feature_cols <- modalities[[modality_name]]
  
  if (length(feature_cols) == 0) next
  
  feature_matrix <- all_data[, feature_cols, drop = FALSE]
  feature_matrix <- feature_matrix[, colSums(!is.na(feature_matrix)) > 0, drop = FALSE]
  
  if (ncol(feature_matrix) == 0) next
  
  ## Fill missing values
  for (j in 1:ncol(feature_matrix)) {
    if (any(is.na(feature_matrix[, j]))) {
      feature_matrix[is.na(feature_matrix[, j]), j] <- mean(feature_matrix[, j], na.rm = TRUE)
    }
  }
  
  expr_matrix <- t(feature_matrix)
  
  ## Design matrix
  group <- factor(all_data$AD_Conversion, levels = c(0, 1), labels = c("NonConverter", "Converter"))
  design <- model.matrix(~0 + group)
  colnames(design) <- c("NonConverter", "Converter")
  
  fit <- lmFit(expr_matrix, design)
  
  contrast_matrix <- makeContrasts(ConverterVsNon = Converter - NonConverter, levels = design)
  
  fit2 <- contrasts.fit(fit, contrast_matrix)
  fit2 <- eBayes(fit2)
  
  results <- topTable(fit2, coef = "ConverterVsNon", number = Inf, sort.by = "P")
  results$Feature <- rownames(results)
  results$Modality <- modality_name
  
  all_diff_results <- rbind(all_diff_results, results)
}

## Filter significant features
all_diff_results$abs_logFC <- abs(all_diff_results$logFC)
all_diff_results$Direction <- ifelse(all_diff_results$logFC > 0, "Up", "Down")

significant_features <- all_diff_results %>%
  filter(abs_logFC > 0.2 & adj.P.Val < 0.1) %>%
  arrange(adj.P.Val, desc(abs_logFC))

## Retain AD core biomarkers
ad_core_patterns <- c("ABETA", "TAU", "PTAU", "ADAS", "MMSE", "FAQ", "APOE",
                       "Hippocampus", "Entorhinal", "Amygdala", "Ventricle")

ad_core_features <- all_diff_results %>%
  filter(grepl(paste(ad_core_patterns, collapse = "|"), Feature, ignore.case = TRUE))

missing_cores <- ad_core_features %>%
  filter(!Feature %in% significant_features$Feature)

if (nrow(missing_cores) > 0) {
  significant_features_enhanced <- rbind(significant_features, missing_cores) %>%
    arrange(adj.P.Val)
} else {
  significant_features_enhanced <- significant_features
}

## Save results
write.csv(all_diff_results, "Conversion_Differential_All.csv", row.names = FALSE)
write.csv(significant_features_enhanced, "Conversion_Differential_Significant_Enhanced.csv", row.names = FALSE)

## Volcano plot
if (nrow(all_diff_results) > 0) {
  all_diff_results$Label <- ""
  top_features <- all_diff_results %>%
    arrange(adj.P.Val) %>%
    head(15)
  all_diff_results$Label[match(top_features$Feature, all_diff_results$Feature)] <- top_features$Feature
  
  all_diff_results$Significance <- case_when(
    all_diff_results$abs_logFC > 0.2 & all_diff_results$adj.P.Val < 0.1 & all_diff_results$logFC > 0 ~ "Up",
    all_diff_results$abs_logFC > 0.2 & all_diff_results$adj.P.Val < 0.1 & all_diff_results$logFC < 0 ~ "Down",
    TRUE ~ "NS"
  )
  
  png("Volcano_Plot_Conversion.png", width = 4000, height = 3200, res = 300)
  
  ggplot(all_diff_results, aes(x = logFC, y = -log10(adj.P.Val), color = Significance, label = Label)) +
    geom_point(alpha = 0.6, size = 2.5) +
    scale_color_manual(values = c("Up" = "#d62728", "Down" = "#2ca02c", "NS" = "grey70")) +
    geom_hline(yintercept = -log10(0.1), linetype = "dashed", color = "blue") +
    geom_vline(xintercept = c(-0.2, 0.2), linetype = "dashed", color = "blue") +
    geom_text_repel(size = 3, max.overlaps = 15) +
    labs(title = "Differential Analysis: AD Converters vs Non-Converters",
         x = "log2 Fold Change", y = "-log10(Adjusted P-value)") +
    theme_minimal() +
    theme(plot.title = element_text(face = "bold", hjust = 0.5))
  
  dev.off()
  
  ## Feature count summary
  if (nrow(significant_features_enhanced) > 0) {
    summary_data <- significant_features_enhanced %>%
      group_by(Modality, Direction) %>%
      summarise(Count = n(), .groups = "drop")
    
    png("Feature_Count_Summary.png", width = 3200, height = 2400, res = 300)
    
    ggplot(summary_data, aes(x = Modality, y = Count, fill = Direction)) +
      geom_bar(stat = "identity", position = "dodge") +
      scale_fill_manual(values = c("Up" = "#d62728", "Down" = "#2ca02c")) +
      geom_text(aes(label = Count), position = position_dodge(width = 0.9), vjust = -0.5) +
      labs(title = "Significant Features by Modality", x = "Modality", y = "Number of Features") +
      theme_minimal()
    
    dev.off()
  }
}

## Overlap with cluster differential features
if (file.exists("Differential_Analysis_Significant_Features.csv")) {
  cluster_diff <- read.csv("Differential_Analysis_Significant_Features.csv", stringsAsFactors = FALSE)
  
  overlap <- intersect(significant_features_enhanced$Feature, cluster_diff$Feature)
  
  if (length(overlap) > 0) {
    overlap_df <- significant_features_enhanced %>%
      filter(Feature %in% overlap) %>%
      select(Feature, Modality, logFC, adj.P.Val)
    
    write.csv(overlap_df, "Overlapping_Features_with_Cluster_Diff.csv", row.names = FALSE)
  }
}
