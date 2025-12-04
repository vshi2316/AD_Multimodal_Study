library(limma)
library(dplyr)
library(ggplot2)
library(ggrepel)
library(pheatmap)
library(tableone)
library(RColorBrewer)

cat("========================================================================\n")
cat("   Differential Analysis: limma + SMD (Integrated)\n")
cat("========================================================================\n\n")

## Load cluster results
cluster_results <- read.csv("cluster_results.csv", stringsAsFactors = FALSE)

## Define risk groups
high_risk_ids <- cluster_results %>% filter(Cluster_Labels == 2) %>% pull(ID)
low_risk_ids <- cluster_results %>% filter(Cluster_Labels == 1) %>% pull(ID)

risk_group <- data.frame(
  ID = c(high_risk_ids, low_risk_ids),
  Risk_Group = c(rep(1, length(high_risk_ids)), rep(0, length(low_risk_ids)))
)

cluster_results$Risk_Group <- ifelse(cluster_results$Cluster_Labels == 2, "HighRisk", "LowRisk")

cat(sprintf("Sample sizes: High-Risk=%d, Low-Risk=%d\n\n",
            length(high_risk_ids), length(low_risk_ids)))

## Define modalities
modalities <- list(
  Clinical = "Clinical_data.csv",
  sMRI = "RNA_plasma.csv",
  CSF = "metabolites.csv"
)

## ========================================================================
## Part 1: limma Differential Analysis
## ========================================================================

cat("========================================================================\n")
cat("PART 1: limma Differential Expression Analysis\n")
cat("========================================================================\n\n")

limma_results <- list()

for (modality_name in names(modalities)) {
  data_file <- modalities[[modality_name]]
  
  if (!file.exists(data_file)) {
    cat(sprintf("  Skipping %s (file not found)\n", modality_name))
    next
  }
  
  cat(sprintf("[%s] Processing...\n", modality_name))
  
  count_data <- read.csv(data_file, stringsAsFactors = FALSE)
  
  data <- count_data %>%
    filter(ID %in% risk_group$ID) %>%
    arrange(match(ID, risk_group$ID))
  
  feature_cols <- sapply(data, is.numeric)
  feature_cols["ID"] <- FALSE
  data_matrix <- data[, feature_cols]
  
  if (ncol(data_matrix) == 0) {
    cat(sprintf("  No numeric features in %s\n", modality_name))
    next
  }
  
  ## Design matrix
  design <- model.matrix(~ factor(risk_group$Risk_Group))
  colnames(design) <- c("Intercept", "HighRisk")
  
  ## limma fit
  fit <- lmFit(t(data_matrix), design)
  fit <- eBayes(fit, trend = TRUE)
  
  results <- topTable(fit, coef = "HighRisk", number = Inf)
  
  significant <- results %>%
    filter(P.Value < 0.05 & abs(logFC) > 0.5) %>%
    arrange(P.Value)
  
  cat(sprintf("  Total features: %d\n", nrow(results)))
  cat(sprintf("  Significant: %d (|logFC|>0.5, P<0.05)\n\n", nrow(significant)))
  
  limma_results[[modality_name]] <- list(
    all_results = results,
    significant = significant,
    data_matrix = data_matrix
  )
  
  ## Save results
  write.csv(results,
            sprintf("DiffExpr_%s_All.csv", modality_name),
            row.names = TRUE)
  
  if (nrow(significant) > 0) {
    write.csv(significant,
              sprintf("DiffExpr_%s_Significant.csv", modality_name),
              row.names = TRUE)
  }
}

## ========================================================================
## Part 2: SMD (Standardized Mean Difference) Analysis
## ========================================================================

cat("\n========================================================================\n")
cat("PART 2: Standardized Mean Difference (SMD) Analysis\n")
cat("========================================================================\n\n")

## Load and merge modality data
clinical <- read.csv("Clinical_data.csv", stringsAsFactors = FALSE)
smri <- read.csv("RNA_plasma.csv", stringsAsFactors = FALSE)
csf <- read.csv("metabolites.csv", stringsAsFactors = FALSE)

clinical_cols <- setdiff(colnames(clinical), "ID")
smri_cols <- setdiff(colnames(smri), "ID")
csf_cols <- setdiff(colnames(csf), "ID")

all_data <- cluster_results %>%
  left_join(clinical, by = "ID") %>%
  left_join(smri, by = "ID") %>%
  left_join(csf, by = "ID")

clinical_cols_valid <- clinical_cols[clinical_cols %in% colnames(all_data)]
smri_cols_valid <- smri_cols[smri_cols %in% colnames(all_data)]
csf_cols_valid <- csf_cols[csf_cols %in% colnames(all_data)]

modality_features <- list(
  Clinical = clinical_cols_valid,
  sMRI = smri_cols_valid,
  CSF = csf_cols_valid
)

all_smd_results <- data.frame()

for (modality_name in names(modality_features)) {
  feature_cols <- modality_features[[modality_name]]
  
  if (length(feature_cols) == 0) next
  
  cat(sprintf("[%s] Calculating SMD...\n", modality_name))
  
  feature_matrix <- all_data[, feature_cols, drop = FALSE]
  feature_matrix <- feature_matrix[, colSums(!is.na(feature_matrix)) > 0, drop = FALSE]
  
  if (ncol(feature_matrix) == 0) next
  
  ## Fill missing values with mean
  for (j in 1:ncol(feature_matrix)) {
    if (any(is.na(feature_matrix[, j]))) {
      feature_matrix[is.na(feature_matrix[, j]), j] <- mean(feature_matrix[, j], na.rm = TRUE)
    }
  }
  
  expr_matrix <- t(feature_matrix)
  
  ## Design matrix
  group <- factor(all_data$Risk_Group, levels = c("LowRisk", "HighRisk"))
  design <- model.matrix(~0 + group)
  colnames(design) <- c("LowRisk", "HighRisk")
  
  fit <- lmFit(expr_matrix, design)
  
  contrast_matrix <- makeContrasts(
    HighVsLow = HighRisk - LowRisk,
    levels = design
  )
  
  fit2 <- contrasts.fit(fit, contrast_matrix)
  fit2 <- eBayes(fit2)
  
  results <- topTable(fit2, coef = "HighVsLow", number = Inf, sort.by = "none")
  results$Feature <- rownames(results)
  results$Modality <- modality_name
  
  ## Calculate SMD manually
  high_risk_data <- feature_matrix[all_data$Risk_Group == "HighRisk", , drop = FALSE]
  low_risk_data <- feature_matrix[all_data$Risk_Group == "LowRisk", , drop = FALSE]
  
  mean_high <- colMeans(high_risk_data, na.rm = TRUE)
  mean_low <- colMeans(low_risk_data, na.rm = TRUE)
  sd_high <- apply(high_risk_data, 2, sd, na.rm = TRUE)
  sd_low <- apply(low_risk_data, 2, sd, na.rm = TRUE)
  
  n_high <- nrow(high_risk_data)
  n_low <- nrow(low_risk_data)
  
  pooled_sd <- sqrt(((n_high - 1) * sd_high^2 + (n_low - 1) * sd_low^2) / (n_high + n_low - 2))
  smd <- (mean_high - mean_low) / pooled_sd
  
  results$SMD <- smd[results$Feature]
  results$Mean_HighRisk <- mean_high[results$Feature]
  results$Mean_LowRisk <- mean_low[results$Feature]
  
  all_smd_results <- rbind(all_smd_results, results)
  
  cat(sprintf("  Features processed: %d\n", nrow(results)))
}

cat("\n")

## Filter significant SMD features
all_smd_results$abs_logFC <- abs(all_smd_results$logFC)
all_smd_results$abs_SMD <- abs(all_smd_results$SMD)

significant_smd <- all_smd_results %>%
  filter(abs_logFC > 0.2 & adj.P.Val < 0.1) %>%
  arrange(adj.P.Val, desc(abs_logFC))

cat(sprintf("Total SMD features analyzed: %d\n", nrow(all_smd_results)))
cat(sprintf("Significant SMD features: %d (|logFC|>0.2, FDR<0.1)\n\n", nrow(significant_smd)))

## Save SMD results
write.csv(all_smd_results, "SMD_All_Features.csv", row.names = FALSE)
write.csv(significant_smd, "SMD_Significant_Features.csv", row.names = FALSE)

## ========================================================================
## Part 3: Visualizations
## ========================================================================

cat("========================================================================\n")
cat("PART 3: Generating Visualizations\n")
cat("========================================================================\n\n")

## 3.1 Volcano plots (limma)
for (modality_name in names(limma_results)) {
  result <- limma_results[[modality_name]]
  if (is.null(result)) next
  
  cat(sprintf("  Creating volcano plot: %s\n", modality_name))
  
  plot_data <- result$all_results
  plot_data$Feature <- rownames(plot_data)
  
  plot_data$Regulation <- "NS"
  plot_data$Regulation[plot_data$P.Value < 0.05 & plot_data$logFC > 0.5] <- "Up in High-Risk"
  plot_data$Regulation[plot_data$P.Value < 0.05 & plot_data$logFC < -0.5] <- "Down in High-Risk"
  
  plot_data <- plot_data %>%
    arrange(P.Value) %>%
    mutate(Label = ifelse(row_number() <= 10 & P.Value < 0.05, Feature, ""))
  
  p <- ggplot(plot_data, aes(x = logFC, y = -log10(P.Value), color = Regulation)) +
    geom_point(aes(size = -log10(P.Value)), alpha = 0.7) +
    scale_size_continuous(range = c(1, 4)) +
    scale_color_manual(values = c("Up in High-Risk" = "#d62728",
                                  "Down in High-Risk" = "#1f77b4",
                                  "NS" = "grey70")) +
    geom_hline(yintercept = -log10(0.05), linetype = "dashed", color = "grey30") +
    geom_vline(xintercept = c(-0.5, 0.5), linetype = "dashed", color = "grey30") +
    geom_text_repel(aes(label = Label), size = 3.5, max.overlaps = 20) +
    labs(title = sprintf("%s: High-Risk vs Low-Risk", modality_name),
         x = expression(log[2]~"Fold Change"),
         y = expression(-log[10]~"P-value")) +
    theme_bw() +
    theme(legend.position = "right", plot.title = element_text(hjust = 0.5, face = "bold"))
  
  ggsave(sprintf("Volcano_%s.png", modality_name), plot = p, width = 10, height = 8, dpi = 300)
}

## 3.2 Heatmaps (limma significant features)
for (modality_name in names(limma_results)) {
  result <- limma_results[[modality_name]]
  
  if (is.null(result) || nrow(result$significant) < 2) next
  
  cat(sprintf("  Creating heatmap: %s\n", modality_name))
  
  sig_features <- rownames(result$significant)
  sig_features <- sig_features[sig_features %in% colnames(result$data_matrix)]
  
  if (length(sig_features) < 2) next
  
  sig_features <- head(sig_features, 50)
  
  heatmap_data <- t(result$data_matrix[, sig_features])
  heatmap_data_scaled <- t(scale(t(heatmap_data)))
  
  annotation_col <- data.frame(
    Risk_Group = factor(
      ifelse(risk_group$Risk_Group == 1, "High-Risk", "Low-Risk"),
      levels = c("High-Risk", "Low-Risk")
    )
  )
  rownames(annotation_col) <- risk_group$ID
  
  heatmap_data_scaled <- heatmap_data_scaled[, rownames(annotation_col)]
  
  ann_colors <- list(
    Risk_Group = c("High-Risk" = "#d62728", "Low-Risk" = "#1f77b4")
  )
  
  png(sprintf("Heatmap_%s.png", modality_name), width = 3600, height = 2400, res = 300)
  
  pheatmap(
    heatmap_data_scaled,
    annotation_col = annotation_col,
    annotation_colors = ann_colors,
    cluster_rows = TRUE,
    cluster_cols = TRUE,
    show_colnames = FALSE,
    show_rownames = TRUE,
    main = sprintf("%s: Differential Features", modality_name),
    color = colorRampPalette(rev(brewer.pal(11, "RdBu")))(100),
    border_color = NA
  )
  
  dev.off()
}

## 3.3 Top SMD features barplot
if (nrow(all_smd_results) > 0) {
  cat("  Creating Top SMD features plot\n")
  
  top_features <- all_smd_results %>%
    arrange(desc(abs_SMD)) %>%
    head(min(30, nrow(all_smd_results)))
  
  png("Top30_SMD_Features.png", width = 4000, height = 3200, res = 300)
  
  print(ggplot(top_features, aes(x = reorder(Feature, abs_SMD), y = abs_SMD, fill = Modality)) +
    geom_bar(stat = "identity") +
    coord_flip() +
    labs(title = "Top Features by Absolute SMD",
         x = "Feature",
         y = "|SMD|") +
    theme_minimal() +
    geom_hline(yintercept = 0.5, linetype = "dashed", color = "red"))
  
  dev.off()
}

## 3.4 SMD Volcano plot
if (nrow(all_smd_results) > 0) {
  cat("  Creating SMD volcano plot\n")
  
  png("Volcano_Plot_SMD.png", width = 4000, height = 3200, res = 300)
  
  all_smd_results$Significance <- ifelse(
    all_smd_results$abs_logFC > 0.2 & all_smd_results$adj.P.Val < 0.1,
    "Significant", "Not Significant"
  )
  
  print(ggplot(all_smd_results, aes(x = logFC, y = -log10(adj.P.Val), color = Significance)) +
    geom_point(alpha = 0.6, size = 3) +
    scale_color_manual(values = c("Significant" = "#e41a1c", "Not Significant" = "grey60")) +
    geom_hline(yintercept = -log10(0.1), linetype = "dashed", color = "blue") +
    geom_vline(xintercept = c(-0.2, 0.2), linetype = "dashed", color = "blue") +
    labs(title = "Volcano Plot: High-Risk vs Low-Risk (SMD)",
         x = "log2 Fold Change",
         y = "-log10(Adjusted P-value)") +
    theme_minimal())
  
  dev.off()
}

cat("\n")

## ========================================================================
## Part 4: Summary Report
## ========================================================================

cat("========================================================================\n")
cat("PART 4: Summary Report\n")
cat("========================================================================\n\n")

summary_report <- data.frame()

for (modality_name in names(limma_results)) {
  result <- limma_results[[modality_name]]
  
  top_features_str <- if(nrow(result$significant) > 0) {
    paste(head(rownames(result$significant), 5), collapse = "; ")
  } else {
    "None"
  }
  
  summary_report <- rbind(summary_report, data.frame(
    Modality = modality_name,
    N_Features = nrow(result$all_results),
    N_Significant_limma = nrow(result$significant),
    Top_Features = top_features_str
  ))
}

## Add SMD counts
smd_summary <- all_smd_results %>%
  group_by(Modality) %>%
  summarise(
    N_Significant_SMD = sum(abs_logFC > 0.2 & adj.P.Val < 0.1),
    .groups = "drop"
  )

summary_report <- summary_report %>%
  left_join(smd_summary, by = "Modality")

print(summary_report)

write.csv(summary_report, "DiffExpr_Summary.csv", row.names = FALSE)

cat("\n========================================================================\n")
cat("Differential Analysis Complete!\n")
cat("========================================================================\n\n")

cat("Output files:\n")
cat("  limma Results:\n")
for (modality_name in names(limma_results)) {
  cat(sprintf("    - DiffExpr_%s_All.csv\n", modality_name))
  if (nrow(limma_results[[modality_name]]$significant) > 0) {
    cat(sprintf("    - DiffExpr_%s_Significant.csv\n", modality_name))
  }
  cat(sprintf("    - Volcano_%s.png\n", modality_name))
  if (nrow(limma_results[[modality_name]]$significant) >= 2) {
    cat(sprintf("    - Heatmap_%s.png\n", modality_name))
  }
}

cat("\n  SMD Results:\n")
cat("    - SMD_All_Features.csv\n")
cat("    - SMD_Significant_Features.csv\n")
cat("    - Top30_SMD_Features.png\n")
cat("    - Volcano_Plot_SMD.png\n")

cat("\n  Summary:\n")
cat("    - DiffExpr_Summary.csv\n")
