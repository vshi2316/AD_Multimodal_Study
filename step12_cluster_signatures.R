library(dplyr)
library(tidyr)
library(ggplot2)
library(pheatmap)
library(RColorBrewer)

## Load cluster results
cluster_results <- read.csv("cluster_results.csv", stringsAsFactors = FALSE)

## Load multimodal data
file_mapping <- list(
  Clinical = "Clinical_data.csv",
  sMRI = "RNA_plasma.csv",
  CSF = "metabolites.csv"
)

full_data <- data.frame(ID = cluster_results$ID)

for (modality in names(file_mapping)) {
  data_file <- file_mapping[[modality]]
  
  if (!file.exists(data_file)) next
  
  count_data <- read.csv(data_file, stringsAsFactors = FALSE)
  data <- count_data %>% filter(ID %in% cluster_results$ID)
  
  feature_cols <- sapply(data, is.numeric)
  feature_cols["ID"] <- FALSE
  
  data_features <- data[, feature_cols, drop = FALSE]
  colnames(data_features) <- paste0(modality, "_", colnames(data_features))
  
  full_data <- cbind(full_data, data_features)
}

## Load significant features
smd_file <- "SMD_Analysis_All_Modalities.csv"

if (file.exists(smd_file)) {
  smd_results <- read.csv(smd_file, stringsAsFactors = FALSE)
  
  top_features_per_cluster <- smd_results %>%
    group_by(Modality) %>%
    slice_max(order_by = abs(SMD), n = 10) %>%
    ungroup()
  
  top_features_per_cluster$Feature_Full <- paste0(
    top_features_per_cluster$Modality, "_",
    top_features_per_cluster$Feature
  )
  
  signature_features <- top_features_per_cluster$Feature_Full
} else {
  signature_features <- colnames(full_data)[-1]
}

available_features <- signature_features[signature_features %in% colnames(full_data)]

if (length(available_features) == 0) {
  feature_vars <- apply(full_data[, -1], 2, var, na.rm = TRUE)
  top_var_features <- names(sort(feature_vars, decreasing = TRUE))[1:min(30, length(feature_vars))]
  available_features <- top_var_features
}

## Prepare visualization data
filtered_data <- full_data[, c("ID", available_features)]
filtered_data$Cluster <- factor(cluster_results$Cluster_Labels)

plot_data <- filtered_data %>%
  pivot_longer(cols = -c(ID, Cluster), names_to = "Feature", values_to = "Value")

summary_plot_data <- plot_data %>%
  group_by(Cluster, Feature) %>%
  summarize(Mean_Value = mean(Value, na.rm = TRUE), .groups = "drop")

## Order features by Cluster 2
cluster2_means <- summary_plot_data %>%
  filter(Cluster == 2) %>%
  arrange(desc(Mean_Value))

summary_plot_data$Feature <- factor(summary_plot_data$Feature, levels = cluster2_means$Feature)

summary_plot_data <- summary_plot_data %>%
  mutate(Cluster_Name = case_when(
    Cluster == 1 ~ "Cluster 1 (Low-Risk)",
    Cluster == 2 ~ "Cluster 2 (High-Risk)",
    TRUE ~ as.character(Cluster)
  ))

## Overall signature profile
p_overall <- ggplot(summary_plot_data,
                    aes(x = Feature, y = Mean_Value, group = Cluster_Name, color = Cluster_Name)) +
  geom_point(size = 2.5, alpha = 0.8) +
  geom_line(linewidth = 1.2) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray40") +
  scale_color_manual(values = c("Cluster 1 (Low-Risk)" = "#1f77b4",
                                "Cluster 2 (High-Risk)" = "#d62728")) +
  labs(title = "Cluster Signature Profiles",
       x = "Feature", y = "Mean Standardized Value", color = "Cluster") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1, size = 8),
        legend.position = "top")

ggsave("Signature_Profile_Overall.png", plot = p_overall, width = 16, height = 10, dpi = 300)

## Per-modality profiles
for (modality in names(file_mapping)) {
  modality_features <- available_features[grepl(paste0("^", modality, "_"), available_features)]
  
  if (length(modality_features) == 0) next
  
  modality_data <- summary_plot_data %>% filter(Feature %in% modality_features)
  
  cluster2_order <- modality_data %>%
    filter(Cluster == 2) %>%
    arrange(desc(Mean_Value))
  
  modality_data$Feature <- factor(modality_data$Feature, levels = cluster2_order$Feature)
  
  modality_data <- modality_data %>%
    mutate(Feature_Clean = gsub(paste0("^", modality, "_"), "", Feature))
  
  p_modality <- ggplot(modality_data,
                       aes(x = Feature_Clean, y = Mean_Value, group = Cluster_Name, color = Cluster_Name)) +
    geom_point(size = 3, alpha = 0.8) +
    geom_line(linewidth = 1.3) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "gray40") +
    scale_color_manual(values = c("Cluster 1 (Low-Risk)" = "#1f77b4",
                                  "Cluster 2 (High-Risk)" = "#d62728")) +
    labs(title = sprintf("%s Signature Profile", modality),
         x = "Feature", y = "Mean Standardized Value", color = "Cluster") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),
          legend.position = "top")
  
  ggsave(sprintf("Signature_Profile_%s.png", modality), plot = p_modality, width = 12, height = 8, dpi = 300)
}

## Heatmap
heatmap_matrix <- summary_plot_data %>%
  select(Cluster, Feature, Mean_Value) %>%
  pivot_wider(names_from = Feature, values_from = Mean_Value) %>%
  column_to_rownames("Cluster") %>%
  as.matrix()

heatmap_matrix <- t(heatmap_matrix)

annotation_col <- data.frame(Risk_Group = c("Low-Risk", "High-Risk"))
rownames(annotation_col) <- c("1", "2")

ann_colors <- list(Risk_Group = c("Low-Risk" = "#1f77b4", "High-Risk" = "#d62728"))

png("Signature_Heatmap.png", width = 2400, height = 3200, res = 300)

pheatmap(heatmap_matrix,
         color = colorRampPalette(rev(brewer.pal(11, "RdBu")))(100),
         scale = "row",
         cluster_cols = FALSE,
         cluster_rows = TRUE,
         annotation_col = annotation_col,
         annotation_colors = ann_colors,
         show_rownames = TRUE,
         show_colnames = TRUE,
         fontsize_row = 6,
         main = "Cluster Signature Heatmap",
         breaks = seq(-2, 2, length.out = 101))

dev.off()

## Save results
write.csv(summary_plot_data, "Cluster_Signature_Data.csv", row.names = FALSE)

signature_wide <- summary_plot_data %>%
  select(Cluster, Feature, Mean_Value) %>%
  pivot_wider(names_from = Cluster, values_from = Mean_Value)

write.csv(signature_wide, "Cluster_Signature_Wide.csv", row.names = FALSE)
