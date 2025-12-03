library(dplyr)
library(ggplot2)
library(pheatmap)
library(car)

## Load data
adni_data <- read.csv("Cohort_A_Integrated.csv", stringsAsFactors = FALSE)
cluster_data <- read.csv("Final_Consensus_Clusters_K3.csv", stringsAsFactors = FALSE)

## Detect cluster label column
possible_cols <- c("Cluster", "Consensus_Cluster_K3", "Subtype")
cluster_col <- NULL

for (col in possible_cols) {
  if (col %in% colnames(cluster_data)) {
    cluster_col <- col
    break
  }
}

if (is.null(cluster_col)) cluster_col <- colnames(cluster_data)[2]

## Merge data by row index
merged_data <- adni_data %>%
  mutate(RowIndex = row_number(),
         Subtype = cluster_data[[cluster_col]][RowIndex])

## CSF data detection
csf_patterns <- list(
  c("ABETA", "TAU", "PTAU"),
  c("ABETA_bl", "TAU_bl", "PTAU_bl"),
  c("AB42", "tTAU", "pTAU")
)

available_csf <- NULL
for (pattern in csf_patterns) {
  matched <- pattern[pattern %in% colnames(merged_data)]
  if (length(matched) >= 2) {
    available_csf <- matched
    break
  }
}

if (is.null(available_csf)) {
  abeta_cols <- grep("ABETA|AB42", colnames(merged_data), value = TRUE, ignore.case = TRUE)
  tau_cols <- grep("TAU", colnames(merged_data), value = TRUE, ignore.case = TRUE)
  
  available_csf <- c(
    if(length(abeta_cols) > 0) abeta_cols[1] else NULL,
    if(length(tau_cols) > 0) tau_cols[1] else NULL
  )
  available_csf <- available_csf[!is.null(available_csf)]
}

## CSF validation
if (!is.null(available_csf) && length(available_csf) >= 2) {
  csf_data <- merged_data %>% select(Subtype, all_of(available_csf))
  csf_complete <- csf_data[complete.cases(csf_data), ]
  
  if (nrow(csf_complete) >= 30) {
    csf_results <- list()
    
    for (biomarker in available_csf) {
      test_data <- csf_complete %>%
        select(Subtype, !!sym(biomarker)) %>%
        filter(!is.na(!!sym(biomarker)))
      
      ## Normality test
      shapiro_pvals <- by(test_data[[biomarker]], test_data$Subtype,
                         function(x) if(length(x) >= 3) shapiro.test(x)$p.value else NA)
      normal_dist <- all(shapiro_pvals > 0.05, na.rm = TRUE)
      
      ## Statistical test
      if (normal_dist) {
        levene_result <- leveneTest(as.formula(paste(biomarker, "~ factor(Subtype)")),
                                    data = test_data)
        equal_var <- levene_result$`Pr(>F)`[1] > 0.05
        
        if (equal_var) {
          aov_model <- aov(as.formula(paste(biomarker, "~ factor(Subtype)")), data = test_data)
          aov_result <- summary(aov_model)
          p_value <- aov_result[[1]]$`Pr(>F)`[1]
        } else {
          welch_result <- oneway.test(as.formula(paste(biomarker, "~ factor(Subtype)")),
                                      data = test_data, var.equal = FALSE)
          p_value <- welch_result$p.value
        }
      } else {
        kw_result <- kruskal.test(as.formula(paste(biomarker, "~ factor(Subtype)")), data = test_data)
        p_value <- kw_result$p.value
      }
      
      csf_results[[biomarker]] <- list(p_value = p_value, significant = p_value < 0.05)
    }
    
    ## CSF boxplot
    png("CSF_Biomarkers_by_Subtype.png", width = 4000, height = 1400, res = 300)
    
    par(mfrow = c(1, length(available_csf)), mar = c(5, 5, 4, 2))
    
    for (biomarker in names(csf_results)) {
      result <- csf_results[[biomarker]]
      test_data <- csf_complete %>%
        select(Subtype, !!sym(biomarker)) %>%
        filter(!is.na(!!sym(biomarker)))
      
      boxplot(as.formula(paste(biomarker, "~ Subtype")),
              data = test_data,
              main = sprintf("%s\np=%.4f", biomarker, result$p_value),
              xlab = "Subtype", ylab = biomarker,
              col = c("#E41A1C", "#377EB8", "#4DAF4A"))
      
      if (result$significant) {
        mtext("*", side = 3, line = -2, cex = 2.5, col = "red")
      }
    }
    
    dev.off()
  }
}

## MRI data detection
mri_features <- grep("^ST\\d+", colnames(merged_data), value = TRUE)

if (length(mri_features) > 0) {
  mri_data <- merged_data %>% select(Subtype, all_of(mri_features))
  mri_complete <- mri_data[complete.cases(mri_data), ]
  
  if (nrow(mri_complete) >= 30) {
    mri_results <- list()
    mri_p_values <- c()
    
    for (region in mri_features) {
      test_data <- mri_complete %>%
        select(Subtype, !!sym(region)) %>%
        filter(!is.na(!!sym(region)))
      
      if (nrow(test_data) < 10) next
      
      ## Statistical test
      kw_result <- kruskal.test(as.formula(paste(region, "~ factor(Subtype)")), data = test_data)
      p_value <- kw_result$p.value
      
      mri_results[[region]] <- list(p_value_raw = p_value)
      mri_p_values <- c(mri_p_values, p_value)
      names(mri_p_values)[length(mri_p_values)] <- region
    }
    
    ## Multiple testing correction
    p_adjusted_fdr <- p.adjust(mri_p_values, method = "fdr")
    
    for (region in names(mri_results)) {
      mri_results[[region]]$p_value_fdr <- p_adjusted_fdr[region]
      mri_results[[region]]$significant <- p_adjusted_fdr[region] < 0.05
    }
    
    ## MRI heatmap
    sig_regions <- names(mri_results)[sapply(mri_results, function(x) x$significant)]
    
    if (length(sig_regions) > 0) {
      mri_matrix <- mri_complete %>%
        select(Subtype, all_of(sig_regions)) %>%
        group_by(Subtype) %>%
        summarise(across(everything(), mean, na.rm = TRUE), .groups = "drop")
      
      subtype_labels <- mri_matrix$Subtype
      mri_matrix <- mri_matrix %>% select(-Subtype) %>% as.matrix()
      rownames(mri_matrix) <- paste0("Subtype ", subtype_labels)
      
      mri_matrix_scaled <- scale(t(mri_matrix))
      
      png("MRI_Significant_Regions_Heatmap.png", width = 3500, height = 2500, res = 300)
      
      pheatmap(mri_matrix_scaled,
               cluster_rows = TRUE,
               cluster_cols = FALSE,
               color = colorRampPalette(c("blue", "white", "red"))(100),
               main = sprintf("Significant MRI Regions (n=%d)", length(sig_regions)),
               fontsize = 11)
      
      dev.off()
    }
    
    ## Save MRI results
    mri_table <- data.frame(
      Region = names(mri_results),
      P_Raw = sapply(mri_results, function(x) x$p_value_raw),
      P_FDR = sapply(mri_results, function(x) x$p_value_fdr),
      Significant = sapply(mri_results, function(x) ifelse(x$significant, "Yes", "No"))
    )
    
    write.csv(mri_table, "MRI_Statistical_Results.csv", row.names = FALSE)
  }
}
