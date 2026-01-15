library(dplyr)
library(ggplot2)
library(pheatmap)
library(car)
library(effsize)
library(optparse)
library(RColorBrewer)

# ==============================================================================
# Parse Command Line Arguments
# ==============================================================================
option_list <- list(
  make_option(c("--integrated_file"), type = "character", 
              default = "Cohort_A_Integrated.csv",
              help = "Path to integrated cohort CSV [default: %default]"),
  make_option(c("--cluster_file"), type = "character", 
              default = "Final_Consensus_Clusters_K3.csv",
              help = "Path to cluster results CSV [default: %default]"),
  make_option(c("--output_dir"), type = "character", 
              default = "./results",
              help = "Output directory [default: %default]"),
  make_option(c("--fdr_threshold"), type = "numeric", default = 0.05,
              help = "FDR significance threshold (Methods 2.4: q < 0.05) [default: %default]"),
  make_option(c("--smd_threshold"), type = "numeric", default = 0.5,
              help = "SMD clinical meaningfulness threshold (Methods 2.4: |SMD| > 0.5) [default: %default]")
)

opt_parser <- OptionParser(option_list = option_list)
opt <- parse_args(opt_parser)

# Create output directory
dir.create(opt$output_dir, showWarnings = FALSE, recursive = TRUE)

#' Calculate Eta-squared effect size

calculate_eta_squared <- function(values, groups) {
  # Remove NA values
  valid_idx <- !is.na(values) & !is.na(groups)
  values <- values[valid_idx]
  groups <- factor(groups[valid_idx])
  
  if (length(unique(groups)) < 2 || length(values) < 10) {
    return(list(eta_sq = NA, interpretation = NA))
  }
  
  # Calculate SS_between and SS_total
  grand_mean <- mean(values)
  group_means <- tapply(values, groups, mean)
  group_ns <- tapply(values, groups, length)
  
  ss_between <- sum(group_ns * (group_means - grand_mean)^2)
  ss_total <- sum((values - grand_mean)^2)
  
  eta_sq <- ss_between / ss_total
  
  # Cohen's criteria (Methods 2.8)
  if (eta_sq >= 0.14) {
    interpretation <- "Large"
  } else if (eta_sq >= 0.06) {
    interpretation <- "Medium"
  } else if (eta_sq >= 0.01) {
    interpretation <- "Small"
  } else {
    interpretation <- "Negligible"
  }
  
  return(list(eta_sq = eta_sq, interpretation = interpretation))
}

#' Calculate pairwise Cohen's d (SMD)
#' Methods 2.4: "|SMD| > 0.5 for clinical meaningfulness"
calculate_pairwise_smd <- function(values, groups) {
  groups <- factor(groups)
  levels_g <- levels(groups)
  n_groups <- length(levels_g)
  
  if (n_groups < 2) return(NULL)
  
  results <- data.frame()
  
  for (i in 1:(n_groups - 1)) {
    for (j in (i + 1):n_groups) {
      g1 <- values[groups == levels_g[i]]
      g2 <- values[groups == levels_g[j]]
      
      g1 <- g1[!is.na(g1)]
      g2 <- g2[!is.na(g2)]
      
      if (length(g1) >= 3 && length(g2) >= 3) {
        d <- cohen.d(g1, g2)
        
        results <- rbind(results, data.frame(
          Comparison = sprintf("%s vs %s", levels_g[i], levels_g[j]),
          Cohen_d = d$estimate,
          CI_Lower = d$conf.int[1],
          CI_Upper = d$conf.int[2],
          Clinically_Meaningful = abs(d$estimate) > opt$smd_threshold
        ))
      }
    }
  }
  
  return(results)
}

# ==============================================================================
# Part 1: Load Data
# ==============================================================================
cat("[1/4] Loading data...\n")

adni_data <- read.csv(opt$integrated_file, stringsAsFactors = FALSE)
cat(sprintf("  Loaded integrated data: %d samples\n", nrow(adni_data)))

# Load cluster data
if (!file.exists(opt$cluster_file)) {
  stop(sprintf("Cluster file not found: %s", opt$cluster_file))
}

cluster_data <- read.csv(opt$cluster_file, stringsAsFactors = FALSE)
cat(sprintf("  Loaded cluster data: %d samples\n", nrow(cluster_data)))

# Detect cluster label column
possible_cols <- c("Cluster", "Consensus_Cluster_K3", "Consensus_Cluster", "Subtype")
cluster_col <- NULL

for (col in possible_cols) {
  if (col %in% colnames(cluster_data)) {
    cluster_col <- col
    break
  }
}

if (is.null(cluster_col)) {
  cluster_col <- colnames(cluster_data)[2]
}

cat(sprintf("  Cluster column: %s\n", cluster_col))

# Merge data by row index
merged_data <- adni_data %>%
  mutate(RowIndex = row_number(),
         Subtype = cluster_data[[cluster_col]][RowIndex])

n_subtypes <- length(unique(merged_data$Subtype))
cat(sprintf("  Number of subtypes: %d\n\n", n_subtypes))

# ==============================================================================
# Part 2: CSF Biomarker Validation
# ==============================================================================
cat("[2/4] CSF Biomarker Validation...\n")

# CSF data detection
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

csf_results <- data.frame()

if (!is.null(available_csf) && length(available_csf) >= 1) {
  csf_data <- merged_data %>% select(Subtype, all_of(available_csf))
  csf_complete <- csf_data[complete.cases(csf_data), ]
  
  cat(sprintf("  CSF biomarkers: %s\n", paste(available_csf, collapse = ", ")))
  cat(sprintf("  Complete cases: %d\n", nrow(csf_complete)))
  
  if (nrow(csf_complete) >= 30) {
    p_values_raw <- c()
    
    for (biomarker in available_csf) {
      test_data <- csf_complete %>%
        select(Subtype, !!sym(biomarker)) %>%
        filter(!is.na(!!sym(biomarker)))
      
      # Normality test
      shapiro_pvals <- by(test_data[[biomarker]], test_data$Subtype,
                         function(x) if(length(x) >= 3) shapiro.test(x)$p.value else NA)
      normal_dist <- all(shapiro_pvals > 0.05, na.rm = TRUE)
      
      # Statistical test
      if (normal_dist && length(unique(test_data$Subtype)) >= 2) {
        levene_result <- leveneTest(as.formula(paste(biomarker, "~ factor(Subtype)")),
                                    data = test_data)
        equal_var <- levene_result$`Pr(>F)`[1] > 0.05
        
        if (equal_var) {
          aov_model <- aov(as.formula(paste(biomarker, "~ factor(Subtype)")), data = test_data)
          aov_result <- summary(aov_model)
          p_value <- aov_result[[1]]$`Pr(>F)`[1]
          test_used <- "ANOVA"
        } else {
          welch_result <- oneway.test(as.formula(paste(biomarker, "~ factor(Subtype)")),
                                      data = test_data, var.equal = FALSE)
          p_value <- welch_result$p.value
          test_used <- "Welch"
        }
      } else {
        kw_result <- kruskal.test(as.formula(paste(biomarker, "~ factor(Subtype)")), data = test_data)
        p_value <- kw_result$p.value
        test_used <- "Kruskal-Wallis"
      }
      
      # Eta-squared (Methods 2.8)
      eta_result <- calculate_eta_squared(test_data[[biomarker]], test_data$Subtype)
      
      # Pairwise SMD (Methods 2.4)
      smd_results <- calculate_pairwise_smd(test_data[[biomarker]], test_data$Subtype)
      max_smd <- if (!is.null(smd_results) && nrow(smd_results) > 0) {
        max(abs(smd_results$Cohen_d), na.rm = TRUE)
      } else {
        NA
      }
      
      p_values_raw <- c(p_values_raw, p_value)
      names(p_values_raw)[length(p_values_raw)] <- biomarker
      
      csf_results <- rbind(csf_results, data.frame(
        Biomarker = biomarker,
        Test = test_used,
        P_Raw = p_value,
        Eta_Squared = eta_result$eta_sq,
        Eta_Interpretation = eta_result$interpretation,
        Max_SMD = max_smd,
        Clinically_Meaningful = ifelse(!is.na(max_smd), abs(max_smd) > opt$smd_threshold, FALSE)
      ))
    }
    
    # FDR correction (Methods 2.4)
    csf_results$P_FDR <- p.adjust(csf_results$P_Raw, method = "fdr")
    csf_results$Significant_FDR <- csf_results$P_FDR < opt$fdr_threshold
    
    cat("\n  CSF Results (Methods 2.4, 2.8):\n")
    for (i in 1:nrow(csf_results)) {
      cat(sprintf("    %s: p_raw=%.4f, p_FDR=%.4f, η²=%.3f (%s), max|SMD|=%.2f\n",
                  csf_results$Biomarker[i],
                  csf_results$P_Raw[i],
                  csf_results$P_FDR[i],
                  csf_results$Eta_Squared[i],
                  csf_results$Eta_Interpretation[i],
                  csf_results$Max_SMD[i]))
    }
    
    # CSF boxplot
    png(file.path(opt$output_dir, "CSF_Biomarkers_by_Subtype.png"), 
        width = 4000, height = 1400, res = 300)
    
    par(mfrow = c(1, length(available_csf)), mar = c(5, 5, 4, 2))
    
    for (i in 1:nrow(csf_results)) {
      biomarker <- csf_results$Biomarker[i]
      test_data <- csf_complete %>%
        select(Subtype, !!sym(biomarker)) %>%
        filter(!is.na(!!sym(biomarker)))
      
      sig_label <- ifelse(csf_results$Significant_FDR[i], "*", "")
      clinical_label <- ifelse(csf_results$Clinically_Meaningful[i], "†", "")
      
      boxplot(as.formula(paste(biomarker, "~ Subtype")),
              data = test_data,
              main = sprintf("%s\np_FDR=%.4f, η²=%.3f%s%s", 
                            biomarker, 
                            csf_results$P_FDR[i],
                            csf_results$Eta_Squared[i],
                            sig_label, clinical_label),
              xlab = "Subtype", ylab = biomarker,
              col = c("#E41A1C", "#377EB8", "#4DAF4A")[1:n_subtypes])
    }
    
    dev.off()
    cat("  Saved: CSF_Biomarkers_by_Subtype.png\n\n")
    
    write.csv(csf_results, 
              file.path(opt$output_dir, "CSF_Statistical_Results.csv"), 
              row.names = FALSE)
  }
} else {
  cat("  No CSF biomarkers available\n\n")
}

# ==============================================================================
# Part 3: MRI Region Validation
# ==============================================================================
cat("[3/4] MRI Region Validation...\n")

mri_features <- grep("^ST\\d+", colnames(merged_data), value = TRUE)
cat(sprintf("  MRI features detected: %d\n", length(mri_features)))

mri_results <- data.frame()

if (length(mri_features) > 0) {
  mri_data <- merged_data %>% select(Subtype, all_of(mri_features))
  mri_complete <- mri_data[complete.cases(mri_data), ]
  
  cat(sprintf("  Complete cases: %d\n", nrow(mri_complete)))
  
  if (nrow(mri_complete) >= 30) {
    p_values_raw <- c()
    
    for (region in mri_features) {
      test_data <- mri_complete %>%
        select(Subtype, !!sym(region)) %>%
        filter(!is.na(!!sym(region)))
      
      if (nrow(test_data) < 10) next
      
      # Kruskal-Wallis test
      kw_result <- kruskal.test(as.formula(paste(region, "~ factor(Subtype)")), data = test_data)
      p_value <- kw_result$p.value
      
      # Eta-squared (Methods 2.8)
      eta_result <- calculate_eta_squared(test_data[[region]], test_data$Subtype)
      
      # Pairwise SMD (Methods 2.4)
      smd_results <- calculate_pairwise_smd(test_data[[region]], test_data$Subtype)
      max_smd <- if (!is.null(smd_results) && nrow(smd_results) > 0) {
        max(abs(smd_results$Cohen_d), na.rm = TRUE)
      } else {
        NA
      }
      
      p_values_raw <- c(p_values_raw, p_value)
      names(p_values_raw)[length(p_values_raw)] <- region
      
      mri_results <- rbind(mri_results, data.frame(
        Region = region,
        P_Raw = p_value,
        Eta_Squared = eta_result$eta_sq,
        Eta_Interpretation = eta_result$interpretation,
        Max_SMD = max_smd,
        Clinically_Meaningful = ifelse(!is.na(max_smd), abs(max_smd) > opt$smd_threshold, FALSE)
      ))
    }
    
    # FDR correction (Methods 2.4)
    mri_results$P_FDR <- p.adjust(mri_results$P_Raw, method = "fdr")
    mri_results$Significant_FDR <- mri_results$P_FDR < opt$fdr_threshold
    
    # Summary
    n_sig <- sum(mri_results$Significant_FDR, na.rm = TRUE)
    n_clinical <- sum(mri_results$Clinically_Meaningful, na.rm = TRUE)
    n_large_effect <- sum(mri_results$Eta_Interpretation == "Large", na.rm = TRUE)
    
    cat(sprintf("\n  MRI Results Summary (Methods 2.4, 2.8):\n"))
    cat(sprintf("    Total regions: %d\n", nrow(mri_results)))
    cat(sprintf("    Significant (FDR < %.2f): %d\n", opt$fdr_threshold, n_sig))
    cat(sprintf("    Clinically meaningful (|SMD| > %.1f): %d\n", opt$smd_threshold, n_clinical))
    cat(sprintf("    Large effect (η² ≥ 0.14): %d\n", n_large_effect))
    
    # MRI heatmap for significant regions
    sig_regions <- mri_results %>% 
      filter(Significant_FDR) %>% 
      pull(Region)
    
    if (length(sig_regions) > 0) {
      # Limit to top 50 for visualization
      if (length(sig_regions) > 50) {
        top_regions <- mri_results %>%
          filter(Significant_FDR) %>%
          arrange(P_FDR) %>%
          head(50) %>%
          pull(Region)
        sig_regions <- top_regions
      }
      
      mri_matrix <- mri_complete %>%
        select(Subtype, all_of(sig_regions)) %>%
        group_by(Subtype) %>%
        summarise(across(everything(), mean, na.rm = TRUE), .groups = "drop")
      
      subtype_labels <- mri_matrix$Subtype
      mri_matrix <- mri_matrix %>% select(-Subtype) %>% as.matrix()
      rownames(mri_matrix) <- paste0("Subtype ", subtype_labels)
      
      mri_matrix_scaled <- scale(t(mri_matrix))
      
      png(file.path(opt$output_dir, "MRI_Significant_Regions_Heatmap.png"), 
          width = 3500, height = 2500, res = 300)
      
      pheatmap(mri_matrix_scaled,
               cluster_rows = TRUE,
               cluster_cols = FALSE,
               color = colorRampPalette(c("blue", "white", "red"))(100),
               main = sprintf("Significant MRI Regions (n=%d, FDR < %.2f)", 
                             length(sig_regions), opt$fdr_threshold),
               fontsize = 11)
      
      dev.off()
      cat("  Saved: MRI_Significant_Regions_Heatmap.png\n")
    }
    
    write.csv(mri_results, 
              file.path(opt$output_dir, "MRI_Statistical_Results.csv"), 
              row.names = FALSE)
    cat("  Saved: MRI_Statistical_Results.csv\n\n")
  }
} else {
  cat("  No MRI features available\n\n")
}

# ==============================================================================
# Part 4: Summary Report
# ==============================================================================
cat("[4/4] Generating summary report...\n\n")

summary_lines <- c(
  "================================================================================",
  "Cross-Modal Validation Report (Methods 2.4, 2.8 Aligned)",
  "================================================================================",
  "",
  sprintf("Generated: %s", Sys.time()),
  "",
  "Methods Requirements:",
  sprintf("  Methods 2.4: Benjamini-Hochberg FDR correction (q < %.2f)", opt$fdr_threshold),
  sprintf("  Methods 2.4: |SMD| > %.1f for clinical meaningfulness", opt$smd_threshold),
  "  Methods 2.8: Eta-squared effect sizes",
  "    - η² ≥ 0.01: Small effect",
  "    - η² ≥ 0.06: Medium effect",
  "    - η² ≥ 0.14: Large effect",
  "",
  "--------------------------------------------------------------------------------",
  "Data Summary",
  "--------------------------------------------------------------------------------",
  sprintf("  Total samples: %d", nrow(merged_data)),
  sprintf("  Number of subtypes: %d", n_subtypes),
  ""
)

if (nrow(csf_results) > 0) {
  summary_lines <- c(summary_lines,
    "--------------------------------------------------------------------------------",
    "CSF Biomarker Results",
    "--------------------------------------------------------------------------------",
    sprintf("  Biomarkers analyzed: %d", nrow(csf_results)),
    sprintf("  Significant (FDR < %.2f): %d", opt$fdr_threshold, 
            sum(csf_results$Significant_FDR, na.rm = TRUE)),
    sprintf("  Clinically meaningful (|SMD| > %.1f): %d", opt$smd_threshold,
            sum(csf_results$Clinically_Meaningful, na.rm = TRUE)),
    ""
  )
  
  for (i in 1:nrow(csf_results)) {
    summary_lines <- c(summary_lines,
      sprintf("  %s:", csf_results$Biomarker[i]),
      sprintf("    p_FDR = %.4f, η² = %.3f (%s), max|SMD| = %.2f",
              csf_results$P_FDR[i],
              csf_results$Eta_Squared[i],
              csf_results$Eta_Interpretation[i],
              csf_results$Max_SMD[i])
    )
  }
  summary_lines <- c(summary_lines, "")
}

if (nrow(mri_results) > 0) {
  summary_lines <- c(summary_lines,
    "--------------------------------------------------------------------------------",
    "MRI Region Results",
    "--------------------------------------------------------------------------------",
    sprintf("  Regions analyzed: %d", nrow(mri_results)),
    sprintf("  Significant (FDR < %.2f): %d", opt$fdr_threshold, 
            sum(mri_results$Significant_FDR, na.rm = TRUE)),
    sprintf("  Clinically meaningful (|SMD| > %.1f): %d", opt$smd_threshold,
            sum(mri_results$Clinically_Meaningful, na.rm = TRUE)),
    sprintf("  Large effect (η² ≥ 0.14): %d", 
            sum(mri_results$Eta_Interpretation == "Large", na.rm = TRUE)),
    sprintf("  Medium effect (η² ≥ 0.06): %d", 
            sum(mri_results$Eta_Interpretation == "Medium", na.rm = TRUE)),
    ""
  )
  
  # Top 5 significant regions
  top_regions <- mri_results %>%
    filter(Significant_FDR) %>%
    arrange(P_FDR) %>%
    head(5)
  
  if (nrow(top_regions) > 0) {
    summary_lines <- c(summary_lines, "  Top 5 significant regions:")
    for (i in 1:nrow(top_regions)) {
      summary_lines <- c(summary_lines,
        sprintf("    %d. %s: p_FDR=%.4e, η²=%.3f (%s)",
                i, top_regions$Region[i], top_regions$P_FDR[i],
                top_regions$Eta_Squared[i], top_regions$Eta_Interpretation[i])
      )
    }
    summary_lines <- c(summary_lines, "")
  }
}

summary_lines <- c(summary_lines,
  "--------------------------------------------------------------------------------",
  "Output Files",
  "--------------------------------------------------------------------------------",
  sprintf("  - %s/CSF_Statistical_Results.csv", opt$output_dir),
  sprintf("  - %s/CSF_Biomarkers_by_Subtype.png", opt$output_dir),
  sprintf("  - %s/MRI_Statistical_Results.csv", opt$output_dir),
  sprintf("  - %s/MRI_Significant_Regions_Heatmap.png", opt$output_dir),
  "",
  "================================================================================",
  "Cross-Modal Validation Complete",
  "================================================================================"
)

# Write report
report_path <- file.path(opt$output_dir, "Cross_Modal_Validation_Report.txt")
writeLines(summary_lines, report_path)

# Print summary
cat(paste(summary_lines, collapse = "\n"))
cat("\n\n")

cat("========================================================================\n")
cat("Step 15: Cross-Modal Validation Complete!\n")
cat("========================================================================\n")
cat(sprintf("Report saved: %s\n", report_path))
cat(sprintf("Output directory: %s\n", opt$output_dir))
