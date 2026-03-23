library(optparse)
library(dplyr)
library(survival)
library(survminer)
library(pROC)
library(ggplot2)

option_list <- list(
  make_option(c("--projection_file"), type = "character", default = "A4_projected_subtypes.csv",
              help = "Projection output from step9A [default: %default]"),
  make_option(c("--baseline_file"), type = "character", default = "A4_Baseline_Integrated.csv",
              help = "A4 baseline file with outcomes [default: %default]"),
  make_option(c("--output_dir"), type = "character", default = "./step21_results",
              help = "Output directory [default: %default]"),
  make_option(c("--id_col"), type = "character", default = "ID",
              help = "ID column [default: %default]"),
  make_option(c("--time_col"), type = "character", default = "Followup_Years",
              help = "Follow-up time column [default: %default]"),
  make_option(c("--event_col"), type = "character", default = "AD_Conversion",
              help = "Event indicator column [default: %default]")
)
opt <- parse_args(OptionParser(option_list = option_list))
dir.create(opt$output_dir, showWarnings = FALSE, recursive = TRUE)

proj <- read.csv(opt$projection_file, stringsAsFactors = FALSE)
base <- read.csv(opt$baseline_file, stringsAsFactors = FALSE)
proj[[opt$id_col]] <- as.character(proj[[opt$id_col]])
base[[opt$id_col]] <- as.character(base[[opt$id_col]])

analysis <- proj %>%
  rename(Projected_Subtype = any_of("Projected_Subtype")) %>%
  left_join(base, by = opt$id_col)

analysis$Risk_Score <- as.numeric(analysis$Projected_Subtype)
write.csv(analysis, file.path(opt$output_dir, "step21_a4_projected_subtypes.csv"), row.names = FALSE)

if (all(c(opt$time_col, opt$event_col) %in% names(analysis))) {
  analysis <- analysis %>% filter(!is.na(.data[[opt$time_col]]), !is.na(.data[[opt$event_col]]), !is.na(Projected_Subtype))
  analysis$Projected_Subtype <- factor(analysis$Projected_Subtype)
  surv_obj <- Surv(analysis[[opt$time_col]], as.numeric(analysis[[opt$event_col]]))
  km_fit <- survfit(surv_obj ~ Projected_Subtype, data = analysis)
  cox_covars <- c("Age", "Gender", "APOE4_Positive", "MMSE_Baseline")
  use_covars <- cox_covars[cox_covars %in% names(analysis)]
  cox_formula <- as.formula(paste("surv_obj ~ Projected_Subtype", if (length(use_covars) > 0) paste("+", paste(use_covars, collapse = " + ")) else ""))
  cox_fit <- coxph(cox_formula, data = analysis)
  cox_out <- data.frame(summary(cox_fit)$coefficients)
  cox_out$Term <- rownames(cox_out)
  write.csv(cox_out, file.path(opt$output_dir, "step21_a4_cox.csv"), row.names = FALSE)

  roc_obj <- roc(as.numeric(analysis[[opt$event_col]]), analysis$Risk_Score, quiet = TRUE)
  auc_out <- data.frame(AUC = as.numeric(auc(roc_obj)), CI_Lower = ci.auc(roc_obj)[1], CI_Upper = ci.auc(roc_obj)[3])
  write.csv(auc_out, file.path(opt$output_dir, "step21_a4_auc.csv"), row.names = FALSE)

  pdf(file.path(opt$output_dir, "step21_a4_kaplan_meier.pdf"), width = 7, height = 6)
  print(ggsurvplot(km_fit, data = analysis, risk.table = TRUE, pval = TRUE))
  dev.off()
}

cat("Saved Step 21 outputs\n")
