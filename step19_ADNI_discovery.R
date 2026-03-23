library(optparse)
library(dplyr)
library(survival)
library(survminer)
library(ggplot2)

option_list <- list(
  make_option(c("--subtype_file"), type = "character", default = "subtype_assignments.csv",
              help = "Subtype assignment file [default: %default]"),
  make_option(c("--clinical_file"), type = "character", default = "Clinical_data.csv",
              help = "Clinical baseline file [default: %default]"),
  make_option(c("--output_dir"), type = "character", default = "./step19_results",
              help = "Output directory [default: %default]"),
  make_option(c("--time_col"), type = "character", default = "Followup_Years",
              help = "Time-to-event column if available [default: %default]"),
  make_option(c("--event_col"), type = "character", default = "AD_Conversion",
              help = "Event indicator column [default: %default]")
)
opt <- parse_args(OptionParser(option_list = option_list))
dir.create(opt$output_dir, showWarnings = FALSE, recursive = TRUE)

subtypes <- read.csv(opt$subtype_file, stringsAsFactors = FALSE)
clinical <- read.csv(opt$clinical_file, stringsAsFactors = FALSE)
subtypes$ID <- as.character(subtypes$ID)
clinical$ID <- as.character(clinical$ID)

if (!"VAE_Subtype" %in% names(subtypes)) stop("VAE_Subtype not found in subtype file")
if (!opt$event_col %in% names(subtypes) && !opt$event_col %in% names(clinical)) stop("AD conversion column not found")

merged <- subtypes %>%
  select(ID, VAE_Subtype, any_of(opt$event_col), any_of(opt$time_col)) %>%
  left_join(clinical, by = "ID", suffix = c("", "_clinical"))

if (!opt$event_col %in% names(merged) && paste0(opt$event_col, "_clinical") %in% names(merged)) {
  merged[[opt$event_col]] <- merged[[paste0(opt$event_col, "_clinical")]]
}
if (!opt$time_col %in% names(merged) && paste0(opt$time_col, "_clinical") %in% names(merged)) {
  merged[[opt$time_col]] <- merged[[paste0(opt$time_col, "_clinical")]]
}

summary_df <- merged %>%
  mutate(Event = as.numeric(.data[[opt$event_col]])) %>%
  group_by(VAE_Subtype) %>%
  summarise(
    N = n(),
    Converters = sum(Event, na.rm = TRUE),
    Conversion_Rate = mean(Event, na.rm = TRUE),
    Age_Mean = mean(Age, na.rm = TRUE),
    Female_Percent = mean(Gender %in% c("F", "Female", 1), na.rm = TRUE) * 100,
    APOE4_Percent = mean(APOE4_Positive == 1 | APOE4_DOSAGE >= 1, na.rm = TRUE) * 100,
    Education_Mean = mean(EDUCATION, na.rm = TRUE),
    MMSE_Mean = mean(MMSE_Baseline, na.rm = TRUE)
  )
write.csv(summary_df, file.path(opt$output_dir, "step19_subtype_summary.csv"), row.names = FALSE)

model_df <- merged %>%
  mutate(
    Event = as.numeric(.data[[opt$event_col]]),
    VAE_Subtype = factor(VAE_Subtype),
    Gender = as.factor(Gender)
  )

covars <- c("Age", "Gender", "EDUCATION")
use_covars <- covars[covars %in% names(model_df)]
formula_logit <- as.formula(paste("Event ~ VAE_Subtype", if (length(use_covars) > 0) paste("+", paste(use_covars, collapse = " + ")) else ""))
logit_fit <- glm(formula_logit, data = model_df, family = binomial())
logit_out <- data.frame(summary(logit_fit)$coefficients)
logit_out$Term <- rownames(logit_out)
write.csv(logit_out, file.path(opt$output_dir, "step19_logistic_regression.csv"), row.names = FALSE)

if (opt$time_col %in% names(model_df) && sum(!is.na(model_df[[opt$time_col]])) > 20) {
  surv_df <- model_df %>% filter(!is.na(.data[[opt$time_col]]), !is.na(Event))
  surv_obj <- Surv(time = surv_df[[opt$time_col]], event = surv_df$Event)
  formula_cox <- as.formula(paste("surv_obj ~ VAE_Subtype", if (length(use_covars) > 0) paste("+", paste(use_covars, collapse = " + ")) else ""))
  cox_fit <- coxph(formula_cox, data = surv_df)
  cox_out <- data.frame(summary(cox_fit)$coefficients)
  cox_out$Term <- rownames(cox_out)
  write.csv(cox_out, file.path(opt$output_dir, "step19_cox_regression.csv"), row.names = FALSE)

  km_fit <- survfit(surv_obj ~ VAE_Subtype, data = surv_df)
  pdf(file.path(opt$output_dir, "step19_kaplan_meier.pdf"), width = 7, height = 6)
  print(ggsurvplot(km_fit, data = surv_df, risk.table = TRUE, pval = TRUE))
  dev.off()
}

writeLines(c(
  sprintf("Discovery cohort participants: %d", nrow(merged)),
  sprintf("Subtype counts: %s", paste(summary_df$VAE_Subtype, summary_df$N, sep = "=", collapse = "; "))
), file.path(opt$output_dir, "step19_summary.txt"))

cat("Saved Step 19 outputs\n")
