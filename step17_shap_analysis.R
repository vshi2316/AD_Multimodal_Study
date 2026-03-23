library(optparse)
library(dplyr)
library(ggplot2)
library(logistf)

if (!requireNamespace("kernelshap", quietly = TRUE)) {
  stop("Package 'kernelshap' is required for manuscript-aligned model-agnostic SHAP analysis")
}

option_list <- list(
  make_option(c("--habs_file"), type = "character", default = "HABS_Baseline_Integrated.csv",
              help = "Integrated HABS baseline file [default: %default]"),
  make_option(c("--output_dir"), type = "character", default = "./step17_results",
              help = "Output directory [default: %default]"),
  make_option(c("--ptau_col"), type = "character", default = "pTau217_Primary",
              help = "Plasma p-tau217 column [default: %default]"),
  make_option(c("--sample_size"), type = "integer", default = 500,
              help = "Maximum sample size for SHAP calculation [default: %default]"),
  make_option(c("--seed"), type = "integer", default = 42,
              help = "Random seed [default: %default]")
)
opt <- parse_args(OptionParser(option_list = option_list))
dir.create(opt$output_dir, showWarnings = FALSE, recursive = TRUE)
set.seed(opt$seed)

habs <- read.csv(opt$habs_file, stringsAsFactors = FALSE)
ptau_col <- opt$ptau_col
if (!ptau_col %in% names(habs)) {
  alt <- c("pTau217", "ptau217", "PTAU217", "pTau217_Primary")
  hit <- alt[alt %in% names(habs)]
  if (length(hit) == 0) stop("Could not find plasma p-tau217 column")
  ptau_col <- hit[1]
}

model_data <- habs %>%
  mutate(
    Age = as.numeric(Age),
    Gender = as.factor(Gender),
    APOE4_Positive = as.numeric(APOE4_Positive),
    MMSE_Baseline = as.numeric(MMSE_Baseline),
    pTau217 = as.numeric(.data[[ptau_col]]),
    AD_Conversion = as.numeric(AD_Conversion)
  ) %>%
  select(Age, Gender, APOE4_Positive, MMSE_Baseline, pTau217, AD_Conversion) %>%
  na.omit()

fit <- logistf(AD_Conversion ~ Age + Gender + APOE4_Positive + MMSE_Baseline + pTau217, data = model_data)

predict_prob <- function(object, newdata) {
  as.numeric(predict(object, newdata = newdata, type = "response"))
}

x <- model_data %>% select(Age, Gender, APOE4_Positive, MMSE_Baseline, pTau217)
if (nrow(x) > opt$sample_size) {
  idx <- sample(seq_len(nrow(x)), opt$sample_size)
  x <- x[idx, , drop = FALSE]
  y <- model_data$AD_Conversion[idx]
} else {
  y <- model_data$AD_Conversion
}

shap <- kernelshap::kernelshap(object = fit, X = x, pred_fun = predict_prob)
shap_values <- as.data.frame(shap$S)
shap_values$Observation <- seq_len(nrow(shap_values))
shap_long <- reshape(
  shap_values,
  varying = names(x),
  v.names = "SHAP",
  timevar = "Feature",
  times = names(x),
  direction = "long"
)
rownames(shap_long) <- NULL
shap_long$Feature <- factor(shap_long$Feature, levels = rev(names(sort(colMeans(abs(shap$S)), decreasing = TRUE))))
shap_long$FeatureValue <- unlist(x)[order(rep(seq_len(ncol(x)), each = nrow(x)))]

importance <- data.frame(
  Feature = names(x),
  MeanAbsSHAP = colMeans(abs(shap$S))
) %>% arrange(desc(MeanAbsSHAP))
write.csv(importance, file.path(opt$output_dir, "step17_shap_importance.csv"), row.names = FALSE)
write.csv(as.data.frame(shap$S), file.path(opt$output_dir, "step17_shap_matrix.csv"), row.names = FALSE)

p1 <- ggplot(importance, aes(x = reorder(Feature, MeanAbsSHAP), y = MeanAbsSHAP)) +
  geom_col(fill = "steelblue") + coord_flip() + theme_minimal() + xlab(NULL) + ylab("Mean |SHAP|")
ggsave(file.path(opt$output_dir, "step17_shap_importance.pdf"), p1, width = 7, height = 5)

p2 <- ggplot(shap_long, aes(x = SHAP, y = Feature, color = FeatureValue)) +
  geom_point(alpha = 0.6, size = 1.6) +
  scale_color_viridis_c() +
  theme_minimal()
ggsave(file.path(opt$output_dir, "step17_shap_summary.pdf"), p2, width = 8, height = 6)

pred <- predict_prob(fit, x)
case_index <- unique(c(which.max(pred), which.min(pred), order(abs(pred - median(pred)))[1]))
force_rows <- list()
for (i in seq_along(case_index)) {
  obs <- case_index[i]
  case_df <- data.frame(
    Feature = names(x),
    SHAP = as.numeric(shap$S[obs, ]),
    Value = as.character(unlist(x[obs, ])),
    Case = paste0("Case_", i)
  )
  force_rows[[i]] <- case_df
  p_case <- ggplot(case_df, aes(x = reorder(Feature, SHAP), y = SHAP, fill = SHAP > 0)) +
    geom_col() + coord_flip() + theme_minimal() + guides(fill = "none") +
    ggtitle(paste0("SHAP contributions for Case ", i))
  ggsave(file.path(opt$output_dir, paste0("step17_case_", i, "_forceplot.pdf")), p_case, width = 7, height = 5)
}
write.csv(bind_rows(force_rows), file.path(opt$output_dir, "step17_forceplot_data.csv"), row.names = FALSE)

writeLines(c(
  sprintf("Samples used for SHAP: %d", nrow(x)),
  sprintf("Top feature by mean |SHAP|: %s", importance$Feature[1]),
  sprintf("Second-ranked feature: %s", importance$Feature[2])
), file.path(opt$output_dir, "step17_summary.txt"))

cat("Saved SHAP outputs\n")
