pkgs <- c("dplyr", "pROC", "ggplot2", "logistf", "PRROC")
need <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(need)) stop("Please install: ", paste(need, collapse = ", "))
library(dplyr); library(pROC); library(ggplot2); library(logistf); library(PRROC)

HABS_FILE <- file.path(Sys.getenv("AD_MULTIMODAL_DATA_ROOT"), "Analysis_Inputs/HABS/HABS_Baseline.csv")
OUT_DIR <- file.path(Sys.getenv("AD_MULTIMODAL_RESULTS_DIR"), "habs_hd")
PTAU_COL <- "pTau217_Primary"
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)
if (!file.exists(HABS_FILE)) stop("HABS file not found: ", HABS_FILE)

h <- read.csv(HABS_FILE, stringsAsFactors = FALSE, check.names = FALSE)
if (!PTAU_COL %in% names(h)) {
  alt <- c("pTau217", "ptau217", "PTAU217", "pTau217_Primary")
  hit <- alt[alt %in% names(h)]
  if (!length(hit)) stop("No plasma p-tau217 column found.")
  PTAU_COL <- hit[1]
}
req <- c("Age", "Gender", "APOE4_Positive", "AD_Conversion")
if (length(setdiff(req, names(h)))) stop("Missing HABS variable(s): ", paste(setdiff(req, names(h)), collapse = ", "))

h <- h %>% mutate(Age = as.numeric(Age), Gender = factor(Gender),
                  APOE4_Positive = as.numeric(APOE4_Positive),
                  AD_Conversion = as.integer(AD_Conversion),
                  pTau217 = as.numeric(.data[[PTAU_COL]]))

# MMSE is deliberately excluded at import/model level; no MMSE subgroup is run.
full <- h %>% select(Age, Gender, APOE4_Positive, AD_Conversion) %>% na.omit()
ptau <- h %>% select(Age, Gender, APOE4_Positive, AD_Conversion, pTau217) %>% na.omit()
if (length(unique(full$AD_Conversion)) != 2 || length(unique(ptau$AD_Conversion)) != 2)
  stop("HABS outcome must contain both converters and non-converters.")

auc_boot <- function(y, p, B = 1000, seed = 42) {
  set.seed(seed); r <- roc(y, p, quiet = TRUE, direction = "auto");
  z <- replicate(B, { i <- sample(seq_along(y), replace = TRUE); if (length(unique(y[i])) < 2) return(NA_real_); as.numeric(auc(roc(y[i], p[i], quiet = TRUE, direction = "auto"))) })
  z <- z[is.finite(z)]
  c(AUC = as.numeric(auc(r)), CI_Lower = unname(quantile(z, .025)), CI_Upper = unname(quantile(z, .975)))
}
auprc <- function(y, p) PRROC::pr.curve(scores.class0 = p[y == 1], scores.class1 = p[y == 0])$auc.integral
net_benefit <- function(y, p, t) sapply(t, function(x) { cl <- p >= x; tp <- sum(cl & y == 1); fp <- sum(cl & y == 0); tp/length(y) - fp/length(y)*x/(1-x) })
metric <- function(y, p) { r <- roc(y, p, quiet=TRUE, direction="auto"); c(AUC=as.numeric(auc(r)), AUPRC=auprc(y,p), N=length(y), Events=sum(y==1), Event_Rate=mean(y)) }

fit_full <- logistf(AD_Conversion ~ Age + Gender + APOE4_Positive, data = full)
fit_base <- logistf(AD_Conversion ~ Age + Gender + APOE4_Positive, data = ptau)
fit_complete <- logistf(AD_Conversion ~ Age + Gender + APOE4_Positive + pTau217, data = ptau)
p_full <- as.numeric(predict(fit_full, newdata = full, type = "response"))
p_base <- as.numeric(predict(fit_base, newdata = ptau, type = "response"))
p_complete <- as.numeric(predict(fit_complete, newdata = ptau, type = "response"))

perf <- rbind(data.frame(Stratum="HABS_Full_Reduced_Clinical", t(metric(full$AD_Conversion, p_full)), t(auc_boot(full$AD_Conversion,p_full))),
              data.frame(Stratum="HABS_pTau217_Subset_Reduced_Clinical", t(metric(ptau$AD_Conversion, p_base)), t(auc_boot(ptau$AD_Conversion,p_base,seed=43))),
              data.frame(Stratum="HABS_pTau217_Subset_Reduced_Clinical_plus_pTau217", t(metric(ptau$AD_Conversion, p_complete)), t(auc_boot(ptau$AD_Conversion,p_complete,seed=44))))
# metric() and auc_boot() both return AUC. Keep one authoritative column.
perf <- perf[, !names(perf) %in% "AUC.1", drop=FALSE]
write.csv(perf, file.path(OUT_DIR, "HABS_performance_no_MMSE.csv"), row.names=FALSE)
write.csv(data.frame(Outcome=ptau$AD_Conversion, Clinical=p_base, Clinical_pTau217=p_complete),
          file.path(OUT_DIR,"HABS_matched_predictions.csv"),row.names=FALSE)
write.csv(as.data.frame(summary(fit_full)$coefficients), file.path(OUT_DIR, "HABS_full_coefficients_no_MMSE.csv"))
write.csv(as.data.frame(summary(fit_complete)$coefficients), file.path(OUT_DIR, "HABS_complete_coefficients_no_MMSE.csv"))

# DeLong comparison and incremental metrics on the identical pTau217 subset.
delong <- roc.test(roc(ptau$AD_Conversion,p_base,quiet=TRUE), roc(ptau$AD_Conversion,p_complete,quiet=TRUE), method="delong")
events <- ptau$AD_Conversion == 1; nonevents <- !events
idi <- (mean(p_complete[events])-mean(p_base[events])) - (mean(p_complete[nonevents])-mean(p_base[nonevents]))
cuts <- c(.05,.10,.20); oldcat <- cut(p_base,c(-Inf,cuts,Inf),labels=FALSE); newcat <- cut(p_complete,c(-Inf,cuts,Inf),labels=FALSE)
nri_e <- mean(newcat[events]>oldcat[events])-mean(newcat[events]<oldcat[events]); nri_n <- mean(newcat[nonevents]<oldcat[nonevents])-mean(newcat[nonevents]>oldcat[nonevents])
write.csv(data.frame(Metric=c("DeLong_p","IDI","NRI_Events","NRI_NonEvents","NRI_Total"), Value=c(delong$p.value,idi,nri_e,nri_n,nri_e+nri_n)), file.path(OUT_DIR,"HABS_incremental_metrics_no_MMSE.csv"), row.names=FALSE)

thr <- seq(.01,.60,.01); dca <- data.frame(Threshold=thr, Reduced_Clinical=net_benefit(ptau$AD_Conversion,p_base,thr), Reduced_Clinical_plus_pTau217=net_benefit(ptau$AD_Conversion,p_complete,thr))
write.csv(dca, file.path(OUT_DIR,"HABS_decision_curve_no_MMSE.csv"), row.names=FALSE)
roc_df <- rbind(data.frame(FPR=1-roc(ptau$AD_Conversion,p_base,quiet=TRUE)$specificities, TPR=roc(ptau$AD_Conversion,p_base,quiet=TRUE)$sensitivities, Model="Reduced clinical"), data.frame(FPR=1-roc(ptau$AD_Conversion,p_complete,quiet=TRUE)$specificities, TPR=roc(ptau$AD_Conversion,p_complete,quiet=TRUE)$sensitivities, Model="Clinical + pTau217"))
ggsave(file.path(OUT_DIR,"HABS_ROC_no_MMSE.pdf"), ggplot(roc_df,aes(FPR,TPR,color=Model))+geom_line()+geom_abline(linetype=2)+theme_minimal(), width=6,height=5)
ggsave(file.path(OUT_DIR,"HABS_DCA_no_MMSE.pdf"), ggplot(dca,aes(Threshold))+geom_line(aes(y=Reduced_Clinical,color="Reduced clinical"))+geom_line(aes(y=Reduced_Clinical_plus_pTau217,color="Clinical + pTau217"))+theme_minimal()+ylab("Net benefit"), width=6,height=5)

write.csv(data.frame(MMSE_Read=FALSE, MMSE_Used=FALSE, Model_Features="Age;Gender;APOE4_Positive", Full_N=nrow(full), Full_Events=sum(full$AD_Conversion==1), pTau_N=nrow(ptau), pTau_Events=sum(ptau$AD_Conversion==1), PTAU_Column=PTAU_COL), file.path(OUT_DIR,"HABS_audit_no_MMSE.csv"), row.names=FALSE)
writeLines(c("HABS-HD no-MMSE framework adaptation", sprintf("Full reduced clinical: N=%d, events=%d",nrow(full),sum(full$AD_Conversion==1)), sprintf("pTau subset: N=%d, events=%d",nrow(ptau),sum(ptau$AD_Conversion==1)), sprintf("DeLong p=%.6g",delong$p.value), "MMSE was not read into any model or subgroup."), file.path(OUT_DIR,"HABS_summary_no_MMSE.txt"))
cat("HABS-HD no-MMSE analysis completed: ", OUT_DIR, "\n", sep="")
