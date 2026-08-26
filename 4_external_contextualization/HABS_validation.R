library(optparse)
library(dplyr)
library(pROC)
library(ggplot2)
library(logistf)
library(PRROC)

option_list <- list(
  make_option(c("--habs_file"), type="character", default="HABS_Baseline_Integrated.csv"),
  make_option(c("--output_dir"), type="character", default="./step16_results"),
  make_option(c("--ptau_col"), type="character", default="pTau217_Primary"),
  make_option(c("--n_bootstrap"), type="integer", default=2000),
  make_option(c("--seed"), type="integer", default=42)
)
opt <- parse_args(OptionParser(option_list=option_list)); dir.create(opt$output_dir, recursive=TRUE, showWarnings=FALSE); set.seed(opt$seed)
h <- read.csv(opt$habs_file, stringsAsFactors=FALSE, check.names=FALSE)
if (!opt$ptau_col %in% names(h)) stop("pTau217 column not found")
required <- c("Age","Gender","APOE4_Positive","AD_Conversion")
if (length(setdiff(required,names(h)))) stop("Missing variables: ",paste(setdiff(required,names(h)),collapse=", "))
h <- h %>% mutate(Age=as.numeric(Age), Gender=factor(Gender), APOE4_Positive=as.numeric(APOE4_Positive), AD_Conversion=as.numeric(AD_Conversion), pTau217=as.numeric(.data[[opt$ptau_col]]))
full <- h %>% select(all_of(required)) %>% na.omit()
ptau <- h %>% select(all_of(required),pTau217) %>% na.omit()
auc_ci <- function(y,p,B=opt$n_bootstrap) { r<-roc(y,p,quiet=TRUE); z<-replicate(B,{i<-sample(seq_along(y),replace=TRUE);if(length(unique(y[i]))<2)return(NA_real_);as.numeric(auc(roc(y[i],p[i],quiet=TRUE)))}); c(AUC=as.numeric(auc(r)),CI_Lower=quantile(z,.025,na.rm=TRUE),CI_Upper=quantile(z,.975,na.rm=TRUE)) }
auprc <- function(y,p) PRROC::pr.curve(scores.class0=p[y==1],scores.class1=p[y==0])$auc.integral
fit_full <- logistf(AD_Conversion~Age+Gender+APOE4_Positive,data=full)
fit_base <- logistf(AD_Conversion~Age+Gender+APOE4_Positive,data=ptau)
fit_complete <- logistf(AD_Conversion~Age+Gender+APOE4_Positive+pTau217,data=ptau)
p_full <- predict(fit_full,newdata=full,type="response"); p_base <- predict(fit_base,newdata=ptau,type="response"); p_complete <- predict(fit_complete,newdata=ptau,type="response")
perf <- rbind(data.frame(Stratum="HABS_Full_Reduced_Clinical",N=nrow(full),Events=sum(full$AD_Conversion),AUPRC=auprc(full$AD_Conversion,p_full),t(auc_ci(full$AD_Conversion,p_full))),data.frame(Stratum="HABS_pTau217_Subset_Reduced_Clinical",N=nrow(ptau),Events=sum(ptau$AD_Conversion),AUPRC=auprc(ptau$AD_Conversion,p_base),t(auc_ci(ptau$AD_Conversion,p_base))),data.frame(Stratum="HABS_pTau217_Subset_Reduced_Clinical_plus_pTau217",N=nrow(ptau),Events=sum(ptau$AD_Conversion),AUPRC=auprc(ptau$AD_Conversion,p_complete),t(auc_ci(ptau$AD_Conversion,p_complete))))
write.csv(perf,file.path(opt$output_dir,"step16_performance_summary.csv"),row.names=FALSE)
write.csv(as.data.frame(summary(fit_full)$coefficients),file.path(opt$output_dir,"step16_full_cohort_model_coefficients.csv"))
write.csv(as.data.frame(summary(fit_complete)$coefficients),file.path(opt$output_dir,"step16_complete_model_coefficients.csv"))
delong <- roc.test(roc(ptau$AD_Conversion,p_base,quiet=TRUE),roc(ptau$AD_Conversion,p_complete,quiet=TRUE),method="delong")
events <- ptau$AD_Conversion==1; nonevents <- !events
idi <- (mean(p_complete[events])-mean(p_base[events]))-(mean(p_complete[nonevents])-mean(p_base[nonevents]))
write.csv(data.frame(Metric=c("DeLong_P","IDI"),Value=c(delong$p.value,idi)),file.path(opt$output_dir,"step16_incremental_metrics.csv"),row.names=FALSE)
write.csv(data.frame(Model_Features="Age;Gender;APOE4_Positive",Incremental_Feature="pTau217",Full_N=nrow(full),Full_Events=sum(full$AD_Conversion),Subset_N=nrow(ptau),Subset_Events=sum(ptau$AD_Conversion)),file.path(opt$output_dir,"step16_analysis_audit.csv"),row.names=FALSE)
writeLines(c("HABS cohort-specific framework evaluation",sprintf("Full cohort: N=%d, events=%d",nrow(full),sum(full$AD_Conversion)),sprintf("pTau217 subset: N=%d, events=%d",nrow(ptau),sum(ptau$AD_Conversion)),sprintf("DeLong P: %.6g",delong$p.value)),file.path(opt$output_dir,"step16_summary.txt"))
cat("Saved HABS validation outputs\n")
