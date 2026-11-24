# A separate frozen clinical classifier supplies nominal classes, with no risk ordering.
library(dplyr)
library(survival)
library(survminer)
library(randomForest)
root <- Sys.getenv("AD_MULTIMODAL_DATA_ROOT")
out <- file.path(Sys.getenv("AD_MULTIMODAL_RESULTS_DIR"),"a4")
dir.create(out,recursive=TRUE,showWarnings=FALSE)
set.seed(42)
classifier <- readRDS(file.path(root,"Analysis_Inputs/Clinical_Classifier/ADNI_RF_Classifier.rds"))
base <- read.csv(file.path(root,"Analysis_Inputs/A4/A4_Baseline_Integrated.csv"),check.names=FALSE)
cdr <- read.csv(file.path(root,"Analysis_Inputs/A4/A4_CDR_Longitudinal.csv"),check.names=FALSE)
base$BID <- as.character(base$BID)
cdr$BID <- as.character(cdr$BID)
if (!"Age_Baseline" %in% names(base)) base$Age_Baseline <- base$Age
if (!"APOE4_Positive" %in% names(base)) base$APOE4_Positive <- base$APOE4_Carrier
features <- rownames(classifier$importance)
stopifnot(setequal(features,c("Age_Baseline","Gender","APOE4_Positive","MMSE_Baseline")),
          !anyDuplicated(base$BID))
followup <- cdr %>% mutate(Followup_Months=Week/4.345,
  Is_Converted=ifelse(!is.na(CDGLOBAL)&CDGLOBAL>0,1L,0L)) %>%
  arrange(BID,Week) %>% group_by(BID) %>% summarise(
    AD_Conversion=as.integer(any(Is_Converted==1L)),
    Time_to_Event_Months=ifelse(AD_Conversion==1L,min(Followup_Months[Is_Converted==1L]),max(Followup_Months)),
    Followup_Years=Time_to_Event_Months/12,N_Visits=n(),Has_Followup=max(Week)>0,.groups="drop") %>%
  filter(Followup_Years>0)
analysis <- base %>% select(-any_of("AD_Conversion")) %>%
  inner_join(followup,by="BID") %>% filter(Followup_Years>=0.5)
missing <- data.frame(Variable=features,N=nrow(analysis),
                     Missing_N=vapply(analysis[features],function(x)sum(is.na(x)),integer(1)))
write.csv(missing,file.path(out,"baseline_predictor_missingness.csv"),row.names=FALSE)
prediction_input <- analysis[features]
for (feature in features) {
  absent <- is.na(prediction_input[[feature]])
  if (any(absent)) {
    value <- if (is.numeric(prediction_input[[feature]])) median(prediction_input[[feature]],na.rm=TRUE) else
      names(sort(table(prediction_input[[feature]]),decreasing=TRUE))[1]
    prediction_input[[feature]][absent] <- value
  }
}
analysis$Classifier_Class <- factor(predict(classifier,newdata=prediction_input),levels=1:3)
analysis$Prediction_Confidence <- apply(predict(classifier,newdata=prediction_input,type="prob"),1,max)
stopifnot(nrow(analysis)==1147L,!anyDuplicated(analysis$BID))
write.csv(analysis,file.path(out,"A4_class_predictions.csv"),row.names=FALSE)
fit <- coxph(Surv(Followup_Years,AD_Conversion)~Classifier_Class+
               Age_Baseline+Gender+APOE4_Positive+MMSE_Baseline,data=analysis)
s <- summary(fit)
cox <- cbind(Term=rownames(s$coefficients),as.data.frame(s$coefficients),as.data.frame(s$conf.int))
write.csv(cox,file.path(out,"A4_adjusted_cox.csv"),row.names=FALSE)
km <- survfit(Surv(Followup_Years,AD_Conversion)~Classifier_Class,data=analysis)
lr <- survdiff(Surv(Followup_Years,AD_Conversion)~Classifier_Class,data=analysis)
write.csv(data.frame(N=nrow(analysis),Events=sum(analysis$AD_Conversion),
                    Logrank_P=pchisq(lr$chisq,2,lower.tail=FALSE)),
          file.path(out,"A4_followup_summary.csv"),row.names=FALSE)
pdf(file.path(out,"A4_KM.pdf"),width=8,height=7)
print(ggsurvplot(km,data=analysis,risk.table=TRUE,pval=TRUE,legend.title="Clinical class"))
dev.off()
