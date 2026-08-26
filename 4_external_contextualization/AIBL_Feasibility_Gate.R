library(optparse)
library(dplyr)
library(lubridate)

option_list <- list(
  make_option(c("--aibl_dir"), type="character", default="./AIBL_Raw_Data/Data_extract_3.3.0"),
  make_option(c("--output_dir"), type="character", default="./Analysis_Inputs/AIBL_Feasibility_Gate")
)
opt <- parse_args(OptionParser(option_list=option_list)); dir.create(opt$output_dir, recursive=TRUE, showWarnings=FALSE)
num <- function(x) suppressWarnings(as.numeric(as.character(x)))
as_date <- function(x) { x<-as.character(x); y<-suppressWarnings(mdy(x)); b<-is.na(y); if(any(b)) y[b]<-suppressWarnings(ymd(x[b])); y }
raw <- opt$aibl_dir
needed <- file.path(raw,c("aibl_pdxconv_01-Jun-2018.csv","aibl_cdr_01-Jun-2018.csv","aibl_apoeres_01-Jun-2018.csv","aibl_ptdemog_01-Jun-2018.csv"))
if(any(!file.exists(needed))) stop("Missing AIBL raw file(s): ",paste(needed[!file.exists(needed)],collapse="; "))
pdx <- read.csv(needed[1],stringsAsFactors=FALSE) %>% transmute(RID=as.character(RID),VISCODE=toupper(trimws(VISCODE)),DXCURREN=num(DXCURREN),DXMCI=num(DXMCI),DXAD=num(DXAD),Is_AD=as.integer(DXAD==1|DXCURREN==3))
cdr <- read.csv(needed[2],stringsAsFactors=FALSE) %>% transmute(RID=as.character(RID),VISCODE=toupper(trimws(VISCODE)),EXAMDATE=as_date(EXAMDATE)) %>% filter(!is.na(EXAMDATE))
apoe <- read.csv(needed[3],stringsAsFactors=FALSE) %>% filter(toupper(trimws(VISCODE))=="BL") %>% transmute(RID=as.character(RID),APOE4_Positive=as.integer(num(APGEN1)==4|num(APGEN2)==4)) %>% distinct(RID,.keep_all=TRUE)
demog <- read.csv(needed[4],stringsAsFactors=FALSE) %>% filter(toupper(trimws(VISCODE))=="BL") %>% transmute(RID=as.character(RID),Sex=num(PTGENDER),Birth_Year=num(sub(".*(\\d{4})$","\\1",PTDOB))) %>% distinct(RID,.keep_all=TRUE)
dates <- cdr %>% group_by(RID) %>% summarise(Baseline_Date=min(EXAMDATE),Last_Followup_Date=max(EXAMDATE),.groups="drop")
first_ad <- pdx %>% filter(Is_AD==1) %>% inner_join(cdr,by=c("RID","VISCODE")) %>% group_by(RID) %>% summarise(First_AD_Date=min(EXAMDATE),.groups="drop")
baseline_mci <- pdx %>% filter(VISCODE=="BL",DXMCI==1|DXCURREN==2) %>% distinct(RID,.keep_all=TRUE)
cohort <- baseline_mci %>% transmute(RID) %>% left_join(dates,by="RID") %>% left_join(first_ad,by="RID") %>% left_join(apoe,by="RID") %>% left_join(demog,by="RID") %>% mutate(Age=year(Baseline_Date)-Birth_Year,Gender_Female=ifelse(Sex==2,1,ifelse(Sex==1,0,NA_real_)),Followup_Years=ifelse(!is.na(First_AD_Date),as.numeric(First_AD_Date-Baseline_Date)/365.25,as.numeric(Last_Followup_Date-Baseline_Date)/365.25),AD_Conversion=as.integer(!is.na(First_AD_Date))) %>% filter(!is.na(Baseline_Date),!is.na(Age),!is.na(Gender_Female),!is.na(APOE4_Positive),!is.na(Followup_Years),Followup_Years>=0)
write.csv(cohort,file.path(opt$output_dir,"02_aibl_eligible_mci_to_ad_conversion_cohort.csv"),row.names=FALSE)
write.csv(data.frame(Stage=c("Baseline MCI","Complete clinical proxy cohort"),N=c(nrow(baseline_mci),nrow(cohort)),Events=c(NA,sum(cohort$AD_Conversion))),file.path(opt$output_dir,"03_aibl_sample_flow.csv"),row.names=FALSE)
write.csv(data.frame(Feature=c("Age","Gender_Female","APOE4_Positive"),Available=TRUE),file.path(opt$output_dir,"04_aibl_feature_availability.csv"),row.names=FALSE)
cat("Saved AIBL clinical-proxy cohort\n")
