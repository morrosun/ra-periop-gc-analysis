library(survival)
# Working directory should be the repository root. The Fine-Gray dataset `fg_dataset.csv`
# is patient-level data and is NOT included; place it under `data/` and set the path below.
fg_csv <- file.path("data", "fg_dataset.csv")  # ADAPT: path to your extracted fg_dataset.csv
d <- read.csv(fg_csv)
d$gc_use[is.na(d$gc_use)] <- 0
for(v in c("age","sex","sofa","charlson")) d[[v]][is.na(d[[v]])] <- median(d[[v]], na.rm=TRUE)
d$age_s      <- as.numeric(scale(d$age))
d$sofa_s     <- as.numeric(scale(d$sofa))
d$charlson_s <- as.numeric(scale(d$charlson))
t <- d$fg_time; st <- d$fg_event
cat("n=", length(t), " inf=", sum(st==1), " death(comp)=", sum(st==2), " cens=", sum(st==0), "\n")

# KM of competing event (death, status==2) treating infection/censoring as censored
fitG  <- survfit(Surv(t, st==2) ~ 1)
Gsurv <- stepfun(fitG$time, c(1, fitG$surv))
Gat   <- function(x){ xc <- pmin(x, max(fitG$time)); pmin(1, pmax(1e-6, Gsurv(xc))) }

base <- data.frame(id=1:length(t), t=t, st=st, gc_use=d$gc_use, sex=d$sex,
                   age_s=d$age_s, sofa_s=d$sofa_s, charlson_s=d$charlson_s)
td   <- tmerge(base, base, id=id, event=event(t, st==1))
Gi    <- Gat(t[td$id])
Gstop <- Gat(td$tstop)
td$w  <- Gi / Gstop

fit_adj  <- coxph(Surv(tstart, tstop, event) ~ gc_use + sex + age_s + sofa_s + charlson_s, data=td, weights=w)
fit_crude<- coxph(Surv(tstart, tstop, event) ~ gc_use, data=td, weights=w)

summ <- function(fit, label){
  b  <- coef(fit)["gc_use"]; se <- sqrt(diag(vcov(fit))["gc_use"])
  sHR<- exp(b); lo <- exp(b-1.96*se); hi <- exp(b+1.96*se); p <- 2*(1-pnorm(abs(b/se)))
  cat(label, " sHR=", round(sHR,2), " (", round(lo,2), "-", round(hi,2), ") P=", round(p,3), "\n")
  data.frame(model=label, sHR=round(sHR,2), lo=round(lo,2), hi=round(hi,2), p=round(p,3), stringsAsFactors=FALSE)
}
res <- rbind(summ(fit_adj,   "ADJ (gc_use+age+sex+sofa+charlson)"),
             summ(fit_crude, "CRUDE (gc_use only)"))
write.csv(res, "fg_result_weightedcox.csv", row.names=FALSE)
cat("=== saved exports/fg_result_weightedcox.csv ===\n")
print(res)
