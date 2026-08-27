# -*- coding: utf-8 -*-
import os as _os
ROOT = _os.path.dirname(_os.path.abspath(__file__))
"""
RA perioperative · MIMIC-IV ICU subcohort (n=668, first ICU stay)
Addresses reviewer #2 (severity adjustment) and #5 (immortal-time / landmark).

Outputs:
  severity_landmark_v3.csv  - all model estimates
  severity_landmark_v3.png  - forest of GC-high -> 30d death under escalating adjustment
"""
import sys, warnings
_pylibs = _os.environ.get("RA_GC_PYLIBS", _os.path.join(ROOT, ".pylibs"))
if _os.path.isdir(_pylibs):
    sys.path.insert(0, _pylibs)
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats
from lifelines import CoxPHFitter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

CSV = "exports/analytic_mimic_icu_v3.csv"
d = pd.read_csv(CSV)
NUM = ['age','charlson','pni','creatinine','gc_use_fulladm','gc_dose_fulladm','gc_use_48h',
       'gc_dose_48h','sofa','respiration','coagulation','liver','cardiovascular','cns','renal',
       'apsiii','oasis','sapsii','vent_24h','vaso_24h','rrt_24h','albumin_min','hemoglobin_min',
       'platelets_min','wbc_min','lactate_max','creatinine_kdigo','inhosp_death',
       'allcause_death_30d','infection']
for c in NUM:
    d[c] = pd.to_numeric(d[c], errors="coerce")
# sex / surgery_category / strata stay as strings (categoricals)
d["sex"] = d["sex"].map({"M":1,"F":0}).fillna(0)   # numeric once, used directly in formulas

# ---- time parsing ---------------------------------------------------------
for col in ["admittime","intime","outtime","deathtime"]:
    d[col] = pd.to_datetime(d[col], errors="coerce")
d["dod"] = pd.to_datetime(d["dod"], errors="coerce")

def death_dt(r):
    if pd.notna(r["deathtime"]): return r["deathtime"]
    if pd.notna(r["dod"]):      return r["dod"]
    return pd.NaT
d["death_dt"] = d.apply(death_dt, axis=1)
d["T_death"] = (d["death_dt"] - d["intime"]).dt.total_seconds()/86400.0
# 30-day mortality measured from ICU intime
d["mort30_intime"] = ((d["death_dt"].notna()) & (d["T_death"] <= 30)).astype(int)
d["T_obs"]   = np.minimum(d["T_death"].fillna(999), 30.0)   # censor at 30d
d["status30"]= d["mort30_intime"].astype(int)
# cross-check with cohort flag (discharge+30d)
print("mort30_intime (from intime) n deaths:", int(d["mort30_intime"].sum()),
      " | cohort allcause_death_30d n:", int(d["allcause_death_30d"].fillna(0).sum()))

# ---- GC exposure definitions ---------------------------------------------
d["gc_full_strata"] = np.where(d["gc_use_fulladm"]==0, "none",
                        np.where(d["gc_dose_fulladm"]>100, "high","low"))
d["gc_early_use"]   = (d["gc_use_48h"].fillna(0)>0).astype(int)
d["gc_early_strata"]= np.where(d["gc_early_use"]==0, "none",
                        np.where(d["gc_dose_48h"]>100, "high","low"))
d["gc_high"]        = (d["gc_full_strata"]=="high").astype(int)   # binary for IPTW
print("\nGC full-adm strata:", d["gc_full_strata"].value_counts().to_dict())
print("GC early-48h strata:", d["gc_early_strata"].value_counts().to_dict())

# ---- landmark at intime+48h ----------------------------------------------
d["reached48"] = ((d["T_death"].fillna(999) > 2)).astype(int)   # survived past 48h
d["T_land"]  = np.minimum((d["T_death"]-2).fillna(999), 28.0)   # follow 48h..30d
d["status_land"] = ((d["death_dt"].notna()) & (d["T_death"]>2) & (d["T_death"]<=30)).astype(int)
print("Reached 48h landmark:", int(d["reached48"].sum()),
      " | landmark deaths:", int(d["status_land"].sum()))

# ---------------------------------------------------------------------------
rows = []
def logit(formula, label, data=None, base=None):
    df = (data if data is not None else d).copy()
    m = smf.logit(formula, data=df).fit(disp=0, maxiter=300)
    out = {"model":label, "n":int(m.nobs), "events":None}
    for t in m.params.index:
        if t in ("Intercept",): continue
        b=m.params[t]; s=m.bse[t]
        if s>0 and np.isfinite(s):
            OR=np.exp(b); lo=np.exp(b-1.96*s); hi=np.exp(b+1.96*s)
            p=2*(1-stats.norm.cdf(abs(b/s)))
            out[t]=f"{OR:.3f}[{lo:.3f}-{hi:.3f}] p={p:.1e}"
    # capture GC-high / GC-low contrast terms (formula names like C(gc_full_strata,...)[T.high])
    for t in m.params.index:
        tl = str(t).lower()
        if ("gc_full_strata" in tl or "gc_early_strata" in tl) and ("high" in tl or "low" in tl):
            b=m.params[t]; s=m.bse[t]
            if s>0 and np.isfinite(s):
                OR=np.exp(b); lo=np.exp(b-1.96*s); hi=np.exp(b+1.96*s)
                p=2*(1-stats.norm.cdf(abs(b/s)))
                term_label = "gc_high" if "high" in tl else "gc_low"
                rows.append(dict(model=label, term=term_label, OR=round(OR,3),
                                 lo=round(lo,3), hi=round(hi,3), p=round(p,4),
                                 n=int(m.nobs)))
    return m, out

# DEMOGRAPHIC BASE (age+sex+charlson) -- replicates prior report on ICU subset
m1,o1 = logit("mort30_intime ~ C(gc_full_strata, Treatment('none')) + age + sex + charlson",
              "Base(age+sex+charlson)")
# + SOFA
m2,o2 = logit("mort30_intime ~ C(gc_full_strata, Treatment('none')) + age + sex + charlson + sofa",
              "Base+SOFA")
# + full severity (SOFA+APSIII+vent+vaso+creatinine)
m3,o3 = logit("mort30_intime ~ C(gc_full_strata, Treatment('none')) + age + sex + charlson + sofa + apsiii + vent_24h + vaso_24h + creatinine",
              "Base+FullSeverity")
# + PNI (attenuation comparison)
m4,o4 = logit("mort30_intime ~ C(gc_full_strata, Treatment('none')) + age + sex + charlson + sofa + pni",
              "Base+SOFA+PNI")

# INFECTION under severity
logit("infection ~ C(gc_full_strata, Treatment('none')) + age + sex + charlson + sofa + apsiii + vent_24h + vaso_24h + creatinine",
      "Infection+FullSeverity")

# ---- IPTW (age+sex+charlson+sofa) marginal OR for GC-high ----------------
def iptw(data, exposure, outcome, confounders):
    df = data.dropna(subset=[exposure, outcome]+confounders).copy()
    Xc = df[confounders].copy()
    Xc = sm.add_constant(Xc, has_constant="add")
    ps = sm.Logit(df[exposure].astype(int), Xc).fit(disp=0)
    pr = ps.predict(Xc)
    pr = np.clip(pr, 0.01, 0.99)
    df["w"] = np.where(df[exposure].astype(int)==1, 1/pr, 1/(1-pr))
    Xw = sm.add_constant(pd.concat([df[[exposure]].astype(int).rename(columns={exposure:"E"}),
                                    Xc.drop(columns="const")], axis=1), has_constant="add")
    mw = sm.GLM(df[outcome].astype(int), Xw, family=sm.families.Binomial(),
                freq_weights=df["w"]).fit()
    b=mw.params["E"]; s=mw.bse["E"]
    OR=np.exp(b); lo=np.exp(b-1.96*s); hi=np.exp(b+1.96*s); p=2*(1-stats.norm.cdf(abs(b/s)))
    rows.append(dict(model="IPTW(sev)", term="gc_high", OR=round(OR,3), lo=round(lo,3),
                     hi=round(hi,3), p=round(p,4), n=int(df["w"].sum())))
iptw(d, "gc_high", "mort30_intime", ["age","sex","charlson","sofa"])

# ---- Absolute risk difference (from FullSeverity model m3) ---------------
def ard(model, exposure_cat, data):
    df = data.dropna(subset=["age","sex","charlson","sofa","apsiii","vent_24h","vaso_24h","creatinine","gc_full_strata","mort30_intime"]).copy()
    base = df.copy()
    g_none = base.copy(); g_none["gc_full_strata"]="none"
    g_high = base.copy(); g_high["gc_full_strata"]="high"
    p_none = model.predict(g_none).mean()
    p_high = model.predict(g_high).mean()
    return p_none, p_high, p_high-p_none
p_none, p_high, ard_val = ard(m3, "high", d)
print(f"\nARD GC-high vs none (30d death, severity-adj): {ard_val*100:.1f} pp  (none={p_none*100:.1f}% high={p_high*100:.1f}%)")

# ---- LANDMARK analysis (early GC, immortal-time-robust) ------------------
lm = d[d["reached48"]==1].copy()
# logistic on landmark
ml,ol = logit("status_land ~ C(gc_early_strata, Treatment('none')) + age + sex + charlson + sofa + apsiii + vent_24h + vaso_24h + creatinine",
              "Landmark(earlyGC)+Severity", data=lm)
# Cox PH full subcohort (time-to-death from intime, early GC time-fixed)
cf = d.dropna(subset=["age","sex","charlson","sofa","apsiii","vent_24h","vaso_24h","creatinine","gc_early_strata","T_obs","status30"]).copy()
cf["gc_early_high"]=(cf["gc_early_strata"]=="high").astype(int)
cf["gc_early_low"] =(cf["gc_early_strata"]=="low").astype(int)
cph = CoxPHFitter()
cph.fit(cf[["T_obs","status30","gc_early_high","gc_early_low","age","sex","charlson","sofa","apsiii","vent_24h","vaso_24h","creatinine"]],
        duration_col="T_obs", event_col="status30")
for term in ["gc_early_high","gc_early_low"]:
    hr=cph.hazard_ratios_[term]; se=cph.summary.loc[term,"se(coef)"]; ci=cph.summary.loc[term,"coef lower 95%"]
    # lifelines summary columns
    lo=cph.summary.loc[term,"exp(coef) lower 95%"]; hi=cph.summary.loc[term,"exp(coef) upper 95%"]; p=cph.summary.loc[term,"p"]
    rows.append(dict(model="Cox(full,earlyGC)+Severity", term=term, OR=round(hr,3),
                     lo=round(lo,3), hi=round(hi,3), p=round(p,4), n=len(cf)))

# Cox on landmark subset
cl = lm.dropna(subset=["age","sex","charlson","sofa","apsiii","vent_24h","vaso_24h","creatinine","gc_early_strata","T_land","status_land"]).copy()
cl["gc_early_high"]=(cl["gc_early_strata"]=="high").astype(int)
cl["gc_early_low"] =(cl["gc_early_strata"]=="low").astype(int)
cphl = CoxPHFitter()
cphl.fit(cl[["T_land","status_land","gc_early_high","gc_early_low","age","sex","charlson","sofa","apsiii","vent_24h","vaso_24h","creatinine"]],
         duration_col="T_land", event_col="status_land")
for term in ["gc_early_high","gc_early_low"]:
    hr=cphl.hazard_ratios_[term]
    lo=cphl.summary.loc[term,"exp(coef) lower 95%"]; hi=cphl.summary.loc[term,"exp(coef) upper 95%"]; p=cphl.summary.loc[term,"p"]
    rows.append(dict(model="Cox(landmark,earlyGC)+Severity", term=term, OR=round(hr,3),
                     lo=round(lo,3), hi=round(hi,3), p=round(p,4), n=len(cl)))

# ---- attenuation of GC-high effect by severity (log-OR) -------------------
def logor(row_model):
    for r in rows:
        if r["model"]==row_model and r["term"]=="gc_high":
            return np.log(r["OR"])
    return None
b0 = logor("Base(age+sex+charlson)")
bS = logor("Base+SOFA")
bF = logor("Base+FullSeverity")
bP = logor("Base+SOFA+PNI")
print("\n=== Attenuation of GC-high log-OR ===")
print(f"  Base:            {b0:.3f}")
print(f"  +SOFA:           {bS:.3f}  attenuate {(b0-bS)/b0*100:.1f}%")
print(f"  +FullSeverity:   {bF:.3f}  attenuate {(b0-bF)/b0*100:.1f}%")
print(f"  +SOFA+PNI:       {bP:.3f}  attenuate {(b0-bP)/b0*100:.1f}%")

res = pd.DataFrame(rows)
res.to_csv("severity_landmark_v3.csv", index=False)
print("\n=== MODEL SUMMARY (GC-high / early-GC terms) ===")
print(res.to_string(index=False))

# ---- forest plot ----------------------------------------------------------
plot_rows = [r for r in rows if (r["term"] in ("gc_full_strata","gc_early_strata","gc_early_use") or "Cox" in r["model"] or "IPTW" in r["model"])]
labels = {
 "Base(age+sex+charlson)":"Base (age+sex+Charlson)",
 "Base+SOFA":"+ SOFA",
 "Base+FullSeverity":"+ Full severity",
 "Base+SOFA+PNI":"+ SOFA + PNI",
 "IPTW(sev)":"IPTW (severity)",
 "Landmark(earlyGC)+Severity":"Landmark (early GC)+sev",
 "Cox(full,earlyGC)+Severity":"Cox full (early GC)+sev",
 "Cox(landmark,earlyGC)+Severity":"Cox landmark (early GC)+sev",
}
fig, ax = plt.subplots(figsize=(8,5))
ys=[]; xs=[]; los=[]; his=[]
order=["Base(age+sex+charlson)","Base+SOFA","Base+FullSeverity","Base+SOFA+PNI","IPTW(sev)","Landmark(earlyGC)+Severity","Cox(full,earlyGC)+Severity","Cox(landmark,earlyGC)+Severity"]
shown=[]
for i,mname in enumerate(order):
    for r in rows:
        if r["model"]==mname and (r["term"]=="gc_full_strata" or "Cox" in mname or "IPTW" in mname) and "high" in str(r["term"]):
            pass
# simpler: pick high stratum / Cox high
for i,mname in enumerate(order):
    rr=[r for r in rows if r["model"]==mname and (("high" in str(r["term"])) or ("IPTW" in mname))]
    if not rr: continue
    r=rr[0]
    shown.append(mname)
    ys.append(len(shown)); xs.append(r["OR"]); los.append(r["lo"]); his.append(r["hi"])
ax.errorbar(xs, ys, xerr=[np.array(xs)-np.array(los), np.array(his)-np.array(xs)], fmt="o", color="#1f4e79", ms=7, lw=2)
ax.axvline(1, color="grey", ls="--")
ax.set_yticks(ys); ax.set_yticklabels([labels[s] for s in shown])
ax.set_xscale("log"); ax.set_xlabel("Odds/Hazard Ratio (high-dose GC vs none), log scale")
ax.set_title("GC-high 30-day mortality: effect under escalating adjustment (MIMIC ICU subcohort)")
plt.tight_layout(); plt.savefig("severity_landmark_v3.png", dpi=150); plt.close()
print("\nSaved severity_landmark_v3.csv / .png")
