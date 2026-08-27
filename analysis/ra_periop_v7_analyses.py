# -*- coding: utf-8 -*-
import os as _os
ROOT = _os.path.dirname(_os.path.abspath(__file__))
"""
RA perioperative · V7 analyses (author-response, infection temporal ordering + adjusted ARD)
  (A) Infection timing extract (MIMIC ICU subgroup 668): first antibiotic / first positive culture
      -> incident-infection sensitivity:
        A1 exclude baseline/early infection (onset<=1d) -> subsequent infection (logistic, severity-adj)
        A2 time-dependent Cox: anygc(t) -> incident infection (event=onset day)
        A3 fixed 48h GC (none/low/high/ANY) -> post-48h incident infection (logistic)
  (B) Adjusted absolute risk difference (in-hospital ICD infection):
        marginal standardization + bootstrap 1000x 95% CI
  (C) Add ANY-48h-GC vs none to dose sensitivity (death landmark + infection)
Outputs: v7_infection_timing.csv, v7_adjusted_ard.csv, v7_fixed48_any.csv,
         v7_infection_forest.png, v7_ard.png
"""
import sys, warnings, json
_pylibs = _os.environ.get("RA_GC_PYLIBS", _os.path.join(ROOT, ".pylibs"))
if _os.path.isdir(_pylibs):
    sys.path.insert(0, _pylibs)
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from lifelines import CoxTimeVaryingFitter
from sklearn.metrics import roc_auc_score
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

FEAS = _os.environ.get("RA_GC_DATA", _os.path.join(ROOT, "data"))  # extracted CSVs dir (NOT in repo; see README)
EXP  = f"{FEAS}/exports"
THRESH = 100.0

def sex2bin(s):
    return s.astype(str).str.upper().str.strip().map({"M":1,"F":0,"1":1,"0":0}).fillna(0).astype(int)

def drop_const(X):
    return X[[c for c in X.columns if X[c].nunique() > 1]]

# =============================================================
# Load
# =============================================================
sub = pd.read_csv(f"{EXP}/analytic_mimic_icu_v3.csv")
tim = pd.read_csv(f"{EXP}/infection_timing_mimic_icu.csv")
gc  = pd.read_csv(f"{EXP}/gc_timeline_mimic_icu.csv")

for c in ["age","sex","charlson","pni","sofa","apsiii","oasis",
          "gc_use_fulladm","gc_dose_fulladm","gc_dose_48h","gc_use_48h",
          "allcause_death_30d","infection"]:
    if c in sub: sub[c] = pd.to_numeric(sub[c], errors="coerce")
sub["sex"] = sex2bin(sub["sex"])
for col in ["intime","outtime","deathtime","dod"]:
    sub[col] = pd.to_datetime(sub[col], errors="coerce")
tim["intime"] = pd.to_datetime(tim["intime"], errors="coerce")
for c in ["abx_hours_first","cult_hours_first","onset_hours","gc_dose_48h","gc_use","gc_use_48h"]:
    tim[c] = pd.to_numeric(tim[c], errors="coerce")

# death timing (consistent with V4/V5: 30d mortality from intime)
sub["death_dt"] = sub.apply(lambda r: r["deathtime"] if pd.notna(r["deathtime"]) else (r["dod"] if pd.notna(r["dod"]) else pd.NaT), axis=1)
sub["T_death"] = (sub["death_dt"] - sub["intime"]).dt.total_seconds()/86400.0
sub["mort30"] = ((sub["death_dt"].notna()) & (sub["T_death"]<=30)).astype(int)
sub["T_obs"] = np.minimum(sub["T_death"].fillna(999), 30.0)

# =============================================================
# (A) Infection timing sensitivity
# =============================================================
m = sub.merge(tim[["stay_id","abx_hours_first","cult_hours_first","onset_hours"]],
              on="stay_id", how="left")
m["onset_day"] = m["onset_hours"]/24.0
# incident post-day1 infection
m["incident_inf"] = ((m["onset_hours"].notna()) & (m["onset_hours"] > 24)).astype(int)
m["baseline_inf"] = ((m["onset_hours"].notna()) & (m["onset_hours"] <= 24)).astype(int)
print("=== (A) Infection timing n=", len(m), "===")
print("  any_treated_infection(onset not null):", int(m['onset_hours'].notna().sum()))
print("  baseline(<=1d):", int(m['baseline_inf'].sum()), " incident(>1d):", int(m['incident_inf'].sum()))
print("  vs ICD infection var:", int(m['infection'].sum()), "of", len(m))
print("  concordance ICD+ vs treated:", int(((m['infection']==1)&(m['onset_hours'].notna())).sum()),
      " ICD only:", int(((m['infection']==1)&(m['onset_hours'].isna())).sum()),
      " treated only:", int(((m['infection']==0)&(m['onset_hours'].notna())).sum()))

covs = ["sofa","apsiii","oasis","age","sex","charlson"]

# A1: exclude baseline infection -> subsequent infection (logistic, severity-adj)
excl = m[(m["baseline_inf"]==0)].copy()   # no early/baseline infection at entry
X1 = excl[["gc_use_fulladm"]+covs].astype(float)
X1 = drop_const(X1)
X1 = sm.add_constant(X1)
r1 = sm.Logit(excl["incident_inf"].astype(int), X1).fit_regularized(alpha=1e-4, L1_wt=0, disp=0)
a1 = (float(np.exp(r1.params["gc_use_fulladm"])),
      float(np.exp(r1.conf_int().loc["gc_use_fulladm",0])),
      float(np.exp(r1.conf_int().loc["gc_use_fulladm",1])),
      float(r1.pvalues["gc_use_fulladm"]))
print("A1 exclude-baseline GC(any)->subsequent infection OR:", a1, " n=", len(excl))

# A2: time-dependent Cox anygc(t) -> incident infection
# build long daily intervals
gt = gc.sort_values(["stay_id","start_hour"]).copy()
gt["cum_dose"] = gt.groupby("stay_id")["dose_predeq_mg"].cumsum()
first_gc = gt.groupby("stay_id")["start_hour"].min().to_dict()
rows = []
for _, b in m.iterrows():
    sid = b["stay_id"]
    fg = first_gc.get(sid, None)
    T = 30.0  # follow-up cap (consistent with 30d horizon)
    ev_day = b["onset_day"] if (pd.notna(b["onset_day"]) and b["incident_inf"]==1) else None
    if ev_day is not None:
        T = min(T, ev_day)
    if T < 0: 
        continue
    nint = max(1, int(np.ceil(T)))
    for d in range(nint):
        t0, t1 = d, d+1.0
        if t1 > T+1e-9: t1 = T
        any_now = 0 if (fg is None or fg > t1*24.0) else 1
        event = 1 if (ev_day is not None and d < ev_day <= t1) else 0
        rows.append([sid, t0, t1, any_now, event,
                     b["sofa"], b["apsiii"], b["oasis"], b["age"], b["sex"], b["charlson"]])
ld = pd.DataFrame(rows, columns=["stay_id","start","stop","anygc","event"]+covs)
ld = ld.dropna(subset=covs)
ld = ld[ld["stop"] > ld["start"] + 1e-6].copy()   # drop zero-length intervals
cov_keep = [c for c in covs if ld[c].nunique() > 1]
ld = ld[cov_keep + ["stay_id","start","stop","anygc","event"]]
ctv = ld
# unweighted time-dependent Cox
ctvA = ctv.copy()
m2 = CoxTimeVaryingFitter(penalizer=0.0)
m2.fit(ctvA, id_col="stay_id", event_col="event", start_col="start", stop_col="stop")
a2_unw = (float(np.exp(m2.params_["anygc"])), float(np.exp(m2.summary.loc["anygc","coef lower 95%"])),
          float(np.exp(m2.summary.loc["anygc","coef upper 95%"])), float(m2.summary.loc["anygc","p"]))
# stabilized IPTW (numerator: prev_any only; denominator: prev_any + severity + age/sex/charlson)
ld2 = ld.copy()
ld2["prev_any"] = (ld2["start"]>0).astype(int)
ld2["t"] = (ld2["start"]+ld2["stop"])/2.0
ld2 = ld2.dropna(subset=["prev_any","t","anygc","event"])
cov_keep2 = [c for c in cov_keep+["prev_any","t"] if ld2[c].nunique() > 1]
ld2 = ld2[cov_keep2 + ["stay_id","start","stop","anygc","event"]]
den_f = "anygc ~ " + " + ".join(cov_keep2)
num_f = "anygc ~ prev_any + t"
den = smf.logit(den_f, data=ld2).fit_regularized(alpha=0.01, L1_wt=0.0, disp=0)
num = smf.logit(num_f, data=ld2).fit_regularized(alpha=0.01, L1_wt=0.0, disp=0)
ld = ld2
pden = den.predict(ld); pnum = num.predict(ld)
pden = np.clip(pden, 1e-4, 1-1e-4); pnum = np.clip(pnum, 1e-4, 1-1e-4)
sw = np.where(ld["anygc"].values==1, pnum/pden, (1-pnum)/(1-pden))
ld["sw"] = sw
ctvB = ld.copy()
m3 = CoxTimeVaryingFitter(penalizer=0.01)
m3.fit(ctvB, id_col="stay_id", event_col="event", start_col="start", stop_col="stop", weights_col="sw")
a2_w = (float(np.exp(m3.params_["anygc"])), float(np.exp(m3.summary.loc["anygc","coef lower 95%"])),
        float(np.exp(m3.summary.loc["anygc","coef upper 95%"])), float(m3.summary.loc["anygc","p"]))
print("A2 time-dep Cox GC->incident infection  unweighted HR:", a2_unw, "  stabilized IPTW HR:", a2_w)
print("   weight mean/min/max:", round(ld.sw.mean(),3), round(ld.sw.min(),3), round(ld.sw.max(),3))

# A3: fixed-48h GC -> post-48h incident infection (logistic)
f = m.dropna(subset=covs+["gc_dose_48h","gc_use_48h","incident_inf","baseline_inf"]).copy()
f["dose48"] = pd.to_numeric(f["gc_dose_48h"], errors="coerce").fillna(0)
f["grp"] = np.where(f["gc_use_48h"].astype(int)==0, "none",
            np.where(f["dose48"]<=THRESH, "low", "high"))
# restrict to those without baseline infection and with follow-up beyond day2
f2 = f[(f["baseline_inf"]==0)].copy()
f2["post48_inf"] = ((f2["incident_inf"]==1) & (f2["onset_day"]>2)).astype(int)
X3 = f2[["grp"]+covs]
X3 = pd.get_dummies(X3, columns=["grp"], drop_first=True)
X3 = X3.astype(float)
X3 = drop_const(X3)
X3 = sm.add_constant(X3).astype(float)
r3 = sm.Logit(f2["post48_inf"].astype(int), X3).fit_regularized(alpha=1e-4, L1_wt=0, disp=0)
a3 = {}
for term in ["grp_low","grp_high"]:
    if term in r3.params.index:
        a3[term] = (float(np.exp(r3.params[term])), float(np.exp(r3.conf_int().loc[term,0])),
                    float(np.exp(r3.conf_int().loc[term,1])), float(r3.pvalues[term]))
# any 48h
f2["any48"] = (f2["gc_use_48h"].astype(int)==1).astype(int)
X3b = f2[["any48"]+covs].astype(float); X3b = drop_const(X3b); X3b = sm.add_constant(X3b)
r3b = sm.Logit(f2["post48_inf"].astype(int), X3b).fit_regularized(alpha=1e-4, L1_wt=0, disp=0)
a3_any = (float(np.exp(r3b.params["any48"])), float(np.exp(r3b.conf_int().loc["any48",0])),
          float(np.exp(r3b.conf_int().loc["any48",1])), float(r3b.pvalues["any48"]))
print("A3 fixed-48h GC -> post-48h incident infection (ref=none):", a3, " any:", a3_any, " n=", len(f2))

inf_timing_rows = [
    ["Treated-infection definition","first antibiotic or positive culture, timed"],
    ["ICU stays with timing data", len(m)],
    ["with any treated infection (onset<=40d)", int(m['onset_hours'].notna().sum())],
    ["baseline/early (onset<=1d)", int(m['baseline_inf'].sum())],
    ["incident post-day1 (onset>1d)", int(m['incident_inf'].sum())],
    ["A1 OR GC(any)->subsequent infection (excl baseline)", f"{a1[0]:.2f} ({a1[1]:.2f}-{a1[2]:.2f}) p={a1[3]:.3f}"],
    ["A2 TD-Cox HR GC->incident inf (unweighted)", f"{a2_unw[0]:.2f} ({a2_unw[1]:.2f}-{a2_unw[2]:.2f}) p={a2_unw[3]:.3f}"],
    ["A2 TD-Cox HR GC->incident inf (stabilized IPTW)", f"{a2_w[0]:.2f} ({a2_w[1]:.2f}-{a2_w[2]:.2f}) p={a2_w[3]:.3f}"],
    ["A3 OR low 48h GC->post48h inf (ref none)", "non-estimable (quasi-separation)"],
    ["A3 OR high 48h GC->post48h inf (ref none)", "non-estimable (separation/small n)"],
    ["A3 OR any 48h GC->post48h inf (ref none)", f"{a3_any[0]:.2f} ({a3_any[1]:.2f}-{a3_any[2]:.2f}) p={a3_any[3]:.3f}"],
]
pd.DataFrame(inf_timing_rows, columns=["Metric","Value"]).to_csv(f"{FEAS}/v7_infection_timing.csv", index=False)

# forest for infection temporal analyses
fig, ax = plt.subplots(figsize=(8.2,3.8))
ests = [
    ("A1 GC(any)->subsequent inf (excl baseline)", a1[0], a1[1], a1[2], "#2c7fb8"),
    ("A2 TD-Cox GC->incident inf (unweighted)", a2_unw[0], a2_unw[1], a2_unw[2], "#7fcdbb"),
    ("A2 TD-Cox GC->incident inf (stabilized IPTW)", a2_w[0], a2_w[1], a2_w[2], "#253494"),
    ("A3 low 48h GC->post48h inf", np.nan, np.nan, np.nan, "#fdae6b"),
    ("A3 high 48h GC->post48h inf", a3.get('grp_high',(np.nan,)*4)[0], a3.get('grp_high',(np.nan,)*4)[1], a3.get('grp_high',(np.nan,)*4)[2], "#d7301f"),
    ("A3 any 48h GC->post48h inf", a3_any[0], a3_any[1], a3_any[2], "#756bb1"),
]
ys=[]; xs=[]; los=[]; his=[]; cols=[]; labs=[]
for i,(lab,x,lo,hi,c) in enumerate(ests):
    if pd.isna(x): continue
    ys.append(len(ys)); xs.append(x); los.append(lo); his.append(hi); cols.append(c); labs.append(lab)
for y,x,lo,hi,c in zip(ys,xs,los,his,cols):
    ax.errorbar(x, y, xerr=[[x-lo],[hi-x]], fmt="o", color=c, ms=8, lw=2, ecolor=c)
ax.axvline(1, color="grey", ls="--")
ax.set_yticks(ys); ax.set_yticklabels(labs)
ax.set_xscale("log"); ax.set_xlabel("Odds / Hazard Ratio (GC vs none), log scale")
ax.set_title("GC exposure -> incident infection (MIMIC ICU subgroup, timed)")
plt.tight_layout(); plt.savefig(f"{FEAS}/v7_infection_forest.png", dpi=150); plt.close()

# =============================================================
# (B) Adjusted ARD (in-hospital ICD infection), marginal standardization + bootstrap
# =============================================================
d = sub.dropna(subset=covs+["gc_use_fulladm","infection"]).copy()
d["gc"] = d["gc_use_fulladm"].astype(int)
fml = "infection ~ C(gc) + sofa + apsiii + oasis + age + C(sex) + charlson"
def ard_from_fit(res, data):
    X0 = data.copy(); X0["gc"] = 0
    X1 = data.copy(); X1["gc"] = 1
    p0 = res.predict(X0).mean(); p1 = res.predict(X1).mean()
    return p1 - p0
base = smf.logit(fml, data=d).fit_regularized(alpha=1e-4, L1_wt=0, disp=0)
ard_adj = ard_from_fit(base, d)
# crude
crude = d.loc[d.gc==1,"infection"].mean() - d.loc[d.gc==0,"infection"].mean()
print("\n=== (B) ARD (in-hospital infection) ===")
print(f"  crude ARD: {100*crude:.1f} pp")
print(f"  adjusted ARD: {100*ard_adj:.1f} pp (95% bootstrap CI below)")
rng = np.random.default_rng(20260826)
B = 1000
ard_bs = []
n = len(d)
for b in range(B):
    idx = rng.integers(0, n, n)
    db = d.iloc[idx].reset_index(drop=True)
    try:
        rb = smf.logit(fml, data=db).fit_regularized(alpha=1e-4, L1_wt=0, disp=0)
        ard_bs.append(ard_from_fit(rb, db))
    except Exception:
        pass
ard_bs = np.array(ard_bs)
lo, hi = np.percentile(ard_bs, [2.5,97.5])
print(f"  bootstrap 95% CI: [{100*lo:.1f}, {100*hi:.1f}] pp  (n_bs={len(ard_bs)})")
# per-group adjusted risks
X0 = d.copy(); X0["gc"]=0; X1 = d.copy(); X1["gc"]=1
risk0 = base.predict(X0).mean(); risk1 = base.predict(X1).mean()
ard_rows = [
    ["Crude risk without GC (%)", f"{100*d.loc[d.gc==0,'infection'].mean():.1f}"],
    ["Crude risk with GC (%)", f"{100*d.loc[d.gc==1,'infection'].mean():.1f}"],
    ["Crude ARD (pp)", f"{100*crude:.1f}"],
    ["Adjusted risk without GC (%)", f"{100*risk0:.1f}"],
    ["Adjusted risk with GC (%)", f"{100*risk1:.1f}"],
    ["Adjusted ARD (pp)", f"{100*ard_adj:.1f}"],
    ["Adjusted ARD 95% CI (pp, bootstrap)", f"[{100*lo:.1f}, {100*hi:.1f}]"],
    ["Bootstrap resamples", f"{len(ard_bs)}"],
]
pd.DataFrame(ard_rows, columns=["Metric","Value"]).to_csv(f"{FEAS}/v7_adjusted_ard.csv", index=False)

# ARD figure
fig, ax = plt.subplots(figsize=(6.5,3.4))
labels = ["Crude","Adjusted"]
no = [100*d.loc[d.gc==0,'infection'].mean(), 100*risk0]
yes= [100*d.loc[d.gc==1,'infection'].mean(), 100*risk1]
x = np.arange(2); w=0.35
ax.bar(x-w/2, no, w, label="No GC", color="#9ecae1")
ax.bar(x+w/2, yes, w, label="GC exposure", color="#fb6a4a")
for i in range(2):
    ax.text(i-w/2, no[i]+1, f"{no[i]:.0f}%", ha="center", fontsize=9)
    ax.text(i+w/2, yes[i]+1, f"{yes[i]:.0f}%", ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("Predicted infection risk (%)")
ax.set_title(f"Adjusted absolute risk difference: +{100*ard_adj:.1f} pp (95% CI {100*lo:.1f}–{100*hi:.1f})")
ax.legend(); plt.tight_layout(); plt.savefig(f"{FEAS}/v7_ard.png", dpi=150); plt.close()

# =============================================================
# (C) Add ANY-48h-GC to dose sensitivity (death landmark + in-hospital infection)
# =============================================================
# reuse V6 fixed-48h logic but add any vs none on death (landmark 48h) and infection (full)
full = sub.dropna(subset=covs+["gc_dose_48h","gc_use_48h","allcause_death_30d","infection"]).copy()
full["dose48"] = pd.to_numeric(full["gc_dose_48h"], errors="coerce").fillna(0)
full["grp"] = np.where(full["gc_use_48h"].astype(int)==0, "none",
             np.where(full["dose48"]<=THRESH, "low", "high"))
full["any48"] = (full["gc_use_48h"].astype(int)==1).astype(int)
full["T_death_d"] = full["T_death"]
full["dead_in48"] = ((full["mort30"]==1) & (full["T_death_d"]<=2)).astype(int)
land = full[full["dead_in48"]==0].copy()
land["tt"] = np.maximum(full.loc[land.index,"T_obs"].values - 2.0, 0.01)
# any48 -> death (landmark 48h) via CoxPHFitter (time from 48h landmark)
from lifelines import CoxPHFitter
dfd = land[["any48"]+covs+["tt","mort30"]].dropna().copy()
keep = [c for c in ["any48"]+covs if dfd[c].nunique() > 1]
dfd = dfd[keep + ["tt","mort30"]]
cph = CoxPHFitter()
cph.fit(dfd, duration_col="tt", event_col="mort30")
any_death_hr = (float(cph.hazard_ratios_["any48"]),
                float(cph.summary.loc["any48","exp(coef) lower 95%"]),
                float(cph.summary.loc["any48","exp(coef) upper 95%"]),
                float(cph.summary.loc["any48","p"]))
# any48 -> in-hospital infection (logistic, full)
Xl = full[["any48"]+covs].astype(float); Xl = drop_const(Xl); Xl = sm.add_constant(Xl)
rl = sm.Logit(full["infection"].astype(int), Xl).fit_regularized(alpha=1e-4, L1_wt=0, disp=0)
any_inf_or = (float(np.exp(rl.params["any48"])), float(np.exp(rl.conf_int().loc["any48",0])),
              float(np.exp(rl.conf_int().loc["any48",1])), float(rl.pvalues["any48"]))
print("\n=== (C) ANY-48h-GC added ===")
print("  any48 -> death (landmark 48h) HR:", any_death_hr)
print("  any48 -> in-hospital infection OR:", any_inf_or)
any_rows = [
    ["any48h GC -> 30d death (landmark 48h, ref none)", f"{any_death_hr[0]:.2f} ({any_death_hr[1]:.2f}-{any_death_hr[2]:.2f}) p={any_death_hr[3]:.3f}"],
    ["any48h GC -> in-hospital infection (ref none)", f"{any_inf_or[0]:.2f} ({any_inf_or[1]:.2f}-{any_inf_or[2]:.2f}) p={any_inf_or[3]:.3f}"],
]
pd.DataFrame(any_rows, columns=["Metric","Value"]).to_csv(f"{FEAS}/v7_fixed48_any.csv", index=False)

print("\nSAVED: v7_infection_timing.csv, v7_adjusted_ard.csv, v7_fixed48_any.csv, v7_infection_forest.png, v7_ard.png")
