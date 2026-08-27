# -*- coding: utf-8 -*-
import os as _os
ROOT = _os.path.dirname(_os.path.abspath(__file__))
"""
RA perioperative · MIMIC-IV ICU subcohort (n=668, first ICU stay)
Upgrade landmark -> Marginal Structural Model / time-dependent Cox.

Time-dependent treatment A(t) = anygc(t): 1 once the patient has received
ANY perioperative GC (first prescription) by time t, else 0.
  - Starts at 0 at ICU intime; turns on only when GC is actually given
    -> eliminates the "must survive to accrue cumulative dose" immortal-time
       artifact that inflated the primary static OR (3.31).
  - Remaining confounding (sicker patients get more GC) handled by MSM
    stabilized IPTW (denominator: severity + prior treatment; numerator:
    prior treatment only).

Outcomes: 30-day mortality (clock from intime).
Models:
  (A) Standard time-dependent Cox (unweighted)  -- biased (guarantee-time),
       shown only to demonstrate why weighting is needed
  (B) Marginal Structural Cox Model w/ stabilized IPTW  -- primary estimate
  (C) Sensitivity: continuous cumulative dose (per 100 mg) as time-varying cov
Outputs: msm_timedependent_v4.csv, msm_timedependent_v4.png
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

COHORT = "exports/analytic_mimic_icu_v3.csv"
TL      = "exports/gc_timeline_mimic_icu.csv"
THRESH  = 100.0

# ---------------- load cohort + times -------------------------------------
d = pd.read_csv(COHORT)
NUM = ['age','charlson','pni','creatinine','gc_use_fulladm','gc_dose_fulladm',
       'sofa','apsiii','vent_24h','vaso_24h','rrt_24h','allcause_death_30d','infection']
for c in NUM:
    if c in d: d[c] = pd.to_numeric(d[c], errors="coerce")
d["sex"] = d["sex"].map({"M":1,"F":0}).fillna(0)
for col in ["admittime","intime","outtime","deathtime"]:
    d[col] = pd.to_datetime(d[col], errors="coerce")
d["dod"] = pd.to_datetime(d["dod"], errors="coerce")
def death_dt(r):
    if pd.notna(r["deathtime"]): return r["deathtime"]
    if pd.notna(r["dod"]):      return r["dod"]
    return pd.NaT
d["death_dt"] = d.apply(death_dt, axis=1)
d["T_death"]  = (d["death_dt"] - d["intime"]).dt.total_seconds()/86400.0
d["mort30_intime"] = ((d["death_dt"].notna()) & (d["T_death"] <= 30)).astype(int)
d["T_obs"]    = np.minimum(d["T_death"].fillna(999), 30.0)
d["status30"] = d["mort30_intime"].astype(int)

# ---------------- load GC timeline -----------------------------------------
tl = pd.read_csv(TL)
tl["start_hour"] = pd.to_numeric(tl["start_hour"], errors="coerce")
tl["dose_predeq_mg"] = pd.to_numeric(tl["dose_predeq_mg"], errors="coerce").fillna(0)
tl["preicu_flag"] = pd.to_numeric(tl["preicu_flag"], errors="coerce").fillna(0).astype(int)

CONF = ["age","sex","charlson","sofa","apsiii","vent_24h","vaso_24h","creatinine"]
rows_long = []
for _, r in d.iterrows():
    sid = r["subject_id"]; hid = r["hadm_id"]; st = r["stay_id"]
    T = max(float(r["T_obs"]), 0.01)
    conf = {c: float(r[c]) for c in CONF}
    if any(pd.isna(v) for v in conf.values()):
        continue
    sub = tl[(tl["subject_id"]==sid) & (tl["hadm_id"]==hid)]
    preicu = sub[sub["preicu_flag"]==1]["dose_predeq_mg"].sum()
    post   = sub[sub["preicu_flag"]==0]
    post = post[post["start_hour"] >= 0].sort_values("start_hour")
    post_list = list(zip(post["start_hour"].clip(lower=0), post["dose_predeq_mg"]))
    preicu_any  = 1 if preicu > 0 else 0
    times_post = sorted(set(post_list[k][0] for k in range(len(post_list))))
    bps = sorted(set([0.0] + [t for t in times_post if 0 < t < T] + [T]))
    prev_any = preicu_any
    n = len(bps) - 1
    for k in range(n):
        s = bps[k]; e = bps[k+1]
        if e <= s: e = s + 0.001
        cum = preicu + sum(dz for (tz, dz) in post_list if tz <= e)
        anyg = 1 if cum > 0 else 0
        cum100 = cum / THRESH
        is_last = (k == n-1)
        ev = 1 if (is_last and int(r["status30"])==1) else 0
        rows_long.append(dict(
            stay_id=st, start=s, stop=e, event=ev,
            anygc=anyg, prev_any=prev_any, cumdose100=cum100, time=s,
            preicu_any=preicu_any, **conf))
        prev_any = anyg

ld = pd.DataFrame(rows_long)
print("Long intervals:", len(ld), "| stays analyzed:", ld["stay_id"].nunique(),
      "| deaths:", int(ld.groupby("stay_id")["event"].max().sum()))
print("Intervals anygc=1:", int((ld["anygc"]==1).sum()),
      "| mean cum dose(100mg) when on:", round(ld.loc[ld.anygc==1,"cumdose100"].mean(),2))

results = []

# ===== (A) Unweighted time-dependent Cox (biased, for contrast) ============
useA = ["start","stop","event","anygc","age","sex","charlson","sofa","apsiii",
        "vent_24h","vaso_24h","creatinine","preicu_any"]
cphA = CoxPHFitter()
cphA.fit(ld[useA], duration_col="stop", event_col="event", entry_col="start")
hrA = cphA.hazard_ratios_["anygc"]
loA = cphA.summary.loc["anygc","exp(coef) lower 95%"]
hiA = cphA.summary.loc["anygc","exp(coef) upper 95%"]
pA  = cphA.summary.loc["anygc","p"]
results.append(dict(model="TimeDepCox(unweighted)", exposure="anygc",
                    HR=round(hrA,3), lo=round(loA,3), hi=round(hiA,3), p=round(pA,4),
                    n=int(ld["stay_id"].nunique())))
print(f"[A] Unweighted time-dependent Cox anygc: HR={hrA:.3f} [{loA:.3f}-{hiA:.3f}] p={pA:.1e}  (guarantee-time biased)")

# ===== (B) Marginal Structural Cox Model w/ stabilized IPTW ===============
pe = "prev_any"
den_f = f"anygc ~ {pe} + time + age + sex + charlson + sofa + apsiii + vent_24h + vaso_24h + creatinine + preicu_any"
num_f = f"anygc ~ {pe} + time"
den = smf.logit(den_f, data=ld).fit_regularized(alpha=0.01, L1_wt=0.0, disp=0)
num = smf.logit(num_f, data=ld).fit_regularized(alpha=0.01, L1_wt=0.0, disp=0)
pden = den.predict(ld); pnum = num.predict(ld)
ld["p_den"] = np.where(ld["anygc"]==1, pden, 1-pden)
ld["p_num"] = np.where(ld["anygc"]==1, pnum, 1-pnum)
ld["w"] = (ld["p_num"]/ld["p_den"]).clip(1e-3, 1e3)
sw = ld.groupby("stay_id")["w"].prod().clip(0.05, 20.0)
ld = ld.merge(sw.rename("sw"), on="stay_id")
print(f"  [MSM] stabilized weights: min {sw.min():.3f}, max {sw.max():.3f}, mean {sw.mean():.3f}")

useB = ["start","stop","event","anygc","stay_id","sw"]
cphB = CoxPHFitter()
cphB.fit(ld[useB], duration_col="stop", event_col="event", entry_col="start",
         weights_col="sw", cluster_col="stay_id", robust=True)
hrB = cphB.hazard_ratios_["anygc"]
loB = cphB.summary.loc["anygc","exp(coef) lower 95%"]
hiB = cphB.summary.loc["anygc","exp(coef) upper 95%"]
pB  = cphB.summary.loc["anygc","p"]
results.append(dict(model="MSM(IPTW)", exposure="anygc",
                    HR=round(hrB,3), lo=round(loB,3), hi=round(hiB,3), p=round(pB,4),
                    n=int(ld["stay_id"].nunique())))
print(f"[B] MSM(IPTW) anygc -> 30d mortality: HR={hrB:.3f} [{loB:.3f}-{hiB:.3f}] p={pB:.1e}")

# ===== (C) Sensitivity: continuous cumulative dose (per 100 mg) ===========
useC = ["start","stop","event","cumdose100","age","sex","charlson","sofa","apsiii",
        "vent_24h","vaso_24h","creatinine","preicu_any"]
cphC = CoxPHFitter()
cphC.fit(ld[useC], duration_col="stop", event_col="event", entry_col="start")
hrC = cphC.hazard_ratios_["cumdose100"]
loC = cphC.summary.loc["cumdose100","exp(coef) lower 95%"]
hiC = cphC.summary.loc["cumdose100","exp(coef) upper 95%"]
pC  = cphC.summary.loc["cumdose100","p"]
results.append(dict(model="TimeDepCox(cont-dose/100mg)", exposure="cumdose100",
                    HR=round(hrC,3), lo=round(loC,3), hi=round(hiC,3), p=round(pC,4),
                    n=int(ld["stay_id"].nunique())))
print(f"[C] Continuous cumulative dose (per 100 mg) HR={hrC:.3f} [{loC:.3f}-{hiC:.3f}] p={pC:.1e}")

res = pd.DataFrame(results)
res.to_csv("msm_timedependent_v4.csv", index=False)
print("\n=== MSM / time-dependent results (30-day mortality) ===")
print(res.to_string(index=False))

# ===== comparison figure (GC exposure -> 30d death) ========================
# Prior estimates from V3: primary static cumulative-dose logistic OR (high),
# landmark early-GC Cox (high & low).
cmp = [
 ("Primary: cumulative-dose\nlogistic OR, high (V3)", 3.31, 1.75, 6.25, "#8B0000"),
 ("Landmark: early-48h GC\nCox HR, high (V3)",       0.85, 0.20, 3.63, "#2E8B57"),
 ("Landmark: early-48h GC\nCox HR, low (V3)",        1.43, 0.91, 2.25, "#3CB371"),
 ("Time-dependent Cox\n(unweighted, anyGC)",         hrA, loA, hiA, "#999999"),
 ("MSM (stabilized IPTW,\nanyGC)  [this study]",     hrB, loB, hiB, "#1f4e79"),
]
fig, ax = plt.subplots(figsize=(8.4,4.6))
ys=[]; xs=[]; los=[]; his=[]; cols=[]; labs=[]
for i,(lab,x,lo,hi,c) in enumerate(cmp):
    ys.append(i); xs.append(x); los.append(lo); his.append(hi); cols.append(c); labs.append(lab)
xs=np.array(xs); los=np.array(los); his=np.array(his); ys=np.array(ys)
for i in range(len(xs)):
    ax.errorbar([xs[i]], [ys[i]], xerr=[[xs[i]-los[i]], [his[i]-xs[i]]],
                fmt="o", color=cols[i], ms=8, lw=2, ecolor=cols[i])
ax.axvline(1, color="grey", ls="--")
ax.set_yticks(ys); ax.set_yticklabels(labs)
ax.set_xscale("log"); ax.set_xlabel("Odds / Hazard Ratio (GC vs none), log scale")
ax.set_title("Perioperative GC \u2192 30-day mortality: immortal-time correction\nlandmark \u2192 time-dependent Cox \u2192 MSM")
plt.tight_layout(); plt.savefig("msm_timedependent_v4.png", dpi=150); plt.close()
print("\nSaved msm_timedependent_v4.csv / .png")
