# -*- coding: utf-8 -*-
import os as _os
ROOT = _os.path.dirname(_os.path.abspath(__file__))
"""
RA perioperative · V5 additions  (faithful extension of V4 MSM)
  (1) Daily-SOFA Marginal Structural Cox (time-dependent confounding upgrade)
      - Model B : baseline-severity MSM  == V4 (reproduces HR~1.07)
      - Model C : MSM with daily SOFA in denominator (NEW, time-dependent confounder)
  (2) Negative-control triangulation
      - Positive : GC(any) -> infection  (expect significant)
      - Negative exposure : ondansetron -> 30d death (expect null)
      - Negative outcome  : GC(any) -> fall (expect null)
Inputs: analytic_mimic_icu_v3.csv, gc_timeline_mimic_icu.csv,
        msm_daily_sofa.csv, msm_negctrl_perstay.csv
Outputs: msm_dailysofa_v5.csv/.png, negctrl_v5.csv/.png
"""
import sys, warnings
_pylibs = _os.environ.get("RA_GC_PYLIBS", _os.path.join(ROOT, ".pylibs"))
if _os.path.isdir(_pylibs):
    sys.path.insert(0, _pylibs)
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from lifelines import CoxPHFitter
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

FEAS = _os.environ.get("RA_GC_DATA", _os.path.join(ROOT, "data"))  # extracted CSVs dir (NOT in repo; see README)
EXP  = f"{FEAS}/exports"
THRESH = 100.0

# ---------------- load cohort + times ----------------
d = pd.read_csv(f"{EXP}/analytic_mimic_icu_v3.csv")
NUM = ['age','charlson','pni','creatinine','gc_use_fulladm','gc_dose_fulladm',
       'sofa','apsiii','vent_24h','vaso_24h','rrt_24h','allcause_death_30d','infection']
for c in NUM:
    if c in d: d[c] = pd.to_numeric(d[c], errors="coerce")
d["sex"] = d["sex"].astype(str).str.upper().map({"M":1,"F":0}).fillna(0)
for col in ["admittime","intime","outtime","deathtime"]:
    d[col] = pd.to_datetime(d[col], errors="coerce")
d["dod"] = pd.to_datetime(d["dod"], errors="coerce")
d["death_dt"] = d.apply(lambda r: r["deathtime"] if pd.notna(r["deathtime"]) else (r["dod"] if pd.notna(r["dod"]) else pd.NaT), axis=1)
d["T_death"] = (d["death_dt"] - d["intime"]).dt.total_seconds()/86400.0
d["mort30_intime"] = ((d["death_dt"].notna()) & (d["T_death"] <= 30)).astype(int)
d["T_obs"] = np.minimum(d["T_death"].fillna(999), 30.0)
d["status30"] = d["mort30_intime"].astype(int)

# ---------------- GC timeline ----------------
tl = pd.read_csv(f"{EXP}/gc_timeline_mimic_icu.csv")
tl["start_hour"] = pd.to_numeric(tl["start_hour"], errors="coerce")
tl["dose_predeq_mg"] = pd.to_numeric(tl["dose_predeq_mg"], errors="coerce").fillna(0)
tl["preicu_flag"] = pd.to_numeric(tl["preicu_flag"], errors="coerce").fillna(0).astype(int)

# ---------------- daily SOFA dict (LOCF) ----------------
sof = pd.read_csv(f"{EXP}/msm_daily_sofa.csv")
sofd = {}
for sid, g in sof.groupby("stay_id"):
    sofd[int(sid)] = dict(zip(g["day_idx"].astype(int), g["daily_sofa"].astype(float)))
def sofa_at(sid, hours, base_sofa):
    day = int(np.floor(hours/24.0)) if hours is not None else 0
    for dd in range(day, -1, -1):
        if sid in sofd and dd in sofd[sid]:
            return sofd[sid][dd]
    return base_sofa if not (isinstance(base_sofa,float) and np.isnan(base_sofa)) else 0.0

# ---------------- build long table (V4 breakpoints + sofa_d) ----------------
CONF = ["age","sex","charlson","sofa","apsiii","vent_24h","vaso_24h","creatinine"]
rows_long = []
for _, r in d.iterrows():
    sid = int(r["subject_id"]); hid = int(r["hadm_id"]); st = int(r["stay_id"])
    T = max(float(r["T_obs"]), 0.01)
    conf = {c: float(r[c]) for c in CONF}
    if any(pd.isna(v) for v in conf.values()):
        continue
    sub = tl[(tl["subject_id"]==sid) & (tl["hadm_id"]==hid)]
    preicu = sub[sub["preicu_flag"]==1]["dose_predeq_mg"].sum()
    post = sub[(sub["preicu_flag"]==0) & (sub["start_hour"]>=0)].sort_values("start_hour")
    post_list = list(zip(post["start_hour"].clip(lower=0), post["dose_predeq_mg"]))
    preicu_any = 1 if preicu > 0 else 0
    times_post = sorted(set(p[0] for p in post_list))
    bps = sorted(set([0.0] + [t for t in times_post if 0 < t < T] + [T]))
    prev_any = preicu_any
    n = len(bps) - 1
    for k in range(n):
        s = bps[k]; e = bps[k+1]
        if e <= s: e = s + 0.001
        cum = preicu + sum(dz for (tz,dz) in post_list if tz <= e)
        anyg = 1 if cum > 0 else 0
        cum100 = cum/THRESH
        sofa_d = sofa_at(st, s, r["sofa"])
        is_last = (k == n-1)
        ev = 1 if (is_last and int(r["status30"])==1) else 0
        rows_long.append(dict(stay_id=st, start=s, stop=e, event=ev,
                              anygc=anyg, prev_any=prev_any, cumdose100=cum100,
                              time=s, preicu_any=preicu_any, sofa_d=sofa_d, **conf))
        prev_any = anyg
ld = pd.DataFrame(rows_long)
print("Long intervals:", len(ld), "| stays:", ld["stay_id"].nunique(),
      "| deaths:", int(ld.groupby("stay_id")["event"].max().sum()))

results = []
# ===== (A) Unweighted time-dependent Cox (V4, guarantee-time biased) =====
useA = ["start","stop","event","anygc","age","sex","charlson","sofa","apsiii",
        "vent_24h","vaso_24h","creatinine","preicu_any"]
cphA = CoxPHFitter(); cphA.fit(ld[useA], duration_col="stop", event_col="event", entry_col="start")
hrA, loA, hiA, pA = (cphA.hazard_ratios_["anygc"], cphA.summary.loc["anygc","exp(coef) lower 95%"],
                     cphA.summary.loc["anygc","exp(coef) upper 95%"], cphA.summary.loc["anygc","p"])
print(f"[A] Unweighted TD Cox: HR={hrA:.3f} [{loA:.3f}-{hiA:.3f}] p={pA:.1e}")

def fit_msm(den_formula, label):
    den = smf.logit(den_formula, data=ld).fit_regularized(alpha=0.01, L1_wt=0.0, disp=0)
    num = smf.logit("anygc ~ prev_any + time", data=ld).fit_regularized(alpha=0.01, L1_wt=0.0, disp=0)
    pden = den.predict(ld); pnum = num.predict(ld)
    ld["p_den"] = np.where(ld["anygc"]==1, pden, 1-pden)
    ld["p_num"] = np.where(ld["anygc"]==1, pnum, 1-pnum)
    ld["w"] = (ld["p_num"]/ld["p_den"]).clip(1e-3, 1e3)
    sw = ld.groupby("stay_id")["w"].prod().clip(0.05, 20.0)
    ld2 = ld.merge(sw.rename("sw"), on="stay_id")
    print(f"  [{label}] stabilized weights: min {sw.min():.3f}, max {sw.max():.3f}, mean {sw.mean():.3f}")
    useB = ["start","stop","event","anygc","stay_id","sw"]
    cph = CoxPHFitter(); cph.fit(ld2[useB], duration_col="stop", event_col="event",
                                 entry_col="start", weights_col="sw", cluster_col="stay_id", robust=True)
    hr = cph.hazard_ratios_["anygc"]
    return (hr, cph.summary.loc["anygc","exp(coef) lower 95%"],
            cph.summary.loc["anygc","exp(coef) upper 95%"], cph.summary.loc["anygc","p"], sw.mean())

# ===== (B) Baseline-severity MSM  (== V4) =====
denB = "anygc ~ prev_any + time + age + sex + charlson + sofa + apsiii + vent_24h + vaso_24h + creatinine + preicu_any"
hrB, loB, hiB, pB, swBm = fit_msm(denB, "B baseline-sev")
print(f"[B] MSM baseline-sev anygc: HR={hrB:.3f} [{loB:.3f}-{hiB:.3f}] p={pB:.1e}")
# ===== (C) Daily-SOFA MSM (NEW) =====
denC = "anygc ~ prev_any + time + age + sex + charlson + sofa_d + apsiii + vent_24h + vaso_24h + creatinine + preicu_any"
hrC, loC, hiC, pC, swCm = fit_msm(denC, "C daily-sofa")
print(f"[C] MSM daily-SOFA anygc: HR={hrC:.3f} [{loC:.3f}-{hiC:.3f}] p={pC:.1e}")

# ---------------- figure: MSM method comparison ----------------
cmp = [
 ("V3 Primary (static cum-dose OR)", 3.31, 1.75, 6.25, "#b22222", "OR"),
 ("V4 Landmark HR48 (high)",         0.85, 0.20, 3.63, "#999999", "HR"),
 ("V4 TD-Cox unweighted",           hrA,  loA,  hiA,  "#d98c00", "HR"),
 ("V4 MSM baseline-sev",             hrB,  loB,  hiB,  "#2c7fb8", "HR"),
 ("V5 MSM daily-SOFA  <- new",       hrC,  loC,  hiC,  "#1a9641", "HR"),
]
fig, ax = plt.subplots(figsize=(9,4.8))
for i,(lab,x,lo,hi,col,_) in enumerate(cmp):
    ax.plot([lo,hi],[i,i], color=col, lw=2)
    ax.plot([x],[i], "o", color=col, ms=9)
    ax.text(hi*1.03, i, f"{x:.2f}", va="center", fontsize=8, color=col)
ax.axvline(1, color="grey", ls="--")
ax.set_yticks(range(len(cmp))); ax.set_yticklabels([c[0] for c in cmp], fontsize=8)
ax.set_xscale("log"); ax.set_xlabel("Odds / Hazard Ratio (GC vs none), log scale")
ax.set_title("Perioperative GC -> 30-day mortality: immortal-time correction\nV4 baseline-severity MSM -> V5 daily-SOFA MSM")
ax.set_xlim(0.15, 8)
plt.tight_layout(); plt.savefig(f"{FEAS}/msm_dailysofa_v5.png", dpi=150); plt.close()

# ---------------- NEGATIVE CONTROL ----------------
nc = pd.read_csv(f"{EXP}/msm_negctrl_perstay.csv")
m2 = d.merge(nc, on="stay_id", how="inner")
m2 = m2.dropna(subset=["sofa","apsiii","age","sex","charlson","allcause_death_30d","infection"])
m2["gc_any"] = (m2["gc_use_fulladm"]==1).astype(int)
m2["ond"] = m2["ondansetron"].astype(int)
m2["fall"] = m2["fall_dx"].astype(int)
adj = ["sofa","apsiii","age","sex","charlson"]
def logit_exp(expcol, ytxt, xcols):
    y = m2[ytxt].astype(int)
    X = sm.add_constant(m2[[*xcols, expcol]].astype(float))
    r = sm.Logit(y, X).fit(disp=0)
    b = r.params[expcol]; ci = r.conf_int().loc[expcol]
    return float(np.exp(b)), float(np.exp(ci[0])), float(np.exp(ci[1])), float(r.pvalues[expcol])
pos  = logit_exp("gc_any", "infection", adj)          # positive control
negE = logit_exp("ond",     "allcause_death_30d", adj) # negative control exposure
negO = logit_exp("gc_any",  "fall", adj)               # negative control outcome
print("Positive GC->infection:", pos)
print("NegativeExp ondansetron->death:", negE)
print("NegativeOut GC->fall:", negO)

ncdf = pd.DataFrame([
 ("Positive control: GC(any) -> infection",        *pos,  "Expected significant"),
 ("Negative ctrl exposure: ondansetron -> death",  *negE, "Expected null"),
 ("Negative ctrl outcome: GC(any) -> fall",        *negO, "Expected null"),
], columns=["contrast","OR","lo","hi","p","expectation"])
ncdf.to_csv(f"{FEAS}/negctrl_v5.csv", index=False)

fig, ax = plt.subplots(figsize=(8.4,3.6))
for i,(_,x,lo,hi,_,_) in enumerate(ncdf.itertuples(index=False)):
    col = "#1a9641" if "Positive" in ncdf.iloc[i]["contrast"] else "#888888"
    ax.plot([lo,hi],[i,i], color=col, lw=2)
    ax.plot([x],[i], "o", color=col, ms=9)
    ax.text(hi*1.03, i, f"{x:.2f}", va="center", fontsize=8, color=col)
ax.axvline(1, color="grey", ls="--")
ax.set_yticks(range(len(ncdf))); ax.set_yticklabels(ncdf["contrast"], fontsize=8)
ax.set_xscale("log"); ax.set_xlabel("Odds Ratio (log scale)")
ax.set_title("Negative-control triangulation (MIMIC ICU subcohort, n~660)")
ax.set_xlim(0.3, 6)
plt.tight_layout(); plt.savefig(f"{FEAS}/negctrl_v5.png", dpi=150); plt.close()

# ---------------- save MSM csv ----------------
msmdf = pd.DataFrame([
 ["V4 TD-Cox unweighted (anygc)", "HR", hrA, loA, hiA, pA],
 ["V4 MSM baseline-severity (anygc)", "HR", hrB, loB, hiB, pB],
 ["V5 MSM daily-SOFA (anygc)  <- new", "HR", hrC, loC, hiC, pC],
], columns=["model","measure","HR_OR","lo","hi","p"])
msmdf.to_csv(f"{FEAS}/msm_dailysofa_v5.csv", index=False)
print("\nsaved msm_dailysofa_v5.csv/.png ; negctrl_v5.csv/.png")
print(msmdf.to_string(index=False))
