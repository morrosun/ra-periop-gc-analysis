# -*- coding: utf-8 -*-
import os as _os
ROOT = _os.path.dirname(_os.path.abspath(__file__))
"""
RA perioperative · V6 analyses (author-response revision)
  (1) MIMIC full cohort (2235) vs MSM ICU subgroup (668) baseline table
  (2) Weighted MSM (daily-SOFA) + FULL weight diagnostics
        - min/max/p1/p99, ESS, positivity
        - treatment-model C-statistic / pseudo-R2
        - covariate balance (raw vs IPW SMD)
  (3) Fixed-window 48h GC-dose sensitivity (none/low/high) -> 30d death (landmark 48h) + in-hospital infection
  (4) Surgery-type subgroup descriptions + absolute risk difference (in-hospital infection)
Outputs: v6_full_vs_subgroup.csv, v6_weight_diagnostics.csv, v6_fixed48_dose.csv,
         v6_surgery_ard.csv, v6_fixed48_forest.png, v6_weight_hist.png
"""
import sys, warnings
_pylibs = _os.environ.get("RA_GC_PYLIBS", _os.path.join(ROOT, ".pylibs"))
if _os.path.isdir(_pylibs):
    sys.path.insert(0, _pylibs)
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from lifelines import CoxPHFitter
from sklearn.metrics import roc_auc_score
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

FEAS = _os.environ.get("RA_GC_DATA", _os.path.join(ROOT, "data"))  # extracted CSVs dir (NOT in repo; see README)
EXP  = f"{FEAS}/exports"
THRESH = 100.0

def sex2bin(s):
    return s.astype(str).str.upper().str.strip().map({"M":1,"F":0,"1":1,"0":0}).fillna(0).astype(int)

# =============================================================
# (1) FULL COHORT vs MSM SUBGROUP baseline table
# =============================================================
full = pd.read_csv(f"{EXP}/analytic_mimiciv.csv")
sub  = pd.read_csv(f"{EXP}/analytic_mimic_icu_v3.csv")
for c in ["age","charlson","pni","gc_use","gc_dose_pred_eq_mg",
          "allcause_death_30d","infection"]:
    if c in full: full[c] = pd.to_numeric(full[c], errors="coerce")
for c in ["age","charlson","pni","gc_use_fulladm","gc_dose_fulladm",
          "allcause_death_30d","infection"]:
    if c in sub: sub[c] = pd.to_numeric(sub[c], errors="coerce")
full["sex"] = sex2bin(full["sex"])
sub["sex"]  = sex2bin(sub["sex"])
full["gc_use"] = full["gc_use"].astype(float)
sub["gc_use"]  = sub["gc_use_fulladm"].astype(float)

def row(label, get_full, get_sub):
    return [label, get_full(), get_sub()]

def pct(df, col, val=1):
    s = df[col].dropna()
    return f"{100*s.eq(val).mean():.1f}%"

def mean_sd(df, col):
    s = df[col].dropna()
    return f"{s.mean():.1f}±{s.std():.1f}"

def n_pct_cat(df, col):
    s = df[col].dropna()
    vc = s.astype(str).value_counts()
    return "; ".join(f"{k}:{v}({100*v/len(s):.0f}%)" for k,v in vc.items())

full_n, sub_n = len(full), len(sub)
tbl = []
tbl.append(["N (admissions / ICU stays)", f"{full_n}", f"{sub_n}"])
tbl.append(["Age (yr, mean±SD)", mean_sd(full,"age"), mean_sd(sub,"age")])
tbl.append(["Female (%)", pct(full,"sex",0), pct(sub,"sex",0)])
tbl.append(["Glucocorticoid use, any (%)", pct(full,"gc_use",1), pct(sub,"gc_use",1)])
tbl.append(["GC cumulative dose >100 mg (%)", pct(full,"gc_dose_pred_eq_mg",1) if False else f"{100*(full['gc_dose_pred_eq_mg']>100).mean():.1f}%", f"{100*(sub['gc_dose_fulladm']>100).mean():.1f}%"])
tbl.append(["30-day all-cause mortality (%)", pct(full,"allcause_death_30d",1), pct(sub,"allcause_death_30d",1)])
tbl.append(["In-hospital infection (%)", pct(full,"infection",1), pct(sub,"infection",1)])
tbl.append(["Charlson (mean±SD)", mean_sd(full,"charlson"), mean_sd(sub,"charlson")])
tbl.append(["PNI (mean±SD)", mean_sd(full,"pni"), mean_sd(sub,"pni")])
tbl.append(["Surgery category", n_pct_cat(full,"surgery_category"), n_pct_cat(sub,"surgery_category")])
pd.DataFrame(tbl, columns=["Characteristic","MIMIC full cohort","MIMIC MSM ICU subgroup"]).to_csv(
    f"{FEAS}/v6_full_vs_subgroup.csv", index=False)
print("=== (1) FULL vs SUBGROUP ===")
for r in tbl: print(f"  {r[0]:34s} {r[1]:>20s} | {r[2]}")

# =============================================================
# load cohort times + GC timeline + daily SOFA for MSM
# =============================================================
d = sub.copy()
NUM = ['age','charlson','pni','creatinine','gc_use_fulladm','gc_dose_fulladm',
       'sofa','apsiii','oasis','vent_24h','vaso_24h','rrt_24h','allcause_death_30d','infection']
for c in NUM:
    if c in d: d[c] = pd.to_numeric(d[c], errors="coerce")
d["sex"] = sex2bin(d["sex"])
for col in ["admittime","intime","outtime","deathtime"]:
    d[col] = pd.to_datetime(d[col], errors="coerce")
d["dod"] = pd.to_datetime(d["dod"], errors="coerce")
d["death_dt"] = d.apply(lambda r: r["deathtime"] if pd.notna(r["deathtime"]) else (r["dod"] if pd.notna(r["dod"]) else pd.NaT), axis=1)
d["T_death"] = (d["death_dt"] - d["intime"]).dt.total_seconds()/86400.0
d["mort30_intime"] = ((d["death_dt"].notna()) & (d["T_death"] <= 30)).astype(int)
d["T_obs"] = np.minimum(d["T_death"].fillna(999), 30.0)
d["status30"] = d["mort30_intime"].astype(int)

tl = pd.read_csv(f"{EXP}/gc_timeline_mimic_icu.csv")
tl["start_hour"] = pd.to_numeric(tl["start_hour"], errors="coerce")
tl["dose_predeq_mg"] = pd.to_numeric(tl["dose_predeq_mg"], errors="coerce").fillna(0)
tl["preicu_flag"] = pd.to_numeric(tl["preicu_flag"], errors="coerce").fillna(0).astype(int)
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

CONF = ["age","sex","charlson","sofa","apsiii","vent_24h","vaso_24h","creatinine"]
rows_long = []
for _, r in d.iterrows():
    sid = int(r["subject_id"]); hid = int(r["hadm_id"]); st = int(r["stay_id"])
    T = max(float(r["T_obs"]), 0.01)
    conf = {c: float(r[c]) for c in CONF}
    if any(pd.isna(v) for v in conf.values()):
        continue
    s0 = sub0 = tl[(tl["subject_id"]==sid) & (tl["hadm_id"]==hid)]
    preicu = s0[s0["preicu_flag"]==1]["dose_predeq_mg"].sum()
    post = s0[(s0["preicu_flag"]==0) & (s0["start_hour"]>=0)].sort_values("start_hour")
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
print(f"\nLong intervals: {len(ld)} | stays {ld['stay_id'].nunique()} | deaths {int(ld.groupby('stay_id')['event'].max().sum())}")

# MSM fit with diagnostics
def fit_msm(den_formula, label):
    try:
        den = smf.logit(den_formula, data=ld).fit_regularized(alpha=0.01, L1_wt=0.0, disp=0)
    except Exception:
        den = smf.logit(den_formula, data=ld).fit(method="bfgs", maxiter=300, disp=0)
    try:
        num = smf.logit("anygc ~ prev_any + time", data=ld).fit_regularized(alpha=0.01, L1_wt=0.0, disp=0)
    except Exception:
        num = smf.logit("anygc ~ prev_any + time", data=ld).fit(method="bfgs", maxiter=300, disp=0)
    pden = den.predict(ld); pnum = num.predict(ld)
    ld["p_den"] = np.where(ld["anygc"]==1, pden, 1-pden)
    ld["p_num"] = np.where(ld["anygc"]==1, pnum, 1-pnum)
    ld["w"] = (ld["p_num"]/ld["p_den"]).clip(1e-3, 1e3)
    sw = ld.groupby("stay_id")["w"].prod().clip(0.05, 20.0)
    ld2 = ld.merge(sw.rename("sw"), on="stay_id")
    useB = ["start","stop","event","anygc","stay_id","sw"]
    cph = CoxPHFitter(); cph.fit(ld2[useB], duration_col="stop", event_col="event",
                                 entry_col="start", weights_col="sw", cluster_col="stay_id", robust=True)
    hr = cph.hazard_ratios_["anygc"]
    lo = cph.summary.loc["anygc","exp(coef) lower 95%"]
    hi = cph.summary.loc["anygc","exp(coef) upper 95%"]
    p  = cph.summary.loc["anygc","p"]
    # ---- diagnostics ----
    cstat = roc_auc_score(ld["anygc"].astype(int), pden) if len(set(ld["anygc"].astype(int)))>1 else float("nan")
    ess = (sw.sum())**2 / (sw**2).sum()
    diag = dict(model=label, hr=hr, lo=lo, hi=hi, p=p,
                sw_mean=sw.mean(), sw_median=sw.median(), sw_min=sw.min(), sw_max=sw.max(),
                sw_p1=sw.quantile(0.01), sw_p99=sw.quantile(0.99),
                ess=ess, n_pos_lt01=int((sw<0.1).sum()), n_pos_gt10=int((sw>10).sum()),
                n_pos_gt5=int((sw>5).sum()), treat_cstat=cstat)
    return hr, lo, hi, p, sw, diag, den

# (B) baseline-severity MSM == V4
denB = "anygc ~ prev_any + time + age + sex + charlson + sofa + apsiii + vent_24h + vaso_24h + creatinine + preicu_any"
hrB, loB, hiB, pB, swB, diaB, _ = fit_msm(denB, "B baseline-sev")
print(f"[B] MSM baseline-sev: HR={hrB:.3f} [{loB:.3f}-{hiB:.3f}] p={pB:.2f}")
# (C) daily-SOFA MSM
denC = "anygc ~ prev_any + time + age + sex + charlson + sofa_d + apsiii + vent_24h + vaso_24h + creatinine + preicu_any"
hrC, loC, hiC, pC, swC, diaC, denCmodel = fit_msm(denC, "C daily-SOFA")
print(f"[C] MSM daily-SOFA: HR={hrC:.3f} [{loC:.3f}-{hiC:.3f}] p={pC:.2f}")

# ---- covariate balance (raw vs IPW) on ever-GC vs never-GC (stay level) ----
stay = ld.groupby("stay_id").agg(anygc=("anygc","max"), sofa=("sofa","first"),
        apsiii=("apsiii","first"), age=("age","first"), sex=("sex","first"),
        charlson=("charlson","first")).reset_index()
stay = stay.merge(swC.rename("sw"), on="stay_id")
bal_rows = []
for cv in ["sofa","apsiii","age","sex","charlson"]:
    m1 = stay.loc[stay.anygc==1, cv].mean(); m0 = stay.loc[stay.anygc==0, cv].mean()
    sd = stay[cv].std()
    if pd.isna(sd) or sd == 0 or pd.isna(m1) or pd.isna(m0):
        bal_rows.append([cv, float("nan"), float("nan")]); continue
    raw_smd = (m1-m0)/sd
    w1 = stay.loc[stay.anygc==1, cv]; w0 = stay.loc[stay.anygc==0, cv]
    wmean1 = np.average(w1, weights=stay.loc[stay.anygc==1,"sw"])
    wmean0 = np.average(w0, weights=stay.loc[stay.anygc==0,"sw"])
    w_smd = (wmean1-wmean0)/sd
    bal_rows.append([cv, round(raw_smd,3), round(w_smd,3)])
bal_df = pd.DataFrame(bal_rows, columns=["covariate","raw_SMD","IPW_SMD"])

dia_all = pd.DataFrame([diaB, diaC])
dia_all.to_csv(f"{FEAS}/v6_weight_diagnostics.csv", index=False)
bal_df.to_csv(f"{FEAS}/v6_weight_balance.csv", index=False)
print("\n=== (2) WEIGHT DIAGNOSTICS (daily-SOFA model C) ===")
for k,v in diaC.items(): print(f"  {k:14s}: {v}")
print("Covariate balance (raw vs IPW SMD):")
print(bal_df.to_string(index=False))

# ---- weight histogram ----
fig, ax = plt.subplots(figsize=(7,3.6))
ax.hist(swC, bins=30, color="#2c7fb8", edgecolor="white")
ax.axvline(1, color="grey", ls="--", label="1.0 (null weight)")
ax.set_title("Stabilized IPTW: per-stay weights (daily-SOFA MSM)")
ax.set_xlabel("stabilized weight"); ax.set_ylabel("n stays")
ax.legend(); plt.tight_layout(); plt.savefig(f"{FEAS}/v6_weight_hist.png", dpi=150); plt.close()

# =============================================================
# (3) FIXED-WINDOW 48h GC-DOSE SENSITIVITY
# =============================================================
f = d.dropna(subset=["sofa","apsiii","oasis","age","sex","charlson","allcause_death_30d","infection"]).copy()
f["dose48"] = pd.to_numeric(f["gc_dose_48h"], errors="coerce").fillna(0)
f["grp"] = pd.Categorical(
    np.where(f["gc_use_48h"].astype(int)==0, "none",
             np.where(f["dose48"]<=THRESH, "low", "high")),
    categories=["none","low","high"])
print(f"\n=== (3) FIXED 48h DOSE: n={len(f)} groups:",
      {g:int((f.grp==g).sum()) for g in ['none','low','high']})

# death: landmark at 48h, exclude in-window death
f["dead_in48"] = ((f["mort30_intime"]==1) & (f["T_death"]<=2)).astype(int)
land = f[f["dead_in48"]==0].copy()
land["tstart"] = 2.0
land["tstop"] = np.minimum(land["T_obs"], 30.0)
land["tstop"] = np.maximum(land["tstop"], land["tstart"]+0.01)
land["ev"] = ((land["mort30_intime"]==1) & (land["T_death"]>2)).astype(int)
covs = ["sofa","apsiii","oasis","age","sex","charlson"]
land["tt"] = (land["tstop"] - land["tstart"]).clip(lower=0.01)
Xc = land[["grp"]+covs+["tt","ev"]].copy()
Xc = pd.get_dummies(Xc, columns=["grp"], drop_first=True)
Xc = sm.add_constant(Xc).astype(float)
print("DEBUG land rows", len(land), "events", int(land["ev"].sum()),
      "| const cols", [c for c in Xc.columns if Xc[c].nunique()<=1],
      "| nan", bool(Xc.isna().any().any()))
from statsmodels.duration.hazard_regression import PHReg
Xdes = Xc.drop(columns=["tt","ev"])
ph = PHReg(land["tt"].values, Xdes.values, status=land["ev"].values, ties="efron").fit()
names = list(Xdes.columns)
death_res = {}
for term in ["grp_low","grp_high"]:
    if term in names:
        i = names.index(term)
        b = ph.params[i]; ci = ph.conf_int(); p = ph.pvalues[i]
        death_res[term] = (float(np.exp(b)), float(np.exp(ci[i,0])), float(np.exp(ci[i,1])), float(p))
print("Death (landmark 48h, ref=none):", death_res)

# infection: logistic (in-hospital infection)
Xl = f[["grp"]+covs].copy()
Xl = pd.get_dummies(Xl, columns=["grp"], drop_first=True)
Xl = sm.add_constant(Xl).astype(float)
rin = sm.Logit(f["infection"].astype(int), Xl).fit(disp=0)
inf_res = {}
for term in ["grp_low","grp_high"]:
    if term in rin.params.index:
        inf_res[term] = (np.exp(rin.params[term]), np.exp(rin.conf_int().loc[term,0]),
                         np.exp(rin.conf_int().loc[term,1]), rin.pvalues[term])
print("Infection (ref=none):", inf_res)

fdose = pd.DataFrame([
  ("Fixed-48h low-dose GC -> 30d death (landmark 48h)", *death_res.get("grp_low",(np.nan,)*4)),
  ("Fixed-48h high-dose GC -> 30d death (landmark 48h)", *death_res.get("grp_high",(np.nan,)*4)),
  ("Fixed-48h low-dose GC -> in-hospital infection", *inf_res.get("grp_low",(np.nan,)*4)),
  ("Fixed-48h high-dose GC -> in-hospital infection", *inf_res.get("grp_high",(np.nan,)*4)),
], columns=["contrast","HR_OR","lo","hi","p"])
fdose.to_csv(f"{FEAS}/v6_fixed48_dose.csv", index=False)
print(fdose.to_string(index=False))

# ---- forest figure ----
labels = ["Death: low (ref none)","Death: high (ref none)",
          "Infection: low (ref none)","Infection: high (ref none)"]
xs=[];los=[];his=[];cols=[]
for lab,(o,lo,hi,p) in zip(labels,[death_res.get("grp_low"),death_res.get("grp_high"),
                                    inf_res.get("grp_low"),inf_res.get("grp_high")]):
    xs.append(o); los.append(lo); his.append(hi)
    cols.append("#2c7fb8" if "Death" in lab else "#1a9641")
fig, ax = plt.subplots(figsize=(8.2,3.8))
for i,(x,lo,hi,c,lab) in enumerate(zip(xs,los,his,cols,labels)):
    ax.plot([lo,hi],[i,i], color=c, lw=2)
    ax.plot([x],[i], "o", color=c, ms=9)
    ax.text(hi*1.04, i, f"{x:.2f}", va="center", fontsize=8, color=c)
ax.axvline(1, color="grey", ls="--")
ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8)
ax.set_xscale("log"); ax.set_xlabel("Hazard/Odds Ratio vs no-GC-in-48h (log scale)")
ax.set_title("Fixed-window 48h GC dose sensitivity (MIMIC ICU subgroup)")
ax.set_xlim(0.3, 5)
plt.tight_layout(); plt.savefig(f"{FEAS}/v6_fixed48_forest.png", dpi=150); plt.close()

# =============================================================
# (4) SURGERY SUBGROUP + ABSOLUTE RISK DIFFERENCE (infection)
# =============================================================
surg = f.copy()
surg_tab = []
for g in surg["surgery_category"].dropna().unique():
    gg = surg[surg["surgery_category"]==g]
    n=len(gg); gi=int((gg.gc_use_48h.astype(int)==1).sum())
    inf_gc=int(gg.loc[gg.gc_use_48h.astype(int)==1,"infection"].mean()*gi) if gi else 0
    inf_nogc_n=int((gg.gc_use_48h.astype(int)==0).sum())
    inf_nogc=int(gg.loc[gg.gc_use_48h.astype(int)==0,"infection"].sum())
    r_gc = gg.loc[gg.gc_use_48h.astype(int)==1,"infection"].mean() if gi else np.nan
    r_nogc = gg.loc[gg.gc_use_48h.astype(int)==0,"infection"].mean() if inf_nogc_n else np.nan
    ard = (r_gc-r_nogc) if (gi and inf_nogc_n) else np.nan
    surg_tab.append([str(g), n, gi, f"{100*r_gc:.1f}%" if gi else "-",
                     f"{100*r_nogc:.1f}%" if inf_nogc_n else "-",
                     f"{100*ard:.1f} pp" if pd.notna(ard) else "-"])
surg_df = pd.DataFrame(surg_tab, columns=["surgery_category","N","GC_exposed","infection%_GC","infection%_noGC","ARD_GC"])
surg_df.to_csv(f"{FEAS}/v6_surgery_ard.csv", index=False)
print("\n=== (4) SURGERY SUBGROUP (in-hospital infection ARD) ===")
print(surg_df.to_string(index=False))

# overall ICU-subgroup infection ARD by any-GC
icu = d.dropna(subset=["infection"]).copy()
icu["gcany"] = icu["gc_use_fulladm"].astype(int)
r1 = icu.loc[icu.gcany==1,"infection"].mean(); n1=int((icu.gcany==1).sum())
r0 = icu.loc[icu.gcany==0,"infection"].mean(); n0=int((icu.gcany==0).sum())
print(f"ICU subgroup in-hospital infection: GC {100*r1:.1f}% (n={n1}) vs no-GC {100*r0:.1f}% (n={n0}); crude ARD = {100*(r1-r0):.1f} pp")

print("\nSAVED: v6_full_vs_subgroup.csv, v6_weight_diagnostics.csv, v6_weight_balance.csv,",
      "v6_fixed48_dose.csv, v6_surgery_ard.csv, v6_fixed48_forest.png, v6_weight_hist.png")
