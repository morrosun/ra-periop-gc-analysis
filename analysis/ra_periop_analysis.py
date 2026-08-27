# -*- coding: utf-8 -*-
import os as _os
ROOT = _os.path.dirname(_os.path.abspath(__file__))
"""
RA perioperative analysis (a/b/c)
  (b) MIMIC-IV modeling: GC dose strata + b/tsDMARD vs death/infection
  (c) Cross-DB external validation + random-effects meta-analysis (I^2)
Inputs : exports/analytic_{mimiciv,nwicu,inspire,eicu}.csv
Outputs: mimic_model_results.csv, meta_results.csv, forest_*.png
"""
import os, warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

FEAS = os.path.dirname(os.path.abspath(__file__))
EXP  = os.path.join(FEAS, "exports")
OUT  = FEAS

libs = {"mimiciv":"MIMIC-IV","nwicu":"NWICU","inspire":"INSPIRE","eicu":"eICU"}
dfs = {k: pd.read_csv(os.path.join(EXP, f"analytic_{k}.csv")) for k in libs}

# ---------- helpers ----------
def gc_strata(gc_use, dose):
    if gc_use == 0: return "none"
    if pd.isna(dose) or dose <= 100: return "low"
    return "high"

def logit_res(frame, outcome, formula):
    """Return dict of OR (95% CI), p for each term in a logistic model."""
    d = frame.dropna(subset=[outcome]).copy()
    try:
        m = smf.logit(formula, data=d).fit(disp=0)
    except Exception as e:
        return {"_error": str(e)}
    out = {}
    for term in m.params.index:
        if term in ("Intercept",): continue
        b = m.params[term]; se = m.bse[term]; p = m.pvalues[term]
        orr = np.exp(b); lo = np.exp(b-1.96*se); hi = np.exp(b+1.96*se)
        out[term] = (orv, lo, hi, p) if False else (round(orr,3), round(lo,3), round(hi,3), f"{p:.2e}" if p<0.001 else round(p,3))
    out["_n"] = int(d.shape[0]); out["_n_event"] = int(d[outcome].sum())
    return out

# ================= (b) MIMIC-IV MODELING =================
m = dfs["mimiciv"].copy()
m["gc_strata"] = [gc_strata(u,d) for u,d in zip(m.gc_use, m.gc_dose_pred_eq_mg)]
m["sex_m"] = (m.sex=="M").astype(int)
m["gc_strata"] = pd.Categorical(m.gc_strata, ["none","low","high"])
m["dmard_class"] = pd.Categorical(m.dmard_class, ["none","csDMARD","TNFi","ILi","JAKi"])
m = m.dropna(subset=["age","sex_m","charlson"]).reset_index(drop=True)  # emergency dropped: ~95% are 1 (near-zero variance) -> 0.0-inf OR artifacts

outcomes = ["inhosp_death","allcause_death_30d","infection"]
rows = []
def addrow(model, exposure, outcome, desc, res):
    if "_error" in res:
        rows.append(dict(model=model, exposure=exposure, outcome=outcome, desc=desc, term="ERR", OR="", CI95="", p=res["_error"], n="", events="")); return
    for term, val in res.items():
        if term.startswith("_") or not isinstance(val, (tuple, list)): continue
        orv, lo, hi, p = val
        rows.append(dict(model=model, exposure=exposure, outcome=outcome, desc=desc, term=term,
                         OR=orv, CI95=f"{lo}-{hi}", p=p, n=res.get("_n",""), events=res.get("_n_event","")))

# --- GC dose strata ---
for oc in outcomes:
    addrow("GC strata (crude)", "gc_strata", oc, "ref=none",
           logit_res(m, oc, f"gc_strata ~ C(gc_strata, Treatment('none'))"))
    addrow("GC strata (adj)", "gc_strata", oc, "ref=none; adj age,sex,emerg,charlson",
           logit_res(m, oc, f"{oc} ~ C(gc_strata, Treatment('none')) + age + sex_m + charlson"))  # emergency removed (near-zero variance)
# --- GC continuous (per 100 mg) ---
m["gc_dose_100"] = m.gc_dose_pred_eq_mg/100.0
for oc in outcomes:
    addrow("GC continuous (adj)", "gc_dose/100mg", oc, "per 100 mg pred-eq; adj",
           logit_res(m[m.gc_use==1], oc, f"{oc} ~ gc_dose_100 + age + sex_m + charlson"))  # emergency removed (near-zero variance)
# --- DMARD class ---
for oc in outcomes:
    addrow("DMARD (crude)", "dmard_class", oc, "ref=none",
           logit_res(m, oc, f"dmard_class ~ C(dmard_class, Treatment('none'))"))
    addrow("DMARD (adj)", "dmard_class", oc, "ref=none; adj",
           logit_res(m, oc, f"{oc} ~ C(dmard_class, Treatment('none')) + age + sex_m + charlson"))  # emergency removed (near-zero variance)
# --- any b/tsDMARD binary ---
m["btsdmard"] = m.btsdmard_use.astype(int)
for oc in outcomes:
    addrow("b/tsDMARD any (adj)", "btsdmard_use", oc, "ref=none; adj",
           logit_res(m, oc, f"{oc} ~ btsdmard + age + sex_m + charlson"))  # emergency removed (near-zero variance)

res_df = pd.DataFrame(rows)
res_df.to_csv(os.path.join(OUT,"mimic_model_results.csv"), index=False)
print("=== MIMIC modeling: key adjusted results ===")
print(res_df[res_df.model.str.contains("adj")][["model","outcome","term","OR","CI95","p","n","events"]].to_string(index=False))

# ================= (c) META-ANALYSIS =================
def two_by_two(df, exp_col, exp_val, out_col):
    ex = df[df[exp_col]==exp_val]; un = df[df[exp_col]!=exp_val]
    a = int(ex[out_col].sum()); b = len(ex)-a
    c = int(un[out_col].sum()); d = len(un)-c
    return a,b,c,d

def logor_se(a,b,c,d):
    if min(a,b,c,d)==0:
        a,b,c,d = [x+0.5 for x in (a,b,c,d)]
    lor = np.log((a*d)/(b*c)); se = np.sqrt(1/a+1/b+1/c+1/d)
    return lor, se, (a,b,c,d)

def dl_meta(studies):
    """studies: list of (name, logor, se). Returns pooled OR, lo, hi, I2, tau2, ks."""
    ys = np.array([s[1] for s in studies]); ses = np.array([s[2] for s in studies])
    w = 1.0/ses**2; k=len(studies)
    if k==1:
        return (np.exp(ys[0]), np.exp(ys[0]-1.96*ses[0]), np.exp(ys[0]+1.96*ses[0]), 0.0, 0.0, k)
    fixed = np.sum(w*ys)/np.sum(w)
    Q = np.sum(w*(ys-fixed)**2); df=k-1
    tau2 = max(0.0, (Q-df)/(np.sum(w)-np.sum(w**2)/np.sum(w)))
    wr = 1.0/(ses**2+tau2)
    pooled = np.sum(wr*ys)/np.sum(wr); se_r = np.sqrt(1/np.sum(wr))
    I2 = max(0.0, (Q-df)/Q*100.0) if Q>0 else 0.0
    return (np.exp(pooled), np.exp(pooled-1.96*se_r), np.exp(pooled+1.96*se_r), I2, tau2, k)

# death column per library
deathcol = {"mimiciv":"allcause_death_30d","nwicu":"inhosp_death","inspire":"allcause_death_30d","eicu":"inhosp_death"}

meta_rows = []
forest_jobs = []  # (title, fname, studies with OR/CI)

def build_studies(df, exp_col, exp_val, out_col):
    a,b,c,d = two_by_two(df, exp_col, exp_val, out_col)
    if a+b==0 or c+d==0: return None
    lor,se,_ = logor_se(a,b,c,d)
    return (lor, se, (a,b,c,d))

# ---- GC any use ----
for oc_label, oc in [("infection","infection"), ("death",None)]:
    studies=[]; detail=[]
    for k in libs:
        df = dfs[k]
        col = oc if oc else deathcol[k]
        if col not in df: continue
        if oc is None and df[col].sum()==0:  # no events in this lib
            continue
        r = build_studies(df, "gc_use", 1, col)
        if r is None: continue
        lor,se,ab = r[0],r[1],r[2]
        orr=np.exp(lor); lo=np.exp(lor-1.96*se); hi=np.exp(lor+1.96*se)
        studies.append((libs[k], lor, se))
        detail.append(dict(library=libs[k], exposure="GC use (any)", outcome=oc_label,
                           a=ab[0],n_exp=ab[0]+ab[1], c=ab[2],n_unexp=ab[2]+ab[3],
                           OR=round(orr,3), lo=round(lo,3), hi=round(hi,3)))
    if len(studies)>=2:
        pooled,plo,phi,I2,tau2,kk = dl_meta(studies)
        for det in detail:
            det.update(pooled_OR=round(pooled,3), pooled_lo=round(plo,3), pooled_hi=round(phi,3), I2=round(I2,1), k=kk)
            meta_rows.append(det)
        forest_jobs.append((f"GC use (any) -> {oc_label}", f"forest_gcuse_{oc_label}.png", studies, (pooled,plo,phi)))

# ---- b/tsDMARD any ----
for oc_label, oc in [("infection","infection"), ("death",None)]:
    studies=[]; detail=[]
    for k in libs:
        if k=="eicu": continue  # eICU has no DMARD
        df = dfs[k]
        col = oc if oc else deathcol[k]
        if col not in df: continue
        if oc is None and df[col].sum()==0: continue
        r = build_studies(df, "btsdmard_use", 1, col)
        if r is None: continue
        lor,se,ab = r[0],r[1],r[2]
        orr=np.exp(lor); lo=np.exp(lor-1.96*se); hi=np.exp(lor+1.96*se)
        studies.append((libs[k], lor, se))
        detail.append(dict(library=libs[k], exposure="b/tsDMARD (any)", outcome=oc_label,
                           a=ab[0],n_exp=ab[0]+ab[1], c=ab[2],n_unexp=ab[2]+ab[3],
                           OR=round(orr,3), lo=round(lo,3), hi=round(hi,3)))
    if len(studies)>=2:
        pooled,plo,phi,I2,tau2,kk = dl_meta(studies)
        for det in detail:
            det.update(pooled_OR=round(pooled,3), pooled_lo=round(plo,3), pooled_hi=round(phi,3), I2=round(I2,1), k=kk)
            meta_rows.append(det)
        forest_jobs.append((f"b/tsDMARD (any) -> {oc_label}", f"forest_dmard_{oc_label}.png", studies, (pooled,plo,phi)))

meta_df = pd.DataFrame(meta_rows)
meta_df.to_csv(os.path.join(OUT,"meta_results.csv"), index=False)
print("\n=== META-ANALYSIS (random-effects) ===")
print(meta_df[["library","exposure","outcome","OR","lo","hi","pooled_OR","I2","k"]].to_string(index=False))

# ---- forest plots ----
def forest(title, fname, studies, pooled):
    names=[s[0] for s in studies]; lors=[s[1] for s in studies]; ses=[s[2] for s in studies]
    ors=[np.exp(l) for l in lors]
    los=[np.exp(l-1.96*s) for l,s in zip(lors,ses)]; his=[np.exp(l+1.96*s) for l,s in zip(lors,ses)]
    fig,ax=plt.subplots(figsize=(7,max(3,0.7*len(names)+2)))
    y=np.arange(len(names))[::-1]
    ax.errorbar(ors,y,xerr=[np.array(ors)-np.array(los),np.array(his)-np.array(ors)],
                fmt='o',color='#1f77b4',ecolor='#888',capsize=4,ms=6)
    ax.axvline(1,color='k',lw=0.8,ls='--')
    px=[pooled[0]]; plx=[pooled[0]-pooled[1]]; phx=[pooled[2]-pooled[0]]
    ax.errorbar(px,[-0.6],xerr=[plx,phx],fmt='s',color='#d62728',ecolor='#d62728',capsize=4,ms=8,label='pooled')
    for yi,nm,o,lo,hi in zip(y,names,ors,los,his):
        ax.text(hi*1.05,yi,f"{o:.2f} ({lo:.2f}-{hi:.2f})",va='center',fontsize=8)
    ax.set_yticks(list(y)+[-0.6]); ax.set_yticklabels(list(names)+['Pooled'],fontsize=9)
    ax.set_xscale('log'); ax.set_title(title,fontsize=11); ax.set_xlabel('Odds Ratio (log scale)')
    ax.legend(loc='upper left',fontsize=8); plt.tight_layout()
    plt.savefig(os.path.join(OUT,fname),dpi=130); plt.close()
    print("saved", fname)

for title,fname,studies,pooled in forest_jobs:
    forest(title,fname,studies,pooled)

print("\nDONE.")
