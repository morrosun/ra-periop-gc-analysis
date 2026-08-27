# -*- coding: utf-8 -*-
import os as _os
ROOT = _os.path.dirname(_os.path.abspath(__file__))
"""
RA 围术期统计报告 V2 复算脚本（响应审稿人意见的可行修正）
仅用现有 analytic CSV，无需回库。
产出：revision_v2_analysis.csv + 控制台摘要
"""
import sys, warnings
_pylibs = _os.environ.get("RA_GC_PYLIBS", _os.path.join(ROOT, ".pylibs"))
if _os.path.isdir(_pylibs):
    sys.path.insert(0, _pylibs)
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats
warnings.filterwarnings("ignore")

CSV = "exports/analytic_mimiciv.csv"
d = pd.read_csv(CSV)
for c in ['age','gc_use','btsdmard_use','inhosp_death','allcause_death_30d',
          'infection','major_comp','charlson','gc_dose_pred_eq_mg','creatinine']:
    d[c] = pd.to_numeric(d[c], errors='coerce')
# sex 为字符串 M/F，单独映射（切勿 to_numeric，否则整列为 NaN）
d['sex'] = d['sex'].astype(str).str.upper().map({'M':1,'F':0})
d['gc_strata'] = np.where(d['gc_use']==0,'none',
                          np.where(d['gc_dose_pred_eq_mg']>100,'high','low'))
d['gc_high'] = (d['gc_strata']=='high').astype(int)
d['gc_any'] = d['gc_use'].astype(int)
d['log_cr'] = np.log(d['creatinine'].clip(lower=0.1))
rows = []
Z = 1.959963985

def orci(m, term):
    b = m.params[term]; s = m.bse[term]
    if s>0 and np.isfinite(s):
        return np.exp(b), np.exp(b-Z*s), np.exp(b+Z*s), 2*(1-stats.norm.cdf(abs(b/s)))
    return float('nan'), float('nan'), float('nan'), float('nan')

def fit(formula, label, out=None):
    try:
        m = smf.logit(formula, data=d).fit(disp=0, maxiter=200)
        return m
    except Exception as e:
        print(f"  !! {label}: {type(e).__name__}: {e}")
        return None

# ---------- 0. 复现基线（age+sex+charlson），与现有报告核对 ----------
print("="*70); print("0. 基线复现（age+sex+charlson），核对现有报告"); print("="*70)
for out in ['inhosp_death','allcause_death_30d','infection']:
    m = fit(f"{out} ~ C(gc_strata, Treatment('none')) + age + sex + charlson", f"base {out}")
    if m:
        for t in ['C(gc_strata, Treatment(\'none\'))[T.low]','C(gc_strata, Treatment(\'none\'))[T.high]']:
            OR,lo,hi,p = orci(m,t)
            print(f"  {out:22s} {t.split('T.')[1].rstrip(']'):5s} OR={OR:.3f} [{lo:.3f}-{hi:.3f}] p={p:.2e}")

# ---------- 1. 扩展调整（+肌酐 + 手术类别）----------
print("\n"+"="*70); print("1. 扩展调整（+log肌酐 + 手术类别），回应审稿人#2"); print("="*70)
for out in ['allcause_death_30d','infection']:
    # base
    mb = fit(f"{out} ~ C(gc_strata, Treatment('none')) + age + sex + charlson", f"base {out}")
    # expanded
    me = fit(f"{out} ~ C(gc_strata, Treatment('none')) + age + sex + charlson + log_cr + C(surgery_category)", f"exp {out}")
    if mb and me:
        for tag,t in [('base','C(gc_strata, Treatment(\'none\'))[T.high]'),
                      ('exp','C(gc_strata, Treatment(\'none\'))[T.high]')]:
            mm = mb if tag=='base' else me
            OR,lo,hi,p = orci(mm,t)
            rows.append(dict(section='GC-high expanded', outcome=out, model=tag, term='gc_high',
                             OR=round(OR,3), lo=round(lo,3), hi=round(hi,3), p=round(p,4)))
            print(f"  {out:22s} {tag:4s} gc_high OR={OR:.3f} [{lo:.3f}-{hi:.3f}] p={p:.3f}")

# ---------- 2. 绝对风险差（marginal standardization）----------
print("\n"+"="*70); print("2. 绝对风险差 ARD（none vs high，marginal standardization）"); print("="*70)
def ard(out, formula_base, formula_exp):
    res={}
    for tag,fmt in [('base',formula_base),('exp',formula_exp)]:
        m = fit(fmt, f"ard {out} {tag}")
        if not m: continue
        df = d.copy()
        # 直接构造：把 gc_strata 全部设为 none，再设为 high
        dn = df.copy(); dn['gc_strata']='none'
        dh = df.copy(); dh['gc_strata']='high'
        # 需保证公式内变量存在；用 C(gc_strata...) 形式
        pN = m.predict(dn); pH = m.predict(dh)
        ard_val = (pH.mean() - pN.mean())*100
        res[tag]=ard_val
        print(f"  {out:22s} {tag:4s} ARD(high-none)={ard_val:+.2f} pp")
    return res
ard('allcause_death_30d',
    "allcause_death_30d ~ C(gc_strata, Treatment('none')) + age + sex + charlson",
    "allcause_death_30d ~ C(gc_strata, Treatment('none')) + age + sex + charlson + log_cr + C(surgery_category)")
ard('infection',
    "infection ~ C(gc_strata, Treatment('none')) + age + sex + charlson",
    "infection ~ C(gc_strata, Treatment('none')) + age + sex + charlson + log_cr + C(surgery_category)")

# ---------- 3. 手术类别分层（GC-high vs none）----------
print("\n"+"="*70); print("3. 手术类别分层（GC-high vs none，调整 age+sex+charlson）"); print("="*70)
sg = d['surgery_category'].dropna().unique()
for out in ['allcause_death_30d','infection']:
    print(f"  -- {out} --")
    for cat in sorted(sg):
        sub = d[d['surgery_category']==cat]
        if sub['gc_high'].sum()<5 or (sub['gc_strata']=='none').sum()<5: continue
        m = fit(f"{out} ~ C(gc_strata, Treatment('none')) + age + sex + charlson", f"sg {cat} {out}", out)
        if m:
            OR,lo,hi,p = orci(m,"C(gc_strata, Treatment('none'))[T.high]")
            n=len(sub); ev=int(sub[out].sum()); gch=int(sub['gc_high'].sum())
            print(f"    {cat:22s} n={n:4d} ev={ev:3d} gc_high={gch:3d} OR={OR:.3f} [{lo:.3f}-{hi:.3f}] p={p:.3f}")
            rows.append(dict(section='surgery subgroup', outcome=out, model=cat, term='gc_high',
                             OR=round(OR,3), lo=round(lo,3), hi=round(hi,3), p=round(p,4)))

# ---------- 4. IPTW 敏感性（GC-high 倾向加权）----------
print("\n"+"="*70); print("4. IPTW 敏感性（GC-high 倾向加权，cov: age/sex/charlson/log_cr）"); print("="*70)
sub = d.dropna(subset=['age','sex','charlson','log_cr','gc_high','allcause_death_30d']).copy()
psm = smf.logit("gc_high ~ age + sex + charlson + log_cr", data=sub).fit(disp=0)
sub['ps'] = psm.predict(sub)
sub['w'] = np.where(sub['gc_high']==1, 1/sub['ps'], 1/(1-sub['ps']))
m_w = smf.logit("allcause_death_30d ~ gc_high", data=sub,
                freq_weights=sub['w']).fit(disp=0, maxiter=200)
OR,lo,hi,p = orci(m_w,'gc_high')
print(f"  IPTW gc_high -> 30d death OR={OR:.3f} [{lo:.3f}-{hi:.3f}] p={p:.3f} (n={len(sub)}, ev={int(sub['allcause_death_30d'].sum())})")
rows.append(dict(section='IPTW', outcome='allcause_death_30d', model='IPTW', term='gc_high',
                 OR=round(OR,3), lo=round(lo,3), hi=round(hi,3), p=round(p,4)))

# ---------- 5. Meta 权重（死亡 meta gc_use->30d death；感染 meta）----------
print("\n"+"="*70); print("5. Meta 权重（回应审稿人#8）"); print("="*70)
def se_from_ci(OR,lo,hi):
    return (np.log(hi)-np.log(lo))/(2*Z)
def weights(name, lib_vals):
    # lib_vals: list of (lib, OR, lo, hi, events)
    print(f"  -- {name} --")
    out=[]
    for lib,OR,lo,hi,ev in lib_vals:
        se=se_from_ci(OR,lo,hi); w=1/se**2
        out.append((lib,OR,lo,hi,ev,se,w))
    tot=sum(o[6] for o in out)
    for lib,OR,lo,hi,ev,se,w in out:
        print(f"    {lib:10s} OR={OR:.2f} [{lo:.2f}-{hi:.2f}] ev={ev:4d} SE={se:.3f} weight={100*w/tot:.1f}%")
    return out
weights("GC use -> 30d death (DL-RE pooled 1.82 [1.32-2.52])",
         [("MIMIC-IV",1.88,1.35,2.63,169),("NWICU",0.97,0.23,3.99,11),("INSPIRE",2.55,0.06,110.3,4)])
weights("GC use -> infection (DL-RE pooled 1.44 [1.22-1.68])",
         [("MIMIC-IV",1.475,1.243,1.751,430),("NWICU",0.929,0.498,1.734,33),
          ("INSPIRE",2.183,0.914,5.213,32),("eICU",1.272,0.7,2.309,27)])

# ---------- 6. Attenuation 修正（logOR 公式 + OR<1 方向说明）----------
print("\n"+"="*70); print("6. Attenuation 修正（logOR 公式，标注 OR<1 方向翻转）"); print("="*70)
# 取自 crossdb_summary.csv
cross = [
 ("MIMIC-IV","30d death",3.28,2.08),("NWICU","30d death",4.29,2.55),("INSPIRE","30d death",1.64,0.99),
 ("MIMIC-IV","infection",1.473,1.036),("NWICU","infection",0.637,0.315),("INSPIRE","infection",2.174,1.506),
]
print(f"  {'db':9s} {'outcome':11s} {'A_OR':>6s} {'B_OR':>6s} {'att(logOR)':>11s}  note")
for db,out,A,B in cross:
    lA,lB=np.log(A),np.log(B)
    att=(lA-lB)/lA if lA!=0 else float('nan')
    # 方向：若 A>1 且 B 更接近1 为正向衰减；若 A<1 则"衰减"符号翻转
    if A>1:
        note=f"{'attenuation' if B<A else 'amplification'} toward null"
    else:
        note="A<1: metric sign flips (NOT interpretable as attenuation)"
    print(f"  {db:9s} {out:11s} {A:6.3f} {B:6.3f} {att*100:10.1f}%  {note}")
    rows.append(dict(section='attenuation(logOR)', outcome=out, model=db, term='gc_high',
                     OR=round(A,3), lo=round(B,3), hi=round(att*100,1), p=float('nan')))

pd.DataFrame(rows).to_csv("revision_v2_analysis.csv", index=False)
print("\n[saved] revision_v2_analysis.csv")
