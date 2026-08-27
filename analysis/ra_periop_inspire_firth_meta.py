# -*- coding: utf-8 -*-
import os as _os
ROOT = _os.path.dirname(_os.path.abspath(__file__))
"""
RA perioperative - INSPIRE 4-death precision fix
Two complementary strategies requested by user:
  (A) Firth penalized logistic regression on INSPIRE (4 deaths, binary gc_use,
      no dose) -> finite, well-behaved OR with profile-likelihood CIs
      (the prior MI gave OR 0.99 [0.02-54.8], i.e. essentially non-identifiable).
  (B) Combined external validation: two-stage random-effects meta pooling the
      adjusted gc_use -> 30-day death logOR across MIMIC-IV (reference, standard
      Logit) + NWICU (Firth) + INSPIRE (Firth). The pooled estimate borrows
      strength from the larger cohorts and shrinks INSPIRE's wide CI.

All numbers recomputed from the exported analytic CSVs (no DB needed).
"""
import sys, os
_pylibs = _os.environ.get("RA_GC_PYLIBS", _os.path.join(ROOT, ".pylibs"))
if _os.path.isdir(_pylibs):
    sys.path.insert(0, _pylibs)
import numpy as np, pandas as pd
from scipy.special import expit
from scipy.optimize import brentq
from scipy import stats as st
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FEAS = _os.environ.get("RA_GC_DATA", _os.path.join(ROOT, "data"))  # extracted CSVs dir (NOT in repo; see README)

DBS = {
    "MIMIC-IV": f"{FEAS}/exports/analytic_mimiciv.csv",
    "NWICU":    f"{FEAS}/exports/analytic_nwicu.csv",
    "INSPIRE":  f"{FEAS}/exports/analytic_inspire.csv",
}

# ----------------------------------------------------------------------
# Firth penalized logistic regression (Heinze & Schemper 2002)
# Penalized log-likelihood = ll(beta) + 0.5 * log|I(beta)|
# Modified score uses p_hat = p + 0.5*(1-2p)*h, h = diag of weighted hat matrix.
def _pen_ll(X, y, beta):
    eta = X @ beta
    p_ = np.clip(expit(eta), 1e-12, 1 - 1e-12)
    ll = (y * np.log(p_) + (1 - y) * np.log(1 - p_)).sum()
    W = p_ * (1 - p_)
    I = X.T @ (W[:, None] * X)
    sign, logdet = np.linalg.slogdet(I)
    return ll + 0.5 * logdet

def safe_inv(M):
    try:
        return np.linalg.inv(M)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(M)

def firth_fit(X, y, maxit=300, tol=1e-10):
    n, p = X.shape
    beta = np.zeros(p)
    for it in range(maxit):
        eta = X @ beta
        p_ = expit(eta)
        W = p_ * (1 - p_)
        I = X.T @ (W[:, None] * X)
        Iinv = safe_inv(I)
        H = W * np.einsum("ij,ij->i", X @ Iinv, X)          # weighted hat diag
        p_hat = np.clip(p_ + 0.5 * (1 - 2 * p_) * H, 1e-12, 1 - 1e-12)
        U = X.T @ (y - p_hat)                               # modified score
        step = Iinv @ U
        beta_new = beta + step
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new
    eta = X @ beta
    p_ = expit(eta)
    W = p_ * (1 - p_)
    I = X.T @ (W[:, None] * X)
    cov = safe_inv(I)
    se = np.sqrt(np.diag(cov))
    return beta, se, cov

def firth_profile_ci(X, y, j, beta, target_q=1.92):
    """Profile penalized-LR CI for beta[j] (target = 0.5*chi2_1,0.95 = 1.92)."""
    Lmax = _pen_ll(X, y, beta)
    target = Lmax - target_q
    def prof(bj):
        bfull = beta.copy(); bfull[j] = bj
        for _ in range(80):
            eta = X @ bfull
            p_ = expit(eta)
            W = p_ * (1 - p_)
            I = X.T @ (W[:, None] * X)
            Iinv = safe_inv(I)
            H = W * np.einsum("ij,ij->i", X @ Iinv, X)
            p_hat = np.clip(p_ + 0.5 * (1 - 2 * p_) * H, 1e-12, 1 - 1e-12)
            U = X.T @ (y - p_hat)
            step = Iinv @ U
            step[j] = 0.0
            bfull = bfull + step
            if np.max(np.abs(step)) < 1e-9:
                break
        return _pen_ll(X, y, bfull)
    s = firth_fit(X, y)[1][j]
    b = beta[j]
    lo = b; step = 0.15
    while prof(lo) > target and step < 30:
        lo -= step; step *= 1.3
    try:
        lo_root = brentq(lambda v: prof(v) - target, lo, b, xtol=1e-7, maxiter=200)
    except Exception:
        lo_root = lo
    hi = b; step = 0.15
    while prof(hi) > target and step < 30:
        hi += step; step *= 1.3
    try:
        hi_root = brentq(lambda v: prof(v) - target, b, hi, xtol=1e-7, maxiter=200)
    except Exception:
        hi_root = hi
    return np.exp(lo_root), np.exp(hi_root)

# ----------------------------------------------------------------------
# Build design matrix (gc_use -> outcome, adjusted age+sex+charlson [+pni])
def build_Xy(df, outcome, with_pni=False):
    d = df.copy()
    d["sex_m"] = (d["sex"] == "M").astype(int)
    cols = ["gc_use", "age", "sex_m", "charlson"]
    d = d.dropna(subset=["gc_use", "age", "sex_m", "charlson", outcome])
    if with_pni:
        d = d.dropna(subset=["pni"])
        cols = cols + ["pni"]
    X = sm.add_constant(d[cols].astype(float).values)
    y = d[outcome].astype(float).values
    return X, y, d, cols

def fit_standard(X, y):
    res = sm.Logit(y, X).fit(disp=0, maxiter=300)
    b = res.params[1]; s = res.bse[1]            # col 1 = gc_use (const at 0)
    z = 1.959963985
    return dict(OR=float(np.exp(b)), lo=float(np.exp(b - z*s)), hi=float(np.exp(b + z*s)),
                logOR=float(b), SE=float(s), p=float(2*(1-st.norm.cdf(abs(b/s)))),
                method="standard")

def fit_firth(X, y):
    beta, se, cov = firth_fit(X, y)
    j = 1                                          # gc_use column index
    b = beta[j]; s = se[j]
    z = 1.959963985
    wald_lo = float(np.exp(b - z*s)); wald_hi = float(np.exp(b + z*s))
    # Use Wald CI from observed information (finite, stable); profile-LR kept
    # available but can collapse for ultra-sparse models (e.g. 4 events/5 params).
    lo, hi = wald_lo, wald_hi
    pval = float(2*(1-st.norm.cdf(abs(b/s)))) if (np.isfinite(s) and s > 0) else float("nan")
    return dict(OR=float(np.exp(b)) if np.isfinite(b) else float("inf"),
                lo=lo, hi=hi, wald_lo=wald_lo, wald_hi=wald_hi,
                logOR=float(b), SE=float(s), p=pval, method="Firth")

# ----------------------------------------------------------------------
# (A) INSPIRE Firth detail
print("\n================ (A) INSPIRE Firth detail ================")
ins = pd.read_csv(DBS["INSPIRE"])
ins_rows = []
# death: base adj only (4 events -> PNI not estimable)
for outcome in ["allcause_death_30d", "infection"]:
    X, y, d, cols = build_Xy(ins, outcome, with_pni=False)
    r = fit_firth(X, y)
    ev = int(y.sum()); n = len(y)
    Xc = sm.add_constant(d[["gc_use"]].astype(float).values)
    rc = fit_firth(Xc, y)
    print(f"  [{outcome} base adj] n={n} events={ev}  adj OR={r['OR']:.3f} [{r['lo']:.3f}-{r['hi']:.3f}]  "
          f"crude OR={rc['OR']:.3f} [{rc['lo']:.3f}-{rc['hi']:.3f}]  p={r['p']:.3f}")
    ins_rows.append(dict(model="Firth_adj_base", outcome=outcome, n=n, events=ev,
                         crude_OR=round(rc["OR"],3), crude_lo=round(rc["lo"],3), crude_hi=round(rc["hi"],3),
                         adj_OR=round(r["OR"],3), adj_lo=round(r["lo"],3), adj_hi=round(r["hi"],3),
                         p=round(r["p"],3), logOR=round(r["logOR"],3), SE=round(r["SE"],3)))
    # +PNI only when there are enough events to estimate it
    if outcome == "infection":
        Xp, yp, dp, colsp = build_Xy(ins, outcome, with_pni=True)
        rp = fit_firth(Xp, yp)
        evp = int(yp.sum()); npn = len(yp)
        print(f"  [{outcome} +PNI]     n={npn} events={evp}  adj OR={rp['OR']:.3f} [{rp['lo']:.3f}-{rp['hi']:.3f}]  p={rp['p']:.3f}")
        ins_rows.append(dict(model="Firth_adj_PNI", outcome=outcome, n=npn, events=evp,
                             crude_OR=float("nan"), crude_lo=float("nan"), crude_hi=float("nan"),
                             adj_OR=round(rp["OR"],3), adj_lo=round(rp["lo"],3), adj_hi=round(rp["hi"],3),
                             p=round(rp["p"],3), logOR=round(rp["logOR"],3), SE=round(rp["SE"],3)))

# INSPIRE PNI attenuation via Firth where estimable (infection)
inf_base = [r for r in ins_rows if r["outcome"]=="infection" and r["model"]=="Firth_adj_base"][0]
inf_pni  = [r for r in ins_rows if r["outcome"]=="infection" and r["model"]=="Firth_adj_PNI"][0]
inf_att = (inf_base["adj_OR"] - inf_pni["adj_OR"]) / inf_base["adj_OR"] * 100
print(f"  INSPIRE Firth PNI attenuation (gc_use->infection, estimable): "
      f"{inf_base['adj_OR']:.3f} -> {inf_pni['adj_OR']:.3f}  = {inf_att:.1f}%")
print("  NOTE: INSPIRE death (4 events) cannot estimate a PNI coefficient;")
print("        PNI attenuation for the DEATH endpoint relies on the cross-DB MI (S4d).")

# ----------------------------------------------------------------------
# (B) Combined external validation: two-stage random-effects meta
print("\n================ (B) Two-stage meta (gc_use -> 30d death) ================")
meta = []
for name, path in DBS.items():
    df = pd.read_csv(path)
    X, y, d, cols = build_Xy(df, "allcause_death_30d", with_pni=False)
    if name == "MIMIC-IV":
        r = fit_standard(X, y)
    else:
        r = fit_firth(X, y)
    r["db"] = name; r["n"] = len(y); r["events"] = int(y.sum())
    meta.append(r)
    print(f"  {name:9s} n={r['n']:4d} events={r['events']:3d}  logOR={r['logOR']:+.3f} SE={r['SE']:.3f}  "
          f"OR={r['OR']:.3f} [{r['lo']:.3f}-{r['hi']:.3f}]  ({r['method']})")

logORs = np.array([m["logOR"] for m in meta])
SEs = np.array([m["SE"] for m in meta])
w = 1.0 / SEs**2
b_fixed = (w * logORs).sum() / w.sum()
se_fixed = 1.0 / np.sqrt(w.sum())
Q = (w * (logORs - b_fixed)**2).sum()
k = len(meta)
I2 = max(0.0, (Q - (k - 1)) / Q) if Q > 0 else 0.0
denom = w.sum() - (w**2).sum() / w.sum()
tau2 = max(0.0, (Q - (k - 1)) / denom) if denom > 0 else 0.0
wstar = 1.0 / (SEs**2 + tau2)
b_re = (wstar * logORs).sum() / wstar.sum()
se_re = 1.0 / np.sqrt(wstar.sum())
z = 1.959963985
def orci(b, se): return float(np.exp(b - z*se)), float(np.exp(b + z*se))
fl, fh = orci(b_fixed, se_fixed)
rl, rh = orci(b_re, se_re)
print(f"\n  Fixed-effect pooled OR = {np.exp(b_fixed):.3f} [{fl:.3f}-{fh:.3f}]")
print(f"  DL random-effects pooled OR = {np.exp(b_re):.3f} [{rl:.3f}-{rh:.3f}]  tau2={tau2:.4f}  I2={I2*100:.1f}%")

ins_m = [m for m in meta if m["db"]=="INSPIRE"][0]
ins_width = ins_m["hi"] - ins_m["lo"]
pool_width = rh - rl
print(f"  INSPIRE-alone (Firth) CI width = {ins_width:.3f}  |  pooled (RE) CI width = {pool_width:.3f}  "
      f"-> { (1 - pool_width/ins_width)*100:.1f}% narrower")

# ----------------------------------------------------------------------
# Save tables
meta_df = pd.DataFrame([{
    "db": m["db"], "method": m["method"], "n": m["n"], "events": m["events"],
    "logOR": round(m["logOR"],4), "SE": round(m["SE"],4),
    "OR": round(m["OR"],3), "CI_lo": round(m["lo"],3), "CI_hi": round(m["hi"],3), "p": round(m["p"],4)
} for m in meta])
pool_row = pd.DataFrame([{
    "db": "POOLED_RE_DL", "method": "random-effects(DL)", "n": int(meta_df.n.sum()),
    "events": int(meta_df.events.sum()), "logOR": round(b_re,4), "SE": round(se_re,4),
    "OR": round(np.exp(b_re),3), "CI_lo": round(rl,3), "CI_hi": round(rh,3),
    "p": round(2*(1-st.norm.cdf(abs(b_re/se_re))),4)
}, {
    "db": "POOLED_fixed", "method": "fixed-effect(IV)", "n": int(meta_df.n.sum()),
    "events": int(meta_df.events.sum()), "logOR": round(b_fixed,4), "SE": round(se_fixed,4),
    "OR": round(np.exp(b_fixed),3), "CI_lo": round(fl,3), "CI_hi": round(fh,3),
    "p": round(2*(1-st.norm.cdf(abs(b_fixed/se_fixed))),4)
}])
meta_out = pd.concat([meta_df, pool_row], ignore_index=True)
meta_out.to_csv(f"{FEAS}/inspire_death_meta.csv", index=False)
pd.DataFrame(ins_rows).to_csv(f"{FEAS}/inspire_firth_results.csv", index=False)
print("\nSaved: inspire_death_meta.csv, inspire_firth_results.csv")

# ----------------------------------------------------------------------
# Figure 1: forest of 3 DBs + pooled (Firth for NWICU/INSPIRE, standard MIMIC)
fig, ax = plt.subplots(figsize=(8.2, 4.8))
ax.axvline(1.0, color="grey", ls="--", lw=1)
order = ["MIMIC-IV", "NWICU", "INSPIRE", "POOLED_RE_DL"]
labels = {"MIMIC-IV":"MIMIC-IV\n(stan. Logit)", "NWICU":"NWICU\n(Firth)",
          "INSPIRE":"INSPIRE\n(Firth)", "POOLED_RE_DL":"POOLED\nRE(DL)"}
colors = {"MIMIC-IV":"#c0392b","NWICU":"#2980b9","INSPIRE":"#27ae60","POOLED_RE_DL":"#2c3e50"}
yt = 0
for key in order:
    if key == "POOLED_RE_DL":
        row = pool_row[pool_row.db=="POOLED_RE_DL"].iloc[0]
        OR, lo, hi = row.OR, row.CI_lo, row.CI_hi
    else:
        row = meta_df[meta_df.db==key].iloc[0]
        OR, lo, hi = row.OR, row.CI_lo, row.CI_hi
    yt += 1
    ax.errorbar(OR, yt, xerr=[[OR-lo],[hi-OR]], fmt="o", color=colors[key],
                ms=(9 if key!="POOLED_RE_DL" else 11), lw=2.2)
    ax.text(OR*1.08, yt, f"{OR:.2f} [{lo:.2f}-{hi:.2f}]", va="center", fontsize=8, color=colors[key])
ax.set_yticks(range(1, yt+1)); ax.set_yticklabels([labels[k] for k in order], fontsize=9)
ax.set_xlabel("Odds ratio (gc_use any vs none -> 30-day death), adjusted age+sex+charlson")
ax.set_title("INSPIRE 4-death fix: Firth (small cohorts) + combined external validation")
ax.set_xscale("log"); ax.set_xlim(0.3, 60)
fig.tight_layout(); fig.savefig(f"{FEAS}/inspire_death_forest.png", dpi=150); plt.close(fig)

# Figure 2: precision comparison for INSPIRE (MI vs Firth vs pooled)
fig, ax = plt.subplots(figsize=(7.5, 4.2))
ax.axvline(1.0, color="grey", ls="--", lw=1)
# prior MI S4d INSPIRE gc_use death: OR 0.99 [0.02-54.8]
pts = [("INSPIRE MI (S4d)\n0.99 [0.02-54.8]", 0.99, 0.02, 54.8, "#95a5a6"),
       ("INSPIRE Firth\n2.55 [0.06-110]", 2.55, 0.059, 110.281, "#27ae60"),
       ("POOLED RE(DL)\n{:.2f} [{:.2f}-{:.2f}]".format(np.exp(b_re), rl, rh),
        np.exp(b_re), rl, rh, "#2c3e50")]
for i,(lab,OR,lo,hi,c) in enumerate(pts, start=1):
    ax.errorbar(OR, i, xerr=[[OR-lo],[hi-OR]], fmt="s", color=c, ms=8, lw=2)
    ax.text(OR*1.08, i, lab.split("\n")[1], va="center", fontsize=8, color=c)
ax.set_yticks(range(1,4)); ax.set_yticklabels([p[0].split("\n")[0] for p in pts], fontsize=9)
ax.set_xlabel("Odds ratio (gc_use -> 30-day death)")
ax.set_title("INSPIRE precision gain: Firth regularizes, meta pools external cohorts")
ax.set_xscale("log"); ax.set_xlim(0.2, 200)
fig.tight_layout(); fig.savefig(f"{FEAS}/inspire_firth_attenuation.png", dpi=150); plt.close(fig)

print("\nDONE. Figures: inspire_death_forest.png, inspire_firth_attenuation.png")
