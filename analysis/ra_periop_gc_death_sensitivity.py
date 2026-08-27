# -*- coding: utf-8 -*-
import os as _os
ROOT = _os.path.dirname(_os.path.abspath(__file__))
"""
RA perioperative (refinement): GC exposure -> death
  1) Competing-risk analysis (Fine-Gray subdistribution hazard) for IN-HOSPITAL death
     time = los_days ; event = inhosp_death ; competing = discharge alive
     Klein-Andersen (2004) data augmentation + lifelines CoxPHFitter (weights, robust, cluster)
  2) Cumulative Incidence Function (Aalen-Johansen) by GC strata + Gray-type global test (FG LRT)
  3) Sensitivity analyses for GC -> death:
       S1 in-hospital death (logistic, alt outcome)
       S2 composite = death OR major_comp (logistic)
       S3 GC continuous dose (per 100 mg pred-eq) among users
       S4 GC-use (any) vs none (binary)
       S5 30-day death adjusted WITH vs WITHOUT PNI (robustness to missing PNI)
     + E-value for unmeasured confounding on GC-high OR
NOTE: 'emergency' dropped from all final models (near-zero variance: ~95% = 1).
Inputs : exports/analytic_mimiciv.csv
Outputs: gc_death_final_model.csv, gc_death_competing_risk.csv, gc_death_sensitivity.csv,
         cif_gc_death.png, sensitivity_forest_gcdeath.png, evalue_gcdeath.csv
"""
import os, warnings, math
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats as sci
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
warnings.filterwarnings("ignore")

FEAS = os.path.dirname(os.path.abspath(__file__))
EXP  = os.path.join(FEAS, "exports")
OUT  = FEAS

d = pd.read_csv(os.path.join(EXP, "analytic_mimiciv.csv"))
d["sex_m"] = (d.sex == "M").astype(int)
d["gc_strata"] = np.where(d.gc_use == 0, "none",
                  np.where(d.gc_dose_pred_eq_mg <= 100, "low", "high"))
d["gc_strata"] = pd.Categorical(d.gc_strata, ["none", "low", "high"])
d["dmard_class"] = pd.Categorical(d.dmard_class, ["none", "csDMARD", "TNFi", "ILi", "JAKi"])
d["btsdmard"] = d.btsdmard_use.astype(int)
d["gc_dose_100"] = d.gc_dose_pred_eq_mg / 100.0
COV = "age + sex_m + charlson"   # FINAL covariate set (emergency removed)

# ---------- helpers ----------
def logit(frame, outcome, formula):
    fr = frame.dropna(subset=[outcome]).copy()
    try:
        m = smf.logit(formula, data=fr).fit(disp=0)
    except Exception as e:
        return None, str(e)
    out = {}
    for term in m.params.index:
        if term == "Intercept":
            continue
        b = m.params[term]; se = m.bse[term]; p = m.pvalues[term]
        out[term] = (round(np.exp(b), 3), round(np.exp(b - 1.96 * se), 3),
                     round(np.exp(b + 1.96 * se), 3), (f"{p:.2e}" if p < 0.001 else round(p, 3)))
    out["_n"] = int(fr.shape[0]); out["_n_event"] = int(fr[outcome].sum())
    return m, out

def get_hi(res):
    """pull (OR, lo, hi, p) for the HIGH GC stratum from a logistic res dict."""
    if res is None:
        return None
    for k, v in res.items():
        if k.startswith("_"):
            continue
        if "[T.high]" in k:
            return v
    return None

def get_lo(res):
    if res is None:
        return None
    for k, v in res.items():
        if k.startswith("_"):
            continue
        if "[T.low]" in k:
            return v
    return None

def evalue(rr, lo):
    """VanderWeele & Ding 2017 E-value for a risk-ratio (use OR as approx)."""
    ev_pt = rr + math.sqrt(rr * (rr - 1)) if rr > 1 else 1.0
    ev_lo = lo + math.sqrt(lo * (lo - 1)) if lo > 1 else 1.0
    return round(ev_pt, 2), round(ev_lo, 2)

# =====================================================================
# 1) FINE-GRAY COMPETING-RISK for IN-HOSPITAL DEATH
# =====================================================================
cr = d.dropna(subset=["los_days", "age", "sex_m", "charlson"]).copy()
cr = cr[cr.los_days > 0].reset_index(drop=True)
cr["subject_id_int"] = np.arange(len(cr))            # unique row id for clustering
cr["status"] = np.where(cr.inhosp_death == 1, 1, 2)  # 1=death(interest), 2=discharge(competing)
cr["gc_low"] = (cr.gc_strata == "low").astype(int)
cr["gc_high"] = (cr.gc_strata == "high").astype(int)

def km_censor_survival(time, status):
    """G(t): KM survival of the CENSORING distribution.
    For G, a 'censoring event' occurs when status in {2,0} (competing or true censor);
    subjects with status==1 (event of interest) are treated as continuing (right-censored at their time)."""
    t = np.asarray(time, float); s = np.asarray(status, int)
    # events that reduce G: status != 1
    ev = np.where(s != 1)[0]
    if len(ev) == 0:
        return lambda x: np.ones_like(np.atleast_1d(x), float)
    times = np.sort(np.unique(t[ev]))
    # risk set size n(u)=#(time>=u); d(u)=#(status!=1 & time==u)
    G = [1.0]; tgrid = [0.0]
    for u in times:
        n = np.sum(t >= u)
        dd = np.sum((s != 1) & (t == u))
        if n > 0:
            G.append(G[-1] * (1 - dd / n))
            tgrid.append(u)
    G = np.array(G); tgrid = np.array(tgrid)
    def Gfun(x):
        x = np.atleast_1d(np.asarray(x, float))
        out = np.empty_like(x, float)
        for i, xx in enumerate(x):
            out[i] = G[tgrid <= xx].max() if np.any(tgrid <= xx) else 1.0
        return out if out.size > 1 else out[0]
    return Gfun

Gfun = km_censor_survival(cr.los_days.values, cr.status.values)

# Klein-Andersen augmentation
tau = np.sort(np.unique(cr.los_days.values[cr.status.isin([1, 2])]))
tau = np.concatenate(([0.0], tau))
rows = []
for i in range(len(cr)):
    ti = cr.los_days.values[i]; si = cr.status.values[i]
    covs = dict(subject_id_int=cr.subject_id_int.values[i],
                age=cr.age.values[i], sex_m=cr.sex_m.values[i], charlson=cr.charlson.values[i],
                gc_low=(cr.gc_strata.values[i] == "low"),
                gc_high=(cr.gc_strata.values[i] == "high"))
    for j in range(1, len(tau)):
        if tau[j] > ti + 1e-9:
            break
        w = Gfun(tau[j]) / Gfun(tau[j - 1])
        status_row = 1 if (si == 1 and abs(ti - tau[j]) < 1e-9) else 0
        rows.append(dict(start=tau[j - 1], stop=tau[j], fg_status=status_row,
                         fg_weight=w, **covs))
aug = pd.DataFrame(rows)
# fit Fine-Gray (subdistribution) via weighted Cox with robust+cluster
cph = CoxPHFitter()
try:
    cph.fit(aug, duration_col="stop", event_col="fg_status", entry_col="start",
            weights_col="fg_weight", cluster_col="subject_id_int", robust=True)
    fg_ok = True
    fg_sum = cph.summary
except Exception as e:
    fg_ok = False
    fg_err = str(e)

# Fine-Gray global test (LRT, 2 df) for GC strata
if fg_ok:
    full_ll = cph.log_likelihood_
    cph_null = CoxPHFitter()
    cph_null.fit(aug.drop(columns=["gc_low", "gc_high"]),
                 duration_col="stop", event_col="fg_status", entry_col="start",
                 weights_col="fg_weight", cluster_col="subject_id_int", robust=True)
    LRT = 2 * (full_ll - cph_null.log_likelihood_)
    fg_global_p = float(sci.chi2.sf(LRT, df=2))
    fg_global = (round(LRT, 3), round(fg_global_p, 4))
else:
    fg_global = None

print("=== Fine-Gray competing-risk (in-hospital death) ===")
if fg_ok:
    for term in ["gc_low", "gc_high"]:
        if term in fg_sum.index:
            r = fg_sum.loc[term]
            print(f"  {term:8s} sHR={np.exp(r['coef']):.3f}  CI={np.exp(r['coef']-1.96*r['se(coef)']):.3f}-{np.exp(r['coef']+1.96*r['se(coef)']):.3f}  p={r['p']:.2e}")
    print(f"  Global test (GC strata, 2 df): LRT={fg_global[0]} p={fg_global[1]}")
else:
    print("  Fine-Gray fit failed:", fg_err)

# ---------------------------------------------------------------------
# 1b) STANDARD COX PH for in-hospital death (discharge-alive = censoring)
#     -> contrasts with the attenuated Fine-Gray subdistribution HR
# ---------------------------------------------------------------------
cox_df = cr[["los_days", "inhosp_death", "gc_low", "gc_high", "age", "sex_m", "charlson"]].copy()
cox_df["event"] = cox_df.inhosp_death.astype(int)
cph_std = CoxPHFitter()
cph_std.fit(cox_df, duration_col="los_days", event_col="event", robust=True)
std_sum = cph_std.summary
print("\n=== Standard Cox PH (in-hosp death; discharge = censoring) ===")
for term in ["gc_low", "gc_high"]:
    if term in std_sum.index:
        rr = std_sum.loc[term]
        print(f"  {term:8s} HR={np.exp(rr['coef']):.3f}  CI={np.exp(rr['coef']-1.96*rr['se(coef)']):.3f}-"
              f"{np.exp(rr['coef']+1.96*rr['se(coef)']):.3f}  p={rr['p']:.2e}")

# =====================================================================
# 2) CIF (Aalen-Johansen) by GC strata + Gray-type test (use FG global above)
# =====================================================================
def cif_by_strata(cr, strata_col, max_t=60):
    """Return time grid and CIF (death) per stratum with pointwise 95% CI."""
    tmax = min(max_t, cr.los_days.max())
    grid = np.arange(0, tmax + 1)
    out = {}
    for s in ["none", "low", "high"]:
        sub = cr[cr[strata_col] == s]
        if len(sub) == 0:
            continue
        t = sub.los_days.values; st = sub.status.values
        # overall event-free survival S (all events: death or discharge)
        ut = np.sort(np.unique(t))
        S = {}; surv = 1.0
        for u in ut:
            n = np.sum(t >= u); d = np.sum(t == u)
            surv *= (1 - d / n) if n > 0 else 1.0
            S[u] = surv
        # CIF death
        cif = []; var = []
        cum = 0.0; varcum = 0.0
        last_S = 1.0
        # iterate unique times sorted
        for u in ut:
            n = np.sum(t >= u)
            d_death = np.sum((st == 1) & (t == u))
            s_minus = S[u - 1] if (u - 1) in S else 1.0
            # S at previous time
            prev = 1.0
            for v in ut:
                if v < u:
                    prev = S[v]
            inc = prev * (d_death / n) if n > 0 else 0.0
            cum += inc
            # variance (Aalen-Johansen, simple form)
            if n > 0 and d_death > 0:
                varcum += (prev ** 2) * (d_death / n) * (1 - d_death / n) / n
            cif.append(cum); var.append(varcum)
        # interpolate onto grid
        cif_grid = np.interp(grid, ut, cif, left=0.0, right=cif[-1])
        se_grid = np.sqrt(np.interp(grid, ut, var, left=0.0, right=var[-1]))
        lo = np.clip(cif_grid - 1.96 * se_grid, 0, 1)
        hi = np.clip(cif_grid + 1.96 * se_grid, 0, 1)
        out[s] = (cif_grid, lo, hi)
    return grid, out

grid, cif_res = cif_by_strata(cr, "gc_strata", max_t=60)

plt.figure(figsize=(7.2, 4.6))
colors = {"none": "#2c7fb8", "low": "#fdae61", "high": "#d7301f"}
for s in ["none", "low", "high"]:
    if s in cif_res:
        cif, lo, hi = cif_res[s]
        n = int((cr.gc_strata == s).sum())
        plt.step(grid, cif, where="post", color=colors[s], lw=2, label=f"{s} (n={n})")
        plt.fill_between(grid, lo, hi, step="post", color=colors[s], alpha=0.15)
plt.xlabel("Hospital length of stay (days)")
plt.ylabel("Cumulative incidence of in-hospital death")
plt.title("Competing-risk CIF: GC exposure vs in-hospital death (MIMIC-IV)")
plt.ylim(0, max(0.05, grid_max := np.max([c[0].max() for c in cif_res.values()]) * 1.15))
plt.legend(loc="upper left", fontsize=9)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "cif_gc_death.png"), dpi=150)
plt.close()
print(f"  CIF at 30 days: " + ", ".join(
    f"{s}={cif_res[s][0][30]:.3f}" for s in cif_res))

# =====================================================================
# 3) SENSITIVITY ANALYSES for GC -> death
# =====================================================================
sens_rows = []
def add_sens(label, outcome, formula, desc, res):
    if res is None:
        sens_rows.append(dict(analysis=label, exposure="gc_high vs none", outcome=outcome,
                              desc=desc, OR="", CI95="", p="FIT_ERR", n="", events=""))
        return
    v = get_hi(res)
    if v is not None:
        orr, lo, hi, p = v
        sens_rows.append(dict(analysis=label, exposure="gc_high vs none", outcome=outcome,
                              desc=desc, OR=orr, CI95=f"{lo}-{hi}", p=p,
                              n=res["_n"], events=res["_n_event"]))

# S1: in-hospital death (logistic, alt outcome)
_, r = logit(cr, "inhosp_death", f"inhosp_death ~ C(gc_strata, Treatment('none')) + {COV}")
add_sens("S1 in-hosp death (logistic)", "inhosp_death", "", "alt outcome", r)
# S2: composite death OR major_comp
cr["comp"] = ((cr.inhosp_death == 1) | (cr.major_comp == 1)).astype(int)
_, r = logit(cr, "comp", f"comp ~ C(gc_strata, Treatment('none')) + {COV}")
add_sens("S2 composite (death/major_comp)", "comp", "", "composite outcome", r)
# S3: GC continuous (per 100 mg) among users -> death 30d
users = cr[cr.gc_use == 1]
_, r = logit(users, "allcause_death_30d", f"allcause_death_30d ~ gc_dose_100 + {COV}")
if r and "gc_dose_100" in r:
    orr, lo, hi, p = r["gc_dose_100"]
    sens_rows.append(dict(analysis="S3 GC dose (per 100mg)", exposure="gc_dose_100", outcome="allcause_death_30d",
                          desc="continuous among users", OR=orr, CI95=f"{lo}-{hi}", p=p,
                          n=r["_n"], events=r["_n_event"]))
# S4: GC-use (any) vs none -> 30d death
cr["gc_any"] = cr.gc_use.astype(int)
_, r = logit(cr, "allcause_death_30d", f"allcause_death_30d ~ gc_any + {COV}")
if r and "gc_any" in r:
    orr, lo, hi, p = r["gc_any"]
    sens_rows.append(dict(analysis="S4 GC-use (any) vs none", exposure="gc_any", outcome="allcause_death_30d",
                          desc="binary exposure", OR=orr, CI95=f"{lo}-{hi}", p=p, n=r["_n"], events=r["_n_event"]))
# S5: 30d death WITH PNI vs WITHOUT (robustness to missing PNI)
_, r0 = logit(cr, "allcause_death_30d", f"allcause_death_30d ~ C(gc_strata, Treatment('none')) + {COV}")
add_sens("S5a 30d death (no PNI)", "allcause_death_30d", "", "base covars", r0)
crp = cr.dropna(subset=["pni"]).copy()
_, r1 = logit(crp, "allcause_death_30d", f"allcause_death_30d ~ C(gc_strata, Treatment('none')) + {COV} + pni")
if r1:
    v = get_hi(r1)
    if v:
        orr, lo, hi, p = v
        sens_rows.append(dict(analysis="S5b 30d death (+PNI)", exposure="gc_high vs none",
                              outcome="allcause_death_30d", desc="add PNI",
                              OR=orr, CI95=f"{lo}-{hi}", p=p, n=r1["_n"], events=r1["_n_event"]))

sens_df = pd.DataFrame(sens_rows)
sens_df.to_csv(os.path.join(OUT, "gc_death_sensitivity.csv"), index=False)
print("\n=== Sensitivity analyses (GC high vs none; OR 95% CI) ===")
print(sens_df.to_string(index=False))

# E-value for GC-high OR (primary 30d death & in-hospital death)
ev_rows = []
# primary 30d death GC-high vs none
v30 = get_hi(r0)
if v30:
    pt, loev = evalue(v30[0], v30[1])
    ev_rows.append(dict(effect="GC high vs none -> 30d death (adj)", OR=v30[0], CI95=f"{v30[1]}-{v30[2]}",
                        Evalue_point=pt, Evalue_CI=loev))
# in-hospital death GC-high vs none (S1)
vih = get_hi(r)
if vih:
    pt, loev = evalue(vih[0], vih[1])
    ev_rows.append(dict(effect="GC high vs none -> in-hosp death (adj)", OR=vih[0], CI95=f"{vih[1]}-{vih[2]}",
                        Evalue_point=pt, Evalue_CI=loev))
ev_df = pd.DataFrame(ev_rows)
ev_df.to_csv(os.path.join(OUT, "evalue_gcdeath.csv"), index=False)
print("\n=== E-value (unmeasured confounding needed to explain away) ===")
print(ev_df.to_string(index=False))

# =====================================================================
# Final model tables (without emergency)
# =====================================================================
final_rows = []
def addfinal(model, outcome, desc, res):
    if res is None:
        return
    for term, v in res.items():
        if term.startswith("_"):
            continue
        if "gc_strata" in term or term in ("gc_high", "gc_low"):
            orr, lo, hi, p = v
            final_rows.append(dict(model=model, outcome=outcome, term=term, OR=orr,
                                   CI95=f"{lo}-{hi}", p=p, n=res["_n"], events=res["_n_event"]))

_, r30 = logit(cr, "allcause_death_30d", f"allcause_death_30d ~ C(gc_strata, Treatment('none')) + {COV}")
addfinal("GC strata (adj, no emergency)", "allcause_death_30d (PRIMARY)", "ref=none; age,sex,charlson", r30)
_, rih = logit(cr, "inhosp_death", f"inhosp_death ~ C(gc_strata, Treatment('none')) + {COV}")
addfinal("GC strata (adj, no emergency)", "inhosp_death", "ref=none; age,sex,charlson", rih)
final_df = pd.DataFrame(final_rows)
final_df.to_csv(os.path.join(OUT, "gc_death_final_model.csv"), index=False)
print("\n=== Final GC-death models (emergency removed) ===")
print(final_df.to_string(index=False))

# Fine-Gray result table
if fg_ok:
    fg_rows = []
    for term in ["gc_low", "gc_high"]:
        if term in fg_sum.index:
            rr = fg_sum.loc[term]
            fg_rows.append(dict(term=term, sHR=round(np.exp(rr["coef"]), 3),
                                CI95=f"{np.exp(rr['coef']-1.96*rr['se(coef)']):.3f}-{np.exp(rr['coef']+1.96*rr['se(coef)']):.3f}",
                                p=round(rr["p"], 4)))
    if fg_global:
        fg_rows.append(dict(term="GLOBAL (2 df)", sHR="", CI95="", p=fg_global[1]))
    fg_df = pd.DataFrame(fg_rows)
    fg_df.to_csv(os.path.join(OUT, "gc_death_competing_risk.csv"), index=False)
    print("\n=== Fine-Gray competing-risk table ===")
    print(fg_df.to_string(index=False))

# =====================================================================
# Sensitivity forest plot (GC high vs none across definitions)
# =====================================================================
plot_df = sens_df[sens_df.exposure == "gc_high vs none"].copy()
if plot_df.empty and len(sens_df) > 0:
    plot_df = sens_df.copy()
fig, ax = plt.subplots(figsize=(7.6, 4.4))
y = np.arange(len(plot_df))
ors = plot_df.OR.astype(float).values
los = [float(x.split("-")[0]) for x in plot_df.CI95]
his = [float(x.split("-")[1]) for x in plot_df.CI95]
ax.errorbar(ors, y, xerr=[np.array(ors)-np.array(los), np.array(his)-np.array(ors)],
            fmt="o", color="#1f78b4", ecolor="#999999", capsize=4, ms=6)
ax.axvline(1, color="k", lw=1, ls="--")
ax.set_yticks(y)
ax.set_yticklabels(plot_df.analysis, fontsize=9)
ax.set_xlabel("Odds ratio (GC high vs none), 95% CI")
ax.set_title("GC high-dose vs none -> death: sensitivity across outcome/exposure definitions")
ax.set_xscale("log")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "sensitivity_forest_gcdeath.png"), dpi=150)
plt.close()
print("\nDONE. Figures + tables written to", OUT)
