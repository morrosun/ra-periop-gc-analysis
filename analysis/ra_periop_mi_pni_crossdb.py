# -*- coding: utf-8 -*-
import os as _os
ROOT = _os.path.dirname(_os.path.abspath(__file__))
"""
RA perioperative - Cross-DB multiple imputation for PNI missingness
Reproduces the MIMIC PNI-confounding analysis (ra_periop_mi_pni.py) in
NWICU and INSPIRE to cross-validate the conclusion that PNI explains
part of the GC -> death association.

For each database we:
  - Impute lab block (albumin, lymphocytes_per_ul, creatinine) + predictors
    + outcome jointly under MAR (sklearn IterativeImputer, sample_posterior).
  - Derive PNI = 10*albumin(g/dL) + 0.005*lymphocytes_per_uL.
  - Fit per imputation (M=50):
        Model A: outcome ~ GC_exposure + age + sex + charlson            (no PNI)
        Model B: outcome ~ GC_exposure + age + sex + charlson + PNI      (with PNI)
  - Rubin-pool; % attenuation of GC-effect OR after adding PNI.
  - Convergence (M=5/10/20/50) and complete-case / full-data references.

Exposure: MIMIC & NWICU use dose strata (gc_low/gc_high vs none);
          INSPIRE has no dose -> binary gc_use.
Outcomes: allcause_death_30d (primary) and infection (robustness).
"""
import numpy as np, pandas as pd
import statsmodels.api as sm
from scipy import stats as st
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FEAS = _os.environ.get("RA_GC_DATA", _os.path.join(ROOT, "data"))  # extracted CSVs dir (NOT in repo; see README)
M = 50

DBS = [
    # name, csv, outcome_primary, binary_gc, note
    ("MIMIC-IV", f"{FEAS}/exports/analytic_mimiciv.csv",  "allcause_death_30d", False, "dose strata (ref=none)"),
    ("NWICU",    f"{FEAS}/exports/analytic_nwicu.csv",    "allcause_death_30d", False, "dose strata (ref=none)"),
    ("INSPIRE",  f"{FEAS}/exports/analytic_inspire.csv",  "allcause_death_30d", True,  "binary gc_use (no dose)"),
]
OUTCOMES = ["allcause_death_30d", "infection"]

# ----------------------------------------------------------------------
def run_mi(path, outcome, binary_gc, seed0=20260826):
    d = pd.read_csv(path)
    d["sex_m"] = (d["sex"] == "M").astype(int)
    if not binary_gc:
        d["gc_strata_num"] = np.where(d["gc_use"] == 0, 0,
                              np.where(d["gc_dose_pred_eq_mg"] <= 100, 1, 2))
        d["gc_low"]  = (d["gc_strata_num"] == 1).astype(int)
        d["gc_high"] = (d["gc_strata_num"] == 2).astype(int)
        exp_terms = ["gc_low", "gc_high"]
        exp_label = "gc_high"
    else:
        d["gc_use_b"] = d["gc_use"].astype(int)
        exp_terms = ["gc_use_b"]
        exp_label = "gc_use_b"
    y = d[outcome].astype(int).values
    events = int(y.sum()); n = len(d)

    age_mean   = d["age"].mean()
    charl_mean = d["charlson"].mean()
    pni_obs    = d["pni"].dropna(); pni_mean = pni_obs.mean() if len(pni_obs) else 0.0
    d["age_c"]       = d["age"] - age_mean
    d["charlson_c"]  = d["charlson"] - charl_mean

    pni_cov = ["age", "sex_m", "charlson", "gc_use", "los_days", "infection",
               "major_comp", "inhosp_death", "btsdmard_use", outcome]
    if not binary_gc:
        pni_cov += ["gc_dose_pred_eq_mg", "gc_strata_num"]
    feat = ["albumin", "lymphocytes_per_ul", "creatinine"] + pni_cov
    # drop fully-missing columns (e.g. INSPIRE creatinine is all NULL) so
    # IterativeImputer keeps them out of the imputed array shape
    keep = [c for c in feat if d[c].notna().any()]
    Ximp = d[keep].copy()
    if "gc_dose_pred_eq_mg" in Ximp:
        Ximp["gc_dose_pred_eq_mg"] = Ximp["gc_dose_pred_eq_mg"].fillna(0.0)
        Ximp["gc_strata_num"]      = Ximp["gc_strata_num"].fillna(0).astype(float)
    feat = keep

    imputed = []
    for m in range(M):
        imp = IterativeImputer(estimator=BayesianRidge(), max_iter=25,
                               sample_posterior=True, random_state=seed0 + m,
                               n_nearest_features=None, skip_complete=False,
                               imputation_order="ascending")
        arr = imp.fit_transform(Ximp.values)
        di = pd.DataFrame(arr, columns=feat)
        di["pni"] = 10.0 * di["albumin"] + 0.005 * di["lymphocytes_per_ul"]
        di["age_c"]       = di["age"] - age_mean
        di["charlson_c"]  = di["charlson"] - charl_mean
        di["pni_c"]       = di["pni"] - pni_mean
        di["gc_low"]  = d["gc_low"].values if not binary_gc else 0
        di["gc_high"] = d["gc_high"].values if not binary_gc else 0
        di["gc_use_b"]= d["gc_use_b"].values if binary_gc else 0
        di["sex_m"]   = d["sex_m"].values
        imputed.append(di)

    def fit_logit(df, with_pni):
        cols = exp_terms + ["age_c", "sex_m", "charlson_c"]
        if with_pni:
            cols = cols + ["pni_c"]
        X = sm.add_constant(df[cols])
        try:
            res = sm.Logit(y, X).fit(disp=0, maxiter=200, method="bfgs")
            if not np.all(np.isfinite(res.params)) or not np.all(np.isfinite(np.diag(res.cov_params()))):
                raise np.linalg.LinAlgError("bad")
            return res
        except Exception:
            return None

    terms_A = exp_terms
    terms_B = exp_terms + ["pni_c"]
    A_beta, A_var = {t: [] for t in terms_A}, {t: [] for t in terms_A}
    B_beta, B_var = {t: [] for t in terms_B}, {t: [] for t in terms_B}
    A_or_h, B_or_h = [], []
    valid = 0
    for di in imputed:
        ra = fit_logit(di, False); rb = fit_logit(di, True)
        if ra is None or rb is None:
            continue
        valid += 1
        for t in terms_A:
            A_beta[t].append(ra.params[t]); A_var[t].append(ra.cov_params().loc[t, t])
        for t in terms_B:
            B_beta[t].append(rb.params[t]); B_var[t].append(rb.cov_params().loc[t, t])
        A_or_h.append(np.exp(ra.params[exp_label]))
        B_or_h.append(np.exp(rb.params[exp_label]))

    def rubin(betas, vars_):
        betas = np.asarray(betas); vars_ = np.asarray(vars_)
        m = len(betas); beta_bar = betas.mean()
        W = vars_.mean(); B = ((betas - beta_bar)**2).sum() / (m - 1)
        T = W + B + (B / m); se = np.sqrt(T)
        r = (B + B/m) / W; df = (m - 1) * (1 + r)**2
        return beta_bar, se, df

    def fmt(bb, se, df):
        tc = st.t.ppf(0.975, df); lo = bb - tc*se; hi = bb + tc*se
        return np.exp(bb), (np.exp(lo), np.exp(hi)), 2*(1 - st.t.cdf(abs(bb/se), df)), df

    pooled_rows = []
    for t in terms_A:
        bb, se, df = rubin(A_beta[t], A_var[t]); OR, CI, p, df = fmt(bb, se, df)
        pooled_rows.append(dict(model="A_noPNI", term=t, OR=round(OR,3), CI=f"{CI[0]:.3f}-{CI[1]:.3f}", p=round(p,4), df=round(df,1)))
    for t in terms_B:
        bb, se, df = rubin(B_beta[t], B_var[t]); OR, CI, p, df = fmt(bb, se, df)
        pooled_rows.append(dict(model="B_PNI", term=t, OR=round(OR,3), CI=f"{CI[0]:.3f}-{CI[1]:.3f}", p=round(p,4), df=round(df,1)))

    A_OR = float(pooled_rows[0]["OR"]) if not binary_gc else None
    # locate exp_label row in B
    B_row = [r for r in pooled_rows if r["model"] == "B_PNI" and r["term"] == exp_label][0]
    B_OR = B_row["OR"]
    A_row = [r for r in pooled_rows if r["model"] == "A_noPNI" and r["term"] == exp_label][0]
    A_OR = A_row["OR"]
    att = (A_OR - B_OR) / A_OR * 100

    # references: full-data A (no pni, complete cases on predictors used) & complete-case B
    def fit_ref(with_pni, idx=None):
        cols = exp_terms + ["age_c", "sex_m", "charlson_c"]
        if with_pni:
            cols += ["pni_c"]
        if idx is None:
            dd = d.copy(); yy = y
        else:
            dd = d.iloc[idx].copy(); yy = y[idx]
        dd["pni_c"] = dd["pni"] - pni_mean
        X = sm.add_constant(dd[cols]); res = sm.Logit(yy, X).fit(disp=0, maxiter=200)
        return res
    ra_full = fit_ref(False)
    OR_A_full = float(np.exp(ra_full.params[exp_label]))
    cc_mask = d["pni"].notna().values
    rb_cc = fit_ref(True, np.where(cc_mask)[0])
    OR_B_cc = float(np.exp(rb_cc.params[exp_label]))
    n_cc = int(cc_mask.sum())

    # convergence
    conv_rows = []
    for k in [5, 10, 20, 50]:
        bb, se, df = rubin(B_beta[exp_label][:k], B_var[exp_label][:k]); OR, CI, p, df = fmt(bb, se, df)
        conv_rows.append(dict(M=k, OR_B_gc=round(OR,3), CI=f"{CI[0]:.3f}-{CI[1]:.3f}", p=round(p,4)))

    return dict(db=path.split("analytic_")[1].split(".")[0], outcome=outcome,
                binary_gc=binary_gc, exp_label=exp_label, events=events, n=n,
                valid=valid, pni_cov=round(d["pni"].notna().mean()*100,1),
                A_OR=A_OR, B_OR=B_OR, att=round(att,1),
                OR_A_full=round(OR_A_full,3), OR_B_cc=round(OR_B_cc,3), n_cc=n_cc,
                pooled=pd.DataFrame(pooled_rows), conv=pd.DataFrame(conv_rows),
                A_or_h=A_or_h, B_or_h=B_or_h,
                B_CI=B_row["CI"], A_CI=A_row["CI"], pni_coef_B=None)

# ----------------------------------------------------------------------
# run all (each DB x each outcome)
results = {}
for (name, path, _, binary_gc, note) in DBS:
    for outcome in OUTCOMES:
        key = (name, outcome)
        print(f"\n########## {name} | {outcome} | {note} ##########")
        r = run_mi(path, outcome, binary_gc)
        results[key] = r
        print(f"  n={r['n']} events={r['events']} valid_imp={r['valid']}/{M} pni_cov={r['pni_cov']}%")
        print(f"  GC-effect OR: A(noPNI)={r['A_OR']:.3f} [{r['A_CI']}]  B(PNI)={r['B_OR']:.3f} [{r['B_CI']}]  attenuation={r['att']:.1f}%")
        print(f"  ref full-data A={r['OR_A_full']:.3f} | complete-case B(n={r['n_cc']})={r['OR_B_cc']:.3f}")

# ----------------------------------------------------------------------
# save per-DB pooled + convergence
for (name, outcome), r in results.items():
    tag = f"{name}_{outcome}"
    r["pooled"].to_csv(f"{FEAS}/crossdb_pooled_{tag}.csv", index=False)
    r["conv"].to_csv(f"{FEAS}/crossdb_conv_{tag}.csv", index=False)

# combined cross-DB summary
summ = []
for (name, outcome), r in results.items():
    summ.append(dict(db=name, outcome=outcome, exposure=("gc_use" if r["binary_gc"] else "gc_high vs none"),
                     n=r["n"], events=r["events"], pni_cov_pct=r["pni_cov"],
                     A_OR_noPNI=r["A_OR"], B_OR_PNI=r["B_OR"], attenuation_pct=r["att"],
                     ref_fullA=r["OR_A_full"], ref_CC_B=r["OR_B_cc"], n_CC=r["n_cc"],
                     valid_imp=r["valid"]))
pd.DataFrame(summ).to_csv(f"{FEAS}/crossdb_summary.csv", index=False)
print("\n=== CROSS-DB SUMMARY ===")
print(pd.DataFrame(summ).to_string(index=False))

# ----------------------------------------------------------------------
# Figures: combined forest (per outcome) + attenuation bars
for outcome in OUTCOMES:
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.axvline(1.0, color="grey", ls="--", lw=1)
    yt = 0; labels = []
    colors = {"MIMIC-IV": "#c0392b", "NWICU": "#2980b9", "INSPIRE": "#27ae60"}
    for (name, oc), r in results.items():
        if oc != outcome:
            continue
        yt += 1; labels.append(f"{name}\nA(noPNI)")
        ax.errorbar(r["A_OR"], yt, xerr=[[r["A_OR"]-float(r["A_CI"].split('-')[0])], [float(r["A_CI"].split('-')[1])-r["A_OR"]]],
                    fmt="o", color=colors.get(name,"#333"), ms=8, lw=2)
        yt += 1; labels.append(f"{name}\nB(PNI)")
        ax.errorbar(r["B_OR"], yt, xerr=[[r["B_OR"]-float(r["B_CI"].split('-')[0])], [float(r["B_CI"].split('-')[1])-r["B_OR"]]],
                    fmt="s", color=colors.get(name,"#333"), ms=8, lw=2)
    ax.set_yticks(range(1, yt+1)); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Odds ratio (GC exposure -> outcome), Rubin-pooled M=50, adjusted")
    ax.set_title(f"Cross-DB GC-effect OR with/without PNI imputation: {outcome}")
    ax.set_xscale("log"); ax.set_xlim(0.2, 12)
    fig.tight_layout(); fig.savefig(f"{FEAS}/crossdb_forest_{outcome}.png", dpi=150); plt.close(fig)

# attenuation bar (death + infection)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
for ax, outcome in zip(axes, OUTCOMES):
    names = []; A=[]; B=[]; atts=[]
    for (name, oc), r in results.items():
        if oc != outcome: continue
        names.append(name); A.append(r["A_OR"]); B.append(r["B_OR"]); atts.append(r["att"])
    x = np.arange(len(names)); w=0.35
    ax.bar(x-w/2, A, w, label="A (no PNI)", color="#34495e")
    ax.bar(x+w/2, B, w, label="B (PNI)", color="#c0392b")
    for i,(a,b,at) in enumerate(zip(A,B,atts)):
        ax.text(i-w/2, a+0.05, f"{a:.2f}", ha="center", fontsize=8)
        ax.text(i+w/2, b+0.05, f"{b:.2f}", ha="center", fontsize=8)
        ax.text(i, max(a,b)+0.3, f"-{at:.0f}%", ha="center", fontsize=8, color="#c0392b", fontweight="bold")
    ax.axhline(1.0, color="grey", ls="--")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("GC-effect OR (vs none / vs no-GC)")
    ax.set_title(f"{outcome}\nPNI attenuation (cross-DB)")
    ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{FEAS}/crossdb_attenuation.png", dpi=150); plt.close(fig)

print("\nDONE. Cross-DB MI outputs written.")
