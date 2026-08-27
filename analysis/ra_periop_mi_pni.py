# -*- coding: utf-8 -*-
import os as _os
ROOT = _os.path.dirname(_os.path.abspath(__file__))
"""
RA perioperative - Multiple Imputation (MICE-style) for PNI missingness
Re-assesses the confounding strength of S5b (adding PNI to GC -> death model).

Method
------
- Impute lab block (albumin, lymphocytes_per_ul, creatinine) jointly with
  outcome + predictors under MAR using sklearn IterativeImputer (BayesianRidge,
  sample_posterior=True for proper between-imputation variance).
- Derive PNI = 10*albumin(g/dL) + 0.005*lymphocytes_per_uL  (matches augment_mimic.sql).
- Fit, on each of M=50 imputed datasets:
    Model A: allcause_death_30d ~ gc_low + gc_high + age + sex + charlson   (no PNI)
    Model B: allcause_death_30d ~ gc_low + gc_high + age + sex + charlson + PNI (with PNI)
- Rubin's rules pool gc_low / gc_high / pni coefficients.
- % attenuation = (OR_A - OR_B) / OR_A  for gc_high.
- Compare MI-pooled B vs complete-case S5b (n~661) and full-data A.
- Robustness: convergence (M=5/10/20/50) + MNAR pattern-mixture tipping point.
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
np.random.seed(20260826)

# ----------------------------------------------------------------------
# 1. Load
# ----------------------------------------------------------------------
d = pd.read_csv(f"{FEAS}/exports/analytic_mimiciv.csv")
print("MIMIC rows:", len(d), "| columns sample:", [c for c in d.columns][:8], "...")

# derived
d["sex_m"] = (d["sex"] == "M").astype(int)
d["gc_strata_num"] = np.where(d["gc_use"] == 0, 0,
                       np.where(d["gc_dose_pred_eq_mg"] <= 100, 1, 2))
d["gc_low"]  = (d["gc_strata_num"] == 1).astype(int)
d["gc_high"] = (d["gc_strata_num"] == 2).astype(int)
y = d["allcause_death_30d"].astype(int).values

# centering constants (observed full-sample means; constant across m)
age_mean   = d["age"].mean()
charl_mean = d["charlson"].mean()
pni_obs    = d["pni"].dropna()
pni_mean   = pni_obs.mean()
d["age_c"]     = d["age"] - age_mean
d["charlson_c"]= d["charlson"] - charl_mean

# ----------------------------------------------------------------------
# 2. Imputation feature matrix (MAR: include outcome + predictors)
# ----------------------------------------------------------------------
feat = ["albumin", "lymphocytes_per_ul", "creatinine", "age", "sex_m",
        "charlson", "gc_use", "gc_dose_pred_eq_mg", "los_days", "infection",
        "major_comp", "inhosp_death", "allcause_death_30d", "btsdmard_use",
        "gc_strata_num"]
Ximp = d[feat].copy()
# fill the 6 missing gc_dose with 0 so strata numeric is finite
Ximp["gc_dose_pred_eq_mg"] = Ximp["gc_dose_pred_eq_mg"].fillna(0.0)
Ximp["gc_strata_num"]      = Ximp["gc_strata_num"].fillna(0).astype(float)
print("Imputation features:", feat)
print("Missingness %:\n", (Ximp.isna().mean()*100).round(1).to_string())

# generate M datasets with independent posterior draws
imputed = []
for m in range(M):
    imp = IterativeImputer(estimator=BayesianRidge(),
                           max_iter=25, sample_posterior=True,
                           random_state=1000 + m, n_nearest_features=None,
                           skip_complete=False, imputation_order="ascending")
    arr = imp.fit_transform(Ximp.values)
    di = pd.DataFrame(arr, columns=feat)
    # derive PNI from imputed components (formula-consistent)
    di["pni"] = 10.0 * di["albumin"] + 0.005 * di["lymphocytes_per_ul"]
    di["age_c"]      = di["age"] - age_mean
    di["charlson_c"] = di["charlson"] - charl_mean
    di["pni_c"]      = di["pni"] - pni_mean
    di["gc_low"]  = d["gc_low"].values
    di["gc_high"] = d["gc_high"].values
    di["sex_m"]   = d["sex_m"].values
    imputed.append(di)
    if (m+1) % 10 == 0:
        print(f"  imputed {m+1}/{M}")

# ----------------------------------------------------------------------
# 3. Fit models per imputation
# ----------------------------------------------------------------------
def fit_logit(df, with_pni):
    cols = ["gc_low", "gc_high", "age_c", "sex_m", "charlson_c"]
    if with_pni:
        cols = cols + ["pni_c"]
    X = sm.add_constant(df[cols])
    res = sm.Logit(y, X).fit(disp=0, maxiter=200)
    return res

terms_A = ["gc_low", "gc_high"]
terms_B = ["gc_low", "gc_high", "pni_c"]
A_beta, A_var = {t: [] for t in terms_A}, {t: [] for t in terms_A}
B_beta, B_var = {t: [] for t in terms_B}, {t: [] for t in terms_B}
A_or_h, B_or_h = [], []

for di in imputed:
    ra = fit_logit(di, False)
    rb = fit_logit(di, True)
    for t in terms_A:
        A_beta[t].append(ra.params[t]); A_var[t].append(ra.cov_params().loc[t, t])
    for t in terms_B:
        B_beta[t].append(rb.params[t]); B_var[t].append(rb.cov_params().loc[t, t])
    A_or_h.append(np.exp(ra.params["gc_high"]))
    B_or_h.append(np.exp(rb.params["gc_high"]))

# ----------------------------------------------------------------------
# 4. Rubin's rules
# ----------------------------------------------------------------------
def rubin(betas, vars_):
    betas = np.asarray(betas); vars_ = np.asarray(vars_)
    m = len(betas)
    beta_bar = betas.mean()
    W = vars_.mean()
    B = ((betas - beta_bar)**2).sum() / (m - 1)
    T = W + B + (B / m)
    se = np.sqrt(T)
    r = (B + B/m) / W
    df = (m - 1) * (1 + r) ** 2
    return beta_bar, se, df

def fmt(beta_bar, se, df):
    tcrit = st.t.ppf(0.975, df)
    lo = beta_bar - tcrit*se; hi = beta_bar + tcrit*se
    OR = np.exp(beta_bar); CI = (np.exp(lo), np.exp(hi))
    p = 2*(1 - st.t.cdf(abs(beta_bar/se), df))
    return OR, CI, p, df

rows = []
for t in terms_A:
    # Model A (no PNI)
    bb, se, df = rubin(A_beta[t], A_var[t]); OR, CI, p, df = fmt(bb, se, df)
    rows.append(dict(model="A_noPNI", term=t, OR=round(OR,3), CI=f"{CI[0]:.3f}-{CI[1]:.3f}", p=round(p,4), df=round(df,1)))
for t in terms_B:
    # Model B (with PNI)
    bb, se, df = rubin(B_beta[t], B_var[t]); OR, CI, p, df = fmt(bb, se, df)
    rows.append(dict(model="B_PNI", term=t, OR=round(OR,3), CI=f"{CI[0]:.3f}-{CI[1]:.3f}", p=round(p,4), df=round(df,1)))

mi_pooled = pd.DataFrame(rows)
print("\n=== RUBIN-POOLED (M=50) ===")
print(mi_pooled.to_string(index=False))

# % attenuation for gc_high
A_gc_high = mi_pooled[(mi_pooled.model=="A_noPNI") & (mi_pooled.term=="gc_high")]["OR"].values[0]
B_gc_high = mi_pooled[(mi_pooled.model=="B_PNI")   & (mi_pooled.term=="gc_high")]["OR"].values[0]
att = (A_gc_high - B_gc_high) / A_gc_high * 100
print(f"\ngc_high OR: A(noPNI)={A_gc_high:.3f}  B(PNI)={B_gc_high:.3f}  attenuation={att:.1f}%")

# ----------------------------------------------------------------------
# 5. References: full-data A (no imputation) and complete-case B (S5b)
# ----------------------------------------------------------------------
# full-data A: all complete vars, n=2235
ra_full = fit_logit(d, False)
OR_A_full = np.exp(ra_full.params["gc_high"]); CI_A_full=(np.exp(ra_full.params["gc_high"]-1.96*np.sqrt(ra_full.cov_params().loc["gc_high","gc_high"])),
                                                        np.exp(ra_full.params["gc_high"]+1.96*np.sqrt(ra_full.cov_params().loc["gc_high","gc_high"])))
# complete-case B (n with pni): from ORIGINAL observed pni (not imputed)
cc_mask = d["pni"].notna().values
cc = d[cc_mask].copy()
cc["pni_c"] = cc["pni"] - pni_mean
y_cc = y[cc_mask]
# temporary override fit_logit to use y_cc
def fit_logit_cc(df, with_pni):
    cols = ["gc_low", "gc_high", "age_c", "sex_m", "charlson_c"]
    if with_pni:
        cols = cols + ["pni_c"]
    X = sm.add_constant(df[cols])
    res = sm.Logit(y_cc, X).fit(disp=0, maxiter=200)
    return res
rb_cc = fit_logit_cc(cc, True)
OR_B_cc = np.exp(rb_cc.params["gc_high"]); CI_B_cc=(np.exp(rb_cc.params["gc_high"]-1.96*np.sqrt(rb_cc.cov_params().loc["gc_high","gc_high"])),
                                                     np.exp(rb_cc.params["gc_high"]+1.96*np.sqrt(rb_cc.cov_params().loc["gc_high","gc_high"])))
print(f"\nReference full-data A (n={len(d)}): gc_high OR={OR_A_full:.3f} {CI_A_full[0]:.3f}-{CI_A_full[1]:.3f}")
print(f"Reference complete-case B / S5b (n={len(cc)}): gc_high OR={OR_B_cc:.3f} {CI_B_cc[0]:.3f}-{CI_B_cc[1]:.3f}")
print(f"PNI coefficient in B (per 1 PNI pt): OR={np.exp(rb_cc.params['pni_c']):.4f}  -> per 5 pts OR={np.exp(rb_cc.params['pni_c']*5):.3f}")

# ----------------------------------------------------------------------
# 6. Convergence (first k imputations)
# ----------------------------------------------------------------------
conv_rows = []
for k in [5, 10, 20, 50]:
    bb, se, df = rubin(B_beta["gc_high"][:k], B_var["gc_high"][:k]); OR, CI, p, df = fmt(bb, se, df)
    conv_rows.append(dict(M=k, OR_B_gc_high=round(OR,3), CI=f"{CI[0]:.3f}-{CI[1]:.3f}", p=round(p,4)))
conv = pd.DataFrame(conv_rows)
print("\n=== CONVERGENCE (Model B gc_high) ===")
print(conv.to_string(index=False))

# ----------------------------------------------------------------------
# 7. MNAR tipping-point (pattern-mixture)
#    shift imputed PNI for ORIGINALLY-MISSING rows in GC-high group by delta
# ----------------------------------------------------------------------
mask_missing_high = (d["pni"].isna().values) & (d["gc_high"].values == 1)
print(f"\nMNAR tipping: originally-missing PNI in GC-high group = {mask_missing_high.sum()} rows")
tip_rows = []
for delta in [-10, -5, 0, 5, 10]:
    Bb = []; Bv = []
    for di in imputed:
        dj = di.copy()
        dj.loc[mask_missing_high, "pni"] = dj.loc[mask_missing_high, "pni"] + delta
        dj["pni_c"] = dj["pni"] - pni_mean
        rb = fit_logit(dj, True)
        Bb.append(rb.params["gc_high"]); Bv.append(rb.cov_params().loc["gc_high","gc_high"])
    bb, se, df = rubin(Bb, Bv); OR, CI, p, df = fmt(bb, se, df)
    tip_rows.append(dict(delta_PNI_in_missing_high=delta, OR_B_gc_high=round(OR,3),
                         CI=f"{CI[0]:.3f}-{CI[1]:.3f}", p=round(p,4)))
tip = pd.DataFrame(tip_rows)
print("\n=== MNAR TIPPING (Model B gc_high) ===")
print(tip.to_string(index=False))

# ----------------------------------------------------------------------
# 8. Save CSVs
# ----------------------------------------------------------------------
mi_pooled.to_csv(f"{FEAS}/mi_pooled.csv", index=False)
conv.to_csv(f"{FEAS}/mi_convergence.csv", index=False)
tip.to_csv(f"{FEAS}/mi_tipping.csv", index=False)
pd.DataFrame({"m": range(M), "OR_A_gc_high": A_or_h, "OR_B_gc_high": B_or_h}).to_csv(f"{FEAS}/mi_perimputation.csv", index=False)
pd.DataFrame([dict(reference="full_data_A_n2235", OR=round(OR_A_full,3), CI=f"{CI_A_full[0]:.3f}-{CI_A_full[1]:.3f}"),
              dict(reference="complete_case_B_S5b_n661", OR=round(OR_B_cc,3), CI=f"{CI_B_cc[0]:.3f}-{CI_B_cc[1]:.3f}"),
              dict(reference="MI_pooled_A_n2235", OR=round(A_gc_high,3), CI=""),
              dict(reference="MI_pooled_B_n2235", OR=round(B_gc_high,3), CI=""),
              dict(reference="attenuation_pct", OR=round(att,1), CI="")]).to_csv(f"{FEAS}/mi_summary.csv", index=False)

# ----------------------------------------------------------------------
# 9. Figures
# ----------------------------------------------------------------------
# 9a. Forest: per-imputation B ORs + pooled Rubin CI + reference markers
fig, ax = plt.subplots(figsize=(7.5, 5.2))
ax.axvline(1.0, color="grey", ls="--", lw=1)
# pooled
bb, se, df = rubin(B_beta["gc_high"], B_var["gc_high"]); ORp, CIp, p, df = fmt(bb, se, df)
ax.errorbar(ORp, 1.5, xerr=[[ORp-CIp[0]], [CIp[1]-ORp]], fmt="o", color="#c0392b", ms=10, lw=2.5, label=f"MI pooled (M=50): {ORp:.2f} [{CIp[0]:.2f}-{CIp[1]:.2f}]")
# per-imputation jitter
ys = np.linspace(2.5, 50.5, M)
ax.scatter(B_or_h, ys, s=14, color="#2980b9", alpha=0.6, label="per-imputation OR (M=50)")
# references
ax.scatter([OR_A_full], [0.6], marker="D", color="black", s=60, label=f"Full-data A (no PNI): {OR_A_full:.2f}")
ax.scatter([OR_B_cc], [1.0], marker="s", color="#27ae60", s=55, label=f"Complete-case S5b: {OR_B_cc:.2f}")
ax.set_yticks([]); ax.set_ylim(0, 52)
ax.set_xlabel("Odds ratio (GC-high vs none) for 30-day death, adjusted")
ax.set_title("GC-high -> 30d death: effect of adding PNI (multiple imputation)")
ax.legend(fontsize=8, loc="upper right")
fig.tight_layout(); fig.savefig(f"{FEAS}/mi_forest.png", dpi=150); plt.close(fig)

# 9b. Attenuation bar
fig, ax = plt.subplots(figsize=(7, 4.2))
labels = ["Full-data A\n(no PNI)", "MI pooled A\n(no PNI)", "Complete-case B\n(S5b, n=661)", "MI pooled B\n(PNI, n=2235)"]
vals = [OR_A_full, A_gc_high, OR_B_cc, B_gc_high]
colors = ["#34495e", "#7f8c8d", "#27ae60", "#c0392b"]
bars = ax.bar(labels, vals, color=colors)
for b, v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, v+0.05, f"{v:.2f}", ha="center", fontsize=10, fontweight="bold")
ax.axhline(1.0, color="grey", ls="--")
ax.set_ylabel("Odds ratio (GC-high vs none)")
ax.set_title(f"Attenuation of GC-death OR after adding PNI: {att:.0f}% (MI pooled)")
fig.tight_layout(); fig.savefig(f"{FEAS}/mi_attenuation.png", dpi=150); plt.close(fig)

print("\nDONE. Figures + CSVs written.")
