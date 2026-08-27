# ra-periop-gc-analysis

**Glucocorticoid exposure, in-hospital infection, and short-term mortality in surgical patients with rheumatoid arthritis: a multi-database intensive care cohort study**

This repository contains the data-extraction queries and analysis code accompanying the manuscript
"类风湿关节炎 ICU 外科患者糖皮质激素暴露、院内感染与短期死亡：多数据库队列研究"
(RA perioperative glucocorticoid ICU cohort study). It is the computational companion to the
multi-database observational analysis across **MIMIC-IV**, **eICU-CRD**, **INSPIRE**, and **NWICU**.

> ⚠️ **Data-use notice.** All four source databases are governed by PhysioNet data-use agreements
> (DUA). **No patient-level data, intermediate CSV extracts, or derived tables are included in this
> repository.** Users must obtain the databases independently, run the SQL extraction under their own
> DUA, and place the resulting CSVs in a local `data/` directory (configurable via the `RA_GC_DATA`
> environment variable) before running the analysis scripts.

---

## Repository structure

```
ra-periop-gc-analysis/
├── sql/                 # Data extraction & cohort-construction queries (per database)
│   ├── build_mimic_cohort.sql        # MIMIC-IV ICU RA surgical cohort
│   ├── build_eicu_cohort.sql         # eICU-CRD cohort
│   ├── build_inspire_cohort.sql      # INSPIRE cohort
│   ├── build_nwicu_cohort.sql        # NWICU cohort
│   ├── extract_gc_timeline.sql       # glucocorticoid exposure timeline (dose → prednisone-equivalent)
│   ├── extract_infection_timing.sql  # infection onset timing (treatment-timed infections)
│   ├── extract_mimic_icu_severity.sql# SOFA / severity extraction
│   ├── augment_*.sql                 # PNI, negative-control, and per-database augmentations
│   └── update_nwicu_pni.sql          # NWICU PNI update
├── analysis/            # Python analysis pipeline (statsmodels / lifelines)
│   ├── ra_periop_analysis.py             # primary infection & mortality associations (logistic)
│   ├── ra_periop_inspire_firth_meta.py   # cross-database random-effects meta (Firth for small strata)
│   ├── ra_periop_msm_timedependent.py    # time-dependent Cox with IPTW marginal structural model
│   ├── ra_periop_msm_dailysofa_negctrl.py# MSM + daily SOFA + negative/positive controls
│   ├── ra_periop_gc_death_sensitivity.py # death sensitivity, PNI MI, Fine–Gray sHR, E-value
│   ├── ra_periop_mi_pni.py / *_crossdb.py# multiple imputation of PNI
│   ├── ra_periop_severity_landmark.py    # 48 h landmark & severity analyses
│   ├── ra_periop_revision_v2.py / _v6 / _v7 # consolidated analyses producing the reported estimates
├── R/
│   └── fg_weighted_cox.R  # Fine–Gray subdistribution hazard via Gray-weighted Cox (competing death)
├── index.html            # Web risk estimator (English)
├── index_zh.html         # Web risk estimator (中文)
├── .nojekyll
├── LICENSE
└── .gitignore
```

## Analysis pipeline (suggested order)

1. **Extract** — run the `sql/` queries against each database under your DUA to build the cohorts and
   extract GC exposure, infection timing, and severity. Output CSVs go to your local `data/` directory.
2. **Primary associations** — `ra_periop_analysis.py` and `ra_periop_inspire_firth_meta.py` produce the
   cross-database pooled OR for GC → in-hospital infection (meta) and the GC → 30-day mortality
   associations.
3. **Time-dependent & MSM** — `ra_periop_msm_timedependent.py` and
   `ra_periop_msm_dailysofa_negctrl.py` fit stabilized-IPTW marginal structural models and negative
   controls to address time-varying confounding and indication bias.
4. **Sensitivity** — `ra_periop_gc_death_sensitivity.py` (PNI multiple imputation, Fine–Gray
   competing-risk analysis, E-value), `ra_periop_mi_pni*.py`, and `ra_periop_severity_landmark.py`.
5. **Competing risk** — `R/fg_weighted_cox.R` fits the Fine–Gray model (death as competing event).

## Dependencies

**Python** (tested with 3.13): `pandas`, `numpy`, `statsmodels`, `lifelines`, `scipy`.
Load your local environment via `PYTHONPATH` (e.g. `export PYTHONPATH=/path/to/.pylibs`) or `pip install -r`
after exporting the requirements.

**R** (tested with 4.6): `survival` (required) and `cmprsk` (reference only; the Fine–Gray estimate uses
the Gray-weighted Cox implementation, which does not depend on `cmprsk`).

Scripts read inputs from `RA_GC_DATA` (default `./data`) and look for a local `.pylibs` via `RA_GC_PYLIBS`
(default `./.pylibs`); adjust these environment variables to your layout.

## Web risk estimator

`index.html` (English) and `index_zh.html` (中文) are a **pure front-end, dependency-free** companion
calculator. It implements a multivariable logistic-regression model for in-hospital infection fitted on
the **MIMIC-IV ICU RA subgroup (n = 668)** with predictors: any systemic glucocorticoid exposure during
admission, age, sex, SOFA, and Charlson comorbidity index.

> The estimator is **research-use only** and reflects an *observational* risk profile, not a causal drug
> effect (glucocorticoids are frequently given to sicker patients — indication/severity confounding).
> External validation is pending.

Model coefficients (hard-coded in the page): β₀ = −1.7189, β_GC = 0.6382, β_age = 0.0042,
β_sex(M) = −0.4373, β_SOFA = 0.1697, β_Charlson = 0.0956.

The calculator is deployed via GitHub Pages:

- English: <https://morrosun.github.io/ra-periop-gc-analysis/>
- 中文: <https://morrosun.github.io/ra-periop-gc-analysis/index_zh.html>

## License

Code is released under the **MIT License** (see `LICENSE`). The manuscript text and figures are not
covered by this license. Database contents remain under their respective PhysioNet DUAs.

## Citation & DOI

A version-specific DOI is issued via Zenodo on each GitHub Release.

- **Release v1.0.0**: <https://github.com/morrosun/ra-periop-gc-analysis/releases/tag/v1.0.0>
- **Zenodo DOI**: *pending* — assigned automatically when Zenodo harvests the GitHub Release (see below).

Cite:

> Wang K, et al. Glucocorticoid exposure, in-hospital infection, and short-term mortality in surgical
> patients with rheumatoid arthritis: a multi-database intensive care cohort study. *[Journal, in press]*.
> Code & data-extraction: https://github.com/morrosun/ra-periop-gc-analysis (DOI: *to be added*).
