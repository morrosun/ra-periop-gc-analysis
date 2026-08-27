-- =============================================================
-- RA perioperative · MIMIC-IV ICU subcohort severity + early GC + time-to-event
-- Goal: address reviewer #2 (severity adjustment) and #5 (immortal-time / landmark)
-- Unit: FIRST ICU STAY of each RA surgical admission (icu_stay = 1)
-- Exports a single analytic CSV for Python (time math done there).
-- =============================================================
\set AUTOCOMMIT on
SET enable_seqscan = off;

-- Stage 1: first ICU stay per admission (among RA surgical cohort)
DROP TABLE IF EXISTS ra_periop._fi;
CREATE TEMP TABLE _fi AS
WITH r AS (
  SELECT i.subject_id, i.hadm_id, i.stay_id, i.intime, i.outtime,
         ROW_NUMBER() OVER (PARTITION BY i.subject_id, i.hadm_id ORDER BY i.intime) rn
  FROM mimiciv_icu.icustays i
  JOIN ra_periop.cohort c
    ON i.subject_id = c.subject_id::integer AND i.hadm_id = c.encounter_id::integer
)
SELECT subject_id, hadm_id, stay_id, intime, outtime FROM r WHERE rn = 1;
CREATE INDEX ON _fi(stay_id);
CREATE INDEX ON _fi(hadm_id);

-- Stage 2: severity from derived tables (per first stay_id)
DROP TABLE IF EXISTS ra_periop._sev;
CREATE TEMP TABLE _sev AS
SELECT f.subject_id, f.hadm_id, f.stay_id,
  fs.sofa, fs.respiration, fs.coagulation, fs.liver, fs.cardiovascular, fs.cns, fs.renal,
  ap.apsiii, oa.oasis, sa.sapsii,
  CASE WHEN v.stay_id IS NOT NULL THEN 1 ELSE 0 END  AS vent_24h,
  CASE WHEN va.stay_id IS NOT NULL THEN 1 ELSE 0 END AS vaso_24h,
  CASE WHEN rrt.stay_id IS NOT NULL THEN 1 ELSE 0 END AS rrt_24h,
  fl.albumin_min, fl.hemoglobin_min, fl.platelets_min, fl.wbc_min,
  fbg.lactate_max,
  kc.creat AS creatinine_kdigo
FROM _fi f
LEFT JOIN mimiciv_derived.first_day_sofa   fs ON f.stay_id = fs.stay_id
LEFT JOIN mimiciv_derived.apsiii           ap ON f.stay_id = ap.stay_id
LEFT JOIN mimiciv_derived.oasis            oa ON f.stay_id = oa.stay_id
LEFT JOIN mimiciv_derived.sapsii           sa ON f.stay_id = sa.stay_id
LEFT JOIN (SELECT DISTINCT v.stay_id
           FROM mimiciv_derived.ventilation v JOIN _fi f2 ON v.stay_id = f2.stay_id
           WHERE v.starttime BETWEEN f2.intime AND f2.intime + INTERVAL '24 hours') v
       ON f.stay_id = v.stay_id
LEFT JOIN (SELECT DISTINCT va.stay_id
           FROM mimiciv_derived.vasoactive_agent va JOIN _fi f2 ON va.stay_id = f2.stay_id
           WHERE va.starttime BETWEEN f2.intime AND f2.intime + INTERVAL '24 hours') va
       ON f.stay_id = va.stay_id
LEFT JOIN (SELECT DISTINCT stay_id FROM mimiciv_derived.first_day_rrt WHERE dialysis_present = 1) rrt
       ON f.stay_id = rrt.stay_id
LEFT JOIN mimiciv_derived.first_day_lab    fl ON f.stay_id = fl.stay_id
LEFT JOIN mimiciv_derived.first_day_bg     fbg ON f.stay_id = fbg.stay_id
LEFT JOIN (SELECT stay_id, MIN(creat) AS creat
           FROM mimiciv_derived.kdigo_creatinine GROUP BY stay_id) kc ON f.stay_id = kc.stay_id;

-- Stage 3: EARLY GC exposure = first 48h of ADMISSION (perioperative window)
-- Reuses the exact GC-drug / pred-equivalent conversion from build_mimic_cohort.sql
DROP TABLE IF EXISTS ra_periop._earlygc;
CREATE TEMP TABLE _earlygc AS
SELECT p.subject_id, p.hadm_id,
  MAX(CASE WHEN
     p.drug ILIKE '%prednisone%' OR p.drug ILIKE '%prednisolone%' OR p.drug ILIKE '%methylprednisolone%'
  OR p.drug ILIKE '%hydrocortisone%' OR p.drug ILIKE '%cortisone%' OR p.drug ILIKE '%dexamethasone%'
  OR p.drug ILIKE '%triamcinolone%' OR p.drug ILIKE '%betamethasone%' THEN 1 ELSE 0 END) AS gc_use_48h,
  SUM(CASE
     WHEN (p.drug ILIKE '%prednisone%' OR p.drug ILIKE '%prednisolone%'
        OR p.drug ILIKE '%methylprednisolone%' OR p.drug ILIKE '%hydrocortisone%' OR p.drug ILIKE '%cortisone%'
        OR p.drug ILIKE '%dexamethasone%' OR p.drug ILIKE '%triamcinolone%' OR p.drug ILIKE '%betamethasone%')
          AND p.dose_val_rx ~ '^[0-9]+(\.[0-9]+)?$' AND p.dose_unit_rx ILIKE '%mg%'
     THEN CASE
       WHEN p.drug ILIKE '%prednisone%' OR p.drug ILIKE '%prednisolone%' THEN 1.0
       WHEN p.drug ILIKE '%methylprednisolone%' THEN 1.25
       WHEN p.drug ILIKE '%hydrocortisone%' OR p.drug ILIKE '%cortisone%' THEN 0.25
       WHEN p.drug ILIKE '%dexamethasone%' OR p.drug ILIKE '%betamethasone%' THEN 6.667
       WHEN p.drug ILIKE '%triamcinolone%' THEN 1.25
       ELSE 0 END * p.dose_val_rx::numeric
     ELSE 0 END) AS gc_dose_48h
FROM ra_periop.cohort c
JOIN mimiciv_hosp.prescriptions p
  ON c.subject_id::integer = p.subject_id AND c.encounter_id::integer = p.hadm_id
JOIN mimiciv_hosp.admissions a ON p.hadm_id = a.hadm_id
WHERE p.starttime IS NOT NULL
  AND p.starttime BETWEEN a.admittime AND a.admittime + INTERVAL '48 hours'
GROUP BY p.subject_id, p.hadm_id;

-- Stage 4: assemble. Times kept raw for Python time-to-event / landmark.
DROP TABLE IF EXISTS ra_periop._out;
CREATE TEMP TABLE _out AS
SELECT
  c.subject_id::integer                       AS subject_id,
  c.encounter_id::integer                     AS hadm_id,
  fi.stay_id                                 AS stay_id,
  a.admittime, fi.intime, fi.outtime,
  a.deathtime, pt.dod,
  c.age, UPPER(pt.gender) AS sex,
  an.charlson, an.pni, an.creatinine,
  c.gc_use                                   AS gc_use_fulladm,
  c.gc_dose_pred_eq_mg                       AS gc_dose_fulladm,
  COALESCE(eg.gc_use_48h,0)::smallint        AS gc_use_48h,
  eg.gc_dose_48h::numeric                    AS gc_dose_48h,
  sv.sofa, sv.respiration, sv.coagulation, sv.liver, sv.cardiovascular, sv.cns, sv.renal,
  sv.apsiii, sv.oasis, sv.sapsii,
  sv.vent_24h, sv.vaso_24h, sv.rrt_24h,
  sv.albumin_min, sv.hemoglobin_min, sv.platelets_min, sv.wbc_min,
  sv.lactate_max, sv.creatinine_kdigo,
  c.surgery_category,
  c.inhosp_death, c.allcause_death_30d, c.infection
FROM ra_periop.cohort c
JOIN _fi fi ON c.subject_id::integer = fi.subject_id AND c.encounter_id::integer = fi.hadm_id
JOIN mimiciv_hosp.admissions a ON c.encounter_id::integer = a.hadm_id
JOIN mimiciv_hosp.patients  pt ON c.subject_id::integer = pt.subject_id
LEFT JOIN ra_periop.analytic an
  ON c.subject_id = an.subject_id AND c.encounter_id = an.encounter_id
LEFT JOIN _sev sv ON fi.stay_id = sv.stay_id
LEFT JOIN _earlygc eg
  ON c.subject_id::integer = eg.subject_id AND c.encounter_id::integer = eg.hadm_id;

-- Stage 5: export
\copy (SELECT * FROM _out ORDER BY subject_id, hadm_id) TO 'exports/analytic_mimic_icu_v3.csv' CSV HEADER;

SELECT 'ICU subcohort rows: ' || count(*) FROM _out;
SELECT 'with SOFA: ' || count(*) FROM _out WHERE sofa IS NOT NULL;
SELECT 'with APSIII: ' || count(*) FROM _out WHERE apsiii IS NOT NULL;
SELECT 'with early GC dose: ' || count(*) FROM _out WHERE gc_dose_48h IS NOT NULL;
