-- =============================================================
-- RA perioperative · unified extraction for:
--  (A) daily-SOFA MSM upgrade  (逐日 SOFA 的 MSM 升级)
--  (B) negative control variables (阳性对照 infection / 阴性对照 ondansetron & fall)
-- Subcohort: first ICU stay of RA surgical admissions (n=668)
-- Output:
--   exports/msm_daily_sofa.csv      long: stay_id, day_idx, daily_sofa (max sofa_24h/day)
--   exports/msm_negctrl_perstay.csv stay_id, hadm_id, ondansetron, ppi, statin,
--                                     acetaminophen, fall_dx
-- =============================================================
\set AUTOCOMMIT on
SET enable_seqscan = off;

-- Stage 1: first ICU stay per RA surgical admission
DROP TABLE IF EXISTS _fi;
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

-- Stage 2: GC prescription timeline (pred-equivalent, reused)
DROP TABLE IF EXISTS _gct;
CREATE TEMP TABLE _gct AS
SELECT fi.subject_id, fi.hadm_id, fi.stay_id, fi.intime,
  EXTRACT(EPOCH FROM (p.starttime - fi.intime))/3600.0 AS start_hour,
  CASE WHEN p.starttime < fi.intime THEN 1 ELSE 0 END AS preicu_flag,
  CASE
    WHEN (p.drug ILIKE '%prednisone%' OR p.drug ILIKE '%prednisolone%'
       OR p.drug ILIKE '%methylprednisolone%' OR p.drug ILIKE '%hydrocortisone%'
       OR p.drug ILIKE '%cortisone%' OR p.drug ILIKE '%dexamethasone%'
       OR p.drug ILIKE '%triamcinolone%' OR p.drug ILIKE '%betamethasone%')
         AND p.dose_val_rx ~ '^[0-9]+(\.[0-9]+)?$' AND p.dose_unit_rx ILIKE '%mg%'
    THEN CASE
      WHEN p.drug ILIKE '%prednisone%' OR p.drug ILIKE '%prednisolone%' THEN 1.0
      WHEN p.drug ILIKE '%methylprednisolone%' THEN 1.25
      WHEN p.drug ILIKE '%hydrocortisone%' OR p.drug ILIKE '%cortisone%' THEN 0.25
      WHEN p.drug ILIKE '%dexamethasone%' OR p.drug ILIKE '%betamethasone%' THEN 6.667
      WHEN p.drug ILIKE '%triamcinolone%' THEN 1.25
      ELSE 0 END * p.dose_val_rx::numeric
    ELSE 0 END AS dose_predeq_mg
FROM _fi fi
JOIN ra_periop.cohort c
  ON fi.subject_id = c.subject_id::integer AND fi.hadm_id = c.encounter_id::integer
JOIN mimiciv_hosp.prescriptions p
  ON c.subject_id::integer = p.subject_id AND c.encounter_id::integer = p.hadm_id
JOIN mimiciv_hosp.admissions a ON p.hadm_id = a.hadm_id
WHERE p.starttime IS NOT NULL
  AND p.starttime BETWEEN a.admittime AND fi.intime + INTERVAL '30 days'
  AND (p.drug ILIKE '%prednisone%' OR p.drug ILIKE '%prednisolone%'
    OR p.drug ILIKE '%methylprednisolone%' OR p.drug ILIKE '%hydrocortisone%'
    OR p.drug ILIKE '%cortisone%' OR p.drug ILIKE '%dexamethasone%'
    OR p.drug ILIKE '%triamcinolone%' OR p.drug ILIKE '%betamethasone%');
CREATE INDEX ON _gct(stay_id);

-- Stage 3: daily SOFA (max sofa_24h within each calendar ICU-day)
DROP TABLE IF EXISTS _sofa_day;
CREATE TEMP TABLE _sofa_day AS
SELECT f.stay_id,
       floor(EXTRACT(EPOCH FROM (s.starttime - f.intime))/86400.0)::integer AS day_idx,
       max(s.sofa_24hours) AS daily_sofa
FROM _fi f
JOIN mimiciv_derived.sofa s ON s.stay_id = f.stay_id
WHERE s.starttime >= f.intime
GROUP BY f.stay_id, floor(EXTRACT(EPOCH FROM (s.starttime - f.intime))/86400.0);
CREATE INDEX ON _sofa_day(stay_id);

-- Stage 4: negative-control exposures (hadm-level prescriptions)
DROP TABLE IF EXISTS _nc;
CREATE TEMP TABLE _nc AS
SELECT f.stay_id, f.hadm_id,
  max(CASE WHEN lower(coalesce(p.drug,'')) LIKE '%ondansetron%' OR lower(coalesce(p.drug,'')) LIKE '%zofran%' THEN 1 ELSE 0 END) AS ondansetron,
  max(CASE WHEN lower(coalesce(p.drug,'')) LIKE '%omeprazole%' OR lower(coalesce(p.drug,'')) LIKE '%pantoprazole%' OR lower(coalesce(p.drug,'')) LIKE '%esomeprazole%' THEN 1 ELSE 0 END) AS ppi,
  max(CASE WHEN lower(coalesce(p.drug,'')) LIKE '%atorvastatin%' OR lower(coalesce(p.drug,'')) LIKE '%rosuvastatin%' OR lower(coalesce(p.drug,'')) LIKE '%simvastatin%' THEN 1 ELSE 0 END) AS statin,
  max(CASE WHEN lower(coalesce(p.drug,'')) LIKE '%acetaminophen%' OR lower(coalesce(p.drug,'')) LIKE '%paracetamol%' THEN 1 ELSE 0 END) AS acetaminophen
FROM _fi f
JOIN mimiciv_hosp.prescriptions p ON p.hadm_id = f.hadm_id
GROUP BY f.stay_id, f.hadm_id;

-- Stage 5: negative-control outcome — in-hospital fall/accidental injury (ICD-10 W00-W19)
DROP TABLE IF EXISTS _fall;
CREATE TEMP TABLE _fall AS
SELECT f.stay_id,
  max(CASE WHEN d.icd_version = 10 AND d.icd_code >= 'W00' AND d.icd_code <= 'W19' THEN 1 ELSE 0 END) AS fall_dx
FROM _fi f
JOIN mimiciv_hosp.diagnoses_icd d ON d.hadm_id = f.hadm_id
GROUP BY f.stay_id;

-- Export A: daily SOFA long
\copy (SELECT stay_id::integer AS stay_id, day_idx::integer AS day_idx, daily_sofa::integer AS daily_sofa FROM _sofa_day ORDER BY stay_id, day_idx) TO 'exports/msm_daily_sofa.csv' CSV HEADER;

-- Export B: negative-control per-stay (single-line \copy)
\copy (SELECT n.stay_id::integer AS stay_id, n.hadm_id::integer AS hadm_id, n.ondansetron::smallint AS ondansetron, n.ppi::smallint AS ppi, n.statin::smallint AS statin, n.acetaminophen::smallint AS acetaminophen, COALESCE(fl.fall_dx,0)::smallint AS fall_dx FROM _nc n LEFT JOIN _fall fl ON n.stay_id = fl.stay_id ORDER BY n.stay_id) TO 'exports/msm_negctrl_perstay.csv' CSV HEADER;

-- Diagnostics
SELECT 'daily sofa rows: ' || count(*) FROM _sofa_day;
SELECT 'stays with daily sofa: ' || count(DISTINCT stay_id) FROM _sofa_day;
SELECT 'ondansetron exposed: ' || count(*) FROM _nc WHERE ondansetron=1;
SELECT 'fall dx: ' || count(*) FROM _fall WHERE fall_dx=1;
