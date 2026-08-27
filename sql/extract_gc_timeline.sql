-- =============================================================
-- RA perioperative · MIMIC-IV ICU subcohort (n=668, first ICU stay)
-- Build GC prescription-level timeline (time-dependent exposure source)
-- for Marginal Structural Model / time-dependent Cox.
-- Output: exports/gc_timeline_mimic_icu.csv  (long, one row per GC Rx)
--   start_hour   = hours from ICU intime (negative = pre-ICU / ward stay)
--   dose_predeq_mg = prednisone-equivalent mg of THIS prescription
--   preicu_flag  = 1 if starttime < intime (baseline exposure)
-- =============================================================
\set AUTOCOMMIT on
SET enable_seqscan = off;

-- Stage 1: first ICU stay per RA surgical admission
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
CREATE INDEX ON _fi(hadm_id);
CREATE INDEX ON _fi(stay_id);

-- Stage 2: GC prescriptions timeline (admission start .. intime+30d)
-- Reuse exact pred-equivalent conversion from build/previous extracts.
DROP TABLE IF EXISTS ra_periop._gct;
CREATE TEMP TABLE _gct AS
SELECT
  fi.subject_id, fi.hadm_id, fi.stay_id, fi.intime,
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

-- Stage 3: export long timeline (single-line \copy)
\copy (SELECT subject_id::integer, hadm_id::integer, stay_id::integer, intime, round(start_hour::numeric,3) AS start_hour, round(dose_predeq_mg::numeric,3) AS dose_predeq_mg, preicu_flag::smallint FROM _gct ORDER BY subject_id, hadm_id, start_hour) TO 'exports/gc_timeline_mimic_icu.csv' CSV HEADER;

-- quick diagnostics
SELECT 'timeline rows: ' || count(*) FROM _gct;
SELECT 'stays with >=1 GC Rx: ' || count(DISTINCT hadm_id) FROM _gct;
SELECT 'pre-ICU GC Rx rows: ' || count(*) FROM _gct WHERE preicu_flag=1;
SELECT 'post-ICU GC Rx rows: ' || count(*) FROM _gct WHERE preicu_flag=0;
