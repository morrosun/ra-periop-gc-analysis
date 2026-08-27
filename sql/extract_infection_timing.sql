-- Infection onset timing for the MIMIC-IV ICU subcohort (first stay per admission)
-- Purpose: enable temporal-ordering sensitivity of GC -> incident infection.
-- Sources:
--   mimiciv_derived.antibiotic   (starttime, stay-level) -> first systemic anti-infective
--   mimiciv_hosp.microbiologyevents (chartdate, positive culture) -> first organism
-- Baseline subcohort = analytic_mimic_icu_v3 (668 first ICU stays)

\set AUTOCOMMIT on
DROP TABLE IF EXISTS _raw;
CREATE TEMP TABLE _raw (
  subject_id text, hadm_id text, stay_id text, admittime text, intime text, outtime text,
  deathtime text, dod text, age text, sex text, charlson text, pni text, creatinine text,
  gc_use_fulladm text, gc_dose_fulladm text, gc_use_48h text, gc_dose_48h text, sofa text,
  respiration text, coagulation text, liver text, cardiovascular text, cns text, renal text,
  apsiii text, oasis text, sapsii text, vent_24h text, vaso_24h text, rrt_24h text,
  albumin_min text, hemoglobin_min text, platelets_min text, wbc_min text, lactate_max text,
  creatinine_kdigo text, surgery_category text, inhosp_death text, allcause_death_30d text, infection text
);
\copy _raw FROM 'exports/analytic_mimic_icu_v3.csv' WITH (FORMAT csv, HEADER)

DROP TABLE IF EXISTS _icu668;
CREATE TEMP TABLE _icu668 AS
SELECT stay_id::integer AS stay_id, hadm_id::integer AS hadm_id, subject_id::integer AS subject_id,
       intime::timestamp AS intime,
       gc_use_fulladm::smallint AS gc_use_fulladm, gc_dose_48h::numeric AS gc_dose_48h,
       gc_use_48h::smallint AS gc_use_48h
FROM _raw;

-- First antibiotic (any route) relative to ICU intime
DROP TABLE IF EXISTS _abx;
CREATE TEMP TABLE _abx AS
SELECT a.stay_id,
       MIN(EXTRACT(EPOCH FROM (a.starttime - i.intime))/3600.0) AS abx_hours_first
FROM _icu668 i
JOIN mimiciv_derived.antibiotic a ON a.stay_id = i.stay_id
WHERE a.starttime IS NOT NULL AND a.starttime >= i.intime - INTERVAL '2 days'
      AND a.starttime <= i.intime + INTERVAL '40 days'
GROUP BY a.stay_id;

-- First positive culture relative to ICU intime
DROP TABLE IF EXISTS _culture;
CREATE TEMP TABLE _culture AS
SELECT i.stay_id,
       MIN(EXTRACT(EPOCH FROM (m.chartdate - i.intime))/3600.0) AS cult_hours_first
FROM _icu668 i
JOIN mimiciv_hosp.microbiologyevents m ON m.hadm_id = i.hadm_id
WHERE m.chartdate IS NOT NULL
  AND m.org_name IS NOT NULL
  AND lower(m.org_name) NOT IN ('no growth','not done','none','no organism isolated','sterile','contaminant')
  AND m.chartdate >= i.intime - INTERVAL '2 days'
  AND m.chartdate <= i.intime + INTERVAL '40 days'
GROUP BY i.stay_id;

-- Assemble
DROP TABLE IF EXISTS _inftime;
CREATE TEMP TABLE _inftime AS
SELECT i.stay_id, i.hadm_id, i.subject_id, i.intime,
       i.gc_use_fulladm, i.gc_dose_48h, i.gc_use_48h,
       ab.abx_hours_first, cu.cult_hours_first,
       LEAST(ab.abx_hours_first, cu.cult_hours_first) AS onset_hours
FROM _icu668 i
LEFT JOIN _abx ab ON ab.stay_id = i.stay_id
LEFT JOIN _culture cu ON cu.stay_id = i.stay_id;

\copy (SELECT stay_id, hadm_id, subject_id, intime, gc_use_fulladm::smallint AS gc_use, gc_dose_48h, gc_use_48h::smallint AS gc_use_48h, abx_hours_first, cult_hours_first, onset_hours FROM _inftime ORDER BY stay_id) TO 'exports/infection_timing_mimic_icu.csv' CSV HEADER;

SELECT 'rows: ' || count(*) FROM _inftime;
SELECT 'with_abx: ' || count(*) FROM _inftime WHERE abx_hours_first IS NOT NULL;
SELECT 'with_culture: ' || count(*) FROM _inftime WHERE cult_hours_first IS NOT NULL;
SELECT 'with_onset: ' || count(*) FROM _inftime WHERE onset_hours IS NOT NULL;
SELECT 'early(<=24h): ' || count(*) FROM _inftime WHERE onset_hours IS NOT NULL AND onset_hours <= 24;
SELECT 'late(>24h): ' || count(*) FROM _inftime WHERE onset_hours IS NOT NULL AND onset_hours > 24;
