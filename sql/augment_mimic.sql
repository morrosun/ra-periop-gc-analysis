\set AUTOCOMMIT on
-- =============================================================
-- RA perioperative · MIMIC-IV covariate augmentation (v2)
-- Charlson: mimiciv_derived.charlson (precomputed, by hadm_id)
-- PNI: baseline labs aggregated from mimiciv_hosp.labevents
--       (first 24h of admission; admission-level coverage)
--       albumin 50862 (g/dL); lymphocytes 51133/51244 (x10^3/uL);
--       creatinine 50912 (mg/dL). Lymph x10^3/uL -> x1000 to /uL.
-- =============================================================
CREATE SCHEMA IF NOT EXISTS ra_periop;
DROP TABLE IF EXISTS ra_periop.analytic;

CREATE TABLE ra_periop.analytic AS
WITH lab AS (
  SELECT le.hadm_id,
         MIN(CASE WHEN le.itemid = 50862 THEN le.valuenum END)                       AS albumin_min,
         MIN(CASE WHEN le.itemid IN (51133, 51244) THEN le.valuenum END)             AS lymphocytes_min,
         MAX(CASE WHEN le.itemid = 50912 THEN le.valuenum END)                       AS creatinine_max
  FROM mimiciv_hosp.labevents le
  JOIN mimiciv_hosp.admissions a ON le.hadm_id = a.hadm_id
  WHERE le.hadm_id IN (SELECT encounter_id::integer FROM ra_periop.cohort)
    AND le.charttime >= a.admittime
    AND le.valuenum IS NOT NULL
  GROUP BY le.hadm_id
)
SELECT
  c.*,
  ch.charlson_comorbidity_index                                            AS charlson,
  lb.albumin_min                                                          AS albumin,
  (lb.lymphocytes_min * 1000.0)                                          AS lymphocytes_per_ul,
  CASE WHEN lb.albumin_min IS NOT NULL AND lb.lymphocytes_min IS NOT NULL
       THEN ROUND((10.0*lb.albumin_min + 0.005*lb.lymphocytes_min*1000.0)::numeric, 2)
       END                                                                AS pni,
  lb.creatinine_max                                                      AS creatinine
FROM ra_periop.cohort c
LEFT JOIN mimiciv_derived.charlson ch
       ON c.subject_id::integer = ch.subject_id
      AND c.encounter_id::integer = ch.hadm_id
LEFT JOIN lab lb
       ON c.encounter_id::integer = lb.hadm_id;

SELECT 'MIMIC analytic rows: ' || count(*) FROM ra_periop.analytic;
SELECT 'MIMIC with PNI: '     || count(*) FROM ra_periop.analytic WHERE pni IS NOT NULL;
SELECT 'MIMIC with charlson: '|| count(*) FROM ra_periop.analytic WHERE charlson IS NOT NULL;
