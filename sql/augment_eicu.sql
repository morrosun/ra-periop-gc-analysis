\set AUTOCOMMIT on
-- =============================================================
-- RA perioperative · eICU covariate augmentation
-- Charlson: NOT available (eICU has no ICD codes) -> NULL
-- PNI: from eicu_sae.lab_agg (albumin_min g/dL, lymphocytes_abs
--       in x10^3/uL -> x1000 to /uL for PNI)
-- encounter_id in cohort = patientunitstayid
-- =============================================================
CREATE SCHEMA IF NOT EXISTS ra_periop;
DROP TABLE IF EXISTS ra_periop.analytic;

CREATE TABLE ra_periop.analytic AS
WITH lab AS (
  SELECT patientunitstayid,
         albumin_min,
         lymphocytes_abs,
         cr_max AS creatinine_max
  FROM eicu_sae.lab_agg
  WHERE patientunitstayid IN (SELECT encounter_id::integer FROM ra_periop.cohort)
)
SELECT
  c.*,
  NULL::integer                                                                               AS charlson,
  lb.albumin_min                                                                              AS albumin,
  (lb.lymphocytes_abs * 1000.0)                                                               AS lymphocytes_per_ul,
  CASE WHEN lb.albumin_min IS NOT NULL AND lb.lymphocytes_abs IS NOT NULL
       THEN ROUND((10.0*lb.albumin_min + 0.005*lb.lymphocytes_abs*1000.0)::numeric, 2)
       END                                                                                    AS pni,
  lb.creatinine_max                                                                           AS creatinine
FROM ra_periop.cohort c
LEFT JOIN lab lb ON c.encounter_id::integer = lb.patientunitstayid;

SELECT 'eICU analytic rows: ' || count(*) FROM ra_periop.analytic;
SELECT 'eICU with PNI: '     || count(*) FROM ra_periop.analytic WHERE pni IS NOT NULL;
SELECT 'eICU charlson (NULL expected): ' || count(*) FROM ra_periop.analytic WHERE charlson IS NULL;
