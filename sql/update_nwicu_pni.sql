-- =============================================================
-- RA perioperative · NWICU PNI re-derivation (v2, FIX)
-- The original augment_nwicu.sql joined nwicu_sae.lab_agg.stay_id
-- -> mimiciv_icu.icustays.stay_id, but lab_agg.stay_id belongs to a
-- DIFFERENT id space (50xxxxxx) that matches NO icustays table, so
-- the bridge collapsed to 2 rows (1.2% PNI coverage).
-- FIX: derive albumin / lymphocytes / creatinine directly from
-- raw hosp.labevents KEYED BY hadm_id (== cohort.encounter_id),
-- using BLOOD-only itemids and excluding sentinel 9999999 + physiologic
-- out-of-range values. PNI formula kept identical to MIMIC/INSPIRE:
--   PNI = 10*albumin(g/dL) + 0.005*lymphocytes_per_uL
-- where lymphocytes_per_uL = stored(x10^3/uL) * 1000.
-- =============================================================
\set AUTOCOMMIT on

WITH labs AS (
  SELECT le.hadm_id,
    MIN(CASE WHEN le.itemid = 100021
              AND le.valuenum BETWEEN 0.5 AND 6.0
             THEN le.valuenum END)                                              AS albumin_min,
    MIN(CASE WHEN le.itemid IN (100023, 100037)        -- Lymphocytes / Absolute Lymphocyte Count (Blood)
              AND le.valuenum BETWEEN 0.1 AND 20.0
             THEN le.valuenum END)                                              AS lymph_min,
    MAX(CASE WHEN le.itemid IN (100002,100085,100097,100099,100189,100209,100269,100314)
              AND le.valuenum BETWEEN 0.1 AND 15.0
             THEN le.valuenum END)                                              AS cr_max
  FROM hosp.labevents le
  WHERE le.hadm_id IN (SELECT encounter_id::int FROM ra_periop.cohort)
    AND le.valuenum IS NOT NULL
    AND le.valuenum < 100                -- drop 9999999 sentinel
  GROUP BY le.hadm_id
)
UPDATE ra_periop.analytic a
SET albumin           = l.albumin_min,
    lymphocytes_per_ul = l.lymph_min * 1000.0,
    creatinine         = l.cr_max,
    pni = CASE WHEN l.albumin_min IS NOT NULL AND l.lymph_min IS NOT NULL
               THEN ROUND((10.0*l.albumin_min + 0.005*l.lymph_min*1000.0)::numeric, 2)
          END
FROM labs l
WHERE a.encounter_id::int = l.hadm_id;

SELECT 'NWICU analytic rows: '        || count(*)                FROM ra_periop.analytic;
SELECT 'NWICU with PNI (recovered): ' || count(*)                FROM ra_periop.analytic WHERE pni IS NOT NULL;
SELECT 'NWICU PNI mean: '             || round(avg(pni),2)       FROM ra_periop.analytic WHERE pni IS NOT NULL;
