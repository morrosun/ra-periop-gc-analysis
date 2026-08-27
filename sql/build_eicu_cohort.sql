-- =============================================================
-- RA perioperative analysis cohort  ·  eICU-CRD
-- Unit of analysis: ICU STAY (patientunitstayid).
-- RA defined by diagnosisstring ILIKE '%rheumatoid%'.
-- No procedure table -> surgery_confirmed = 0 (ICU = peri-procedural critical care).
-- No external follow-up -> allcause_death_30d = in-unit death.
-- Infection: diagnosisstring keywords UNION positive microlab culture.
-- Schema: ra_periop.cohort  (created in the eicu database)
-- =============================================================
CREATE SCHEMA IF NOT EXISTS ra_periop;
DROP TABLE IF EXISTS ra_periop.cohort;

CREATE TABLE ra_periop.cohort AS
WITH ra AS (
  SELECT DISTINCT patientunitstayid FROM eicu_crd.diagnosis
  WHERE diagnosisstring ILIKE '%rheumatoid%'
),
base AS (
  SELECT p.patientunitstayid, p.gender, p.age,
         p.unitdischargestatus, p.unitadmittime24, p.unitdischargetime24
  FROM eicu_crd.patient p JOIN ra ON p.patientunitstayid = ra.patientunitstayid
),
gc AS (                         -- GC exposure from ICU medication orders
  SELECT m.patientunitstayid, 1 AS gc_use
  FROM eicu_crd.medication m JOIN base b ON m.patientunitstayid = b.patientunitstayid
  WHERE m.drugname ILIKE '%prednisone%' OR m.drugname ILIKE '%prednisolone%'
     OR m.drugname ILIKE '%methylprednisolone%' OR m.drugname ILIKE '%hydrocortisone%'
     OR m.drugname ILIKE '%cortisone%' OR m.drugname ILIKE '%dexamethasone%'
     OR m.drugname ILIKE '%triamcinolone%' OR m.drugname ILIKE '%betamethasone%'
  GROUP BY m.patientunitstayid
),
inf AS (                        -- infection: diagnosis keywords UNION positive culture
  SELECT patientunitstayid, 1 AS infection FROM (
    SELECT d.patientunitstayid FROM eicu_crd.diagnosis d JOIN base b ON d.patientunitstayid = b.patientunitstayid
    WHERE d.diagnosisstring ILIKE '%pneumonia%' OR d.diagnosisstring ILIKE '%sepsis%'
       OR d.diagnosisstring ILIKE '%septic%' OR d.diagnosisstring ILIKE '%urinary tract infection%'
       OR d.diagnosisstring ILIKE '%cellulitis%' OR d.diagnosisstring ILIKE '%wound infection%'
       OR d.diagnosisstring ILIKE '%clostridium%' OR d.diagnosisstring ILIKE '%bacteremia%'
       OR d.diagnosisstring ILIKE '%osteomyelitis%' OR d.diagnosisstring ILIKE '%abscess%'
    UNION
    SELECT m.patientunitstayid FROM eicu_crd.microlab m JOIN base b ON m.patientunitstayid = b.patientunitstayid
    WHERE m.organism IS NOT NULL AND m.organism <> '' AND m.organism NOT ILIKE '%no growth%'
  ) x GROUP BY patientunitstayid
),
mc AS (                         -- major complications (diagnosisstring keywords)
  SELECT d.patientunitstayid, 1 AS major_comp
  FROM eicu_crd.diagnosis d JOIN base b ON d.patientunitstayid = b.patientunitstayid
  WHERE d.diagnosisstring ILIKE '%myocardial infarction%' OR d.diagnosisstring ILIKE '%stroke%'
     OR d.diagnosisstring ILIKE '%pulmonary embolism%' OR d.diagnosisstring ILIKE '%deep vein thrombosis%'
     OR d.diagnosisstring ILIKE '%acute kidney injury%' OR d.diagnosisstring ILIKE '%renal failure%'
     OR d.diagnosisstring ILIKE '%cardiac arrest%' OR d.diagnosisstring ILIKE '%gastrointestinal hemorrhage%'
     OR d.diagnosisstring ILIKE '%gastrointestinal bleed%' OR d.diagnosisstring ILIKE '%postoperative complication%'
  GROUP BY d.patientunitstayid
)
SELECT
  'eICU'::text                                       AS library,
  b.patientunitstayid::text                          AS subject_id,
  b.patientunitstayid::text                          AS encounter_id,
  'icu_stay'::text                                   AS encounter_unit,
  0                                                  AS surgery_confirmed,
  1                                                  AS ra_flag,
  CASE WHEN b.age = '> 89' THEN 90
       WHEN b.age ~ '^[0-9]+$' THEN b.age::int
       ELSE NULL END::numeric(6,2)                   AS age,
  CASE WHEN b.gender ILIKE 'M%' THEN 'M' WHEN b.gender ILIKE 'F%' THEN 'F' ELSE UPPER(b.gender) END AS sex,
  NULL::text                                         AS asa,
  NULL::smallint                                     AS asa_class,
  NULL::smallint                                     AS emergency,
  NULL::text                                         AS surgery_category,
  NULL::text                                         AS primary_proc,
  1                                                  AS icu_stay,
  COALESCE(gc.gc_use,0)::smallint                    AS gc_use,
  NULL::numeric                                      AS gc_dose_pred_eq_mg,
  'none'::text                                       AS dmard_class,
  0::smallint                                        AS btsdmard_use,
  CASE WHEN b.unitdischargestatus = 'Expired' THEN 1 ELSE 0 END::smallint AS inhosp_death,
  CASE WHEN b.unitdischargestatus = 'Expired' THEN 1 ELSE 0 END::smallint AS allcause_death_30d,
  COALESCE(inf.infection,0)::smallint                AS infection,
  COALESCE(mc.major_comp,0)::smallint                AS major_comp,
  CASE WHEN b.unitadmittime24 ~ '^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$'
         AND b.unitdischargetime24 ~ '^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$'
       THEN EXTRACT(EPOCH FROM (to_timestamp(b.unitdischargetime24,'YYYY-MM-DD HH24:MI:SS')
                              - to_timestamp(b.unitadmittime24,'YYYY-MM-DD HH24:MI:SS')))/86400.0
       ELSE NULL END::numeric(8,2) AS los_days,
  CASE WHEN b.unitadmittime24 ~ '^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$'
         AND b.unitdischargetime24 ~ '^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$'
       THEN EXTRACT(EPOCH FROM (to_timestamp(b.unitdischargetime24,'YYYY-MM-DD HH24:MI:SS')
                              - to_timestamp(b.unitadmittime24,'YYYY-MM-DD HH24:MI:SS')))/86400.0
       ELSE NULL END::numeric(8,2) AS icu_los_days
FROM base b
LEFT JOIN gc  gc  ON b.patientunitstayid = gc.patientunitstayid
LEFT JOIN inf inf ON b.patientunitstayid = inf.patientunitstayid
LEFT JOIN mc  mc  ON b.patientunitstayid = mc.patientunitstayid;

CREATE INDEX IF NOT EXISTS idx_eicu_cohort_subj ON ra_periop.cohort(subject_id);
CREATE INDEX IF NOT EXISTS idx_eicu_cohort_enc  ON ra_periop.cohort(encounter_id);
