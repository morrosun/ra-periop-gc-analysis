-- =============================================================
-- RA perioperative analysis cohort  ·  INSPIRE (Korean HIRA-OPS)
-- Unit of analysis: OPERATION  (restricted to SINGLE-OPERATION RA
--   patients so medication exposure can be cleanly attributed to the
--   one admission; multi-op patients excluded -- see memo)
-- Time scale: relative minutes from hospital admission
--   (admission_time = 0 reference). Validated 2026-08-26.
-- Schema: ra_periop.cohort  (created in the inspire database)
-- =============================================================
CREATE SCHEMA IF NOT EXISTS ra_periop;
DROP TABLE IF EXISTS ra_periop.cohort;

CREATE TABLE ra_periop.cohort AS
WITH ra_subj AS (
  SELECT DISTINCT subject_id FROM inspire.diagnosis
  WHERE icd10_cm LIKE 'M05%' OR icd10_cm LIKE 'M06%'
),
ops AS (                        -- all RA operations with valid time window
  SELECT o.op_id, o.subject_id, o.hadm_id,
         o.age, o.sex, o.asa, o.emop, o.department, o.antype, o.icd10_pcs,
         o.inhosp_death_time, o.allcause_death_time,
         CAST(o.admission_time  AS bigint) AS adm,
         CAST(o.discharge_time  AS bigint) AS dis,
         CAST(o.opdate          AS bigint) AS op_min,
         CAST(NULLIF(o.icuin_time,'')  AS bigint) AS icuin,
         CAST(NULLIF(o.icuout_time,'') AS bigint) AS icuout
  FROM inspire.operations o
  WHERE o.subject_id IN (SELECT subject_id FROM ra_subj)
    AND o.opdate ~ '^[0-9]+$'          AND CAST(o.opdate AS bigint) BETWEEN 1 AND 60*1440
    AND o.admission_time ~ '^[0-9]+$'  AND o.discharge_time ~ '^[0-9]+$'
    AND CAST(o.discharge_time AS bigint) > CAST(o.admission_time AS bigint)
),
single AS (                     -- keep only single-operation RA patients
  SELECT subject_id FROM ops GROUP BY subject_id HAVING count(*) = 1
),
base AS ( SELECT o.* FROM ops o JOIN single s ON o.subject_id = s.subject_id ),
rx AS (                         -- GC + b/tsDMARD exposure within admission window
  SELECT m.subject_id,
    MAX(CASE WHEN m.atc_code LIKE 'H02%' THEN 1 ELSE 0 END) AS gc_use,
    BOOL_OR((m.atc_code LIKE 'L04AA%' AND m.atc_code <> 'L04AA29')
            OR m.atc_code LIKE 'L04AX%' OR m.atc_code LIKE 'L04AD%') AS any_cs,
    BOOL_OR(m.atc_code = 'L04AA29')                       AS any_jaki,
    BOOL_OR(m.atc_code LIKE 'L04AB%')                     AS any_tnfi,
    BOOL_OR(m.atc_code IN ('L04AC02','L04AC07'))          AS any_ili
  FROM inspire.medications m JOIN base b ON m.subject_id = b.subject_id
  WHERE m.chart_time ~ '^[0-9]+$'
    AND CAST(m.chart_time AS bigint) BETWEEN b.adm AND b.dis
  GROUP BY m.subject_id
),
inf AS (                        -- in-hospital infection (ICD-10-CM)
  SELECT d.subject_id, 1 AS infection
  FROM inspire.diagnosis d JOIN base b ON d.subject_id = b.subject_id
  WHERE d.icd10_cm LIKE 'J1%' OR d.icd10_cm LIKE 'J85%' OR d.icd10_cm LIKE 'N39%' OR d.icd10_cm LIKE 'N10%'
     OR d.icd10_cm LIKE 'A40%' OR d.icd10_cm LIKE 'A41%' OR d.icd10_cm LIKE 'A49%' OR d.icd10_cm LIKE 'R65%'
     OR d.icd10_cm LIKE 'A04%' OR d.icd10_cm LIKE 'T81%' OR d.icd10_cm LIKE 'T79%' OR d.icd10_cm LIKE 'T84%'
     OR d.icd10_cm LIKE 'T82%' OR d.icd10_cm LIKE 'L03%' OR d.icd10_cm LIKE 'L08%' OR d.icd10_cm LIKE 'L02%'
     OR d.icd10_cm LIKE 'M86%' OR d.icd10_cm LIKE 'K65%' OR d.icd10_cm LIKE 'K61%'
  GROUP BY d.subject_id
),
mc AS (                         -- major complications (ICD-10-CM)
  SELECT d.subject_id, 1 AS major_comp
  FROM inspire.diagnosis d JOIN base b ON d.subject_id = b.subject_id
  WHERE d.icd10_cm LIKE 'I21%' OR d.icd10_cm LIKE 'I22%'
     OR d.icd10_cm LIKE 'I6%'
     OR d.icd10_cm LIKE 'I26%'
     OR d.icd10_cm LIKE 'I80%' OR d.icd10_cm LIKE 'I82%'
     OR d.icd10_cm LIKE 'N17%' OR d.icd10_cm LIKE 'N99%'
     OR d.icd10_cm LIKE 'I46%'
     OR d.icd10_cm LIKE 'K92%'
     OR d.icd10_cm LIKE 'T81%'
  GROUP BY d.subject_id
)
SELECT
  'INSPIRE'::text                                    AS library,
  b.subject_id::text                                 AS subject_id,
  b.op_id::text                                      AS encounter_id,
  'operation'::text                                  AS encounter_unit,
  1                                                  AS surgery_confirmed,
  1                                                  AS ra_flag,
  b.age::numeric(6,2)                                AS age,
  UPPER(b.sex)                                       AS sex,
  NULLIF(b.asa,'')                                   AS asa,
  NULLIF(regexp_replace(b.asa,'[^0-9]',''),'')::smallint AS asa_class,
  CASE b.emop WHEN '1' THEN 1 WHEN '0' THEN 0 ELSE NULL END::smallint AS emergency,
  CASE WHEN b.department ILIKE '%ortho%' THEN 'ortho_msk'
       ELSE 'unknown' END                            AS surgery_category,
  b.icd10_pcs                                        AS primary_proc,
  CASE WHEN b.icuin IS NOT NULL AND b.icuin > 0 THEN 1 ELSE 0 END::smallint AS icu_stay,
  COALESCE(rx.gc_use,0)::smallint                    AS gc_use,
  NULL::numeric                                      AS gc_dose_pred_eq_mg,
  CASE WHEN rx.any_tnfi THEN 'TNFi'
       WHEN rx.any_ili  THEN 'ILi'
       WHEN rx.any_jaki THEN 'JAKi'
       WHEN rx.any_cs   THEN 'csDMARD'
       ELSE 'none' END                               AS dmard_class,
  CASE WHEN (rx.any_tnfi OR rx.any_ili OR rx.any_jaki) THEN 1 ELSE 0 END::smallint AS btsdmard_use,
  CASE WHEN b.inhosp_death_time IS NOT NULL AND b.inhosp_death_time ~ '^[0-9]+$'
        AND CAST(b.inhosp_death_time AS bigint) > 0 THEN 1 ELSE 0 END::smallint AS inhosp_death,
  CASE WHEN (b.inhosp_death_time IS NOT NULL AND b.inhosp_death_time ~ '^[0-9]+$'
             AND CAST(b.inhosp_death_time AS bigint) > 0)
        OR (b.allcause_death_time IS NOT NULL AND b.allcause_death_time ~ '^[0-9]+$'
             AND CAST(b.allcause_death_time AS bigint) <= b.dis + 30*1440)
       THEN 1 ELSE 0 END::smallint                   AS allcause_death_30d,
  COALESCE(inf.infection,0)::smallint                AS infection,
  COALESCE(mc.major_comp,0)::smallint                AS major_comp,
  ((b.dis - b.adm)/1440.0)::numeric(8,2)             AS los_days,
  CASE WHEN b.icuin IS NOT NULL AND b.icuout IS NOT NULL AND b.icuout > b.icuin
       THEN ((b.icuout - b.icuin)/1440.0)::numeric(8,2) ELSE NULL END AS icu_los_days
FROM base b
LEFT JOIN rx  rx  ON b.subject_id = rx.subject_id
LEFT JOIN inf inf ON b.subject_id = inf.subject_id
LEFT JOIN mc  mc  ON b.subject_id = mc.subject_id;

CREATE INDEX IF NOT EXISTS idx_inspire_cohort_subj ON ra_periop.cohort(subject_id);
CREATE INDEX IF NOT EXISTS idx_inspire_cohort_enc  ON ra_periop.cohort(encounter_id);
