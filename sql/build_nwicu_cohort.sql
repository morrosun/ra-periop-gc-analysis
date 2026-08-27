-- =============================================================
-- RA perioperative analysis cohort  ·  NWICU (Northwestern ICU)
-- Unit of analysis: HOSPITAL ADMISSION (ICU admission).
-- NOTE: NWICU has NO procedure table (0 rows in procedures_icd),
--   so surgery cannot be confirmed -> surgery_confirmed = 0.
--   Treated as a peri-procedural ICU validation set.
-- Schema: ra_periop.cohort  (created in the nwicu database)
-- =============================================================
CREATE SCHEMA IF NOT EXISTS ra_periop;
DROP TABLE IF EXISTS ra_periop.cohort;

CREATE TABLE ra_periop.cohort AS
WITH ra_adm AS (
  SELECT DISTINCT d.subject_id, d.hadm_id
  FROM hosp.diagnoses_icd d
  WHERE d.icd_version = 10 AND (d.icd_code LIKE 'M05%' OR d.icd_code LIKE 'M06%')
),
base AS (
  SELECT r.subject_id, r.hadm_id,
         a.admittime, a.dischtime, a.deathtime,
         a.hospital_expire_flag, a.admission_type,
         pt.gender,
         CASE WHEN (pt.anchor_age + (EXTRACT(YEAR FROM a.admittime) - pt.anchor_year)) BETWEEN 0 AND 110
              THEN (pt.anchor_age + (EXTRACT(YEAR FROM a.admittime) - pt.anchor_year))::int
              ELSE pt.anchor_age END AS age_at_adm,
         pt.dod
  FROM ra_adm r
  JOIN hosp.admissions a  ON r.subject_id = a.subject_id AND r.hadm_id = a.hadm_id
  JOIN hosp.patients   pt ON r.subject_id = pt.subject_id
),
icu AS (
  SELECT i.subject_id, i.hadm_id, COUNT(*) icu_stays, SUM(i.los) icu_los_days
  FROM mimiciv_icu.icustays i JOIN base b ON i.subject_id = b.subject_id AND i.hadm_id = b.hadm_id
  GROUP BY i.subject_id, i.hadm_id
),
rx AS (
  SELECT p.subject_id, p.hadm_id,
    MAX(CASE WHEN
       p.drug ILIKE '%prednisone%' OR p.drug ILIKE '%prednisolone%' OR p.drug ILIKE '%methylprednisolone%'
    OR p.drug ILIKE '%hydrocortisone%' OR p.drug ILIKE '%cortisone%' OR p.drug ILIKE '%dexamethasone%'
    OR p.drug ILIKE '%triamcinolone%' OR p.drug ILIKE '%betamethasone%' THEN 1 ELSE 0 END) AS gc_use,
    SUM(CASE
       WHEN p.dose_val_rx ~ '^[0-9]+(\.[0-9]+)?$' AND p.dose_unit_rx ILIKE '%mg%' THEN
         CASE
           WHEN p.drug ILIKE '%prednisone%' OR p.drug ILIKE '%prednisolone%' THEN 1.0
           WHEN p.drug ILIKE '%methylprednisolone%' THEN 1.25
           WHEN p.drug ILIKE '%hydrocortisone%' OR p.drug ILIKE '%cortisone%' THEN 0.25
           WHEN p.drug ILIKE '%dexamethasone%' OR p.drug ILIKE '%betamethasone%' THEN 6.667
           WHEN p.drug ILIKE '%triamcinolone%' THEN 1.25
           ELSE 0 END * p.dose_val_rx::numeric
       ELSE 0 END) AS gc_dose_pred_eq_mg,
    BOOL_OR(p.drug ILIKE ANY (ARRAY['%methotrexate%','%leflunomide%','%azathioprine%','%mycophenolate%',
            '%cyclosporine%','%tacrolimus%','%hydroxychloroquine%','%sulfasalazine%','%mizoribine%',
            '%sirolimus%','%everolimus%'])) AS any_cs,
    BOOL_OR(p.drug ILIKE ANY (ARRAY['%tofacitinib%','%baricitinib%','%upadacitinib%'])) AS any_jaki,
    BOOL_OR(p.drug ILIKE ANY (ARRAY['%adalimumab%','%etanercept%','%infliximab%','%golimumab%','%certolizumab%'])) AS any_tnfi,
    BOOL_OR(p.drug ILIKE ANY (ARRAY['%tocilizumab%','%rituximab%','%abatacept%','%anakinra%',
            '%secukinumab%','%ustekinumab%','%basiliximab%'])) AS any_ili
  FROM base b JOIN hosp.prescriptions p
       ON b.subject_id = p.subject_id AND b.hadm_id = p.hadm_id
  GROUP BY p.subject_id, p.hadm_id
),
inf AS (
  SELECT d.subject_id, d.hadm_id, 1 AS infection
  FROM hosp.diagnoses_icd d JOIN base b ON d.subject_id = b.subject_id AND d.hadm_id = b.hadm_id
  WHERE d.icd_version = 10 AND (
      d.icd_code LIKE 'J1%' OR d.icd_code LIKE 'J85%' OR d.icd_code LIKE 'N39%' OR d.icd_code LIKE 'N10%'
   OR d.icd_code LIKE 'A40%' OR d.icd_code LIKE 'A41%' OR d.icd_code LIKE 'A49%' OR d.icd_code LIKE 'R65%'
   OR d.icd_code LIKE 'A04%' OR d.icd_code LIKE 'T81%' OR d.icd_code LIKE 'T79%' OR d.icd_code LIKE 'T84%'
   OR d.icd_code LIKE 'T82%' OR d.icd_code LIKE 'L03%' OR d.icd_code LIKE 'L08%' OR d.icd_code LIKE 'L02%'
   OR d.icd_code LIKE 'M86%' OR d.icd_code LIKE 'K65%' OR d.icd_code LIKE 'K61%')
  GROUP BY d.subject_id, d.hadm_id
),
mc AS (
  SELECT d.subject_id, d.hadm_id, 1 AS major_comp
  FROM hosp.diagnoses_icd d JOIN base b ON d.subject_id = b.subject_id AND d.hadm_id = b.hadm_id
  WHERE d.icd_version = 10 AND (
      d.icd_code LIKE 'I21%' OR d.icd_code LIKE 'I22%'
   OR d.icd_code LIKE 'I6%'
   OR d.icd_code LIKE 'I26%'
   OR d.icd_code LIKE 'I80%' OR d.icd_code LIKE 'I82%'
   OR d.icd_code LIKE 'N17%' OR d.icd_code LIKE 'N99%'
   OR d.icd_code LIKE 'I46%'
   OR d.icd_code LIKE 'K92%'
   OR d.icd_code LIKE 'T81%')
  GROUP BY d.subject_id, d.hadm_id
)
SELECT
  'NWICU'::text                                      AS library,
  b.subject_id::text                                 AS subject_id,
  b.hadm_id::text                                    AS encounter_id,
  'admission'::text                                  AS encounter_unit,
  0                                                  AS surgery_confirmed,
  1                                                  AS ra_flag,
  b.age_at_adm::numeric(6,2)                         AS age,
  UPPER(b.gender)                                    AS sex,
  NULL::text                                         AS asa,
  NULL::smallint                                     AS asa_class,
  CASE WHEN b.admission_type IN ('EMERGENCY','URGENT','TRAUMA') THEN 1
       WHEN b.admission_type ILIKE '%ELECTIVE%' THEN 0
       ELSE NULL END::smallint                       AS emergency,
  NULL::text                                         AS surgery_category,
  NULL::text                                         AS primary_proc,
  1                                                  AS icu_stay,
  COALESCE(rx.gc_use,0)::smallint                    AS gc_use,
  rx.gc_dose_pred_eq_mg::numeric                     AS gc_dose_pred_eq_mg,
  CASE WHEN rx.any_tnfi THEN 'TNFi'
       WHEN rx.any_ili  THEN 'ILi'
       WHEN rx.any_jaki THEN 'JAKi'
       WHEN rx.any_cs   THEN 'csDMARD'
       ELSE 'none' END                               AS dmard_class,
  CASE WHEN (rx.any_tnfi OR rx.any_ili OR rx.any_jaki) THEN 1 ELSE 0 END::smallint AS btsdmard_use,
  CASE WHEN b.hospital_expire_flag = 1 THEN 1 ELSE 0 END::smallint AS inhosp_death,
  CASE WHEN b.hospital_expire_flag = 1
        OR (b.dod IS NOT NULL AND b.dod <= b.dischtime + INTERVAL '30 days') THEN 1 ELSE 0 END::smallint AS allcause_death_30d,
  COALESCE(inf.infection,0)::smallint                AS infection,
  COALESCE(mc.major_comp,0)::smallint                AS major_comp,
  EXTRACT(DAY FROM (b.dischtime - b.admittime))::numeric(8,2) AS los_days,
  icu.icu_los_days::numeric(8,2)                     AS icu_los_days
FROM base b
LEFT JOIN icu  icu ON b.subject_id = icu.subject_id AND b.hadm_id = icu.hadm_id
LEFT JOIN rx   rx  ON b.subject_id = rx.subject_id AND b.hadm_id = rx.hadm_id
LEFT JOIN inf  inf ON b.subject_id = inf.subject_id AND b.hadm_id = inf.hadm_id
LEFT JOIN mc   mc  ON b.subject_id = mc.subject_id AND b.hadm_id = mc.hadm_id;

CREATE INDEX IF NOT EXISTS idx_nwicu_cohort_subj ON ra_periop.cohort(subject_id);
CREATE INDEX IF NOT EXISTS idx_nwicu_cohort_enc  ON ra_periop.cohort(encounter_id);
