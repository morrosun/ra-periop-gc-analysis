-- =============================================================
-- RA perioperative analysis cohort  ·  MIMIC-IV
-- Unit: HOSPITAL ADMISSION (surgical = has >=1 ICD-10-PCS procedure)
-- Staged via TEMP tables + indexes so the 20M-row prescriptions
-- and 6M-row diagnoses_icd are reached only by index lookup on the
-- ~2,235 RA surgical admissions (avoids full-table seq scans).
-- Schema: ra_periop.cohort  (created in the mimiciv database)
-- =============================================================
CREATE SCHEMA IF NOT EXISTS ra_periop;
DROP TABLE IF EXISTS ra_periop.cohort;

-- Indexes to make the RA-code filters fast (full scans of 6M-row diagnoses_icd
-- and procedures_icd were the bottleneck). Idempotent.
CREATE INDEX IF NOT EXISTS idx_dx_icdcode   ON mimiciv_hosp.diagnoses_icd(icd_code);
CREATE INDEX IF NOT EXISTS idx_proc_icdcode ON mimiciv_hosp.procedures_icd(icd_code);

-- Force index nested-loop joins (the 20M-row prescriptions would otherwise be
-- seq-scanned on this disk, taking minutes). base is tiny + indexed.
SET enable_seqscan = off;
SET enable_hashjoin = off;

-- Stage 1: RA surgical admissions (materialized, indexed)
CREATE TEMP TABLE base AS
WITH ra_adm AS (
  SELECT DISTINCT d.subject_id, d.hadm_id
  FROM mimiciv_hosp.diagnoses_icd d
  WHERE (d.icd_version = 10 AND (d.icd_code LIKE 'M05%' OR d.icd_code LIKE 'M06%'))
     OR (d.icd_version = 9  AND d.icd_code = '7140')
),
surg AS (
  SELECT DISTINCT p.subject_id, p.hadm_id
  FROM mimiciv_hosp.procedures_icd p
  JOIN ra_adm r ON p.subject_id = r.subject_id AND p.hadm_id = r.hadm_id
  WHERE p.icd_version = 10
)
SELECT r.subject_id, r.hadm_id,
       a.admittime, a.dischtime, a.deathtime,
       a.hospital_expire_flag, a.admission_type,
       pt.gender,
       CASE WHEN (pt.anchor_age + (EXTRACT(YEAR FROM a.admittime) - pt.anchor_year)) BETWEEN 0 AND 110
            THEN (pt.anchor_age + (EXTRACT(YEAR FROM a.admittime) - pt.anchor_year))::int
            ELSE pt.anchor_age END AS age_at_adm,
       pt.dod
FROM ra_adm r
JOIN surg s ON r.subject_id = s.subject_id AND r.hadm_id = s.hadm_id
JOIN mimiciv_hosp.admissions a  ON r.subject_id = a.subject_id AND r.hadm_id = a.hadm_id
JOIN mimiciv_hosp.patients   pt ON r.subject_id = pt.subject_id;
CREATE INDEX idx_base ON base(subject_id, hadm_id);
ANALYZE base;

-- Stage 2: GC + b/tsDMARD exposure (one pass over RA prescriptions only)
CREATE TEMP TABLE rx AS
SELECT p.subject_id, p.hadm_id,
  MAX(CASE WHEN
     p.drug ILIKE '%prednisone%' OR p.drug ILIKE '%prednisolone%' OR p.drug ILIKE '%methylprednisolone%'
  OR p.drug ILIKE '%hydrocortisone%' OR p.drug ILIKE '%cortisone%' OR p.drug ILIKE '%dexamethasone%'
  OR p.drug ILIKE '%triamcinolone%' OR p.drug ILIKE '%betamethasone%' THEN 1 ELSE 0 END) AS gc_use,
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
     ELSE 0 END) AS gc_dose_pred_eq_mg,
  BOOL_OR(p.drug ILIKE ANY (ARRAY['%methotrexate%','%leflunomide%','%azathioprine%','%mycophenolate%',
          '%cyclosporine%','%tacrolimus%','%hydroxychloroquine%','%sulfasalazine%','%mizoribine%',
          '%sirolimus%','%everolimus%'])) AS any_cs,
  BOOL_OR(p.drug ILIKE ANY (ARRAY['%tofacitinib%','%baricitinib%','%upadacitinib%'])) AS any_jaki,
  BOOL_OR(p.drug ILIKE ANY (ARRAY['%adalimumab%','%etanercept%','%infliximab%','%golimumab%','%certolizumab%'])) AS any_tnfi,
  BOOL_OR(p.drug ILIKE ANY (ARRAY['%tocilizumab%','%rituximab%','%abatacept%','%anakinra%',
          '%secukinumab%','%ustekinumab%','%basiliximab%'])) AS any_ili
FROM base b JOIN mimiciv_hosp.prescriptions p
     ON b.subject_id = p.subject_id AND b.hadm_id = p.hadm_id
GROUP BY p.subject_id, p.hadm_id;
CREATE INDEX idx_rx ON rx(subject_id, hadm_id);

-- Stage 3: in-hospital infection (ICD-10-CM) on RA diagnoses only
CREATE TEMP TABLE inf AS
SELECT d.subject_id, d.hadm_id, 1 AS infection
FROM base b JOIN mimiciv_hosp.diagnoses_icd d
     ON b.subject_id = d.subject_id AND b.hadm_id = d.hadm_id
WHERE d.icd_version = 10 AND (
    d.icd_code LIKE 'J1%' OR d.icd_code LIKE 'J85%' OR d.icd_code LIKE 'N39%' OR d.icd_code LIKE 'N10%'
 OR d.icd_code LIKE 'A40%' OR d.icd_code LIKE 'A41%' OR d.icd_code LIKE 'A49%' OR d.icd_code LIKE 'R65%'
 OR d.icd_code LIKE 'A04%' OR d.icd_code LIKE 'T81%' OR d.icd_code LIKE 'T79%' OR d.icd_code LIKE 'T84%'
 OR d.icd_code LIKE 'T82%' OR d.icd_code LIKE 'L03%' OR d.icd_code LIKE 'L08%' OR d.icd_code LIKE 'L02%'
 OR d.icd_code LIKE 'M86%' OR d.icd_code LIKE 'K65%' OR d.icd_code LIKE 'K61%')
GROUP BY d.subject_id, d.hadm_id;
CREATE INDEX idx_inf ON inf(subject_id, hadm_id);

-- Stage 4: major complications (ICD-10-CM) on RA diagnoses only
CREATE TEMP TABLE mc AS
SELECT d.subject_id, d.hadm_id, 1 AS major_comp
FROM base b JOIN mimiciv_hosp.diagnoses_icd d
     ON b.subject_id = d.subject_id AND b.hadm_id = d.hadm_id
WHERE d.icd_version = 10 AND (
    d.icd_code LIKE 'I21%' OR d.icd_code LIKE 'I22%'
 OR d.icd_code LIKE 'I6%'
 OR d.icd_code LIKE 'I26%'
 OR d.icd_code LIKE 'I80%' OR d.icd_code LIKE 'I82%'
 OR d.icd_code LIKE 'N17%' OR d.icd_code LIKE 'N99%'
 OR d.icd_code LIKE 'I46%'
 OR d.icd_code LIKE 'K92%'
 OR d.icd_code LIKE 'T81%')
GROUP BY d.subject_id, d.hadm_id;
CREATE INDEX idx_mc ON mc(subject_id, hadm_id);

-- Stage 5: ICU stay + ICU LOS
CREATE TEMP TABLE icu AS
SELECT i.subject_id, i.hadm_id, COUNT(*) icu_stays, SUM(i.los) icu_los_days
FROM base b JOIN mimiciv_icu.icustays i
     ON b.subject_id = i.subject_id AND b.hadm_id = i.hadm_id
GROUP BY i.subject_id, i.hadm_id;
CREATE INDEX idx_icu ON icu(subject_id, hadm_id);

-- Stage 6: primary procedure (for surgery category)
CREATE TEMP TABLE proc AS
SELECT p.subject_id, p.hadm_id, p.icd_code AS primary_proc
FROM (SELECT p.subject_id, p.hadm_id, p.icd_code,
             ROW_NUMBER() OVER (PARTITION BY p.subject_id, p.hadm_id ORDER BY p.seq_num) rn
      FROM base b JOIN mimiciv_hosp.procedures_icd p
           ON b.subject_id = p.subject_id AND b.hadm_id = p.hadm_id
      WHERE p.icd_version = 10) p
WHERE rn = 1;
CREATE INDEX idx_proc ON proc(subject_id, hadm_id);

-- Final assembly
CREATE TABLE ra_periop.cohort AS
SELECT
  'MIMIC-IV'::text                                   AS library,
  b.subject_id::text                                 AS subject_id,
  b.hadm_id::text                                    AS encounter_id,
  'admission'::text                                  AS encounter_unit,
  1                                                  AS surgery_confirmed,
  1                                                  AS ra_flag,
  b.age_at_adm::numeric(6,2)                         AS age,
  UPPER(b.gender)                                    AS sex,
  NULL::text                                         AS asa,
  NULL::smallint                                     AS asa_class,
  CASE WHEN b.admission_type ILIKE '%ELECTIVE%' THEN 0 ELSE 1 END AS emergency,
  CASE WHEN pr.primary_proc ~ '^0[LMQ]' THEN 'ortho_msk'
       WHEN pr.primary_proc IS NOT NULL THEN 'non_ortho'
       ELSE 'unknown' END                            AS surgery_category,
  pr.primary_proc                                    AS primary_proc,
  CASE WHEN icu.icu_stays IS NOT NULL THEN 1 ELSE 0 END AS icu_stay,
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
LEFT JOIN rx   rx  ON b.subject_id = rx.subject_id AND b.hadm_id = rx.hadm_id
LEFT JOIN inf  inf ON b.subject_id = inf.subject_id AND b.hadm_id = inf.hadm_id
LEFT JOIN mc   mc  ON b.subject_id = mc.subject_id AND b.hadm_id = mc.hadm_id
LEFT JOIN icu  icu ON b.subject_id = icu.subject_id AND b.hadm_id = icu.hadm_id
LEFT JOIN proc pr  ON b.subject_id = pr.subject_id AND b.hadm_id = pr.hadm_id;

CREATE INDEX idx_mimic_cohort_subj ON ra_periop.cohort(subject_id);
CREATE INDEX idx_mimic_cohort_enc  ON ra_periop.cohort(encounter_id);
