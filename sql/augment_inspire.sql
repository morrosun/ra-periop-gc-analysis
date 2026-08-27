\set AUTOCOMMIT on
-- =============================================================
-- RA perioperative · INSPIRE covariate augmentation
-- Charlson: computed from inspire.diagnosis.icd10_cm (3-char, Quan 2005)
--           aggregated per subject_id (patient-level comorbidity)
-- PNI: from inspire.labs (item_name 'albumin','lymphocyte';
--       lymphocyte in x10^9/L -> x1000 to /uL for PNI)
-- NOTE: chart_time is relative-minutes text; pre-op vs post-op
--       unknown -> take worst (min) value as conservative baseline
-- =============================================================
CREATE SCHEMA IF NOT EXISTS ra_periop;
DROP TABLE IF EXISTS ra_periop.analytic;

CREATE TABLE ra_periop.analytic AS
WITH dx AS (
  SELECT d.subject_id,
         regexp_replace(d.icd10_cm, '\.', '', 'g') AS k
  FROM inspire.diagnosis d
  WHERE d.icd10_cm IS NOT NULL
    AND d.subject_id IN (SELECT subject_id::bigint FROM ra_periop.cohort)
),
cat AS (
  SELECT subject_id,
    MAX(CASE WHEN k LIKE 'I21%' OR k LIKE 'I22%' THEN 1 ELSE 0 END)                                              AS mi,
    MAX(CASE WHEN k IN ('I092','I110','I130','I132','I255') OR k LIKE 'I42%' OR k LIKE 'I43%' OR k LIKE 'I50%' THEN 1 ELSE 0 END) AS chf,
    MAX(CASE WHEN k LIKE 'I70%' OR k LIKE 'I71%' OR k LIKE 'I73%' OR k LIKE 'I77%' OR k LIKE 'I78%'
                  OR k IN ('I790','I792') OR k LIKE 'K55%' OR k IN ('Z958','Z959') THEN 1 ELSE 0 END)              AS pvd,
    MAX(CASE WHEN k LIKE 'G45%' OR k LIKE 'G46%' OR k LIKE 'I6%' OR k='H340' THEN 1 ELSE 0 END)                    AS cvd,
    MAX(CASE WHEN k LIKE 'F0%' OR k IN ('F051','G300','G311') THEN 1 ELSE 0 END)                                   AS dementia,
    MAX(CASE WHEN k LIKE 'J4%' OR k LIKE 'J6%' OR k IN ('J684','J701','J703','J704') THEN 1 ELSE 0 END)            AS copd,
    MAX(CASE WHEN k LIKE 'M05%' OR k LIKE 'M06%' OR k LIKE 'M08%' OR k LIKE 'M09%'
                  OR k LIKE 'M3%' OR k LIKE 'M4%' THEN 1 ELSE 0 END)                                               AS rheum,
    MAX(CASE WHEN k LIKE 'K2%' THEN 1 ELSE 0 END)                                                                  AS pud,
    MAX(CASE WHEN k LIKE 'B18%' OR k LIKE 'K7%' OR k='Z944' THEN 1 ELSE 0 END)                                     AS mildliver,
    MAX(CASE WHEN k LIKE 'G81%' OR k LIKE 'G82%' OR k LIKE 'G83%' OR k IN ('G041','G114','G801','G802') THEN 1 ELSE 0 END) AS hemi,
    MAX(CASE WHEN k LIKE 'N03%' OR k LIKE 'N05%' OR k LIKE 'N18%' OR k LIKE 'N19%' OR k LIKE 'N25%'
                  OR k IN ('I120','I130','I131','Z490','Z491','Z492','Z940','Z942','Z990','Z992') THEN 1 ELSE 0 END) AS renal,
    MAX(CASE WHEN k IN ('E100','E101','E106','E108','E109','E110','E111','E116','E118','E119',
                        'E120','E121','E126','E128','E129','E130','E131','E136','E138','E139',
                        'E140','E141','E146','E148','E149') THEN 1 ELSE 0 END)                                      AS dmsimple,
    MAX(CASE WHEN k IN ('E102','E103','E104','E105','E107','E112','E113','E114','E115','E117',
                        'E122','E123','E124','E125','E127','E132','E133','E134','E135','E137',
                        'E142','E143','E144','E145','E147') THEN 1 ELSE 0 END)                                      AS dmcomp,
    MAX(CASE WHEN k LIKE 'C%' OR k LIKE 'D0%' THEN 1 ELSE 0 END)                                                   AS anytumor,
    MAX(CASE WHEN k LIKE 'I85%' OR k LIKE 'I86%' OR k LIKE 'I98%' OR k LIKE 'K72%' OR k LIKE 'K76%' THEN 1 ELSE 0 END) AS modliver,
    MAX(CASE WHEN k LIKE 'C77%' OR k LIKE 'C78%' OR k LIKE 'C79%' OR k LIKE 'C80%' THEN 1 ELSE 0 END)               AS metas,
    MAX(CASE WHEN k LIKE 'B2%' THEN 1 ELSE 0 END)                                                                  AS aids
  FROM dx
  GROUP BY subject_id
),
charlson AS (
  SELECT subject_id,
     mi+chf+pvd+cvd+dementia+copd+rheum+pud+mildliver+dmsimple
   + 2*(hemi+renal+dmcomp+anytumor)
   + 3*(modliver) + 6*(metas+aids) AS charlson_index
  FROM cat
),
lab AS (
  SELECT subject_id,
         MIN(CASE WHEN item_name='albumin'    AND value ~ '^[0-9]+(\.[0-9]+)?$' THEN value::numeric END) AS albumin_min,
         MIN(CASE WHEN item_name='lymphocyte' AND value ~ '^[0-9]+(\.[0-9]+)?$' THEN value::numeric END) AS lymphocyte_min
  FROM inspire.labs
  WHERE item_name IN ('albumin','lymphocyte')
    AND subject_id IN (SELECT subject_id::bigint FROM ra_periop.cohort)
  GROUP BY subject_id
)
SELECT
  c.*,
  ch.charlson_index                                                                            AS charlson,
  lb.albumin_min                                                                               AS albumin,
  (lb.lymphocyte_min * 1000.0)                                                                 AS lymphocytes_per_ul,
  CASE WHEN lb.albumin_min IS NOT NULL AND lb.lymphocyte_min IS NOT NULL
       THEN ROUND((10.0*lb.albumin_min + 0.005*lb.lymphocyte_min*1000.0)::numeric, 2)
       END                                                                                     AS pni,
  NULL::numeric                                                                                AS creatinine
FROM ra_periop.cohort c
LEFT JOIN charlson ch ON c.subject_id::bigint = ch.subject_id
LEFT JOIN lab lb      ON c.subject_id::bigint = lb.subject_id;

SELECT 'INSPIRE analytic rows: ' || count(*) FROM ra_periop.analytic;
SELECT 'INSPIRE with PNI: '     || count(*) FROM ra_periop.analytic WHERE pni IS NOT NULL;
SELECT 'INSPIRE with charlson: '|| count(*) FROM ra_periop.analytic WHERE charlson IS NOT NULL;
