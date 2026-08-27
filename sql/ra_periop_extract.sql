-- ============================================================================
-- RA 围术期研究：四库统一提取 SQL 骨架
-- 生成 2026-08-26；基于本机 PostgreSQL 18 直接 SQL 核查（口令已授权）
-- 四个库各自独立运行对应 SECTION（psql -d <db> -f 本文件 或直接粘贴）
--
-- 统一输出列（四库一致，便于后续纵向堆叠/合并分析）：
--   library         TEXT    库名
--   subject_id      BIGINT  患者
--   encounter_id    BIGINT  就诊/手术/ICU 停留标识
--   ra_flag         BOOL    是否满足 RA 定义
--   gc_use          BOOL    围术期是否使用糖皮质激素
--   gc_dose_eq_mg   NUMERIC 糖皮质激素累计 prednisone 等效剂量(mg)；无剂量库为 NULL
--   dmard_class     TEXT    none / csDMARD / tsDMARD_JAKi / bDMARD_TNFi / bDMARD_nonTNF
--   inhosp_death    BOOL    院内死亡
--   allcause_death  BOOL    全因死亡（不可用库为 NULL）
--   major_comp      BOOL    主要并发症(composite)（另行定义，骨架留 NULL）
--   infection       BOOL    术后感染（ICD/微生物学组合，骨架留 NULL）
--
-- 经核查确认的各库关键事实（详见 FEASIBILITY_RA_PERIOP.md 与各 SECTION 注释）：
--   INSPIRE : time=分钟相对入院(adm=0)；medications 无剂量、无入院键；TNFi(L04AB) 缺失；
--             多手术患者用药无法归因 → 主分析限单手术患者；drug_name 含 CP949 字节，
--             只读 ATC 或 SET CLIENT_ENCODING TO 'EUC_KR' 后读。
--   MIMIC-IV: RA 3511人/8208次；GC 处方 1557人、8279行 100% 带 dose_val_rx → 剂量-反应主库。
--   NWICU   : 真实处方表是 hosp.prescriptions（mimiciv_hosp.prescriptions 为空！）；
--             RA 133人/170次；GC 71、csDMARD 41、TNFi 2、JAKi 2、bDMARD 1。
--   eICU    : diagnosis.diagnosisstring ILIKE '%rheumatoid%' → 250 stays；
--             ICU 医嘱基本不含慢性 DMARD（仅 GC 56 stays），故 b/tsDMARD 不可用。
-- ============================================================================


-- ============================ SECTION 1: INSPIRE ============================
-- 围术期主库。暴露=ATC 编码（避免 CP949 药名）。时间=分钟相对入院。
-- 主分析限单手术患者以保证用药→手术的干净归因。
-- 运行前如需读 drug_name 文本： SET CLIENT_ENCODING TO 'EUC_KR';
WITH ra_subj AS (
  SELECT DISTINCT subject_id FROM inspire.diagnosis
  WHERE icd10_cm LIKE 'M05%' OR icd10_cm LIKE 'M06%'
),
ra_op AS (
  SELECT o.subject_id, o.op_id,
         CAST(o.opdate AS bigint)            AS op_min,
         CAST(o.admission_time AS bigint)    AS adm_min,
         CAST(o.discharge_time AS bigint)    AS dis_min,
         (o.inhosp_death_time IS NOT NULL AND o.inhosp_death_time <> '')   AS inhosp_death,
         (o.allcause_death_time IS NOT NULL AND o.allcause_death_time <> '') AS allcause_death
  FROM inspire.operations o
  JOIN ra_subj r ON o.subject_id = r.subject_id
  WHERE o.opdate ~ '^[0-9]+$'
    AND CAST(o.opdate AS bigint) BETWEEN 1 AND 60*1440          -- 排除缺失(=0)与异常(>60天)
    AND o.admission_time ~ '^[0-9]+$' AND o.discharge_time ~ '^[0-9]+$'
    AND CAST(o.discharge_time AS bigint) > CAST(o.admission_time AS bigint)
),
single AS (SELECT subject_id FROM ra_op GROUP BY subject_id HAVING count(*) = 1),
clean_op AS (SELECT * FROM ra_op WHERE subject_id IN (SELECT subject_id FROM single)),
exp AS (
  SELECT c.op_id,
         bool_or(m.atc_code LIKE 'H02%')                                                  AS gc_use,
         bool_or(m.atc_code LIKE 'L04%')                                                  AS dmard_any,
         bool_or(m.atc_code LIKE 'L04AA%' OR m.atc_code LIKE 'L04AX%')                    AS csdmard,
         bool_or(m.atc_code = 'L04AA29')                                                  AS jakii,
         bool_or(m.atc_code IN ('L04AC02','L04AC07'))                                     AS bdmard_nontnf,
         bool_or(m.atc_code LIKE 'L04AB%')                                                AS bdmard_tnf  -- 恒为 false（TNFi 缺失）
  FROM clean_op c
  JOIN inspire.medications m
    ON m.subject_id = c.subject_id
   AND m.chart_time ~ '^[0-9]+$'
   AND CAST(m.chart_time AS bigint) BETWEEN c.adm_min AND c.dis_min       -- 仅取本次入院窗口内用药
  GROUP BY c.op_id
)
SELECT 'INSPIRE' AS library, c.subject_id, c.op_id AS encounter_id,
       true AS ra_flag,
       COALESCE(e.gc_use, false) AS gc_use,
       NULL::numeric            AS gc_dose_eq_mg,                          -- INSPIRE 无剂量字段
       CASE WHEN e.bdmard_tnf     THEN 'bDMARD_TNFi'
            WHEN e.bdmard_nontnf  THEN 'bDMARD_nonTNF'
            WHEN e.jakii          THEN 'tsDMARD_JAKi'
            WHEN e.csdmard        THEN 'csDMARD'
            WHEN e.dmard_any      THEN 'other_DMARD'
            ELSE 'none' END       AS dmard_class,
       c.inhosp_death, c.allcause_death,
       NULL::boolean AS major_comp,
       NULL::boolean AS infection
FROM clean_op c
LEFT JOIN exp e ON e.op_id = c.op_id
ORDER BY c.subject_id, c.op_id;


-- ============================ SECTION 2: MIMIC-IV ============================
-- 剂量-反应主库。GC 用 dose_val_rx 转 prednisone 等效剂量；b/tsDMARD 用药名识别。
-- 如需"外科亚群"，可 JOIN 你已有的 surg 模式或 procedures_icd/DRG 外科定义（见注释）。
WITH ra_adm AS (
  SELECT DISTINCT subject_id, hadm_id FROM mimiciv_hosp.diagnoses_icd
  WHERE (icd_version = 10 AND (icd_code LIKE 'M05%' OR icd_code LIKE 'M06%'))
     OR (icd_version = 9  AND icd_code = '7140')
),
gc_rx AS (
  SELECT p.subject_id, p.hadm_id,
         (CASE WHEN p.dose_val_rx ~ '^[0-9]+(\.[0-9]+)?$' THEN p.dose_val_rx::numeric END) *
         CASE WHEN p.drug ILIKE '%dexamethasone%'    THEN 6.67
              WHEN p.drug ILIKE '%methylprednisolone%' THEN 1.25
              WHEN p.drug ILIKE '%prednisolone%'     THEN 1.0
              WHEN p.drug ILIKE '%prednisone%'       THEN 1.0
              WHEN p.drug ILIKE '%hydrocortisone%'   THEN 4.0
              WHEN p.drug ILIKE '%betamethasone%'    THEN 2.5
              ELSE NULL END AS pe_mg
  FROM mimiciv_hosp.prescriptions p
  JOIN ra_adm a ON p.subject_id = a.subject_id AND p.hadm_id = a.hadm_id
  WHERE p.drug ILIKE ANY (ARRAY['%prednisone%','%methylprednisolone%','%hydrocortisone%',
                               '%dexamethasone%','%prednisolone%','%betamethasone%'])
    AND p.dose_val_rx ~ '^[0-9]+(\.[0-9]+)?$' AND p.dose_unit_rx ILIKE '%mg%'
),
gc_agg AS (
  SELECT subject_id, hadm_id, count(*) AS gc_rows, SUM(pe_mg) AS gc_pe_total
  FROM gc_rx GROUP BY subject_id, hadm_id
),
dmard AS (
  SELECT p.subject_id, p.hadm_id,
    CASE WHEN p.drug ILIKE ANY(ARRAY['%adalimumab%','%etanercept%','%infliximab%','%golimumab%','%certolizumab%']) THEN 'bDMARD_TNFi'
         WHEN p.drug ILIKE ANY(ARRAY['%tocilizumab%','%rituximab%','%abatacept%','%secukinumab%','%ustekinumab%','%anakinra%']) THEN 'bDMARD_nonTNF'
         WHEN p.drug ILIKE ANY(ARRAY['%tofacitinib%','%baricitinib%','%upadacitinib%']) THEN 'tsDMARD_JAKi'
         WHEN p.drug ILIKE ANY(ARRAY['%methotrexate%','%leflunomide%','%azathioprine%','%mycophenolate%','%cyclosporine%','%tacrolimus%','%hydroxychloroquine%','%sulfasalazine%']) THEN 'csDMARD'
         ELSE NULL END AS dmard_class
  FROM mimiciv_hosp.prescriptions p
  JOIN ra_adm a ON p.subject_id = a.subject_id AND p.hadm_id = a.hadm_id
),
dmard_prio AS (
  SELECT subject_id, hadm_id, dmard_class,
         ROW_NUMBER() OVER (PARTITION BY subject_id, hadm_id
            ORDER BY CASE dmard_class
              WHEN 'bDMARD_TNFi'   THEN 1 WHEN 'bDMARD_nonTNF' THEN 2
              WHEN 'tsDMARD_JAKi'  THEN 3 WHEN 'csDMARD'       THEN 4 END) rn
  FROM dmard WHERE dmard_class IS NOT NULL
)
SELECT 'MIMIC-IV' AS library, a.subject_id, a.hadm_id AS encounter_id,
       true AS ra_flag,
       (g.hadm_id IS NOT NULL) AS gc_use,
       ROUND(g.gc_pe_total)    AS gc_dose_eq_mg,
       COALESCE(d.dmard_class, 'none') AS dmard_class,
       ad.hospital_expire_flag::boolean AS inhosp_death,
       NULL::boolean           AS allcause_death,
       NULL::boolean           AS major_comp,
       NULL::boolean           AS infection
FROM ra_adm a
JOIN mimiciv_hosp.admissions ad ON a.hadm_id = ad.hadm_id
LEFT JOIN gc_agg g ON g.subject_id = a.subject_id AND g.hadm_id = a.hadm_id
LEFT JOIN dmard_prio d ON d.subject_id = a.subject_id AND d.hadm_id = a.hadm_id AND d.rn = 1
ORDER BY a.subject_id, a.hadm_id;


-- ============================ SECTION 3: NWICU ============================
-- 注意：真实处方/诊断/入院表在 hosp 模式（mimiciv_hosp.prescriptions 为空！）
-- RA 133人/170次；GC 71、csDMARD 41、TNFi 2、JAKi 2、bDMARD 1（验证集，n 小）。
WITH ra_adm AS (
  SELECT DISTINCT subject_id, hadm_id FROM hosp.diagnoses_icd
  WHERE icd_version = 10 AND (icd_code LIKE 'M05%' OR icd_code LIKE 'M06%')
),
gc_rx AS (
  SELECT p.subject_id, p.hadm_id,
         (CASE WHEN p.dose_val_rx ~ '^[0-9]+(\.[0-9]+)?$' THEN p.dose_val_rx::numeric END) *
         CASE WHEN p.drug ILIKE '%dexamethasone%'    THEN 6.67
              WHEN p.drug ILIKE '%methylprednisolone%' THEN 1.25
              WHEN p.drug ILIKE '%prednisolone%'     THEN 1.0
              WHEN p.drug ILIKE '%prednisone%'       THEN 1.0
              WHEN p.drug ILIKE '%hydrocortisone%'   THEN 4.0
              WHEN p.drug ILIKE '%betamethasone%'    THEN 2.5
              ELSE NULL END AS pe_mg
  FROM hosp.prescriptions p
  JOIN ra_adm a ON p.subject_id = a.subject_id AND p.hadm_id = a.hadm_id
  WHERE p.drug ILIKE ANY (ARRAY['%prednisone%','%methylprednisolone%','%hydrocortisone%',
                               '%dexamethasone%','%prednisolone%','%betamethasone%'])
    AND p.dose_val_rx ~ '^[0-9]+(\.[0-9]+)?$' AND p.dose_unit_rx ILIKE '%mg%'
),
gc_agg AS (
  SELECT subject_id, hadm_id, count(*) AS gc_rows, SUM(pe_mg) AS gc_pe_total
  FROM gc_rx GROUP BY subject_id, hadm_id
),
dmard AS (
  SELECT p.subject_id, p.hadm_id,
    CASE WHEN p.drug ILIKE ANY(ARRAY['%adalimumab%','%etanercept%','%infliximab%','%golimumab%','%certolizumab%']) THEN 'bDMARD_TNFi'
         WHEN p.drug ILIKE ANY(ARRAY['%tocilizumab%','%rituximab%','%abatacept%','%secukinumab%','%ustekinumab%','%anakinra%']) THEN 'bDMARD_nonTNF'
         WHEN p.drug ILIKE ANY(ARRAY['%tofacitinib%','%baricitinib%','%upadacitinib%']) THEN 'tsDMARD_JAKi'
         WHEN p.drug ILIKE ANY(ARRAY['%methotrexate%','%leflunomide%','%azathioprine%','%mycophenolate%','%cyclosporine%','%tacrolimus%','%hydroxychloroquine%','%sulfasalazine%']) THEN 'csDMARD'
         ELSE NULL END AS dmard_class
  FROM hosp.prescriptions p
  JOIN ra_adm a ON p.subject_id = a.subject_id AND p.hadm_id = a.hadm_id
),
dmard_prio AS (
  SELECT subject_id, hadm_id, dmard_class,
         ROW_NUMBER() OVER (PARTITION BY subject_id, hadm_id
            ORDER BY CASE dmard_class
              WHEN 'bDMARD_TNFi' THEN 1 WHEN 'bDMARD_nonTNF' THEN 2
              WHEN 'tsDMARD_JAKi' THEN 3 WHEN 'csDMARD' THEN 4 END) rn
  FROM dmard WHERE dmard_class IS NOT NULL
)
SELECT 'NWICU' AS library, a.subject_id, a.hadm_id AS encounter_id,
       true AS ra_flag,
       (g.hadm_id IS NOT NULL) AS gc_use,
       ROUND(g.gc_pe_total)    AS gc_dose_eq_mg,
       COALESCE(d.dmard_class, 'none') AS dmard_class,
       ad.hospital_expire_flag::boolean AS inhosp_death,
       NULL::boolean           AS allcause_death,
       NULL::boolean           AS major_comp,
       NULL::boolean           AS infection
FROM ra_adm a
JOIN hosp.admissions ad ON a.hadm_id = ad.hadm_id
LEFT JOIN gc_agg g ON g.subject_id = a.subject_id AND g.hadm_id = a.hadm_id
LEFT JOIN dmard_prio d ON d.subject_id = a.subject_id AND d.hadm_id = a.hadm_id AND d.rn = 1
ORDER BY a.subject_id, a.hadm_id;


-- ============================ SECTION 4: eICU ============================
-- RA 经 diagnosis.diagnosisstring ILIKE '%rheumatoid%' → 250 stays。
-- ICU 医嘱基本不含慢性 DMARD（仅 GC 可用）；死亡用 patient.unitdischargestatus。
-- 时间=相对 ICU 入院分钟(drugstartoffset)，便于围术期窗口化。
WITH ra_stay AS (
  SELECT DISTINCT patientunitstayid FROM eicu_crd.diagnosis
  WHERE diagnosisstring ILIKE '%rheumatoid%'
),
gc AS (
  SELECT m.patientunitstayid,
         bool_or(m.drugname ILIKE ANY(ARRAY['%prednisone%','%methylprednisolone%','%hydrocortisone%',
                                        '%dexamethasone%','%prednisolone%','%betamethasone%'])) AS gc_use
  FROM eicu_crd.medication m
  JOIN ra_stay r ON m.patientunitstayid = r.patientunitstayid
  GROUP BY m.patientunitstayid
)
SELECT 'eICU' AS library, s.patientunitstayid AS subject_id, s.patientunitstayid AS encounter_id,
       true AS ra_flag,
       COALESCE(g.gc_use, false) AS gc_use,
       NULL::numeric AS gc_dose_eq_mg,                            -- dosage 为文本，需另解析
       'NA_ICU'::text AS dmard_class,                            -- ICU 不含慢性 DMARD
       (p.unitdischargestatus = 'Expired') AS inhosp_death,
       NULL::boolean AS allcause_death,
       NULL::boolean AS major_comp,
       NULL::boolean AS infection
FROM ra_stay s
JOIN eicu_crd.patient p ON p.patientunitstayid = s.patientunitstayid
LEFT JOIN gc g ON g.patientunitstayid = s.patientunitstayid
ORDER BY s.patientunitstayid;
