"""Mock of the All of Us Researcher Workbench: same tables, same column names, same traps, mock data.

WHAT IT IS FOR
  You run the BigQuery SQL and pandas code you will write in the Workbench on your own machine (or in Colab),
  in exactly the same form, without touching real data. When the same cell is moved to the Workbench it runs
  without any change: run_sql() checks the environment and goes to BigQuery in the Workbench, to DuckDB otherwise.

USAGE
    pip install duckdb sqlglot pandas
    import aou_mock
    con = aou_mock.connect()                 # builds the mock CDR (deterministic, seed=20261014)
    df  = aou_mock.run_sql(f'''
        SELECT measurement_concept_id, unit_concept_id, COUNT(*) AS n
        FROM `{aou_mock.CDR}.measurement`
        WHERE measurement_concept_id IN (3016723, 3020564)
        GROUP BY 1, 2 ORDER BY 1, 2''')
    truth = aou_mock.mock_truth()           # per-person PLANTED ground truth (exists only in the mock)
    aou_mock.self_test()                    # the mock's self-test

SAME AS THE REAL THING
  * Table and column names: OMOP CDM 5.3.1 + All of Us additions (person.sex_at_birth_concept_id, aou_death,
    cb_search_person). Source: Controlled Tier v9 (C2025Q4R6) data dictionary and the CDM 5.3 specification.
  * Table addressing: `WORKSPACE_CDR.table` (project.dataset.table). Locally CDR = "aou-cdr-mock.C2025Q4R6".
  * Documented traps (EXTRACTION_CODE_REQUEST_2026-10-03): mixed units (eGFR sitting in the creatinine slot,
    µmol/L, NULL and 0 units), censoring operators on measurements (<, <=, >, >=), visit_occurrence_id /
    measurement_id larger than 2^53, half of the inpatient stays lacking a time of day (00:00:00),
    midnight-stamped measurements, stays with a NULL or reversed end, multiple death records for the same person
    (some undated), LOINC answer concepts for sex (45878463 / 45880669; no 8532 / 8507),
    "Dependence on renal dialysis" living in the observation table, cb_search_person genomic flags all 0.
  * Concept IDs: real IDs verified against the OHDSI WebAPI on 29 Sep / 14 Oct 2026. Concepts whose real ID
    could not be verified are flagged with an ID above 99_000_000 and vocabulary_id = 'MOCK': do not rely on
    these in your code; resolve the real one from the concept table in the Workbench.

NOT THE SAME AS THE REAL THING
  * The data is mock: ~400 persons, planted according to scenarios. Distributions (unit shares, AKI frequency) are made up.
  * concept / concept_ancestor contain only the few hundred rows this project needs; in reality there are
    millions of rows and far more descendants. That is why your code MUST PRINT its descendant list.
  * Genomic files (bucket) are not in this mock.
  * BigQuery SQL is translated to DuckDB with sqlglot. Not every BigQuery function translates; if you use
    something that cannot be translated, run_sql raises an explicit error. Translated SQL still does NOT PROVE that BigQuery accepts it.
  * Returned DataFrame dtypes mimic BigQuery to_dataframe(): INT64 -> nullable Int64, TIMESTAMP -> UTC
    tz-aware datetime64, DATE -> datetime64 (dbdate in BigQuery), STRING -> object/str.
"""
import datetime as dt
import math
import os

import numpy as np
import pandas as pd

CDR_DEFAULT = "aou-cdr-mock.C2025Q4R6"
CDR = os.environ.get("WORKSPACE_CDR", CDR_DEFAULT)
PROJECT = os.environ.get("GOOGLE_PROJECT", "aou-mock-project")
SEED = 20261014
N_PERSONS = 400
MOCK_ID_FLOOR = 99_000_000            # IDs above this are made up (vocabulary_id = 'MOCK')
DATA_CUTOFF = dt.date(2024, 6, 30)    # clinical data cutoff date of the mock CDR
BIG_ID_LOW, BIG_ID_HIGH = 1_000_000_000_000_000, 76_000_000_000_000_000   # visit/measurement ID range (real: 1.0e15 - 7.6e16)

_CON = None   # default connection after connect()

# ============================================================================================
# 1. VOCABULARY (concept). Real IDs verified against the OHDSI WebAPI; MOCK ones are flagged.
# ============================================================================================
# (concept_id, concept_name, domain_id, vocabulary_id, concept_class_id, standard_concept, concept_code)
CONCEPTS = [
    # --- measurements ---
    (3016723, "Creatinine [Mass/volume] in Serum or Plasma", "Measurement", "LOINC", "Lab Test", "S", "2160-0"),
    (3020564, "Creatinine [Moles/volume] in Serum or Plasma", "Measurement", "LOINC", "Lab Test", "S", "14682-9"),
    (1619025, "Glomerular filtration rate/1.73 sq M.predicted [Volume Rate/Area] in Serum, Plasma or Blood by Creatinine-based formula (CKD-EPI 2021)", "Measurement", "LOINC", "Lab Test", "S", "98979-8"),
    (3033837, "Creatinine/Urea nitrogen ratio (per example file; name not re-verified)", "Measurement", "LOINC", "Lab Test", "S", "MOCK-NAME"),
    (3034485, "Albumin/Creatinine [Mass Ratio] in Urine", "Measurement", "LOINC", "Lab Test", "S", "9318-7"),
    (3001802, "Microalbumin/Creatinine [Mass Ratio] in Urine", "Measurement", "LOINC", "Lab Test", "S", "14959-1"),
    (3020682, "Albumin/Creatinine [Ratio] in Urine", "Measurement", "LOINC", "Lab Test", "S", "32294-1"),
    (3022826, "Microalbumin/Creatinine [Ratio] in Urine", "Measurement", "LOINC", "Lab Test", "S", "30000-4"),
    (3001582, "Protein/Creatinine [Mass Ratio] in Urine", "Measurement", "LOINC", "Lab Test", "S", "2890-2"),
    (3045462, "Protein/Creatinine [Ratio] in Urine", "Measurement", "LOINC", "Lab Test", "S", "34366-5"),
    (3014051, "Protein [Presence] in Urine by Test strip", "Measurement", "LOINC", "Lab Test", "S", "20454-5"),
    (3005897, "Protein [Mass/volume] in Urine by Test strip", "Measurement", "LOINC", "Lab Test", "S", "5804-0"),
    (3038553, "Body mass index (BMI) [Ratio]", "Measurement", "LOINC", "Clinical Observation", "S", "39156-5"),
    (3036277, "Body height", "Measurement", "LOINC", "Clinical Observation", "S", "8302-2"),
    (3023540, "Body height Measured", "Measurement", "LOINC", "Clinical Observation", "S", "3137-7"),
    (3025315, "Body weight", "Measurement", "LOINC", "Clinical Observation", "S", "29463-7"),
    (3013762, "Body weight Measured", "Measurement", "LOINC", "Clinical Observation", "S", "3141-9"),
    (903124, "Body Mass Index (program physical measurement; name approximate)", "Measurement", "PPI", "Clinical Observation", "S", "PM-BMI"),
    # --- units ---
    (0, "No matching concept", "Metadata", "None", "Undefined", None, "No matching concept"),
    (8840, "milligram per deciliter", "Unit", "UCUM", "Unit", "S", "mg/dL"),
    (8749, "micromole per liter", "Unit", "UCUM", "Unit", "S", "umol/L"),
    (8753, "millimole per liter", "Unit", "UCUM", "Unit", "S", "mmol/L"),
    (720870, "milliliter per minute per 1.73 square meter", "Unit", "UCUM", "Unit", "S", "mL/min/(173.10*-2.m2)"),
    (8795, "milliliter per minute", "Unit", "UCUM", "Unit", "S", "mL/min"),
    (8510, "unit", "Unit", "UCUM", "Unit", "S", "[U]"),
    (8923, "international unit per liter", "Unit", "UCUM", "Unit", "S", "[IU]/L"),
    (9017, "milligram per gram of creatinine", "Unit", "UCUM", "Unit", "S", "mg/g{creat}"),
    (8723, "milligram per gram", "Unit", "UCUM", "Unit", "S", "mg/g"),
    (8582, "centimeter", "Unit", "UCUM", "Unit", "S", "cm"),
    (9529, "kilogram", "Unit", "UCUM", "Unit", "S", "kg"),
    (9531, "kilogram per square meter", "Unit", "UCUM", "Unit", "S", "kg/m2"),
    (99000301, "milligram per millimole (MOCK id; real UCUM id to resolve)", "Unit", "MOCK", "Unit", "S", "mg/mmol"),
    # --- operators and measurement values ---
    (4172703, "=", "Meas Value Operator", "SNOMED", "Qualifier Value", "S", "276136004"),
    (4171754, "<=", "Meas Value Operator", "SNOMED", "Qualifier Value", "S", "276140008"),
    (4171755, ">=", "Meas Value Operator", "SNOMED", "Qualifier Value", "S", "276139006"),
    (4171756, "<", "Meas Value Operator", "SNOMED", "Qualifier Value", "S", "276137008"),
    (4172704, ">", "Meas Value Operator", "SNOMED", "Qualifier Value", "S", "276138003"),
    (9189, "Negative", "Meas Value", "SNOMED", "Qualifier Value", "S", "260385009"),
    (9191, "Positive", "Meas Value", "SNOMED", "Qualifier Value", "S", "10828004"),
    (4123508, "+", "Meas Value", "SNOMED", "Qualifier Value", "S", "260347006"),
    (4125547, "+++", "Meas Value", "SNOMED", "Qualifier Value", "S", "260349009"),
    (99000101, "Trace (MOCK id; real SNOMED id to resolve)", "Meas Value", "MOCK", "Qualifier Value", "S", "MOCK"),
    (99000102, "++ (MOCK id; real SNOMED id to resolve)", "Meas Value", "MOCK", "Qualifier Value", "S", "MOCK"),
    # --- visits ---
    (9201, "Inpatient Visit", "Visit", "Visit", "Visit", "S", "IP"),
    (262, "Emergency Room and Inpatient Visit", "Visit", "Visit", "Visit", "S", "ERIP"),
    (9202, "Outpatient Visit", "Visit", "Visit", "Visit", "S", "OP"),
    (9203, "Emergency Room Visit", "Visit", "Visit", "Visit", "S", "ER"),
    (32037, "Intensive Care", "Visit", "Visit", "Visit", "S", "OMOP4822460"),
    # --- diagnoses ---
    (201820, "Diabetes mellitus", "Condition", "SNOMED", "Clinical Finding", "S", "73211009"),
    (201826, "Type 2 diabetes mellitus", "Condition", "SNOMED", "Clinical Finding", "S", "44054006"),
    (201254, "Type 1 diabetes mellitus", "Condition", "SNOMED", "Clinical Finding", "S", "46635009"),
    (4193704, "Type 2 diabetes mellitus without complication", "Condition", "SNOMED", "Clinical Finding", "S", "313436004"),
    (99000401, "Gestational diabetes mellitus (MOCK id)", "Condition", "MOCK", "Clinical Finding", "S", "MOCK"),
    (320128, "Essential hypertension", "Condition", "SNOMED", "Clinical Finding", "S", "59621000"),
    (316866, "Hypertensive disorder", "Condition", "SNOMED", "Clinical Finding", "S", "38341003"),
    (99000402, "Benign essential hypertension (MOCK id)", "Condition", "MOCK", "Clinical Finding", "S", "MOCK"),
    (432867, "Hyperlipidemia", "Condition", "SNOMED", "Clinical Finding", "S", "55822004"),
    (316139, "Heart failure", "Condition", "SNOMED", "Clinical Finding", "S", "84114007"),
    (99000403, "Congestive heart failure (MOCK id)", "Condition", "MOCK", "Clinical Finding", "S", "MOCK"),
    (317576, "Coronary arteriosclerosis", "Condition", "SNOMED", "Clinical Finding", "S", "53741008"),
    (312327, "Acute myocardial infarction", "Condition", "SNOMED", "Clinical Finding", "S", "57054005"),
    (4212540, "Chronic liver disease", "Condition", "SNOMED", "Clinical Finding", "S", "328383001"),
    (4064161, "Cirrhosis of liver", "Condition", "SNOMED", "Clinical Finding", "S", "19943007"),
    (255573, "Chronic obstructive pulmonary disease", "Condition", "SNOMED", "Clinical Finding", "S", "13645005"),
    (443392, "Malignant neoplastic disease", "Condition", "SNOMED", "Clinical Finding", "S", "363346000"),
    (99000404, "Malignant neoplasm of colon (MOCK id)", "Condition", "MOCK", "Clinical Finding", "S", "MOCK"),
    (99000405, "Basal cell carcinoma of skin (MOCK id)", "Condition", "MOCK", "Clinical Finding", "S", "MOCK"),
    (433736, "Obesity", "Condition", "SNOMED", "Clinical Finding", "S", "414916001"),
    (99000406, "Morbid obesity (MOCK id)", "Condition", "MOCK", "Clinical Finding", "S", "MOCK"),
    (132797, "Sepsis", "Condition", "SNOMED", "Clinical Finding", "S", "91302008"),
    (196236, "Septic shock", "Condition", "SNOMED", "Clinical Finding", "S", "76571007"),
    (99000407, "Neonatal sepsis (MOCK id)", "Condition", "MOCK", "Clinical Finding", "S", "MOCK"),
    (197320, "Acute kidney injury", "Condition", "SNOMED", "Clinical Finding", "S", "14669001"),
    (444044, "Acute tubular necrosis", "Condition", "SNOMED", "Clinical Finding", "S", "35455006"),
    (46271022, "Chronic kidney disease", "Condition", "SNOMED", "Clinical Finding", "S", "709044004"),
    (443601, "Chronic kidney disease stage 2", "Condition", "SNOMED", "Clinical Finding", "S", "431856006"),
    (443597, "Chronic kidney disease stage 3", "Condition", "SNOMED", "Clinical Finding", "S", "433144002"),
    (443612, "Chronic kidney disease stage 4", "Condition", "SNOMED", "Clinical Finding", "S", "431857002"),
    (443611, "Chronic kidney disease stage 5", "Condition", "SNOMED", "Clinical Finding", "S", "433146000"),
    (193782, "End-stage renal disease", "Condition", "SNOMED", "Clinical Finding", "S", "46177005"),
    (42539502, "Transplanted kidney present", "Condition", "SNOMED", "Clinical Finding", "S", "737295003"),
    (99000408, "Pregnancy (MOCK id)", "Condition", "MOCK", "Clinical Finding", "S", "MOCK"),
    # --- observations ---
    (4019967, "Dependence on renal dialysis", "Observation", "SNOMED", "Clinical Finding", "S", "105502003"),
    (45878463, "Female", "Observation", "LOINC", "Answer", "S", "LA3-6"),
    (45880669, "Male", "Observation", "LOINC", "Answer", "S", "LA2-8"),
    (903096, "PMI: Skip", "Observation", "PPI", "Answer", "S", "PMI_Skip"),
    (903079, "PMI: Prefer Not To Answer", "Observation", "PPI", "Answer", "S", "PMI_PreferNotToAnswer"),
    # --- procedures ---
    (4032243, "Dialysis procedure", "Procedure", "SNOMED", "Procedure", "S", "108241001"),
    (37018292, "Continuous renal replacement therapy", "Procedure", "SNOMED", "Procedure", "S", "714749008"),
    (619390, "Prolonged intermittent renal replacement therapy", "Procedure", "SNOMED", "Procedure", "S", "895382009"),
    (4050863, "Continuous hemodialysis", "Procedure", "SNOMED", "Procedure", "S", "233578004"),
    (4052537, "Continuous hemofiltration", "Procedure", "SNOMED", "Procedure", "S", "233583007"),
    (46270933, "Emergency hemodialysis", "Procedure", "SNOMED", "Procedure", "S", "708932005"),
    (4238836, "Initial hemodialysis", "Procedure", "SNOMED", "Procedure", "S", "57274006"),
    (4051326, "Intermittent hemodialysis", "Procedure", "SNOMED", "Procedure", "S", "233575001"),
    (1314323, "Dialysis procedure at a medicare certified esrd facility for acute kidney injury without esrd", "Procedure", "HCPCS", "HCPCS", "S", "G0491"),
    (4322471, "Transplant of kidney", "Procedure", "SNOMED", "Procedure", "S", "70536003"),
    (2514441, "Critical care, evaluation and management of the critically ill or critically injured patient; first 30-74 minutes", "Procedure", "CPT4", "CPT4", "S", "99291"),
    (2514442, "Critical care, evaluation and management of the critically ill or critically injured patient; each additional 30 minutes", "Procedure", "CPT4", "CPT4", "S", "99292"),
    (2788036, "Respiratory Ventilation, Less than 24 Consecutive Hours", "Procedure", "ICD10PCS", "ICD10PCS", "S", "5A1935Z"),
    (2788037, "Respiratory Ventilation, 24-96 Consecutive Hours", "Procedure", "ICD10PCS", "ICD10PCS", "S", "5A1945Z"),
    (2788038, "Respiratory Ventilation, Greater than 96 Consecutive Hours", "Procedure", "ICD10PCS", "ICD10PCS", "S", "5A1955Z"),
    (2008008, "Continuous invasive mechanical ventilation for less than 96 consecutive hours", "Procedure", "ICD9Proc", "4-dig billing code", "S", "96.71"),
    (2008009, "Continuous invasive mechanical ventilation for 96 consecutive hours or more", "Procedure", "ICD9Proc", "4-dig billing code", "S", "96.72"),
    (99000501, "CT of abdomen with contrast (MOCK id)", "Procedure", "MOCK", "Procedure", "S", "MOCK"),
    # --- drugs ---
    (1321341, "norepinephrine", "Drug", "RxNorm", "Ingredient", "S", "7512"),
    (1343916, "epinephrine", "Drug", "RxNorm", "Ingredient", "S", "3992"),
    (1507835, "vasopressin (USP)", "Drug", "RxNorm", "Ingredient", "S", "11149"),
    (1135766, "phenylephrine", "Drug", "RxNorm", "Ingredient", "S", "8163"),
    (1337860, "dopamine", "Drug", "RxNorm", "Ingredient", "S", "3628"),
    (99000601, "norepinephrine 1 MG/ML Injectable Solution (MOCK id)", "Drug", "MOCK", "Clinical Drug", "S", "MOCK"),
    (99000602, "vasopressin 20 UNT/ML Injectable Solution (MOCK id)", "Drug", "MOCK", "Clinical Drug", "S", "MOCK"),
    (99000603, "phenylephrine 2.5 % Ophthalmic Solution (MOCK id)", "Drug", "MOCK", "Clinical Drug", "S", "MOCK"),
    (99000604, "phenylephrine 0.1 MG/ML Injectable Solution (MOCK id)", "Drug", "MOCK", "Clinical Drug", "S", "MOCK"),
    (99000605, "iohexol (MOCK ingredient id; ATC V08AB02)", "Drug", "MOCK", "Ingredient", "S", "MOCK"),
    (99000606, "iohexol 300 MG/ML Injectable Solution (MOCK id)", "Drug", "MOCK", "Clinical Drug", "S", "MOCK"),
    # --- routes ---
    (4171047, "Intravenous", "Route", "SNOMED", "Qualifier Value", "S", "47625008"),
    (4132161, "Oral", "Route", "SNOMED", "Qualifier Value", "S", "26643006"),
    (99000201, "Ophthalmic (MOCK id; real SNOMED route id to resolve)", "Route", "MOCK", "Qualifier Value", "S", "MOCK"),
    # --- type concepts ---
    (32817, "EHR", "Type Concept", "Type Concept", "Type Concept", "S", "OMOP4976890"),
    (32865, "Patient self-report", "Type Concept", "Type Concept", "Type Concept", "S", "OMOP4976926"),
    (32879, "Registry", "Type Concept", "Type Concept", "Type Concept", "S", "OMOP4976936"),
]

# (ancestor, descendant, min_levels, max_levels); each concept's self-row (0,0) is added at build time (it exists in real OMOP)
ANCESTORS = [
    (201820, 201826, 1, 1), (201820, 201254, 1, 1), (201820, 4193704, 2, 2), (201826, 4193704, 1, 1), (201820, 99000401, 1, 1),
    (316866, 320128, 1, 1), (316866, 99000402, 2, 2), (320128, 99000402, 1, 1),
    (316139, 99000403, 1, 1),
    (4212540, 4064161, 1, 1),
    (443392, 99000404, 1, 1), (443392, 99000405, 1, 1),
    (433736, 99000406, 1, 1),
    (132797, 196236, 1, 1), (132797, 99000407, 1, 1),
    (197320, 444044, 1, 1),
    (46271022, 443601, 1, 1), (46271022, 443597, 1, 1), (46271022, 443612, 1, 1), (46271022, 443611, 1, 1),
    (46271022, 193782, 2, 2), (443611, 193782, 1, 1),
    (4032243, 37018292, 1, 1), (4032243, 619390, 1, 1), (4032243, 4050863, 2, 2), (4032243, 4052537, 2, 2),
    (4032243, 46270933, 2, 2), (4032243, 4238836, 2, 2), (4032243, 4051326, 2, 2),
    (37018292, 4050863, 1, 1), (37018292, 4052537, 1, 1),
    (1321341, 99000601, 1, 1), (1507835, 99000602, 1, 1), (1135766, 99000603, 1, 1), (1135766, 99000604, 1, 1),
    (99000605, 99000606, 1, 1),
]

# ============================================================================================
# 2. SCHEMA: column names from the CT v9 data dictionary (CDM 5.3.1 + All of Us additions)
# ============================================================================================
DDL = {
    "person": """person_id BIGINT, gender_concept_id BIGINT, year_of_birth BIGINT, month_of_birth BIGINT, day_of_birth BIGINT,
        birth_datetime TIMESTAMP, race_concept_id BIGINT, ethnicity_concept_id BIGINT, location_id BIGINT, provider_id BIGINT,
        care_site_id BIGINT, person_source_value VARCHAR, gender_source_value VARCHAR, gender_source_concept_id BIGINT,
        race_source_value VARCHAR, race_source_concept_id BIGINT, ethnicity_source_value VARCHAR, ethnicity_source_concept_id BIGINT,
        state_of_residence_concept_id BIGINT, state_of_residence_source_value VARCHAR,
        sex_at_birth_concept_id BIGINT, sex_at_birth_source_concept_id BIGINT, sex_at_birth_source_value VARCHAR,
        self_reported_category_concept_id BIGINT, self_reported_category_source_value VARCHAR""",
    "visit_occurrence": """visit_occurrence_id BIGINT, person_id BIGINT, visit_concept_id BIGINT, visit_start_date DATE,
        visit_start_datetime TIMESTAMP, visit_end_date DATE, visit_end_datetime TIMESTAMP, visit_type_concept_id BIGINT,
        provider_id BIGINT, care_site_id BIGINT, visit_source_value VARCHAR, visit_source_concept_id BIGINT,
        admitting_source_concept_id BIGINT, admitting_source_value VARCHAR, discharge_to_concept_id BIGINT,
        discharge_to_source_value VARCHAR, preceding_visit_occurrence_id BIGINT""",
    "visit_detail": """visit_detail_id BIGINT, person_id BIGINT, visit_detail_concept_id BIGINT, visit_detail_start_date DATE,
        visit_detail_start_datetime TIMESTAMP, visit_detail_end_date DATE, visit_detail_end_datetime TIMESTAMP,
        visit_detail_type_concept_id BIGINT, provider_id BIGINT, care_site_id BIGINT, visit_detail_source_value VARCHAR,
        visit_detail_source_concept_id BIGINT, admitting_source_value VARCHAR, admitting_source_concept_id BIGINT,
        discharge_to_source_value VARCHAR, discharge_to_concept_id BIGINT, preceding_visit_detail_id BIGINT,
        visit_detail_parent_id BIGINT, visit_occurrence_id BIGINT""",
    "measurement": """measurement_id BIGINT, person_id BIGINT, measurement_concept_id BIGINT, measurement_date DATE,
        measurement_datetime TIMESTAMP, measurement_time VARCHAR, measurement_type_concept_id BIGINT, operator_concept_id BIGINT,
        value_as_number DOUBLE, value_as_concept_id BIGINT, unit_concept_id BIGINT, range_low DOUBLE, range_high DOUBLE,
        provider_id BIGINT, visit_occurrence_id BIGINT, visit_detail_id BIGINT, measurement_source_value VARCHAR,
        measurement_source_concept_id BIGINT, unit_source_value VARCHAR, value_source_value VARCHAR""",
    "condition_occurrence": """condition_occurrence_id BIGINT, person_id BIGINT, condition_concept_id BIGINT, condition_start_date DATE,
        condition_start_datetime TIMESTAMP, condition_end_date DATE, condition_end_datetime TIMESTAMP, condition_type_concept_id BIGINT,
        condition_status_concept_id BIGINT, stop_reason VARCHAR, provider_id BIGINT, visit_occurrence_id BIGINT, visit_detail_id BIGINT,
        condition_source_value VARCHAR, condition_source_concept_id BIGINT, condition_status_source_value VARCHAR""",
    "procedure_occurrence": """procedure_occurrence_id BIGINT, person_id BIGINT, procedure_concept_id BIGINT, procedure_date DATE,
        procedure_datetime TIMESTAMP, procedure_type_concept_id BIGINT, modifier_concept_id BIGINT, quantity BIGINT, provider_id BIGINT,
        visit_occurrence_id BIGINT, visit_detail_id BIGINT, procedure_source_value VARCHAR, procedure_source_concept_id BIGINT,
        modifier_source_value VARCHAR""",
    "drug_exposure": """drug_exposure_id BIGINT, person_id BIGINT, drug_concept_id BIGINT, drug_exposure_start_date DATE,
        drug_exposure_start_datetime TIMESTAMP, drug_exposure_end_date DATE, drug_exposure_end_datetime TIMESTAMP, verbatim_end_date DATE,
        drug_type_concept_id BIGINT, stop_reason VARCHAR, refills BIGINT, quantity DOUBLE, days_supply BIGINT, sig VARCHAR,
        route_concept_id BIGINT, lot_number VARCHAR, provider_id BIGINT, visit_occurrence_id BIGINT, visit_detail_id BIGINT,
        drug_source_value VARCHAR, drug_source_concept_id BIGINT, route_source_value VARCHAR, dose_unit_source_value VARCHAR""",
    "observation": """observation_id BIGINT, person_id BIGINT, observation_concept_id BIGINT, observation_date DATE,
        observation_datetime TIMESTAMP, observation_type_concept_id BIGINT, value_as_number DOUBLE, value_as_string VARCHAR,
        value_as_concept_id BIGINT, qualifier_concept_id BIGINT, unit_concept_id BIGINT, provider_id BIGINT, visit_occurrence_id BIGINT,
        visit_detail_id BIGINT, observation_source_value VARCHAR, observation_source_concept_id BIGINT, unit_source_value VARCHAR,
        qualifier_source_value VARCHAR, value_source_concept_id BIGINT, value_source_value VARCHAR, questionnaire_response_id BIGINT""",
    "aou_death": """person_id BIGINT, death_date DATE, death_datetime TIMESTAMP, death_type_concept_id BIGINT, cause_concept_id BIGINT,
        cause_source_value VARCHAR, cause_source_concept_id BIGINT, src_id VARCHAR, primary_death_record BOOLEAN""",
    "cb_search_person": """person_id BIGINT, gender VARCHAR, sex_at_birth VARCHAR, race VARCHAR, ethnicity VARCHAR, dob DATE,
        age_at_consent BIGINT, age_at_cdr BIGINT, has_ehr_data BIGINT, has_physical_measurement_data BIGINT, has_ppi_survey_data BIGINT,
        has_fitbit BIGINT, has_whole_genome_variant BIGINT, has_array_data BIGINT, has_lr_whole_genome_variant BIGINT,
        has_structural_variant_data BIGINT""",
    "concept": """concept_id BIGINT, concept_name VARCHAR, domain_id VARCHAR, vocabulary_id VARCHAR, concept_class_id VARCHAR,
        standard_concept VARCHAR, concept_code VARCHAR, valid_start_date DATE, valid_end_date DATE, invalid_reason VARCHAR""",
    "concept_ancestor": """ancestor_concept_id BIGINT, descendant_concept_id BIGINT, min_levels_of_separation BIGINT,
        max_levels_of_separation BIGINT""",
    "concept_relationship": """concept_id_1 BIGINT, concept_id_2 BIGINT, relationship_id VARCHAR, valid_start_date DATE,
        valid_end_date DATE, invalid_reason VARCHAR""",
}

EHR, SELF_REPORT = 32817, 32865
D = dt.timedelta


# ============================================================================================
# 3. DATA GENERATION
# ============================================================================================
class _World:
    """Accumulates tables as lists of rows; generates IDs; records each person's PLANTED ground truth."""

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.rows = {t: [] for t in DDL}
        self.truth = []
        self._used_big = set()
        self._small = 1

    # ---- helpers ----
    def big_id(self):
        while True:
            v = int(self.rng.integers(BIG_ID_LOW, BIG_ID_HIGH))
            if v not in self._used_big:
                self._used_big.add(v)
                return v

    def small_id(self):
        self._small += 1
        return self._small

    def p(self, prob):
        return bool(self.rng.random() < prob)

    def u(self, a, b):
        return float(self.rng.uniform(a, b))

    def choice(self, seq, p=None):
        i = int(self.rng.choice(len(seq), p=p))
        return seq[i]

    def dtime(self, day, midnight=False):
        if midnight:
            return dt.datetime(day.year, day.month, day.day)
        return dt.datetime(day.year, day.month, day.day, int(self.rng.integers(6, 22)), int(self.rng.integers(0, 60)), 0)

    # ---- row appenders (column names are those of the real schema) ----
    def visit(self, pid, concept, start, end, start_time_known=True, end_time_known=True, vid=None):
        vid = vid or self.big_id()
        sdt = self.dtime(start, midnight=not start_time_known)
        edt = None if end is None else self.dtime(end, midnight=not end_time_known)
        if edt is not None and end == start and edt < sdt:
            edt = sdt + D(hours=2)
        self.rows["visit_occurrence"].append(dict(
            visit_occurrence_id=vid, person_id=pid, visit_concept_id=concept, visit_start_date=start,
            visit_start_datetime=sdt, visit_end_date=end, visit_end_datetime=edt, visit_type_concept_id=EHR,
            visit_source_value={9201: "IP", 262: "ERIP", 9202: "OP", 9203: "ER"}[concept], visit_source_concept_id=0))
        return vid, sdt, edt

    def meas(self, pid, concept, when, value, unit, operator=0, vid=None, value_concept=None, source_concept=0, time_known=True):
        day = when.date() if isinstance(when, dt.datetime) else when
        when_dt = when if isinstance(when, dt.datetime) else self.dtime(when, midnight=not time_known)
        self.rows["measurement"].append(dict(
            measurement_id=self.big_id(), person_id=pid, measurement_concept_id=concept, measurement_date=day,
            measurement_datetime=when_dt, measurement_time=None, measurement_type_concept_id=EHR, operator_concept_id=operator,
            value_as_number=value, value_as_concept_id=value_concept if value_concept is not None else 0,
            unit_concept_id=unit, visit_occurrence_id=vid, measurement_source_concept_id=source_concept,
            measurement_source_value=str(concept), unit_source_value=None, value_source_value=None if value is None else str(value)))
        return when_dt

    def cond(self, pid, concept, day, vid=None):
        self.rows["condition_occurrence"].append(dict(
            condition_occurrence_id=self.big_id(), person_id=pid, condition_concept_id=concept, condition_start_date=day,
            condition_start_datetime=self.dtime(day, midnight=True), condition_end_date=None, condition_type_concept_id=EHR,
            condition_status_concept_id=0, visit_occurrence_id=vid, condition_source_concept_id=0, condition_source_value=str(concept)))

    def proc(self, pid, concept, day, vid=None):
        self.rows["procedure_occurrence"].append(dict(
            procedure_occurrence_id=self.big_id(), person_id=pid, procedure_concept_id=concept, procedure_date=day,
            procedure_datetime=self.dtime(day), procedure_type_concept_id=EHR, modifier_concept_id=0, quantity=1,
            visit_occurrence_id=vid, procedure_source_concept_id=0, procedure_source_value=str(concept)))

    def drug(self, pid, concept, day, route, vid=None, days=1):
        self.rows["drug_exposure"].append(dict(
            drug_exposure_id=self.big_id(), person_id=pid, drug_concept_id=concept, drug_exposure_start_date=day,
            drug_exposure_start_datetime=self.dtime(day), drug_exposure_end_date=day + D(days=days - 1),
            drug_exposure_end_datetime=None, drug_type_concept_id=EHR, route_concept_id=route, visit_occurrence_id=vid,
            drug_source_concept_id=0, drug_source_value=str(concept), route_source_value=None, days_supply=days))

    def obs(self, pid, concept, day, vid=None, value_concept=0):
        self.rows["observation"].append(dict(
            observation_id=self.big_id(), person_id=pid, observation_concept_id=concept, observation_date=day,
            observation_datetime=self.dtime(day, midnight=True), observation_type_concept_id=EHR, value_as_concept_id=value_concept,
            unit_concept_id=0, qualifier_concept_id=0, visit_occurrence_id=vid, observation_source_concept_id=0,
            observation_source_value=str(concept), value_source_concept_id=0))

    def death(self, pid, day, primary=True, src="hospital"):
        self.rows["aou_death"].append(dict(
            person_id=pid, death_date=day, death_datetime=None if day is None else self.dtime(day, midnight=True),
            death_type_concept_id=EHR, cause_concept_id=0, cause_source_value=None, cause_source_concept_id=0,
            src_id=src, primary_death_record=primary))


def _vocab(w):
    for c in CONCEPTS:
        w.rows["concept"].append(dict(concept_id=c[0], concept_name=c[1], domain_id=c[2], vocabulary_id=c[3], concept_class_id=c[4],
                                      standard_concept=c[5], concept_code=c[6], valid_start_date=dt.date(1970, 1, 1),
                                      valid_end_date=dt.date(2099, 12, 31), invalid_reason=None))
    ids = {c[0] for c in CONCEPTS}
    for a, d_, lo, hi in ANCESTORS:
        assert a in ids and d_ in ids, (a, d_)
        w.rows["concept_ancestor"].append(dict(ancestor_concept_id=a, descendant_concept_id=d_, min_levels_of_separation=lo, max_levels_of_separation=hi))
    for cid in ids:                      # self-row: exists in real OMOP concept_ancestor
        w.rows["concept_ancestor"].append(dict(ancestor_concept_id=cid, descendant_concept_id=cid, min_levels_of_separation=0, max_levels_of_separation=0))
    for a, d_, lo, hi in ANCESTORS:
        if lo == 1:
            w.rows["concept_relationship"].append(dict(concept_id_1=d_, concept_id_2=a, relationship_id="Is a", valid_start_date=dt.date(1970, 1, 1), valid_end_date=dt.date(2099, 12, 31), invalid_reason=None))
            w.rows["concept_relationship"].append(dict(concept_id_1=a, concept_id_2=d_, relationship_id="Subsumes", valid_start_date=dt.date(1970, 1, 1), valid_end_date=dt.date(2099, 12, 31), invalid_reason=None))


def _ckd_epi_2021(scr, age, female):
    k, a = (0.7, -0.241) if female else (0.9, -0.302)
    r = scr / k
    e = 142 * min(r, 1.0) ** a * max(r, 1.0) ** -1.200 * 0.9938 ** age
    return e * 1.012 if female else e


def _descendants_of(root):
    out = {root}
    for a, d_, _, _ in ANCESTORS:
        if a == root:
            out.add(d_)
    return out


def _person(w, pid, enroll_min=dt.date(2018, 1, 1)):
    """Adds all records of one person and returns the planted ground truth."""
    T = dict(person_id=pid, scenario=[])
    tag = T["scenario"].append

    # ---- demographics ----
    sex = w.choice(["F", "M", "other"], p=[0.495, 0.495, 0.01])
    yob = int(w.rng.integers(1935, 2001)) if not w.p(0.02) else int(w.rng.integers(2007, 2010))
    sex_concept = {"F": 45878463, "M": 45880669, "other": w.choice([903096, 903079])}[sex]
    birth = dt.datetime(yob, 6, 15)   # All of Us: birth_datetime is masked, YYYY-06-15 for everyone
    w.rows["person"].append(dict(person_id=pid, gender_concept_id=sex_concept, year_of_birth=yob, month_of_birth=None, day_of_birth=None,
                                 birth_datetime=birth, race_concept_id=0, ethnicity_concept_id=0, sex_at_birth_concept_id=sex_concept,
                                 sex_at_birth_source_concept_id=sex_concept, sex_at_birth_source_value={"F": "SexAtBirth_Female", "M": "SexAtBirth_Male"}.get(sex, "PMI_Skip"),
                                 self_reported_category_concept_id=0))
    T.update(sex=sex, year_of_birth=yob)
    has_ehr = w.p(0.97)
    enroll = enroll_min + D(days=int(w.rng.integers(0, 2200)))
    w.rows["cb_search_person"].append(dict(person_id=pid, gender=sex, sex_at_birth=sex, race=None, ethnicity=None, dob=birth.date(),
                                           age_at_consent=enroll.year - yob, age_at_cdr=DATA_CUTOFF.year - yob, has_ehr_data=int(has_ehr),
                                           has_physical_measurement_data=1, has_ppi_survey_data=1, has_fitbit=0, has_whole_genome_variant=0,
                                           has_array_data=0, has_lr_whole_genome_variant=0, has_structural_variant_data=0))
    # program BMI (enrollment visit): measurement_source_concept_id = 903124, concept 3038553
    bmi_prog = round(w.u(19, 42), 1)
    w.meas(pid, 3038553, enroll, bmi_prog, 9531, source_concept=903124)
    T.update(has_ehr_data=int(has_ehr), enrollment_date=enroll, program_bmi=bmi_prog)
    if not has_ehr:
        tag("no_ehr")
        return T

    # ---- is there an index admission ----
    female = sex == "F"
    base = max(0.45, float(w.rng.normal(0.80 if female else 1.00, 0.12)))
    ckd_baseline = w.p(0.08)
    if ckd_baseline:
        base = w.u(1.6, 2.6); tag("baseline_egfr_lt60")
    T["true_baseline_scr"] = round(base, 3)
    has_admission = w.p(0.85)
    if has_admission:
        admit = dt.date(2018, 6, 1) + D(days=int(w.rng.integers(0, 2100)))
        if w.p(0.05):
            admit = DATA_CUTOFF - D(days=int(w.rng.integers(20, 170))); tag("discharge_too_close_to_cutoff")
    else:
        admit = dt.date(2021, 1, 1) + D(days=int(w.rng.integers(0, 900)))   # to anchor the windows to this virtual date
        tag("no_inpatient_stay")
    age_at_admit = (admit - birth.date()).days / 365.25
    T.update(index_admit_date=admit if has_admission else None, age_at_admit=round(age_at_admit, 2))
    if age_at_admit < 18:
        tag("minor_at_admission")
    hist_days = int(w.rng.integers(30, 2200)) if w.p(0.9) else int(w.rng.integers(30, 364))
    ehr_start = admit - D(days=hist_days)
    T["history_days_before_admit"] = hist_days
    if hist_days < 365:
        tag("history_lt_365d")

    stays = []   # (start, end) inpatient stay intervals: for the 'outpatient' rule

    # ---- prior (within-window) inpatient stay trap ----
    if w.p(0.10):
        ps = admit - D(days=int(w.rng.integers(100, 300))); pe = ps + D(days=3)
        pvid, _, _ = w.visit(pid, 9201, ps, pe, start_time_known=w.p(0.5))
        stays.append((ps, pe))
        for k in range(2):
            w.meas(pid, 3016723, ps + D(days=k), round(base * w.u(1.3, 1.8), 2), 8840, vid=pvid)   # in-stay: must not enter the baseline
        tag("prior_stay_in_window")

    # ---- baseline creatinine (outpatient, -365..-8) ----
    n_valid = w.choice([0, 1, 2, 3, 4, 5, 6], p=[0.08, 0.15, 0.20, 0.22, 0.17, 0.10, 0.08])
    valid_vals, valid_dates = [], []
    rows_mass_only = True

    def outpatient_day(lo, hi):
        for _ in range(50):
            d_ = admit - D(days=int(w.rng.integers(lo, hi + 1)))
            if ehr_start <= d_ and not any(s <= d_ <= e for s, e in stays):
                return d_
        return None

    for _ in range(n_valid):
        d_ = outpatient_day(8, 365)
        if d_ is None:
            continue
        val = round(base * w.u(0.9, 1.1), 2)
        vid = w.visit(pid, 9202, d_, d_)[0] if w.p(0.7) else None
        r = w.rng.random()
        if r < 0.10:
            w.meas(pid, 3016723, d_, round(val * 88.4, 1), 8749, vid=vid); tag("baseline_umol_row")
        elif r < 0.20:
            w.meas(pid, 3016723, d_, val, None, vid=vid); tag("baseline_null_unit_row")
        elif r < 0.25:
            w.meas(pid, 3016723, d_, val, 0, vid=vid); tag("baseline_zero_unit_row")
        elif r < 0.27:
            w.meas(pid, 3016723, d_, val, 8510, vid=vid)
        elif r < 0.35:
            w.meas(pid, 3020564, d_, round(val * 88.4, 1), 8749, vid=vid); tag("baseline_molar_concept_row"); rows_mass_only = False
        elif r < 0.38:
            w.meas(pid, 3020564, d_, round(val * 88.4, 1), None, vid=vid); tag("baseline_molar_null_unit_row"); rows_mass_only = False
        else:
            w.meas(pid, 3016723, d_, val, 8840, vid=vid, operator=w.choice([0, 4172703]))
            if w.p(0.10):
                w.meas(pid, 3016723, d_, val, 8840, vid=vid); tag("baseline_exact_duplicate")   # same day, same value: a second row (datetime may differ)
        valid_vals.append(val); valid_dates.append(d_)
    # trap rows: SQL or pandas must drop them
    if w.p(0.08):
        d_ = outpatient_day(8, 365)
        if d_: w.meas(pid, 3016723, d_, round(w.u(60, 110), 0), 720870); tag("egfr_in_creatinine_box")
    if w.p(0.06):
        d_ = outpatient_day(8, 365)
        if d_: w.meas(pid, 3016723, d_, 0.5, 8840, operator=4171756); tag("baseline_censored_lt")
    if w.p(0.03):
        d_ = outpatient_day(8, 365)
        if d_: w.meas(pid, 3016723, d_, w.choice([0.05, 0.0, 35.0, -1.0]), 8840); tag("baseline_out_of_range")
    if w.p(0.15):
        d_ = outpatient_day(1, 7)
        if d_: w.meas(pid, 3016723, d_, round(base * w.u(1.2, 2.0), 2), 8840); tag("creatinine_day_minus7_to_minus1")
    if w.p(0.20) and hist_days > 400:
        d_ = outpatient_day(366, min(hist_days, 900))
        if d_: w.meas(pid, 3016723, d_, round(base * w.u(0.9, 1.1), 2), 8840); tag("creatinine_before_day_minus365")
    if w.p(0.05):
        d_ = outpatient_day(8, 365)
        if d_: w.meas(pid, 3033837, d_, 9.9, 8840); tag("wrong_concept_creat_bun_ratio")
    T["baseline_scr_values"] = valid_vals
    T["baseline_scr_median"] = float(np.median(valid_vals)) if valid_vals else None
    T["baseline_scr_n"] = len(valid_vals)
    if valid_vals and sex != "other":
        egfrs = [_ckd_epi_2021(v, (d_ - birth.date()).days / 365.25, female) for v, d_ in zip(valid_vals, valid_dates)]
        T["baseline_egfr_median"] = round(float(np.median(egfrs)), 2)
    else:
        T["baseline_egfr_median"] = None
    T["baseline_rows_mass_concept_only"] = rows_mass_only

    # ---- UACR / dipstick (window) ----
    T.update(uacr_mg_g=None, uacr_date=None, dipstick_value=None, proteinuria_flag=None)
    if w.p(0.40):
        d_ = outpatient_day(8, 365)
        if d_:
            uacr = round(w.u(3, 25), 1) if w.p(0.7) else round(w.u(31, 600), 1)
            r = w.rng.random()
            if r < 0.5:   w.meas(pid, 3034485, d_, uacr, 9017)
            elif r < 0.7: w.meas(pid, 3001802, d_, uacr, 8723)
            elif r < 0.85: w.meas(pid, 3001802, d_, round(uacr / 8.84, 2), 99000301); tag("uacr_mg_mmol_row")
            else:         w.meas(pid, 3034485, d_, uacr, None); tag("uacr_null_unit_row")
            T.update(uacr_mg_g=uacr, uacr_date=d_)
    if w.p(0.18):
        d_ = outpatient_day(8, 365)
        if d_:
            vc = w.choice([9189, 99000101, 4123508, 99000102, 4125547], p=[0.55, 0.15, 0.15, 0.1, 0.05])
            w.meas(pid, 3014051, d_, None, 0, value_concept=vc)
            T["dipstick_value"] = {9189: "neg", 99000101: "trace", 4123508: "1+", 99000102: "2+", 4125547: "3+"}[vc]
    if w.p(0.08):
        d_ = outpatient_day(366, 700) if hist_days > 400 else None
        if d_: w.meas(pid, 3034485, d_, 150.0, 9017); tag("uacr_outside_window")
    pos = []
    if T["uacr_mg_g"] is not None: pos.append(T["uacr_mg_g"] >= 30)
    if T["dipstick_value"] is not None: pos.append(T["dipstick_value"] in ("1+", "2+", "3+"))
    T["proteinuria_flag"] = (1 if any(pos) else 0) if pos else None

    # ---- EHR height / weight (for BMI) ----
    T.update(ehr_bmi=None)
    if w.p(0.6):
        d_ = outpatient_day(10, 400)
        if d_:
            h = round(w.u(150, 195), 1); wt = round(w.u(45, 140), 1)
            w.meas(pid, 3036277, d_, h, 8582); w.meas(pid, 3025315, d_ + D(days=int(w.rng.integers(0, 20))), wt, 9529)
            T["ehr_bmi"] = round(wt / (h / 100) ** 2, 1)
    elif w.p(0.3):
        d_ = outpatient_day(10, 400)
        if d_: w.meas(pid, 3025315, d_, round(w.u(45, 140), 1), 9529); tag("weight_without_height")

    # ---- pre-admission diagnoses ----
    def prior_day():
        return admit - D(days=int(w.rng.integers(2, max(3, hist_days))))
    flags = dict(diabetes=0, hypertension=0, heart_failure=0, coronary_artery_disease=0, hyperlipidemia=0, chronic_liver_disease=0,
                 COPD=0, malignancy=0, obesity_dx=0, prior_AKI_flag=0, prior_CKD_dx=0, dialysis_dependence_prior=0, transplant_prior=0)
    menu = [("diabetes", 0.25, [201826, 4193704, 201254, 201820]), ("hypertension", 0.45, [320128, 99000402, 316866]),
            ("heart_failure", 0.12, [316139, 99000403]), ("coronary_artery_disease", 0.15, [317576, 312327]),
            ("hyperlipidemia", 0.30, [432867]), ("chronic_liver_disease", 0.05, [4064161, 4212540]), ("COPD", 0.10, [255573]),
            ("malignancy", 0.10, [99000404, 443392]), ("obesity_dx", 0.10, [433736, 99000406]), ("prior_AKI_flag", 0.08, [197320, 444044]),
            ("prior_CKD_dx", 0.07, [46271022, 443597, 443601, 443612]), ("transplant_prior", 0.01, [42539502])]
    for name, prob, concepts in menu:
        if w.p(prob):
            flags[name] = 1
            for _ in range(int(w.rng.integers(1, 4))):
                w.cond(pid, w.choice(concepts), prior_day())
    if flags["hypertension"] and w.p(0.3):
        w.cond(pid, 316866, prior_day())   # broad root: an ancestor, not a descendant; does not count as a descendant of 320128
    if w.p(0.05):
        w.cond(pid, 99000405, prior_day()); tag("skin_bcc_only_or_extra")   # non-melanoma skin: excluded from the malignancy definition (decision)
    if w.p(0.02):
        w.cond(pid, 99000401, prior_day()); tag("gestational_dm_record")       # gestational: does not count as chronic DM (decision)
    if w.p(0.015):
        w.obs(pid, 4019967, prior_day()); flags["dialysis_dependence_prior"] = 1; tag("dialysis_dependence_observation_prior")
    if w.p(0.05):
        c = w.choice([201826, 320128, 316139, 197320]); w.cond(pid, c, admit)
        tag("dx_on_admission_day_%d" % c)   # admission day: does not count as 'prior'
    if w.p(0.10) and has_admission:
        c = w.choice([201826, 320128, 255573]); w.cond(pid, c, admit + D(days=int(w.rng.integers(3, 400))))
        tag("dx_after_admission_%d" % c)    # after: does not count
    T.update(flags)
    T["obesity"] = int(flags["obesity_dx"] == 1 or (T["ehr_bmi"] is not None and T["ehr_bmi"] >= 30))
    T["prior_ckd_excluded"] = int(flags["prior_CKD_dx"] or flags["dialysis_dependence_prior"] or flags["transplant_prior"])
    if not has_admission:
        return T

    # ---- index admission ----
    vconcept = w.choice([9201, 262], p=[0.8, 0.2])
    los = int(w.rng.integers(2, 15))
    start_known = w.p(0.5)
    disch = admit + D(days=los)
    end_known = w.p(0.6)
    er_attached = w.p(0.25)
    transfer = w.choice(["none", "contiguous", "gap1"], p=[0.85, 0.10, 0.05])
    constituent = []
    if er_attached:
        er_day = admit - D(days=int(w.rng.integers(0, 2)))
        evid = w.visit(pid, 9203, er_day, admit, start_time_known=start_known)[0]
        constituent.append(("ER", evid)); tag("er_visit_attached")
    if transfer == "none":
        vid, sdt, edt = w.visit(pid, vconcept, admit, disch, start_known, end_known)
        constituent.append(("IP", vid)); stays.append((admit, disch)); main_vid = vid
    else:
        split = admit + D(days=max(1, los // 2))
        vid1, sdt, _ = w.visit(pid, vconcept, admit, split, start_known, True)
        s2 = split if transfer == "contiguous" else split + D(days=1)
        vid2, _, edt = w.visit(pid, 9201, s2, disch, True, end_known)
        constituent += [("IP", vid1), ("IP", vid2)]; stays += [(admit, split), (s2, disch)]; main_vid = vid1
        tag("transfer_" + transfer)
    T.update(index_visit_concept=vconcept, index_admit_datetime=sdt, index_admit_time_known=start_known, index_discharge_date=disch,
             index_discharge_time_known=end_known, length_of_stay_days=los, constituent_visit_ids=[str(v) for _, v in constituent],
             episode_merge=transfer if transfer != "none" else ("er_attached" if er_attached else "single_visit"))
    if w.p(0.02):
        w.rows["visit_occurrence"][-1]["visit_end_date"] = None; w.rows["visit_occurrence"][-1]["visit_end_datetime"] = None; tag("index_visit_end_null")
    elif w.p(0.01):
        w.rows["visit_occurrence"][-1]["visit_end_date"] = admit - D(days=2); tag("index_visit_end_before_start")

    # ---- in-stay creatinine series ----
    aki = w.choice(["none", "r03", "r15", "stage2", "stage3", "community", "decreasing"], p=[0.35, 0.20, 0.15, 0.12, 0.08, 0.05, 0.05])
    r = round(base * w.u(0.95, 1.05), 2)
    if aki == "r03":   r = round(w.u(0.95, 1.2), 2)
    if aki in ("r15",): r = round(w.u(0.7, 1.15), 2)
    if aki in ("stage2", "stage3"): r = round(w.u(0.8, 1.0), 2)
    seq = {
        "none":       [r, round(r * 1.05, 2), round(r * 0.98, 2), round(r * 1.1, 2), round(r * 1.0, 2)],
        "r03":        [r, round(r + 0.35, 2), round(r + 0.42, 2), round(r + 0.3, 2), round(r + 0.1, 2)],
        "r15":        [r] + [round(r * (1 + 0.11 * k), 2) for k in range(1, 6)] + [round(r * 1.4, 2)],
        "stage2":     [r, round(r * 1.4, 2), round(r * 2.3, 2), round(r * 2.0, 2), round(r * 1.6, 2)],
        "stage3":     [r, round(r * 1.6, 2), round(r * 2.5, 2), round(r * 3.4, 2), round(r * 3.0, 2), round(r * 2.2, 2)],
        "community":  [round(base * 2.0, 2), round(base * 1.9, 2), round(base * 1.7, 2), round(base * 1.5, 2), round(base * 1.3, 2)],
        "decreasing": [round(base * 1.8, 2), round(base * 1.5, 2), round(base * 1.2, 2), round(base * 1.0, 2), round(base * 0.9, 2)],
    }[aki]
    if aki == "stage3" and w.p(0.4):
        seq = [1.3, 2.1, 3.2, 4.2, 3.8, 3.0]; r = 1.3; tag("stage3_by_absolute_4mgdl")
    seq = seq[:max(2, min(len(seq), los + 1))]
    t0 = sdt if start_known else sdt + D(hours=int(w.rng.integers(8, 20)))
    umol_inpatient = w.p(0.05)
    midnight_rows = w.p(0.07)
    stamps, vals = [], []
    for k, v in enumerate(seq):
        ts = t0 + D(hours=24 * k + int(w.rng.integers(-3, 4)))
        if k == 0: ts = t0 + D(hours=int(w.rng.integers(0, 3)))
        if midnight_rows and k == 1: ts = dt.datetime(ts.year, ts.month, ts.day)
        if ts.date() > disch: break
        if umol_inpatient: w.meas(pid, 3016723, ts, round(v * 88.4, 1), 8749, vid=main_vid)
        else:              w.meas(pid, 3016723, ts, v, 8840, vid=main_vid)
        if w.p(0.08):      w.meas(pid, 3016723, ts, v, 8840, vid=main_vid); tag("inpatient_exact_duplicate")
        stamps.append(ts); vals.append(v)
    if umol_inpatient: tag("inpatient_umol_rows")
    if midnight_rows: tag("inpatient_midnight_timestamp")
    # planted AKI ground truth (v5: only an earlier value can serve as the reference)
    ref = onset = None; stage = 0; rule = None
    if aki in ("r03", "r15", "stage2", "stage3") and len(vals) >= 2:
        for j in range(1, len(vals)):
            for i in range(j):
                hrs = (stamps[j] - stamps[i]).total_seconds() / 3600
                if (vals[j] - vals[i] >= 0.3 and hrs <= 48) or (vals[j] >= 1.5 * vals[i] and hrs <= 24 * 7):
                    ref, onset = i, j; rule = "0.3mg/dL" if (vals[j] - vals[i] >= 0.3 and hrs <= 48) else "1.5x"; break
            if onset is not None: break
    peak_i = None
    if onset is not None:
        peak_i = max(range(onset, len(vals)), key=lambda i: (vals[i], -i))
        ratio = vals[peak_i] / vals[ref]
        stage = 3 if (ratio >= 3.0 or vals[peak_i] >= 4.0) else 2 if ratio >= 2.0 else 1
    T.update(aki_scenario=aki, aki_rule=rule, aki_v5=int(onset is not None), aki_reference_scr=vals[ref] if ref is not None else None,
             aki_onset_scr=vals[onset] if onset is not None else None, aki_onset_datetime=stamps[onset] if onset is not None else None,
             peak_scr=vals[peak_i] if peak_i is not None else None, peak_datetime=stamps[peak_i] if peak_i is not None else None,
             last_inpatient_scr=vals[-1] if vals else None, last_inpatient_datetime=stamps[-1] if stamps else None,
             episode_max_scr=max(vals) if vals else None, kdigo_stage_creatinine=stage)

    # ---- acute severity ----
    sev = dict(ICU_admission=0, icu_route=None, mechanical_ventilation=0, vasopressor_exposure=0, sepsis_flag=0, contrast_exposure=0, acute_KRT=0)
    icu = w.p(0.45 if stage >= 2 else 0.08)
    if icu:
        sev["ICU_admission"] = 1
        route = w.choice(["visit_detail", "cpt", "both"])
        sev["icu_route"] = route
        if route in ("visit_detail", "both"):
            w.rows["visit_detail"].append(dict(visit_detail_id=w.big_id(), person_id=pid, visit_detail_concept_id=32037, visit_detail_start_date=admit,
                                               visit_detail_start_datetime=sdt, visit_detail_end_date=admit + D(days=min(los, 3)),
                                               visit_detail_end_datetime=None, visit_detail_type_concept_id=EHR, visit_detail_source_concept_id=0,
                                               visit_occurrence_id=main_vid))
        if route in ("cpt", "both"):
            w.proc(pid, 2514441, admit + D(days=1), vid=main_vid)
            if w.p(0.5): w.proc(pid, 2514442, admit + D(days=1), vid=main_vid)
        if w.p(0.5):
            sev["mechanical_ventilation"] = 1
            w.proc(pid, w.choice([2788036, 2788037, 2788038, 2008008]), admit + D(days=1), vid=main_vid)
        if w.p(0.6):
            sev["vasopressor_exposure"] = 1
            w.drug(pid, w.choice([99000601, 99000602, 99000604]), admit + D(days=1), 4171047, vid=main_vid, days=2)
    if w.p(0.05):
        w.drug(pid, 99000603, admit + D(days=1), 99000201, vid=main_vid); tag("phenylephrine_ophthalmic_trap")
    if w.p(0.20):
        sev["sepsis_flag"] = 1
        w.cond(pid, w.choice([132797, 196236]), admit + D(days=int(w.rng.integers(0, 2))), vid=main_vid)
    if w.p(0.01):
        w.cond(pid, 99000407, admit, vid=main_vid); tag("neonatal_sepsis_descendant_trap")
    if w.p(0.20):
        sev["contrast_exposure"] = 1
        cday = admit + D(days=int(w.rng.integers(0, min(3, los))))
        w.drug(pid, 99000606, cday, 4171047, vid=main_vid); w.proc(pid, 99000501, cday, vid=main_vid)
        T["contrast_before_onset"] = (onset is None) or (cday <= stamps[onset].date())
    if stage == 3 and w.p(0.5):
        sev["acute_KRT"] = 1
        for k in range(2):
            w.proc(pid, w.choice([4051326, 37018292, 46270933]), admit + D(days=2 + k), vid=main_vid)
        if sev["acute_KRT"]: T["kdigo_stage_creatinine"] = 3
    if w.p(0.02):
        w.obs(pid, 4019967, admit + D(days=1), vid=main_vid); tag("dialysis_dependence_observation_during_stay")
    T.update(sev)
    T["kdigo_max_stage"] = 3 if sev["acute_KRT"] else stage

    # ---- post-discharge ----
    T.update(death_date=None, death_known=0, day90_scr=None, day90_date=None, day90_window=None, recurrent_aki_date=None,
             maintenance_krt_date=None, readmission_date=None)
    alive_until = None
    if w.p(0.08):
        dd = disch + D(days=int(w.rng.integers(1, 400)))
        if w.p(0.10):
            w.death(pid, None, primary=False, src="registry"); T["death_known"] = 1; tag("death_record_without_date")
        else:
            w.death(pid, dd); T.update(death_date=dd, death_known=1); alive_until = dd
            if w.p(0.3):
                w.death(pid, dd + D(days=int(w.rng.integers(0, 10))), primary=False, src="registry"); tag("second_death_record")
            if w.p(0.1):
                w.death(pid, None, primary=False, src="registry")
    def after(lo, hi):
        d_ = disch + D(days=int(w.rng.integers(lo, hi + 1)))
        if alive_until and d_ > alive_until: return None
        if d_ > DATA_CUTOFF: return None
        return d_
    recovered = w.p(0.85 if stage <= 1 else 0.55)
    fu_val = round(base * (w.u(0.9, 1.15) if recovered else w.u(1.4, 2.2)), 2)
    T["day90_recovered_planted"] = recovered
    rr = w.rng.random()
    if rr < 0.60:
        d_ = after(75, 105); win = "75-105"
    elif rr < 0.75:
        d_ = after(60, 120) ; win = "60-120"
        if d_ and 75 <= (d_ - disch).days <= 105: d_ = disch + D(days=w.choice([65, 112])); d_ = None if (alive_until and d_ > alive_until) or d_ > DATA_CUTOFF else d_
    elif rr < 0.85:
        d_ = after(106, 180); win = "106-180"
    else:
        d_ = None; win = None
    if d_:
        vid = w.visit(pid, 9202, d_, d_)[0]
        w.meas(pid, 3016723, d_, fu_val, 8840, vid=vid)
        T.update(day90_scr=fu_val, day90_date=d_, day90_window=win)
        if w.p(0.3):
            d2 = after(1, 60)
            if d2: w.meas(pid, 3016723, d2, round(base * w.u(1.1, 1.5), 2), 8840)   # early follow-up: not a day-90 candidate
    if w.p(0.08):
        d_ = after(75, 105)
        if d_:
            evid = w.visit(pid, 9203, d_, d_)[0]
            w.meas(pid, 3016723, d_, round(base * w.u(1.5, 2.5), 2), 8840, vid=evid); tag("acute_ed_creatinine_in_day90_window")
    if w.p(0.10):
        rd = after(20, 70)
        if rd:
            re_ = rd + D(days=4); rvid = w.visit(pid, 9201, rd, re_)[0]; stays.append((rd, re_))
            seq2 = [round(base, 2), round(base * 1.3, 2), round(base * 1.7, 2), round(base * 1.4, 2)]
            for k, v in enumerate(seq2):
                w.meas(pid, 3016723, w.dtime(rd + D(days=k)), v, 8840, vid=rvid)
            w.cond(pid, 197320, rd + D(days=1), vid=rvid)
            T.update(recurrent_aki_date=rd + D(days=2), readmission_date=rd); tag("recurrent_aki_readmission")
    if w.p(0.03):
        md = after(30, 60)
        if md:
            w.obs(pid, 4019967, md)
            for k in range(8): w.proc(pid, 4051326, md + D(days=7 * k))
            T["maintenance_krt_date"] = md; tag("maintenance_dialysis_after_discharge")
    return T


def build(con, seed=SEED, n_persons=N_PERSONS, cdr=None):
    """Builds the mock CDR into a DuckDB connection: catalog = project name, schema = dataset name (the same three-part name as BigQuery)."""
    cdr = cdr or CDR
    project, dataset = cdr.split(".")
    w = _World(seed)
    _vocab(w)
    pids = sorted(set(int(x) for x in w.rng.integers(1_000_000, 9_999_999, size=n_persons * 2)))[:n_persons]
    for pid in pids:
        w.truth.append(_person(w, pid))
    con.execute('ATTACH \':memory:\' AS "%s"' % project)
    con.execute('CREATE SCHEMA "%s"."%s"' % (project, dataset))
    for t, cols in DDL.items():
        full = '"%s"."%s"."%s"' % (project, dataset, t)
        con.execute("CREATE TABLE %s (%s)" % (full, cols))
        rows = w.rows[t]
        if not rows:
            continue
        names = [c.strip().split()[0] for c in cols.replace("\n", " ").split(",")]
        data = [[_py(r.get(n)) for n in names] for r in rows]
        con.executemany("INSERT INTO %s VALUES (%s)" % (full, ", ".join("?" * len(names))), data)
    truth = pd.DataFrame(w.truth)
    truth["scenario"] = truth["scenario"].map(lambda s: ";".join(s))
    for c in ("baseline_scr_values", "constituent_visit_ids"):
        truth[c] = truth[c].map(lambda v: None if not isinstance(v, list) else ";".join(str(x) for x in v))
    con.register("_truth_df", truth)
    con.execute("CREATE TABLE main._mock_truth AS SELECT * FROM _truth_df")
    con.unregister("_truth_df")
    return con


def _py(v):
    if v is None: return None
    if isinstance(v, float) and math.isnan(v): return None
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating,)): return float(v)
    if isinstance(v, (np.bool_,)): return bool(v)
    return v


# ============================================================================================
# 4. EXECUTION LAYER: the same run_sql works both in the Workbench (BigQuery) and locally (DuckDB)
# ============================================================================================
def in_workbench():
    if "WORKSPACE_CDR" not in os.environ:
        return False
    try:
        import google.cloud.bigquery  # noqa: F401
        return True
    except ImportError:
        return False


def connect(seed=SEED, n_persons=N_PERSONS, cdr=None):
    """Builds the local mock CDR and makes it the default connection. Not needed in the Workbench (run_sql goes to BigQuery)."""
    global _CON
    if in_workbench():
        print("Workbench environment: WORKSPACE_CDR = %s; mock not built, run_sql goes to BigQuery" % CDR)
        return None
    import duckdb
    con = duckdb.connect(":memory:")
    build(con, seed=seed, n_persons=n_persons, cdr=cdr)
    _CON = con
    return con


def _to_duckdb(sql):
    import sqlglot
    try:
        return sqlglot.transpile(sql, read="bigquery", write="duckdb")[0]
    except Exception as e:
        raise RuntimeError("BigQuery SQL could not be translated to DuckDB (sqlglot): %s\nSQL:\n%s" % (e, sql)) from e


def run_sql(sql, con=None):
    """Runs a query written in the BigQuery (GoogleSQL) dialect and returns a pandas DataFrame.
    In the Workbench: via google.cloud.bigquery against `{CDR}.table` addresses. Locally: translated to DuckDB with sqlglot and run against the mock."""
    if in_workbench():
        from google.cloud import bigquery
        client = bigquery.Client(project=PROJECT)
        return client.query(sql).to_dataframe()
    con = con or _CON
    if con is None:
        raise RuntimeError("call aou_mock.connect() first")
    cur = con.execute(_to_duckdb(sql))
    df = cur.df()
    # mimic BigQuery to_dataframe() dtypes
    for name, tcode, *_ in cur.description:
        t = str(tcode).upper()
        if name not in df:
            continue
        if t in ("BIGINT", "INTEGER", "HUGEINT", "SMALLINT", "TINYINT", "UBIGINT", "UINTEGER"):
            df[name] = df[name].astype("Int64")
        elif t == "TIMESTAMP" and getattr(df[name].dt, "tz", None) is None:
            df[name] = df[name].dt.tz_localize("UTC")
        elif t == "DATE":
            df[name] = pd.to_datetime(df[name])
    return df


def dry_run(sql, con=None):
    """Returns how many bytes the query would read in BigQuery (cost check). In the mock it only translates the syntax and returns 0."""
    if in_workbench():
        from google.cloud import bigquery
        client = bigquery.Client(project=PROJECT)
        job = client.query(sql, job_config=bigquery.QueryJobConfig(dry_run=True, use_query_cache=False))
        return int(job.total_bytes_processed)
    _to_duckdb(sql)
    return 0


def columns(table):
    """Returns a table's columns and types (INFORMATION_SCHEMA in BigQuery, DuckDB in the mock)."""
    if in_workbench():
        project, dataset = CDR.split(".")
        return run_sql("SELECT column_name, data_type FROM `%s.%s.INFORMATION_SCHEMA.COLUMNS` WHERE table_name = '%s' ORDER BY ordinal_position"
                       % (project, dataset, table))
    project, dataset = CDR.split(".")
    return _CON.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_catalog = ? AND table_schema = ? AND table_name = ? ORDER BY ordinal_position",
                        [project, dataset, table]).df()


def mock_truth(con=None):
    """Per-person PLANTED ground truth: scenario tags and expected values. Exists only in the mock; not in the Workbench."""
    con = con or _CON
    return con.execute("SELECT * FROM main._mock_truth").df()


# ============================================================================================
# 5. SELF-TEST
# ============================================================================================
def self_test(verbose=True):
    con = connect()
    tr = mock_truth(con)
    cnt = run_sql("SELECT (SELECT COUNT(*) FROM `%s.person`) AS n_person, (SELECT COUNT(*) FROM `%s.measurement`) AS n_meas, "
                  "(SELECT COUNT(*) FROM `%s.visit_occurrence`) AS n_visit, (SELECT COUNT(*) FROM `%s.condition_occurrence`) AS n_cond"
                  % ((CDR,) * 4)).iloc[0]
    assert int(cnt["n_person"]) == N_PERSONS == len(tr), cnt
    # schema: every table has the expected columns
    for t, cols in DDL.items():
        names = {c.strip().split()[0] for c in cols.replace("\n", " ").split(",")}
        assert set(columns(t)["column_name"]) == names, t
    # are the traps really there
    units = run_sql("SELECT measurement_concept_id, unit_concept_id, COUNT(*) AS n FROM `%s.measurement` WHERE measurement_concept_id IN (3016723, 3020564) GROUP BY 1, 2 ORDER BY 1, 2" % CDR)
    seen = set(zip(units["measurement_concept_id"].astype(int), units["unit_concept_id"].astype("float")))
    for pair in [(3016723, 8840.0), (3016723, 8749.0), (3016723, 720870.0), (3016723, 0.0), (3020564, 8749.0)]:
        assert pair in seen, pair
    assert units["unit_concept_id"].isna().any(), "no NULL unit"
    ops = run_sql("SELECT operator_concept_id, COUNT(*) AS n FROM `%s.measurement` GROUP BY 1" % CDR)
    assert 4171756 in set(ops["operator_concept_id"].dropna().astype(int)), "no censoring operator"
    big = run_sql("SELECT COUNTIF(visit_occurrence_id > 9007199254740992) AS n_big, COUNT(*) AS n FROM `%s.visit_occurrence`" % CDR).iloc[0]
    assert int(big["n_big"]) > 0, "no ID above 2^53"
    mid = run_sql("SELECT COUNTIF(EXTRACT(HOUR FROM visit_start_datetime) = 0 AND EXTRACT(MINUTE FROM visit_start_datetime) = 0) AS n0, COUNT(*) AS n "
                  "FROM `%s.visit_occurrence` WHERE visit_concept_id IN (9201, 262)" % CDR).iloc[0]
    assert 0.3 < int(mid["n0"]) / int(mid["n"]) < 0.7, "about half of the inpatient stays should lack a time of day: %s" % dict(mid)
    dom = run_sql("SELECT concept_id, domain_id FROM `%s.concept` WHERE concept_id IN (4019967, 197320, 2514441)" % CDR).set_index("concept_id")["domain_id"]
    assert dom[4019967] == "Observation" and dom[197320] == "Condition"
    n_dial_cond = run_sql("SELECT COUNT(*) AS n FROM `%s.condition_occurrence` WHERE condition_concept_id = 4019967" % CDR).iloc[0]["n"]
    assert int(n_dial_cond) == 0, "dialysis dependence must not be in condition_occurrence"
    sx = run_sql("SELECT sex_at_birth_concept_id, COUNT(*) AS n FROM `%s.person` GROUP BY 1" % CDR)
    assert set(sx["sex_at_birth_concept_id"].astype(int)) <= {45878463, 45880669, 903096, 903079}
    dth = run_sql("SELECT person_id, COUNT(*) AS n, COUNTIF(death_date IS NULL) AS n_null FROM `%s.aou_death` GROUP BY 1" % CDR)
    assert (dth["n"] > 1).any() and (dth["n_null"] > 0).any(), "no multiple / undated death records"
    # is the table consistent with the planted ground truth: baseline (only in persons whose baseline rows are all 3016723) via the omop_example.py pattern
    tz = run_sql("SELECT measurement_datetime FROM `%s.measurement` LIMIT 1" % CDR)["measurement_datetime"]
    assert str(tz.dtype).endswith("UTC]"), tz.dtype
    try:
        import omop_example as ex
        d = ex.Dialect("bigquery", cdr=CDR)
        stays = run_sql("SELECT person_id, CAST(visit_occurrence_id AS STRING) AS stay_id, visit_start_date FROM `%s.visit_occurrence` WHERE visit_concept_id IN (9201, 262)" % CDR)
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):
            res, wf = ex.extract_baseline_creatinine(run_sql, d, stays_for_output=stays[["person_id", "stay_id"]])
        chk = res.merge(tr[["person_id", "index_admit_date", "baseline_scr_median", "baseline_rows_mass_concept_only", "scenario"]], on="person_id")
        chk = chk.merge(stays, on=["person_id", "stay_id"])
        chk = chk[(chk["visit_start_date"].dt.date == chk["index_admit_date"]) & chk["baseline_rows_mass_concept_only"]
                  & ~chk["scenario"].str.contains("prior_stay_in_window|baseline_exact_duplicate|baseline_out_of_range")]
        both = chk.dropna(subset=["baseline_scr_median", "baseline_creatinine_mg_dL"])
        # the example drops values outside the 0.1-20 range and converts µmol: must match the planted median within ±0.02
        bad = both[(both["baseline_scr_median"] - both["baseline_creatinine_mg_dL"]).abs() > 0.02]
        assert len(both) > 50 and len(bad) == 0, (len(both), bad.head())
        if verbose: print("omop_example.py baseline pattern ran on the mock: equals the planted median in %d index admissions" % len(both))
    except ImportError:
        if verbose: print("omop_example.py not found, baseline cross-check skipped")
    if verbose:
        print("Tables:", {t: int(con.execute('SELECT COUNT(*) FROM "%s"."%s"."%s"' % (tuple(CDR.split(".")) + (t,))).fetchone()[0]) for t in DDL})
        print("Scenario tags (number of persons):")
        tags = tr["scenario"].str.split(";").explode()
        print(tags[tags != ""].value_counts().to_string())
        print("MOCK CDR READY: CDR = %s" % CDR)
    return con


if __name__ == "__main__":
    self_test()
