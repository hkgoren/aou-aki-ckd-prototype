# Instructions for running this on the All of Us Researcher Workbench

Written for the collaborator who runs the notebooks on the real CDR — and for any AI assistant helping them. The code was written by Odak and validated on the mock CDR. The job on the Workbench is to **run it unchanged**, report the numbers, and apply only the technical fixes listed here.

# Part A — `01_omop_aki_ckd_phenotype.ipynb`

## 1. Rules

**Do not**
- Change the logic of any function, "improve", vectorize or rewrite it.
- Change thresholds: 0.3 mg/dL, 1.5×, 3 months, 365 days, 30 %, eGFR 90/60, > 10 mg/dL.
- Widen or narrow the cohort (e.g. add 9202 outpatient visits, add the 48 h / 7 d windows).
- Remove the `try/except` in `detect_ckd`; an empty window there is meant to produce `drop_flag = 1`.

**You may**
- Add `LIMIT` to the SQL for a dry run (remove it afterwards).
- Apply the fixes in §6.
- Report runtime and memory.
- On an error, send the error, the line and a de-identified data sample to Odak.

## 2. Environment

- `aou_mock.connect()` is a no-op on the Workbench; `run_sql` goes straight to `google.cloud.bigquery` with `{CDR}.table` addresses. `CDR` is read from `WORKSPACE_CDR`.
- `aou_mock.mock_truth()` **does not exist** on the Workbench; it is not called in the notebook — do not call it.
- Concept and unit ids in the mock are the real OMOP ids (3016723, 3020564, 8840, 8749, 720870).

## 3. Run order

Run top to bottom ("Restart & Run all"). Cells depend on each other; do not skip.

| Step | What it does | Result |
|---|---|---|
| 1 | SQL: concept, creatinine measurements, person, stays (9201, 262) | `concept, measurement, person, stays` |
| 2 | Sex (1 M / 0 F / NaN) and age at measurement year | `measurement` + `sex, age_at_measurement` |
| 3 | Unit codes → labels (`mg/dL`, `umol/L`) | `measurement` |
| 4 | `stay_no` on every measurement (NaN if outpatient) | `measurement_with_stays` → inpatient / outpatient |
| 5 | Inpatient: µmol÷88.4, drop > 10, AKI flag, months since peak | `measurement_aki` |
| 6 | Concat; µmol÷88.4 on outpatient rows (`convert_units`) | `all_measurements` |
| 7 | CKD-EPI 2021 eGFR | same table |
| 8 | Months since each AKI peak on all rows | same table |
| 9 | Per person: baseline vs follow-up → CKD, drop | **`aki_ckd_table`** |

## 4. Clinical decisions (fixed)

- **Stay** = `visit_concept_id` 9201 (inpatient) or 262 (ER → inpatient). 9202 and 9203 are not stays.
- **AKI** = within one stay, max − min ≥ 0.3 mg/dL **or** max ≥ 1.5 × min. A falling series counts (KDIGO retrospective baseline). The whole stay is one window; no 48 h / 7 d limit.
- **Peak (anchor)** = date of the highest creatinine in the stay. Time unit: (date − peak) / 30 → months.
- **Multiple AKI stays:** the last AKI stay is the anchor. First-to-last peak gap > 3 months → `drop_flag = 1` (separate episodes, outside the single-episode cohort). ≤ 3 months → one episode.
- **Baseline eGFR** = highest eGFR among all measurements (inpatient + outpatient) in the 365 days before the peak.
- **Follow-up eGFR** = first measurement, by date, ≥ 3 months after the peak.
- **CKD** = (baseline − follow-up) ≥ 0.3 × baseline **or** (baseline > 90 and follow-up < 60).
- Empty baseline window or no follow-up → `drop_flag = 1`, `ckd_developed = NaN`.
- **eGFR**: CKD-EPI 2021 race-free. Persons with `sex = NaN` get no eGFR.

## 5. Output: `aki_ckd_table`

One row per creatinine measurement. Person-level decisions are written on every row.

| Column | Meaning |
|---|---|
| `person_id` | OMOP person id (join key to the genetics table) |
| `measurement_date` | measurement date |
| `value_as_number` | creatinine, mg/dL (µmol converted; > 10 dropped on inpatient rows) |
| `sex` | 1 male, 0 female, NaN other |
| `age_at_measurement` | measurement year − birth year |
| `egfr` | CKD-EPI 2021 eGFR |
| `stay_no` | the person's n-th stay; NaN on outpatient rows |
| `stay_start_date`, `stay_end_date` | dates of that stay |
| `aki_flag` | AKI in this row's stay (0 on outpatient rows) |
| `months_since_peak` | months relative to this stay's peak (inpatient rows only) |
| `{n}_stay_months_since_peak` | months relative to the peak of AKI stay *n*, on all of the person's rows |
| `{n}_stay_aki_flag` | AKI in stay *n* |
| `ckd_developed` | 1 / 0 / NaN (undecided); 0 for persons without AKI |
| `drop_flag` | 1 = outside cohort; 0 = multi-AKI single episode; NaN = single-AKI decided **or** no AKI |

Person table: `aki_ckd_table.groupby('person_id').agg(aki=('aki_flag','max'), ckd=('ckd_developed','max'), drp=('drop_flag','max'))`.
**Model cohort** = `aki == 1` and `ckd.notna()`. Do not use `drop_flag` alone as the filter (it mixes NaN meanings).

## 6. Permitted technical fixes

Only these; tell Odak which ones you applied.

**a) pandas ≥ 3.** `groupby(...).apply(f)` no longer passes the grouping column to `f`; `dataset['person_id']` inside `assign_stays` raises KeyError. Check `pd.__version__` first. On 3.x wrap the five `apply` calls like this (do not touch the functions):
```python
.groupby('person_id', group_keys=False).apply(lambda d: assign_stays(d.assign(person_id=d.name)))
.groupby(['person_id','stay_no'], group_keys=False).apply(lambda d: detect_aki(d.assign(person_id=d.name[0], stay_no=d.name[1])))
.groupby('person_id', group_keys=False).apply(lambda d: convert_units(d.assign(person_id=d.name)))
.groupby('person_id', group_keys=False).apply(lambda d: compute_time_since_each_aki(d.assign(person_id=d.name)))
.groupby('person_id', group_keys=False).apply(lambda d: detect_ckd(d.assign(person_id=d.name)))
```

**b) Date dtype.** If BigQuery `DATE` columns come back without `.dt` support (object / `dbdate`), add `measurement['measurement_date'] = pd.to_datetime(measurement['measurement_date'])` right after the SQL; same for `stays['visit_start_date']`, `stays['visit_end_date']` if needed.

**c) Performance pre-filter (does not change results).** Persons without a stay are outside the cohort; you may add to the `measurement` query:
```sql
AND person_id IN (SELECT person_id FROM `{CDR}.visit_occurrence` WHERE visit_concept_id IN (9201, 262))
```
On the mock, the stay-matching loop took ~36 min for 300 k persons; real runtime scales with measurement count.

**d) `person` query** uses `SELECT *`; if large, select only `person_id, sex_at_birth_source_value, year_of_birth`.

## 7. Numbers to report

The last cell prints; send all of it:
- persons, with AKI, dropped, CKD=1, CKD=0
- persons with `stay_no >= 10` (if not 0, tell Odak; those are reviewed by hand)
- outpatient rows with `value_as_number > 10` (not dropped there; eGFR values filed under creatinine — unit 720870 — stay. Odak decides.)

Also send (SQL or pandas):
- Pre-cleaning row counts and median value by `unit_concept_id` × `measurement_concept_id`. In particular 8840, 8749, 720870, 0/NULL and **anything else**. Any unexpected unit (e.g. 8510) with its value range; the code treats it as mg/dL.
- Rows with concept 3020564 (molar) and `unit_concept_id` NULL. The code cannot convert them; they are removed by the > 10 filter. If many, Odak decides.
- Rows removed by `value_as_number > 10`.
- Persons with `sex = NaN`.

## 8. Not yet implemented — Odak decides, do not add

- KDIGO stage (1/2/3). Only AKI yes/no exists; stage would come from the in-stay max/min ratio.
- > 10 mg/dL filter on outpatient rows (currently inpatient only).
- Baseline source column (outpatient vs inpatient).
- Upper bound on the follow-up window (currently ≥ 3 months, no upper bound).
- Merge with the genetics table (PRS, APOL1, PC1–10) on `person_id`.

## 9. Reference: mock result

400-person mock CDR, 382 with creatinine: AKI 210, dropped 99, CKD=1 43, CKD=0 68; outpatient rows > 10: 56. Real-data proportions will differ; these numbers only prove the code runs.

---

# Part B — `02_genetics_apol1_prs.ipynb`

Developed on synthetic files. Computation logic is fixed; the job is to point the inputs at the real Workbench files, run, and report.

## B1. Input mapping (synthetic → real)

| Variable | Synthetic file | Workbench equivalent |
|---|---|---|
| `dosage_raw` | `acaf_threshold.dummy.raw.gz` | `plink2 --export A` output from the ACAF threshold callset (PLINK bed), restricted to PGS002237 variants. Column headers must be `chr:pos:ref:alt_COUNTED`; `FID`, `IID` required |
| `beta_scores` | `PGS002237_hmPOS_GRCh38.dummy.txt.gz` | `PGS002237_hmPOS_GRCh38.txt.gz` from the PGS Catalog (harmonized, GRCh38). Needed columns: `chr_name, chr_position, effect_allele, effect_weight, hm_chr, hm_pos` |
| `ancestry_preds` | `ancestry_preds.dummy.tsv` | All of Us `ancestry_preds.tsv` (`research_id, ancestry_pred, pca_features`) |
| `apol1_raw` | xlsx sheet `apol1_person_table` | **No ready-made table on the Workbench (as far as we know).** Derive the APOL1 genotype from three variants: G1 = rs73885319 (S342G) + rs60910145 (I384M), G2 = rs71785313 (6-bp deletion). The output must carry the **same column names** as the synthetic file: `IID`, `apol1_genotype` (`G0/G0, G0/G1, G0/G2, G1/G1, G1/G2, G2/G2`), `related_flag`, `sample_qc_flag`, `known_issue_8`, `low_coverage_flag`. Fill the flags from the All of Us relatedness and flagged-samples lists; for a flag that does not exist, provide an all-False column — do not edit `qc_flag_cols` |

Changing file paths (`DATA_DIR`) is fine; changing column names is not.

## B2. Rules

- Do not change the orientation rule, the one-hot, the QC logic, or the score sum.
- `apol1_high_risk` (two risk alleles) is fixed.
- Do not add the items in B3; Odak decides.

## B3. Known open points — Odak decides

1. **`np.arange(0, 1800)`.** `weight_matrix` is replicated to the synthetic row count (1800). With more persons, the extra rows multiply by NaN and `prs_raw` becomes NaN. The commented line `dosage_raw.loc[:, cols] * weight_row.iloc[0]` does the same job independent of row count. If `prs_raw NaN` is not 0 after the run, tell Odak.
2. **Orientation compares the ALT allele.** The code parses `counted_allele` but compares `alt_allele` and `ref_allele` to `effect_allele`. If `--export A` did not count ALT for every variant, the sign is wrong for those. Before running, report the share of `.raw` headers whose `_` suffix equals ALT.
3. **Sign flip vs `2 − dosage`.** When the effect allele is the opposite of the counted allele, the exact contribution is `w × (2 − dosage)`; the code uses `−w × dosage`. The difference is a constant `2w` per variant, i.e. the same shift for everyone; irrelevant after standardization.
4. **Missing genotypes.** NA dosages count as 0 in `sum`; persons with many missing calls get a lower score. Report the per-person NA rate.
5. **PC1–10 are not extracted yet.** `pca_features` is a string list; the PCs for the model will be parsed from it.
6. **Standardization by ancestry.** Not done; Odak decides after seeing the B4 table.
7. **pandas ≥ 3.** `weight_matrix` is built from an object-dtype frame; on pandas 3 the element-wise product cannot be written back into float columns (`TypeError: Invalid value ... for dtype 'float64'`). Permitted fix: append `.astype(float)` to the `weight_matrix = ...ffill()` line. On pandas 2.x no change is needed.

## B4. Numbers to report

The last cell prints:
- `.raw` variant count, matched to the score file, oriented (both subsets), dropped
- persons, `prs_raw` NaN count
- APOL1 persons before/after QC, `apol1_high_risk` distribution
- mean `prs_raw` by `ancestry_pred` (previous cell)

Also: the overlap between `.raw` persons and `person_id` in `aki_ckd_table`. Join key `IID` = `person_id`.

## B5. Outputs

| Table | Row | Columns |
|---|---|---|
| `apol1_table` | person (QC-passed) | `IID`, `apol1_genotype`, `apol1_genotype_*` (one-hot), `apol1_high_risk` |
| `dosage_raw[['IID','prs_raw']]` | person | raw PRS (not standardized) |
