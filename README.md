# aou-aki-ckd-prototype

Prototype pipeline for an *All of Us* study of progression from a single acute kidney injury (AKI) episode to chronic kidney disease (CKD), combining a clinical phenotype derived from OMOP tables with genetic features (APOL1 genotype, a CKD polygenic risk score, ancestry PCs).

Everything here runs **without access to real data**: the OMOP notebook runs on a deterministic mock of the Researcher Workbench CDR, and the genetics notebook runs on synthetic files with the real layouts. The same cells run unchanged on the Workbench.

> **Status: prototype.** The clinical logic is final and validated on the mock; the code is a first implementation (per-person `groupby.apply`, wide per-stay columns). A long-format rewrite is planned — see *Roadmap*.

## What it does

| Notebook | Input | Output |
|---|---|---|
| `notebooks/01_omop_aki_ckd_phenotype.ipynb` | OMOP `measurement`, `person`, `visit_occurrence`, `concept` | `aki_ckd_table`: one row per creatinine measurement with `stay_no`, `egfr`, `aki_flag`, `months_since_peak`, `ckd_developed`, `drop_flag` |
| `notebooks/02_genetics_apol1_prs.ipynb` | PLINK `.raw` dosages, PGS Catalog score file (PGS002237), `ancestry_preds.tsv`, APOL1 person table | `apol1_table` (one-hot genotype, high-risk flag), `prs_raw` per person |

Join key between the two: `person_id` = `IID`.

### Clinical decisions (fixed)

- **Inpatient stay** = `visit_concept_id` 9201 (inpatient) or 262 (ER → inpatient). 9202 and 9203 are not stays.
- **Creatinine**: concepts 3016723 (mg/dL) and 3020564 (molar). µmol/L ÷ 88.4; values > 10 mg/dL dropped (eGFR filed in the creatinine box, entry errors).
- **AKI** = within one stay, max − min ≥ 0.3 mg/dL **or** max ≥ 1.5 × min. A falling series (high on admission, recovered at discharge) counts; the whole stay is one window, the 48 h / 7 d limits are deliberately not applied.
- **Episode**: if a person has several AKI stays and the first and last peaks are > 3 months apart, the person is dropped (separate episodes). Otherwise the last peak is the anchor.
- **Baseline eGFR** = highest eGFR in the 365 days before the peak (inpatient and outpatient rows).
- **Follow-up eGFR** = first measurement ≥ 3 months after the peak.
- **CKD** = (baseline − follow-up) ≥ 30 % of baseline **or** (baseline > 90 and follow-up < 60). No baseline or no follow-up → `NaN` and `drop_flag = 1`.
- **eGFR**: CKD-EPI 2021, race-free.
- **APOL1 high risk** = two risk alleles (G1/G1, G1/G2, G2/G2).
- **PRS**: PGS002237 weights, oriented to the `.raw` counted allele, summed per person; raw, not standardized.

## Repository layout

```
notebooks/
  01_omop_aki_ckd_phenotype.ipynb   OMOP → AKI/CKD phenotype
  02_genetics_apol1_prs.ipynb       APOL1 + CKD-PRS
mock/
  aou_mock.py                       mock Workbench CDR (DuckDB; BigQuery SQL via sqlglot)
  make_synthetic_genetics.py        tiny synthetic inputs for notebook 02
docs/
  WORKBENCH_INSTRUCTIONS.md         rules for whoever (or whatever) runs this on the real Workbench
requirements.txt
```

## Run it locally

```bash
pip install -r requirements.txt
python mock/make_synthetic_genetics.py        # writes data/ for notebook 02
jupyter lab notebooks/
```

Notebook 01 builds the mock CDR on first run (`aou_mock.connect()`, ~400 persons, seed 20261014) and should print:

```
persons: 382
with AKI: 210
dropped: 99
CKD=1: 43 | CKD=0 (with AKI): 68
```

Notebook 02 reads from `DATA_DIR` (set in the first cell; defaults to the Colab Drive path — point it at `../data` for the synthetic set).

The mock plants the documented traps of the real CDR: mixed and missing units, eGFR rows under the creatinine concept, censoring operators, visit ids above 2^53, stays without clock time or with missing/inverted end dates, duplicate death rows, LOINC sex-at-birth concepts. `aou_mock.mock_truth()` returns the planted ground truth per person; `aou_mock.self_test()` checks the mock itself.

## Run it on the Workbench

Copy `mock/aou_mock.py` next to the notebooks. With `WORKSPACE_CDR` set, `connect()` is a no-op and `run_sql` goes to BigQuery through `google.cloud.bigquery`; `mock_truth()` does not exist there. Read `docs/WORKBENCH_INSTRUCTIONS.md` first — it lists what may be changed (file paths, a pandas ≥ 3 wrapper, a performance pre-filter) and what may not (every threshold and rule above).

## Roadmap

- Long-format rewrite: stay-level and person-level summary tables instead of per-stay columns (`{n}_stay_months_since_peak`), vectorized stay matching instead of the per-person loop.
- Apply the > 10 mg/dL filter to outpatient rows as well (currently inpatient only).
- Add KDIGO stage from the in-stay max/min ratio.
- PRS: compare the effect allele to the counted allele (currently to ALT), use `2 − dosage` instead of a sign flip, handle missing genotypes, standardize within ancestry, parse PC1–10 from `pca_features`.
- Replace the hard-coded `np.arange(0, 1800)` replication with column-wise broadcasting.
- Sensitivity analyses: median instead of max baseline; outpatient-only baseline.

## Notes

- No real *All of Us* data, identifiers or results are in this repository. Mock person ids are synthetic; mock concept ids that could not be verified against OHDSI are ≥ 99 000 000 and tagged `vocabulary_id = 'MOCK'`.
- The PGS002237 score file is not redistributed; download it from the [PGS Catalog](https://www.pgscatalog.org/score/PGS002237/).
- Written in Turkish first; identifiers and comments were translated for publication. Logic unchanged.

## License

MIT — see `LICENSE`.
