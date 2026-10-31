"""Writes a tiny synthetic input set for notebooks/02_genetics_apol1_prs.ipynb into ../data/.

Same file names, separators and column layouts as the real inputs (PLINK --export A dosage matrix,
PGS Catalog harmonized score file, ancestry_preds.tsv, APOL1 person table). Values are random.
Run:  python mock/make_synthetic_genetics.py
Then set DATA_DIR = "../data" in the first cell of notebook 02.
"""
import gzip
import os

import numpy as np
import pandas as pd

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(1)
n = 60
ids = np.arange(1000, 1000 + n)

# PLINK .raw: FID IID PAT MAT SEX PHENOTYPE + one column per variant, named chr:pos:ref:alt_COUNTED
cols = ["chr1:100:A:G_G", "chr2:200:C:T_T", "chr3:300:G:A_A", "chr4:400:T:C_C", "chr5:500:A:T_T"]
raw = pd.DataFrame({"FID": ids, "IID": ids, "PAT": 0, "MAT": 0, "SEX": rng.integers(1, 3, n), "PHENOTYPE": -9})
for c in cols:
    v = rng.integers(0, 3, n).astype(float)
    v[rng.random(n) < 0.05] = np.nan          # a few missing calls
    raw[c] = v
raw.to_csv(os.path.join(OUT, "acaf_threshold.dummy.raw.gz"), sep="\t", index=False, compression="gzip")

# PGS Catalog harmonized score file: '#' header lines, then tab-separated columns.
# rs1: effect = ALT (kept as is); rs2: effect = REF (sign flipped); rs3: neither (dropped);
# rs4: empty hm_pos (dropped); rs5: duplicated row (de-duplicated).
pgs = """# PGS CATALOG SCORING FILE (synthetic)
# pgs_id=PGS002237
rsID\tchr_name\tchr_position\teffect_allele\tother_allele\teffect_weight\thm_chr\thm_pos
rs1\t1\t99\tG\tA\t0.12\t1\t100
rs2\t2\t199\tC\tT\t0.30\t2\t200
rs3\t3\t299\tT\tC\t0.05\t3\t300
rs4\t4\t399\tC\tT\t-0.20\t4\t
rs5\t5\t499\tT\tA\t0.08\t5\t500
rs5\t5\t499\tT\tA\t0.08\t5\t500
"""
with gzip.open(os.path.join(OUT, "PGS002237_hmPOS_GRCh38.dummy.txt.gz"), "wt") as f:
    f.write(pgs)

# ancestry_preds.tsv
anc = pd.DataFrame({
    "research_id": ids,
    "ancestry_pred": rng.choice(["afr", "eur", "amr"], n),
    "pca_features": ["[" + ",".join(map(str, rng.normal(size=16).round(3))) + "]" for _ in range(n)],
})
anc.to_csv(os.path.join(OUT, "ancestry_preds.dummy.tsv"), sep="\t", index=False)

# APOL1 person table (sheet apol1_person_table)
ap = pd.DataFrame({
    "IID": ids,
    "apol1_genotype": rng.choice(["G0/G0", "G0/G1", "G1/G1", "G1/G2", "G2/G2", "G0/G2"], n, p=[.5, .2, .1, .1, .05, .05]),
    "related_flag": rng.random(n) < 0.05,
    "sample_qc_flag": rng.random(n) < 0.03,
    "known_issue_8": False,
    "low_coverage_flag": rng.random(n) < 0.02,
})
with pd.ExcelWriter(os.path.join(OUT, "aou_genomic_dummy_2026-09-12_VIEWER_ONLY.xlsx")) as w:
    ap.to_excel(w, sheet_name="apol1_person_table", index=False)

print("synthetic genetics inputs written to", os.path.abspath(OUT))
