"""Build the evaluation-protocol split files and compare the project split with de Chazal DS1/DS2.

Writes backend/splits/ds1_ds2.json, cv5_dev.json and cv5_dev_it0..4.json. No model predictions are used,
and test records are never read when assigning folds.
Run: backend/.venv/bin/python backend/scripts/make_splits.py
"""
import json
from pathlib import Path

import numpy as np
import wfdb

ROOT = Path(__file__).resolve().parents[2]
CFG = json.loads((ROOT / "backend/models/preprocessing_config.json").read_text())
SPLITS = ROOT / "backend/splits"
C = ["N", "S", "V", "F"]
# de Chazal et al., IEEE TBME 2004 (record lists as reproduced in the literature)
DS1 = "101 106 108 109 112 114 115 116 118 119 122 124 201 203 205 207 208 209 215 220 223 230".split()
DS2 = "100 103 105 111 113 117 121 123 200 202 210 212 213 214 219 221 222 228 231 232 233 234".split()
PACED = ["102", "104", "107", "217"]
SEED, DRAWS = 20241127, 20000


def counts(rec):
    ann = wfdb.rdann(str(ROOT / "results/mitdb" / rec), "atr")
    k = [C.index(CFG["aami_mapping"][s]) for s in ann.symbol if s in CFG["aami_mapping"]]
    return np.bincount(k, minlength=4)


s = CFG["split"]
train, val, test = s["train_records"], s["validation_records"], s["test_records"]
cmp = {"project_test_equals_DS2_minus_202": set(test) == set(DS2) - {"202"},
       "project_train_plus_validation_equals_DS1_plus_202": set(train) | set(val) == set(DS1) | {"202"},
       "DS1_DS2_known_subject_overlap": "records 201 (DS1) and 202 (DS2) come from the same subject"}
print(json.dumps(cmp, indent=1))
ds1_counts = sum(counts(r) for r in DS1)
print("DS1 annotated N,S,V,F:", ds1_counts.tolist())

inner_val = [r for r in val if r != "202"]
ds = {"name": "ds1_ds2", "description": "de Chazal DS1 (training, with an inner validation subset) / DS2 (test).",
      "train": [r for r in DS1 if r not in inner_val], "validation": inner_val, "test": DS2,
      "subject_groups": {}, "known_subject_overlap": cmp["DS1_DS2_known_subject_overlap"], "excluded": {"paced": PACED},
      "comparison_with_project_split": cmp}
(SPLITS / "ds1_ds2.json").write_text(json.dumps(ds, indent=1))
print("DS1 inner-train equals project train:", set(ds["train"]) == set(train))

# 5 subject-wise folds over DS1 only. Record 202 is a DS2 (final test) record, so cross-validation never uses it.
dev = sorted(set(DS1))
subjects = {}
for r in dev:
    subjects.setdefault(s["subject_groups_merged"].get(r, r), []).append(r)
names = sorted(subjects)
mat = np.array([sum(counts(r) for r in subjects[n]) for n in names], float)
rng = np.random.default_rng(SEED)
best = None
sizes = np.array([5, 5, 4, 4, 4])
for _ in range(DRAWS):
    fold = np.repeat(np.arange(5), sizes)[rng.permutation(len(names))]
    tot = np.array([mat[fold == k].sum(0) for k in range(5)])
    share = tot / mat.sum(0)
    score = ((share[:, [1, 2]] - 0.2) ** 2).sum() + ((share[:, 0] - 0.2) ** 2).sum()   # balance N, S, V
    if best is None or score < best[0]:
        best = (score, fold, tot)
_, fold, tot = best
folds = [[r for n, f in zip(names, fold) if f == k for r in subjects[n]] for k in range(5)]
cv = {"name": "cv5_dev", "description": "5 subject-wise folds over DS1 only (no DS2 record). Iteration k: evaluate fold k, "
      "early-stop/tune on fold (k+1)%5, train on the other three.", "seed": SEED, "draws": DRAWS,
      "criterion": "minimise squared deviation from 20% of N, S and V per fold; F cannot be balanced",
      "folds": folds, "fold_counts_N_S_V_F": tot.astype(int).tolist(), "excluded": {"paced": PACED, "held_out_test": DS2}}
(SPLITS / "cv5_dev.json").write_text(json.dumps(cv, indent=1))
assert sorted(sum(folds, [])) == dev and not set(sum(folds, [])) & set(DS2)
for k in range(5):
    print(f"fold {k}: {len(folds[k])} records {folds[k]}  N,S,V,F={tot[k].astype(int).tolist()}")
r208 = counts("208")[3]
print(f"record 208 holds {r208} of {int(mat[:, 3].sum())} development F beats")
# per-iteration iteration files in the format train.py reads
for k in range(5):
    es = (k + 1) % 5
    it = {"name": f"cv5_dev_it{k}", "train": [r for j in range(5) if j not in (k, es) for r in folds[j]], "validation": folds[es],
          "evaluation": folds[k], "test": DS2, "subject_groups": {}, "excluded": {"paced": PACED}}
    (SPLITS / f"cv5_dev_it{k}.json").write_text(json.dumps(it, indent=1))
    tr = sum(counts(r) for r in it["train"])
    print(f"iteration {k}: train N,S,V,F={tr.tolist()}  early-stop F={int(tot[es][3])}  eval F={int(tot[k][3])}")
