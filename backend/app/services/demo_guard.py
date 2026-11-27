"""Which records may appear in the Demo Playground and Challenge Mode.

Only MIT-BIH records that are in training or validation under BOTH the app model's split and the study split, and whose
patient appears in no test part of either. This excludes all of DS2 and every INCART record.
"""
from __future__ import annotations

import json

from app.core.config import CONFIG_PATH, SPLITS_DIR


def eligible_records() -> dict[str, dict[str, str]]:
    """record id -> {"app_model_split": ..., "study_split": ...}"""
    app_split = json.loads(CONFIG_PATH.read_text())["split"]
    study = json.loads((SPLITS_DIR / "multidb_v1.json").read_text())
    patients = study["patients"]
    test_records = set(study["test_ds2"]) | set(study["test_incart"]) | {f"mitdb/{r}" for r in app_split["test_records"]}
    test_patients = {patients.get(r, r) for r in test_records}
    # record 202 is the same subject as record 201 (study split file and app config both record this)
    for a, b in app_split.get("subject_groups_merged", {}).items():
        if f"mitdb/{a}" in test_records:
            test_records.add(f"mitdb/{b}")
    allowed = {}
    for part in ("train", "validation"):
        for record in app_split[f"{part}_records"]:
            key = f"mitdb/{record}"
            study_part = next((p for p in ("train", "validation") if key in study[p]), None)
            if study_part and key not in test_records and patients.get(key, key) not in test_patients:
                allowed[record] = {"app_model_split": part, "study_split": study_part}
    return allowed
