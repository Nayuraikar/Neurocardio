from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from app.ml.preprocessing import NORMALIZATION_EPSILON
from app.services.demo_guard import eligible_records
from app.services.study_service import read_json


def load_samples(artifacts_dir: Path) -> tuple[dict | None, str]:
    """-> (payload with only eligible samples, problem). Eligibility is re-checked on every load."""
    data, problem = read_json(artifacts_dir / "demo_samples.json")
    if data is None:
        return None, problem or "Demo samples have not been built. Run backend/tools/build_demo_samples.py."
    allowed = eligible_records()
    bad = [s["id"] for s in data["samples"] if s.get("database") != "mitdb" or s["record"] not in allowed
           or s["app_model_split"] != allowed[s["record"]]["app_model_split"]]
    if bad:
        return None, f"Demo file refused: {len(bad)} samples are not from eligible training or validation patients."
    return data, ""


def standardize(window: list[float]) -> np.ndarray:
    x = np.asarray(window, dtype=np.float32)
    return ((x - x.mean()) / (x.std() + NORMALIZATION_EPSILON)).astype(np.float32)


def describe(s: dict, with_label: bool = True) -> dict:
    out = {k: s[k] for k in ("id", "database", "record", "sample", "app_model_split", "study_split")}
    if with_label:
        out["label"] = s["label"]
    return out


def challenge_ids(samples: list[dict], per_class: int, seed: int) -> list[dict]:
    """A class-balanced, shuffled round. Labels are withheld."""
    rng = random.Random(seed)
    picked = []
    for label in "NSVF":
        group = [s for s in samples if s["label"] == label]
        picked += rng.sample(group, min(per_class, len(group)))
    rng.shuffle(picked)
    return [describe(s, with_label=False) for s in picked]
