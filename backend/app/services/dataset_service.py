"""Reads the tracked MIT-BIH records for the Dataset Explorer."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import wfdb

from app.core.config import CONFIG_PATH
from app.ml.preprocessing import extract_windows, filter_continuous

CLASSES = ("N", "S", "V", "F")


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text())


def record_split(record: str) -> str:
    split = _config()["split"]
    if record in split["excluded_paced_records"]:
        return "excluded (paced)"
    return next((p for p in ("train", "validation", "test") if record in split[f"{p}_records"]), "not used")


@lru_cache(maxsize=64)
def load_record(mitdb: Path, record: str) -> dict:
    data = wfdb.rdrecord(str(mitdb / record), physical=True)
    lead = "MLII" if "MLII" in data.sig_name else data.sig_name[0]
    index = data.sig_name.index(lead)
    raw = data.p_signal[:, index].astype(np.float32)
    ann = wfdb.rdann(str(mitdb / record), "atr")
    mapping = _config()["aami_mapping"]
    beats = [(int(s), mapping[sym]) for s, sym in zip(ann.sample, ann.symbol) if sym in mapping]
    return {"raw": raw, "filtered": filter_continuous(raw, data.fs), "fs": float(data.fs), "lead": lead,
            "units": data.units[index], "beats": beats}


def list_records(mitdb: Path) -> list[dict]:
    out = []
    for header in sorted(mitdb.glob("*.hea")):
        record = load_record(mitdb, header.stem)
        counts = {c: sum(1 for _, k in record["beats"] if k == c) for c in CLASSES}
        out.append({"record": header.stem, "split": record_split(header.stem), "lead": record["lead"],
                    "duration_seconds": len(record["raw"]) / record["fs"], "annotated_beats": counts})
    return out


def segment(mitdb: Path, record: str, start: float, seconds: float) -> dict:
    r = load_record(mitdb, record)
    a = int(min(max(0.0, start) * r["fs"], max(0, len(r["raw"]) - 1)))
    b = min(len(r["raw"]), a + int(seconds * r["fs"]))
    return {"record": record, "split": record_split(record), "lead": r["lead"], "units": r["units"], "sampling_rate": r["fs"],
            "start_sample": a, "duration_seconds": len(r["raw"]) / r["fs"],
            "raw": r["raw"][a:b].astype(float).tolist(), "filtered": r["filtered"][a:b].astype(float).tolist(),
            "beats": [{"offset": p - a, "label": k} for p, k in r["beats"] if a <= p < b]}


def window_payload(r: dict, position: int) -> dict:
    batch = extract_windows(r["filtered"], np.asarray([position], dtype=np.int64))
    lo, hi = position - 360, position + 360
    z = batch.windows[0, :, 0]
    stats = lambda x: {"mean": float(np.mean(x)), "sd": float(np.std(x))}
    return {"sample": position, "start_sample": lo, "sampling_rate": r["fs"], "units": r["units"],
            "raw": r["raw"][lo:hi].astype(float).tolist(), "filtered": r["filtered"][lo:hi].astype(float).tolist(),
            "model_input": z.astype(float).tolist(),
            "stats": {"raw": stats(r["raw"][lo:hi]), "filtered": stats(r["filtered"][lo:hi]), "model_input": stats(z)}}


def class_positions(r: dict, label: str) -> list[int]:
    return [p for p, k in r["beats"] if k == label and 360 <= p <= len(r["raw"]) - 360]


@lru_cache(maxsize=1)
def representatives(mitdb: Path) -> list[dict]:
    """Per class, the TRAINING window closest (L2) to the mean of that class's standardized training windows."""
    train = _config()["split"]["train_records"]
    out = []
    for label in CLASSES:
        windows, where = [], []
        for record in train:
            r = load_record(mitdb, record)
            positions = class_positions(r, label)
            if positions:
                windows.append(extract_windows(r["filtered"], np.asarray(positions, dtype=np.int64)).windows[..., 0])
                where += [(record, p) for p in positions]
        stack = np.concatenate(windows)
        best = int(np.argmin(((stack - stack.mean(axis=0)) ** 2).sum(axis=1)))
        record, position = where[best]
        out.append({"label": label, "record": record, "sample": position, "candidates": len(where),
                    "model_input": stack[best].astype(float).tolist()})
    return out
