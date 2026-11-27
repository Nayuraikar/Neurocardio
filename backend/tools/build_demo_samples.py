"""Builds backend/artifacts/demo_samples.json: a few labelled windows from training and validation patients only.

    python backend/tools/build_demo_samples.py

Windows are picked at random with a fixed seed, spread across records. The model is never consulted, so samples are
not chosen by whether it gets them right. Each window is stored after the bandpass filter and before standardization.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import wfdb

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
from app.core.config import ARTIFACTS_DIR, CONFIG_PATH, PREPROCESSING_VERSION, PROJECT_DIR  # noqa: E402
from app.ml.preprocessing import extract_windows, filter_continuous  # noqa: E402
from app.services.demo_guard import eligible_records  # noqa: E402

SEED = 20261127
PER_CLASS_PER_SPLIT = 12
MAX_BYTES = 2 * 1024 * 1024


def main() -> None:
    mapping = json.loads(CONFIG_PATH.read_text())["aami_mapping"]
    allowed = eligible_records()
    pool: dict[tuple[str, str], dict[str, list[int]]] = {}
    filtered = {}
    for record, splits in sorted(allowed.items()):
        base = str(PROJECT_DIR / "results" / "mitdb" / record)
        data = wfdb.rdrecord(base, physical=True)
        filtered[record] = filter_continuous(data.p_signal[:, data.sig_name.index("MLII")], data.fs)
        ann = wfdb.rdann(base, "atr")
        for sample, symbol in zip(ann.sample, ann.symbol):
            if symbol in mapping and 360 <= sample <= len(filtered[record]) - 360:
                pool.setdefault((mapping[symbol], splits["app_model_split"]), {}).setdefault(record, []).append(int(sample))

    rng = np.random.default_rng(SEED)
    samples = []
    for (label, part), by_record in sorted(pool.items()):
        queues = {r: list(rng.permutation(v)) for r, v in by_record.items()}
        order = list(rng.permutation(sorted(queues)))
        taken = 0
        while taken < PER_CLASS_PER_SPLIT and any(queues.values()):
            for record in order:  # one per record per round
                if taken >= PER_CLASS_PER_SPLIT or not queues[record]:
                    continue
                position = int(queues[record].pop())
                window = filtered[record][position - 360:position + 360]
                assert len(extract_windows(filtered[record], np.array([position])).windows) == 1
                samples.append({"id": f"{record}-{position}", "database": "mitdb", "record": record, "sample": position,
                                "label": label, **allowed[record], "filtered": [round(float(v), 4) for v in window]})
                taken += 1

    payload = {"source": "MIT-BIH Arrhythmia Database (PhysioNet), lead MLII, 360 Hz", "seed": SEED,
               "preprocessing_version": PREPROCESSING_VERSION,
               "selection": "random with a fixed seed, spread across records; the model is not consulted",
               "eligibility": "training or validation patients under both splits; no DS2 and no INCART records",
               "signal": "bandpass filtered, before per-window standardization", "samples": samples}
    text = json.dumps(payload, separators=(",", ":"))
    assert len(text.encode()) < MAX_BYTES, f"demo file is {len(text)} bytes"
    target = ARTIFACTS_DIR / "demo_samples.json"
    target.write_text(text)
    counts: dict = {}
    for s in samples:
        counts[(s["label"], s["app_model_split"])] = counts.get((s["label"], s["app_model_split"]), 0) + 1
    print(f"wrote {target} ({len(text.encode()):,} bytes, {len(samples)} windows)")
    print(counts)
    print("records used:", sorted({s["record"] for s in samples}))


if __name__ == "__main__":
    main()
