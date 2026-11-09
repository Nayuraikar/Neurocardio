"""Evaluate WFDB XQRS against MIT-BIH beat annotations (150 ms one-to-one match)."""
from __future__ import annotations
import csv
from pathlib import Path
import wfdb
from app.ml.preprocessing import FS_HZ, filter_continuous
from app.ml.rpeak_detection import detect_r_peaks, score_detector

ROOT = Path(__file__).resolve().parents[2] / "results" / "mitdb"
RECORDS = ("100", "103", "105", "111", "113", "117", "121", "123", "200", "210", "212", "213", "214", "219", "221", "222", "228", "231", "232", "233", "234")
BEAT_SYMBOLS = set("NLR ejAaJSVEF/fQ".replace(" ", ""))

def main() -> None:
    rows = []
    for record in RECORDS:
        rec = wfdb.rdrecord(str(ROOT / record), physical=True)
        lead_idx = rec.sig_name.index("MLII") if "MLII" in rec.sig_name else 0
        ann = wfdb.rdann(str(ROOT / record), "atr")
        ref = ann.sample[[s in BEAT_SYMBOLS for s in ann.symbol]]
        filtered = filter_continuous(rec.p_signal[:, lead_idx], float(rec.fs))
        detected = detect_r_peaks(filtered, float(rec.fs))
        rows.append({"record": record, "lead": rec.sig_name[lead_idx], **score_detector(ref, detected, int(round(0.15 * FS_HZ)))})
        print(record, rows[-1])
    keys = list(rows[0])
    out = Path(__file__).resolve().parents[2] / "docs" / "rpeak-evaluation.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    tp = sum(r["tp"] for r in rows); fp = sum(r["fp"] for r in rows); fn = sum(r["fn"] for r in rows)
    # Reconstruct pooled metrics directly; beat timing errors are averaged over true matches.
    import numpy as np
    sens = tp / (tp + fn) if tp + fn else float("nan")
    ppv = tp / (tp + fp) if tp + fp else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else float("nan")
    print("POOLED", {"records": len(rows), "tp": tp, "fp": fp, "fn": fn, "sensitivity": sens,
                    "positive_predictivity": ppv, "f1": f1,
                    "mean_record_sensitivity": float(np.mean([r["sensitivity"] for r in rows])),
                    "mean_record_positive_predictivity": float(np.mean([r["positive_predictivity"] for r in rows])),
                    "mean_record_f1": float(np.mean([r["f1"] for r in rows]))})
    print("CSV", out)

if __name__ == "__main__": main()
