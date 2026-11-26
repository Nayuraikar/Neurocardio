"""Run LOCALLY: download SVDB and INCART, build beat windows for three databases, make patient-wise splits.

  kaggle/data/multidb_dev.npz                  training + validation patients only   -> upload
  kaggle/data/multidb_test_ds2_SEALED.npz      MIT-BIH DS2                           -> upload only for the final evaluation
  kaggle/data/multidb_test_incart_SEALED.npz   held-out INCART patients              -> upload only for the final evaluation
  backend/splits/multidb_v1.json, backend/artifacts/multidb_data_report.json, docs/multidb-data-report.md

Every window comes from the deployed app's filter_continuous + extract_windows. Signals not at 360 Hz are first
resampled (polyphase FIR) and annotation positions rescaled. Windows stay centred on the database's annotation;
no re-alignment is done. The split uses record/patient label counts only, never a model output.

Run: backend/.venv/bin/python -u backend/scripts/build_multidb.py [--skip-download]
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

import numpy as np
import wfdb
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app.core.config import PREPROCESSING_VERSION  # noqa: E402
from app.ml.preprocessing import FS_HZ, extract_windows, filter_continuous  # noqa: E402
sys.path.insert(0, str(ROOT / "backend/scripts"))
from split_checks import PARTS, check_split  # noqa: E402

CFG = json.loads((ROOT / "backend/models/preprocessing_config.json").read_text())
DS = json.loads((ROOT / "backend/splits/ds1_ds2.json").read_text())
CLASSES = ["N", "S", "V", "F"]
RR_NAMES = ["pre_rr_s", "post_rr_s", "pre_over_record_median", "post_over_record_median", "pre_over_local_mean10"]
DIRS = {"mitdb": ROOT / "results/mitdb", "svdb": ROOT / "backend/data/external/svdb",
        "incartdb": ROOT / "backend/data/external/incartdb"}
LEAD = {"mitdb": "MLII", "svdb": "ECG1", "incartdb": "II"}
BEAT_SYMBOLS = set("NLRBAaJSVrFejnE/fQ?")
PACED_SYMBOLS = set("/f")
MITDB_PACED = DS["excluded"]["paced"]
SAME_SUBJECT_AS_DS2 = {"201": "same subject as DS2 record 202"}
MITDB_VAL = ["108", "114", "124", "205", "223"]          # the earlier inner validation records, without 201
MAX_UNMAPPED, MIN_BEATS = 0.10, 100                       # "unusable annotations" rule
SEED, DRAWS = 20241127, 40000
OUT = ROOT / "kaggle/data"


# ------------------------------------------------------------------ download
def fetch(url: str, dst: Path, attempts: int = 5) -> None:
    for i in range(attempts):
        try:
            tmp = dst.with_name(dst.name + ".part")
            with urllib.request.urlopen(url, timeout=300) as r, tmp.open("wb") as f:
                while block := r.read(1 << 20):
                    f.write(block)
            tmp.replace(dst)
            return
        except OSError:
            if i == attempts - 1:
                raise
            time.sleep(3 * (i + 1))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(db: str, skip: bool) -> dict:
    """Fetch .hea/.dat/.atr with wfdb's PhysioNet layout, then verify against PhysioNet's SHA256SUMS."""
    d, base = DIRS[db], f"https://physionet.org/files/{db}/1.0.0/"
    d.mkdir(parents=True, exist_ok=True)
    if not (d / "SHA256SUMS.txt").exists():
        fetch(base + "SHA256SUMS.txt", d / "SHA256SUMS.txt")
    sums = {name: digest for digest, name in (ln.split() for ln in (d / "SHA256SUMS.txt").read_text().splitlines() if ln.strip())}
    records = wfdb.get_record_list(db) if not skip else (d / "RECORDS").read_text().split()
    if not skip:
        (d / "RECORDS").write_text("\n".join(records) + "\n")
    names = [f"{r}.{e}" for r in records for e in ("hea", "atr", "dat")]
    if db == "incartdb":
        names.append("files-patients-diagnoses.txt")
    todo = [n for n in names if not (d / n).exists()]
    if todo and skip:
        sys.exit(f"{db}: missing files {todo[:5]} (run without --skip-download)")
    if todo:
        print(f"{db}: downloading {len(todo)} files", flush=True)
        with ThreadPoolExecutor(4) as pool:
            for i, _ in enumerate(pool.map(lambda n: fetch(base + n, d / n), todo), 1):
                if i % 25 == 0 or i == len(todo):
                    print(f"  {db} {i}/{len(todo)}", flush=True)
    bad = [n for n in names if n in sums and sha256(d / n) != sums[n]]
    if bad:
        sys.exit(f"{db}: checksum mismatch for {bad[:5]}; delete those files and rerun")
    return {"records": len(records), "files_checksum_verified": sum(n in sums for n in names)}


def record_list(db: str) -> list[str]:
    if db == "mitdb":
        return sorted(p.stem for p in DIRS[db].glob("*.hea"))
    return (DIRS[db] / "RECORDS").read_text().split()


def patient_of(db: str, rec: str) -> str:
    if db == "mitdb":
        return "mitdb:201" if rec == "202" else f"mitdb:{rec}"
    if db == "incartdb":
        m = re.search(r"patient\s+(\d+)", " ".join(wfdb.rdheader(str(DIRS[db] / rec)).comments))
        return f"incartdb:p{int(m.group(1)):02d}"
    return f"svdb:{rec}"            # SVDB publishes no record-to-subject map: one record = one subject is ASSUMED


# ------------------------------------------------------------------ per-record processing
def rr_features(peaks: np.ndarray, fs: float = FS_HZ) -> np.ndarray:
    """RR context from beat POSITIONS only (no labels). Uses the following beat and the record median: offline only."""
    rr = np.diff(peaks) / fs
    pre = np.concatenate([[rr[0]], rr])
    post = np.concatenate([rr, [rr[-1]]])
    med = np.median(rr)
    local = np.array([pre[max(0, i - 9):i + 1].mean() for i in range(len(pre))])
    return np.stack([pre, post, pre / med, post / med, pre / local], axis=1).astype(np.float32)


def annotated_beats(db: str, rec: str):
    ann = wfdb.rdann(str(DIRS[db] / rec), "atr")
    return np.asarray(ann.sample, np.int64), list(ann.symbol)


def screen(db: str, rec: str) -> tuple[str | None, dict]:
    """-> (exclusion reason or None, annotation statistics)."""
    _, symbols = annotated_beats(db, rec)
    beats = collections.Counter(s for s in symbols if s in BEAT_SYMBOLS)
    total = sum(beats.values())
    mapped = sum(v for s, v in beats.items() if s in CFG["aami_mapping"])
    stats = {"beat_annotations": total, "mapped_beats": mapped, "unmapped": {s: v for s, v in beats.items() if s not in CFG["aami_mapping"]},
             "paced_beats": sum(beats[s] for s in PACED_SYMBOLS),
             "class_counts": [sum(v for s, v in beats.items() if CFG["aami_mapping"].get(s) == c) for c in CLASSES]}
    if db == "mitdb" and rec in MITDB_PACED:
        return "paced record (AAMI practice)", stats
    if stats["paced_beats"] > 0.01 * max(total, 1):
        return f"paced beats ({stats['paced_beats']} of {total})", stats
    if LEAD[db] not in wfdb.rdheader(str(DIRS[db] / rec)).sig_name:
        return f"lead {LEAD[db]} missing", stats
    if mapped < MIN_BEATS:
        return f"unusable annotations: only {mapped} mapped beats", stats
    if 1 - mapped / total > MAX_UNMAPPED:
        return f"unusable annotations: {100 * (1 - mapped / total):.1f}% of beats have no N/S/V/F label", stats
    return None, stats


def process(db: str, rec: str) -> dict:
    r = wfdb.rdrecord(str(DIRS[db] / rec), channel_names=[LEAD[db]])
    signal = r.p_signal[:, 0].astype(np.float64)
    nonfinite = int((~np.isfinite(signal)).sum())
    if nonfinite:
        assert nonfinite < 0.01 * len(signal), f"{db}/{rec}: {nonfinite} invalid samples"
        good = np.isfinite(signal)
        signal = np.interp(np.arange(len(signal)), np.flatnonzero(good), signal[good])
    fs = float(r.fs)
    if fs != FS_HZ:
        ratio = Fraction(int(FS_HZ), int(fs))
        assert float(fs).is_integer()
        signal = resample_poly(signal, ratio.numerator, ratio.denominator)
    samples, symbols = annotated_beats(db, rec)
    keep = [(int(round(int(p) * FS_HZ / fs)), CLASSES.index(CFG["aami_mapping"][s])) for p, s in zip(samples, symbols)
            if s in CFG["aami_mapping"]]
    peaks = np.array([p for p, _ in keep], np.int64)
    labels = np.array([c for _, c in keep], np.int8)
    assert (np.diff(peaks) > 0).all(), f"{db}/{rec}: beat positions are not strictly increasing"
    filtered = filter_continuous(signal)
    batch = extract_windows(filtered, peaks)
    valid = np.isin(peaks, batch.sample_indices)
    # diagnostic only: distance from the annotation to the largest deflection within +/-100 ms (windows are NOT moved)
    offsets = np.array([int(np.abs(filtered[p - 36:p + 37]).argmax()) - 36 for p in batch.sample_indices
                        if 36 <= p < len(filtered) - 36])
    return {"X": batch.windows[..., 0], "y": labels[valid], "rr": rr_features(peaks)[valid], "sample_index": peaks[valid],
            "edge_rejected": int((~valid).sum()), "offsets": offsets, "source_fs": fs, "nonfinite_samples": nonfinite}


# ------------------------------------------------------------------ splits
def search_split(units: list[str], counts: np.ndarray, sizes: list[int], targets: list[float], minimums: list[tuple[int, int, int]],
                 seed: int) -> list[list[str]]:
    """Random search over partitions of `units` (patients). counts: (units, 4) beat counts.
    minimums: (class index, beats a patient needs to count, patients required in every part)."""
    assert sum(sizes) == len(units)
    rng = np.random.default_rng(seed)
    base = np.repeat(np.arange(len(sizes)), sizes)
    total = counts.sum(0)
    best = None
    for _ in range(DRAWS):
        part = base[rng.permutation(len(units))]
        if any((counts[part == k, c] >= beats).sum() < need for c, beats, need in minimums for k in range(len(sizes))):
            continue
        share = np.array([counts[part == k].sum(0) for k in range(len(sizes))]) / total
        score = float(((share - np.array(targets)[:, None]) ** 2).sum())
        if best is None or score < best[0]:
            best = (score, part)
    assert best, "no partition satisfies the constraints"
    return [sorted(u for u, p in zip(units, best[1]) if p == k) for k in range(len(sizes))]


def make_split(usable: dict, stats: dict) -> dict:
    def patient_counts(db):
        by = collections.defaultdict(lambda: np.zeros(4, int))
        for rec in usable[db]:
            by[patient_of(db, rec)] += np.array(stats[db][rec]["class_counts"])
        names = sorted(by)
        return names, np.stack([by[n] for n in names])

    sv_names, sv_counts = patient_counts("svdb")
    n_val = round(0.2 * len(sv_names))
    sv_train, sv_val = search_split(sv_names, sv_counts, [len(sv_names) - n_val, n_val], [0.8, 0.2],
                                    [(1, 20, 5), (3, 1, 2)], SEED)
    ic_names, ic_counts = patient_counts("incartdb")
    ic_train, ic_val, ic_test = search_split(ic_names, ic_counts, [15, 7, len(ic_names) - 22], [0.45, 0.20, 0.35],
                                             [(1, 5, 3), (3, 5, 1)], SEED + 1)

    def records(db, patients):
        return [f"{db}/{r}" for r in usable[db] if patient_of(db, r) in set(patients)]

    ds1 = sorted(set(DS["train"]) | set(DS["validation"]))
    mit_train = [f"mitdb/{r}" for r in ds1 if r not in MITDB_VAL and r not in SAME_SUBJECT_AS_DS2]
    split = {
        "name": "multidb_v1", "seed": SEED, "draws": DRAWS,
        "description": "Patient-wise. train/validation: MIT-BIH DS1, SVDB, INCART. test_ds2: MIT-BIH DS2. "
                       "test_incart: INCART patients unseen in training (the INCART database itself IS in training).",
        "criterion": "squared deviation of per-class patient shares from target shares; label counts only",
        "train": mit_train + records("svdb", sv_train) + records("incartdb", ic_train),
        "validation": [f"mitdb/{r}" for r in MITDB_VAL] + records("svdb", sv_val) + records("incartdb", ic_val),
        "test_ds2": [f"mitdb/{r}" for r in DS["test"]],
        "test_incart": records("incartdb", ic_test),
        "excluded_from_train_validation": {f"mitdb/{r}": why for r, why in SAME_SUBJECT_AS_DS2.items()},
    }
    split["patients"] = {r: patient_of(*r.split("/")) for part in ("train", "validation", "test_ds2", "test_incart") for r in split[part]}
    check_split(split)
    return split


# ------------------------------------------------------------------ main
def stack(split_records: list[str], cache: dict, split: dict) -> dict:
    parts = collections.defaultdict(list)
    for key in split_records:
        c, n = cache[key], len(cache[key]["y"])
        for k in ("X", "y", "rr", "sample_index"):
            parts[k].append(c[k])
        parts["record"].append(np.full(n, key, dtype="<U16"))
        parts["patient"].append(np.full(n, split["patients"][key], dtype="<U16"))
        parts["db"].append(np.full(n, key.split("/")[0], dtype="<U8"))
    data = {k: np.concatenate(v) for k, v in parts.items()}
    data.update(class_names=np.array(CLASSES), rr_names=np.array(RR_NAMES), preprocessing_version=np.array(PREPROCESSING_VERSION))
    return data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()
    source = {db: download(db, args.skip_download) for db in ("svdb", "incartdb")}

    usable, excluded, stats = {}, {}, {}
    for db in DIRS:
        usable[db], excluded[db], stats[db] = [], {}, {}
        for rec in record_list(db):
            reason, st = screen(db, rec)
            stats[db][rec] = st
            if reason:
                excluded[db][rec] = reason
            else:
                usable[db].append(rec)
        print(f"{db}: {len(usable[db])} usable records, {len(excluded[db])} excluded {excluded[db]}", flush=True)

    # cross-check the INCART header patient numbers against PhysioNet's text file
    listed, p = {}, None
    for line in (DIRS["incartdb"] / "files-patients-diagnoses.txt").read_text().splitlines():
        if m := re.fullmatch(r"patient\s+(\d+)", line.strip()):
            p = int(m.group(1))
        elif re.fullmatch(r"(I\d\d\s*)+", line.strip()):
            listed.update({r: f"incartdb:p{p:02d}" for r in line.split()})
    assert all(patient_of("incartdb", r) == listed[r] for r in record_list("incartdb")), "INCART patient maps disagree"

    split = make_split(usable, stats)
    cache = {}
    wanted = [r for part in PARTS for r in split[part]]
    for i, key in enumerate(wanted, 1):
        cache[key] = process(*key.split("/"))
        if i % 20 == 0 or i == len(wanted):
            print(f"processed {i}/{len(wanted)} records", flush=True)

    # ---- report
    def summary(keys):
        y = np.concatenate([cache[k]["y"] for k in keys])
        return {"records": len(keys), "patients": len({split["patients"][k] for k in keys}),
                "windows": dict(zip(CLASSES, np.bincount(y, minlength=4).tolist())), "total": int(len(y))}

    def concentration(keys):
        out = {}
        for i, c in ((1, "S"), (3, "F")):
            by = collections.Counter()
            for k in keys:
                by[split["patients"][k]] += int((cache[k]["y"] == i).sum())
            total = sum(by.values())
            top, n = by.most_common(1)[0]
            out[c] = {"total": total, "patients_with_any": sum(v > 0 for v in by.values()), "top_patient": top,
                      "top_share": n / total if total else None}
        return out

    report = {"preprocessing_version": PREPROCESSING_VERSION, "source": source, "lead": LEAD, "databases": {}, "splits": {},
              "excluded_records": excluded,
              "rules": {"paced": "MIT-BIH 102/104/107/217; any record with > 1% paced-beat symbols (/ f)",
                        "unusable": f"fewer than {MIN_BEATS} mapped beats, or > {MAX_UNMAPPED:.0%} of beat annotations without an N/S/V/F mapping",
                        "same_subject": split["excluded_from_train_validation"]}}
    for db in DIRS:
        keys = [k for k in cache if k.startswith(db + "/")]
        off = np.concatenate([cache[k]["offsets"] for k in keys]) * 1000 / FS_HZ
        unmapped = collections.Counter()
        for rec in usable[db]:
            unmapped.update(stats[db][rec]["unmapped"])
        report["databases"][db] = {
            **summary(keys), "source_fs_hz": cache[keys[0]]["source_fs"],
            "dropped_beat_symbols_in_usable_records": dict(unmapped),
            "edge_rejected_beats": sum(cache[k]["edge_rejected"] for k in keys),
            "paced_beat_annotations_in_usable_records": sum(stats[db][r]["paced_beats"] for r in usable[db]),
            "interpolated_invalid_samples": sum(cache[k]["nonfinite_samples"] for k in keys),
            "annotation_to_peak_ms": {"median_signed": float(np.median(off)), "median_abs": float(np.median(np.abs(off))),
                                      "p90_abs": float(np.percentile(np.abs(off), 90)),
                                      "fraction_over_42ms": float((np.abs(off) > 41.7).mean())}}
    for part in PARTS:
        report["splits"][part] = {**summary(split[part]), "concentration": concentration(split[part]),
                                  "by_database": {db: summary(ks) for db in DIRS
                                                  if (ks := [k for k in split[part] if k.startswith(db + "/")])}}

    OUT.mkdir(parents=True, exist_ok=True)
    dev_keys = split["train"] + split["validation"]
    files = {"multidb_dev.npz": dev_keys, "multidb_test_ds2_SEALED.npz": split["test_ds2"],
             "multidb_test_incart_SEALED.npz": split["test_incart"]}
    hashes = {name: {hashlib.md5(w.tobytes()).digest() for k in keys for w in cache[k]["X"]} for name, keys in files.items()}
    names = list(files)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not hashes[a] & hashes[b], f"identical windows in {a} and {b}"
    del hashes

    report["files"] = {}
    for name, keys in files.items():
        print(f"writing {name}", flush=True)
        data = stack(keys, cache, split)
        if name == "multidb_dev.npz":
            test_people = {split["patients"][r] for r in split["test_ds2"] + split["test_incart"]}
            assert not set(data["patient"].tolist()) & test_people, "a test patient is in the dev file"
        np.savez_compressed(OUT / name, **data)
        report["files"][name] = {"windows": int(len(data["y"])), "bytes": (OUT / name).stat().st_size, "sha256": sha256(OUT / name)}
        del data
    split["data_sha256"] = {n: report["files"][n]["sha256"] for n in files}
    (ROOT / "backend/splits/multidb_v1.json").write_text(json.dumps(split, indent=1))
    (ROOT / "backend/artifacts/multidb_data_report.json").write_text(json.dumps(report, indent=1))
    write_markdown(report, split)


def write_markdown(rep: dict, split: dict) -> None:
    row = lambda name, s: f"| {name} | {s['records']} | {s['patients']} | " + " | ".join(f"{s['windows'][c]:,}" for c in CLASSES) + f" | {s['total']:,} |"
    head = ["| | records | patients | N | S | V | F | total |", "|---|---|---|---|---|---|---|---|"]
    L = ["# Multi-database data report", "", "Generated by `backend/scripts/build_multidb.py`. Counts are usable windows.",
         "", "## Per database (usable records)", *head, *[row(d, s) for d, s in rep["databases"].items()],
         "", "## Per split", *head, *[row(p, s) for p, s in rep["splits"].items()],
         "", "## Per split and database", *head,
         *[row(f"{p} / {d}", s) for p, e in rep["splits"].items() for d, s in e["by_database"].items()],
         "", "## S and F concentration", "| split | class | beats | patients with any | largest patient | its share |", "|---|---|---|---|---|---|"]
    for p, e in rep["splits"].items():
        for c, v in e["concentration"].items():
            L.append(f"| {p} | {c} | {v['total']} | {v['patients_with_any']} | {v['top_patient']} | {v['top_share']:.1%} |")
    L += ["", "## Excluded records", ""]
    for db, ex in rep["excluded_records"].items():
        L += [f"- **{db}**: " + (", ".join(f"{r} ({why})" for r, why in ex.items()) if ex else "none met an exclusion rule")]
    L += [f"- **Train/validation only**: " + ", ".join(f"{r} ({why})" for r, why in rep["rules"]["same_subject"].items()),
          "", f"Paced rule: {rep['rules']['paced']}.", f"Unusable-annotation rule: {rep['rules']['unusable']}.",
          "", "## Signal details", "| database | lead | source Hz | dropped beat symbols | edge-rejected | paced symbols | interpolated samples |",
          "|---|---|---|---|---|---|---|"]
    for db, s in rep["databases"].items():
        L.append(f"| {db} | {rep['lead'][db]} | {s['source_fs_hz']:g} | {s['dropped_beat_symbols_in_usable_records'] or 'none'} | "
                 f"{s['edge_rejected_beats']} | {s['paced_beat_annotations_in_usable_records']} | {s['interpolated_invalid_samples']} |")
    L += ["", "## Annotation position (diagnostic; windows are not moved)",
          "Offset from the annotation to the largest absolute deflection within ±100 ms, in the filtered 360 Hz signal. "
          "The largest deflection is not always the R peak, so compare databases rather than reading absolute values.",
          "| database | median signed ms | median abs ms | 90th pct abs ms | share > 42 ms (augmentation shift) |", "|---|---|---|---|---|"]
    for db, s in rep["databases"].items():
        o = s["annotation_to_peak_ms"]
        L.append(f"| {db} | {o['median_signed']:+.1f} | {o['median_abs']:.1f} | {o['p90_abs']:.1f} | {o['fraction_over_42ms']:.1%} |")
    L += ["", "## What cannot be verified",
          "- SVDB publishes no record-to-subject map. Each record is treated as one subject.",
          "- SVDB names its channels ECG1/ECG2 only. ECG1 is used; its electrode placement is not documented in the headers.",
          "- No subject map links SVDB to MIT-BIH, so cross-database subject overlap cannot be checked.",
          "- INCART annotations were made by an algorithm and corrected by hand; PhysioNet says positions were not manually corrected."]
    (ROOT / "docs/multidb-data-report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


if __name__ == "__main__":
    main()
