"""Run LOCALLY: build the Kaggle data files and the code zip, and verify the data.

  kaggle/neurocardio_kaggle.zip             code, splits, model config (no data, no .venv)
  kaggle/data/neurocardio_dev.npz           development records only  -> upload for stages 1-3
  kaggle/data/neurocardio_test_SEALED.npz   DS2 records               -> upload only for the one final evaluation
Windows are produced by the app's own preprocessing, so Kaggle does no signal processing.

Run: backend/.venv/bin/python backend/scripts/build_kaggle_package.py
"""
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import wfdb

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "backend/scripts"))
from app.core.config import PREPROCESSING_VERSION  # noqa: E402
from app.ml.preprocessing import extract_windows, filter_continuous  # noqa: E402
from train import CLASSES, RR_NAMES  # noqa: E402

CFG = json.loads((ROOT / "backend/models/preprocessing_config.json").read_text())
SPLITS = {n: json.loads((ROOT / f"backend/splits/{n}.json").read_text()) for n in ("interpatient_v1", "ds1_ds2")}
OUT = ROOT / "kaggle"


def rr_features(peaks: np.ndarray, fs: float = 360.0) -> np.ndarray:
    """RR context from beat POSITIONS only (no labels). Uses the following beat, so it is offline only."""
    rr = np.diff(peaks) / fs
    pre = np.concatenate([[rr[0]], rr])
    post = np.concatenate([rr, [rr[-1]]])
    med = np.median(rr)
    local = np.array([pre[max(0, i - 9):i + 1].mean() for i in range(len(pre))])
    return np.stack([pre, post, pre / med, post / med, pre / local], axis=1).astype(np.float32)


def build(records):
    parts = {k: [] for k in ("X", "y", "record", "subject", "sample_index", "rr")}
    for rec in records:
        r = wfdb.rdrecord(str(ROOT / "results/mitdb" / rec))
        signal = r.p_signal[:, r.sig_name.index("MLII")]
        ann = wfdb.rdann(str(ROOT / "results/mitdb" / rec), "atr")
        keep = [(int(p), CLASSES.index(CFG["aami_mapping"][s])) for p, s in zip(ann.sample, ann.symbol) if s in CFG["aami_mapping"]]
        peaks = np.array([p for p, _ in keep], np.int64)
        labels = np.array([c for _, c in keep], np.int8)
        batch = extract_windows(filter_continuous(signal), peaks)
        valid = np.isin(peaks, batch.sample_indices)
        n = int(valid.sum())
        parts["X"].append(batch.windows[..., 0])
        parts["y"].append(labels[valid])
        parts["record"].append(np.full(n, rec, dtype="<U3"))
        parts["subject"].append(np.full(n, CFG["split"]["subject_groups_merged"].get(rec, rec), dtype="<U3"))
        parts["sample_index"].append(peaks[valid])
        parts["rr"].append(rr_features(peaks)[valid])
    data = {k: np.concatenate(v) for k, v in parts.items()}
    data.update(class_names=np.array(CLASSES), rr_names=np.array(RR_NAMES), preprocessing_version=np.array(PREPROCESSING_VERSION))
    return data


def counts(d, records):
    return np.bincount(d["y"][np.isin(d["record"], records)], minlength=4).tolist()


def main():
    base, ds = SPLITS["interpatient_v1"], SPLITS["ds1_ds2"]
    dev_records = sorted(set(base["train"]) | set(base["validation"]))
    (OUT / "data").mkdir(parents=True, exist_ok=True)
    metrics = json.loads((ROOT / "backend/artifacts/metrics.json").read_text())

    dev = build(dev_records)
    for part in ("train", "validation"):
        want = [metrics["dataset"]["class_counts"][part]["usable_windows"][c] for c in CLASSES]
        assert counts(dev, base[part]) == want, (part, counts(dev, base[part]), want)
    assert not set(dev["record"]) & set(base["test"]), "a project test record is in the dev file"
    assert set(dev["record"]) & set(ds["test"]) == {"202"}, "only 202 (reproduction early-stop) may overlap DS2"
    print("dev file verified: window counts equal metrics.json; no project test record; only DS2 record present is 202")

    test = build(ds["test"])
    ref_path = ROOT / "results/test_examples_for_xai.npz"
    if not ref_path.exists():
        sys.exit(f"{ref_path} is needed for the deployed model's predictions in the sealed file")
    with np.load(ref_path) as z:                       # deployed model's saved test predictions, for the paired baseline
        key = {(str(r), int(p)): i for i, (r, p) in enumerate(zip(z["record"], z["position"]))}
        prob, ref_x, ref_y = z["prob"].astype(np.float32), z["X"].astype(np.float32), z["y"]
    baseline = np.full((len(test["y"]), 4), np.nan, np.float32)
    rows = np.array([key.get((str(r), int(p)), -1) for r, p in zip(test["record"], test["sample_index"])])
    has = rows >= 0
    assert has.sum() == len(ref_y) and np.array_equal(test["record"][~has], np.full((~has).sum(), "202"))
    baseline[has] = prob[rows[has]]
    diff = float(np.abs(test["X"][has] - ref_x[rows[has]].reshape(len(ref_y), 720)).max())
    assert diff < 1e-5 and np.array_equal(test["y"][has], ref_y[rows[has]])
    test["baseline_probs"] = baseline
    print(f"sealed file verified against the training run's saved test windows (max difference {diff:.1e})")

    files = [OUT / "data/neurocardio_dev.npz", OUT / "data/neurocardio_test_SEALED.npz"]
    np.savez_compressed(files[0], **dev)
    np.savez_compressed(files[1], **test)

    zpath = OUT / "neurocardio_kaggle.zip"
    members = [(p, p.relative_to(ROOT).as_posix()) for p in sorted((ROOT / "backend/scripts").glob("*.py"))]
    members += [(p, p.relative_to(ROOT).as_posix()) for p in sorted((ROOT / "backend/splits").glob("*.json"))]
    members += [(ROOT / p, p) for p in ("backend/models/preprocessing_config.json", "backend/requirements.txt",
                                        "kaggle/requirements-kaggle.txt", "kaggle/reproduction_reference.json",
                                        "kaggle/run_all.sh", "kaggle/stage1.sh", "kaggle/stage2.sh", "kaggle/stage3.sh",
                                        "kaggle/README.md", "docs/experiment-protocol.md")]
    members.append((ROOT / "backend/artifacts/experiments.json", "backend/artifacts/experiments_local.json"))
    private = str(Path.home())
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for src, name in members:
            text = src.read_text()
            assert private not in text, f"{name} contains a local home-directory path"
            z.writestr(name, text)
    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
    assert not any(".venv" in n or n.endswith((".npz", ".keras", ".sqlite3")) for n in names)

    print(f"\n{'file':45} {'size':>10}")
    for f in [zpath, *files]:
        print(f"{f.relative_to(ROOT).as_posix():45} {f.stat().st_size / 1e6:8.2f} MB")
    print(f"zip: {len(names)} files")
    for label, d in (("dev", dev), ("sealed test", test)):
        print(f"{label}: {len(d['y'])} windows, {len(set(d['record']))} records, arrays "
              + ", ".join(f"{k}{list(v.shape)}" for k, v in d.items() if v.ndim))


if __name__ == "__main__":
    main()
