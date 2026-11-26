"""Run LOCALLY after build_multidb.py: build the Kaggle code zip and check what is about to be uploaded.

  kaggle/neurocardio_kaggle.zip   code, split, protocol, data report, deployed model (for the final comparison)
No data goes in the zip; the three .npz files come from build_multidb.py.

Run: backend/.venv/bin/python backend/scripts/build_kaggle_package.py
"""
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend/scripts"))
from split_checks import check_arrays, check_split  # noqa: E402

OUT = ROOT / "kaggle"
DATA = {"multidb_dev.npz": ("train", "validation"), "multidb_test_ds2_SEALED.npz": ("test_ds2",),
        "multidb_test_incart_SEALED.npz": ("test_incart",)}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    split = json.loads((ROOT / "backend/splits/multidb_v1.json").read_text())
    check_split(split)
    for name, parts in DATA.items():                      # reads only the record/patient columns
        path = OUT / "data" / name
        assert sha256(path) == split["data_sha256"][name], f"{name} does not match the split; rerun build_multidb.py"
        with np.load(path, allow_pickle=False) as z:
            check_arrays(split, z["record"].astype(str), z["patient"].astype(str), parts)
        print(f"verified {name}: hash, records and patients")

    scripts = ["train.py", "train_multidb.py", "run_multidb.py", "evaluate_multidb.py", "final_test.py", "models_multidb.py",
               "split_checks.py", "analysis_common.py", "gpu_check.py", "experiments.py", "build_multidb.py"]
    members = [(ROOT / "backend/scripts" / s, f"backend/scripts/{s}") for s in scripts]
    members += [(ROOT / p, p) for p in (
        "backend/splits/multidb_v1.json", "backend/splits/ds1_ds2.json", "backend/models/preprocessing_config.json",
        "backend/models/neurocardio_final.keras", "backend/models/neurocardio_final.keras.sha256",
        "backend/artifacts/multidb_data_report.json", "kaggle/requirements-kaggle.txt", "kaggle/reproduction_reference.json",
        "kaggle/run_multidb.sh", "kaggle/README.md", "docs/multidb-protocol.md", "docs/multidb-data-report.md")]
    members.append((ROOT / "backend/artifacts/experiments.json", "backend/artifacts/experiments_local.json"))
    zpath, private = OUT / "neurocardio_kaggle.zip", str(Path.home()).encode()
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for src, name in members:
            raw = src.read_bytes()
            assert private not in raw, f"{name} contains a local home-directory path"
            z.writestr(name, raw)
    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
        assert z.testzip() is None
    assert not any(".venv" in n or n.endswith((".npz", ".sqlite3")) for n in names)

    print(f"\n{'file':52} {'size':>10}")
    for f in [zpath, *(OUT / "data" / n for n in DATA)]:
        print(f"{f.relative_to(ROOT).as_posix():52} {f.stat().st_size / 1e6:8.2f} MB")
    print(f"zip: {len(names)} files")


if __name__ == "__main__":
    main()
