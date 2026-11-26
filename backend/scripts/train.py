"""Reproducible training for the NeuroCardio beat classifier (local or Kaggle).

THIS IS A RECONSTRUCTION. The original training notebook is not in the repository.
Where each setting comes from:

  VERIFIED from saved artifacts
    architecture, initialisers           backend/models/neurocardio_final.keras (config.json)
    Adam(1e-3), sparse cross-entropy     compile_config in the .keras archives
    batch size 64                        optimizer step counts: 630/epoch (augmented), 602 (not)
    EarlyStopping patience 8 on val macro-F1, best weights restored
                                         all 9 histories end 8 epochs after the best; saved model
                                         reproduces the best-epoch validation loss exactly
    ReduceLROnPlateau(val_macro_f1, max, factor 0.5, patience 4)   reproduces 9/9 LR schedules
    validation loss is unweighted
    preprocessing, split, augmentation parameter ranges and copy counts   preprocessing_config.json
  ASSUMED (not recoverable)
    sqrt class weight w_c = sqrt(N / (K * n_c)) with n_c = pre-augmentation training counts
    each extra copy applies all four augmentations, in config order, without re-z-scoring;
    the original window is kept unchanged
    full reshuffle each epoch; seed 42 as the first seed

Data comes from a prepared .npz (build_kaggle_package.py). This script never opens the sealed test
file, and rejects a split that places one of its own test records in a development part.

Run: python -u train.py --variant baseline --seed 42 [--data FILE] [--splits DIR] [--out DIR]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import signal
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CODE = HERE.parent                                   # backend/ in the repo and in the Kaggle zip
CLASSES = ["N", "S", "V", "F"]
BATCH, LR, MAX_EPOCHS, STOP_PATIENCE, LR_PATIENCE, LR_FACTOR = 64, 1e-3, 100, 8, 4, 0.5
DEV_FILE = "neurocardio_dev.npz"
KAGGLE_INPUT, KAGGLE_WORKING = Path("/kaggle/input"), Path("/kaggle/working")

# every variant is a delta from the reconstructed baseline
VARIANTS = {
    "baseline":         {"loss": "sqrt_weighted_ce",     "sampler": "shuffle",  "context": "none"},
    "focal_g2":         {"loss": "focal_g2",             "sampler": "shuffle",  "context": "none"},
    "cb_focal":         {"loss": "class_balanced_focal", "sampler": "shuffle",  "context": "none"},
    "balanced_sampler": {"loss": "ce",                   "sampler": "balanced", "context": "none"},  # no loss weights: no double correction
    "rr_context":       {"loss": "sqrt_weighted_ce",     "sampler": "shuffle",  "context": "rr"},
}
RR_NAMES = ["pre_rr_s", "post_rr_s", "pre_over_record_median", "post_over_record_median", "pre_over_local_mean10"]


# ------------------------------------------------------------------ paths
def find_data(name: str = DEV_FILE) -> Path:
    if KAGGLE_INPUT.exists():
        hits = sorted(KAGGLE_INPUT.rglob(name))
        if hits:
            return hits[0]
    return CODE.parent / "kaggle/data" / name


def default_out() -> Path:
    return KAGGLE_WORKING if KAGGLE_WORKING.exists() else CODE / "experiments/kaggle_out"


class Env:
    """All file locations and run-wide switches."""

    def __init__(self, data=None, splits=None, out=None, model_config=None, allow_cpu=False):
        self.data = Path(data) if data else find_data()
        self.splits = Path(splits) if splits else CODE / "splits"
        self.out = Path(out) if out else default_out()
        self.model_config = Path(model_config) if model_config else CODE / "models/preprocessing_config.json"
        self.allow_cpu = allow_cpu or os.environ.get("NC_ALLOW_CPU") == "1"
        self.runs = self.out / "runs"
        self.registry = self.out / "experiments.json"
        self.settings = self.out / "settings.json"
        self._data = None

    @property
    def cfg(self) -> dict:
        return json.loads(self.model_config.read_text())

    def dataset(self) -> dict:
        if self._data is None:
            if not self.data.exists():
                sys.exit(f"data file not found: {self.data}  (pass --data or attach the Kaggle dataset)")
            with np.load(self.data, allow_pickle=False) as z:
                self._data = {k: z[k] for k in z.files}
            self._data["sha256"] = sha256(self.data)
        return self._data


def add_path_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--data", help=f"path to {DEV_FILE} (default: search /kaggle/input, then kaggle/data)")
    ap.add_argument("--splits", help="directory of split JSON files")
    ap.add_argument("--out", help="output directory (default: /kaggle/working, else backend/experiments/kaggle_out)")
    ap.add_argument("--model-config", help="preprocessing_config.json")
    ap.add_argument("--allow-cpu", action="store_true", help="permit running without a GPU (smoke tests only)")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)                            # atomic: a killed session never leaves half a file


# ------------------------------------------------------------------ splits and data
def load_split(env: Env, name: str) -> dict:
    split = json.loads((env.splits / f"{name}.json").read_text())
    groups = split.get("subject_groups", {})
    held_out = set(split["test"])
    development = split["train"] + split["validation"] + split.get("evaluation", [])
    assert not held_out & set(development), "a test record is in a development part"
    subj = lambda recs: {groups.get(r, r) for r in recs}
    assert not subj(split["train"]) & subj(split["validation"]), "train/validation subject overlap"
    assert not subj(held_out) & subj(development), "test/development subject overlap"
    if "evaluation" in split:
        assert not subj(split["evaluation"]) & subj(split["train"] + split["validation"]), "evaluation fold overlaps train/early-stop"
    return split


def take(data: dict, records: list[str]):
    """-> windows (n,720,1), labels, record ids, RR features, in the order of `records`."""
    rec = data["record"].astype(str)
    missing = sorted(set(records) - set(rec))
    if missing:
        sys.exit(f"records {missing} are not in the data file")
    idx = np.concatenate([np.flatnonzero(rec == r) for r in records])
    return data["X"][idx][..., None], data["y"][idx].astype(np.int64), rec[idx], data["rr"][idx]


def augment(cfg, X, y, rr, seed: int):
    """Training data only."""
    a = cfg["augmentation_training_only"]
    rng = np.random.default_rng(seed)
    t = np.arange(X.shape[1]) / cfg["sampling_frequency"]
    new_x, new_y, new_rr = [X], [y], [rr]
    for name, copies in a["copies_per_class_incl_original"].items():
        idx = np.flatnonzero(y == CLASSES.index(name))
        for _ in range(copies - 1):
            w = X[idx, :, 0].astype(np.float64)
            n = len(idx)
            w *= rng.uniform(*a["amplitude_scale_uniform"], (n, 1))
            w += rng.normal(0, a["gaussian_noise_sd_on_zscored_window"], w.shape)
            amp = rng.uniform(*a["baseline_wander_amplitude_uniform"], (n, 1))
            freq = rng.uniform(*a["baseline_wander_freq_hz_uniform"], (n, 1))
            phase = rng.uniform(0, 2 * np.pi, (n, 1))
            w += amp * np.sin(2 * np.pi * freq * t[None] + phase)
            lo, hi = a["time_shift_samples_uniform_int"]
            for i, s in enumerate(rng.integers(lo, hi + 1, n)):
                w[i] = np.roll(w[i], int(s))
            new_x.append(w[..., None].astype(np.float32))
            new_y.append(y[idx])
            new_rr.append(rr[idx])                                   # copied from the source beat
    return np.concatenate(new_x), np.concatenate(new_y), np.concatenate(new_rr)


# ------------------------------------------------------------------ model and loss
def build_model(context: str):
    import keras
    from keras import layers
    inp = keras.Input((720, 1), name="ecg_input")
    h = layers.Conv1D(32, 7, padding="same", activation="relu", name="conv1")(inp)
    h = layers.MaxPooling1D(2)(layers.BatchNormalization()(h))
    h = layers.Conv1D(64, 5, padding="same", activation="relu", name="conv2")(h)
    h = layers.MaxPooling1D(2)(layers.BatchNormalization()(h))
    h = layers.Conv1D(128, 3, padding="same", activation="relu", name="conv3")(h)
    h = layers.Bidirectional(layers.LSTM(64), name="bilstm")(h)       # default args: cuDNN is used on GPU
    h = layers.Dropout(0.4)(h)
    inputs = [inp]
    if context == "rr":
        rr_in = keras.Input((len(RR_NAMES),), name="rr_input")
        h = layers.Concatenate()([h, rr_in])
        inputs.append(rr_in)
    h = layers.Dense(64, activation="relu")(h)
    logits = layers.Dense(4, name="logits")(h)
    return keras.Model(inputs if len(inputs) > 1 else inp, layers.Softmax(name="probs")(logits), name="cnn_bilstm")


def class_weights(kind: str, counts: np.ndarray) -> np.ndarray:
    if kind == "sqrt_weighted_ce":
        return np.sqrt(counts.sum() / (len(counts) * counts))
    if kind == "class_balanced_focal":                       # Cui et al. 2019, beta = 0.999
        w = (1 - 0.999) / (1 - np.power(0.999, counts))
        return w * len(counts) / w.sum()
    return np.ones(len(counts))


def make_loss(kind: str):
    import keras
    if "focal" not in kind:
        return "sparse_categorical_crossentropy"

    def focal(y_true, p):
        y_true = keras.ops.cast(keras.ops.reshape(y_true, (-1,)), "int32")
        pt = keras.ops.sum(p * keras.ops.one_hot(y_true, 4), axis=1)
        pt = keras.ops.clip(pt, 1e-7, 1.0)
        return -keras.ops.power(1 - pt, 2.0) * keras.ops.log(pt)
    return focal


def macro_f1(y, pred) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(y, pred, labels=range(4), average="macro", zero_division=0))


def class_report(y, pred) -> dict:
    from sklearn.metrics import precision_recall_fscore_support
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=range(4), zero_division=0)
    return {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])} for i, c in enumerate(CLASSES)}


# ------------------------------------------------------------------ experiment registry
def read_registry(env: Env) -> dict:
    return json.loads(env.registry.read_text()) if env.registry.exists() else {"runs": []}


def register(env: Env, entry: dict, new_attempt: bool = False) -> None:
    """Insert or update one run, immediately. Earlier attempts (failed, interrupted) are kept."""
    data = read_registry(env)
    old = next((r for r in data["runs"] if r["run_id"] == entry["run_id"]), None)
    attempts = list(old.get("previous_attempts", [])) if old else []
    if old and new_attempt:
        prior = {k: v for k, v in old.items() if k != "previous_attempts"}
        if prior.get("status") == "running":                 # the session died mid-run
            prior["status"] = "interrupted"
        attempts.append(prior)
    entry = {**entry, **({"previous_attempts": attempts} if attempts else {})}
    data["runs"] = [r for r in data["runs"] if r["run_id"] != entry["run_id"]] + [entry]
    write_json(env.registry, data)


def run_id_for(split_name: str, variant: str, seed: int, tag: str = "") -> str:
    return f"{split_name}__{variant}__seed{seed}{tag}"


def finished_run(env: Env, run_id: str, fingerprint: dict) -> dict | None:
    """A run is skipped only if it completed with the same settings and its predictions exist."""
    for r in read_registry(env)["runs"]:
        if r["run_id"] == run_id and r["status"] == "completed" and r.get("fingerprint") == fingerprint:
            if (env.runs / run_id / "validation_predictions.npz").exists() and \
               (not r.get("has_evaluation_fold") or (env.runs / run_id / "evaluation_predictions.npz").exists()):
                return r
    return None


# ------------------------------------------------------------------ device
def setup_tf(env: Env) -> dict:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")       # one GPU, also on dual-T4 sessions
    import keras
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except RuntimeError:
            pass
    if not gpus and not env.allow_cpu:
        sys.exit("No GPU visible to TensorFlow. Enable a GPU accelerator, or pass --allow-cpu for a smoke test.")
    keras.mixed_precision.set_global_policy("float32")       # mixed precision stays OFF
    settings = json.loads(env.settings.read_text()) if env.settings.exists() else {}
    deterministic = bool(settings.get("deterministic_ops", True))     # decided once by gpu_check.py
    if deterministic:
        tf.config.experimental.enable_op_determinism()
    names = [tf.config.experimental.get_device_details(g).get("device_name", "GPU") for g in gpus]
    return {"gpu": bool(gpus), "gpu_names": names, "deterministic_ops": deterministic,
            "precision": keras.mixed_precision.global_policy().name, "python": platform.python_version(),
            "tensorflow": tf.__version__, "keras": keras.__version__}


class _Terminated(KeyboardInterrupt):
    pass


def _on_sigterm(signum, frame):
    raise _Terminated(f"received signal {signum}")


# ------------------------------------------------------------------ training
def train(env: Env, variant: str, split_name: str, seed: int, spec: dict | None = None, max_epochs: int = MAX_EPOCHS,
          tag: str = "", limit: int = 0) -> dict:
    if limit and not tag:
        raise ValueError("--limit is only for smoke runs; give them a --tag")
    spec = spec or VARIANTS[variant]
    run_id = run_id_for(split_name, variant, seed, tag)
    split = load_split(env, split_name)
    data = env.dataset()
    split_sha = hashlib.sha256((env.splits / f"{split_name}.json").read_bytes()).hexdigest()
    fingerprint = {"spec": spec, "split_sha256": split_sha, "data_sha256": data["sha256"], "seed": seed,
                   "max_epochs": max_epochs, "limit": limit, "batch_size": BATCH}
    done = finished_run(env, run_id, fingerprint)
    if done:
        print(f"[{run_id}] already completed; skipping", flush=True)
        return done

    signal.signal(signal.SIGTERM, _on_sigterm)
    out = env.runs / run_id
    out.mkdir(parents=True, exist_ok=True)
    entry = {"run_id": run_id, "kind": "training", "variant": variant, "spec": spec, "split": split_name, "seed": seed,
             "status": "running", "started_at": now(), "fingerprint": fingerprint, "script_sha256": sha256(Path(__file__))}
    if tag:
        entry["note"] = "smoke run; not a result"
    register(env, entry, new_attempt=True)
    try:
        device = setup_tf(env)
        import keras
        keras.backend.clear_session()
        keras.utils.set_random_seed(seed)
        Xtr, ytr, _, rrtr = take(data, split["train"])
        Xva, yva, rva, rrva = take(data, split["validation"])
        counts = np.bincount(ytr, minlength=4)                       # training only, before augmentation
        Xtr, ytr, rrtr = augment(env.cfg, Xtr, ytr, rrtr, seed)
        if limit:                                                    # smoke only: random subset
            pick = np.random.default_rng(0).choice(len(ytr), min(limit, len(ytr)), replace=False)
            Xtr, ytr, rrtr = Xtr[pick], ytr[pick], rrtr[pick]
        mean, std = rrtr.mean(0), rrtr.std(0) + 1e-8                 # training only
        rrtr, rrva = (rrtr - mean) / std, (rrva - mean) / std
        weights = class_weights(spec["loss"], counts.astype(np.float64))
        model = build_model(spec["context"])
        if spec["context"] == "none":
            assert model.count_params() == 142980, model.count_params()
        model.compile(optimizer=keras.optimizers.Adam(LR), loss=make_loss(spec["loss"]), metrics=["accuracy"])
        use_weights = spec["loss"] in ("sqrt_weighted_ce", "class_balanced_focal")
        config = {**entry, "batch_size": BATCH, "learning_rate": LR, "early_stop_patience": STOP_PATIENCE,
                  "lr_plateau": {"monitor": "val_macro_f1", "patience": LR_PATIENCE, "factor": LR_FACTOR},
                  "train_records": split["train"], "validation_records": split["validation"],
                  "evaluation_records": split.get("evaluation"),
                  "train_counts_pre_augmentation": counts.tolist(),
                  "train_counts_post_augmentation": np.bincount(ytr, minlength=4).tolist(),
                  "class_weights": weights.tolist(), "class_weights_applied": use_weights,
                  "augmentation": env.cfg["augmentation_training_only"], "parameter_count": int(model.count_params()),
                  "rr_features": RR_NAMES if spec["context"] == "rr" else None,
                  "rr_mean": mean.tolist(), "rr_std": std.tolist(), "environment": device}
        write_json(out / "config.json", config)

        def inputs(X, rr):
            return [X, rr] if spec["context"] == "rr" else X

        class MacroF1(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                logs["val_macro_f1"] = macro_f1(yva, self.model.predict(inputs(Xva, rrva), batch_size=512, verbose=0).argmax(1))

        t0, history = time.time(), []

        class Progress(keras.callbacks.Callback):
            def on_epoch_begin(self, epoch, logs=None):
                self.s = time.time()

            def on_epoch_end(self, epoch, logs=None):
                row = {"epoch": epoch + 1, "seconds": round(time.time() - self.s, 1),
                       **{k: float(logs[k]) for k in ("loss", "accuracy", "val_loss", "val_accuracy", "val_macro_f1")},
                       "learning_rate": float(keras.ops.convert_to_numpy(self.model.optimizer.learning_rate))}
                history.append(row)
                with (out / "history.csv").open("w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=row)
                    w.writeheader()
                    w.writerows(history)
                print(f"[{run_id}] epoch {row['epoch']} {row['seconds']}s loss {row['loss']:.4f} val_loss {row['val_loss']:.4f} "
                      f"val_macro_f1 {row['val_macro_f1']:.4f} lr {row['learning_rate']:.6f} | elapsed {time.time() - t0:.0f}s", flush=True)

        callbacks = [MacroF1(),
                     keras.callbacks.EarlyStopping("val_macro_f1", mode="max", patience=STOP_PATIENCE, restore_best_weights=True),
                     keras.callbacks.ReduceLROnPlateau("val_macro_f1", mode="max", factor=LR_FACTOR, patience=LR_PATIENCE),
                     Progress()]
        kwargs = {"validation_data": (inputs(Xva, rrva), yva), "epochs": max_epochs, "callbacks": callbacks, "verbose": 0}
        if spec["sampler"] == "balanced":
            by_class = [np.flatnonzero(ytr == k) for k in range(4)]
            present = np.array([k for k in range(4) if len(by_class[k])])
            rng = np.random.default_rng(seed + 1)
            steps = int(np.ceil(len(ytr) / BATCH))                   # same steps per epoch as shuffling

            class Balanced(keras.utils.PyDataset):
                def __len__(self):
                    return steps

                def __getitem__(self, i):
                    k = present[rng.integers(0, len(present), BATCH)]   # each class equally likely
                    j = np.array([by_class[c][rng.integers(len(by_class[c]))] for c in k])
                    x = (Xtr[j], rrtr[j]) if spec["context"] == "rr" else Xtr[j]
                    return x, ytr[j]
            model.fit(Balanced(), **kwargs)
        else:
            sw = weights[ytr].astype(np.float32) if use_weights else None   # sample weights: training loss only
            model.fit(inputs(Xtr, rrtr), ytr, sample_weight=sw, batch_size=BATCH, shuffle=True, **kwargs)

        probs = model.predict(inputs(Xva, rrva), batch_size=512, verbose=0)
        model.save(out / "model.keras", include_optimizer=False)
        np.savez_compressed(out / "validation_predictions.npz", probs=probs.astype(np.float32), y=yva, record=rva)
        if "evaluation" in split:                                    # CV: held-out fold, not used for fitting or stopping
            Xev, yev, rev, rrev = take(data, split["evaluation"])
            ev = model.predict(inputs(Xev, (rrev - mean) / std), batch_size=512, verbose=0)
            np.savez_compressed(out / "evaluation_predictions.npz", probs=ev.astype(np.float32), y=yev, record=rev)
            entry["has_evaluation_fold"] = True
            entry["evaluation_fold"] = {"macro_f1": macro_f1(yev, ev.argmax(1)), "per_class": class_report(yev, ev.argmax(1))}
        best = max(history, key=lambda r: r["val_macro_f1"])
        entry.update(status="completed", finished_at=now(), epochs=len(history), best_epoch=best["epoch"],
                     seconds=round(time.time() - t0), mean_epoch_seconds=round(float(np.mean([r["seconds"] for r in history])), 1),
                     steps_per_epoch=int(np.ceil(len(ytr) / BATCH)), class_weights=weights.tolist(),
                     class_weights_applied=use_weights, rr_mean=mean.tolist(), rr_std=std.tolist(), environment=device,
                     validation={"macro_f1": macro_f1(yva, probs.argmax(1)), "per_class": class_report(yva, probs.argmax(1))})
        register(env, entry)
        print(f"[{run_id}] completed: val macro-F1 {entry['validation']['macro_f1']:.4f}", flush=True)
        return entry
    except BaseException as exc:                                      # failures and interrupts are logged too
        entry.update(status="failed" if isinstance(exc, Exception) else "interrupted", finished_at=now(),
                     error="".join(traceback.format_exception_only(type(exc), exc)).strip()[:2000])
        register(env, entry)
        raise


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=[*VARIANTS, "combined"], default="baseline")
    ap.add_argument("--spec-json", help="explicit variant spec (needed for the 'combined' variant)")
    ap.add_argument("--split", default="interpatient_v1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    ap.add_argument("--tag", default="")
    ap.add_argument("--limit", type=int, default=0, help="smoke runs only: train on N random windows")
    add_path_args(ap)
    a = ap.parse_args()
    train(Env(a.data, a.splits, a.out, a.model_config, a.allow_cpu), a.variant, a.split, a.seed,
          spec=json.loads(a.spec_json) if a.spec_json else None, max_epochs=a.max_epochs, tag=a.tag, limit=a.limit)
