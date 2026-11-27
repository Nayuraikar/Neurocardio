"""One training run of the multi-database study (fixed patient-wise split, one model, one seed).

  python -u train_multidb.py --model rr_fusion --seed 42 [--data multidb_dev.npz] [--out DIR]

Protocol (docs/multidb-protocol.md): batch 64, Adam 1e-3, at most 25 epochs, early stopping patience 5 on validation
macro-F1 with best weights restored, learning rate halved after 2 epochs without improvement, sqrt-inverse-frequency
class weights from training counts, the same training-only augmentation as the deployed model.
Only multidb_dev.npz is read. Its SHA-256 must equal the one recorded in the split file, so a file holding test
patients is rejected. A completed run with the same settings is skipped.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import signal
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analysis_common as A  # noqa: E402
import models_multidb as M  # noqa: E402
import split_checks  # noqa: E402
import train as T  # noqa: E402

SPLIT = "multidb_v1"
DEV_FILE = "multidb_dev.npz"
PROTOCOL = 1                                    # bump when a change would make old runs incomparable
SEEDS = [42, 11, 22]
BATCH, LR, MAX_EPOCHS, PATIENCE, LR_PATIENCE, LR_FACTOR = 64, 1e-3, 25, 5, 2, 0.5
S_HEAD_LAMBDA = 0.5
GBM = {"learning_rate": 0.1, "max_leaf_nodes": 31, "l2_regularization": 1.0, "min_samples_leaf": 50, "max_features": 0.8}
GBM_STEP, GBM_MAX_ITER, GBM_PATIENCE = 20, 400, 5
SMOKE_VAL_STRIDE = 20


def make_env(a, smoke: bool = False) -> T.Env:
    data = a.data or T.find_data(DEV_FILE)
    env = T.Env(data, a.splits, a.out, a.model_config, a.allow_cpu)
    if smoke:
        env = T.Env(env.data, env.splits, env.out / "smoke", env.model_config, a.allow_cpu)
    return env


def load_split(env: T.Env) -> tuple[dict, str]:
    raw = (env.splits / f"{SPLIT}.json").read_bytes()
    split = json.loads(raw)
    split_checks.check_split(split)
    return split, hashlib.sha256(raw).hexdigest()


def load_dev(env: T.Env, split: dict) -> dict:
    data = env.dataset()
    want = split["data_sha256"][DEV_FILE]
    assert data["sha256"] == want, f"{env.data} is not the dev file this split was built with (sha256 {data['sha256'][:12]} != {want[:12]})"
    split_checks.check_arrays(split, data["record"].astype(str), data["patient"].astype(str), ("train", "validation"))
    return data


def part(data: dict, split: dict, name: str, stride: int = 1) -> dict:
    idx = np.flatnonzero(np.isin(data["record"], split[name]))[::stride]
    return {"X": data["X"][idx], "y": data["y"][idx].astype(np.int64), "rr": data["rr"][idx],
            **{k: data[k][idx].astype(str) for k in ("record", "patient", "db")}}


def run_id(model: str, seed: int, tag: str = "") -> str:
    return f"{SPLIT}__{model}__seed{seed}{tag}"


def model_file(env: T.Env, rid: str, model: str) -> Path:
    return env.runs / rid / ("model.joblib" if model == "gbm" else "model.keras")


def completed(env: T.Env, rid: str, fingerprint: dict | None = None) -> dict | None:
    for r in T.read_registry(env)["runs"]:
        if r["run_id"] == rid and r["status"] == "completed" and (fingerprint is None or r.get("fingerprint") == fingerprint):
            if (env.runs / rid / "validation_predictions.npz").exists() and model_file(env, rid, r["variant"]).exists():
                return r
    return None


def fingerprint_for(split: dict, split_sha: str, model: str, seed: int, max_epochs: int, limit: int) -> dict:
    return {"protocol": PROTOCOL, "model": model, "seed": seed, "split_sha256": split_sha,
            "data_sha256": split["data_sha256"][DEV_FILE], "max_epochs": max_epochs, "limit": limit, "batch_size": BATCH}


def predict_proba(model_name: str, model, X: np.ndarray, rr: np.ndarray, rr_mean, rr_std) -> tuple[np.ndarray, np.ndarray | None]:
    """-> (n,4) probabilities and the auxiliary S probability if the model has one. Also used by final_test.py."""
    if model_name == "gbm":
        return model.predict_proba(M.handcrafted_features(X, rr)).astype(np.float32), None
    x = X[..., None]
    if model_name in M.USES_RR:
        x = [x, (rr - np.asarray(rr_mean, np.float32)) / np.asarray(rr_std, np.float32)]
    out = model.predict(x, batch_size=512, verbose=0).astype(np.float32)
    if model_name in M.S_HEAD:
        return out[:, :4], out[:, 4]
    return out, None


def train_neural(env, name, seed, tr, va, max_epochs, rid, out):
    import keras
    keras.backend.clear_session()
    keras.utils.set_random_seed(seed)
    counts = np.bincount(tr["y"], minlength=4)                       # training only, before augmentation
    X, y, rr = T.augment(env.cfg, tr["X"][..., None], tr["y"], tr["rr"], seed)
    mean, std = rr.mean(0), rr.std(0) + 1e-8                         # training only
    weights = T.class_weights("sqrt_weighted_ce", counts.astype(np.float64))
    model = M.build(name)
    use_rr = name in M.USES_RR
    x_train = [X, (rr - mean) / std] if use_rr else X
    extra = {"train_counts_post_augmentation": np.bincount(y, minlength=4).tolist(), "class_weights": weights.tolist(),
             "rr_mean": mean.tolist(), "rr_std": std.tolist(), "parameter_count": int(model.count_params()),
             "trainable_parameter_count": int(sum(int(np.prod(v.shape)) for v in model.trainable_weights)),
             "steps_per_epoch": int(np.ceil(len(y) / BATCH))}
    if name in M.S_HEAD:
        pos = float((y == 1).sum())
        extra["s_head"] = {"lambda": S_HEAD_LAMBDA, "positive_weight": float(np.sqrt((len(y) - pos) / pos)), "used_at_inference": False}
        model.compile(optimizer=keras.optimizers.Adam(LR), loss=M.s_head_loss(weights, extra["s_head"]["positive_weight"], S_HEAD_LAMBDA))
        sample_weight = None                                          # class weights are inside the loss
    else:
        model.compile(optimizer=keras.optimizers.Adam(LR), loss="sparse_categorical_crossentropy")
        sample_weight = weights[y].astype(np.float32)
    history, t0, steps = [], time.time(), extra["steps_per_epoch"]

    class Monitor(keras.callbacks.Callback):
        """Computes validation macro-F1 (must run before the callbacks that read it) and prints progress."""

        def on_epoch_begin(self, epoch, logs=None):
            self.start = time.time()

        def on_train_batch_end(self, batch, logs=None):
            if batch == 49:
                self.mark = time.time()
            elif batch > 49 and (batch + 1) % 1000 == 0:
                print(f"[{rid}] step {batch + 1}/{steps} {1000 * (time.time() - self.mark) / (batch - 49):.1f} ms/batch", flush=True)

        def on_epoch_end(self, epoch, logs=None):
            train_s = time.time() - self.start
            probs, _ = predict_proba(name, self.model, va["X"], va["rr"], mean, std)
            logs["val_macro_f1"] = A.from_cm(A.cm(va["y"], probs.argmax(1)))["macro_f1"]
            row = {"epoch": epoch + 1, "train_seconds": round(train_s, 1), "seconds": round(time.time() - self.start, 1),
                   "ms_per_batch": round(1000 * train_s / steps, 1), "loss": float(logs["loss"]),
                   "val_nll_unweighted": A.calibration(probs, va["y"])["nll"], "val_macro_f1": float(logs["val_macro_f1"]),
                   "learning_rate": float(keras.ops.convert_to_numpy(self.model.optimizer.learning_rate))}
            history.append(row)
            with (out / "history.csv").open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=row)
                w.writeheader()
                w.writerows(history)
            print(f"[{rid}] epoch {row['epoch']}/{max_epochs} {row['seconds']}s ({row['ms_per_batch']} ms/batch) loss {row['loss']:.4f} "
                  f"val_macro_f1 {row['val_macro_f1']:.4f} lr {row['learning_rate']:.6f} | elapsed {(time.time() - t0) / 60:.1f} min", flush=True)

    callbacks = [Monitor(),
                 keras.callbacks.EarlyStopping("val_macro_f1", mode="max", patience=PATIENCE, restore_best_weights=True),
                 keras.callbacks.ReduceLROnPlateau("val_macro_f1", mode="max", factor=LR_FACTOR, patience=LR_PATIENCE)]
    model.fit(x_train, y, sample_weight=sample_weight, batch_size=BATCH, shuffle=True, epochs=max_epochs,
              callbacks=callbacks, verbose=0)
    probs, s_aux = predict_proba(name, model, va["X"], va["rr"], mean, std)
    model.save(out / "model.keras", include_optimizer=False)
    best = max(history, key=lambda r: r["val_macro_f1"])
    extra.update(epochs=len(history), best_epoch=best["epoch"], mean_ms_per_batch=round(float(np.mean([r["ms_per_batch"] for r in history])), 1),
                 mean_epoch_seconds=round(float(np.mean([r["seconds"] for r in history])), 1))
    return probs, s_aux, extra


def train_gbm(seed, tr, va, rid, out, smoke):
    import joblib
    import sklearn
    from sklearn.ensemble import HistGradientBoostingClassifier
    t0 = time.time()
    Xtr, Xva = M.handcrafted_features(tr["X"], tr["rr"]), M.handcrafted_features(va["X"], va["rr"])
    counts = np.bincount(tr["y"], minlength=4)
    weights = T.class_weights("sqrt_weighted_ce", counts.astype(np.float64))
    step, max_iter = (2, 4) if smoke else (GBM_STEP, GBM_MAX_ITER)
    clf = HistGradientBoostingClassifier(**GBM, early_stopping=False, warm_start=True, max_iter=step, random_state=seed)
    best, best_model, stale, history = -1.0, None, 0, []
    for it in range(step, max_iter + 1, step):                       # early stopping on patient-held-out validation macro-F1
        clf.set_params(max_iter=it)
        clf.fit(Xtr, tr["y"], sample_weight=weights[tr["y"]])
        score = A.from_cm(A.cm(va["y"], clf.predict_proba(Xva).argmax(1)))["macro_f1"]
        history.append({"iterations": it, "val_macro_f1": score})
        print(f"[{rid}] {it} boosting iterations val_macro_f1 {score:.4f} | elapsed {(time.time() - t0) / 60:.1f} min", flush=True)
        if score > best:
            best, best_model, stale = score, copy.deepcopy(clf), 0
        else:
            stale += 1
            if stale >= GBM_PATIENCE:
                break
    assert list(best_model.classes_) == [0, 1, 2, 3], "a class is missing from training"
    joblib.dump(best_model, out / "model.joblib", compress=3)
    extra = {"parameter_count": None, "feature_count": int(Xtr.shape[1]), "gbm": GBM, "best_iterations": int(best_model.max_iter),
             "tree_count": int(sum(len(i) for i in best_model._predictors)),
             "leaf_count": int(sum(p.get_n_leaf_nodes() for i in best_model._predictors for p in i)),
             "history": history, "class_weights": weights.tolist(), "augmentation_used": False, "sklearn": sklearn.__version__}
    return best_model.predict_proba(Xva).astype(np.float32), None, extra


def train(env: T.Env, name: str, seed: int, max_epochs: int = MAX_EPOCHS, tag: str = "", limit: int = 0) -> dict:
    if limit and not tag:
        raise ValueError("--limit is only for smoke runs; give them a --tag")
    rid = run_id(name, seed, tag)
    split, split_sha = load_split(env)
    fp = fingerprint_for(split, split_sha, name, seed, max_epochs, limit)
    done = completed(env, rid, fp)
    if done:
        print(f"[{rid}] already completed; skipping", flush=True)
        return done
    signal.signal(signal.SIGTERM, T._on_sigterm)
    out = env.runs / rid
    out.mkdir(parents=True, exist_ok=True)
    entry = {"run_id": rid, "kind": "multidb_training", "variant": name, "split": SPLIT, "seed": seed, "status": "running",
             "started_at": T.now(), "fingerprint": fp, "script_sha256": T.sha256(Path(__file__)),
             "models_sha256": T.sha256(HERE / "models_multidb.py")}
    if tag:
        entry["note"] = "smoke run; not a result"
    T.register(env, entry, new_attempt=True)
    t0 = time.time()
    try:
        data = load_dev(env, split)
        tr = part(data, split, "train")
        va = part(data, split, "validation", SMOKE_VAL_STRIDE if limit else 1)
        env._data = None                                              # free the combined copy
        del data
        if limit:                                                     # smoke only
            rng = np.random.default_rng(0)
            pick = np.unique(np.concatenate([rng.choice(len(tr["y"]), min(limit, len(tr["y"])), replace=False),
                                             *[rng.choice(np.flatnonzero(tr["y"] == c), 8, replace=False) for c in range(4)]]))
            tr = {k: v[pick] for k, v in tr.items()}
        entry["train_counts"] = np.bincount(tr["y"], minlength=4).tolist()
        entry["validation_counts"] = np.bincount(va["y"], minlength=4).tolist()
        if name == "gbm":
            probs, s_aux, extra = train_gbm(seed, tr, va, rid, out, bool(limit))
        else:
            entry["environment"] = T.setup_tf(env)
            probs, s_aux, extra = train_neural(env, name, seed, tr, va, max_epochs, rid, out)
        arrays = {"probs": probs, "y": va["y"], "record": va["record"], "patient": va["patient"], "db": va["db"]}
        if s_aux is not None:
            arrays["s_aux"] = s_aux
        np.savez_compressed(out / "validation_predictions.npz", **arrays)
        T.write_json(out / "config.json", {**entry, **extra, "protocol": {
            "batch_size": BATCH, "learning_rate": LR, "max_epochs": max_epochs, "early_stop_patience": PATIENCE,
            "lr_plateau": {"patience": LR_PATIENCE, "factor": LR_FACTOR}, "augmentation": env.cfg["augmentation_training_only"]}})
        entry.update(extra)
        entry.update(status="completed", finished_at=T.now(), seconds=round(time.time() - t0),
                     validation={"macro_f1": A.report(va["y"], probs.argmax(1))["macro_f1"],
                                 "per_class": A.report(va["y"], probs.argmax(1))["per_class"]})
        T.register(env, entry)
        print(f"[{rid}] completed in {entry['seconds'] / 60:.1f} min: val macro-F1 (argmax) {entry['validation']['macro_f1']:.4f}", flush=True)
        return entry
    except BaseException as exc:                                      # failures and interrupts are logged too
        entry.update(status="failed" if isinstance(exc, Exception) else "interrupted", finished_at=T.now(),
                     error="".join(traceback.format_exception_only(type(exc), exc)).strip()[:2000])
        T.register(env, entry)
        raise


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=M.ALL_MODELS, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    ap.add_argument("--tag", default="")
    ap.add_argument("--limit", type=int, default=0, help="smoke runs only: train on N random windows")
    T.add_path_args(ap)
    a = ap.parse_args()
    train(make_env(a), a.model, a.seed, max_epochs=a.max_epochs, tag=a.tag, limit=a.limit)
