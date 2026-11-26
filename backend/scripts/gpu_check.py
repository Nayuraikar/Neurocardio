"""Confirm the GPU, decide once whether deterministic ops are free, and project stage durations.

Writes <out>/settings.json. The decision is reused by every later run, so all results share it.
Deterministic ops stay ON if they work with every loss/sampler path and cost at most 10%.
Run: python -u gpu_check.py [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as T  # noqa: E402

BATCHES = 60
MEAN_EPOCHS, MIN_EPOCHS, MAX_EPOCHS_SEEN = 25, 14, 42     # from the 9 saved training histories
FIXED_STEPS, CV_STEPS = 630, 500                          # steps/epoch: fixed split, typical CV iteration
STARTUP_S = 40                                            # process start + data load per run (assumed)
STAGE_RUNS = {1: (3, 45, 45), 2: (0, 30, 30), 3: (0, 10, 35)}   # fixed runs, min CV runs, max CV runs


def bench(deterministic: bool) -> dict:
    import numpy as np
    import keras
    import tensorflow as tf
    keras.mixed_precision.set_global_policy("float32")
    keras.utils.set_random_seed(0)
    if deterministic:
        tf.config.experimental.enable_op_determinism()
    x = np.random.default_rng(0).normal(size=(64 * BATCHES, 720, 1)).astype("float32")
    y = np.arange(len(x)) % 4
    out = {}
    for loss, ctx in (("sqrt_weighted_ce", "none"), ("focal_g2", "none"), ("sqrt_weighted_ce", "rr")):   # every op path
        m = T.build_model(ctx)
        m.compile("adam", T.make_loss(loss))
        xs = [x, np.zeros((len(x), 5), "float32")] if ctx == "rr" else x
        head = [a[:640] for a in xs] if ctx == "rr" else x[:640]
        m.fit(head, y[:640], sample_weight=np.ones(640, "float32"), batch_size=64, verbose=0)    # warm-up/compile
        if (loss, ctx) == ("sqrt_weighted_ce", "none"):
            t = time.time()
            m.fit(x, y, sample_weight=np.ones(len(x), "float32"), batch_size=64, verbose=0)
            out["sec_per_train_batch"] = (time.time() - t) / BATCHES
            m.predict(x[:1024], batch_size=512, verbose=0)
            t = time.time()
            m.predict(x, batch_size=512, verbose=0)
            out["sec_per_1000_predictions"] = (time.time() - t) / len(x) * 1000
    return out


def run_bench(flag: str) -> dict:
    p = subprocess.run([sys.executable, "-u", __file__, "--bench", flag], capture_output=True, text=True)
    line = next((ln for ln in p.stdout.splitlines() if ln.startswith("BENCH ")), None)
    if p.returncode or not line:
        return {"error": (p.stderr or p.stdout)[-1500:]}
    return json.loads(line[6:])


def project(spb: float, pred: float) -> None:
    def run_s(steps, epochs, val_windows):
        return STARTUP_S + epochs * (steps * spb + val_windows / 1000 * pred)
    print(f"Projection from {spb * 1000:.1f} ms/train batch (epochs per run assumed {MEAN_EPOCHS}, range {MIN_EPOCHS}-{MAX_EPOCHS_SEEN}):")
    print(f"  one fixed-split run ~{run_s(FIXED_STEPS, MEAN_EPOCHS, 14600) / 60:.1f} min, one CV run ~{run_s(CV_STEPS, MEAN_EPOCHS, 10000) / 60:.1f} min")
    for stage, (fixed, lo, hi) in STAGE_RUNS.items():
        a = fixed * run_s(FIXED_STEPS, MEAN_EPOCHS, 14600) + lo * run_s(CV_STEPS, MEAN_EPOCHS, 10000)
        b = fixed * run_s(FIXED_STEPS, MEAN_EPOCHS, 14600) + hi * run_s(CV_STEPS, MEAN_EPOCHS, 10000)
        worst = fixed * run_s(FIXED_STEPS, MAX_EPOCHS_SEEN, 14600) + hi * run_s(CV_STEPS, MAX_EPOCHS_SEEN, 10000)
        rng = f"{a / 3600:.1f}" if lo == hi else f"{a / 3600:.1f}-{b / 3600:.1f}"
        print(f"  stage {stage}: {fixed + lo}{'' if lo == hi else f'-{fixed + hi}'} runs, ~{rng} h (if every run took {MAX_EPOCHS_SEEN} epochs: {worst / 3600:.1f} h)")


def ensure_settings(env: "T.Env") -> dict:
    if env.settings.exists():
        s = json.loads(env.settings.read_text())
        print(f"using stored settings: deterministic_ops={s['deterministic_ops']} ({s['reason']})")
        return s
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    print(f"TensorFlow {tf.__version__}; GPUs visible: {len(gpus)} {[tf.config.experimental.get_device_details(g).get('device_name') for g in gpus]}")
    if not gpus and not env.allow_cpu:
        sys.exit("No GPU visible to TensorFlow. Enable a GPU accelerator, or pass --allow-cpu for a smoke test.")
    del tf
    on, off = run_bench("on"), run_bench("off")
    if "error" in off:
        sys.exit("benchmark failed even without deterministic ops:\n" + off["error"])
    if "error" in on:
        det, reason, used = False, "deterministic ops failed on this device", off
    else:
        ratio = on["sec_per_train_batch"] / off["sec_per_train_batch"]
        det = ratio <= 1.10
        reason = f"on/off time ratio {ratio:.2f}"
        used = on if det else off
    s = {"deterministic_ops": det, "reason": reason, "gpu": bool(gpus), "mixed_precision": False, "batch_size": T.BATCH,
         "bench_on": on, "bench_off": off, "sec_per_train_batch": used["sec_per_train_batch"],
         "sec_per_1000_predictions": used["sec_per_1000_predictions"], "decided_at": T.now()}
    T.write_json(env.settings, s)
    print(f"deterministic_ops={det} ({reason}); mixed precision OFF; batch size stays {T.BATCH}")
    return s


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", choices=["on", "off"], help=argparse.SUPPRESS)
    T.add_path_args(ap)
    a = ap.parse_args()
    if a.bench:
        print("BENCH " + json.dumps(bench(a.bench == "on")), flush=True)
    else:
        st = ensure_settings(T.Env(a.data, a.splits, a.out, a.model_config, a.allow_cpu))
        project(st["sec_per_train_batch"], st["sec_per_1000_predictions"])
        if not st["gpu"]:
            print("NOTE: these timings are CPU timings. They say nothing about a T4.")
