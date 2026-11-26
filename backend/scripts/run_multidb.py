"""Runs every (model, seed) of the multi-database study, then the validation evaluation. Resumable.

  python -u run_multidb.py --estimate             time projection only
  python -u run_multidb.py --smoke                tiny end-to-end check (results go to <out>/smoke and mean nothing)
  python -u run_multidb.py [--max-hours 8.5]      the real runs; rerun the same command to resume

Neural runs: one process per GPU, in parallel. The boosted-tree runs use the CPU and run alone afterwards, so they
do not starve the GPU input pipelines. No test file is ever opened here.
Exit codes: 0 done, 1 some runs failed, 3 time budget reached (rerun to resume).
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import experiments as E  # noqa: E402
import gpu_check  # noqa: E402
import models_multidb as M  # noqa: E402
import train as T  # noqa: E402
import train_multidb as TM  # noqa: E402

MEASURED_MS = 24.1              # T4, ms per training batch of 64, CNN-BiLSTM (user's measurement)
STARTUP_S = 90                  # ASSUMED: process start, loading ~1 GB, augmentation
VAL_S_PER_EPOCH = 30            # ASSUMED: predicting ~90k validation windows once per epoch
GBM_MIN = (15, 60)              # ASSUMED range in minutes per boosted-tree run on 4 CPU cores
START = time.time()


def estimate(workers: int) -> None:
    rep = json.loads((T.CODE / "artifacts/multidb_data_report.json").read_text())
    cfg = json.loads((T.CODE / "models/preprocessing_config.json").read_text())
    counts = rep["splits"]["train"]["windows"]
    copies = cfg["augmentation_training_only"]["copies_per_class_incl_original"]
    n = sum(v * copies.get(c, 1) for c, v in counts.items())
    steps = int(np.ceil(n / TM.BATCH))
    epoch = steps * MEASURED_MS / 1000 + VAL_S_PER_EPOCH
    runs = len(M.NEURAL) * len(TM.SEEDS)
    print(f"training windows after augmentation {n:,} -> {steps:,} steps/epoch; one epoch ~{epoch:.0f} s at {MEASURED_MS} ms/batch")
    print(f"{runs} neural runs on {workers} GPU(s) + {len(TM.SEEDS)} boosted-tree runs on CPU")
    for label, epochs in (("early stop around epoch 10", 10), ("around epoch 15", 15), (f"all {TM.MAX_EPOCHS} epochs", TM.MAX_EPOCHS)):
        one = (STARTUP_S + epochs * epoch) / 3600
        neural = one * runs / workers
        lo, hi = (len(TM.SEEDS) * m / 60 for m in GBM_MIN)
        print(f"  {label:28s}: {one * 60:4.0f} min/run | neural {neural:5.1f} h | total {neural + lo:.1f}-{neural + hi:.1f} h")
    print("Caveats: 24.1 ms/batch was measured for CNN-BiLSTM only; the other architectures are assumed equal. Two processes share\n"
          "the CPUs, which can slow both. Each run prints its real ms/batch; use that to correct this estimate.")


def gpu_count() -> int:
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.TimeoutExpired):
        return 0
    return sum(ln.startswith("GPU ") for ln in out.splitlines())


class Runner:
    def __init__(self, env, a):
        self.env, self.a, self.tag = env, a, "__smoke" if a.smoke else ""
        self.failed, self.durations, self.deferred = [], [], 0
        self.children: set[subprocess.Popen] = set()
        self.lock = threading.Lock()

    def command(self, model, seed):
        e = self.env
        cmd = [sys.executable, "-u", str(HERE / "train_multidb.py"), "--model", model, "--seed", str(seed), "--data", str(e.data),
               "--splits", str(e.splits), "--out", str(e.out), "--model-config", str(e.model_config)]
        if e.allow_cpu:
            cmd.append("--allow-cpu")
        if self.a.smoke:
            cmd += ["--tag", self.tag, "--limit", "2048", "--max-epochs", "1"]
        return cmd

    def longest(self, model) -> float:
        past = [r["seconds"] for r in T.read_registry(self.env)["runs"] if r.get("kind") == "multidb_training"
                and r.get("status") == "completed" and not r.get("note") and (r["variant"] == "gbm") == (model == "gbm")]
        return max(past + self.durations, default=0)

    def one(self, slot: str, gpu: str | None, model: str, seed: int, position: str) -> None:
        rid = TM.run_id(model, seed, self.tag)
        env = dict(os.environ)
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = gpu
        log = self.env.out / "logs" / f"{rid}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        print(f"--- [{slot}] {position} {rid} | elapsed {(time.time() - START) / 60:.1f} min", flush=True)
        t = time.time()
        with log.open("a") as f:
            p = subprocess.Popen(self.command(model, seed), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            self.children.add(p)
            tail = []
            for line in p.stdout:
                f.write(line)
                f.flush()
                tail = (tail + [line])[-25:]
                if line.startswith(f"[{rid}]"):
                    print(f"[{slot}] {line}", end="", flush=True)
            p.wait()
            self.children.discard(p)
        with self.lock:
            if p.returncode:
                print(f"[FAILED] {rid} exit {p.returncode}\n" + "".join(tail), flush=True)
                entry = next((r for r in T.read_registry(self.env)["runs"] if r["run_id"] == rid), None)
                if entry is None or entry["status"] == "running":      # killed (e.g. out of memory)
                    T.register(self.env, {**(entry or {"run_id": rid, "kind": "multidb_training", "variant": model, "seed": seed}),
                                          "status": "failed", "finished_at": T.now(), "error": "".join(tail)[-2000:]})
                self.failed.append(rid)
            else:
                self.durations.append(time.time() - t)

    def block(self, jobs: list[tuple[str, int]], slots: list[tuple[str, str | None]]) -> None:
        split, sha = TM.load_split(self.env)
        epochs, limit = (1, 2048) if self.a.smoke else (TM.MAX_EPOCHS, 0)
        todo = [(m, s) for m, s in jobs
                if not TM.completed(self.env, TM.run_id(m, s, self.tag), TM.fingerprint_for(split, sha, m, s, epochs, limit))]
        print(f"{len(jobs)} runs in this block; {len(jobs) - len(todo)} already finished; {len(todo)} to do on {len(slots)} slot(s)", flush=True)
        q: queue.Queue = queue.Queue()
        for i, job in enumerate(todo, 1):
            q.put((i, job))

        def worker(slot, gpu):
            while True:
                try:
                    i, (model, seed) = q.get_nowait()
                except queue.Empty:
                    return
                need = 1.2 * self.longest(model)
                if self.a.max_hours and not self.a.smoke and time.time() - START + need > self.a.max_hours * 3600:
                    with self.lock:
                        self.deferred += 1
                    continue
                self.one(slot, gpu, model, seed, f"run {i}/{len(todo)}:")

        threads = [threading.Thread(target=worker, args=s, daemon=True) for s in slots]
        for t in threads:
            t.start()
        try:
            for t in threads:
                while t.is_alive():
                    t.join(1)
        except KeyboardInterrupt:
            for p in list(self.children):
                p.terminate()                                          # train_multidb logs the interrupt
            for p in list(self.children):
                p.wait()
            raise


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--max-hours", type=float, default=8.5, help="stop cleanly before this many hours (0 = no limit)")
    ap.add_argument("--workers", type=int, help="parallel neural processes (default: one per GPU)")
    ap.add_argument("--models", nargs="*", default=M.ALL_MODELS, choices=M.ALL_MODELS)
    ap.add_argument("--resume-from", nargs="*", default=[], help="earlier results folders or zips")
    T.add_path_args(ap)
    a = ap.parse_args()
    if a.estimate:
        estimate(a.workers or 2)
        return
    env = TM.make_env(a, a.smoke)
    if not a.smoke:
        E.restore(env, a.resume_from)                                  # merges; earlier registry entries are never dropped
    env.out.mkdir(parents=True, exist_ok=True)
    split, _ = TM.load_split(env)
    if not env.data.exists():
        sys.exit(f"data file not found: {env.data}")
    assert T.sha256(env.data) == split["data_sha256"][TM.DEV_FILE], f"{env.data} does not match the split's recorded SHA-256"
    gpus = gpu_count()
    if not gpus and not env.allow_cpu:
        sys.exit("No GPU found. Choose the GPU T4 x2 accelerator, or pass --allow-cpu for a smoke test.")
    slots = [(f"gpu{i}", str(i)) for i in range(gpus)][: a.workers or gpus] if gpus else [(f"cpu{i}", None) for i in range(a.workers or 1)]
    if gpus == 1:
        print("Only one GPU visible: neural runs go one at a time.", flush=True)
    gpu_check.ensure_settings(env)
    seeds = TM.SEEDS[:2] if a.smoke else TM.SEEDS
    runner = Runner(env, a)
    neural = [(m, s) for s in seeds for m in M.NEURAL if m in a.models]            # seed by seed: partial output stays comparable
    runner.block(neural, slots)
    if "gbm" in a.models:
        runner.block([("gbm", s) for s in seeds], [("cpu", "")])
    if runner.deferred:
        print(f"[TIME BUDGET] {runner.deferred} runs not started. Attach this output and run the same command again.", flush=True)
        sys.exit(3)
    if runner.failed:
        sys.exit(f"{len(runner.failed)} runs failed (see experiments.json and logs/): {runner.failed}")
    if set(a.models) == set(M.ALL_MODELS):
        cmd = [sys.executable, "-u", str(HERE / "evaluate_multidb.py"), "--out", str(env.out), "--splits", str(env.splits)]
        subprocess.run(cmd + (["--smoke"] if a.smoke else []), check=True)
    print(f"[ALL RUNS COMPLETE] {(time.time() - START) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
