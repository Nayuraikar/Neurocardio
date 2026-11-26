"""Stage runner for the model-improvement track. Resumable: finished runs are skipped.

  python -u experiments.py --stage 1|2|3 [--smoke] [--max-hours 8]

Stage 1: reproduction (gate), then cross-validation of baseline, focal_g2, cb_focal
Stage 2: cross-validation of balanced_sampler, rr_context
Stage 3: thresholds (post-hoc), combined variant, 5-seed ensembles, ablation table
Only development records are used. The sealed test file is never opened here.
Exit codes: 0 done, 1 some runs failed, 2 reproduction gate failed, 3 time budget reached (rerun to resume).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analysis_common as A  # noqa: E402
import gpu_check  # noqa: E402
import train as T  # noqa: E402

FOLDS = range(5)
SINGLES = ["baseline", "focal_g2", "cb_focal", "balanced_sampler", "rr_context"]
IMBALANCE = ["baseline", "focal_g2", "cb_focal", "balanced_sampler"]
START = time.time()


class Ctx:
    def __init__(self, env, smoke, max_hours):
        self.env, self.smoke, self.max_hours = env, smoke, max_hours
        self.seeds = [42, 11] if smoke else [42, 11, 22]
        self.extra = [33] if smoke else [33, 44]
        self.folds = [0] if smoke else list(FOLDS)
        self.tag = "__smoke" if smoke else ""
        self.analysis = env.out / "analysis"
        self.failed: list[str] = []


# ------------------------------------------------------------------ resume across sessions
def merge_registry(env, source: Path) -> int:
    mine = {r["run_id"]: r for r in T.read_registry(env)["runs"]}
    added = 0
    for r in json.loads(source.read_text()).get("runs", []):
        old = mine.get(r["run_id"])
        if old is None or (old["status"] != "completed" and r["status"] == "completed"):
            mine[r["run_id"]] = r
            added += 1
    T.write_json(env.registry, {"runs": list(mine.values())})
    return added


def restore(env, resume_from: list[str]) -> None:
    """Copy earlier results (folders or neurocardio_results*.zip) into the output directory."""
    sources = [Path(p) for p in resume_from]
    if T.KAGGLE_INPUT.exists():
        sources += sorted({p.parent for p in T.KAGGLE_INPUT.rglob("experiments.json") if "smoke" not in p.parts})
        sources += sorted(T.KAGGLE_INPUT.rglob("neurocardio_results*.zip"))
    seed_log = T.CODE / "artifacts/experiments_local.json"
    if not seed_log.exists():
        seed_log = T.CODE / "artifacts/experiments.json"
    if seed_log.exists():
        merge_registry(env, seed_log)                      # keeps the earlier local attempts in one log
    for src in sources:
        if src.suffix == ".zip":
            tmp = Path(tempfile.mkdtemp())
            with zipfile.ZipFile(src) as z:
                z.extractall(tmp)
            src = tmp
        if not (src / "experiments.json").exists() or src.resolve() == env.out.resolve():
            continue
        for sub in ("runs", "analysis", "final", "final_multidb"):
            if (src / sub).exists():
                for item in (src / sub).iterdir():
                    dst = env.out / sub / item.name
                    if not dst.exists():
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        (shutil.copytree if item.is_dir() else shutil.copy)(item, dst)
        if (src / "settings.json").exists() and not env.settings.exists():
            env.out.mkdir(parents=True, exist_ok=True)
            shutil.copy(src / "settings.json", env.settings)
        print(f"restored {merge_registry(env, src / 'experiments.json')} registry entries from {src}", flush=True)


# ------------------------------------------------------------------ running jobs
def completed(ctx, run_id) -> dict | None:
    return next((r for r in T.read_registry(ctx.env)["runs"] if r["run_id"] == run_id and r["status"] == "completed"
                 and (ctx.env.runs / run_id / "validation_predictions.npz").exists()), None)


def run_jobs(ctx, jobs: list[tuple]) -> None:
    """jobs: (variant, split, seed, spec-or-None)"""
    todo = [j for j in jobs if not completed(ctx, T.run_id_for(j[1], j[0], j[2], ctx.tag))]
    print(f"{len(jobs)} runs in this block; {len(jobs) - len(todo)} already finished; {len(todo)} to do", flush=True)
    durations: list[float] = []
    for i, (variant, split, seed, spec) in enumerate(todo):
        rid = T.run_id_for(split, variant, seed, ctx.tag)
        elapsed = time.time() - START
        longest = max(durations or [r.get("seconds", 0) for r in T.read_registry(ctx.env)["runs"] if r.get("status") == "completed" and not r.get("note")] or [0])
        if ctx.max_hours and not ctx.smoke and elapsed + 1.25 * longest > ctx.max_hours * 3600:
            print(f"[TIME BUDGET] {elapsed / 3600:.2f} h used; the next run may not fit in {ctx.max_hours} h. "
                  f"{len(todo) - i} runs left. Run the same stage again to resume.", flush=True)
            sys.exit(3)
        eta = f"ETA for block {np.mean(durations) * (len(todo) - i) / 60:.0f} min" if durations else "ETA after first run"
        print(f"--- run {i + 1}/{len(todo)}: {rid} | elapsed {elapsed / 60:.1f} min | {eta}", flush=True)
        t = time.time()
        try:
            if ctx.smoke:                                    # in-process keeps the smoke test short
                T.train(ctx.env, variant, split, seed, spec=spec, max_epochs=1, tag=ctx.tag, limit=2048)
            else:                                            # one process per run: clean GPU memory and seeds
                cmd = [sys.executable, "-u", str(HERE / "train.py"), "--variant", variant, "--split", split, "--seed", str(seed),
                       "--data", str(ctx.env.data), "--splits", str(ctx.env.splits), "--out", str(ctx.env.out),
                       "--model-config", str(ctx.env.model_config)]
                if spec:
                    cmd += ["--spec-json", json.dumps(spec)]
                if ctx.env.allow_cpu:
                    cmd.append("--allow-cpu")
                subprocess.run(cmd, check=True)
        except Exception as exc:                              # logged in experiments.json by train(); carry on
            print(f"[FAILED] {rid}: {exc}", flush=True)
            entry = next((r for r in T.read_registry(ctx.env)["runs"] if r["run_id"] == rid), None)
            if entry is None or entry["status"] == "running":     # process was killed (e.g. out of memory)
                T.register(ctx.env, {**(entry or {"run_id": rid, "kind": "training", "variant": variant, "split": split, "seed": seed}),
                                     "status": "failed", "finished_at": T.now(), "error": str(exc)[:2000]})
            ctx.failed.append(rid)
        durations.append(time.time() - t)


def cv_jobs(ctx, variant, seeds, spec=None):
    return [(variant, f"cv5_dev_it{k}", s, spec) for s in seeds for k in ctx.folds]


# ------------------------------------------------------------------ reproduction gate
def gate(ctx) -> dict:
    ref = json.loads((T.CODE.parent / "kaggle/reproduction_reference.json").read_text())
    runs = [completed(ctx, T.run_id_for("interpatient_v1", "baseline", s, ctx.tag)) for s in ctx.seeds]
    if any(r is None for r in runs):
        sys.exit("reproduction runs are incomplete; run stage 1 again")
    vals = np.array([r["validation"]["macro_f1"] for r in runs])
    mean, sd = float(vals.mean()), float(vals.std(ddof=1))
    target = ref["validation_macro_f1"]
    result = {"rule": "deployed validation macro-F1 within mean +/- 2 SD of the reproduction seeds", "target": target,
              "seeds": ctx.seeds, "values": vals.tolist(), "mean": mean, "sd": sd, "interval": [mean - 2 * sd, mean + 2 * sd],
              "passed": bool(mean - 2 * sd <= target <= mean + 2 * sd), "steps_per_epoch": [r["steps_per_epoch"] for r in runs],
              "per_class_by_seed": [r["validation"]["per_class"] for r in runs], "smoke": ctx.smoke, "checked_at": T.now()}
    T.write_json(ctx.analysis / "gate.json", result)
    print(f"[REPRODUCTION GATE] values {np.round(vals, 4).tolist()} mean {mean:.4f} sd {sd:.4f} "
          f"interval [{mean - 2 * sd:.4f}, {mean + 2 * sd:.4f}] target {target:.4f} -> {'PASS' if result['passed'] else 'FAIL'}"
          + (" (smoke: meaningless, not enforced)" if ctx.smoke else ""), flush=True)
    if sd > 0.05:
        print("  WARNING: seed SD is above 0.05, so this gate is weak even if it passes.", flush=True)
    return result


def require_gate(ctx):
    p = ctx.analysis / "gate.json"
    if not ctx.smoke and (not p.exists() or not json.loads(p.read_text())["passed"]):
        sys.exit("The reproduction gate has not passed. Run stage 1 first (attach its output when using a new session).")


# ------------------------------------------------------------------ pooled out-of-fold predictions
def load_pred(ctx, rid, which):
    with np.load(ctx.env.runs / rid / f"{which}_predictions.npz", allow_pickle=False) as z:
        return z["probs"].astype(np.float64), z["y"], z["record"].astype(str)


def pooled(ctx, variant, seeds: list[int], thresholds: bool):
    """Held-out-fold predictions for one model per seed (len(seeds)==1) or a mean-probability ensemble."""
    ys, preds, recs, scales = [], [], [], []
    for k in ctx.folds:
        ids = [T.run_id_for(f"cv5_dev_it{k}", variant, s, ctx.tag) for s in seeds]
        missing = [i for i in ids if not completed(ctx, i)]
        if missing:
            raise FileNotFoundError(f"missing runs: {missing[:3]}")
        ev = [load_pred(ctx, i, "evaluation") for i in ids]
        assert all(np.array_equal(ev[0][1], e[1]) and np.array_equal(ev[0][2], e[2]) for e in ev)
        probs = np.mean([e[0] for e in ev], axis=0)
        w = np.ones(4)
        if thresholds:                                       # fit on the early-stop fold, apply to the held-out fold
            va = [load_pred(ctx, i, "validation") for i in ids]
            w = A.fit_thresholds(np.mean([v[0] for v in va], axis=0), va[0][1])
        ys.append(ev[0][1]); recs.append(ev[0][2]); preds.append((probs * w).argmax(1)); scales.append(w.tolist())
    return np.concatenate(ys), np.concatenate(preds), np.concatenate(recs), scales


def evaluate_config(ctx, label, variant, seed_groups, thresholds):
    """seed_groups: [[42],[11],[22]] for per-seed models, or [[42,11,22]] for one ensemble."""
    results = [pooled(ctx, variant, g, thresholds) for g in seed_groups]
    reports = [A.report(y, p) for y, p, _, _ in results]
    stat = lambda f: {"mean": float(np.mean([f(r) for r in reports])),
                      "sd": float(np.std([f(r) for r in reports], ddof=1)) if len(reports) > 1 else None}
    order = sorted(set(results[0][2]))
    row = {"label": label, "variant": variant, "thresholds": thresholds, "seed_groups": seed_groups,
           "ensemble_size": len(seed_groups[0]), "macro_f1": stat(lambda r: r["macro_f1"]),
           "accuracy": stat(lambda r: r["accuracy"]), "majority_baseline_accuracy": reports[0]["majority_baseline_accuracy"],
           "per_class": {c: {m: stat(lambda r, c=c, m=m: r["per_class"][c][m]) for m in ("precision", "recall", "f1")}
                         for c in A.CLASSES},
           "threshold_scales": [r[3] for r in results] if thresholds else None, "per_seed_group": reports}
    row["_cms"] = [A.record_cms(y, p, rec, order) for y, p, rec, _ in results]
    return row


def posthoc_log(ctx, row):
    rid = f"posthoc__{row['label']}{ctx.tag}"
    T.register(ctx.env, {"run_id": rid, "kind": "posthoc", "variant": row["variant"], "status": "completed", "finished_at": T.now(),
                         "thresholds": row["thresholds"], "seed_groups": row["seed_groups"], "macro_f1": row["macro_f1"],
                         "per_class": row["per_class"], **({"note": "smoke run; not a result"} if ctx.smoke else {})})


def stage3(ctx):
    require_gate(ctx)
    per_seed = lambda seeds: [[s] for s in seeds]
    try:
        rows = {v: evaluate_config(ctx, v, v, per_seed(ctx.seeds), False) for v in SINGLES}
    except FileNotFoundError as exc:
        sys.exit(f"stage 3 needs every cross-validation run from stages 1 and 2. {exc}")
    score = lambda v: rows[v]["macro_f1"]["mean"]
    best_imbalance = max(IMBALANCE, key=score)
    rr_helps = score("rr_context") > score("baseline")
    combined_spec = None
    if rr_helps and best_imbalance != "baseline":              # rule fixed in docs/experiment-protocol.md
        combined_spec = {**T.VARIANTS[best_imbalance], "context": "rr"}
    elif ctx.smoke:                                            # exercise the combined path in the smoke test
        combined_spec = {**T.VARIANTS["focal_g2"], "context": "rr"}
    T.write_json(ctx.analysis / "combined_spec.json", {"best_imbalance": best_imbalance, "rr_helps": rr_helps, "spec": combined_spec})
    print(f"best imbalance method: {best_imbalance}; RR context helps: {rr_helps}; combined spec: {combined_spec}", flush=True)
    if combined_spec:
        run_jobs(ctx, cv_jobs(ctx, "combined", ctx.seeds, combined_spec))
        rows["combined"] = evaluate_config(ctx, "combined", "combined", per_seed(ctx.seeds), False)
    best = max(rows, key=lambda v: rows[v]["macro_f1"]["mean"])
    spec_of = lambda v: combined_spec if v == "combined" else None
    targets = list(dict.fromkeys(["baseline", best]))
    run_jobs(ctx, [j for v in targets for j in cv_jobs(ctx, v, ctx.extra, spec_of(v))])
    five = ctx.seeds + ctx.extra

    table = list(rows.values())
    table.append(evaluate_config(ctx, "baseline+thresholds", "baseline", per_seed(ctx.seeds), True))
    for v in targets:
        table.append(evaluate_config(ctx, f"{v}+ensemble{len(ctx.seeds)}", v, [ctx.seeds], False))
        table.append(evaluate_config(ctx, f"{v}+ensemble{len(five)}", v, [five], False))
    if best != "baseline":
        table.append(evaluate_config(ctx, f"{best}+thresholds", best, per_seed(ctx.seeds), True))
    table.append(evaluate_config(ctx, f"{best}+thresholds+ensemble{len(five)}", best, [five], True))

    base_cms = rows["baseline"]["_cms"]
    for row in table:
        row["vs_baseline"] = None if row["label"] == "baseline" else A.compare(row["_cms"], base_cms)
        posthoc_log(ctx, row)
    for row in table:
        row.pop("_cms")
    suggestion = max(table, key=lambda r: r["macro_f1"]["mean"])
    T.write_json(ctx.analysis / "ablation.json", {"rows": table, "best_single_or_combined": best, "highest_cv_macro_f1": suggestion["label"],
                 "note": "Choosing the highest of many rows on the same folds is optimistic. The final single test evaluation is the unbiased check.",
                 "smoke": ctx.smoke})
    pm = lambda s: f"{s['mean']:.3f}" + ("" if s["sd"] is None else f"±{s['sd']:.3f}")
    lines = ["# Ablation (pooled held-out folds, DS1 only)" + (" - SMOKE, NOT RESULTS" if ctx.smoke else ""), "",
             "| Config | macro-F1 | " + " | ".join(f"{c} P | {c} R | {c} F1" for c in A.CLASSES) + " | accuracy (majority) | S meaningful | F meaningful |",
             "|" + "---|" * 17]
    for r in table:
        cells = " | ".join(f"{pm(r['per_class'][c]['precision'])} | {pm(r['per_class'][c]['recall'])} | {pm(r['per_class'][c]['f1'])}" for c in A.CLASSES)
        v = r["vs_baseline"]
        lines.append(f"| {r['label']} | {pm(r['macro_f1'])} | {cells} | {pm(r['accuracy'])} ({r['majority_baseline_accuracy']:.3f}) | "
                     + (f"{'YES' if v['S_f1']['meaningful'] else 'no'} | {'YES' if v['F_f1']['meaningful'] else 'no'} |" if v else "- | - |"))
    improved = [r["label"] for r in table if r["vs_baseline"] and (r["vs_baseline"]["S_f1"]["meaningful"] or r["vs_baseline"]["F_f1"]["meaningful"])]
    lines += ["", f"Highest CV macro-F1: **{suggestion['label']}** (selection on these folds is optimistic).",
              "S or F meaningfully improved by the pre-registered rule: " + (", ".join(improved) if improved else "**none. S/F did not meaningfully improve.**"),
              "", "F caveat: record 208 holds 373 of 415 development F beats; F results do not support general claims."]
    (ctx.analysis / "ablation.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, choices=[1, 2, 3], required=True)
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end check; results go to <out>/smoke and mean nothing")
    ap.add_argument("--max-hours", type=float, default=8.0, help="stop cleanly before this many hours (0 = no limit)")
    ap.add_argument("--resume-from", nargs="*", default=[], help="earlier results folders or zips")
    T.add_path_args(ap)
    a = ap.parse_args()
    env = T.Env(a.data, a.splits, a.out, a.model_config, a.allow_cpu)
    if a.smoke:
        env = T.Env(env.data, env.splits, env.out / "smoke", env.model_config, a.allow_cpu)
    else:
        restore(env, a.resume_from)
    env.out.mkdir(parents=True, exist_ok=True)
    ctx = Ctx(env, a.smoke, a.max_hours)
    print(f"stage {a.stage}{' (SMOKE)' if a.smoke else ''} | data {env.data} | out {env.out}", flush=True)
    gpu_check.ensure_settings(env)
    if a.stage == 1:
        run_jobs(ctx, [("baseline", "interpatient_v1", s, None) for s in ctx.seeds])
        if ctx.failed:
            sys.exit(f"reproduction runs failed: {ctx.failed}")
        if not gate(ctx)["passed"] and not ctx.smoke:
            print("[STOP] Reproduction failed the pre-registered rule. Experiments stop until this is understood.", flush=True)
            sys.exit(2)
        for v in ("baseline", "focal_g2", "cb_focal"):
            run_jobs(ctx, cv_jobs(ctx, v, ctx.seeds))
    elif a.stage == 2:
        require_gate(ctx)
        for v in ("balanced_sampler", "rr_context"):
            run_jobs(ctx, cv_jobs(ctx, v, ctx.seeds))
    else:
        stage3(ctx)
    if ctx.failed:
        sys.exit(f"{len(ctx.failed)} runs failed (see experiments.json): {ctx.failed[:5]}")
    print(f"[STAGE {a.stage} COMPLETE] {(time.time() - START) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
