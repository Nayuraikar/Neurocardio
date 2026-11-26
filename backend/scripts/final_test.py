"""THE ONE FINAL TEST EVALUATION. Run by hand, once, after choosing a configuration.

  python -u final_test.py --config baseline [--thresholds] [--ensemble 1|3|5] \
         --test-data /kaggle/input/<dataset>/neurocardio_test_SEALED.npz --confirm EVALUATE-TEST-ONCE

Order of work
  1. Train the chosen configuration on ds1_ds2 (DS1; DS2 is never used for fitting).
  2. Fit temperature (and thresholds, if chosen) on the inner validation records only.
  3. Write a lock file, open the sealed file, run inference once, save the predictions at once.
  4. Report DS2 (22 records) and its 21-record subset, compared with the deployed model and the majority class.
If it is interrupted after step 3, running it again with the SAME arguments rebuilds the report from the
saved predictions without opening the sealed file. A different configuration is refused.
--smoke rehearses with development records standing in as "test"; it refuses the sealed file.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analysis_common as A  # noqa: E402
import experiments as E  # noqa: E402
import train as T  # noqa: E402

CONFIRM = "EVALUATE-TEST-ONCE"
SEEDS = [42, 11, 22, 33, 44]


def temperature(probs, y) -> float:
    lp = np.log(np.clip(probs, 1e-12, 1))

    def nll(t):
        z = lp / np.exp(t)
        z -= z.max(1, keepdims=True)
        return -(z[np.arange(len(y)), y] - np.log(np.exp(z).sum(1))).mean()
    return float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x))


def scaled(probs, t):
    z = np.log(np.clip(probs, 1e-12, 1)) / t
    z -= z.max(1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(1, keepdims=True)


def calibration(probs, y, bins=15) -> dict:
    conf, ok = probs.max(1), (probs.argmax(1) == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    ece = sum(m.mean() * abs(ok[m].mean() - conf[m].mean())
              for lo, hi in zip(edges[:-1], edges[1:]) if (m := (conf > lo) & (conf <= hi)).any())
    return {"ece_15": float(ece), "brier": float(((probs - np.eye(4)[y]) ** 2).sum(1).mean()),
            "nll": float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1)).mean())}


def per_record(y, pred, rec):
    out = []
    for r in sorted(set(rec)):
        c = A.cm(y[rec == r], pred[rec == r])
        out.append({"record": r, "n": int(c.sum()),
                    **{k: {"support": int(c[i].sum()), "true_positive": int(c[i, i]), "false_positive": int(c[:, i].sum() - c[i, i]),
                           "recall": None if c[i].sum() == 0 else float(c[i, i] / c[i].sum())}
                       for i, k in ((1, "S"), (3, "F"))}})
    return out


def bootstrap(y, pred, rec, n=2000):
    order = sorted(set(rec))
    cms = A.record_cms(y, pred, rec, order)
    picks = np.random.default_rng(20241127).integers(0, len(order), (n, len(order)))
    vals = {k: [fn(A.from_cm(cms[i].sum(0))) for i in picks] for k, fn in A.METRICS.items()}
    return {k: [float(v) for v in np.percentile(x, [2.5, 97.5])] for k, x in vals.items()}


def report(out: Path, meta: dict) -> None:
    with np.load(out / "predictions.npz", allow_pickle=False) as z:
        probs, y, rec, base = z["probs"].astype(np.float64), z["y"], z["record"].astype(str), z["baseline_probs"].astype(np.float64)
    w, t = np.array(meta["threshold_scales"]), meta["temperature"]
    pred = (probs * w).argmax(1)
    res = {**meta, "sets": {}}
    sets = {"DS2_all_22_records": np.ones(len(y), bool)}
    if (rec != "202").any() and (rec == "202").any():
        sets["DS2_without_202_21_records"] = rec != "202"
    for name, m in sets.items():
        entry = {"model": A.report(y[m], pred[m]), "per_record_S_F": per_record(y[m], pred[m], rec[m]),
                 "bootstrap_ci95_record_level": bootstrap(y[m], pred[m], rec[m]),
                 "calibration_raw": calibration(probs[m], y[m]), "calibration_temperature_scaled": calibration(scaled(probs[m], t), y[m]),
                 "majority_class": A.report(y[m], np.zeros(m.sum(), int))}
        if np.isfinite(base[m]).all():                     # deployed model: same records, same windows
            bp = base[m].argmax(1)
            order = sorted(set(rec[m]))
            entry["deployed_model"] = A.report(y[m], bp)
            cmp = A.compare([A.record_cms(y[m], pred[m], rec[m], order)], [A.record_cms(y[m], bp, rec[m], order)])
            entry["vs_deployed_model"] = cmp
            entry["vs_deployed_note"] = "Single paired comparison: the seed-SD part of the rule does not apply; CI and leave-one-record-out do."
        res["sets"][name] = entry
    T.write_json(out / "final_report.json", res)
    lines = [f"# Final test evaluation{' - SMOKE REHEARSAL, NOT TEST RESULTS' if meta['smoke'] else ''}",
             f"Config: {meta['config']} | thresholds: {meta['thresholds']} | ensemble: {meta['ensemble']} | opened {meta['opened_at']}", "",
             "The test set had been scored before this track (nine earlier variants and the audit). This is one evaluation within this track."]
    for name, e in res["sets"].items():
        lines += ["", f"## {name}", "| Model | accuracy | macro-F1 | " + " | ".join(f"{c} P / R / F1" for c in A.CLASSES) + " |", "|" + "---|" * 7]
        for label, key in (("chosen", "model"), ("deployed baseline", "deployed_model"), ("majority class", "majority_class")):
            if key in e:
                r = e[key]
                lines.append(f"| {label} | {r['accuracy']:.3f} | {r['macro_f1']:.3f} | " + " | ".join(
                    f"{r['per_class'][c]['precision']:.3f} / {r['per_class'][c]['recall']:.3f} / {r['per_class'][c]['f1']:.3f}" for c in A.CLASSES) + " |")
        lines.append(f"\nECE: raw {e['calibration_raw']['ece_15']:.4f}, temperature-scaled {e['calibration_temperature_scaled']['ece_15']:.4f}")
        if "vs_deployed_model" in e:
            for k in ("S_f1", "F_f1", "macro_f1"):
                v = e["vs_deployed_model"][k]
                lines.append(f"- {k}: gain {v['gain']:+.3f}, 95% CI [{v['bootstrap_ci95'][0]:+.3f}, {v['bootstrap_ci95'][1]:+.3f}] -> "
                             + ("meaningful improvement" if v["meaningful"] else "NOT a meaningful improvement"))
        lines += ["", "Per record (S, F): true positives / support, false positives"]
        for r in e["per_record_S_F"]:
            if r["S"]["support"] or r["F"]["support"] or r["S"]["false_positive"] or r["F"]["false_positive"]:
                lines.append(f"- {r['record']}: S {r['S']['true_positive']}/{r['S']['support']} (fp {r['S']['false_positive']}), "
                             f"F {r['F']['true_positive']}/{r['F']['support']} (fp {r['F']['false_positive']})")
    (out / "final_report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, choices=[*T.VARIANTS, "combined"])
    ap.add_argument("--thresholds", action="store_true")
    ap.add_argument("--ensemble", type=int, choices=[1, 3, 5], default=1)
    ap.add_argument("--test-data", help="neurocardio_test_SEALED.npz")
    ap.add_argument("--confirm", default="")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume-from", nargs="*", default=[])
    T.add_path_args(ap)
    a = ap.parse_args()
    env = T.Env(a.data, a.splits, a.out, a.model_config, a.allow_cpu)
    if a.smoke:
        if a.test_data:
            sys.exit("--smoke never opens a test file; remove --test-data")
        env = T.Env(env.data, env.splits, env.out / "smoke", env.model_config, a.allow_cpu)
    else:
        if a.confirm != CONFIRM:
            sys.exit(f"This opens the test set. Re-run with --confirm {CONFIRM}")
        E.restore(env, a.resume_from)
    env.out.mkdir(parents=True, exist_ok=True)
    out = env.out / "final"
    lock = out / "FINAL_TEST_LOCK.json"
    chosen = {"config": a.config, "thresholds": a.thresholds, "ensemble": a.ensemble, "smoke": a.smoke}

    logged = next((r for r in T.read_registry(env)["runs"] if r.get("kind") == "final_test"), None)
    if not a.smoke and (lock.exists() or logged):
        prior = json.loads(lock.read_text()) if lock.exists() else logged
        if {k: prior.get(k) for k in chosen} != chosen:
            sys.exit(f"REFUSED: the test set was already opened for {({k: prior.get(k) for k in chosen})}.")
        if (out / "predictions.npz").exists():
            print("Test predictions already exist. Rebuilding the report only; the sealed file is not opened.")
            report(out, prior)
            return
        sys.exit("REFUSED: the lock exists but no saved predictions were found. Attach the earlier output. "
                 "If inference never completed, record why before removing the lock by hand.")

    spec = T.VARIANTS.get(a.config)
    if not a.smoke:
        gate = env.out / "analysis/gate.json"
        if not gate.exists() or not json.loads(gate.read_text())["passed"] or not (env.out / "analysis/ablation.json").exists():
            sys.exit("Stages 1-3 must be complete first (gate.json passed and ablation.json present in the output folder).")
        if a.config == "combined":
            spec = json.loads((env.out / "analysis/combined_spec.json").read_text())["spec"]
            if not spec:
                sys.exit("no combined variant was defined by stage 3")
    spec = spec or {**T.VARIANTS["focal_g2"], "context": "rr"}

    # 1. train members on DS1
    seeds = SEEDS[:a.ensemble]
    tag = "__smoke" if a.smoke else ""
    ids = []
    for s in seeds:
        if a.smoke:
            T.train(env, a.config, "ds1_ds2", s, spec=spec, max_epochs=1, tag=tag, limit=2048)
        else:
            cmd = [sys.executable, "-u", str(HERE / "train.py"), "--variant", a.config, "--split", "ds1_ds2", "--seed", str(s),
                   "--spec-json", json.dumps(spec), "--data", str(env.data), "--splits", str(env.splits), "--out", str(env.out),
                   "--model-config", str(env.model_config)] + (["--allow-cpu"] if env.allow_cpu else [])
            subprocess.run(cmd, check=True)
        ids.append(T.run_id_for("ds1_ds2", a.config, s, tag))

    # 2. fit temperature / thresholds on inner validation only
    va = []
    for rid in ids:
        with np.load(env.runs / rid / "validation_predictions.npz") as z:
            va.append((z["probs"].astype(np.float64), z["y"]))
    vp, vy = np.mean([v[0] for v in va], axis=0), va[0][1]
    meta = {**chosen, "spec": spec, "member_runs": ids, "temperature": temperature(vp, vy),
            "threshold_scales": A.fit_thresholds(vp, vy).tolist() if a.thresholds else [1.0] * 4,
            "fitted_on": "ds1_ds2 inner validation", "opened_at": T.now()}

    # 3. open the test data once
    if a.smoke:
        data = env.dataset()
        pseudo = ["108", "114", "124", "201"]
        m = np.isin(data["record"].astype(str), pseudo)
        test = {k: data[k][m] for k in ("X", "y", "record", "rr")}
        test["baseline_probs"] = np.full((int(m.sum()), 4), np.nan, np.float32)
        test_sha = "smoke: development records"
    else:
        path = Path(a.test_data) if a.test_data else T.find_data("neurocardio_test_SEALED.npz")
        if not path.exists():
            sys.exit(f"sealed test file not found: {path}")
        test_sha = T.sha256(path)
    import keras
    T.setup_tf(env)
    models = [keras.models.load_model(env.runs / rid / "model.keras", compile=False) for rid in ids]   # load before locking
    entries = {r["run_id"]: r for r in T.read_registry(env)["runs"]}
    meta["test_file_sha256"] = test_sha
    T.write_json(lock, {**meta, "state": "opened"})
    T.register(env, {"run_id": f"final_test{tag}", "kind": "final_test", "status": "running", **meta})
    if not a.smoke:
        with np.load(path, allow_pickle=False) as z:
            test = {k: z[k] for k in ("X", "y", "record", "rr", "baseline_probs")}
    probs = []
    for rid, model in zip(ids, models):
        e = entries[rid]
        x = test["X"][..., None]
        if spec["context"] == "rr":
            x = [x, (test["rr"] - np.array(e["rr_mean"], np.float32)) / np.array(e["rr_std"], np.float32)]
        probs.append(model.predict(x, batch_size=512, verbose=0))
    np.savez_compressed(out / "predictions.npz", probs=np.mean(probs, axis=0).astype(np.float32),
                        member_probs=np.stack(probs).astype(np.float32), y=test["y"].astype(np.int64),
                        record=test["record"], baseline_probs=test["baseline_probs"])
    T.write_json(lock, {**meta, "state": "predicted"})
    report(out, meta)
    T.register(env, {"run_id": f"final_test{tag}", "kind": "final_test", "status": "completed", "finished_at": T.now(), **meta,
                     **({"note": "smoke rehearsal; not a test result"} if a.smoke else {})})


if __name__ == "__main__":
    main()
