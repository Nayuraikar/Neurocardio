"""THE ONE FINAL TEST EVALUATION (multi-database study). Run by hand, once, after evaluate_multidb.py.

  python -u final_test.py --ds2 .../multidb_test_ds2_SEALED.npz --incart .../multidb_test_incart_SEALED.npz \
         --confirm EVALUATE-TEST-ONCE [--secondary-all]

Nothing is trained or tuned here. Models are the validation runs; temperature, thresholds, review cut-offs, stacker
and the primary model come from analysis/multidb/selection.json.

Order of work
  1. Load every model that will be scored, including the deployed model (SHA-256 checked).
  2. Check both sealed files against the SHA-256 recorded in the split file, write a lock, open them, predict,
     save all predictions at once.
  3. Report DS2 (22 records), DS2 without record 202 (21; the only DS2 set on which the deployed model is comparable,
     because 202 drove its early stopping) and the held-out INCART patients.
Rerunning with the same arguments after step 2 rebuilds the report from saved predictions without opening the
sealed files. Anything else is refused. The lock is a safeguard, not tamper-proof: keep the output folder.
--secondary-all also scores every other frozen candidate in the same single pass, labelled secondary. It is part of
the lock, so decide before the first run. --smoke uses validation patients as stand-ins and refuses sealed files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analysis_common as A  # noqa: E402
import evaluate_multidb as EV  # noqa: E402
import experiments as E  # noqa: E402
import models_multidb as M  # noqa: E402
import split_checks  # noqa: E402
import train as T  # noqa: E402
import train_multidb as TM  # noqa: E402

CONFIRM = "EVALUATE-TEST-ONCE"
FILES = {"ds2": "multidb_test_ds2_SEALED.npz", "incart": "multidb_test_incart_SEALED.npz"}
TITLES = {"ds2_22_records": "MIT-BIH DS2, 22 records", "ds2_without_202": "MIT-BIH DS2 without record 202, 21 records",
          "incart_heldout": "INCART held-out patients (other INCART patients were in training)"}


def deployed_model():
    path = T.CODE / "models/neurocardio_final.keras"
    want = (T.CODE / "models/neurocardio_final.keras.sha256").read_text().split()[0]
    assert T.sha256(path) == want, "the deployed model file does not match its pinned SHA-256"
    import keras
    return keras.models.load_model(path, compile=False)


def sets_from(preds: dict):
    rec = preds["ds2"]["record"].astype(str)
    yield "ds2_22_records", preds["ds2"], np.ones(len(rec), bool)
    if (rec == "mitdb/202").any():
        yield "ds2_without_202", preds["ds2"], rec != "mitdb/202"
    yield "incart_heldout", preds["incart"], np.ones(len(preds["incart"]["y"]), bool)


def score_candidate(name, sel, data, mask, y, patient, db) -> tuple[dict, list]:
    frozen = [sel["candidates"][name]["frozen"][str(s)] for s in sel["seeds"]]
    raw = [data[f"probs__{name}__{s}"][mask].astype(np.float64) for s in sel["seeds"]]
    raw = [p / p.sum(1, keepdims=True) for p in raw]
    preds = [(p * np.array(f["thresholds"])).argmax(1) for p, f in zip(raw, frozen)]
    cal = [A.scaled(p, f["temperature"]) for p, f in zip(raw, frozen)]
    out = EV.summarize(y, patient, db, preds, raw, cal)
    # the review cut-off a deployment would have to use: fixed on validation, so the reviewed share can differ from the budget
    out["review_with_validation_cutoff"] = {
        b: EV.average([EV.review_stats(y, p, c.max(1), 0, c.max(1) <= f["review_confidence_cutoffs"][b])
                       for p, c, f in zip(preds, cal, frozen)]) for b in frozen[0]["review_confidence_cutoffs"]}
    out["rule"] = sel["candidates"][name]["rule"]
    out["risk_coverage"] = EV.risk_coverage(y, preds, cal)
    return out, preds


def report(out: Path, meta: dict, sel: dict) -> None:
    preds = {}
    for key in FILES:
        with np.load(out / f"predictions_{key}.npz", allow_pickle=False) as z:
            preds[key] = {k: z[k] for k in z.files}
    primary = sel["primary"]
    result = {**meta, "sets": {}}
    L = ["# Final test evaluation" + (" - SMOKE REHEARSAL, NOT TEST RESULTS" if meta["smoke"] else ""),
         f"Primary model (chosen on validation): **{primary}**, rule `{sel['candidates'][primary]['rule']}`, "
         f"seeds {sel['seeds']} (mean±SD). Opened {meta['opened_at']}.", "",
         "Disclosure: DS2 had been scored before this study (nine earlier variants and an audit). This is one evaluation "
         "within this study, not a first look."]
    for set_name, data, mask in sets_from(preds):
        y, patient, db = data["y"][mask], data["patient"][mask].astype(str), data["db"][mask].astype(str)
        order = sorted(set(patient))
        entry, pred_by = {}, {}
        names = meta["scored"]
        for name in names:
            entry[name], pred_by[name] = score_candidate(name, sel, data, mask, y, patient, db)
        dep_raw = data["probs__deployed"][mask].astype(np.float64)
        dep_pred = dep_raw.argmax(1)
        entry["deployed_model"] = EV.summarize(y, patient, db, [dep_pred], [dep_raw], [dep_raw])
        entry["deployed_model"].pop("calibration_temperature_scaled")
        entry["majority_class"] = A.report(y, np.zeros(len(y), int))
        cms = lambda ps: [A.record_cms(y, p, patient, order) for p in ps]
        comparable = set_name != "ds2_22_records"                  # 202 was one of the deployed model's validation records
        cmp = A.compare(cms(pred_by[primary]), cms([dep_pred]), EV.RESAMPLES)
        entry["primary_vs_deployed"] = {**cmp, "headline_comparison": comparable}
        entry["primary_vs_majority_macro_f1_gain"] = entry[primary]["macro_f1"]["mean"] - entry["majority_class"]["macro_f1"]
        result["sets"][set_name] = entry

        P = entry[primary]
        L += ["", f"## {TITLES[set_name]}", f"{P['n_beats']:,} beats, {P['n_patients']} patients; "
              + ", ".join(f"{c} {P['per_class'][c]['support']:,}" for c in A.CLASSES)]
        maj = entry["majority_class"]
        rows = {f"{primary} (primary)": P, "deployed model": entry["deployed_model"]}
        rows["majority class"] = {"macro_f1": {"mean": maj["macro_f1"], "sd": None}, "accuracy": {"mean": maj["accuracy"], "sd": None},
                                  "per_class": {c: {k: {"mean": maj["per_class"][c][k], "sd": None} for k in ("precision", "recall", "f1")}
                                                for c in A.CLASSES}}
        rows.update({f"{n} (secondary)": entry[n] for n in names if n != primary})
        L += EV.table(rows)
        L += ["", "95% CI over patients, primary: " + "; ".join(
            f"{k} [{v[0]:.3f}, {v[1]:.3f}]" for k, v in P["bootstrap_ci95_patients"].items() if k in ("macro_f1", "S_f1", "S_recall", "F_f1", "F_recall"))]
        L += ["95% CI over patients, deployed: " + "; ".join(
            f"{k} [{v[0]:.3f}, {v[1]:.3f}]" for k, v in entry["deployed_model"]["bootstrap_ci95_patients"].items()
            if k in ("macro_f1", "S_f1", "S_recall", "F_f1", "F_recall"))]
        L += ["", "Primary versus deployed model" + ("" if comparable else
              " (NOT a fair comparison on this set: record 202 drove the deployed model's early stopping; use the 21-record set)") + ":"]
        for k, label in (("S_f1", "S F1"), ("F_f1", "F F1"), ("macro_f1", "macro-F1"), ("S_recall", "S recall"), ("F_recall", "F recall")):
            v = cmp[k]
            verdict = ("improved (meets the pre-declared rule)" if v["meaningful"] else
                       "lower" if v["gain"] < 0 else "did NOT meaningfully improve")
            L.append(f"- {label}: {v['gain']:+.3f}, 95% CI [{v['bootstrap_ci95'][0]:+.3f}, {v['bootstrap_ci95'][1]:+.3f}], "
                     f"smallest gain leaving one patient out {v['leave_one_record_out_min_gain']:+.3f} -> {verdict}")
        L += ["", f"Calibration (ECE, 15 bins): primary raw {P['calibration_raw']['ece_15']:.4f}, temperature-scaled "
              f"{P['calibration_temperature_scaled']['ece_15']:.4f}; deployed raw {entry['deployed_model']['calibration_raw']['ece_15']:.4f}.",
              "", "Selective review (primary). Recall 'after review' assumes a perfect reviewer: an upper bound.",
              "| budget | cut-off | reviewed | class | model recall | after review | after random review | missed beats sent to review |",
              "|---|---|---|---|---|---|---|---|"]
        for label, block in (("rank within set", P["review"]), ("fixed on validation", P["review_with_validation_cutoff"])):
            for b, st in block.items():
                for c in ("S", "F"):
                    if st[c]:
                        L.append(f"| {float(b):.0%} | {label} | {st['reviewed_share']:.1%} | {c} | {st[c]['model_recall_all_beats']:.3f} | "
                                 f"{st[c]['recall_after_perfect_review']:.3f} | {st[c]['recall_after_random_review']:.3f} | "
                                 f"{st[c]['missed_beats_sent_to_review']:.3f} |")
        L += ["", "Per patient: true positives / support (false positives); primary is the mean over seeds",
              "| patient | S primary | S deployed | F primary | F deployed |", "|---|---|---|---|---|"]
        dep_rows = {r["patient"]: r for r in entry["deployed_model"]["per_patient"]}
        cell = lambda r, c: f"{r[c]['true_positive']:.0f}/{r[c]['support']} ({r[c]['false_positive']:.0f})"
        for r in P["per_patient"]:
            d = dep_rows[r["patient"]]
            if any(x[c]["support"] or x[c]["false_positive"] >= 1 for x in (r, d) for c in "SF"):
                L.append(f"| {r['patient']} | {cell(r, 'S')} | {cell(d, 'S')} | {cell(r, 'F')} | {cell(d, 'F')} |")
        suffix = "_smoke" if meta["smoke"] else ""
        EV.plot_curves(out / f"risk_coverage_{set_name}{suffix}.png", {n: entry[n]["risk_coverage"] for n in names}, TITLES[set_name])
    T.write_json(out / "final_report.json", result)
    (out / "final_report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ds2", help=FILES["ds2"])
    ap.add_argument("--incart", help=FILES["incart"])
    ap.add_argument("--confirm", default="")
    ap.add_argument("--secondary-all", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume-from", nargs="*", default=[])
    T.add_path_args(ap)
    a = ap.parse_args()
    if a.smoke and (a.ds2 or a.incart):
        sys.exit("--smoke never opens a test file; remove --ds2/--incart")
    if not a.smoke and a.confirm != CONFIRM:
        sys.exit(f"This opens both test sets. Re-run with --confirm {CONFIRM}")
    env = TM.make_env(a, a.smoke)
    if not a.smoke:
        E.restore(env, a.resume_from)
    split, split_sha = TM.load_split(env)
    sel_path = env.out / "analysis/multidb/selection.json"
    if not sel_path.exists():
        sys.exit(f"{sel_path} not found: run run_multidb.py to completion (it ends with evaluate_multidb.py)")
    sel = json.loads(sel_path.read_text())
    assert sel["smoke"] == a.smoke, "selection.json comes from the other mode (smoke vs real)"
    assert sel["split_sha256"] == split_sha and sel["data_sha256"] == split["data_sha256"], "selection was made on another split or data"
    out = env.out / ("final_multidb" if not a.smoke else "final")
    out.mkdir(parents=True, exist_ok=True)
    lock = out / "FINAL_TEST_LOCK.json"
    scored = [sel["primary"]] + ([n for n in sel["candidates"] if n != sel["primary"]] if a.secondary_all else [])
    chosen = {"study": TM.SPLIT, "primary": sel["primary"], "scored": scored, "smoke": a.smoke,
              "selection_sha256": hashlib.sha256(sel_path.read_bytes()).hexdigest()}

    logged = next((r for r in T.read_registry(env)["runs"] if r.get("kind") == "final_test" and r.get("study") == TM.SPLIT
                   and not r.get("smoke")), None)
    if not a.smoke and (lock.exists() or logged):
        prior = json.loads(lock.read_text()) if lock.exists() else logged
        if {k: prior.get(k) for k in chosen} != chosen:
            sys.exit(f"REFUSED: the test sets were already opened with {({k: prior.get(k) for k in chosen})}.")
        if all((out / f"predictions_{k}.npz").exists() for k in FILES):
            print("Test predictions already exist. Rebuilding the report only; the sealed files are not opened.")
            report(out, prior, sel)
            return
        sys.exit("REFUSED: a lock exists but saved predictions are missing. Attach the earlier output. "
                 "If inference never completed, record why before removing the lock by hand.")

    # 1. load every model before anything is opened
    paths = {}
    if not a.smoke:
        for key, given in (("ds2", a.ds2), ("incart", a.incart)):
            paths[key] = Path(given) if given else T.find_data(FILES[key])
            if not paths[key].exists():
                sys.exit(f"sealed file not found: {paths[key]}")
            assert T.sha256(paths[key]) == split["data_sha256"][FILES[key]], f"{paths[key]} is not the sealed file recorded in the split"
    T.setup_tf(env)
    import joblib
    import keras
    base_needed = sorted({m for n in scored for m in (M.ALL_MODELS if n == EV.STACK else [n])}, key=M.ALL_MODELS.index)
    registry = {r["run_id"]: r for r in T.read_registry(env)["runs"]}
    loaded = {}
    for name in base_needed:
        for s in sel["seeds"]:
            rid = sel["members"][name][str(s)]
            assert TM.completed(env, rid), f"run {rid} is missing from this output folder"
            f = TM.model_file(env, rid, name)
            loaded[name, s] = (joblib.load(f) if name == "gbm" else keras.models.load_model(f, compile=False), registry[rid])
    deployed = deployed_model()
    print(f"loaded {len(loaded)} study models and the deployed model", flush=True)

    # 2. open once
    meta = {**chosen, "opened_at": T.now()}
    T.write_json(lock, {**meta, "state": "opened"})
    rid_final = f"final_test_{TM.SPLIT}" + ("__smoke" if a.smoke else "")
    entry = {"run_id": rid_final, "kind": "final_test", "status": "running", **meta,
             **({"note": "smoke rehearsal; not a test result"} if a.smoke else {})}
    T.register(env, entry)
    if a.smoke:
        dev = TM.load_dev(env, split)
        va = TM.part(dev, split, "validation", TM.SMOKE_VAL_STRIDE)
        ic = va["db"] == "incartdb"
        fake = {"ds2": ~ic, "incart": ic}
        tests = {k: {key: v[m] for key, v in va.items()} for k, m in fake.items()}
    else:
        tests = {}
        for key, path in paths.items():
            with np.load(path, allow_pickle=False) as z:
                tests[key] = {k: z[k] for k in ("X", "y", "rr", "record", "patient", "db")}
            split_checks.check_arrays(split, tests[key]["record"].astype(str), tests[key]["patient"].astype(str), (f"test_{key}",))
    for key, d in tests.items():
        probs = {}
        for (name, s), (model, run) in loaded.items():
            probs[name, s] = TM.predict_proba(name, model, d["X"], d["rr"], run.get("rr_mean"), run.get("rr_std"))[0]
            print(f"  {key}: {name} seed {s} predicted", flush=True)
        if EV.STACK in scored:
            for s in sel["seeds"]:
                feats = EV.stack_features([probs[n, s].astype(np.float64) / probs[n, s].sum(1, keepdims=True) for n in sel["stack_member_order"]])
                probs[EV.STACK, s] = EV.apply_stacker(sel["stackers"][str(s)], feats).astype(np.float32)
        dep = deployed.predict(d["X"][..., None], batch_size=512, verbose=0).astype(np.float32)
        np.savez_compressed(out / f"predictions_{key}.npz", y=d["y"].astype(np.int64), record=d["record"], patient=d["patient"],
                            db=d["db"], probs__deployed=dep, **{f"probs__{n}__{s}": p for (n, s), p in probs.items()})
    T.write_json(lock, {**meta, "state": "predicted"})

    # 3. report
    report(out, meta, sel)
    T.register(env, {**entry, "status": "completed", "finished_at": T.now()})


if __name__ == "__main__":
    main()
