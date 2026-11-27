"""Writes docs/RESULTS.md from the committed study reports. Every number is read from a file; none is typed here.

    backend/.venv/bin/python backend/tools/build_results_doc.py

It reads JSON reports only: no data file, no predictions file, no model.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "backend" / "artifacts"
SOURCES = {"selection": "analysis/multidb/selection.json", "validation": "analysis/multidb/validation_report.json",
           "final": "analysis/multidb_final/final_report.json", "data": "multidb_data_report.json", "app": "metrics.json"}
NAMES = {"cnn_bilstm": "CNN-BiLSTM (study retrain)", "rr_fusion": "CNN-BiLSTM + RR fusion", "rr_fusion_attn": "RR fusion + attention",
         "rr_fusion_attn_shead": "RR fusion + attention + S head", "cnn": "CNN only", "lstm": "LSTM only", "resnet1d": "ResNet1D",
         "gbm": "Gradient-boosted trees (GBM)", "stacked_ensemble": "Stacked ensemble"}
ORDER = list(NAMES)  # fixed; secondary models are never ordered by a score
SETS = {"ds2_22_records": "MIT-BIH DS2, all records", "ds2_without_202": "MIT-BIH DS2 without record 202",
        "incart_heldout": "INCART held-out patients (unseen patients)"}
METRICS = [("macro_f1", "Macro-F1", "macro_f1"), ("S_f1", "S F1", "per_class.S.f1"), ("F_f1", "F F1", "per_class.F.f1"),
           ("S_recall", "S recall", "per_class.S.recall"), ("F_recall", "F recall", "per_class.F.recall")]
COLUMNS = [("Macro-F1", "macro_f1", True), ("S recall", "per_class.S.recall", True), ("F recall", "per_class.F.recall", True),
           ("S F1", "per_class.S.f1", False), ("F F1", "per_class.F.f1", False), ("S precision", "per_class.S.precision", False),
           ("F precision", "per_class.F.precision", False), ("V recall", "per_class.V.recall", False), ("Accuracy", "accuracy", False)]


def pick(obj, path: str):
    for key in path.split("."):
        obj = obj[key]
    return obj


def f3(v) -> str:
    return f"{v:.3f}"


def table(head: list[str], rows: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head), *("| " + " | ".join(r) + " |" for r in rows), ""]


def scores(primary: str, entry: dict, majority: dict, deployed: dict | None) -> list[str]:
    def row(name, role, summary):
        cells = []
        for _, path, spread in COLUMNS:
            stat = pick(summary, path)
            cells.append(f3(stat["mean"]) + (f" ±{f3(stat['sd'])}" if spread and stat["sd"] is not None else ""))
        return [name, role, *cells]

    rows = [row(NAMES[primary], "**PRIMARY**", entry[primary])]
    if deployed:
        rows.append(row("Deployed CNN-BiLSTM (the app's model)", "Reference", deployed))
    rows.append(["Majority class (always N)", "Baseline", *(f3(pick(majority, path)) for _, path, _ in COLUMNS)])
    rows += [row(NAMES[m], "SECONDARY", entry[m]) for m in ORDER if m != primary and m in entry]
    return table(["Model", "Role", *(c[0] for c in COLUMNS)], rows)


def verdict(gain: dict) -> str:
    return "improved" if gain["meaningful"] else "lower; not a meaningful improvement" if gain["gain"] < 0 else "did NOT meaningfully improve"


def interval(ci) -> str:
    return f"{f3(ci[0])} to {f3(ci[1])}"


def build(art: Path = ART) -> str:
    load = {k: json.loads((art / v).read_text()) for k, v in SOURCES.items()}
    selection, validation, final, data, app = (load[k] for k in ("selection", "validation", "final", "data", "app"))
    for name in ("selection", "validation", "final"):
        assert selection.get("smoke") is False and load[name].get("smoke") is False, f"{name} is not a real run"
    p = final["primary"]
    assert selection["primary"] == validation["primary"] == p, "the reports disagree about the primary model"
    digest = hashlib.sha256((art / SOURCES["selection"]).read_bytes()).hexdigest()
    assert digest == final["selection_sha256"], "selection.json is not the file the final report was made with"
    assert selection["created_at"] < final["opened_at"], "selection is not dated before the test"
    sets = final["sets"]
    headline = {k: v for k, v in sets.items() if v["primary_vs_deployed"]["headline_comparison"]}
    name = NAMES[p]
    seeds = len(selection["seeds"])

    out = ["# Results: multi-database study (multidb_v1)", "",
           "**Generated file.** `backend/tools/build_results_doc.py` writes it from the committed reports; do not edit it by hand.", "",
           "## Sources", ""]
    out += table(["Content", "File (under `backend/artifacts/`)"], [
        ["Frozen choice: primary model, thresholds, temperatures, review cut-offs", f"`{SOURCES['selection']}`"],
        ["Validation scores", f"`{SOURCES['validation']}`"],
        ["Test scores, intervals, comparisons, review (one evaluation)",
         f"`{SOURCES['final']}` (readable form: `analysis/multidb_final/final_report.md`; lock: `FINAL_TEST_LOCK.json`)"],
        ["Beat and patient counts", f"`{SOURCES['data']}`"],
        ["The app's own model evaluation", f"`{SOURCES['app']}`"]])
    out += ["Protocol and the disclosures fixed before training: [multidb-protocol.md](multidb-protocol.md).", "",
            f"**{name} is the PRIMARY model.** It was chosen on validation (rule: {selection['selection_rule']}). The other study "
            "models are SECONDARY and are listed in a fixed order, never ranked by test score.", "",
            f"Checked by the generator: the three reports name the same primary; the SHA-256 of `selection.json` equals the one "
            f"recorded in the final report; selection is dated {selection['created_at']} and the test was opened {final['opened_at']}. "
            "These dates were written by the study scripts; they are not independent proof.", "",
            "**The app does not run the primary model.** Its live model is the deployed CNN-BiLSTM. No study model, threshold, "
            "temperature or review cut-off is applied to uploaded recordings, and the API cannot serve the primary.", "",
            "## What the final test showed", "",
            "A result counts as improved only if, as declared before the test, the gain exceeds twice the larger seed SD, the paired "
            "patient-bootstrap 95% CI excludes zero, and the gain stays positive with any one patient left out. No multiple-comparison "
            "correction is applied.", ""]

    out.append("- **Macro-F1 against the deployed model:** " + "; ".join(
        f"{'improved' if e['primary_vs_deployed']['macro_f1']['meaningful'] else 'did not meaningfully improve'} on {SETS[k]} "
        f"({f3(e[p]['macro_f1']['mean'])} against {f3(e['deployed_model']['macro_f1']['mean'])})" for k, e in headline.items())
        + ". The comparison is not like for like: the study models trained on more data from more databases. The `cnn_bilstm` row "
          "(deployed architecture, new data) is the control for that.")
    won = [f"{label} on {SETS[k]}" for k, e in headline.items() for key, label, _ in METRICS[1:] if e["primary_vs_deployed"][key]["meaningful"]]
    lower = [f"{label} on {SETS[k]} ({e['primary_vs_deployed'][key]['gain']:+.3f})" for k, e in headline.items()
             for key, label, _ in METRICS if e["primary_vs_deployed"][key]["gain"] < 0]
    out.append("- **S and F:** " + (f"meaningful improvement only for {', '.join(won)}." if won else
               "S and F recall and F1 did NOT meaningfully improve on either comparable test set.")
               + (f" Lower than the deployed model: {', '.join(lower)}." if lower else ""))
    out.append("- **F recall of the primary:** " + "; ".join(
        f"{SETS[k]} {f3(e[p]['per_class']['F']['recall']['mean'])}, 95% CI {interval(e[p]['bootstrap_ci95_patients']['F_recall'])} "
        f"({'includes zero' if e[p]['bootstrap_ci95_patients']['F_recall'][0] <= 0 else 'excludes zero'})" for k, e in sets.items()) + ".")
    val_recall = validation["results"][p]["chosen"]["per_class"]["S"]["recall"]["mean"]
    out.append(f"- **Validation S recall ({f3(val_recall)}) against test:** " + "; ".join(
        f"{SETS[k]} {f3(e[p]['per_class']['S']['recall']['mean'])} "
        f"(validation {'overstated it' if val_recall > e[p]['per_class']['S']['recall']['mean'] else 'did not overstate it'})"
        for k, e in sets.items()) + ".")

    def share(split, cls):
        c = data["splits"][split]["concentration"][cls]
        return f"patient {c['top_patient']} holds {c['top_share']:.1%} of the {c['total']:,} {cls} beats"

    out.append(f"- **S and F depend on a few patients.** DS2: {share('test_ds2', 'S')}; {share('test_ds2', 'F')}. INCART test: "
               f"{share('test_incart', 'S')}; {share('test_incart', 'F')}.")
    out.append("- **Recall after review is an upper bound:** it assumes a reviewer who corrects every reviewed beat. It is shown beside "
               "random review of the same share.")
    out.append("- **DS2 is not untouched.** It had been scored before this study. This is one evaluation within the study, not a first look.")
    out.append("")

    out += ["## Test results", "",
            f"Study models: mean over {seeds} training seeds (±SD where shown). The seeds measure training randomness only. "
            "Read macro-F1 and S/F recall first; accuracy is dominated by N and must be read beside the majority-class row."]
    for key, entry in sets.items():
        P, maj = entry[p], entry["majority_class"]
        fair = entry["primary_vs_deployed"]["headline_comparison"]
        note = ("The fair comparison with the deployed model." if key == "ds2_without_202" else
                "Not a fair comparison with the deployed model: record 202 drove its early stopping." if key == "ds2_22_records" else
                "Unseen patients, not an unseen database: other INCART patients were in training.")
        records = " records" if key.startswith("ds2") else " patients"
        out += ["", f"### {SETS[key]} ({P['n_patients']}{records})", "",
                f"{note} {P['n_beats']:,} beats: " + ", ".join(f"{c} {P['per_class'][c]['support']:,}" for c in "NSVF") + ".", ""]
        out += scores(p, entry, maj, entry["deployed_model"])
        out += ["95% CI, bootstrap over patients:", ""]
        out += table(["Model", "Macro-F1", "S recall", "F recall"], [
            *[[label, *(interval(entry[m]["bootstrap_ci95_patients"][k]) for k in ("macro_f1", "S_recall", "F_recall"))]
              for m, label in ((p, f"{name} (PRIMARY)"), ("deployed_model", "Deployed CNN-BiLSTM"))],
            ["Majority class", *["no interval computed"] * 3]])
        if fair:
            out += ["Primary against the deployed model:", ""]
            out += table(["Metric", "Primary", "Deployed", "Majority class", "Gain", "95% CI of gain", "Smallest gain, one patient out", "Verdict"], [
                [label, f3(pick(P, path)["mean"]), f3(pick(entry["deployed_model"], path)["mean"]), f3(pick(maj, path)),
                 f"{g['gain']:+.3f}", f"{g['bootstrap_ci95'][0]:+.3f} to {g['bootstrap_ci95'][1]:+.3f}",
                 f"{g['leave_one_record_out_min_gain']:+.3f}", verdict(g)]
                for k, label, path in METRICS for g in [entry["primary_vs_deployed"][k]]])
        rows = []
        for model_label, block, cutoff in ((f"{name} (PRIMARY)", P["review"], "ranked within set"),
                                           (f"{name} (PRIMARY)", P["review_with_validation_cutoff"], "fixed on validation"),
                                           ("Deployed CNN-BiLSTM", entry["deployed_model"]["review"], "ranked within set")):
            rows += [[f"{float(b):.0%}", cutoff, f"{st['reviewed_share']:.1%}", model_label, c, f3(st[c]["model_recall_all_beats"]),
                      f3(st[c]["recall_after_perfect_review"]), f3(st[c]["recall_after_random_review"])]
                     for b, st in block.items() for c in "SF" if st[c]]
        rows += [["—", "—", "—", "Majority class", c, f3(maj["per_class"][c]["recall"]), "not defined (no confidence score)", "not defined"]
                 for c in "SF"]
        out += ["Selective review. **After review is an upper bound (perfect reviewer).**", ""]
        out += table(["Budget", "Cut-off", "Reviewed", "Model", "Class", "Model recall", "After review (upper bound)", "After random review"], rows)
        out += [f"Calibration, expected calibration error (15 bins): primary {P['calibration_raw']['ece_15']:.4f} raw, "
                f"{P['calibration_temperature_scaled']['ece_15']:.4f} temperature-scaled; deployed {entry['deployed_model']['calibration_raw']['ece_15']:.4f} raw."]

    out += ["", "## Validation results (not test results)", "",
            "Validation drove early stopping and chose the primary from all the candidates, so these figures are optimistic. "
            "The deployed model was not scored on this data.", ""]
    out += scores(p, {m: r["chosen"] for m, r in validation["results"].items()}, validation["majority_class"], None)

    deployed = sets["ds2_without_202"]["deployed_model"]
    same = abs(deployed["macro_f1"]["mean"] - app["test"]["macro"]["f1"]) < 1e-9 and deployed["n_beats"] == app["test"]["n_windows"]
    out += ["## Consistency checks", "",
            f"- Deployed model on DS2 without record 202: the final report and `metrics.json` "
            f"{'agree on beat count and macro-F1 (within 1e-9)' if same else 'DISAGREE on beat count or macro-F1'}."]
    for key, entry in sets.items():
        for cls in "SF":
            summed, true = sum(r[cls]["support"] for r in entry[p]["per_patient"]), entry[p]["per_class"][cls]["support"]
            if summed != true:
                out.append(f"- **Known defect, {SETS[key]}:** the per-patient {cls} supports in `final_report.json` add to {summed}, "
                           f"but the set has {true}.")
    out += ["- Cause of the per-patient gaps: `backend/scripts/evaluate_multidb.py`, function `per_patient`, applies `int()` to a "
            "floating-point mean over seeds, which can round a whole number down. Pooled per-class scores are computed separately and "
            "are not affected. Patient shares above come from the data report, which counts beats directly. The study scripts and the "
            "final report were left unchanged.", "",
            "## Limitations", "",
            "- One fixed split. Seed SD does not measure how much results would move with other patients.",
            "- Few patients carry most S and F beats; the patient-bootstrap intervals for S and F are wide.",
            f"- INCART F is a very small sample ({sets['incart_heldout'][p]['per_class']['F']['support']} beats).",
            "- DS2 was scored before the study, and the RR variants' stored scores were known when RR fusion was chosen.",
            "- INCART is not an external database: other INCART patients were used in training and validation.",
            "- Offline method: RR features use the following beat and the whole-record median, and are computed from reference beat "
            "positions after beats outside N/S/V/F are removed. No beat-detector error is included.",
            "- SVDB has no record-to-subject map (one record is assumed to be one subject), and overlap with MIT-BIH subjects cannot be checked.",
            "- Windows are centred on each database's own annotations, which are not aligned the same way ([multidb-data-report.md](multidb-data-report.md)).",
            "- The comparison with the deployed model mixes the effect of more training data with the effect of the method.",
            "- Window-level results on retrospective recordings. Not a diagnosis, and no patient-level claim.",
            "- Packaging did not retrain or re-evaluate anything. This page reports the committed files; it does not re-verify the "
            "evaluation that produced them.", ""]
    return "\n".join(out)


if __name__ == "__main__":
    (ROOT / "docs" / "RESULTS.md").write_text(build())
    print("wrote docs/RESULTS.md")
