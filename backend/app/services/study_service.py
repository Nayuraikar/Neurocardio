"""Reads evaluation artifacts. Nothing here computes a model score; a missing artifact is reported as missing."""
from __future__ import annotations

import ast
import csv
import json
from pathlib import Path

from app.core.config import BACKEND_DIR, SPLITS_DIR

STUDY_FILES = {"selection": "selection.json", "validation": "validation_report.json", "final": "final_report.json"}
NOT_RUN = {"selection": "Threshold selection not run yet.", "validation": "Study validation not run yet.",
           "final": "Final evaluation not run yet."}
STACK = "stacked_ensemble"


def read_json(path: Path) -> tuple[dict | None, str | None]:
    """-> (content, problem). A missing file is not a problem; an unreadable one is."""
    if not path.exists():
        return None, None
    try:
        return json.loads(path.read_text()), None
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"{path.name} could not be read: {exc.__class__.__name__}"


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle))


def _number(value: str):
    try:
        return float(value)
    except ValueError:
        return value


def split_facts(metrics: dict, results_dir: Path) -> dict:
    """Facts about the loaded model's split, checked here where they can be."""
    records = metrics["split"]["records"]
    test, other = set(records["test"]), set(records["train"]) | set(records["validation"])
    selection, _ = read_json(results_dir / "split_selection.json")
    equals_de_chazal = None
    multidb, _ = read_json(SPLITS_DIR / "multidb_v1.json")
    if multidb:
        ds2 = {r.split("/")[1] for r in multidb["test_ds2"]}
        equals_de_chazal = test == ds2 - {"202"}
    return {"records_disjoint": not (test & other), "test_equals_ds2_without_202": equals_de_chazal,
            "split_file_statement": selection and {k: selection.get(k) for k in
                                                   ("method", "selection_rule", "assignments_drawn", "uses_model_predictions")}}


def model_metrics(state) -> dict:
    data, problem = read_json(state.artifacts_dir / "metrics.json")
    if data is None:
        return {"available": False, "problem": problem, "message": "Model evaluation has not been generated."}
    stale = []
    if state.classifier is not None and data["model"]["sha256"] != state.classifier.sha256:
        stale.append("the loaded model file differs from the evaluated one")
    if data["model"]["config_sha256"] != state.config_sha256:
        stale.append("the preprocessing config differs from the evaluated one")
    results = state.results_dir
    return {"available": True, "stale": stale, "metrics": data, "split_facts": split_facts(data, results),
            "variant_test_scores": [{k: _number(v) for k, v in row.items()} for row in read_csv(results / "experiment_table_test.csv")],
            "minority_concentration": [{k: (v if k in ("split", "class", "top_record") else _number(v)) for k, v in row.items()}
                                       for row in read_csv(results / "minority_concentration_by_split.csv")]}


def study_models() -> list[str]:
    """Model names as the study code defines them (read, not imported)."""
    tree = ast.parse((BACKEND_DIR / "scripts" / "models_multidb.py").read_text())
    values: dict = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ("NEURAL", "ALL_MODELS"):
            if isinstance(node.value, ast.List):
                values[node.targets[0].id] = ast.literal_eval(node.value)
            else:  # ALL_MODELS = NEURAL + ["gbm"]
                values[node.targets[0].id] = values[node.value.left.id] + ast.literal_eval(node.value.right)
    return [*values["ALL_MODELS"], STACK]


def study_artifact(artifacts_dir: Path, kind: str) -> dict:
    """-> {available, message, data, path}. Smoke-run outputs are refused: they are not results."""
    name = STUDY_FILES[kind]
    for path in (artifacts_dir / "multidb" / name, artifacts_dir / name):
        data, problem = read_json(path)
        if problem:
            return {"available": False, "message": problem, "data": None, "error": True}
        if data is not None:
            if data.get("smoke") is not False:
                return {"available": False, "data": None, "error": True,
                        "message": f"{name} is marked as a smoke run (or not marked as a real run), so it is not shown."}
            return {"available": True, "message": None, "data": data, "error": False,
                    "path": path.relative_to(artifacts_dir).as_posix()}
    return {"available": False, "message": NOT_RUN[kind], "data": None, "error": False}


def study(artifacts_dir: Path) -> dict:
    data_report, problem = read_json(artifacts_dir / "multidb_data_report.json")
    return {"models": study_models(), "data_report": data_report, "data_report_problem": problem,
            **{kind: study_artifact(artifacts_dir, kind) for kind in STUDY_FILES}}


def _point(summary: dict) -> dict:
    return {"macro_f1": summary["macro_f1"]["mean"], "accuracy": summary["accuracy"]["mean"],
            **{f"{c}_{m}": summary["per_class"][c][m]["mean"] for c in "SF" for m in ("precision", "recall")}}


def operating_points(artifacts_dir: Path) -> dict:
    """The decision rules frozen on validation data: plain argmax, and per-class thresholds where selection kept them."""
    selection = study_artifact(artifacts_dir, "selection")
    if not selection["available"]:
        return {"available": False, "message": selection["message"], "error": selection["error"]}
    validation = study_artifact(artifacts_dir, "validation")
    chosen = selection["data"]
    models = {}
    for name, entry in chosen["candidates"].items():
        points = []
        results = validation["data"]["results"].get(name) if validation["available"] else None
        keys = ["argmax"] + (["thresholds"] if entry["rule"] == "thresholds" else [])
        for key in keys:
            point = {"key": key, "validation": None,
                     "thresholds": {seed: f["thresholds"] for seed, f in entry["frozen"].items()} if key == "thresholds" else None}
            if results:
                point["validation"] = _point(results["argmax" if key == "argmax" else "chosen"])
            points.append(point)
        models[name] = {"rule": entry["rule"], "points": points}
    majority = validation["data"]["majority_class"] if validation["available"] else None
    return {"available": True, "primary": chosen["primary"], "models": models, "validation_message": validation["message"],
            "majority": majority and {"accuracy": majority["accuracy"], "macro_f1": majority["macro_f1"]}}
