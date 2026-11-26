"""Reads evaluation artifacts. Nothing here computes a model score; a missing artifact is reported as missing."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from app.core.config import SPLITS_DIR


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
