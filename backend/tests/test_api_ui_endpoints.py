from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import create_app

MITDB = Path(__file__).resolve().parents[2] / "results" / "mitdb"


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    temp = tmp_path_factory.mktemp("ui-api")
    app = create_app(database_url=f"sqlite:///{temp / 'ui.sqlite3'}", data_dir=temp / "data")
    with TestClient(app) as c:
        yield c


def files(record, parts=("hea", "dat", "atr")):
    return [("files", (f"{record}.{s}", (MITDB / f"{record}.{s}").read_bytes())) for s in parts]


def wait(client, analysis_id):
    for _ in range(300):
        row = client.get(f"/api/analyses/{analysis_id}").json()
        if row["status"] in ("completed", "failed"):
            return row
        time.sleep(0.2)
    raise AssertionError("analysis did not finish")


@pytest.fixture(scope="module")
def analysed(client):
    record = client.post("/api/upload", files=files("101"))
    assert record.status_code == 200, record.text
    started = client.post("/api/analyze", params={"record_id": record.json()["id"]})
    assert started.status_code == 202 and started.json()["status"] == "running"
    done = wait(client, started.json()["id"])
    assert done["status"] == "completed" and done["summary"]["stage"] == "done", done
    return record.json(), done


def test_status_and_model_config(client):
    status = client.get("/api/status").json()
    assert status["model_loaded"] and status["database_ok"]
    assert client.get("/api/model/config").json()["architecture"] == "CNN-BiLSTM"


def test_paced_recording_is_rejected(client):
    response = client.post("/api/upload", files=files("217"))
    assert response.status_code == 422 and "paced" in response.json()["detail"]


def test_warnings_without_annotation(client):
    record = client.post("/api/upload", files=files("103", ("hea", "dat"))).json()
    assert record["units"] == "mV" and any("XQRS" in w for w in record["input_warnings"])


def test_recording_views(client, analysed):
    record, analysis = analysed
    rid = record["id"]
    preds = client.get(f"/api/recordings/{rid}/predictions").json()
    assert preds["analysis_id"] == analysis["id"] and len(preds["predictions"]) == analysis["beat_count"]
    sample = preds["predictions"][5]["sample_index"]
    win = client.get(f"/api/recordings/{rid}/window", params={"sample": sample}).json()
    assert len(win["raw"]) == len(win["model_input"]) == 720
    assert abs(win["stats"]["model_input"]["mean"]) < 1e-4 and abs(win["stats"]["model_input"]["sd"] - 1) < 1e-3
    assert len(client.get(f"/api/recordings/{rid}/envelope").json()["min"]) == 600
    assert client.get("/api/recordings").json()[0]["id"]
    assert client.get("/api/recordings/nope").status_code == 404


def test_overview_and_review_state(client, analysed):
    _, analysis = analysed
    before = client.get("/api/overview").json()
    assert before["completed_analyses"] >= 1
    beats = client.get(f"/api/analyses/{analysis['id']}/beats?limit=10000").json()
    flagged = [b for b in beats if b["prediction"]["review_flag"]]
    if flagged:
        client.post("/api/reviews", json={"beat_id": flagged[0]["id"], "reviewed_label": "N"})
        assert client.get("/api/overview").json()["pending_review_windows"] == before["pending_review_windows"] - 1
        withdrawn = client.post("/api/reviews", json={"beat_id": flagged[0]["id"], "reviewed_label": None}).json()
        assert withdrawn["status"] == "withdrawn"
        assert client.get("/api/overview").json()["pending_review_windows"] == before["pending_review_windows"]


def test_signal_quality_uses_only_measured_metrics(client, analysed):
    record, _ = analysed
    q = client.get(f"/api/recordings/{record['id']}/quality").json()
    assert [m["key"] for m in q["metrics"]] == ["missing_fraction", "clipping_fraction", "baseline_wander_power_fraction",
                                               "high_frequency_power_fraction", "invalid_window_fraction"]
    assert q["overall"] in ("Good", "Moderate", "Poor") and all(0 <= m["value"] <= 1 for m in q["metrics"])


def test_quality_measures_on_synthetic_signals():
    import numpy as np
    from app.services.quality_service import band_fractions, clipping_fraction
    t = np.arange(0, 60, 1 / 360)
    assert clipping_fraction(np.clip(np.sin(2 * np.pi * t), -0.5, 0.5)) > 0.4
    assert clipping_fraction(np.sin(2 * np.pi * t)) < 0.001
    low, high = band_fractions(np.sin(2 * np.pi * 0.2 * t), 360)
    assert low > 0.95 and high < 0.01
    low, high = band_fractions(np.sin(2 * np.pi * 60 * t), 360)
    assert high > 0.95 and low < 0.01


def test_model_metrics_current_and_missing(client, tmp_path):
    payload = client.get("/api/model/metrics").json()
    assert payload["available"] and payload["stale"] == [] and payload["split_facts"]["records_disjoint"]
    app = create_app(database_url=f"sqlite:///{tmp_path / 'x.sqlite3'}", data_dir=tmp_path, artifacts_dir=tmp_path,
                     initialize_model=False)
    with TestClient(app) as empty:
        assert empty.get("/api/model/metrics").json()["available"] is False
        (tmp_path / "metrics.json").write_text("{broken")
        assert "could not be read" in empty.get("/api/model/metrics").json()["problem"]


def test_study_empty_smoke_broken_and_present(tmp_path):
    import json
    import study_fixtures as F
    app = create_app(database_url=f"sqlite:///{tmp_path / 's.sqlite3'}", data_dir=tmp_path, artifacts_dir=tmp_path,
                     initialize_model=False)
    with TestClient(app) as c:
        empty = c.get("/api/study").json()
        assert empty["final"] == {"available": False, "message": "Final evaluation not run yet.", "data": None, "error": False}
        assert {"cnn_bilstm", "rr_fusion_attn_shead", "cnn", "lstm", "resnet1d", "gbm", "stacked_ensemble"} <= set(empty["models"])
        assert c.get("/api/study/operating-points").json()["available"] is False
        (tmp_path / "selection.json").write_text(json.dumps(F.selection(smoke=True)))
        assert "smoke" in c.get("/api/study").json()["selection"]["message"]
        (tmp_path / "selection.json").write_text("not json")
        assert c.get("/api/study").json()["selection"]["error"] is True
        (tmp_path / "selection.json").write_text(json.dumps(F.selection()))
        (tmp_path / "validation_report.json").write_text(json.dumps(F.validation()))
        points = c.get("/api/study/operating-points").json()
        assert [p["key"] for p in points["models"]["rr_fusion"]["points"]] == ["argmax", "thresholds"]
        assert points["models"]["rr_fusion"]["points"][1]["validation"]["macro_f1"] == 0.6


def test_explanations_reference_and_comparison(client, analysed):
    _, analysis = analysed
    aid = analysis["id"]
    beat = client.get(f"/api/analyses/{aid}/beats?limit=5").json()[3]
    cls = beat["prediction"]["predicted_class"]
    params = {"beat": beat["id"], "class": cls}
    assert client.get(f"/api/analyses/{aid}/explanations/compare", params=params).status_code == 409
    cam = client.get(f"/api/analyses/{aid}/explanations", params={**params, "method": "gradcam"}).json()
    ig = client.get(f"/api/analyses/{aid}/explanations", params={**params, "method": "ig"}).json()
    assert ig["metadata"]["steps"] == 512 and "completeness_error" in ig["metadata"] and cam["model_version"]
    assert ig["reference_label"] in ("N", "S", "V", "F") and ig["top_regions"]
    assert "spearman_gradcam_vs_abs_ig" in client.get(f"/api/analyses/{aid}/explanations/compare", params=params).json()
    ref = client.get(f"/api/analyses/{aid}/reference").json()
    assert ref["available"] and len(ref["labels"]) == analysis["beat_count"]
    assert client.get("/api/model/xai-evaluation").json()["available"] is True


def test_demo_samples_come_only_from_training_and_validation(client):
    import json
    from app.core.config import CONFIG_PATH, SPLITS_DIR, ARTIFACTS_DIR
    app_split = json.loads(CONFIG_PATH.read_text())["split"]
    study = json.loads((SPLITS_DIR / "multidb_v1.json").read_text())
    test_keys = set(study["test_ds2"]) | set(study["test_incart"]) | {f"mitdb/{r}" for r in app_split["test_records"]}
    test_patients = {study["patients"].get(k, k) for k in test_keys}
    listed = client.get("/api/demo/samples").json()["samples"]
    assert listed and (ARTIFACTS_DIR / "demo_samples.json").stat().st_size < 2 * 1024 * 1024
    for s in listed:
        key = f"mitdb/{s['record']}"
        assert s["database"] == "mitdb" and key not in test_keys and study["patients"].get(key, key) not in test_patients
        assert s["record"] in app_split[f"{s['app_model_split']}_records"] and s["app_model_split"] in ("train", "validation")
        assert s["study_split"] in ("train", "validation") and key in study[s["study_split"]]
    assert not {"201", "202"} & {s["record"] for s in listed}
    items = client.get("/api/demo/challenge?seed=3").json()["items"]
    assert len(items) == 8 and all("label" not in i for i in items)
    hidden = client.get(f"/api/demo/samples/{items[0]['id']}?reveal=false").json()
    assert "label" not in hidden and hidden["prediction"] is None
    shown = client.get(f"/api/demo/samples/{items[0]['id']}").json()
    assert shown["label"] in "NSVF" and shown["prediction"]["predicted_class"] in "NSVF"


def test_demo_file_with_a_test_patient_is_refused(tmp_path):
    import json
    from app.core.config import ARTIFACTS_DIR
    data = json.loads((ARTIFACTS_DIR / "demo_samples.json").read_text())
    data["samples"][0]["record"] = "232"  # a DS2 record
    (tmp_path / "demo_samples.json").write_text(json.dumps(data))
    app = create_app(database_url=f"sqlite:///{tmp_path / 'd.sqlite3'}", data_dir=tmp_path, artifacts_dir=tmp_path,
                     initialize_model=False)
    with TestClient(app) as c:
        response = c.get("/api/demo/samples")
        assert response.status_code == 503 and "refused" in response.json()["detail"]


def test_dataset_endpoints(client):
    summary = client.get("/api/dataset/summary").json()
    assert summary["app_model"]["dataset"]["records_total"] == 48 and summary["study"]["splits"]
    records = client.get("/api/dataset/records").json()
    assert len(records) == 48
    examples = client.get("/api/dataset/representatives").json()["examples"]
    from app.services.dataset_service import record_split
    assert [e["label"] for e in examples] == list("NSVF") and all(record_split(e["record"]) == "train" for e in examples)
    assert len(client.get("/api/dataset/records/208/window?class=F&index=2").json()["model_input"]) == 720
    assert client.get("/api/dataset/records/..%2F208/segment").status_code == 404


def test_history_sign_off_and_predictions_are_never_overwritten(client, analysed):
    _, analysis = analysed
    aid = analysis["id"]
    before = client.get(f"/api/analyses/{aid}/beats?limit=10000").json()
    client.post("/api/reviews", json={"beat_id": before[0]["id"], "reviewed_label": "F", "note": "=1+1 <b>x"})
    assert client.post(f"/api/analyses/{aid}/sign-offs", json={"note": "checked <ok> & done"}).status_code == 200
    row = next(r for r in client.get("/api/history").json() if r["id"] == aid)
    assert row["analysis_reviewed"] and not row["needs_review"]
    client.post(f"/api/analyses/{aid}/sign-offs", json={"status": "reopened"})
    assert not next(r for r in client.get("/api/history").json() if r["id"] == aid)["analysis_reviewed"]
    after = client.get(f"/api/analyses/{aid}/beats?limit=10000").json()
    assert [b["prediction"] for b in before] == [b["prediction"] for b in after]


def test_report_sections_pdf_and_csv(client, analysed):
    _, analysis = analysed
    aid = analysis["id"]
    first = client.get(f"/api/analyses/{aid}/beats?limit=1").json()[0]
    client.post("/api/reviews", json={"beat_id": first["id"], "reviewed_label": "F", "note": "=1+1 <b>x"})
    full = client.get(f"/api/reports/{aid}?format=pdf")
    assert full.status_code == 200 and full.content[:4] == b"%PDF"
    small = client.get(f"/api/reports/{aid}?format=pdf&sections=recording,limitations&inline=true")
    assert small.headers["content-disposition"].startswith("inline") and len(small.content) < len(full.content)
    text = client.get(f"/api/reports/{aid}?format=csv").text
    assert f"# model_version: {analysis['model_version']}" in text and "# generated_at_utc:" in text and "'=1+1" in text
    only = client.get(f"/api/reports/{aid}?format=csv&sections=recording").text
    assert "predicted_class" not in only and "# recording: 101.hea" in only
    assert client.get(f"/api/reports/{aid}?sections=bogus").status_code == 422
