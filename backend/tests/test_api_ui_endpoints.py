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
