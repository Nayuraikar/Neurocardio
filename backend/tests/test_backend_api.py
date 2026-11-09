from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import wfdb
from fastapi.testclient import TestClient

from app.main import create_app

ROOT = Path(__file__).resolve().parents[2]
MITDB = ROOT / "results" / "mitdb"


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    temp = tmp_path_factory.mktemp("neurocardio-api")
    app = create_app(database_url=f"sqlite:///{temp / 'api.sqlite3'}", data_dir=temp / "data")
    with TestClient(app) as test_client:
        yield test_client


def wfdb_upload(record="213"):
    return [(f"{record}.{suffix}", (MITDB / f"{record}.{suffix}").read_bytes())
            for suffix in ("hea", "dat", "atr")]


def ensure_record(client):
    known = getattr(client, "_nc_record_id", None)
    if known:
        return known
    response = client.post("/api/v1/ecg/upload", files=[("files", (name, data)) for name, data in wfdb_upload()])
    assert response.status_code == 200, response.text
    client._nc_record_id = response.json()["id"]
    return client._nc_record_id


def ensure_analysis(client):
    known = getattr(client, "_nc_analysis", None)
    if known:
        return known
    rid = ensure_record(client)
    response = client.post("/api/v1/analyses", params={"record_id": rid})
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "completed"
    client._nc_analysis = response.json()["id"]
    return client._nc_analysis


def test_health_and_model_info_use_loaded_artifacts(client):
    health = client.get("/api/v1/health")
    assert health.status_code == 200 and health.json()["model_loaded"] is True
    info = client.get("/api/v1/model-info")
    assert info.status_code == 200
    payload = info.json()
    assert payload["model"] == "bilstm_aug_sqrtw"
    assert payload["classes"] == ["N", "S", "V", "F"]
    assert payload["rr_features"] is None
    assert payload["test_metrics"]["class_metrics"]["S"]["recall"] == pytest.approx(0.09652076318742986)


def test_upload_wfdb_preview_and_metadata(client):
    rid = ensure_record(client)
    response = client.get("/api/v1/ecg/records")
    assert response.status_code == 200
    rec = next(x for x in response.json() if x["id"] == rid)
    assert rec["format"] == "wfdb" and rec["lead"] == "MLII" and rec["sampling_rate"] == 360


def test_valid_csv_from_real_mit_bih_signal(client):
    record = wfdb.rdrecord(str(MITDB / "213"), physical=True)
    channel = record.p_signal[:18000, record.sig_name.index("MLII")]
    output = io.StringIO()
    np.savetxt(output, channel, delimiter=",", header="MLII", comments="", fmt="%.8f")
    response = client.post("/api/v1/ecg/upload", data={"sampling_rate": "360", "lead": "MLII"},
                           files={"files": ("mitdb-213.csv", output.getvalue().encode(), "text/csv")})
    assert response.status_code == 200, response.text
    assert response.json()["format"] == "csv" and response.json()["sample_count"] == len(channel)
    assert response.json()["preview"]


@pytest.mark.parametrize("filename,payload,expected", [
    ("trace.txt", b"not a signal", 415),
    ("input.csv", b"time,other\n0,1\n", 422),
])
def test_upload_wrong_format_and_missing_lead(client, filename, payload, expected):
    if filename.endswith(".csv"):
        resp = client.post("/api/v1/ecg/upload", data={"sampling_rate": "360", "lead": "MLII"},
                           files={"files": (filename, payload, "text/csv")})
    else:
        resp = client.post("/api/v1/ecg/upload", files={"files": (filename, payload, "text/plain")})
    assert resp.status_code == expected


def test_upload_csv_validation_cases(client):
    missing_rate = client.post("/api/v1/ecg/upload", data={"lead": "MLII"},
                               files={"files": ("x.csv", b"MLII\n0.1\n0.2\n", "text/csv")})
    assert missing_rate.status_code == 422
    unsupported = client.post("/api/v1/ecg/upload", data={"sampling_rate": "250", "lead": "MLII"},
                              files={"files": ("x.csv", b"MLII\n0.1\n0.2\n", "text/csv")})
    assert unsupported.status_code == 422 and "360 Hz" in unsupported.text
    for bad in (b"MLII\nNaN\n", b"MLII\ninf\n"):
        resp = client.post("/api/v1/ecg/upload", data={"sampling_rate": "360", "lead": "MLII"},
                           files={"files": ("x.csv", bad, "text/csv")})
        assert resp.status_code == 422
    flat = client.post("/api/v1/ecg/upload", data={"sampling_rate": "360", "lead": "MLII"},
                       files={"files": ("flat.csv", (b"MLII\n" + b"0\n" * 800), "text/csv")})
    assert flat.status_code == 422 and "flatline" in flat.text
    short = client.post("/api/v1/ecg/upload", data={"sampling_rate": "360", "lead": "MLII"},
                        files={"files": ("short.csv", b"MLII\n0.1\n0.2\n", "text/csv")})
    assert short.status_code == 422 and "short" in short.text.lower()


def test_upload_wfdb_requires_companion_and_rejects_wrong_lead(client):
    hea = ("213.hea", (MITDB / "213.hea").read_bytes())
    missing = client.post("/api/v1/ecg/upload", files={"files": hea})
    assert missing.status_code == 422 and "213.dat" in missing.text
    files = [("files", (name, data)) for name, data in wfdb_upload()]
    wrong_lead = client.post("/api/v1/ecg/upload", data={"lead": "NOT_A_LEAD"}, files=files)
    assert wrong_lead.status_code == 422 and "available leads" in wrong_lead.text
    second = client.post("/api/v1/ecg/upload", files=[("files", (name, data)) for name, data in wfdb_upload("232")])
    assert second.status_code == 200, second.text
    assert second.json()["id"] != ensure_record(client)


def test_wfdb_mismatched_companion_filenames_are_rejected(client):
    files = [
        ("files", ("223.hea", (MITDB / "223.hea").read_bytes())),
        ("files", ("223.dat", (MITDB / "223.dat").read_bytes())),
        ("files", ("228.atr", (MITDB / "228.atr").read_bytes())),
    ]
    response = client.post("/api/v1/ecg/upload", files=files)
    assert response.status_code == 422
    assert "223.hea" in response.text and "223.dat" in response.text and "228.atr" in response.text


@pytest.mark.parametrize("field,value,expected", [(0, "228", "declares record"), (2, "250", "sampling rate")])
def test_wfdb_header_identity_and_rate_are_verified(client, field, value, expected):
    lines = (MITDB / "223.hea").read_text().splitlines()
    fields = lines[0].split()
    fields[field] = value
    lines[0] = " ".join(fields)
    files = [
        ("files", ("223.hea", "\n".join(lines).encode())),
        ("files", ("223.dat", (MITDB / "223.dat").read_bytes())),
    ]
    response = client.post("/api/v1/ecg/upload", files=files)
    assert response.status_code == 422
    assert expected in response.text


def test_misaligned_annotations_are_rejected_even_with_matching_filename(client):
    files = [
        ("files", ("223.hea", (MITDB / "223.hea").read_bytes())),
        ("files", ("223.dat", (MITDB / "223.dat").read_bytes())),
        # Rename annotation contents only in the multipart request; /results remains read-only.
        ("files", ("223.atr", (MITDB / "228.atr").read_bytes())),
    ]
    response = client.post("/api/v1/ecg/upload", files=files)
    assert response.status_code == 422
    assert "annotation alignment check failed" in response.text
    assert "different recording" in response.text


def test_record_analysis_matches_stage3_annotation_reference(client):
    analysis_id = ensure_analysis(client)
    detail = client.get(f"/api/v1/analyses/{analysis_id}").json()
    assert detail["model_version"] == "bilstm_aug_sqrtw"
    assert detail["peak_source"] == "wfdb_annotation"
    assert detail["status"] == "completed"
    beats = client.get(f"/api/v1/analyses/{analysis_id}/beats?limit=10000").json()
    assert len(beats) > 0
    ref = np.load(ROOT / "results" / "test_examples_for_xai.npz")
    selected = np.flatnonzero(ref["record"].astype(str) == "213")
    by_position = {int(ref["position"][i]): ref["prob"][i] for i in selected}
    matched = [b for b in beats if b["sample_index"] in by_position]
    assert matched
    max_diff = max(float(np.max(np.abs(np.asarray(b["prediction"]["probabilities"]) - by_position[b["sample_index"]])))
                   for b in matched)
    assert max_diff <= 1e-4
    assert len(matched) == len(selected)


def _independent_conv3_gradcam(model, window, target_class):
    import tensorflow as tf

    x = tf.convert_to_tensor(np.asarray(window, dtype=np.float32).reshape(1, 720, 1))
    grad_model = tf.keras.Model(model.inputs, [model.get_layer("conv3").output, model.get_layer("logits").output])
    with tf.GradientTape() as tape:
        conv, logits = grad_model(x, training=False)
        score = logits[0, target_class]
    grads = tape.gradient(score, conv)
    assert grads is not None
    weights = tf.reduce_mean(grads, axis=1, keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(conv * weights, axis=-1, keepdims=True))
    upsampled = tf.squeeze(tf.image.resize(cam[..., None], (720, 1), method="bilinear"), axis=[-1, -2])
    maximum = tf.reduce_max(upsampled, axis=1, keepdims=True)
    normalized = tf.where(maximum > 0, upsampled / tf.maximum(maximum, tf.keras.backend.epsilon()), tf.zeros_like(upsampled))
    return normalized.numpy()[0], float(tf.norm(grads, axis=[1, 2]).numpy()[0])


def test_api_gradcam_is_conv3_map_matches_independent_reference_and_is_smooth(client):
    from app.ml.preprocessing import extract_windows, filter_continuous
    from app.models import Beat as BeatEntity

    aid = ensure_analysis(client)
    beats = client.get(f"/api/v1/analyses/{aid}/beats?limit=10000").json()
    reference = np.load(ROOT / "results" / "test_examples_for_xai.npz")
    ref_index = int(np.flatnonzero((reference["record"].astype(str) == "213") &
                                   (reference["position"].astype(int) == 478))[0])
    sample_index = int(reference["position"][ref_index])
    beat = next(b for b in beats if b["sample_index"] == sample_index)
    # Use the exact deployed preprocessing path for the same real reference
    # beat so the independent Grad-CAM comparison isolates attribution math.
    db = client.app.state.SessionLocal()
    try:
        beat_entity = db.get(BeatEntity, beat["id"])
        raw = np.asarray(np.load(beat_entity.analysis.record.signal_path), dtype=np.float32)
        filtered = filter_continuous(raw, beat_entity.analysis.record.sampling_rate)
        window_batch = extract_windows(filtered, np.asarray([sample_index], dtype=np.int64))
        assert len(window_batch.windows) == 1
        window = window_batch.windows[0]
    finally:
        db.close()
    target = 0

    gradcam_response = client.get(f"/api/v1/analyses/{aid}/explanations",
                                  params={"beat": beat["id"], "class": "N", "method": "gradcam"})
    assert gradcam_response.status_code == 200, gradcam_response.text
    gradcam = gradcam_response.json()
    expected, expected_norm = _independent_conv3_gradcam(client.app.state.classifier.model, window, target)
    actual = np.asarray(gradcam["attributions"], dtype=np.float32)
    assert actual.shape == (720,)
    assert float(np.max(np.abs(actual - expected))) <= 1e-5
    assert gradcam["metadata"]["target_layer"] == "conv3"
    assert np.isfinite(gradcam["metadata"]["gradient_l2_norm"])
    assert gradcam["metadata"]["gradient_l2_norm"] == pytest.approx(expected_norm, rel=1e-6)
    assert "ReLU" in gradcam["metadata"]["normalisation"] and "max" in gradcam["metadata"]["normalisation"]

    ig_response = client.get(f"/api/v1/analyses/{aid}/explanations",
                             params={"beat": beat["id"], "class": "N", "method": "ig"})
    assert ig_response.status_code == 200, ig_response.text
    ig = ig_response.json()
    integrated = np.asarray(ig["attributions"], dtype=np.float32)
    assert gradcam["metadata"]["cache_key"] != ig["metadata"]["cache_key"]
    assert not np.allclose(actual, integrated)
    gradcam_tv = float(np.abs(np.diff(actual)).sum())
    ig_tv = float(np.abs(np.diff(integrated)).sum())
    assert ig_tv > 0
    assert gradcam_tv < 0.25 * ig_tv


def test_signal_explanation_review_and_reports(client):
    aid = ensure_analysis(client)
    beats = client.get(f"/api/v1/analyses/{aid}/beats?limit=2").json()
    beat = beats[0]
    sig = client.get(f"/api/v1/analyses/{aid}/signal?start_sample=1000&count=32&filtered=true")
    assert sig.status_code == 200 and len(sig.json()["amplitude"]) == 32
    xai = client.get(f"/api/v1/analyses/{aid}/explanations", params={"beat": beat["id"], "class": "N", "method": "gradcam"})
    assert xai.status_code == 200 and len(xai.json()["attributions"]) == 720
    cached_xai = client.get(f"/api/v1/analyses/{aid}/explanations", params={"beat": beat["id"], "class": "N", "method": "gradcam"})
    assert cached_xai.status_code == 200 and cached_xai.json()["cached"] is True
    review = client.post("/api/v1/reviews", json={"beat_id": beat["id"], "reviewed_label": "V", "note": "reviewed locally"})
    assert review.status_code == 200
    assert review.json()["original_prediction"] == beat["prediction"]["predicted_class"]
    after = client.get(f"/api/v1/analyses/{aid}/beats?limit=1").json()[0]
    assert after["prediction"]["predicted_class"] == beat["prediction"]["predicted_class"]
    assert client.get("/api/v1/reviews").status_code == 200
    assert any(row["id"] == aid for row in client.get("/api/v1/analyses").json())
    assert client.get(f"/api/v1/reports/{aid}?format=csv").headers["content-type"].startswith("text/csv")
    assert client.get(f"/api/v1/reports/{aid}?format=pdf").headers["content-type"].startswith("application/pdf")
    assert client.get(f"/api/v1/reports/{aid}/csv").headers["content-type"].startswith("text/csv")
    bad_xai = client.get(f"/api/v1/analyses/{aid}/explanations", params={"beat": beat["id"], "class": "N", "method": "bad"})
    assert bad_xai.status_code == 422


def test_explanation_methods_cache_key_and_completeness(client):
    import uuid
    from app.models import Explanation

    aid = ensure_analysis(client)
    beats = client.get(f"/api/v1/analyses/{aid}/beats?limit=5").json()
    beat = beats[1]
    bid = beat["id"]

    # (a) gradcam and ig for the same beat/class return different arrays
    g_res = client.get(f"/api/v1/analyses/{aid}/explanations", params={"beat": bid, "class": "N", "method": "gradcam"})
    assert g_res.status_code == 200
    g_data = g_res.json()
    assert g_data["cached"] is False
    assert g_data["method"] == "gradcam"
    assert g_data["class"] == "N"

    ig_res = client.get(f"/api/v1/analyses/{aid}/explanations", params={"beat": bid, "class": "N", "method": "ig"})
    assert ig_res.status_code == 200
    ig_data = ig_res.json()
    assert ig_data["cached"] is False
    assert ig_data["method"] == "ig"
    assert ig_data["class"] == "N"

    g_arr = np.asarray(g_data["attributions"], dtype=np.float32)
    ig_arr = np.asarray(ig_data["attributions"], dtype=np.float32)
    assert not np.allclose(g_arr, ig_arr)
    assert np.all(g_arr >= 0.0)
    assert np.max(g_arr) <= 1.0 + 1e-6
    assert np.any(ig_arr < 0.0) or not np.allclose(g_arr, ig_arr)

    # (b) repeat requests return identical arrays (determinism/caching) and report cached=true
    g_rep = client.get(f"/api/v1/analyses/{aid}/explanations", params={"beat": bid, "class": "N", "method": "gradcam"})
    assert g_rep.status_code == 200
    g_rep_data = g_rep.json()
    assert g_rep_data["cached"] is True
    assert g_rep_data["attributions"] == g_data["attributions"]

    ig_rep = client.get(f"/api/v1/analyses/{aid}/explanations", params={"beat": bid, "class": "N", "method": "ig"})
    assert ig_rep.status_code == 200
    ig_rep_data = ig_rep.json()
    assert ig_rep_data["cached"] is True
    assert ig_rep_data["attributions"] == ig_data["attributions"]

    # (c) different target classes give different results
    g_v = client.get(f"/api/v1/analyses/{aid}/explanations", params={"beat": bid, "class": "V", "method": "gradcam"})
    assert g_v.status_code == 200
    g_v_arr = np.asarray(g_v.json()["attributions"], dtype=np.float32)
    assert not np.allclose(g_arr, g_v_arr)

    # (d) the stored method label matches the request
    assert g_data["method"] == "gradcam"
    assert g_rep_data["method"] == "gradcam"
    assert ig_data["method"] == "ig"
    assert ig_rep_data["method"] == "ig"

    # (e) IG completeness error is returned and finite
    assert "completeness_error" in ig_data["metadata"]
    c_err = ig_data["metadata"]["completeness_error"]
    assert np.isfinite(c_err)
    assert isinstance(c_err, float)
    assert ig_data["metadata"]["baseline"] == "all-zero normalized window"
    assert ig_data["metadata"]["steps"] == 512

    # (f) Stale entries without cache_key are not served
    db = client.app.state.SessionLocal()
    stale_id = str(uuid.uuid4())
    db.add(Explanation(id=stale_id, cache_key=None, beat_id=bid, method="gradcam", target_class="S", attributions=[888.0]*720, metadata_json={"stale": True}))
    db.commit()
    db.close()
    s_res = client.get(f"/api/v1/analyses/{aid}/explanations", params={"beat": bid, "class": "S", "method": "gradcam"})
    assert s_res.status_code == 200
    assert s_res.json()["attributions"][0] != 888.0
    assert s_res.json().get("metadata", {}).get("stale") is not True


def test_delete_analysis_cascades_dependents_and_report_files(client):
    from sqlalchemy import func, select
    from app.models import Analysis, Beat, Explanation, Prediction, Report, Review
    aid = ensure_analysis(client)
    beats = client.get(f"/api/v1/analyses/{aid}/beats?limit=2").json()
    beat_id = beats[0]["id"]
    client.get(f"/api/v1/analyses/{aid}/explanations", params={"beat": beat_id, "class": "N", "method": "gradcam"})
    client.post("/api/v1/reviews", json={"beat_id": beat_id, "reviewed_label": "N", "note": "delete cascade fixture"})
    client.get(f"/api/v1/reports/{aid}?format=csv")
    db = client.app.state.SessionLocal()
    try:
        report_paths = [Path(r.path) for r in db.scalars(select(Report).where(Report.analysis_id == aid)).all()]
    finally:
        db.close()
    response = client.delete(f"/api/v1/analyses/{aid}")
    assert response.status_code == 200 and response.json()["deleted"] is True
    assert client.get(f"/api/v1/analyses/{aid}").status_code == 404
    assert client.get(f"/api/v1/analyses/{aid}/beats").status_code == 404
    assert client.get(f"/api/v1/reports/{aid}").status_code == 404
    assert all(not path.exists() for path in report_paths)
    db = client.app.state.SessionLocal()
    try:
        assert db.scalar(select(func.count()).select_from(Analysis).where(Analysis.id == aid)) == 0
        assert db.scalar(select(func.count()).select_from(Beat).where(Beat.analysis_id == aid)) == 0
        assert db.scalar(select(func.count()).select_from(Prediction).join(Beat).where(Beat.analysis_id == aid)) == 0
        assert db.scalar(select(func.count()).select_from(Report).where(Report.analysis_id == aid)) == 0
        assert db.scalar(select(func.count()).select_from(Review).where(Review.beat_id == beat_id)) == 0
        assert db.scalar(select(func.count()).select_from(Explanation).where(Explanation.beat_id == beat_id)) == 0
    finally:
        db.close()


def test_not_found_and_failed_analysis_states(client):
    assert client.get("/api/v1/analyses/missing").status_code == 404
    assert client.get("/api/v1/analyses/missing/beats").status_code == 404
    assert client.get("/api/v1/reports/missing").status_code == 404
    rid = ensure_record(client)
    from app.models import ECGRecord
    db = client.app.state.SessionLocal()
    try:
        rec = db.get(ECGRecord, rid)
        signal_path = Path(rec.signal_path)
        good = np.load(signal_path)
        np.save(signal_path, np.full(good.shape, np.nan, dtype=np.float32))
        failed = client.post("/api/v1/analyses", params={"record_id": rid})
        assert failed.status_code == 500
        failure_id = failed.json()["detail"]["analysis_id"]
        assert failure_id
        detail = client.get(f"/api/v1/analyses/{failure_id}").json()
        assert detail["status"] == "failed" and detail["error_message"]
    finally:
        np.save(signal_path, good)
        db.close()


def test_persistence_visible_after_app_restart(client):
    rid = ensure_record(client)
    db_url = str(client.app.state.SessionLocal.kw["bind"].url)
    restarted = create_app(database_url=db_url, data_dir=client.app.state.data_dir, initialize_model=False)
    with TestClient(restarted) as second:
        records = second.get("/api/v1/ecg/records").json()
        assert any(row["id"] == rid for row in records)


def test_no_r_peaks_returns_explicit_empty_analysis_state():
    from app.ml.analysis import analyze_recording
    record = wfdb.rdrecord(str(MITDB / "213"), physical=True)
    signal = record.p_signal[:, record.sig_name.index("MLII")]
    with patch("app.ml.analysis.detect_r_peaks", return_value=np.array([], dtype=np.int64)):
        result = analyze_recording(signal, 360, "MLII")
    assert result["status"] == "no_r_peaks"
    assert result["predictions"] == []
    assert result["r_peak_count"] == 0


def test_oversize_upload_is_rejected(tmp_path):
    app = create_app(database_url=f"sqlite:///{tmp_path / 'large.sqlite3'}", data_dir=tmp_path / "data",
                     max_upload_bytes=8, initialize_model=False)
    with TestClient(app) as c:
        response = c.post("/api/v1/ecg/upload", files={"files": ("large.txt", b"0123456789")})
    assert response.status_code == 413
