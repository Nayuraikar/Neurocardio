from __future__ import annotations

import numpy as np
from fastapi import APIRouter, HTTPException, Query, Request

from app.api.routes import XAI_LOCK, get_explainer
from app.services.demo_service import challenge_ids, describe, load_samples, standardize
from app.services.xai_service import top_regions

router = APIRouter(prefix="/demo")
CLASSES = ("N", "S", "V", "F")


def _payload(request: Request) -> dict:
    data, problem = load_samples(request.app.state.artifacts_dir)
    if data is None:
        raise HTTPException(503, problem)
    return data


def _sample(request: Request, sample_id: str) -> dict:
    for s in _payload(request)["samples"]:
        if s["id"] == sample_id:
            return s
    raise HTTPException(404, "demo sample not found")


@router.get("/samples")
def samples(request: Request):
    data = _payload(request)
    return {**{k: data[k] for k in ("source", "selection", "eligibility", "signal")},
            "samples": [describe(s) for s in data["samples"]]}


@router.get("/challenge")
def challenge(request: Request, per_class: int = Query(2, ge=1, le=5), seed: int = Query(0, ge=0)):
    return {"seed": seed, "items": challenge_ids(_payload(request)["samples"], per_class, seed)}


@router.get("/samples/{sample_id}")
def sample(sample_id: str, request: Request, reveal: bool = True):
    """The window; with reveal, also its reference label and the live model output."""
    s = _sample(request, sample_id)
    window = standardize(s["filtered"])
    out = {**describe(s, with_label=reveal), "model_input": window.astype(float).tolist(), "sampling_rate": 360, "prediction": None}
    if reveal:
        classifier = request.app.state.classifier
        if classifier is None:
            raise HTTPException(503, "the model is not loaded")
        row = classifier.predict(window[None, :, None])[0]
        out["prediction"] = {"predicted_class": row["predicted_class"], "probabilities": row["probabilities"]}
        out["model_version"] = request.app.state.config["selected_model"]
    return out


@router.get("/samples/{sample_id}/gradcam")
def gradcam(sample_id: str, request: Request, target_class: str = Query(..., alias="class", pattern="^[NSVF]$")):
    window = standardize(_sample(request, sample_id)["filtered"])
    with XAI_LOCK:
        values = get_explainer(request).gradcam(np.asarray(window)[None, :, None], CLASSES.index(target_class))[0]
    return {"attributions": values.astype(float).tolist(), "top_regions": top_regions(values.tolist(), 360.0)}
