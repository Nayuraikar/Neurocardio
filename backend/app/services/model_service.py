"""Facts about the loaded model, generated from its layers and the preprocessing config."""
from __future__ import annotations

from app.core.config import PREPROCESSING_VERSION

NORMALIZATION_LABELS = {"per_window_zscore": "per-window z-score"}


def architecture_label(types: list[str]) -> str:
    """e.g. Conv1D + Bidirectional(LSTM) layers -> 'CNN-BiLSTM'."""
    parts = []
    if "Conv1D" in types:
        parts.append("CNN")
    if "Bidirectional" in types:
        parts.append("BiLSTM")
    elif "LSTM" in types:
        parts.append("LSTM")
    return "-".join(parts) or "unrecognised architecture"


def describe_layers(model) -> list[dict]:
    rows = []
    for layer in model.layers:
        try:
            shape = layer.output.shape
        except AttributeError:
            shape = ()
        rows.append({"name": layer.name, "type": type(layer).__name__,
                     "output_shape": [None if d is None else int(d) for d in shape], "params": int(layer.count_params())})
    return rows


def model_config(classifier, config: dict, config_sha256: str) -> dict:
    layers = describe_layers(classifier.model)
    dims = [int(d) for d in classifier.model.inputs[0].shape[1:]]
    band = config["bandpass_filter"]
    norm = NORMALIZATION_LABELS.get(config["normalization"], config["normalization"])
    shape = "×".join(map(str, dims))
    pipeline = [
        {"key": "validation", "label": "Validation",
         "detail": f"Format, {config['sampling_frequency']} Hz sampling rate, finite values and beat positions are checked."},
        {"key": "filter", "label": "Bandpass filter",
         "detail": f"{band['type'].title()} order {band['order']}, {band['low_hz']}–{band['high_hz']} Hz, "
                   f"{'zero-phase, ' if band['zero_phase'] else ''}on the whole recording."},
        {"key": "segmentation", "label": "Segmentation",
         "detail": f"{config['window_size']}-sample windows {config['window_alignment']}. Edge beats are skipped."},
        {"key": "standardization", "label": "Input standardization", "detail": f"{norm}, giving a {shape} input."},
        {"key": "inference", "label": "Window classification",
         "detail": f"Softmax over {', '.join(config['classes'])} for each window."},
    ]
    return {"architecture": architecture_label([x["type"] for x in layers]), "input_shape": shape, "input_dims": dims,
            "normalization": norm, "sampling_rate_hz": config["sampling_frequency"], "window_samples": config["window_size"],
            "window_alignment": config["window_alignment"], "lead": config["lead"],
            "bandpass": {k: band[k] for k in ("type", "order", "low_hz", "high_hz", "zero_phase")},
            "classes": config["classes"], "model_version": config["selected_model"], "model_sha256": classifier.sha256,
            "config_sha256": config_sha256, "preprocessing_version": PREPROCESSING_VERSION,
            "parameter_count": int(classifier.model.count_params()), "uses_rr_features": config.get("rr_features") is not None,
            "layers": layers, "pipeline": pipeline}
