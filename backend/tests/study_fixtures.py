"""Synthetic study artifacts for tests. Every number is made up and exists only to exercise the readers."""


def _summary(base: float) -> dict:
    stat = lambda v: {"mean": v, "sd": 0.001}
    return {"n_beats": 10, "n_patients": 2, "macro_f1": stat(base), "accuracy": stat(base + 0.1),
            "per_class": {c: {"precision": stat(base - 0.1), "recall": stat(base - 0.2), "f1": stat(base - 0.15), "support": 5}
                          for c in "NSVF"}}


def selection(smoke=False):
    frozen = {"1": {"temperature": 1.0, "thresholds": [1, 2, 1, 4], "review_confidence_cutoffs": {"0.05": 0.5}}}
    return {"smoke": smoke, "primary": "rr_fusion", "candidates": {"rr_fusion": {"rule": "thresholds", "frozen": frozen}}}


def validation(smoke=False):
    return {"smoke": smoke, "primary": "rr_fusion", "majority_class": {"accuracy": 0.9, "macro_f1": 0.237},
            "results": {"rr_fusion": {"rule": "thresholds", "chosen": _summary(0.6), "argmax": _summary(0.5)}}}
