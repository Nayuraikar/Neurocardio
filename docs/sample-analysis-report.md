# Sample report excerpt — real reference beat

This measured sample is one beat from the saved MIT-BIH test reference (`results/test_examples_for_xai.npz`), not a fabricated waveform or a clinical report.

| Field | Value |
|---|---|
| Dataset record | 213 |
| R-peak sample position | 478 |
| Reference class | N |
| Saved model class | N |
| Probabilities N / S / V / F | 0.9996776581 / 0.0000098508 / 0.0000077557 / 0.0003046927 |
| Selected model | `bilstm_aug_sqrtw` |
| Input | one normalized 720-sample MLII window at 360 Hz |

Source command:

```sh
MPLCONFIGDIR=/tmp/neurocardio-mpl backend/.venv/bin/python - <<'PY'
import numpy as np
d = np.load("results/test_examples_for_xai.npz")
i = np.flatnonzero((d["record"].astype(str) == "213") & (d["position"].astype(int) == 478))[0]
print("y, pred, prob:", d["y"][i], d["pred"][i], d["prob"][i].tolist())
PY
```

The fresh inference check against the saved test set recorded maximum absolute probability difference `9.268522262573242e-06` and exact argmax agreement. This beat-level output is not a recording-level rhythm diagnosis. Dataset-label semantics, model uncertainty, weak S/F recall, and retrospective-only evaluation still apply.
