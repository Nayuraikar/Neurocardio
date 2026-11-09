# NeuroCardio backend ML setup

This stage contains the selected Keras model, exact configured preprocessing,
WFDB XQRS retrospective detection, and inference utilities. The selected model
is ECG-only (one 720×1 input); no RR features are accepted.

## Local environment

On Apple Silicon, Python 3.11 is the chosen runtime. TensorFlow CPU inference
is sufficient; TensorFlow Metal is optional and is not required. From this
directory:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Check required imports and versions:

```sh
python - <<'PY'
import tensorflow, keras, numpy, scipy, pandas, wfdb, neurokit2
for m in (tensorflow, keras, numpy, scipy, pandas, wfdb, neurokit2):
    print(m.__name__, getattr(m, "__version__", "available"))
PY
```

The checked-in artifact metadata records Keras 3.13.2. The source notebook
runtime printed TensorFlow 2.20.0. This local environment must still pass the
model-loading test before being considered verified.

## Tests and evaluation

```sh
PYTHONPATH=. python -m pytest -q
PYTHONPATH=. python scripts/compare_reference_inputs.py
PYTHONPATH=. python scripts/evaluate_rpeaks.py
```

The reference scripts use the project-adjacent local `results/` artifacts and
MIT-BIH files. They do not download or upload data. `test_model_loading.py`
labels generated random input as a technical smoke test; the golden test uses
the actual notebook XAI reference window/probability.

The inference path is offline/retrospective. Inputs must be a finite single
channel at exactly 360 Hz; the training pipeline specifies no resampling.
