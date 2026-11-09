# Explainability methods and measured examples

This document describes model attributions for research inspection. These maps
are **not clinically validated explanations**, do not establish causality, and
are not evidence that a beat is clinically abnormal.

## Implemented methods

### Grad-CAM

Grad-CAM uses gradients from the selected output logit back to the existing
`conv3` activations. Temporal gradients are averaged into channel weights;
the weighted activations are summed, ReLU'd, bilinearly resized to 720 samples,
and normalized by that window's maximum. If the post-ReLU map is all zero, the
method returns a zero map rather than fabricating a region. ReLU can therefore
produce an all-zero map for some inputs, including high-confidence N/V beats;
confidence alone does not guarantee a nonzero attribution. The saved N/S/V/F
examples were selected from correctly classified reference beats, preferring
high confidence and requiring a nonzero post-ReLU map. They are illustrations
selected by that rule, not representative samples.

### Integrated Gradients

Integrated Gradients targets the selected class logit. The primary baseline is
an all-zero 720-sample window, which represents a mean-level signal in the
model's per-window z-scored input space. Integration uses a 512-interval
trapezoidal rule, accumulated in chunks to bound memory. Completeness error is
reported as `sum(attributions) - (logit(input) - logit(baseline))`.

Baseline sensitivity was measured against a fixed-seed-0 Gaussian baseline
with per-sample `N(0,0.1)` noise in normalized input units. This is a
sensitivity check, not a clinically meaningful baseline. Attribution changes
are substantial, so the maps depend on baseline choice.

## Real test examples

Examples come from `results/test_examples_for_xai.npz`, selecting correctly
classified test beats and preferring a high-confidence example with a
nonzero target-class Grad-CAM map. Fresh model probabilities are included in
`docs/xai-examples/attributions.npz`.

| Target class | Record / sample | Target probability | Grad-CAM gradient L2 | Zero-baseline IG logit difference | IG attribution sum | Completeness error | Zero vs noise relative L1 |
|---|---|---:|---:|---:|---:|---:|---:|
| N | 219 / 181165 | 0.9999903440 | 0.4967239797 | 4.4388875961 | 4.4430055618 | 0.0041179657 | 4.4854278564 |
| S | 100 / 520982 | 0.9979189634 | 1.0797091722 | 8.4941062927 | 8.5117244720 | 0.0176181793 | 1.2038128376 |
| V | 221 / 221268 | 0.9999457598 | 0.3831043541 | 9.4736089706 | 9.4469528198 | -0.0266561508 | 1.5394897461 |
| F | 213 / 529130 | 0.9768355489 | 0.8215988278 | 6.8176813126 | 6.8893556595 | 0.0716743469 | 1.2737861872 |

These numbers were emitted by:
`MPLCONFIGDIR=/tmp/neurocardio-mpl TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=1 backend/.venv/bin/python backend/scripts/generate_xai_examples.py`.
Completeness errors are displayed as measured; no post-hoc threshold is
claimed. The Gaussian-baseline completeness errors are recorded in
`docs/xai-examples/measurements.json`.

Images: `docs/xai-examples/N.png`, `S.png`, `V.png`, and `F.png`.
The compressed arrays and coordinates are in `attributions.npz`.

## Reproduction and tests

Run the generator command above to rebuild the examples from the local model
and saved test windows. The test command
`MPLCONFIGDIR=/tmp/neurocardio-mpl TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=1 backend/.venv/bin/python -m pytest backend/tests/test_explainability.py -v`
passed 3 tests covering finite maps, shapes, nonzero gradient flow,
class-specific variation, repeat determinism, batch Grad-CAM, Integrated
Gradients completeness quantities, baseline sensitivity and invalid inputs.

## API/UI Grad-CAM regression (2026-11-09)

The API computes Grad-CAM from `conv3`; it does not derive Grad-CAM from
Integrated Gradients or from input gradients. The reported UI mismatch was an
asynchronous frontend race: after switching method or target class, a slower
response for the previous selection could arrive last and replace the displayed
response. The renderer could then show stale attribution under the currently
selected method. The frontend now cancels/ignores stale requests and displays
data only when its returned method and class match the current selection. API
cache keys use a bumped version so older cached maps cannot mask the corrected
response. Grad-CAM metadata explicitly reports `target_layer=conv3`, a finite
gradient L2 norm, and ReLU/per-window-max normalization after upsampling. The
page clips attribution overlays to plot regions and reports IG completeness
error as an absolute value and percentage, with a warning above 5%.

The API regression test uses record 213, sample 478 from the saved reference,
reconstructs its deployed-preprocessed window, and compares the API result to
an independent test-side `tf.GradientTape` computation on `conv3` with maximum
absolute tolerance `1e-5`. It also checks total variation against IG and
separate method cache keys. Full-suite results are in `docs/testing-report.md`.
