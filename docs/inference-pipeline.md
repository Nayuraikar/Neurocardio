# NeuroCardio inference pipeline

This describes the measured selected model and implemented retrospective path.
It does not describe real-time prediction and is not a clinical protocol.

## Model contract

The canonical artifact is `backend/models/neurocardio_final.keras`, checked by
SHA-256 at load. Its config identifies `bilstm_aug_sqrtw`, one input named
`ecg_input` shaped `(batch,720,1)`, one four-class softmax output, and class
order N, S, V, F. No RR input is used. The model was loaded with
`compile=False`; the layer archive contains no inference-time custom objects.
The saved XAI reference includes an `R` array, but equivalence inference passes
only `X`.

## Recorded signal-to-prediction path

1. Select a single lead. The evaluated MIT-BIH path prefers MLII and uses the
   first available lead only when MLII is absent.
2. Require a finite one-dimensional signal sampled at exactly 360 Hz. The
   configured implementation rejects other rates; no resampling is specified.
3. Apply a fourth-order Butterworth bandpass (0.5–40 Hz) as second-order
   sections using zero-phase `sosfiltfilt` to the **full continuous recording**.
4. Find beat centers. Annotation-based positions are used for equivalence
   evaluation only. Automatic retrospective detection uses WFDB XQRS on the
   filtered signal and does not receive annotations.
5. For a peak at sample `p`, extract `[p-360,p+360)`, putting the peak at
   sample index 360. Skip a window that crosses a recording edge and count it.
6. For each complete window, compute population mean and standard deviation
   and normalize as `(x-mean)/(std+1e-8)`. Cast to float32 and append one signal
   channel, yielding `(beats,720,1)`. A zero-variance window remains finite.
7. Run the loaded Keras model in batches. Check output shape, finiteness,
   probability bounds and row sums. Return N/S/V/F probabilities, raw argmax,
   maximum probability as confidence, and original sample position.

The analysis is retrospective. The configured model does not use post-beat RR
features, and no RR values are passed to inference. Sampling rates 250 Hz and
500 Hz were tested and rejected with a clear error.

## Measured equivalence

The one-command verification is:

```sh
MPLCONFIGDIR=/tmp/neurocardio-mpl backend/.venv/bin/python backend/scripts/verify_model_reference.py
```

It compared all 47,518 saved windows and probabilities; max probability
difference was 9.268522262573242e-06, mean difference 2.9065758149077487e-08,
and argmax agreement was 100%. Annotation-centered raw-record reconstruction
had maximum input difference 1.1920928955078125e-07 and also matched every
saved prediction. Full results are in
[`model-reference-comparison.json`](model-reference-comparison.json).

## Detector and classifier are separate measurements

At 150 ms one-to-one tolerance over 21 fixed MIT-BIH test records, WFDB XQRS
had sensitivity 0.9966995312, positive predictivity 0.9986309160 and F1
0.9976642889. These are beat-detection metrics. On detector-matched beats,
classifier accuracy was 0.9101004983 and macro-F1 was 0.4755221943; these are
classifier metrics. The separate record-level results are in
[`rpeak-evaluation.csv`](rpeak-evaluation.csv). False detections and missed
beats are not treated as classifier errors in the detector table.

## Research limits

Results are retrospective and use MIT-BIH, a fixed patient-wise DS2 test
partition, one lead (MLII), and 360 Hz. The grouped CV documentation notes
that pooled folds include original test patients. S and F recall are weak:
fixed test recalls were 0.0965207632 and 0.0568475452; grouped CV recalls were
0.0535971223 and 0.01246882793. Performance on other patients, acquisition
systems, leads, and sampling rates has not been established. This is not a
clinically validated diagnostic device.
