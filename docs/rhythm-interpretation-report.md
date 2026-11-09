# Rhythm interpretation: scope and evidence

## What the deployed model can identify

NeuroCardio currently uses `bilstm_aug_sqrtw`, a per-beat CNN-BiLSTM. It receives one 720-sample MLII window centered on each detected or annotated R peak, at 360 Hz, and assigns one AAMI-mapped class: N, S, V, or F. The deployed input has no RR-interval features. The model's output is a beat class and its four class probabilities, not a recording-level rhythm label.

The interface now describes those outputs as normal, supraventricular ectopic, ventricular ectopic, and fusion beats. A V-class output may include PVC-like morphology, but it does not independently diagnose a PVC or ventricular tachycardia. F is not presented as an independent arrhythmia diagnosis. Neither S nor any other beat class is converted into atrial fibrillation, atrial flutter, supraventricular tachycardia, or another named rhythm.

The model's documented fixed-test recall is weak for S (9.7%) and F (5.7%). A class not predicted by this model cannot be treated as absent. The application therefore presents rhythm-level classification as unavailable and keeps beat predictions separate from any human review labels.

## Rhythm annotations and available signal information

The local MIT-BIH Arrhythmia Database files were inspected, including the auxiliary notes in the available 48 `.atr` annotation files. Some records contain rhythm-change notes such as `(AFIB`, `(AFL`, `(VT`, and `(SVTA`, as well as other rhythm annotations. Their presence is record-dependent; the annotations are not uniform coverage across all records or all rhythm categories. Record 228's annotation contains `(N` and `(B` notes and no AFIB, AFL, VT, or SVTA note.

The current data flow uses annotation symbols only for the existing AAMI beat mapping and, when available, annotation-peak alignment. The runtime analysis stores beat timestamps/sample indices, the single-lead signal, four-class predictions/probabilities, and separate review rows. It does not expose a validated rhythm-label target, rhythm-segment evaluation, or rhythm-classifier output. The existence of some rhythm notes in source annotations is not evidence that the deployed model detects those rhythms.

## What a separate rhythm component would require

A future rhythm-analysis module should be independent of the beat classifier and its stored predictions. Before implementation, it would need:

- A defined target taxonomy and explicit mapping from reference rhythm annotations to time intervals, including treatment of transitions, ambiguous segments, and missing labels.
- A separate model or justified signal-processing algorithm operating on continuous recording segments and suitable rhythm features. The existing per-beat morphology model, which has no RR input, is not sufficient for this purpose.
- Patient-wise development and evaluation that prevents leakage, with per-rhythm sensitivity, specificity, precision, and interval-level/event-level metrics; independent external validation would be needed before making stronger generalization claims.
- Explicit abstention when annotation coverage, signal quality, or evidence is insufficient. Unsupported or uncertain cases should remain “Insufficient evidence for rhythm classification” or “Not supported by the current model.”
- Separate API, persistence, UI, and report fields so rhythm findings cannot overwrite or be inferred from N/S/V/F predictions or human-reviewed labels.

The available annotations could support a carefully scoped retrospective research study for selected rhythms and records with usable reference intervals. Their partial coverage does not justify implementing a named-rhythm detector in the current application. No rhythm classifier or rhythm diagnosis was implemented.

## Implemented and unsupported

- Implemented: class-specific recording findings and timestamps derived from saved beat predictions; an explicit rhythm-classification-unavailable notice; explanation of the V and F class limits.
- Not implemented: AF, AFL, VT, SVT, or other named-rhythm predictions; clinical interpretation; any alteration to the selected model, preprocessing, signal input, probabilities, beat predictions, or database.

This application remains a local research prototype and is not clinically validated or a diagnostic device.
