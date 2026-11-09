# Limitations

- Research prototype only. It is not clinically validated and is not a diagnostic device.
- Model evaluation and explainability examples are retrospective MIT-BIH results. They do not establish prospective or clinical performance.
- Patient-wise held-out testing and grouped cross-validation address leakage within the source evaluation; they do not demonstrate generalization to new hospitals, devices, populations, or acquisition protocols.
- The model uses a single MLII signal at 360 Hz. Uploads at other sample rates are rejected; there is no configured resampling path.
- Fixed test recall is weak for S (9.7%) and F (5.7%). Grouped-CV pooled S and F recall were 5.4% and 1.2%; V recall was 89.2% on the fixed test and 73.0% pooled in grouped CV. The application should direct uncertain/rare-class cases to human review.
- Beat classification does not provide a recording-level rhythm diagnosis. Beat detection errors can change which windows reach the classifier; detector metrics and classifier metrics must remain separate.
- Annotation peaks are available only for permitted WFDB records with annotation files. Other recordings use XQRS.
- Grad-CAM and Integrated Gradients are input- and baseline-dependent model attributions, not clinically validated explanations.
- The backend runs analyses synchronously and has no background job queue or authentication. It is intended for a trusted local session.
- The app depends on local model and `results/` artifacts remaining available at their configured paths.
- User-interface browser automation and screenshots remain pending because Playwright/browser tooling is not installed in the local environment.
