import csv
import json
from pathlib import Path


def read_model_info(results_dir: Path) -> dict:
    raw = json.loads((results_dir / "test_raw_primary.json").read_text())
    test = json.loads((results_dir / "test_metrics.json").read_text())
    cv = json.loads((results_dir / "cv_results.json").read_text())
    summary_rows = list(csv.DictReader((results_dir / "final_summary.csv").open()))
    fixed = next(r for r in summary_rows if r["protocol"] == "Fixed test, seed 42")
    selected = test["selected_test_metrics"]
    labels = ["N", "S", "V", "F"]
    confusion = selected["confusion_matrix"]
    class_metrics = {}
    for i, label in enumerate(labels):
        tp = confusion[i][i]
        fp = sum(confusion[row][i] for row in range(4)) - tp
        fn = sum(confusion[i]) - tp
        support = tp + fn
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        class_metrics[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    pooled = cv.get("pooled", {})
    return {"model": test.get("selected_model"), "evaluation_protocol": fixed["protocol"],
            "test_metrics": {"accuracy": selected["accuracy"], "macro_f1": selected["macro_f1"],
                             "macro_roc_auc": selected["macro_roc_auc"], "class_metrics": class_metrics,
                             "confusion_matrix_true_rows_pred_columns": confusion,
                             "raw_primary_precision": raw.get("precision"), "raw_primary_recall": raw.get("recall")},
            "grouped_cv_pooled": {k: pooled.get(k) for k in ("accuracy", "macro_f1", "sensitivity", "confusion_matrix")},
            "fixed_test_summary_row": fixed,
            "limitations": ["Retrospective MIT-BIH only; single lead MLII at 360 Hz.",
                            "S and F recall are weak on both fixed test and grouped cross-validation.",
                            "Grouped CV pooled folds include patients from the fixed test workflow.",
                            "Research prototype, not clinically validated and not a diagnostic device."]}
