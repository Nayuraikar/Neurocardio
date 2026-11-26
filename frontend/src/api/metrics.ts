export type ClassMetric = { precision: number; recall: number; specificity: number; f1: number; support: number; roc_auc: number; pr_auc: number; prevalence: number }
export type SplitMetrics = {
  n_windows: number; accuracy: number; majority_class_baseline_accuracy: number; balanced_accuracy: number; mcc: number; cohen_kappa: number; role: string
  macro: Record<'precision' | 'recall' | 'f1' | 'roc_auc' | 'pr_auc', number>; weighted: Record<'precision' | 'recall' | 'f1' | 'roc_auc' | 'pr_auc', number>
  per_class: Record<string, ClassMetric>
  confusion_matrix: { labels: string[]; counts: number[][] }
  roc_curves: Record<string, { fpr: number[]; tpr: number[] }>; pr_curves: Record<string, { recall: number[]; precision: number[] }>
}
export type ModelMetrics = {
  available: boolean; message?: string; problem?: string | null; stale: string[]
  metrics: {
    evaluated_at_utc: string; test: SplitMetrics; validation: SplitMetrics
    test_bootstrap: { unit: string; resamples: number; ci95: Record<string, [number, number]> }
    training: {
      loss_as_recorded_in_config: string; verification: string; decision_rule: string; selection_rule: string
      validation_macro_f1_by_variant: Record<string, number>
      history: { epoch: number; accuracy: number; loss: number; val_accuracy: number; val_loss: number; val_macro_f1: number }[]
    }
    split: { type: string; records: Record<string, string[]>; patient_group_ids: Record<string, string[]>; excluded_paced_records: string[]; disjoint_assertion_passed: boolean }
    dataset: { name: string; records_total: number; records_used: number; subject_groups_used: number; class_counts: Record<string, { annotated_beats: Record<string, number>; usable_windows: Record<string, number> }> }
    calibration: { temperature_refit_on_validation: number; test_raw: { ece_15: number }; test_temperature_scaled: { ece_15: number } }
    review_flag_policy: { temperature: number; app_uses: string; operating_points: Record<string, { tau: number; margin: number; test: { review_rate: number; error_rate_on_unflagged: number; fraction_of_all_errors_flagged: number; review_rate_by_true_class: Record<string, number> }; validation: { review_rate: number; error_rate_on_unflagged: number } }> }
  }
  split_facts: { records_disjoint: boolean; test_equals_ds2_without_202: boolean | null; split_file_statement: null | { method: string; selection_rule: string; assignments_drawn: number; uses_model_predictions: boolean } }
  variant_test_scores: Record<string, string | number>[]
  minority_concentration: { split: string; class: string; total_beats: number; top_record: string; top_record_beats: number; top_record_share: number }[]
}
