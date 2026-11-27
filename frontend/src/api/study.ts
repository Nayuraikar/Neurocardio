export type Artifact<T> = { available: boolean; message: string | null; data: T | null; error: boolean; path?: string }
export type Stat = { mean: number; sd: number | null }
export type ReviewClass = null | { model_recall_all_beats: number; recall_after_perfect_review: number; recall_after_random_review: number }
export type Review = Record<string, { reviewed_share: number } & Record<string, ReviewClass>>
export type Summary = {
  n_beats: number; n_patients: number; seeds?: number; macro_f1: Stat; accuracy: Stat
  per_class: Record<string, { precision: Stat; recall: Stat; f1: Stat; support: number }>
  review?: Review; review_with_validation_cutoff?: Review
  bootstrap_ci95_patients?: Record<string, [number, number]>
  per_patient?: { patient: string; S: { support: number }; F: { support: number } }[]
}
export type Majority = { accuracy: number; macro_f1: number; per_class: Record<string, { precision: number; recall: number; f1: number }> }
export type GainKey = 'macro_f1' | 'S_f1' | 'F_f1' | 'S_recall' | 'F_recall'
export type Gain = { gain: number; bootstrap_ci95: [number, number]; leave_one_record_out_min_gain: number; meaningful: boolean }
export type TestSet = Record<string, Summary> & {
  majority_class: Majority
  primary_vs_deployed: Record<GainKey, Gain> & { headline_comparison: boolean }
  primary_vs_majority_macro_f1_gain: number
}
export type ValidationData = { primary: string; majority_class: Majority; results: Record<string, { rule: string; chosen: Summary }> }
export type FinalData = { primary: string; opened_at: string; scored: string[]; sets: Record<string, TestSet> }
export type Study = {
  models: string[]
  data_report: null | DataReport
  data_report_problem: string | null
  provenance: null | { primary_matches: boolean; selection_hash_matches: boolean; selection_dated_before_test: boolean }
  selection: Artifact<{ primary: string }>
  validation: Artifact<ValidationData>
  final: Artifact<FinalData>
}
export type SplitCounts = { records: number; patients: number; windows: Record<string, number>; total: number; concentration?: Record<string, { total: number; patients_with_any: number; top_patient: string; top_share: number }>; by_database?: Record<string, { records: number; patients: number; windows: Record<string, number>; total: number }> }
export type DataReport = {
  splits: Record<string, SplitCounts>
  databases: Record<string, { records: number; patients: number; windows: Record<string, number>; total: number; source_fs_hz: number }>
  lead: Record<string, string>
  excluded_records: Record<string, Record<string, string>>
}
export const MODEL_LABELS: Record<string, string> = {
  cnn_bilstm: 'CNN-BiLSTM', rr_fusion: 'CNN-BiLSTM + RR fusion', rr_fusion_attn: 'RR fusion + attention', rr_fusion_attn_shead: 'RR fusion + attention + S head',
  cnn: 'CNN only', lstm: 'LSTM only', resnet1d: 'ResNet1D', gbm: 'Gradient-boosted trees (GBM)', stacked_ensemble: 'Stacked ensemble',
}
export const RR_MODELS = ['rr_fusion', 'rr_fusion_attn', 'rr_fusion_attn_shead', 'gbm', 'stacked_ensemble']
