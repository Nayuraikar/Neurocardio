export type Artifact<T> = { available: boolean; message: string | null; data: T | null; error: boolean; path?: string }
export type Stat = { mean: number; sd: number | null }
export type Summary = {
  n_beats: number; n_patients: number; seeds?: number; macro_f1: Stat; accuracy: Stat
  per_class: Record<string, { precision: Stat; recall: Stat; f1: Stat; support: number }>
  review?: Record<string, { reviewed_share: number } & Record<string, null | { model_recall_all_beats: number; recall_after_perfect_review: number; recall_after_random_review: number }>>
}
export type Majority = { accuracy: number; macro_f1: number; per_class: Record<string, { recall: number }> }
export type Study = {
  models: string[]
  data_report: null | DataReport
  data_report_problem: string | null
  selection: Artifact<{ primary: string }>
  validation: Artifact<{ primary: string; majority_class: Majority; results: Record<string, { rule: string; chosen: Summary }> }>
  final: Artifact<{ primary: string; opened_at: string; sets: Record<string, Record<string, Summary> & { majority_class: Majority }> }>
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
