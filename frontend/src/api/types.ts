export type Prediction = { predicted_class: string; probabilities: number[]; confidence: number; margin: number; review_flag: boolean; review_reason: string | null }
export type Review = { id: string; beat_id: string; original_prediction: string; reviewed_label: string | null; note: string | null; reviewer: string; status: string; created_at: string; model_version?: string }
export type Beat = { id: string; beat_index: number; sample_index: number; timestamp_seconds: number; prediction: Prediction; review: Review | null }

export type Detection = { source: string; peak_count: number; annotation_alignment: null | { matched_fraction: number; matched_count: number; annotation_count: number } }
export type AnalysisSummary = {
  stage?: string
  class_counts?: Record<string, number>
  flagged_for_review_count?: number
  detected_peak_count?: number
  valid_window_count?: number
  skipped_edge_count?: number
  detection?: Detection
  uncertainty_policy?: { name: string; provenance: string; temperature: number; tau: number; margin: number; label: string }
}
export type Analysis = {
  id: string; record_id: string; status: string; progress: number; model_version: string; model_sha256: string; config_sha256: string
  preprocessing_version: string; peak_source: string | null; beat_count: number; skipped_edge_count: number
  summary: AnalysisSummary; error_message: string | null; started_at: string; completed_at: string | null
}
export type RecordInfo = {
  id: string; filename: string; format: string; sampling_rate: number; lead: string; duration_seconds: number; sample_count: number; uploaded_at: string
  units?: string | null; signals?: string[]; has_annotation?: boolean; input_warnings?: string[]
}
export type RecordingRow = RecordInfo & { latest_analysis: Analysis | null; pending_review_count: number; analysis_reviewed: boolean }

export type Status = { api: string; model_loaded: boolean; database_ok: boolean; model_sha256: string | null; timestamp: string }
export type ModelConfig = {
  architecture: string; input_shape: string; input_dims: number[]; normalization: string; sampling_rate_hz: number; window_samples: number
  window_alignment: string; lead: string; bandpass: { type: string; order: number; low_hz: number; high_hz: number; zero_phase: boolean }
  classes: string[]; model_version: string; model_sha256: string; config_sha256: string; preprocessing_version: string
  parameter_count: number; uses_rr_features: boolean
  layers: { name: string; type: string; output_shape: (number | null)[]; params: number }[]
  pipeline: { key: string; label: string; detail: string }[]
}
export type Signal = { sampling_rate: number; start_sample: number; time_seconds: number[]; amplitude: number[]; lead: string; units: string | null; filtered: boolean }
