export type Explanation = {
  id: string; method: 'gradcam' | 'ig'; class: string; attributions: number[]; cached: boolean; model_version: string; created_at: string
  predicted_class: string; reference_label: string | null
  metadata: { steps?: number; baseline?: string; score_difference?: number; attribution_sum?: number; completeness_error?: number; gradient_l2_norm?: number; target_layer?: string }
  top_regions: { start_sample: number; end_sample: number; start_ms: number; end_ms: number; share_of_total_absolute: number; signed_share: number; contains_centre_beat: boolean }[]
}

export const explanationUrl = (a: string, beat: string, cls: string, method: string) => `/analyses/${a}/explanations?beat=${beat}&class=${cls}&method=${method}`
