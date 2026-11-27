// SYNTHETIC study artifacts. Every number is invented, to test that the UI renders what the files contain.
// They are written only to a temporary directory and must never be copied into backend/artifacts/.
const MODELS = ['cnn_bilstm', 'rr_fusion', 'rr_fusion_attn', 'rr_fusion_attn_shead', 'cnn', 'lstm', 'resnet1d', 'gbm', 'stacked_ensemble']

export function studyFixtures() {
  let seed = 7
  const rnd = () => { seed = (seed * 48271) % 2147483647; return seed / 2147483647 }
  const stat = () => ({ mean: rnd(), sd: 0.01 })
  const summary = () => ({
    n_beats: 10, n_patients: 2, macro_f1: stat(), accuracy: stat(),
    per_class: Object.fromEntries([...'NSVF'].map(c => [c, { precision: stat(), recall: stat(), f1: stat(), support: 5 }])),
    review: Object.fromEntries(['0.05', '0.1'].map(b => [b, { reviewed_share: rnd(), ...Object.fromEntries(['S', 'F'].map(c => [c, { model_recall_all_beats: rnd(), recall_after_perfect_review: rnd(), recall_after_random_review: rnd() }])) }])),
  })
  const baseline = () => ({ accuracy: rnd(), macro_f1: rnd(), per_class: Object.fromEntries([...'NSVF'].map(c => [c, { recall: c === 'N' ? 1 : 0 }])) })
  const majority = baseline()
  const validation = { smoke: false, primary: 'rr_fusion', majority_class: majority, results: Object.fromEntries(MODELS.map(m => [m, { rule: 'thresholds', chosen: summary(), argmax: summary() }])) }
  const frozen = { 1: { temperature: 1, thresholds: [1, 2, 1, 4], review_confidence_cutoffs: { '0.05': 0.5 } } }
  const set = () => ({ rr_fusion: summary(), cnn_bilstm: summary(), deployed_model: summary(), majority_class: baseline() })
  return {
    'selection.json': { smoke: false, primary: 'rr_fusion', candidates: { rr_fusion: { rule: 'thresholds', frozen } } },
    'validation_report.json': validation,
    'final_report.json': { smoke: false, primary: 'rr_fusion', opened_at: '2000-01-01T00:00:00+00:00', sets: { ds2_22_records: set(), ds2_without_202: set(), incart_heldout: set() } },
  }
}
