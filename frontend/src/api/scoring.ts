import { CLASS_CODES } from '../theme/theme'

/** Scores for a small labelled set. Accuracy is always returned together with the majority-class baseline and macro-F1. */
export function score(pairs: { truth: string; predicted: string }[]) {
  const matrix = CLASS_CODES.map(t => CLASS_CODES.map(p => pairs.filter(x => x.truth === t && x.predicted === p).length))
  const per = CLASS_CODES.map((c, i) => {
    const support = matrix[i].reduce((a, b) => a + b, 0)
    const called = matrix.reduce((a, row) => a + row[i], 0)
    const tp = matrix[i][i]
    return { code: c, support, tp, fp: called - tp, fn: support - tp, recall: support ? tp / support : null, precision: called ? tp / called : null, f1: support + called ? 2 * tp / (support + called) : 0 }
  })
  const n = pairs.length
  return {
    n, matrix, per,
    accuracy: n ? per.reduce((a, p) => a + p.tp, 0) / n : null,
    baseline: n ? Math.max(...per.map(p => p.support)) / n : null,
    macroF1: n ? per.reduce((a, p) => a + p.f1, 0) / per.length : null,
  }
}
