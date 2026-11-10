import axios from 'axios'

export const api = axios.create({ baseURL: '/api/v1' })
export const colors: Record<string, string> = { N: '#668B70', S: '#D5A34C', V: '#C66B52', F: '#9881B6' }
export const CLASS_CODES = ['N', 'S', 'V', 'F'] as const
export type ClassCode = typeof CLASS_CODES[number]
export const CLASS_NAMES: Record<ClassCode, string> = { N: 'Normal', S: 'Supraventricular ectopic', V: 'Ventricular ectopic', F: 'Fusion' }
export const DISCLAIMER = 'Decision-support only, not a diagnosis. S and F recall are low on held-out test.'

export type Review = { id: string; original_prediction: string; reviewed_label: string | null; note: string | null; status: string }
export type Beat = { id: string; beat_index: number; sample_index: number; timestamp_seconds: number; prediction: { predicted_class: string; probabilities: number[]; confidence: number; margin: number; review_flag: boolean }; review?: Review | null }

/** Doctor-reviewed label if present, otherwise the model's own class. Never alters model confidence. */
export const effectiveClass = (b: Beat): string => b.review?.reviewed_label || b.prediction.predicted_class

export function summarise(beats: Beat[]) {
  const model: Record<ClassCode, number> = { N: 0, S: 0, V: 0, F: 0 }
  const reviewed: Record<ClassCode, number> = { N: 0, S: 0, V: 0, F: 0 }
  const confSum: Record<ClassCode, number> = { N: 0, S: 0, V: 0, F: 0 }
  const meanProb = [0, 0, 0, 0]
  let edited = 0
  let flagged = 0
  let n = 0
  for (const b of beats) {
    const m = b.prediction.predicted_class as ClassCode
    if (!(m in model)) continue
    n += 1
    model[m] += 1
    confSum[m] += b.prediction.confidence
    const e = effectiveClass(b) as ClassCode
    reviewed[e in reviewed ? e : m] += 1
    if (b.review?.reviewed_label) edited += 1
    if (b.prediction.review_flag) flagged += 1
    b.prediction.probabilities.forEach((p, i) => { if (i < 4) meanProb[i] += p })
  }
  const dom = (c: Record<ClassCode, number>) => CLASS_CODES.reduce<ClassCode>((a, k) => c[k] > c[a] ? k : a, 'N')
  const abn = (c: Record<ClassCode, number>): ClassCode | null => {
    const k = (['S', 'V', 'F'] as ClassCode[]).reduce<ClassCode>((a, x) => c[x] > c[a] ? x : a, 'S')
    return c[k] > 0 ? k : null
  }
  const dominantModel = dom(model)
  return {
    n, model, reviewed, edited, flagged, confSum,
    meanProb: meanProb.map(p => n ? p / n : 0),
    dominantModel, dominantReviewed: dom(reviewed),
    confidence: model[dominantModel] ? confSum[dominantModel] / model[dominantModel] : 0,
    abnModel: abn(model), abnReviewed: abn(reviewed),
  }
}
