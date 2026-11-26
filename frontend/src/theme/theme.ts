import data from './classColors.json'

export type ClassCode = 'N' | 'S' | 'V' | 'F'
export const CLASS_CODES = data.order as ClassCode[]
export const CLASSES = data.classes as Record<ClassCode, { name: string; colorName: string; fill: string; text: string }>
export const classFill = (c: string) => CLASSES[c as ClassCode]?.fill ?? '#777'
export const classText = (c: string) => CLASSES[c as ClassCode]?.text ?? '#333'
export const className_ = (c: string) => CLASSES[c as ClassCode]?.name ?? c

/** Fixed interface wording. Numbers never go here. */
export const NOTICE = 'Research prototype. Output is window/beat-level model classification, not a diagnosis.'
export const XAI_NOTICE = 'Explains model behaviour, not medical causation. A highlighted region is not evidence of a clinical cause.'
export const SOFTMAX_NOTICE = 'Softmax probability is a model score. It is not clinical certainty.'
export const REVIEW_BOUND_NOTICE = '“Recall after review” assumes a reviewer who corrects every reviewed beat. Treat it as an upper bound, not a measured result.'
export const INCART_NOTICE = 'INCART results are for unseen patients, not an unseen database: other INCART patients were in training.'
