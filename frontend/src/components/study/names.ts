import { MODEL_LABELS, type Study } from '../../api/study'

export const modelName = (m: string) => m === 'deployed_model' ? 'Deployed CNN-BiLSTM (the app’s model)' : MODEL_LABELS[m] ?? m

/** The study's primary model as recorded by whichever artifact is readable. Selection comes first: it is what froze the choice. */
export const primaryOf = (s: Study): string | null => s.selection.data?.primary ?? s.final.data?.primary ?? s.validation.data?.primary ?? null

/** Primary first, then the others in the study code's fixed order. Never ordered by a score. */
export const ordered = (models: string[], primary: string) => [primary, ...models.filter(m => m !== primary)]

export const SHORT_SETS: Record<string, string> = {
  ds2_22_records: 'DS2, all records', ds2_without_202: 'DS2 without record 202', incart_heldout: 'INCART unseen patients',
}
