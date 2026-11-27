import type { RecordInfo } from '../api/types'
import { PartialNotice } from './states'

const TEXT: Record<string, string> = {
  train: 'was used to train this model. Results on it are optimistic and do not show how the model does on new patients.',
  validation: 'was used to select this model. Results on it are optimistic.',
  test: 'is in this model’s held-out test split.',
}

export function SplitNote({ record }: { record: RecordInfo }) {
  const part = record.model_split_by_name
  return part ? <PartialNotice>The file name matches an MIT-BIH record that {TEXT[part]} (Matched by name only.)</PartialNotice> : null
}
