import type { Analysis, RecordInfo } from './types'
import { useApi } from './useApi'

export function useAnalysisWithRecord(id: string) {
  const analysis = useApi<Analysis>(`/analyses/${id}`)
  const record = useApi<RecordInfo>(analysis.data ? `/recordings/${analysis.data.record_id}` : null)
  return { analysis, record }
}
