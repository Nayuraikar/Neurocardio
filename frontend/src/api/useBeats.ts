import { useApi } from './useApi'
import type { Beat } from './types'

/** All windows of an analysis. The API returns at most 10,000 per request. */
export function useBeats(analysisId: string | null) {
  return useApi<Beat[]>(analysisId ? `/analyses/${analysisId}/beats?limit=10000` : null)
}
