import { useEffect, useState } from 'react'
import { get, toApiError, type ApiError } from '../../api/client'
import type { Signal } from '../../api/types'

export const MAX_SPAN = 20 // seconds; one 10,000-sample chunk covers any view up to this wide

export type View = { a: number; span: number }
export const clampView = (a: number, span: number, duration: number): View => {
  const s = Math.min(Math.max(span, 1), Math.min(MAX_SPAN, duration))
  return { a: Math.min(Math.max(a, 0), Math.max(0, duration - s)), span: s }
}

/** The signal chunk covering [a, a + MAX_SPAN]. The previous chunk stays visible while the next one loads. */
export function useChunk(recordId: string, fs: number, a: number, filtered: boolean) {
  const start = Math.max(0, Math.round((Math.floor(a / 5) * 5 - 2) * fs))
  const key = `${recordId}|${start}|${filtered}`
  const [state, setState] = useState<{ key: string; data: Signal | null; error: ApiError | null }>({ key: '', data: null, error: null })
  const [tick, setTick] = useState(0)
  useEffect(() => {
    let live = true
    get<Signal>(`/recordings/${recordId}/signal`, { start_sample: start, count: 10000, filtered }).then(
      data => { if (live) setState({ key, data, error: null }) },
      e => { if (live) setState(s => ({ key, data: s.data, error: toApiError(e) })) })
    return () => { live = false }
  }, [recordId, start, filtered, key, tick])
  const usable = state.key.startsWith(`${recordId}|`) && state.key.endsWith(`|${filtered}`)
  return { chunk: usable ? state.data : null, error: state.key === key ? state.error : null, loading: state.key !== key, retry: () => setTick(t => t + 1) }
}
