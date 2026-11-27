import { useCallback, useEffect, useState } from 'react'
import { get, toApiError, type ApiError } from './client'

type State<T> = { key: string; data: T | null; error: ApiError | null }

/** GET with loading / error / retry. Pass null to skip. Stale responses are ignored. */
export function useApi<T>(url: string | null) {
  const [tick, setTick] = useState(0)
  const key = `${url}#${tick}`
  const [state, setState] = useState<State<T>>({ key: '', data: null, error: null })
  useEffect(() => {
    if (!url) return
    let live = true
    get<T>(url).then(
      data => { if (live) setState({ key, data, error: null }) },
      e => { if (live) setState({ key, data: null, error: toApiError(e) }) })
    return () => { live = false }
  }, [url, key])
  const reload = useCallback(() => setTick(t => t + 1), [])
  const current = state.key === key
  // keep previous data of the same url visible while reloading
  const sameUrl = state.key.startsWith(`${url}#`)
  return { data: url && sameUrl ? state.data : null, error: url && current ? state.error : null, loading: !!url && !current, reload }
}
