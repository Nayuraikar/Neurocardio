/* eslint-disable react-refresh/only-export-components */
import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react'
import { useSearchParams } from 'react-router-dom'
import { useApi } from '../api/useApi'
import type { Analysis, RecordInfo } from '../api/types'

const KEY = 'neurocardio.selection'
type Stored = { analysis: string | null; window: number }
const read = (): Stored => {
  try { return { analysis: null, window: 0, ...JSON.parse(localStorage.getItem(KEY) ?? '{}') } } catch { return { analysis: null, window: 0 } }
}

type Ctx = {
  analysisId: string | null; windowIndex: number
  analysis: Analysis | null; record: RecordInfo | null
  loading: boolean; missing: boolean; error: string | null
  select: (analysisId: string | null, windowIndex?: number) => void
  setWindow: (i: number) => void
  link: (path: string, windowIndex?: number) => string
}
const Context = createContext<Ctx | null>(null)

/** The URL (?analysis=&window=) wins; localStorage restores it; the API confirms the id still exists. */
export function RecordingProvider({ children }: { children: ReactNode }) {
  const [params, setParams] = useSearchParams()
  const [stored, setStored] = useState<Stored>(read)
  const urlAnalysis = params.get('analysis')
  const analysisId = urlAnalysis ?? stored.analysis
  const urlWindow = params.get('window')
  const windowIndex = Math.max(0, Number(urlWindow ?? (analysisId === stored.analysis ? stored.window : 0)) || 0)

  const a = useApi<Analysis>(analysisId ? `/analyses/${analysisId}` : null)
  const r = useApi<RecordInfo>(a.data ? `/recordings/${a.data.record_id}` : null)
  const missing = a.error?.status === 404

  const persist = useCallback((next: Stored) => {
    setStored(next)
    try { localStorage.setItem(KEY, JSON.stringify(next)) } catch { /* storage may be blocked */ }
  }, [])

  useEffect(() => {
    if (a.data && (a.data.id !== stored.analysis || windowIndex !== stored.window)) persist({ analysis: a.data.id, window: windowIndex })
    if (missing && stored.analysis === analysisId) persist({ analysis: null, window: 0 })
  }, [a.data, missing, analysisId, windowIndex, stored, persist])

  const select = useCallback((id: string | null, w = 0) => {
    persist({ analysis: id, window: w })
    setParams(prev => {
      const p = new URLSearchParams(prev)
      if (id) { p.set('analysis', id); p.set('window', String(w)) } else { p.delete('analysis'); p.delete('window') }
      return p
    }, { replace: true })
  }, [persist, setParams])

  const setWindow = useCallback((i: number) => { if (analysisId) select(analysisId, i) }, [analysisId, select])
  const link = useCallback((path: string, w?: number) =>
    analysisId ? `${path}?analysis=${analysisId}&window=${w ?? windowIndex}` : path, [analysisId, windowIndex])

  const value = useMemo<Ctx>(() => ({
    analysisId, windowIndex, analysis: a.data, record: r.data, loading: a.loading, missing,
    error: a.error?.message ?? r.error?.message ?? null, select, setWindow, link,
  }), [analysisId, windowIndex, a.data, a.loading, a.error, r.data, r.error, missing, select, setWindow, link])
  return <Context.Provider value={value}>{children}</Context.Provider>
}

export function useRecording() {
  const c = useContext(Context)
  if (!c) throw new Error('useRecording needs RecordingProvider')
  return c
}
