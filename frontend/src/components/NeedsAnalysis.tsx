import type { ReactNode } from 'react'
import { Link } from 'react-router-dom'
import { useApi } from '../api/useApi'
import type { Analysis, RecordInfo, RecordingRow } from '../api/types'
import { useRecording } from '../context/RecordingContext'
import { Empty, ErrorState, Load, Skeleton } from './states'
import { when } from './format'
import { Table } from './ui'

export function AnalysisPicker() {
  const q = useApi<RecordingRow[]>('/recordings')
  const { select } = useRecording()
  return <Load q={q}>{rows => {
    const done = rows.filter(r => r.latest_analysis?.status === 'completed')
    if (!done.length) return <Empty title="No completed analysis yet"><Link className="btn primary" to="/studio">Upload a recording</Link></Empty>
    return <Table head={['Recording', 'Analysed', 'Windows', '']} rows={done.slice(0, 15).map(r => [
      r.filename, when(r.latest_analysis!.completed_at), r.latest_analysis!.beat_count,
      <button type="button" key="b" onClick={() => select(r.latest_analysis!.id)}>Open</button>])} />
  }}</Load>
}

/** Renders children once a completed analysis is selected; otherwise explains what is missing. */
export function NeedsAnalysis({ children }: { children: (a: Analysis, r: RecordInfo) => ReactNode }) {
  const rec = useRecording()
  if (rec.analysisId && !rec.missing) {
    if (rec.error) return <ErrorState error={{ status: null, message: rec.error }} />
    if (!rec.analysis || !rec.record) return <Skeleton lines={6} />
    if (rec.analysis.status === 'completed') return <>{children(rec.analysis, rec.record)}</>
  }
  return <>
    {rec.missing && <ErrorState error={{ status: 404, message: 'The selected analysis no longer exists.' }} />}
    {rec.analysis && rec.analysis.status !== 'completed' && <Empty title={`This analysis is ${rec.analysis.status}`}>{rec.analysis.error_message}</Empty>}
    <section className="card"><div className="card-head"><h2>Choose an analysed recording</h2></div><AnalysisPicker /></section>
  </>
}
