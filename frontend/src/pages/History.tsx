import { useState } from 'react'
import { Link } from 'react-router-dom'
import { http, toApiError, type ApiError } from '../api/client'
import type { Analysis } from '../api/types'
import { useApi } from '../api/useApi'
import { when } from '../components/format'
import { Empty, ErrorState, Load } from '../components/states'
import { Card, PageHead, Table } from '../components/ui'
import { useRecording } from '../context/RecordingContext'

export type HistoryRow = Analysis & { filename: string; pending_review_count: number; analysis_reviewed: boolean; needs_review: boolean }
const FILTERS: Record<string, (r: HistoryRow) => boolean> = {
  All: () => true, Completed: r => r.status === 'completed', Failed: r => r.status === 'failed', 'Needs review': r => r.needs_review,
}

function Rows({ rows, reload }: { rows: HistoryRow[]; reload: () => void }) {
  const [filter, setFilter] = useState('All')
  const [search, setSearch] = useState('')
  const [error, setError] = useState<ApiError | null>(null)
  const rec = useRecording()
  const shown = rows.filter(r => FILTERS[filter](r) && r.filename.toLowerCase().includes(search.toLowerCase()))
  const remove = async (r: HistoryRow) => {
    if (!window.confirm(`Delete the analysis of ${r.filename}? Its predictions, reviews and reports are removed. This cannot be undone.`)) return
    try { await http.delete(`/analyses/${r.id}`); if (rec.analysisId === r.id) rec.select(null); setError(null); reload() }
    catch (e) { setError(toApiError(e)) }
  }
  return <Card>
    <div className="row between">
      <div className="row" role="group" aria-label="Filter">{Object.keys(FILTERS).map(f => <button type="button" key={f} aria-pressed={filter === f} onClick={() => setFilter(f)}>{f} ({rows.filter(FILTERS[f]).length})</button>)}</div>
      <input type="search" placeholder="Search file name" aria-label="Search file name" value={search} onChange={e => setSearch(e.target.value)} />
    </div>
    {error && <ErrorState title="Delete failed" error={error} />}
    {!rows.length ? <Empty title="No analyses yet"><Link className="btn primary" to="/studio">Analyse a recording</Link></Empty>
      : !shown.length ? <Empty title="Nothing matches this filter" />
        : <Table head={['Recording', 'Started', 'Status', 'Windows', 'Pending review', 'Model', '']} rows={shown.map(r => [r.filename, when(r.started_at),
          <span key="s">{r.status}{r.analysis_reviewed && <span className="tag good"> marked reviewed</span>}{r.error_message && <div className="small muted">{r.error_message}</div>}</span>,
          r.beat_count.toLocaleString(), r.pending_review_count, r.model_version,
          <div className="row" key="a">{r.status === 'completed' && <><Link to={`/viewer?analysis=${r.id}&window=0`}>Open</Link><Link to={`/history/${r.id}`}>Review</Link><Link to={`/history/${r.id}/report`}>Report</Link></>}<button type="button" onClick={() => void remove(r)}>Delete</button></div>])} />}
  </Card>
}

export default function History() {
  const q = useApi<HistoryRow[]>('/history')
  return <><PageHead title="History & Review" sub="Stored analyses. Opening one shows its saved predictions; nothing is recomputed." />
    <Load q={q} lines={8}>{rows => <Rows rows={rows} reload={q.reload} />}</Load></>
}
