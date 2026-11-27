import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { get, http, post, toApiError } from '../api/client'
import type { Analysis, RecordInfo } from '../api/types'
import { ClassDistribution } from '../components/ClassDistribution'
import { PartialNotice, SuccessBanner, ErrorState } from '../components/states'
import { Card, PageHead, Table } from '../components/ui'
import { WeaknessNote } from '../components/WeaknessNote'
import { csvColumns } from './studioErrors'
import { groupFiles, type Job } from './batchGroups'

type Result = { state: 'waiting' | 'uploading' | 'analysing' | 'done' | 'failed' | 'skipped'; message?: string; analysis?: Analysis; warnings?: string[] }

export default function Batch() {
  const [jobs, setJobs] = useState<Job[]>([])
  const [results, setResults] = useState<Record<string, Result>>({})
  const [rate, setRate] = useState('360')
  const [running, setRunning] = useState(false)
  const stop = useRef(false)
  const alive = useRef(true)
  useEffect(() => { alive.current = true; return () => { alive.current = false; stop.current = true } }, [])

  const run = async () => {
    stop.current = false; setRunning(true)
    for (const job of jobs) {
      const set = (r: Result) => setResults(all => ({ ...all, [job.name]: r }))
      if (stop.current) { set({ state: 'skipped', message: 'Stopped before this file' }); continue }
      if (job.problem) { set({ state: 'failed', message: job.problem }); continue }
      try {
        set({ state: 'uploading' })
        const body = new FormData()
        job.files.forEach(f => body.append('files', f))
        if (job.csv) {
          const columns = await csvColumns(job.files[0])
          body.append('sampling_rate', rate); body.append('lead', columns.includes('MLII') ? 'MLII' : columns[0] ?? '')
        }
        const record = (await http.post<RecordInfo>('/upload', body)).data
        let analysis = await post<Analysis>('/analyze', undefined, { record_id: record.id })
        set({ state: 'analysing', analysis, warnings: record.input_warnings })
        while (analysis.status === 'running' && alive.current) {
          await new Promise(r => window.setTimeout(r, 700))
          analysis = await get<Analysis>(`/analyses/${analysis.id}`)
        }
        set(analysis.status === 'completed' ? { state: 'done', analysis, warnings: record.input_warnings } : { state: 'failed', message: analysis.error_message ?? 'analysis did not complete', analysis })
      } catch (e) { set({ state: 'failed', message: toApiError(e).message }) }
    }
    setRunning(false)
  }

  const all = Object.values(results)
  const ok = all.filter(r => r.state === 'done').length, bad = all.filter(r => r.state === 'failed').length
  const finished = !running && all.length > 0
  return <>
    <PageHead title="Batch Analysis" sub="Several recordings, processed one at a time. One failure does not stop the others." />
    <WeaknessNote />
    <Card title="Files">
      <input type="file" multiple accept=".hea,.dat,.atr,.csv" aria-label="Choose files for batch analysis" data-testid="batch-input" disabled={running}
        onChange={e => { setJobs(groupFiles(Array.from(e.target.files ?? []))); setResults({}) }} />
      <p className="small muted">WFDB files are grouped by base name. For CSV files the MLII column is used, or the first column if there is none.</p>
      {jobs.some(j => j.csv) && <label className="field" style={{ maxWidth: 200 }}>CSV sampling rate (Hz)<input type="number" value={rate} onChange={e => setRate(e.target.value)} /></label>}
      <div className="row"><button type="button" className="primary" disabled={!jobs.length || running} onClick={() => void run()}>Analyse {jobs.length} recording{jobs.length === 1 ? '' : 's'}</button>
        {running && <button type="button" onClick={() => { stop.current = true }}>Stop after the current file</button>}</div>
    </Card>
    {finished && ok > 0 && bad === 0 && <SuccessBanner>All attempted recordings completed.</SuccessBanner>}
    {finished && ok > 0 && bad > 0 && <PartialNotice><strong>Partial result:</strong> {ok} completed and {bad} failed. See the reason beside each failure.</PartialNotice>}
    {finished && ok === 0 && bad > 0 && <ErrorState title="No recording could be analysed" error={{ status: null, message: 'See the reason beside each file.' }} />}
    {jobs.length > 0 && <Card title="Progress"><Table head={['Recording', 'Files', 'State', 'Windows', 'By model class', '']} rows={jobs.map(j => {
      const r = results[j.name] ?? { state: 'waiting' as const, message: j.problem ?? undefined }
      return [j.name, <span key="f" className="small">{j.files.map(f => f.name).join(', ')}</span>,
        <div key="s"><span className={`tag ${r.state === 'done' ? 'good' : r.state === 'failed' ? 'poor' : ''}`}>{r.state}</span>{r.message && <div className="small">{r.message}</div>}{r.warnings?.map(w => <div key={w} className="small muted">{w}</div>)}</div>,
        r.state === 'done' ? r.analysis!.beat_count.toLocaleString() : '—',
        r.state === 'done' ? <div key="d" style={{ minWidth: 200 }}><ClassDistribution counts={r.analysis!.summary.class_counts ?? {}} source={`analysis#${r.analysis!.id}.class_counts`} /></div> : '—',
        r.state === 'done' ? <Link key="l" to={`/viewer?analysis=${r.analysis!.id}&window=0`}>Open</Link> : '']
    })} /></Card>}
  </>
}
