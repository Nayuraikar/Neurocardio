import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { get, http, post, toApiError, type ApiError } from '../api/client'
import type { Analysis, ModelConfig, RecordInfo } from '../api/types'
import { useApi } from '../api/useApi'
import { useBeats } from '../api/useBeats'
import { ClassDistribution } from '../components/ClassDistribution'
import { secs, when } from '../components/format'
import { PipelineSteps } from '../components/PipelineSteps'
import { SplitNote } from '../components/SplitNote'
import { ErrorState, Load, PartialNotice, ReviewReady, Skeleton, SuccessBanner } from '../components/states'
import { Card, KeyValues, PageHead } from '../components/ui'
import { WindowStages } from '../components/WindowStages'
import { WeaknessNote } from '../components/WeaknessNote'
import { WindowTable } from '../components/WindowTable'
import { Trace } from '../components/waveform/Trace'
import { useRecording } from '../context/RecordingContext'
import { csvColumns, explain } from './studioErrors'

const Rejected = ({ error, onRetry }: { error: ApiError; onRetry?: () => void }) => {
  const e = explain(error.message)
  return <ErrorState title={e.title} error={{ ...error, message: `${e.fix} Server message: ${error.message}` }} onRetry={onRetry} />
}

function Upload({ onUploaded }: { onUploaded: (r: RecordInfo) => void }) {
  const [files, setFiles] = useState<File[]>([])
  const [columns, setColumns] = useState<string[]>([])
  const [column, setColumn] = useState('')
  const [rate, setRate] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<ApiError | null>(null)
  const [over, setOver] = useState(false)
  const csv = files.find(f => f.name.toLowerCase().endsWith('.csv'))
  const set = async (next: File[]) => {
    setFiles(next); setError(null)
    const c = next.find(f => f.name.toLowerCase().endsWith('.csv'))
    const cols = c ? await csvColumns(c) : []
    setColumns(cols); setColumn(cols.includes('MLII') ? 'MLII' : cols[0] ?? '')
  }
  const add = (list: FileList | null) => {
    if (!list) return
    const byName = new Map(files.map(f => [f.name, f]))
    for (const f of Array.from(list)) byName.set(f.name, f)
    void set([...byName.values()])
  }
  const send = async () => {
    setBusy(true); setError(null)
    const body = new FormData()
    files.forEach(f => body.append('files', f))
    if (csv) { if (rate) body.append('sampling_rate', rate); if (column) body.append('lead', column) }
    try { onUploaded((await http.post<RecordInfo>('/upload', body)).data) }
    catch (e) { setError(toApiError(e)) }
    finally { setBusy(false) }
  }
  return <Card title="1 · Upload">
    <div className={`drop ${over ? 'over' : ''}`} onDragOver={e => { e.preventDefault(); setOver(true) }} onDragLeave={() => setOver(false)}
      onDrop={e => { e.preventDefault(); setOver(false); add(e.dataTransfer.files) }}>
      <p><strong>Drop files here</strong>, or choose them.</p>
      <p className="small muted">WFDB: one .hea and its .dat, plus an optional .atr. CSV: one file with a header row. 360 Hz only.</p>
      <input type="file" multiple accept=".hea,.dat,.atr,.csv" aria-label="Choose recording files" data-testid="file-input" onChange={e => { add(e.target.files); e.target.value = '' }} />
    </div>
    {files.length > 0 && <>
      <ul>{files.map(f => <li key={f.name}><span className="num">{f.name}</span> · {(f.size / 1024).toFixed(0)} KB <button type="button" onClick={() => void set(files.filter(x => x !== f))}>Remove</button></li>)}</ul>
      <button type="button" onClick={() => void set([])}>Replace all</button>
    </>}
    {csv && <div className="row">
      <label className="field">Sampling rate (Hz)<input type="number" value={rate} onChange={e => setRate(e.target.value)} placeholder="360" /></label>
      <label className="field">Signal column<select value={column} onChange={e => setColumn(e.target.value)}>{columns.map(c => <option key={c}>{c}</option>)}</select></label>
    </div>}
    {error && <Rejected error={error} />}
    <p><button type="button" className="primary" disabled={!files.length || busy} onClick={() => void send()}>{busy ? 'Validating…' : 'Upload and validate'}</button></p>
  </Card>
}

function Metadata({ record }: { record: RecordInfo }) {
  return <>
    <KeyValues rows={[['File', record.filename], ['Format', record.format.toUpperCase()], ['Lead', record.lead], ['Sampling rate', `${record.sampling_rate} Hz`],
      ['Duration', secs(record.duration_seconds)], ['Samples', record.sample_count.toLocaleString()], ['Units', record.units ?? 'unknown'],
      ['Annotation file', record.has_annotation ? 'present: beat positions come from it' : 'none: XQRS detector is used'], ['Uploaded', when(record.uploaded_at)], ['Recording ID', record.id]]} />
    {record.input_warnings?.map(w => <PartialNotice key={w}>{w}</PartialNotice>)}
  </>
}

function RawPreview({ record }: { record: RecordInfo }) {
  const raw = useApi<{ amplitude: number[] }>(`/recordings/${record.id}/signal?count=3600`)
  const filtered = useApi<{ amplitude: number[] }>(`/recordings/${record.id}/signal?count=3600&filtered=true`)
  const unit = record.units ?? 'signal units'
  return <div className="stack">
    <Load q={raw}>{d => <Trace values={d.amplitude} fs={record.sampling_rate} label="Raw, first 10 s" unit={unit} />}</Load>
    <Load q={filtered}>{d => <Trace values={d.amplitude} fs={record.sampling_rate} label="Bandpass filtered, first 10 s" unit={unit} />}</Load>
  </div>
}

function Results({ analysis, record }: { analysis: Analysis; record: RecordInfo }) {
  const beats = useBeats(analysis.id)
  const { link } = useRecording()
  const s = analysis.summary
  const source = analysis.peak_source === 'wfdb_annotation' ? 'WFDB annotation file' : 'XQRS detector'
  return <>
    <SuccessBanner>Analysis complete. These are window-level model outputs, not a finding about the recording or the patient.</SuccessBanner>
    {record.input_warnings?.map(w => <PartialNotice key={w}>{w}</PartialNotice>)}
    <SplitNote record={record} />
    <WeaknessNote />
    <div className="grid-2">
      <Card title="Windows by model class">
        <ClassDistribution counts={s.class_counts ?? {}} source="analysis#summary.class_counts" />
        <p><strong>V-class windows present: <span className="num">{s.class_counts?.V ?? 0}</span></strong></p>
      </Card>
      <Card title="Run details">
        <KeyValues rows={[['Windows analysed', analysis.beat_count.toLocaleString()], ['Flagged for review', String(s.flagged_for_review_count ?? 0)],
          ['Beat positions from', source], ['Edge beats skipped', String(analysis.skipped_edge_count)], ['Model version', analysis.model_version], ['Completed', when(analysis.completed_at)]]} />
        <div className="row"><Link className="btn primary" to={link('/viewer')}>Open ECG Viewer</Link><Link className="btn" to={link('/quality')}>Signal quality</Link><Link className="btn" to={link('/confidence')}>Confidence</Link><Link className="btn" to={link('/xai')}>XAI Studio</Link><Link className="btn" to={`/history/${analysis.id}/report`}>Report</Link></div>
      </Card>
    </div>
    {!!s.flagged_for_review_count && <ReviewReady pending={s.flagged_for_review_count} to={`/history/${analysis.id}`} />}
    <Card title="Every window"><Load q={beats}>{b => <WindowTable beats={b} />}</Load></Card>
  </>
}

export default function AnalysisStudio() {
  const rec = useRecording()
  const config = useApi<ModelConfig>('/model/config')
  const [uploaded, setUploaded] = useState<RecordInfo | null>(null)
  const [running, setRunning] = useState<Analysis | null>(null)
  const [error, setError] = useState<ApiError | null>(null)
  const [fresh, setFresh] = useState(false)
  const stored = !fresh && !uploaded && !running && rec.analysis && rec.record ? { analysis: rec.analysis, record: rec.record } : null
  const done = running?.status === 'completed' && uploaded ? { analysis: running, record: uploaded } : stored?.analysis.status === 'completed' ? stored : null
  const stage = done ? 3 : uploaded ? 2 : 1
  const select = rec.select
  const pollId = running?.status === 'running' ? running.id : null

  useEffect(() => {
    if (!pollId) return
    let live = true
    const timer = window.setInterval(() => {
      get<Analysis>(`/analyses/${pollId}`).then(a => {
        if (!live) return
        setRunning(a); setError(null)
        if (a.status === 'completed') select(a.id, 0)
      }, e => { if (live) setError(toApiError(e)) })
    }, 500)
    return () => { live = false; window.clearInterval(timer) }
  }, [pollId, select])

  const start = async () => {
    if (!uploaded) return
    setError(null)
    try { setRunning(await post<Analysis>('/analyze', undefined, { record_id: uploaded.id })) }
    catch (e) { setError(toApiError(e)) }
  }
  const reset = () => { setFresh(true); setUploaded(null); setRunning(null); setError(null); select(null) }
  const status = done ? 'completed' : running?.status === 'failed' ? 'failed' : running ? 'running' : 'idle'

  return <>
    <PageHead title="Analysis Studio" sub="Upload, process, then read the window-level results."
      actions={stage > 1 || done ? <button type="button" onClick={reset}>New upload</button> : undefined} />
    <ol className="stages" aria-label="Stages">{['Upload', 'Process', 'Results'].map((s, i) => <li key={s} aria-current={stage === i + 1 ? 'step' : undefined}>{i + 1}. {s}</li>)}</ol>
    {stage === 1 && <Upload onUploaded={r => { setFresh(true); select(null); setUploaded(r) }} />}
    {stage === 2 && uploaded && <>
      <SuccessBanner>The file passed upload validation.</SuccessBanner>
      <Card title="Recording"><Metadata record={uploaded} /></Card>
      <div className="grid-2">
        <Card title="2 · Process">
          <Load q={config}>{c => <PipelineSteps pipeline={c.pipeline} stage={running?.summary.stage} status={status} />}</Load>
          {running?.status === 'failed' && <Rejected error={{ status: null, message: running.error_message ?? 'analysis failed' }} onRetry={() => { setRunning(null) }} />}
          {error && <Rejected error={error} />}
          {!running && <p><button type="button" className="primary" onClick={() => void start()}>Analyse recording</button></p>}
          {status === 'running' && <Skeleton lines={1} label="Analysis running" />}
        </Card>
        <Card title="Signal"><RawPreview record={uploaded} /></Card>
      </div>
    </>}
    {done && <>
      <Results analysis={done.analysis} record={done.record} />
      <PreviewWindow analysis={done.analysis} record={done.record} />
    </>}
  </>
}

function PreviewWindow({ analysis, record }: { analysis: Analysis; record: RecordInfo }) {
  const beats = useBeats(analysis.id)
  return <Card title="Raw to model input: first window"><Load q={beats}>{b => b[0] ? <WindowStages record={record} sample={b[0].sample_index} /> : <p>No windows.</p>}</Load></Card>
}
