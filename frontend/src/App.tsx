import { useEffect, useRef, useState, type ReactNode } from 'react'
import { BrowserRouter, Link, Navigate, Route, Routes, useLocation, useParams, useSearchParams } from 'react-router-dom'
import { Activity, ArrowRight, BrainCircuit, ClipboardCheck, FileText, ScanLine, Upload, Waves } from 'lucide-react'
import HomePage from './HomePage'
import TopNav from './Nav'
import EcgPlot from './EcgPlot'
import ExplainDrawer from './ExplainDrawer'
import ModelFlow from './ModelFlow'
import UploadCard from './UploadCard'
import { api, colors, CLASS_CODES, CLASS_NAMES, DISCLAIMER, effectiveClass, summarise, type Beat, type Review, type ClassCode } from './shared'
import './App.css'
import './pulmify.css'

const SIG_COUNT = 10000
type Analysis = { id: string; record_id: string; status: string; model_version: string; preprocessing_version: string; beat_count: number; started_at: string; completed_at: string | null; skipped_edge_count: number; summary: { class_counts?: Record<string, number>; abnormal_beat_count?: number; flagged_for_review_count?: number }; error_message?: string; peak_source?: string }

function Shell({ children }: { children: ReactNode }) {
  return <div className="pf-shell"><TopNav/><main className="pf-main">{children}</main><footer className="pf-footer"><span>NeuroCardio · Understand the rhythm. Explore the signal.</span><span>Research use only · Not a clinical diagnosis</span></footer></div>
}
function Disclaimer() { return <p className="pf-disclaimer" role="note">{DISCLAIMER}</p> }
function LatestAnalysis() {
  const [to, setTo] = useState<string | null>(null)
  useEffect(() => { api.get('/analyses').then(r => { const a = (r.data as Analysis[]).find(x => x.status === 'completed'); setTo(a ? `/app/analyses/${a.id}` : '/app/dashboard#upload') }).catch(() => setTo('/app/dashboard')) }, [])
  return to ? <Navigate to={to} replace/> : <Page tag="ANALYSIS" title="Opening latest analysis…"><Empty title="Loading" text="Looking for your most recent completed analysis."/></Page>
}
function Page({ tag,title,subtitle,children }: { tag:string;title:string;subtitle?:string;children:ReactNode }) { return <section className="page"><span className="pf-corner">CARDIOLOGY</span><div className="heading"><div className="eyebrow">{tag}</div><h1>{title}</h1>{subtitle&&<p>{subtitle}</p>}</div>{children}</section> }
function Empty({title,text}:{title:string;text:string}) { return <div className="empty"><Activity/><b>{title}</b><span>{text}</span></div> }
function ErrorBox({text}:{text:string}) { return <div className="error" role="alert">{text}</div> }
function Metric({label,value,note}:{label:string;value:string;note:string}) { return <article className="metric"><span>{label}</span><b>{value}</b><small>{note}</small></article> }
type ECGRecord = { id: string; filename: string; uploaded_at: string; duration_seconds: number }
type ReviewBeat = Beat & { review?: unknown | null }
type DashboardReviewCounts = { flagged: number; pending: number }

function Dash() {
  const loc = useLocation()
  useEffect(() => { if (loc.hash === '#upload') document.getElementById('upload')?.scrollIntoView({ behavior: 'smooth' }) }, [loc.hash])
  const [analyses, setAnalyses] = useState<Analysis[] | null>(null)
  const [records, setRecords] = useState<ECGRecord[] | null>(null)
  const [analysisError, setAnalysisError] = useState('')
  const [reviewCounts, setReviewCounts] = useState<Record<string, DashboardReviewCounts> | null>(null)
  const [reviewError, setReviewError] = useState('')

  useEffect(() => {
    Promise.all([api.get('/analyses'), api.get('/ecg/records', { params: { limit: 500 } })])
      .then(([analysisResponse, recordResponse]) => {
        setAnalyses(analysisResponse.data)
        setRecords(recordResponse.data)
      })
      .catch(error => setAnalysisError(error.response?.data?.detail || 'Local API unavailable. Start the backend and refresh.'))
  }, [])

  useEffect(() => {
    if (!analyses) return
    let cancelled = false
    const completed = analyses.filter(analysis => analysis.status === 'completed')
    Promise.all(completed.map(async analysis => {
      let offset = 0
      let flagged = 0
      let pending = 0
      while (true) {
        const response = await api.get<ReviewBeat[]>(`/analyses/${analysis.id}/beats`, { params: { offset, limit: 1000 } })
        const beats: ReviewBeat[] = response.data
        for (const beat of beats) {
          if (beat.prediction.review_flag) {
            flagged += 1
            if (!beat.review) pending += 1
          }
        }
        if (beats.length < 1000) break
        offset += beats.length
      }
      return [analysis.id, { flagged, pending }] as const
    }))
      .then(entries => { if (!cancelled) setReviewCounts(Object.fromEntries(entries)) })
      .catch(error => { if (!cancelled) setReviewError(error.response?.data?.detail || error.message || 'Review queue data is unavailable.') })
    return () => { cancelled = true }
  }, [analyses])

  const recordById = new Map((records || []).map(record => [record.id, record]))
  const completed = analyses?.filter(analysis => analysis.status === 'completed') || []
  const analyzedRecordCount = analyses ? new Set(analyses.filter(analysis => analysis.status === 'completed').map(analysis => analysis.record_id)).size : null
  const beatsProcessed = analyses ? completed.reduce((sum, analysis) => sum + analysis.beat_count, 0) : null
  const pendingReviews = reviewCounts ? Object.values(reviewCounts).reduce((sum, counts) => sum + counts.pending, 0) : null
  const recent = analyses?.slice(0, 6) || []
  const classTotals = completed.reduce((acc, a) => { CLASS_CODES.forEach(c => { acc[c] += a.summary.class_counts?.[c] || 0 }); return acc }, { N: 0, S: 0, V: 0, F: 0 })
  const classSum = CLASS_CODES.reduce((t, c) => t + classTotals[c], 0)
  const flaggedTotal = reviewCounts ? Object.values(reviewCounts).reduce((t, c) => t + c.flagged, 0) : analyses ? completed.reduce((t, a) => t + (a.summary.flagged_for_review_count || 0), 0) : null
  const metricValue = (value: number | null) => value === null ? '—' : value.toLocaleString()
  const formatDate = (value: string | null) => value ? new Intl.DateTimeFormat(undefined, { dateStyle: 'medium' }).format(new Date(value)) : 'Date unavailable'
  const actions = [
    { title: 'New ECG analysis', detail: 'Validate a signal and start a run', href: '/app/new', icon: Upload, color: 'cloud' },
    { title: 'Recordings', detail: 'Browse local ECG files', href: '/app/recordings', icon: Waves, color: 'pistachio' },
    { title: 'Human review', detail: 'Open the flagged beat queue', href: '/app/reviews', icon: ClipboardCheck, color: 'blush' },
    { title: 'Explainability', detail: 'Choose a result to inspect', href: '/app/recordings', icon: ScanLine, color: 'lavender' },
    { title: 'Reports', detail: 'Export completed analyses', href: '/app/reports', icon: FileText, color: 'butter' },
  ]

  return <Page tag="LOCAL WORKSPACE" title="Your ECG workspace." subtitle="Review local recordings, follow analysis activity and choose what to work on next.">
    {analysisError && <ErrorBox text={analysisError} />}
    <div className="pf-dash-top"><UploadCard/>
      <div className="pf-kpis" aria-label="Key figures">
        <article><span className="pf-label">BEATS ANALYSED</span><b className="pf-big">{metricValue(beatsProcessed)}</b><small>Across {completed.length} completed {completed.length === 1 ? 'analysis' : 'analyses'}</small></article>
        <article><span className="pf-label">CLASS SHARE</span>
          <div className="pf-share">{CLASS_CODES.map(c => <div key={c}><i style={{ background: colors[c] }}/><b>{c}</b><span>{classSum ? (classTotals[c] / classSum * 100).toFixed(1) + '%' : '—'}</span></div>)}</div>
          <div className="pf-stack" aria-hidden="true">{CLASS_CODES.map(c => <i key={c} style={{ width: `${classSum ? classTotals[c] / classSum * 100 : 0}%`, background: colors[c] }}/>)}</div></article>
        <article><span className="pf-label">FLAGGED FOR REVIEW</span><b className="pf-big">{metricValue(flaggedTotal)}</b><small>Low-confidence beats · {metricValue(pendingReviews)} still unreviewed</small></article>
      </div>
    </div>
    <section className="workspace-welcome">
      <div className="workspace-welcome-copy"><span className="workspace-kicker">NEUROCARDIO / RESEARCH DESK</span><p>ECG beat analysis, made navigable. Inspect a recording, follow its model results and bring flagged beats into human review.</p><Link className="workspace-primary-action" to="/app/new"><Upload size={17} /> New ECG Analysis <ArrowRight size={16} /></Link><span className="workspace-local-note"><i /> Local recordings · Retrospective research</span></div>
      <div className="workspace-ecg-visual"><span className="workspace-figure-label">SIGNAL / STUDY VIEW</span><svg viewBox="0 0 640 190" role="img" aria-labelledby="workspace-ecg-title workspace-ecg-description"><title id="workspace-ecg-title">Illustrative ECG waveform</title><desc id="workspace-ecg-description">Decorative waveform graphic, not a patient recording.</desc><defs><pattern id="workspace-small-grid" width="16" height="16" patternUnits="userSpaceOnUse"><path d="M16 0H0V16" fill="none" stroke="#343448" strokeOpacity=".08" strokeWidth="1"/></pattern><pattern id="workspace-large-grid" width="80" height="80" patternUnits="userSpaceOnUse"><rect width="80" height="80" fill="url(#workspace-small-grid)"/><path d="M80 0H0V80" fill="none" stroke="#343448" strokeOpacity=".12"/></pattern></defs><rect width="640" height="190" fill="url(#workspace-large-grid)"/><path d="M0 96h100l12-1 14 2 14-1 14 1 10-2 12 1 12-1 12 2 12-1 10 1 8-2 8 2 9-1 8 1 10-2 9 2 10-1 10 1 11-2 11 2 13-1 13 1 14-2 14 2 14-1 15 1 16-2 16 2h137" fill="none" stroke="#697565" strokeWidth="2.4" strokeLinejoin="round"/><path d="M248 97h25l12-1 14 1 10-2 11 1 12-1 11 2 14-1 10-1 9 2 8-1 11 1 9-2 8 2 6-1 4-42 5 77 5-28 11-10 14 2 13-1 14 2 15-1 15 1 14-1 16 2 18-1h58" fill="none" stroke="#a85d55" strokeWidth="3" strokeLinejoin="round" strokeLinecap="round"/><circle cx="423" cy="56" r="4" fill="#a85d55"/><text x="435" y="51" fill="#655f67" fontSize="9" letterSpacing="1.5">R-PEAK</text></svg><div className="workspace-ecg-caption"><span>ILLUSTRATIVE TRACE · NOT PATIENT DATA</span><b>Signal in context.</b></div></div>
    </section>

    <section className="dashboard-metrics" aria-label="Workspace overview">
      <article><span>RECORDINGS ANALYSED</span><b>{metricValue(analyzedRecordCount)}</b><small>{analyses ? 'Unique recordings with completed analyses' : 'Loading analysis history'}</small></article>
      <article><span>BEATS PROCESSED</span><b>{metricValue(beatsProcessed)}</b><small>From completed local analyses</small></article>
      <article className="metric-review"><span>AWAITING HUMAN REVIEW</span><b>{metricValue(pendingReviews)}</b><small>{reviewCounts ? 'Flagged beats without a saved review' : reviewError ? 'Review queue unavailable' : 'Loading flagged beat queue'}</small></article>
      <article className="metric-reports"><span>REPORTS GENERATED</span><b>—</b><small>Export history is not exposed by the current API</small></article>
    </section>
    {reviewError && <div className="dashboard-data-note" role="status">Could not load the flagged-beat review queue: {reviewError}</div>}

    <section className="dashboard-section dashboard-actions-section"><div className="dashboard-section-heading"><div><div className="eyebrow">WORKSPACE SHORTCUTS</div><h2>Pick up where you need to be.</h2></div><span>01 / NAVIGATE</span></div><div className="dashboard-actions">{actions.map(({ title, detail, href, icon: Icon, color }, index) => <Link className={`dashboard-action dashboard-action-${color}`} to={href} key={title}><span className="dashboard-action-index">0{index + 1}</span><Icon size={20} strokeWidth={1.7}/><b>{title}</b><small>{detail}</small><ArrowRight className="dashboard-action-arrow" size={15}/></Link>)}</div></section>

    <div className="dashboard-lower-grid"><section className="dashboard-section dashboard-activity"><div className="dashboard-section-heading"><div><div className="eyebrow">RECENT ACTIVITY</div><h2>Latest recordings.</h2></div><Link to="/app/recordings">All recordings <ArrowRight size={14}/></Link></div>{recent.length ? <div className="activity-table-wrap"><table className="activity-table"><thead><tr><th>RECORDING</th><th>DATE ANALYSED</th><th>BEATS</th><th>STATUS</th><th></th></tr></thead><tbody>{recent.map(analysis => {const record = recordById.get(analysis.record_id);const review = reviewCounts?.[analysis.id];return <tr key={analysis.id}><td><span className="activity-record-name">{record?.filename || `Recording ${analysis.record_id.slice(0, 8)}`}</span><small>{analysis.id.slice(0, 8)}</small></td><td>{formatDate(analysis.completed_at || analysis.started_at)}</td><td>{analysis.beat_count.toLocaleString()}</td><td><span className={`activity-status status-${analysis.status}`}>{analysis.status}</span>{review?.pending ? <small className="activity-review-note">{review.pending} awaiting review</small> : review?.flagged ? <small className="activity-review-note">Review complete</small> : null}</td><td><Link aria-label={`Open results for ${record?.filename || analysis.id.slice(0, 8)}`} to={`/app/analyses/${analysis.id}`}>Open <ArrowRight size={13}/></Link></td></tr>})}</tbody></table></div> : <article className="dashboard-empty"><Activity size={21}/><b>{analyses ? 'No recordings analysed yet' : 'Loading activity'}</b><span>{analyses ? 'Validate a local ECG recording to see saved analyses here.' : 'The latest analysis history is loaded from the local API.'}</span>{analyses && <Link to="/app/new">Start an analysis <ArrowRight size={14}/></Link>}</article>}</section>

      <aside className="dashboard-model"><div className="eyebrow">MODEL AT A GLANCE</div><div className="dashboard-model-name"><BrainCircuit size={20}/><span><b>CNN–BiLSTM</b><small>Beat classification model</small></span></div><div className="dashboard-model-facts"><span>WINDOW <b>720 samples</b></span><span>INPUT <b>Single lead · 360 Hz</b></span><span>CLASSES <b>N · S · V · F</b></span></div><div className="dashboard-model-scores"><div><span>FIXED-TEST ACCURACY</span><b>90.6%</b></div><div><span>MACRO-F1</span><b>47.3%</b></div><div className="dashboard-class-recall"><span>S · V · F RECALL</span><b>9.7% · 89.2% · 5.7%</b></div></div><p className="dashboard-model-warning">Weak S and F recall means many beats in these groups may be missed.</p><Link to="/app/model">Detailed model information <ArrowRight size={14}/></Link></aside>
    </div>
  </Page>
}
function NewPage() {
  const [files, setFiles] = useState<File[]>([]), [lead, setLead] = useState('MLII'), [rate, setRate] = useState('360'), [preview, setPreview] = useState<any>(null), [err, setErr] = useState(''), [busy, setBusy] = useState(false), [busyAction, setBusyAction] = useState<'validate' | 'run' | null>(null), [dragging, setDragging] = useState(false)
  async function upload() {
    if (!files.length) return
    setBusy(true); setBusyAction('validate'); setErr('')
    try { const f = new FormData(); files.forEach(file => f.append('files', file)); f.append('lead', lead); if (files.some(x => x.name.toLowerCase().endsWith('.csv'))) f.append('sampling_rate', rate); const r = await api.post('/ecg/upload', f); setPreview(r.data) }
    catch (e: any) { setErr(e.response?.data?.detail || e.message) }
    finally { setBusy(false); setBusyAction(null) }
  }
  async function run() {
    setBusy(true); setBusyAction('run'); setErr('')
    try { const r = await api.post('/analyses', null, { params: { record_id: preview.id } }); location.href = `/app/analyses/${r.data.id}` }
    catch (e: any) { const d = e.response?.data?.detail; setErr(typeof d === 'string' ? d : d?.message || JSON.stringify(d)) }
    finally { setBusy(false); setBusyAction(null) }
  }
  function selectFiles(selected: FileList | null) { if (!selected) return; setFiles(Array.from(selected)); setPreview(null); setErr('') }
  const fileKind = (name: string) => name.toLowerCase().endsWith('.csv') ? 'CSV' : name.toLowerCase().endsWith('.hea') ? 'WFDB header' : name.toLowerCase().endsWith('.dat') ? 'WFDB signal' : 'WFDB annotations'
  const formatSize = (size: number) => size < 1024 ? `${size} B` : size < 1024 * 1024 ? `${(size / 1024).toFixed(1)} KB` : `${(size / (1024 * 1024)).toFixed(2)} MB`
  return <Page tag="NEW ECG ANALYSIS" title="Bring a recording into view." subtitle="Validate a local ECG file, inspect its real signal, then choose when to run a retrospective analysis.">
    <div className="new-analysis-layout"><article className="panel upload-panel">
      <div className="upload-section-heading"><span className="eyebrow">01 / RECORDING</span><h2>Select ECG files</h2><p>Files are validated by the research backend before analysis.</p></div>
      <div className={`drop ${dragging ? 'drop-active' : ''}`} onDragEnter={event => { event.preventDefault(); setDragging(true) }} onDragOver={event => event.preventDefault()} onDragLeave={event => { if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setDragging(false) }} onDrop={event => { event.preventDefault(); setDragging(false); selectFiles(event.dataTransfer.files) }} role="group" aria-label="ECG file drop area">
        <div className="drop-symbol"><Upload size={23} /></div><b>Drop ECG files here</b><span>or choose files from this device</span><label className="file-picker" htmlFor="ecg-file-input">Choose files</label><input id="ecg-file-input" className="visually-hidden-file" type="file" multiple accept=".csv,.hea,.dat,.atr" onChange={e => selectFiles(e.target.files)} /><small>CSV · WFDB (.hea + .dat, optional .atr)</small>
      </div>
      {files.length > 0 && <div className="selected-files" aria-live="polite"><div className="selected-files-heading"><b>Selected files</b><span>{files.length} {files.length === 1 ? 'file' : 'files'}</span></div>{files.map((file, index) => <div className="selected-file" key={`${file.name}-${index}`}><span className="file-type-mark">{fileKind(file.name).slice(0, 3).toUpperCase()}</span><span className="selected-file-name"><b>{file.name}</b><small>{fileKind(file.name)}</small></span><span className="file-size">{formatSize(file.size)}</span></div>)}</div>}
      <div className="upload-controls"><label>Lead or CSV column<input value={lead} onChange={e => { setLead(e.target.value); setPreview(null); setErr('') }} aria-label="Lead or CSV column" /></label><label>Sampling frequency<input inputMode="decimal" value={rate} onChange={e => { setRate(e.target.value); setPreview(null); setErr('') }} aria-label="Sampling frequency in hertz" /><small>Hz</small></label></div>
      <div className="format-guidance"><b>File guide</b><p><strong>CSV</strong> — upload the ECG signal file, select its signal column above and enter the recording frequency. <strong>WFDB</strong> — upload matching <code>.hea</code> and <code>.dat</code> files; <code>.atr</code> annotations are optional.</p><small>Expected sampling frequency: 360 Hz. Other rates are rejected.</small></div>
      {err && <ErrorBox text={err} />}
      {preview && <div className="validation-result" role="status"><span className="validation-check">✓</span><div><b>Recording validated</b><small>Preview below uses the signal returned by the backend. Review the waveform before starting analysis.</small></div></div>}
      <div className="upload-actions"><div><span>Step 1</span><button disabled={!files.length || busy} onClick={upload}>{busyAction === 'validate' ? 'Validating…' : 'Validate & preview'}</button></div><div><span>Step 2</span><button className="primary" disabled={!preview || busy} onClick={run}>{busyAction === 'run' ? 'Starting analysis…' : 'Run analysis'}</button></div></div>
      {preview && <Wave points={preview.preview} />}
    </article><aside className="upload-aside"><div className="upload-aside-feature"><div className="eyebrow">BEFORE YOU BEGIN</div><h2>Read the signal.<br /><em>Keep context.</em></h2><p>Review the ECG signal, then explore beat-level predictions and optional model explanations.</p></div><div className="upload-aside-note"><span className="aside-note-number">A</span><div><b>Check the source</b><p>Confirm the ECG lead, sampling rate and file pairing before validation.</p></div></div><div className="upload-aside-note"><span className="aside-note-number">B</span><div><b>Inspect before analysis</b><p>Use the actual signal preview to check that the recording looks as expected.</p></div></div></aside></div>
  </Page>
}
function Wave({points, beats = [], selectedBeat}: {points: {time_seconds: number; amplitude: number}[]; beats?: Beat[]; selectedBeat?: Beat | null}) {
  const [zoom, setZoom] = useState(1)
  if (!points?.length) return <Empty title="Waveform unavailable" text="No samples were returned from the backend."/>
  const values = points.map(point => point.amplitude)
  const low = Math.min(...values)
  const span = Math.max(...values) - low || 1
  const path = points.map((point, index) => `${index / Math.max(1, points.length - 1) * 1000},${220 - (point.amplitude - low) / span * 190}`).join(' ')
  const startTime = points[0].time_seconds
  const endTime = points.at(-1)?.time_seconds ?? startTime
  const visibleBeats = [...new Map([...beats, ...(selectedBeat ? [selectedBeat] : [])].map(beat => [beat.id, beat])).values()]
    .filter(beat => beat.timestamp_seconds >= startTime && beat.timestamp_seconds <= endTime)
  return <div className="wave"><div className="wave-controls"><span>ECG waveform · actual uploaded signal</span><span><button aria-label="Zoom out" onClick={() => setZoom(Math.max(1, zoom - 1))}>−</button> {zoom}× <button aria-label="Zoom in" onClick={() => setZoom(Math.min(8, zoom + 1))}>+</button></span></div><div className="wave-scroll"><svg viewBox="0 0 1000 240" style={{width: `${zoom * 100}%`}} onWheel={event => { if (event.deltaY < 0) setZoom(Math.min(8, zoom + 1)); else setZoom(Math.max(1, zoom - 1)) }}><defs><pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse"><path d="M20 0H0V20" fill="none" stroke="#e8ded0"/></pattern></defs><rect width="1000" height="240" fill="url(#grid)"/><polyline points={path} fill="none" stroke="#263B32" strokeWidth="2"/>{visibleBeats.map(beat => { const x = (beat.timestamp_seconds - startTime) / Math.max(endTime - startTime, 1e-9) * 1000; const selected = beat.id === selectedBeat?.id; return <line key={beat.id} x1={x} x2={x} y1="0" y2="240" stroke={colors[beat.prediction.predicted_class]} strokeDasharray={selected ? undefined : '4 4'} strokeWidth={selected ? 2.5 : 1} /> })}</svg></div><small>Scroll horizontally to pan · time in seconds · amplitude scaled to visible segment</small></div>
}
function Result() {
  const { id } = useParams()
  const [searchParams] = useSearchParams()
  const targetBeatId = searchParams.get('beat')
  const [d, setD] = useState<any>(null)
  const [error, setError] = useState('')
  const [recordingName, setRecordingName] = useState('')
  const [recordingDuration, setRecordingDuration] = useState<number | null>(null)
  const [recordLookupError, setRecordLookupError] = useState('')
  const [findingBeats, setFindingBeats] = useState<Beat[] | null>(null)
  const [findingError, setFindingError] = useState('')
  const [selectedBeat, setSelectedBeat] = useState<Beat | null>(null)
  const [reviewClass, setReviewClass] = useState('N')
  const [note, setNote] = useState('')
  const [reviewMessage, setReviewMessage] = useState('')
  const [filtered, setFiltered] = useState(false)
  const [classFilter, setClassFilter] = useState('all')
  const [flaggedOnly, setFlaggedOnly] = useState(false)
  const [hasMore, setHasMore] = useState(true)
  const [loadingMore, setLoadingMore] = useState(false)
  const [drawerOpen, setDrawerOpen] = useState(false)
  const [heat, setHeat] = useState<{ t: number; dur: number; values: number[] } | null>(null)
  const [showId, setShowId] = useState(false)
  const [focus, setFocus] = useState<{ t: number; k: number } | null>(null)
  const [rowLimit, setRowLimit] = useState(200)
  const classificationRef = useRef<HTMLElement | null>(null)

  useEffect(() => {
    setD(null)
    setError('')
    setRecordingName('')
    setRecordingDuration(null)
    setRecordLookupError('')
    setFindingBeats(null)
    setFindingError('')
    setSelectedBeat(null)
    setDrawerOpen(false)
    Promise.all([api.get(`/analyses/${id}`), api.get(`/analyses/${id}/beats?limit=500`), api.get(`/analyses/${id}/signal?count=${SIG_COUNT}`)]).then(([a, b, s]) => {
      setD({ analysis: a.data, beats: b.data, signal: s.data })
      setHasMore(b.data.length === 500)
    }).catch(e => setError(e.response?.data?.detail?.message || e.response?.data?.detail || e.message))
  }, [id])
  const recordId = d?.analysis?.record_id
  useEffect(() => {
    if (!recordId) return
    api.get('/ecg/records', { params: { limit: 500 } }).then(({ data }) => {
      const record = data.find((item: ECGRecord) => item.id === recordId)
      if (record) {
        setRecordingName(record.filename)
        setRecordingDuration(record.duration_seconds)
      }
      else setRecordLookupError('Recording name unavailable; showing its saved ID.')
    }).catch(() => setRecordLookupError('Recording name unavailable; showing its saved ID.'))
  }, [recordId])
  useEffect(() => {
    if (!d?.analysis) return
    let cancelled = false
    const analysis = d.analysis as Analysis
    if (analysis.status !== 'completed') {
      setFindingBeats([])
      return
    }
    setFindingBeats(null)
    setFindingError('')
    const loadPredictions = async () => {
      const allBeats: Beat[] = [...d.beats]
      while (allBeats.length < analysis.beat_count) {
        const response = await api.get<Beat[]>(`/analyses/${analysis.id}/beats`, {
          params: { offset: allBeats.length, limit: 1000 },
        })
        if (cancelled) return
        allBeats.push(...response.data)
        if (!response.data.length) break
      }
      if (cancelled) return
      if (allBeats.length !== analysis.beat_count) {
        setFindingError(`Received ${allBeats.length} of ${analysis.beat_count} saved beat predictions.`)
        return
      }
      setFindingBeats(allBeats)
      setD((current: any) => current ? { ...current, beats: allBeats } : current)
      setHasMore(false)
    }
    void loadPredictions().catch((e: any) => {
      if (!cancelled) setFindingError(e.response?.data?.detail || e.message || 'Saved beat predictions are unavailable.')
    })
    return () => { cancelled = true }
  }, [d?.analysis?.id])
  useEffect(() => {
    if (!targetBeatId || !d?.beats || selectedBeat?.id === targetBeatId) return
    let cancelled = false
    const findBeat = async () => {
      let allBeats: Beat[] = d.beats
      let target = allBeats.find(beat => beat.id === targetBeatId)
      let offset = allBeats.length
      while (!target && offset < d.analysis.beat_count) {
        const response = await api.get<Beat[]>(`/analyses/${id}/beats`, { params: { offset, limit: 500 } })
        if (cancelled) return
        allBeats = [...allBeats, ...response.data]
        target = response.data.find(beat => beat.id === targetBeatId)
        if (!response.data.length) break
        offset += response.data.length
      }
      if (cancelled) return
      if (allBeats.length > d.beats.length) setD((current: any) => current ? { ...current, beats: allBeats } : current)
      setHasMore(allBeats.length < d.analysis.beat_count)
      if (target) void selectBeat(target)
    }
    void findBeat().catch((e: any) => { if (!cancelled) setError(e.response?.data?.detail || e.message) })
    return () => { cancelled = true }
  }, [targetBeatId, d?.beats])
  if (error) return <Page tag="ANALYSIS" title="Analysis unavailable"><ErrorBox text={error}/></Page>
  if (!d) return <Page tag="ANALYSIS" title="Loading analysis…"><Empty title="Waiting for local API" text="Analysis data is retrieved from the backend."/></Page>
  const analysis: Analysis = d.analysis
  const visibleBeats = d.beats.filter((b: Beat) => (classFilter === 'all' || b.prediction.predicted_class === classFilter) && (!flaggedOnly || b.prediction.review_flag))
  const classCodes = ['N', 'S', 'V', 'F'] as const
  const classCounts = (findingBeats || []).reduce<Record<string, number>>((counts, beat) => {
    const code = beat.prediction?.predicted_class
    if (code && classCodes.includes(code as typeof classCodes[number])) counts[code] += 1
    return counts
  }, { N: 0, S: 0, V: 0, F: 0 })
  const predictionsValid = findingBeats !== null && findingBeats.every(beat =>
    classCodes.includes(beat.prediction?.predicted_class as typeof classCodes[number]) &&
    Number.isFinite(beat.timestamp_seconds) && beat.timestamp_seconds >= 0
  )
  const classTotal = classCodes.reduce((sum, code) => sum + classCounts[code], 0)
  const hasValidDistribution = analysis.status === 'completed' && analysis.beat_count > 0 &&
    findingBeats !== null && predictionsValid && classTotal === analysis.beat_count
  const abnormalClasses = hasValidDistribution ? classCodes.filter(code => code !== 'N' && classCounts[code] > 0) : []
  const abnormalCount = abnormalClasses.reduce((sum, code) => sum + classCounts[code], 0)
  const flaggedCount = findingBeats ? findingBeats.filter(beat => beat.prediction.review_flag).length : null
  const formatTime = (seconds: number) => {
    const whole = Math.max(0, Math.floor(seconds))
    const hours = Math.floor(whole / 3600)
    const minutes = Math.floor((whole % 3600) / 60)
    const remainder = whole % 60
    return hours ? `${hours}:${String(minutes).padStart(2, '0')}:${String(remainder).padStart(2, '0')}` : `${minutes}:${String(remainder).padStart(2, '0')}`
  }
  const formatOccurrenceTime = (seconds: number) => {
    const milliseconds = Math.max(0, Math.round(seconds * 1000))
    return `${formatTime(Math.floor(milliseconds / 1000))}.${String(milliseconds % 1000).padStart(3, '0')}`
  }
  const classTitles: Record<typeof classCodes[number], string> = {
    N: 'Normal beats',
    S: 'Supraventricular ectopic beats',
    V: 'Ventricular ectopic beats',
    F: 'Fusion beats',
  }
  const classNotes: Record<typeof classCodes[number], string> = {
    N: 'Beats assigned to the AAMI normal class.',
    S: 'Beats assigned to the AAMI supraventricular ectopic class.',
    V: 'May include PVC-like beats; this class does not independently diagnose PVCs or ventricular tachycardia.',
    F: 'Fusion-class beat morphology; not an independent arrhythmia diagnosis.',
  }
  const classBeats = (code: typeof classCodes[number]) => (findingBeats || []).filter(beat => beat.prediction.predicted_class === code)
  const classRange = (code: typeof classCodes[number]) => {
    const matches = classBeats(code)
    if (!matches.length) return 'No occurrences detected'
    const ordered = [...matches].sort((a, b) => a.timestamp_seconds - b.timestamp_seconds)
    return `First beat ${ordered[0].beat_index + 1} · ${formatOccurrenceTime(ordered[0].timestamp_seconds)} s / Last beat ${ordered.at(-1)!.beat_index + 1} · ${formatOccurrenceTime(ordered.at(-1)!.timestamp_seconds)} s`
  }
  const timelineDuration = recordingDuration ?? 0
  const timelineBins = Array.from({ length: 24 }, (_, index) => {
    const start = timelineDuration * index / 24
    const end = timelineDuration * (index + 1) / 24
    const beats = (findingBeats || []).filter(beat => beat.timestamp_seconds >= start && (index === 23 ? beat.timestamp_seconds <= end : beat.timestamp_seconds < end))
    const counts = classCodes.reduce<Record<string, number>>((result, code) => {
      result[code] = beats.filter(beat => beat.prediction.predicted_class === code).length
      return result
    }, { N: 0, S: 0, V: 0, F: 0 })
    return { index, start, end, beats, counts }
  })
  const sum = summarise(d.beats)
  const recordingTitle = recordingName || `Recording ${analysis.record_id.slice(0, 8)}`
  async function loadMore() {
    setLoadingMore(true)
    try {
      const r = await api.get(`/analyses/${id}/beats`, { params: { offset: d.beats.length, limit: 500 } })
      setD({ ...d, beats: [...d.beats, ...r.data] })
      setHasMore(r.data.length === 500)
    } catch (e: any) { setError(e.response?.data?.detail || e.message) }
    finally { setLoadingMore(false) }
  }
  async function setSignalMode(on: boolean) {
    try {
      const startSample = d.signal.start_sample ?? 0
      const s = await api.get(`/analyses/${id}/signal`, { params: { start_sample: startSample, count: SIG_COUNT, filtered: on } })
      setD((current: any) => current ? { ...current, signal: s.data } : current)
      setFiltered(on)
    } catch (e: any) { setError(e.response?.data?.detail || e.message) }
  }
  async function selectBeat(beat: Beat, open = false) {
    setSelectedBeat(beat)
    setReviewClass(beat.review?.reviewed_label || beat.prediction.predicted_class)
    setNote(beat.review?.note || '')
    setReviewMessage('')
    if (open) setDrawerOpen(true)
    try {
      const sig = d.signal
      const inside = beat.sample_index >= sig.start_sample && beat.sample_index < sig.start_sample + sig.sample_indices.length
      if (!inside) {
        const startSample = Math.max(0, beat.sample_index - SIG_COUNT / 2)
        const signal = await api.get(`/analyses/${id}/signal`, { params: { start_sample: startSample, count: SIG_COUNT, filtered } })
        setD((current: any) => current ? { ...current, signal: signal.data } : current)
      }
      setFocus({ t: beat.timestamp_seconds, k: Date.now() })
    } catch (e: any) { setError(e.response?.data?.detail || e.message) }
  }
  async function shiftWindow(dir: -1 | 1) {
    try {
      const start = Math.max(0, d.signal.start_sample + dir * Math.round(SIG_COUNT * 0.8))
      const s = await api.get(`/analyses/${id}/signal`, { params: { start_sample: start, count: SIG_COUNT, filtered } })
      if (!s.data.amplitude.length) return
      setD((current: any) => current ? { ...current, signal: s.data } : current)
      setFocus({ t: dir > 0 ? s.data.time_seconds[0] + 5 : s.data.time_seconds[s.data.time_seconds.length - 1] - 5, k: Date.now() })
    } catch (e: any) { setError(e.response?.data?.detail || e.message) }
  }
  function patchBeat(beatId: string, review: Review | null) {
    setD((current: any) => current ? { ...current, beats: current.beats.map((b: Beat) => b.id === beatId ? { ...b, review } : b) } : current)
    setSelectedBeat(current => current && current.id === beatId ? { ...current, review } : current)
  }
  async function setReview(beat: Beat, label: string | null) {
    const prev = beat.review ?? null
    patchBeat(beat.id, label ? { id: prev?.id ?? 'pending', original_prediction: beat.prediction.predicted_class, reviewed_label: label, note: prev?.note ?? null, status: 'reviewed' } : null)
    try {
      const r = await api.post('/reviews', { beat_id: beat.id, reviewed_label: label, note: prev?.note ?? undefined })
      if (label) patchBeat(beat.id, r.data as Review)
      setReviewMessage('')
    } catch (e: any) { patchBeat(beat.id, prev); setReviewMessage(e.response?.data?.detail || e.message) }
  }
  function openFromCard() {
    if (selectedBeat) { setDrawerOpen(true); return }
    const pick = d.beats.filter((b: Beat) => b.prediction.predicted_class === sum.dominantModel).sort((x: Beat, y: Beat) => y.prediction.confidence - x.prediction.confidence)[0]
    if (pick) void selectBeat(pick, true)
  }
  function showClassBeats(code: typeof classCodes[number]) {
    setClassFilter(code)
    setFlaggedOnly(false)
    classificationRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }
  function inspectTimelineBin(bin: typeof timelineBins[number]) {
    const beat = bin.beats.find(item => item.prediction.predicted_class !== 'N') || bin.beats[0]
    if (!beat) return
    setClassFilter(beat.prediction.predicted_class)
    setFlaggedOnly(false)
    void selectBeat(beat)
    classificationRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }
  async function saveReview() {
    if (!selectedBeat) return
    try {
      const response = await api.post('/reviews', { beat_id: selectedBeat.id, reviewed_label: reviewClass, note })
      const review = response.data as Review
      setSelectedBeat(current => current ? { ...current, review } : current)
      setD((current: any) => current ? { ...current, beats: current.beats.map((beat: Beat) => beat.id === selectedBeat.id ? { ...beat, review } : beat) } : current)
      setReviewMessage(`Review saved separately. Original model class: ${review.original_prediction}.`)
    } catch (e: any) { setReviewMessage(e.response?.data?.detail || e.message) }
  }
  return <Page tag="ANALYSIS RESULTS" title={recordingTitle} subtitle={undefined}>
    <div className="pf-idline"><button type="button" className="pf-info" aria-expanded={showId} onClick={() => setShowId(v => !v)}>ⓘ details</button>{showId && <code>{analysis.status} · {analysis.model_version} · {analysis.preprocessing_version}</code>}</div>
    {analysis.status === 'completed' && <>
    <div className="pf-analysis-top">
      <article id="beat-waveform" className="panel pf-plot-panel"><span className="pf-corner">WAVEFORM</span><div className="panel-head"><h2>ECG waveform</h2><div className="actions"><button aria-pressed={!filtered} onClick={() => setSignalMode(false)}>Raw</button><button aria-pressed={filtered} onClick={() => setSignalMode(true)}>Filtered</button><span>{filtered ? 'Filtered' : 'Raw'} · 360 Hz</span></div></div>
        <EcgPlot times={d.signal.time_seconds} amps={d.signal.amplitude} beats={d.beats} selectedId={selectedBeat?.id} focus={focus} heat={drawerOpen ? heat : null} onSelect={b => void selectBeat(b, true)} onShift={dir => void shiftWindow(dir)}/></article>
      <aside className="pf-result" onClick={openFromCard} aria-label="Result summary">
        {sum.n ? (() => {
          const share = (c: ClassCode) => sum.model[c] / sum.n
          const head: ClassCode | null = share('V') >= 0.01 ? 'V' : (['S', 'F'] as ClassCode[]).filter(c => share(c) >= 0.05).sort((x, y) => sum.model[y] - sum.model[x])[0] ?? null
          const hc: ClassCode = head ?? 'N'
          const title = head === 'V' ? 'Ventricular ectopic beats present' : head === 'S' ? 'Supraventricular ectopic beats present' : head === 'F' ? 'Fusion beats present' : 'Predominantly normal'
          const conf = sum.model[hc] ? sum.confSum[hc] / sum.model[hc] : 0
          const dv = sum.edited > 0 ? `${sum.edited} ${sum.edited === 1 ? 'beat' : 'beats'} edited · ${hc} ${(share(hc) * 100).toFixed(1)}% → ${(sum.reviewed[hc] / sum.n * 100).toFixed(1)}%` : null
          return <>
            <div className="pf-verdict">
              <span className="pf-mlabel">RESULT</span>
              <h2>{title}</h2>
              <div className="pf-big num">{(conf * 100).toFixed(1)}<small>%</small></div>
              <div className="pf-mlabel">avg. model confidence for {hc} beats</div>
              <div className="pf-sub num">{sum.model[hc].toLocaleString()} of {sum.n.toLocaleString()} beats · {(share(hc) * 100).toFixed(1)}%</div>
              <div className="pf-stackbar" role="img" aria-label="Beat class composition">{CLASS_CODES.map(c => sum.model[c] > 0 && <i key={c} style={{ width: `${share(c) * 100}%`, background: colors[c] }}/>)}</div>
              <div className="pf-legend num">{CLASS_CODES.map(c => <span key={c} title={CLASS_NAMES[c]}><i style={{ background: colors[c] }}/>{c} {sum.model[c].toLocaleString()}</span>)}</div>
              <div className="pf-delta num">{dv ?? <em>No edits yet</em>}</div>
            </div>
            {sum.flagged > 0 && <div className="pf-flag">⚑ Review recommended · {sum.flagged.toLocaleString()} {sum.flagged === 1 ? 'beat' : 'beats'} flagged low-confidence</div>}
            <button type="button" className="pf-cta-btn">EXPLAIN →</button>
          </>
        })() : <p>Loading beat predictions…</p>}
      </aside>
    </div>
    <article ref={classificationRef} id="beat-classifications" className="panel table-scroll pf-beats"><span className="pf-corner">BEATS</span><div className="panel-head"><h2>Beat list</h2><span>{findingBeats ? `Showing ${Math.min(rowLimit, visibleBeats.length).toLocaleString()} of ${visibleBeats.length.toLocaleString()} beats` : `Loading saved predictions · ${analysis.beat_count.toLocaleString()} total`}</span></div><div className="table-filters"><label>Class<select value={classFilter} onChange={e => setClassFilter(e.target.value)}><option value="all">All classes</option>{classCodes.map(c => <option key={c} value={c}>{c} · {classTitles[c]}</option>)}</select></label><label className="flag-filter"><input type="checkbox" checked={flaggedOnly} onChange={e => setFlaggedOnly(e.target.checked)}/> Flagged only</label></div>
      {reviewMessage && <p className="review-message" role="status">{reviewMessage}</p>}
      <table><thead><tr><th>#</th><th>Time</th><th>Model</th><th>Conf.</th><th>N</th><th>S</th><th>V</th><th>F</th><th>Reviewed</th><th></th></tr></thead><tbody>{visibleBeats.slice(0, rowLimit).map((b: Beat) => { const mc = b.prediction.predicted_class; const edited = !!b.review?.reviewed_label; return <tr key={b.id} className={selectedBeat?.id === b.id ? 'selected-row' : ''} onClick={() => void selectBeat(b, true)}><td>{b.beat_index + 1}</td><td>{b.timestamp_seconds.toFixed(2)}s{b.prediction.review_flag ? ' ⚑' : ''}</td><td><span className="pf-chip" style={{ color: colors[mc], borderColor: colors[mc] }} title={CLASS_NAMES[mc as keyof typeof CLASS_NAMES]}>{mc}</span></td><td>{(b.prediction.confidence * 100).toFixed(1)}%</td>{b.prediction.probabilities.map((p: number, i: number) => <td key={i}>{p.toFixed(3)}</td>)}<td className="pf-review-cell"><select aria-label={`Reviewed class for beat ${b.beat_index + 1}`} value={effectiveClass(b)} onClick={e => e.stopPropagation()} onChange={e => void setReview(b, e.target.value)}>{CLASS_CODES.map(c => <option key={c} value={c}>{c} · {CLASS_NAMES[c]}</option>)}</select>{edited && <span className="pf-reviewed">REVIEWED</span>}</td><td className="pf-actions-cell"><button type="button" disabled={!b.review?.reviewed_label} onClick={e => { e.stopPropagation(); void setReview(b, null) }}>Reset to model</button><button type="button" onClick={e => { e.stopPropagation(); void selectBeat(b, true) }}>Explain</button></td></tr> })}</tbody></table>{!visibleBeats.length && <Empty title="No matching beats" text="Change the class or review-flag filters."/>}{visibleBeats.length > rowLimit && <button type="button" onClick={() => setRowLimit(n => n + 300)}>Show more rows ({(visibleBeats.length - rowLimit).toLocaleString()} left)</button>}{hasMore && <button disabled={loadingMore} onClick={loadMore}>{loadingMore ? 'Loading…' : 'Load more beats'}</button>}</article>
    </>}
    <section className={`analysis-summary ${hasValidDistribution ? 'analysis-summary-ready' : 'analysis-summary-insufficient'}`}>
      <div className="analysis-summary-heading"><div><span className="eyebrow">RECORDING OVERVIEW</span><h2>Analysis summary</h2></div><span className={`analysis-status status-${analysis.status}`}>{analysis.status}</span></div>
      {hasValidDistribution ? <>
        <div className="analysis-summary-result">
          <strong>{(() => { const n = analysis.beat_count || 1; return classCounts.V / n >= 0.01 ? 'Ventricular ectopic beats present' : classCounts.S / n >= 0.05 ? 'Supraventricular ectopic beats present' : classCounts.F / n >= 0.05 ? 'Fusion beats present' : 'Predominantly normal' })()}</strong>
          <p>{classCounts.V > 0
            ? `The model classified ${classCounts.V.toLocaleString()} beats as ventricular ectopic (${(classCounts.V / analysis.beat_count * 100).toFixed(1)}%). This beat class does not independently establish PVCs, ventricular tachycardia, or another rhythm diagnosis.`
            : classCounts.S > 0
              ? `The model classified ${classCounts.S.toLocaleString()} beats as supraventricular ectopic (${(classCounts.S / analysis.beat_count * 100).toFixed(1)}%). This beat class does not independently establish a rhythm diagnosis.`
              : classCounts.F > 0
                ? `The model classified ${classCounts.F.toLocaleString()} beats as fusion beats (${(classCounts.F / analysis.beat_count * 100).toFixed(1)}%). Fusion class is not an independent arrhythmia diagnosis.`
                : 'This does not rule out arrhythmia; the current model has weak recall for S and F classes.'}</p>
        </div>
        <div className="analysis-summary-facts">
          <div><span>RECORDING</span><b title={recordingTitle}>{recordingTitle}</b>{recordLookupError && <small>{recordLookupError}</small>}</div>
          <div><span>BEATS ANALYSED</span><b>{analysis.beat_count.toLocaleString()}</b></div>
          <div><span>FLAGGED FOR REVIEW</span><b>{flaggedCount === null ? '—' : flaggedCount.toLocaleString()}</b></div>
          <div><span>ABNORMAL-CLASS BEATS</span><b>{abnormalCount.toLocaleString()} · {(abnormalCount / analysis.beat_count * 100).toFixed(1)}%</b></div>
        </div>
        <p className="analysis-summary-notice">Experimental beat classifications, not a clinical diagnosis. Weak S and F recall means a class not detected by this model may still be present.</p>
      </> : <div className="analysis-summary-result">
        <strong>Review required / insufficient analysis.</strong>
        <p>{analysis.status === 'failed' ? analysis.error_message || 'The recording could not be analysed.' : analysis.status !== 'completed' ? 'Analysis has not completed.' : analysis.beat_count === 0 ? 'No valid beats were available for classification.' : findingError || (findingBeats === null ? 'Loading saved beat predictions for recording-level findings.' : 'The saved beat predictions do not match the analysed beat total, so a recording summary is unavailable.')}</p>
        <small>{analysis.beat_count.toLocaleString()} valid beats recorded · {flaggedCount === null ? 'review count unavailable' : `${flaggedCount.toLocaleString()} flagged for human review`}</small>
      </div>}
    </section>
    <section className="analysis-distribution-section" aria-label="Beat class distribution">
      <div className="analysis-section-heading"><div><span className="eyebrow">CLASS DISTRIBUTION</span><h2>Beats by model class.</h2></div><span>Counts from this analysis</span></div>
      {hasValidDistribution ? <div className="analysis-class-grid">{classCodes.map(code => {
        const count = classCounts?.[code] ?? 0
        const percent = analysis.beat_count ? (count / analysis.beat_count) * 100 : 0
        return <article className={`analysis-class analysis-class-${code.toLowerCase()}`} key={code}><div><span>CLASS {code}</span><b>{count.toLocaleString()}</b></div><strong>{classTitles[code]}</strong><small>{percent.toFixed(1)}% of analysed beats</small><div className="class-bar"><i style={{ width: `${percent}%` }}/></div><p>{classNotes[code]}</p></article>
      })}</div> : <div className="analysis-distribution-unavailable">Class distribution is unavailable until a complete, internally consistent analysis is returned.</div>}
    </section>
    <section className="detected-patterns" aria-labelledby="detected-patterns-title">
      <div className="analysis-section-heading"><div><span className="eyebrow">RECORDING FINDINGS</span><h2 id="detected-patterns-title">Detected patterns.</h2></div><span>Beat classes only · no rhythm diagnosis</span></div>
      {hasValidDistribution ? <>
        <div className="pattern-list">
          {([
            { code: 'V', title: 'Ventricular ectopic activity detected' },
            { code: 'S', title: 'Supraventricular ectopic activity detected' },
            { code: 'F', title: 'Fusion beats detected' },
          ] as const).map(({ code, title }) => {
            const count = classCounts[code]
            const percent = count / analysis.beat_count * 100
            return <article className={`pattern-item pattern-${code.toLowerCase()}`} key={code}>
              <div className="pattern-symbol">{code}</div>
              <div className="pattern-copy">
                <h3>{count ? title : `${classTitles[code]} not detected`}</h3>
                <p>{count ? classNotes[code] : 'No beats were assigned to this class in the saved model predictions.'}</p>
                <small>{classRange(code)}</small>
              </div>
              <div className="pattern-count"><b>{count.toLocaleString()}</b><span>{percent.toFixed(1)}% of analysed beats</span></div>
              <button type="button" disabled={!count} onClick={() => showClassBeats(code)}>{count ? 'View matching beats' : 'No matching beats'}</button>
            </article>
          })}
        </div>
        {!abnormalCount && <p className="no-abnormal-note">No abnormal-class beats detected by the model. This does not rule out arrhythmia.</p>}
      </> : <div className="analysis-distribution-unavailable">{findingError || 'Detected patterns will appear after all saved beat predictions are loaded and checked.'}</div>}
      <section className="rhythm-unavailable" aria-label="Rhythm-level interpretation status">
        <span className="eyebrow">RHYTHM-LEVEL INTERPRETATION</span>
        <strong>Rhythm-level classification unavailable</strong>
        <p>The current model classifies individual beats as N, S, V, or F. It does not identify atrial fibrillation, atrial flutter, ventricular tachycardia, or supraventricular tachycardia.</p>
      </section>
    </section>
    <section className="temporal-findings" aria-labelledby="temporal-findings-title">
      <div className="analysis-section-heading"><div><span className="eyebrow">ACTUAL BEAT TIMESTAMPS</span><h2 id="temporal-findings-title">Temporal distribution.</h2></div><span>{timelineDuration ? `${formatTime(timelineDuration)} recording` : 'Recording duration unavailable'}</span></div>
      {hasValidDistribution && timelineDuration > 0 ? <>
        <div className="temporal-legend">{classCodes.map(code => <span key={code}><i style={{ background: colors[code] }}/>{code} · {classTitles[code]}</span>)}</div>
        <div className="temporal-timeline" role="group" aria-label="Clickable timeline of model beat classifications">
          {timelineBins.map(bin => {
            const count = bin.beats.length
            const abnormalInBin = count - bin.counts.N
            return <button type="button" className={`timeline-bin${abnormalInBin ? ' timeline-bin-abnormal' : ''}`} key={bin.index} disabled={!count} aria-label={`${formatTime(bin.start)} to ${formatTime(bin.end)}: ${bin.counts.N} N, ${bin.counts.S} S, ${bin.counts.V} V, ${bin.counts.F} F; ${count} beats`} title={`${formatTime(bin.start)}–${formatTime(bin.end)} · N ${bin.counts.N} · S ${bin.counts.S} · V ${bin.counts.V} · F ${bin.counts.F}`} onClick={() => inspectTimelineBin(bin)}>
              <span className="timeline-stack">{classCodes.map(code => bin.counts[code] ? <i key={code} className={`timeline-segment timeline-segment-${code.toLowerCase()}`} style={{ width: `${bin.counts[code] / count * 100}%` }}/> : null)}</span>
              <small>{formatTime(bin.start)}</small>
              {abnormalInBin > 0 && <b>{abnormalInBin}</b>}
            </button>
          })}
        </div>
        <p className="timeline-hint">Select a recording segment to open its first abnormal-class beat (or the first beat when none occur) in the detailed waveform and matching class results. Segment counts use actual saved beat timestamps.</p>
      </> : <div className="analysis-distribution-unavailable">{findingBeats === null ? 'Loading timestamped predictions…' : 'Temporal distribution is unavailable without a complete prediction list and recording duration.'}</div>}
    </section>
    {error && <ErrorBox text={error}/>}
    {analysis.status === 'completed' && <article className="panel selected-beat-panel"><div className="analysis-section-heading"><div><span className="eyebrow">DETAIL 01</span><h2>Selected beat.</h2></div>{selectedBeat && <span>Beat {selectedBeat.beat_index + 1} · {selectedBeat.timestamp_seconds.toFixed(3)} s</span>}</div>{selectedBeat ? <><p>Model prediction: <b style={{ color: colors[selectedBeat.prediction.predicted_class] }}>{selectedBeat.prediction.predicted_class}</b> · confidence {(selectedBeat.prediction.confidence * 100).toFixed(1)}%{selectedBeat.prediction.review_flag ? ' · flagged for review' : ''}</p><div className="selected-probabilities">{classCodes.map((code, index) => <div key={code}><b style={{ color: colors[code] }}>{code}</b><span>{(selectedBeat.prediction.probabilities[index] * 100).toFixed(1)}%</span></div>)}</div></> : <p>Select a row above to inspect its class probabilities and waveform segment.</p>}</article>}
    {analysis.status === 'completed' && <section className="panel analysis-next-steps"><div className="analysis-section-heading"><div><span className="eyebrow">OPTIONAL EXPLORATION</span><h2>Explainability.</h2></div></div>{selectedBeat ? <><p>Explore how the selected beat contributed to its model output. Attributions are experimental, not clinical evidence.</p><div className="analysis-link-row"><Link to={`/app/explainability?analysis=${id}&beat=${selectedBeat.id}&method=gradcam`}>Grad-CAM <ArrowRight size={14}/></Link><Link to={`/app/explainability?analysis=${id}&beat=${selectedBeat.id}&method=ig`}>Integrated Gradients <ArrowRight size={14}/></Link></div></> : <p>Select a beat above to enable Grad-CAM and Integrated Gradients exploration.</p>}</section>}
    {analysis.status === 'completed' && <section className="panel human-review-panel"><div className="analysis-section-heading"><div><span className="eyebrow">SEPARATE HUMAN LABEL</span><h2>Human review.</h2></div>{selectedBeat?.review && <span className="review-state">Saved · {selectedBeat.review.status}</span>}</div>{selectedBeat ? <><p>The model prediction and probabilities remain unchanged when you save a review.</p><div className="human-review-controls"><label>Reviewed label<select value={reviewClass} onChange={e => setReviewClass(e.target.value)}>{classCodes.map(code => <option key={code} value={code}>{code}</option>)}</select></label><label>Note<input value={note} onChange={e => setNote(e.target.value)} placeholder="Optional context"/></label><button onClick={saveReview}>Save separate review</button></div>{selectedBeat.review && <p className="saved-review-note">Saved label: <b>{selectedBeat.review.reviewed_label}</b>{selectedBeat.review.note ? ` · ${selectedBeat.review.note}` : ''}</p>}{reviewMessage && <p className="review-message" role="status">{reviewMessage}</p>}</> : <p>Select a flagged beat above to add or inspect a separate human label.</p>}</section>}
    <Disclaimer/>
    {drawerOpen && selectedBeat && <ExplainDrawer analysisId={analysis.id} beat={selectedBeat} beats={d.beats} onClose={() => setDrawerOpen(false)} onJump={b => void selectBeat(b, true)} onHeat={setHeat}/>}
  </Page>
}
function Explain() {
  const [q] = useSearchParams()
  const analysis = q.get('analysis')
  const beat = q.get('beat')
  const [selectedBeat, setSelectedBeat] = useState<any>(null)
  const [klass, setClass] = useState<string>('')
  const [method, setMethod] = useState<'gradcam' | 'ig'>(q.get('method') === 'ig' ? 'ig' : 'gradcam')
  const [responseData, setData] = useState<any>(null)
  const data = responseData?.method === method && responseData?.class === klass ? responseData : null
  const [signal, setSignal] = useState<number[]>([])
  const [err, setErr] = useState('')

  useEffect(() => {
    if (!analysis || !beat) return
    api.get(`/analyses/${analysis}/beats?limit=10000`).then(async (b) => {
      const selected = b.data.find((r: any) => r.id === beat)
      if (!selected) throw new Error('Selected beat is unavailable')
      setSelectedBeat(selected)
      setClass(prev => prev || selected.prediction.predicted_class)
      const start = Math.max(0, selected.sample_index - 360)
      const sig = await api.get(`/analyses/${analysis}/signal`, { params: { start_sample: start, count: 720, filtered: true } })
      setSignal(sig.data.amplitude)
    }).catch(e => setErr(e.response?.data?.detail || e.message))
  }, [analysis, beat])

  useEffect(() => {
    if (!analysis || !beat || !klass) return
    let cancelled = false
    setData(null)
    setErr('')
    api.get(`/analyses/${analysis}/explanations`, { params: { beat, class: klass, method } })
      .then(x => {
        if (cancelled) return
        if (x.data.method !== method || x.data.class !== klass) {
          setErr('The API returned an explanation for a different method or target class; it was not displayed.')
          return
        }
        setData(x.data)
      })
      .catch(e => { if (!cancelled) setErr(e.response?.data?.detail || e.message) })
    return () => { cancelled = true }
  }, [analysis, beat, klass, method])

  if (!analysis || !beat) {
    return (
      <Page tag="EXPLAINABILITY" title="Explore model attribution." subtitle="Optional deeper exploration of a selected beat.">
        <Empty title="Select a beat from a result" text="Open an analysis and choose Explore attribution." />
      </Page>
    )
  }

  const isGradcam = method === 'gradcam'
  const isIg = method === 'ig'
  const attrs = data?.attributions || []
  const isAllZeros = isGradcam && attrs.length > 0 && attrs.every((v: number) => Math.abs(v) < 1e-6)

  const minSig = signal.length ? Math.min(...signal) : 0
  const maxSig = signal.length ? Math.max(...signal) : 1
  const spanSig = maxSig - minSig || 1
  const maxAbsAttr = attrs.length ? Math.max(...attrs.map((v: number) => Math.abs(v)), 0.001) : 1

  const W_LEFT = 65
  const W_WIDTH = 870
  const W_TOP = 30
  const W_HEIGHT = 245
  const W_BOTTOM = W_TOP + W_HEIGHT
  const R_PEAK_X = W_LEFT + (360 / 719) * W_WIDTH

  const sigPoints = signal.map((v, i) => {
    const x = W_LEFT + (i / 719) * W_WIDTH
    const y = W_BOTTOM - ((v - minSig) / spanSig) * W_HEIGHT
    return `${x.toFixed(1)},${y.toFixed(1)}`
  }).join(' ')

  const ZERO_Y = W_TOP + W_HEIGHT / 2
  const igZeroPoints = attrs.map((v: number, i: number) => {
    const x = W_LEFT + (i / 719) * W_WIDTH
    const y = ZERO_Y - (v / maxAbsAttr) * (W_HEIGHT / 2 - 5)
    return `${x.toFixed(1)},${y.toFixed(1)}`
  }).join(' ')

  const gradcamPoints = attrs.map((v: number, i: number) => {
    const x = W_LEFT + (i / 719) * W_WIDTH
    const y = W_BOTTOM - v * (W_HEIGHT - 10)
    return `${x.toFixed(1)},${y.toFixed(1)}`
  }).join(' ')

  const gradcamAreaPath = attrs.length ? `M ${W_LEFT} ${W_BOTTOM} ` + attrs.map((v: number, i: number) => {
    const x = W_LEFT + (i / 719) * W_WIDTH
    const y = W_BOTTOM - v * (W_HEIGHT - 10)
    return `L ${x.toFixed(1)} ${y.toFixed(1)}`
  }).join(' ') + ` L ${W_LEFT + W_WIDTH} ${W_BOTTOM} Z` : ''

  const igPosArea = attrs.length ? `M ${W_LEFT} ${ZERO_Y} ` + attrs.map((v: number, i: number) => {
    const x = W_LEFT + (i / 719) * W_WIDTH
    const y = v > 0 ? ZERO_Y - (v / maxAbsAttr) * (W_HEIGHT / 2 - 5) : ZERO_Y
    return `L ${x.toFixed(1)} ${y.toFixed(1)}`
  }).join(' ') + ` L ${W_LEFT + W_WIDTH} ${ZERO_Y} Z` : ''

  const igNegArea = attrs.length ? `M ${W_LEFT} ${ZERO_Y} ` + attrs.map((v: number, i: number) => {
    const x = W_LEFT + (i / 719) * W_WIDTH
    const y = v < 0 ? ZERO_Y - (v / maxAbsAttr) * (W_HEIGHT / 2 - 5) : ZERO_Y
    return `L ${x.toFixed(1)} ${y.toFixed(1)}`
  }).join(' ') + ` L ${W_LEFT + W_WIDTH} ${ZERO_Y} Z` : ''
  const igAbsoluteError = typeof data?.metadata?.completeness_error === 'number'
    ? Math.abs(data.metadata.completeness_error) : null
  const igScoreMagnitude = typeof data?.metadata?.score_difference === 'number'
    ? Math.abs(data.metadata.score_difference) : null
  const igErrorPercent = igAbsoluteError === null || igScoreMagnitude === null
    ? null : igScoreMagnitude === 0 ? (igAbsoluteError === 0 ? 0 : Infinity) : 100 * igAbsoluteError / igScoreMagnitude
  const igErrorOverFivePercent = igErrorPercent !== null && igErrorPercent > 5

  return (
    <Page tag="EXPLAINABILITY" title="Explore model attribution." subtitle="Optional deeper exploration of a selected beat.">
      <div className="xai-info" role="note">
        <BrainCircuit size={16} />
        <span>{isGradcam ? 'Grad-CAM highlights signal regions associated with a model prediction.' : 'Integrated Gradients attributes the model output to individual input samples.'} Experimental explanation only; not clinical evidence.</span>
      </div>

      <article className="panel xai-workbench">
        <div className="panel-head">
          <div className="eyebrow">
            {data ? (data.cached ? 'CACHED' : 'COMPUTED') : 'LOADING'} · {isGradcam ? 'GRAD-CAM' : 'INTEGRATED GRADIENTS'} · TARGET CLASS {klass}
          </div>
          <Link to={`/app/analyses/${analysis}`}>← Back to analysis</Link>
        </div>

        {selectedBeat && (
          <div className="xai-pred-bar" aria-label="Selected beat prediction summary">
            <span><b>Selected beat {selectedBeat.beat_index + 1}:</b> Model predicted <b style={{ color: colors[selectedBeat.prediction.predicted_class], fontSize: '14px' }}>Class {selectedBeat.prediction.predicted_class}</b> ({(selectedBeat.prediction.confidence * 100).toFixed(1)}% conf{selectedBeat.prediction.review_flag ? ' · flagged' : ''})</span>
            <div style={{ display: 'flex', gap: '8px', marginLeft: 'auto', flexWrap: 'wrap' }}>
              {['N', 'S', 'V', 'F'].map((c, i) => (
                <span key={c} className="pred-pill" style={{ background: c === klass ? `${colors[c]}22` : '#eee', border: c === klass ? `1.5px solid ${colors[c]}` : '1px solid #ddd', color: colors[c] }}>
                  <b>{c}</b>: {(selectedBeat.prediction.probabilities[i] * 100).toFixed(1)}%
                </span>
              ))}
            </div>
          </div>
        )}

        <div className="xai-controls">
          <label>
            Target class:
            <select value={klass} onChange={e => setClass(e.target.value)}>
              {['N', 'S', 'V', 'F'].map(c => (
                <option key={c} value={c}>
                  {c} {selectedBeat?.prediction.predicted_class === c ? '(Predicted class)' : ''}
                </option>
              ))}
            </select>
          </label>
          <label>
            Method:
            <select value={method} onChange={e => setMethod(e.target.value as any)}>
              <option value="gradcam">Grad-CAM (ReLU conv3 feature map)</option>
              <option value="ig">Integrated Gradients (Signed input attribution)</option>
            </select>
          </label>
        </div>

        {err && <ErrorBox text={err} />}

        {data && signal.length > 0 && (
          <>
            <div className="xai-chart-box">
              <div className="panel-head">
                <h3>Beat waveform with {isGradcam ? 'Grad-CAM' : 'Integrated Gradients'} attribution overlay</h3>
                <span className="xai-chart-metadata">720 samples · 360 Hz · 2000 ms</span>
              </div>

              <div className="xai-legend">
                <span className="xai-legend-item"><span className="xai-legend-color" style={{ background: '#263B32' }} /> ECG Waveform</span>
                <span className="xai-legend-item"><span className="xai-legend-color" style={{ background: '#C6A96B', height: '2px' }} /> Window Center (R-peak, 1000 ms)</span>
                {isGradcam ? (
                  <span className="xai-legend-item"><span className="xai-legend-color" style={{ background: 'rgba(152, 129, 182, 0.5)' }} /> Grad-CAM Relevance [0..1]</span>
                ) : (
                  <>
                    <span className="xai-legend-item"><span className="xai-legend-color" style={{ background: 'rgba(46, 125, 50, 0.6)' }} /> Positive Evidence (+ attribution)</span>
                    <span className="xai-legend-item"><span className="xai-legend-color" style={{ background: 'rgba(198, 107, 82, 0.6)' }} /> Negative Evidence (- attribution)</span>
                    <span className="xai-legend-item"><span className="xai-legend-color" style={{ background: '#888', height: '2px' }} /> Zero Baseline</span>
                  </>
                )}
              </div>

              {isAllZeros ? (
                <div className="warning" style={{ margin: '14px 0', background: '#fcf6e8', borderColor: '#d5a34c', color: '#684e1b' }}>
                  No positive evidence for this class in this window (ReLU Grad-CAM)
                </div>
              ) : null}

              <svg viewBox="0 0 1000 320" style={{ background: '#faf9f6', border: '1px solid #e8e3d8', borderRadius: '6px' }}>
                <defs>
                  <pattern id="grid-xai" width="20" height="20" patternUnits="userSpaceOnUse">
                    <path d="M 20 0 L 0 0 0 20" fill="none" stroke="#ede7dc" strokeWidth="0.8" />
                  </pattern>
                </defs>
                <defs>
                  <clipPath id="xai-overlay-plot"><rect x={W_LEFT} y={W_TOP} width={W_WIDTH} height={W_HEIGHT} /></clipPath>
                </defs>
                <rect x={W_LEFT} y={W_TOP} width={W_WIDTH} height={W_HEIGHT} fill="url(#grid-xai)" />

                {isGradcam && !isAllZeros && (
                  <>
                    <path d={gradcamAreaPath} clipPath="url(#xai-overlay-plot)" fill="rgba(152, 129, 182, 0.28)" />
                    <polyline points={gradcamPoints} clipPath="url(#xai-overlay-plot)" fill="none" stroke="#9881B6" strokeWidth="1.8" strokeDasharray="3 3" />
                  </>
                )}

                {isIg && (
                  <>
                    <line x1={W_LEFT} y1={ZERO_Y} x2={W_LEFT + W_WIDTH} y2={ZERO_Y} clipPath="url(#xai-overlay-plot)" stroke="#888" strokeWidth="1.5" strokeDasharray="4 3" />
                    <path d={igPosArea} clipPath="url(#xai-overlay-plot)" fill="rgba(46, 125, 50, 0.25)" />
                    <path d={igNegArea} clipPath="url(#xai-overlay-plot)" fill="rgba(198, 107, 82, 0.25)" />
                    <polyline points={igZeroPoints} clipPath="url(#xai-overlay-plot)" fill="none" stroke="#5E4080" strokeWidth="1.5" />
                  </>
                )}

                <line x1={R_PEAK_X} y1={W_TOP} x2={R_PEAK_X} y2={W_BOTTOM} stroke="#C6A96B" strokeWidth="2" strokeDasharray="5 3" />
                <text x={R_PEAK_X} y={W_TOP - 8} textAnchor="middle" fill="#A88742" fontSize="11" fontWeight="bold">R-peak (1000 ms)</text>

                <polyline points={sigPoints} fill="none" stroke="#263B32" strokeWidth="2.4" />

                <line x1={W_LEFT} y1={W_BOTTOM} x2={W_LEFT + W_WIDTH} y2={W_BOTTOM} stroke="#a0a89f" strokeWidth="1.2" />
                <line x1={W_LEFT} y1={W_TOP} x2={W_LEFT} y2={W_BOTTOM} stroke="#a0a89f" strokeWidth="1.2" />
                <line x1={W_LEFT + W_WIDTH} y1={W_TOP} x2={W_LEFT + W_WIDTH} y2={W_BOTTOM} stroke="#a0a89f" strokeWidth="1.2" />

                {[0, 500, 1000, 1500, 2000].map(ms => {
                  const x = W_LEFT + (ms / 2000) * W_WIDTH
                  return (
                    <g key={ms}>
                      <line x1={x} y1={W_BOTTOM} x2={x} y2={W_BOTTOM + 5} stroke="#888" />
                      <text x={x} y={W_BOTTOM + 18} textAnchor="middle" fontSize="10" fill="#666">{ms} ms</text>
                    </g>
                  )
                })}
                <text x={W_LEFT + W_WIDTH / 2} y={W_BOTTOM + 36} textAnchor="middle" fontSize="11" fill="#333" fontWeight="bold">Time (milliseconds) · 720 samples at 360 Hz</text>

                <text x={18} y={W_TOP + W_HEIGHT / 2} textAnchor="middle" transform={`rotate(-90 18 ${W_TOP + W_HEIGHT / 2})`} fontSize="11" fill="#263B32" fontWeight="bold">ECG Amplitude</text>
                <text x={W_LEFT - 6} y={W_TOP + 10} textAnchor="end" fontSize="9" fill="#263B32">{maxSig.toFixed(2)}</text>
                <text x={W_LEFT - 6} y={W_BOTTOM} textAnchor="end" fontSize="9" fill="#263B32">{minSig.toFixed(2)}</text>

                <text x={982} y={W_TOP + W_HEIGHT / 2} textAnchor="middle" transform={`rotate(90 982 ${W_TOP + W_HEIGHT / 2})`} fontSize="11" fill="#5E4080" fontWeight="bold">
                  Attribution ({isGradcam ? 'Relevance 0..1' : 'Signed'})
                </text>
                {isGradcam ? (
                  <>
                    <text x={W_LEFT + W_WIDTH + 6} y={W_TOP + 10} textAnchor="start" fontSize="9" fill="#5E4080">1.00</text>
                    <text x={W_LEFT + W_WIDTH + 6} y={W_BOTTOM} textAnchor="start" fontSize="9" fill="#5E4080">0.00</text>
                  </>
                ) : (
                  <>
                    <text x={W_LEFT + W_WIDTH + 6} y={W_TOP + 10} textAnchor="start" fontSize="9" fill="#2E7D32">+{maxAbsAttr.toFixed(2)}</text>
                    <text x={W_LEFT + W_WIDTH + 6} y={ZERO_Y + 3} textAnchor="start" fontSize="9" fill="#888">0.00</text>
                    <text x={W_LEFT + W_WIDTH + 6} y={W_BOTTOM} textAnchor="start" fontSize="9" fill="#C66B52">-{maxAbsAttr.toFixed(2)}</text>
                  </>
                )}
              </svg>
            </div>

            <div className="xai-chart-box">
              <h3>{isGradcam ? 'Grad-CAM' : 'Integrated Gradients'} attribution curve & diagnostics</h3>

              {isIg && data.metadata && (
                <div className="xai-meta-grid">
                  <div className="xai-meta-card">
                    <small>Baseline used</small>
                    <b>{data.metadata.baseline || 'all-zero normalized window'}</b>
                  </div>
                  <div className="xai-meta-card">
                    <small>Absolute completeness error</small>
                    <b>{igAbsoluteError === null ? '—' : igAbsoluteError.toFixed(6)}</b>
                    <small>As % of |score difference|</small>
                    <b>{igErrorPercent === null ? '—' : Number.isFinite(igErrorPercent) ? `${igErrorPercent.toFixed(2)}%` : '∞'}</b>
                  </div>
                  <div className="xai-meta-card">
                    <small>Integration steps</small>
                    <b>{data.metadata.steps || 512} (trapezoidal)</b>
                  </div>
                  <div className="xai-meta-card">
                    <small>Score difference (Δ logit)</small>
                    <b>{typeof data.metadata.score_difference === 'number' ? data.metadata.score_difference.toFixed(4) : '—'}</b>
                  </div>
                </div>
              )}

              {isIg && igErrorOverFivePercent && (
                <div className="warning" role="alert">
                  Integrated Gradients completeness error exceeds 5% of |score difference| ({Number.isFinite(igErrorPercent) ? `${igErrorPercent!.toFixed(2)}%` : '∞'}). Treat this attribution with caution.
                </div>
              )}

              {isGradcam && data.metadata && (
                <div className="xai-meta-grid">
                  <div className="xai-meta-card">
                    <small>Target layer</small>
                    <b>{data.metadata.target_layer || data.metadata.layer || 'Unavailable'}</b>
                  </div>
                  <div className="xai-meta-card">
                    <small>Gradient L2 norm</small>
                    <b>{typeof data.metadata.gradient_l2_norm === 'number' ? data.metadata.gradient_l2_norm.toFixed(4) : '—'}</b>
                  </div>
                  <div className="xai-meta-card">
                    <small>Normalization</small>
                    <b>{data.metadata.normalization || 'ReLU, per-window max'}</b>
                  </div>
                </div>
              )}

              {isAllZeros ? (
                <div className="warning" style={{ background: '#fcf6e8', borderColor: '#d5a34c', color: '#684e1b' }}>
                  No positive evidence for this class in this window (ReLU Grad-CAM)
                </div>
              ) : (
                <svg viewBox="0 0 1000 240" style={{ background: '#fffdf9', border: '1px solid #ede7de', borderRadius: '6px' }}>
                  <defs>
                    <clipPath id="xai-curve-plot"><rect x={W_LEFT} y={15} width={W_WIDTH} height={195} /></clipPath>
                  </defs>
                  <line x1={W_LEFT} y1={isIg ? 120 : 200} x2={W_LEFT + W_WIDTH} y2={isIg ? 120 : 200} stroke="#ccc" strokeWidth="1" strokeDasharray={isIg ? "4 3" : undefined} />
                  <line x1={R_PEAK_X} y1={15} x2={R_PEAK_X} y2={210} stroke="#C6A96B" strokeWidth="1.5" strokeDasharray="4 3" />
                  <text x={R_PEAK_X} y={12} textAnchor="middle" fill="#A88742" fontSize="10">R-peak (1000 ms)</text>

                  {isIg ? (
                    <>
                      <path
                        d={`M ${W_LEFT} 120 ` + attrs.map((v: number, i: number) => {
                          const x = W_LEFT + (i / 719) * W_WIDTH
                          const y = v > 0 ? 120 - (v / maxAbsAttr) * 95 : 120
                          return `L ${x.toFixed(1)} ${y.toFixed(1)}`
                        }).join(' ') + ` L ${W_LEFT + W_WIDTH} 120 Z`}
                        clipPath="url(#xai-curve-plot)"
                        fill="rgba(46, 125, 50, 0.25)"
                      />
                      <path
                        d={`M ${W_LEFT} 120 ` + attrs.map((v: number, i: number) => {
                          const x = W_LEFT + (i / 719) * W_WIDTH
                          const y = v < 0 ? 120 - (v / maxAbsAttr) * 95 : 120
                          return `L ${x.toFixed(1)} ${y.toFixed(1)}`
                        }).join(' ') + ` L ${W_LEFT + W_WIDTH} 120 Z`}
                        clipPath="url(#xai-curve-plot)"
                        fill="rgba(198, 107, 82, 0.25)"
                      />
                      <polyline
                        points={attrs.map((v: number, i: number) => {
                          const x = W_LEFT + (i / 719) * W_WIDTH
                          const y = 120 - (v / maxAbsAttr) * 95
                          return `${x.toFixed(1)},${y.toFixed(1)}`
                        }).join(' ')}
                        clipPath="url(#xai-curve-plot)"
                        fill="none"
                        stroke="#5E4080"
                        strokeWidth="2"
                      />
                      <text x={W_LEFT - 6} y={30} textAnchor="end" fontSize="9" fill="#2E7D32">+{maxAbsAttr.toFixed(2)}</text>
                      <text x={W_LEFT - 6} y={123} textAnchor="end" fontSize="9" fill="#666">0.00</text>
                      <text x={W_LEFT - 6} y={215} textAnchor="end" fontSize="9" fill="#C66B52">-{maxAbsAttr.toFixed(2)}</text>
                    </>
                  ) : (
                    <>
                      <path
                        d={`M ${W_LEFT} 210 ` + attrs.map((v: number, i: number) => {
                          const x = W_LEFT + (i / 719) * W_WIDTH
                          const y = 210 - v * 180
                          return `L ${x.toFixed(1)} ${y.toFixed(1)}`
                        }).join(' ') + ` L ${W_LEFT + W_WIDTH} 210 Z`}
                        clipPath="url(#xai-curve-plot)"
                        fill="rgba(152, 129, 182, 0.3)"
                      />
                      <polyline
                        points={attrs.map((v: number, i: number) => {
                          const x = W_LEFT + (i / 719) * W_WIDTH
                          const y = 210 - v * 180
                          return `${x.toFixed(1)},${y.toFixed(1)}`
                        }).join(' ')}
                        clipPath="url(#xai-curve-plot)"
                        fill="none"
                        stroke="#9881B6"
                        strokeWidth="2"
                      />
                      <text x={W_LEFT - 6} y={35} textAnchor="end" fontSize="9" fill="#9881B6">1.00</text>
                      <text x={W_LEFT - 6} y={213} textAnchor="end" fontSize="9" fill="#9881B6">0.00</text>
                    </>
                  )}

                  {[0, 500, 1000, 1500, 2000].map(ms => {
                    const x = W_LEFT + (ms / 2000) * W_WIDTH
                    return (
                      <g key={ms}>
                        <line x1={x} y1={210} x2={x} y2={215} stroke="#888" />
                        <text x={x} y={228} textAnchor="middle" fontSize="10" fill="#666">{ms} ms</text>
                      </g>
                    )
                  })}
                </svg>
              )}
            </div>
          </>
        )}
      </article>
      <Disclaimer/>
    </Page>
  )
}
function Records() {
  const [records, setRecords] = useState<any[]>([])
  const [analyses, setAnalyses] = useState<Analysis[]>([])
  const [query, setQuery] = useState('')
  const [error, setError] = useState('')
  useEffect(() => { Promise.all([api.get('/ecg/records'), api.get('/analyses')]).then(([r, a]) => { setRecords(r.data); setAnalyses(a.data) }).catch(e => setError(e.message)) }, [])
  const visible = records.filter(x => `${x.filename} ${x.lead} ${x.id}`.toLowerCase().includes(query.toLowerCase()))
  async function deleteAnalysis(analysisId: string) {
    if (!window.confirm('Delete this analysis and its beats, predictions, explanations, reviews, and report files? The uploaded recording will remain.')) return
    try { await api.delete(`/analyses/${analysisId}`); setAnalyses(old => old.filter(a => a.id !== analysisId)) }
    catch (e: any) { setError(e.response?.data?.detail || e.message) }
  }
  return <Page tag="LIBRARY" title="Recordings" subtitle="Validated ECG files stored on this device.">{error && <ErrorBox text={error}/>}<article className="panel table-scroll"><label>Search recordings<input value={query} onChange={e => setQuery(e.target.value)} placeholder="File name, lead or recording ID"/></label><table className="recordings-table"><thead><tr><th>ECG recording</th><th>Lead</th><th>Frequency</th><th>Duration</th><th>Samples</th><th>Uploaded</th><th>Analysis status</th></tr></thead><tbody>{visible.map(r => <tr key={r.id}><td><b>{r.filename}</b><small className="record-id">ID {r.id}</small></td><td>{r.lead}</td><td>{r.sampling_rate} Hz</td><td>{r.duration_seconds.toFixed(1)} s</td><td>{r.sample_count.toLocaleString()}</td><td>{r.uploaded_at ? new Intl.DateTimeFormat(undefined, { dateStyle: 'medium' }).format(new Date(r.uploaded_at)) : '—'}</td><td>{analyses.filter(a => a.record_id === r.id).length ? analyses.filter(a => a.record_id === r.id).map(a => <div key={a.id} className="analysis-delete"><Link to={`/app/analyses/${a.id}`}><span className={`activity-status status-${a.status}`}>{a.status}</span> · {a.id.slice(0, 8)}</Link><button onClick={() => deleteAnalysis(a.id)}>Delete</button></div>) : <span className="record-not-analysed">Not analysed</span>}</td></tr>)}</tbody></table>{!visible.length && <Empty title={records.length ? 'No matching recordings' : 'No recordings saved'} text="Validated files appear here."/>}</article></Page>
}
function Reviews() { const [r,setR]=useState<any[]>([]),[queue,setQueue]=useState<any[]>([]);useEffect(()=>{Promise.all([api.get('/reviews'),api.get('/analyses')]).then(async([saved,analyses])=>{setR(saved.data);const batches=await Promise.all(analyses.data.filter((a:Analysis)=>a.status==='completed').map((a:Analysis)=>api.get(`/analyses/${a.id}/beats?limit=10000`).then(x=>x.data.map((b:any)=>({...b,analysis_id:a.id,model_version:a.model_version})))));setQueue(batches.flat().filter((b:any)=>b.prediction?.review_flag&&!b.review))})},[]);return <Page tag="HUMAN REVIEW" title="Review flagged beats." subtitle="Human labels are separate; model predictions are preserved."><article className="panel"><h2>Flagged prediction queue</h2><table><thead><tr><th>Analysis</th><th>Beat</th><th>Model class</th><th>Confidence</th><th/></tr></thead><tbody>{queue.map(b=><tr key={b.id}><td>{b.analysis_id.slice(0,8)}</td><td>{b.beat_index+1}</td><td>{b.prediction.predicted_class}</td><td>{(b.prediction.confidence*100).toFixed(1)}%</td><td><Link to={`/app/analyses/${b.analysis_id}?beat=${b.id}`}>Open beat for review</Link></td></tr>)}</tbody></table>{!queue.length&&<Empty title="No flagged beats waiting" text="The queue is derived from actual saved analyses."/>}</article><article className="panel"><h2>Saved reviews</h2><table><thead><tr><th>Beat</th><th>Original model class</th><th>Reviewed class</th><th>Status</th><th>Note</th></tr></thead><tbody>{r.map(x=><tr key={x.id}><td>{x.beat_id.slice(0,8)}</td><td>{x.original_prediction}</td><td>{x.reviewed_label||'—'}</td><td>{x.status}</td><td>{x.note||'—'}</td></tr>)}</tbody></table>{!r.length&&<Empty title="No reviews saved" text="Submitted reviews will appear here."/>}</article></Page> }
function Reports() { const [r,setR]=useState<Analysis[]>([]);useEffect(()=>{api.get('/analyses').then(x=>setR(x.data))},[]);return <Page tag="REPORTS" title="Research exports." subtitle="PDF summaries and beat-level CSVs from completed analyses.">{r.filter(a=>a.status==='completed').map(a=><article className="panel report" key={a.id}><div><h2>{a.id.slice(0,8)} · {a.beat_count} beats</h2><p>{a.model_version} · {a.preprocessing_version}</p></div><span><a href={`/api/v1/reports/${a.id}?format=pdf`}>PDF</a> · <a href={`/api/v1/reports/${a.id}?format=csv`}>CSV</a></span></article>)}{!r.some(a=>a.status==='completed')&&<article className="panel"><Empty title="No reports available" text="Complete an analysis to export a report."/></article>}</Page> }
function Model() { const [d,setD]=useState<any>(null),[e,setE]=useState('');useEffect(()=>{api.get('/model-info').then(x=>setD(x.data)).catch(x=>setE(x.message))},[]);const shape=(x:any[])=>`(${x.map(v=>v===null?'batch':v).join(', ')})`;return <Page tag="MODEL INFORMATION" title="Evidence before interpretation." subtitle="Metrics are read from local evaluation artifacts."><ModelFlow acc={d?.test_metrics?.accuracy ?? 0.906} f1={d?.test_metrics?.macro_f1 ?? 0.473} auc={d?.test_metrics?.macro_roc_auc ?? 0.837} fromApi={!!d?.test_metrics}/>{e&&<ErrorBox text={e}/ >}{d&&<><article className="panel"><div className="eyebrow">SELECTED EXPORT</div><h2>{d.model}</h2><p>Input {shape(d.input_shape)} · output {shape(d.output_shape)} · classes {d.classes.join(' / ')} · {d.lead} at {d.sampling_rate} Hz</p><small>SHA-256 {d.model_sha256}</small></article><div className="metrics"><Metric label="Test accuracy" value={d.test_metrics.accuracy.toFixed(4)} note="Fixed MIT-BIH test"/><Metric label="Macro F1" value={d.test_metrics.macro_f1.toFixed(4)} note="Fixed test"/><Metric label="Macro ROC-AUC" value={d.test_metrics.macro_roc_auc.toFixed(4)} note="One-vs-rest"/><Metric label="Grouped CV F1" value={Number(d.grouped_cv_pooled.macro_f1).toFixed(4)} note="Pooled folds"/></div><article className="panel"><h2>Class-wise test performance</h2><table><thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead><tbody>{Object.entries(d.test_metrics.class_metrics).map(([c,v]:[string,any])=><tr key={c}><td style={{color:colors[c]}}>{c}</td><td>{v.precision.toFixed(4)}</td><td>{v.recall.toFixed(4)}</td><td>{v.f1.toFixed(4)}</td><td>{v.support}</td></tr>)}</tbody></table><div className="warning">S and F recall are weak: S recall {(d.test_metrics.class_metrics.S.recall*100).toFixed(1)}% (precision {(d.test_metrics.class_metrics.S.precision*100).toFixed(1)}%); F recall {(d.test_metrics.class_metrics.F.recall*100).toFixed(1)}% (precision {(d.test_metrics.class_metrics.F.precision*100).toFixed(1)}%).</div></article><article className="panel"><h2>Confusion matrix · true rows / predicted columns</h2><pre>{d.test_metrics.confusion_matrix_true_rows_pred_columns.map((x:number[])=>x.join('   ')).join('\n')}</pre></article></>}</Page> }
function SettingsPage(){return <Page tag="SETTINGS" title="Local by design." subtitle="Runtime details and model input expectations."><section className="settings-layout"><article className="settings-intro"><div className="eyebrow">RUNTIME NOTES</div><h2>Research stays close to the signal.</h2><p>NeuroCardio runs as a local research workspace. Recordings and analysis records are stored in the project's local SQLite database; uploaded signals are not a cloud service.</p><Link to="/app/model">Review model information <ArrowRight size={15}/></Link></article><article className="settings-runtime"><div className="eyebrow">CURRENT CONFIGURATION</div><h2>Analysis environment</h2><div className="runtime-facts"><div><span>Storage</span><b>Local SQLite</b><small>backend/data</small></div><div><span>Model input</span><b>360 Hz</b><small>Other sampling rates are rejected</small></div><div><span>Signal window</span><b>720 samples</b><small>Single ECG lead</small></div><div><span>Analysis mode</span><b>Retrospective</b><small>Beat-level classification</small></div></div></article></section></Page>}
function AppRoutes(){return <Routes><Route path="dashboard" element={<Dash/>}/><Route path="analysis" element={<LatestAnalysis/>}/><Route path="new" element={<NewPage/>}/><Route path="analyses/:id" element={<Result/>}/><Route path="explainability" element={<Explain/>}/><Route path="recordings" element={<Records/>}/><Route path="reviews" element={<Reviews/>}/><Route path="reports" element={<Reports/>}/><Route path="model" element={<Model/>}/><Route path="settings" element={<SettingsPage/>}/><Route path="*" element={<Navigate to="/app/dashboard" replace/>}/></Routes>}
export default function App(){return <BrowserRouter><div className="pf"><Routes><Route path="/" element={<HomePage/>}/><Route path="/app" element={<Navigate to="/app/dashboard" replace/>}/><Route path="/app/*" element={<Shell><AppRoutes/></Shell>}/><Route path="/*" element={<Shell><AppRoutes/></Shell>}/></Routes></div></BrowserRouter>}
