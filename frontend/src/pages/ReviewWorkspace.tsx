import { useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { post, toApiError, type ApiError } from '../api/client'
import type { Analysis, Beat, RecordInfo, Review } from '../api/types'
import { useApi } from '../api/useApi'
import { useAnalysisWithRecord } from '../api/useAnalysisWithRecord'
import { useBeats } from '../api/useBeats'
import { when } from '../components/format'
import { SplitNote } from '../components/SplitNote'
import { Empty, ErrorState, Load, ReviewReady, SuccessBanner } from '../components/states'
import { Card, ClassBadge, PageHead, ProbBars, Table } from '../components/ui'
import type { WindowData } from '../components/WindowStages'
import { WindowTable } from '../components/WindowTable'
import { Trace } from '../components/waveform/Trace'
import { classFill, CLASS_CODES, CLASSES } from '../theme/theme'

type SignOff = { id: string; status: string; note: string | null; reviewer: string; created_at: string }

function Assess({ analysis, record, beat, onSaved }: { analysis: Analysis; record: RecordInfo; beat: Beat; onSaved: (message: string) => void }) {
  const win = useApi<WindowData>(`/recordings/${record.id}/window?sample=${beat.sample_index}`)
  const history = useApi<Review[]>(`/reviews?analysis_id=${analysis.id}&limit=5000`)
  const [label, setLabel] = useState(beat.review?.reviewed_label ?? '')
  const [note, setNote] = useState(beat.review?.note ?? '')
  const [error, setError] = useState<ApiError | null>(null)
  const [busy, setBusy] = useState(false)
  const save = async (value: string | null) => {
    setBusy(true); setError(null)
    try { await post('/reviews', { beat_id: beat.id, reviewed_label: value, note: note || null }); history.reload(); onSaved(value ? `Saved reviewer label ${value} for window ${beat.beat_index + 1}.` : `Withdrew the reviewer label for window ${beat.beat_index + 1}.`) }
    catch (e) { setError(toApiError(e)) }
    finally { setBusy(false) }
  }
  const mine = history.data?.filter(r => r.beat_id === beat.id) ?? []
  return <Card title={`Window #${beat.beat_index + 1} at ${beat.timestamp_seconds.toFixed(2)} s`} aside={<Link to={`/xai?analysis=${analysis.id}&window=${beat.beat_index}`}>Explain in XAI Studio</Link>}>
    <Load q={win}>{w => <Trace values={w.model_input} fs={w.sampling_rate} startSeconds={w.start_sample / w.sampling_rate} label="Model input window" unit="unitless z-score" height={200} band={{ from: 270, to: 450, color: classFill(beat.prediction.predicted_class) }} />}</Load>
    <p>Model prediction (read-only) <ClassBadge code={beat.prediction.predicted_class} long /></p>
    <ProbBars probs={beat.prediction.probabilities} />
    <h3>Reviewer assessment</h3>
    <div className="row" role="group" aria-label="Reviewer label">{CLASS_CODES.map(c => <button type="button" key={c} aria-pressed={label === c} onClick={() => setLabel(c)}>{c} · {CLASSES[c].name}</button>)}</div>
    <label className="field" style={{ marginTop: 8 }}>Note<textarea rows={2} value={note} onChange={e => setNote(e.target.value)} /></label>
    <div className="row" style={{ marginTop: 8 }}><button type="button" className="primary" disabled={!label || busy} onClick={() => void save(label)}>Save assessment</button>
      <button type="button" disabled={beat.review?.status !== 'reviewed' || busy} onClick={() => void save(null)}>Withdraw label</button></div>
    {error && <ErrorState title="The assessment was not saved" error={error} />}
    {!!mine.length && <Table caption="Every saved assessment of this window (nothing is overwritten)" head={['When', 'Status', 'Label', 'Reviewer', 'Note']} rows={mine.map(r => [when(r.created_at), r.status, r.reviewed_label ?? '—', r.reviewer, r.note ?? ''])} />}
  </Card>
}

function Workspace({ analysis, record }: { analysis: Analysis; record: RecordInfo }) {
  const beats = useBeats(analysis.id)
  const signOffs = useApi<SignOff[]>(`/analyses/${analysis.id}/sign-offs`)
  const [picked, setPicked] = useState<string | null>(null)
  const [all, setAll] = useState(false)
  const [note, setNote] = useState('')
  const [saved, setSaved] = useState<string | null>(null)
  const [error, setError] = useState<ApiError | null>(null)
  const sign = async (status: 'reviewed' | 'reopened') => {
    try { await post(`/analyses/${analysis.id}/sign-offs`, { status, note: note || null }); setNote(''); setError(null); setSaved(status === 'reviewed' ? 'Analysis marked reviewed.' : 'Analysis reopened.'); signOffs.reload() }
    catch (e) { setError(toApiError(e)) }
  }
  const done = signOffs.data?.[0]?.status === 'reviewed'
  return <Load q={beats} lines={8}>{rows => {
    const pending = rows.filter(b => b.prediction.review_flag && b.review?.status !== 'reviewed')
    const shown = all ? rows : pending
    const beat = rows.find(b => b.id === picked) ?? shown[0]
    return <>
      <SplitNote record={record} />
      {saved && <SuccessBanner>{saved}</SuccessBanner>}
      {!done && pending.length > 0 && <ReviewReady pending={pending.length} to={`/history/${analysis.id}#queue`} />}
      <Card title={`Windows (${shown.length})`} aside={<label className="row"><input type="checkbox" checked={all} onChange={e => setAll(e.target.checked)} /> Show every window</label>}>
        <div id="queue">{shown.length ? <WindowTable beats={shown} selectedId={beat?.id} onSelect={b => setPicked(b.id)} /> : <Empty title="No flagged window is waiting">Unflagged windows can still be wrong. Tick “Show every window” to assess any of them.</Empty>}</div>
      </Card>
      {beat && <Assess key={`${beat.id}${beat.review?.id}`} analysis={analysis} record={record} beat={beat} onSaved={message => { setPicked(null); setSaved(message); beats.reload() }} />}
      <Card title="Analysis sign-off" aside={<Link className="btn" to={`/history/${analysis.id}/report`}>Build report</Link>}>
        <label className="field">Note on the whole analysis<textarea rows={3} value={note} onChange={e => setNote(e.target.value)} /></label>
        <div className="row" style={{ marginTop: 8 }}>{done ? <button type="button" onClick={() => void sign('reopened')}>Reopen</button> : <button type="button" className="primary" onClick={() => void sign('reviewed')}>Mark analysis reviewed</button>}
          {!done && pending.length > 0 && <span className="muted">{pending.length} flagged windows are still unassessed.</span>}</div>
        {error && <ErrorState title="Sign-off was not saved" error={error} />}
        <Load q={signOffs}>{s => s.length ? <Table head={['When', 'Status', 'Reviewer', 'Note']} rows={s.map(r => [when(r.created_at), r.status, r.reviewer, r.note ?? ''])} /> : <p className="muted">Not signed off yet.</p>}</Load>
      </Card>
    </>
  }}</Load>
}

export default function ReviewWorkspace() {
  const { id = '' } = useParams()
  const { analysis, record } = useAnalysisWithRecord(id)
  return <><PageHead title="Review workspace" sub="Reviewer assessments are stored in their own table and never change a prediction." />
    <Load q={analysis}>{a => a.status !== 'completed' ? <Empty title={`This analysis is ${a.status}`}>{a.error_message}</Empty> : <Load q={record}>{r => <Workspace key={a.id} analysis={a} record={r} />}</Load>}</Load></>
}
