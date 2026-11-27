import { useSearchParams } from 'react-router-dom'
import { useApi } from '../api/useApi'
import { useAnalysisWithRecord } from '../api/useAnalysisWithRecord'
import { ClassDistribution } from '../components/ClassDistribution'
import { secs } from '../components/format'
import { SplitNote } from '../components/SplitNote'
import { Empty, Load, PartialNotice } from '../components/states'
import { Card, KeyValues, PageHead } from '../components/ui'
import { WaveLink } from '../components/WaveLink'
import { WeaknessNote } from '../components/WeaknessNote'
import { Trace } from '../components/waveform/Trace'
import type { HistoryRow } from './History'

function Side({ id, start }: { id: string; start: number }) {
  const { analysis, record } = useAnalysisWithRecord(id)
  const r = record.data
  const signal = useApi<{ amplitude: number[] }>(r ? `/recordings/${r.id}/signal?filtered=true&count=3600&start_sample=${Math.round(Math.min(start, Math.max(0, r.duration_seconds - 10)) * r.sampling_rate)}` : null)
  const quality = useApi<{ overall: string }>(r ? `/recordings/${r.id}/quality?analysis_id=${id}` : null)
  return <Load q={analysis} lines={8}>{a => <Load q={record} lines={8}>{rec => <Card title={rec.filename} aside={<WaveLink analysisId={a.id} index={0} label="Open in viewer" />}>
    <SplitNote record={rec} />
    {rec.input_warnings?.map(w => <PartialNotice key={w}>{w}</PartialNotice>)}
    <KeyValues rows={[['Lead', rec.lead], ['Duration', secs(rec.duration_seconds)], ['Windows', a.beat_count.toLocaleString()], ['Flagged for review', String(a.summary.flagged_for_review_count ?? 0)],
      ['Beat positions from', a.peak_source ?? '—'], ['Model version', a.model_version], ['Signal quality', quality.data ? <span className={`tag ${quality.data.overall.toLowerCase()}`}>{quality.data.overall}</span> : quality.error ? 'unavailable' : 'measuring']]} />
    <h3>Windows by model class</h3><ClassDistribution counts={a.summary.class_counts ?? {}} source={`analysis#${a.id}.class_counts`} />
    <h3>10 s from {Math.min(start, Math.max(0, rec.duration_seconds - 10)).toFixed(0)} s</h3>
    <Load q={signal}>{s => <Trace values={s.amplitude} fs={rec.sampling_rate} startSeconds={Math.min(start, Math.max(0, rec.duration_seconds - 10))} label="Bandpass filtered" unit={rec.units ?? 'signal units'} height={220} />}</Load>
  </Card>}</Load>}</Load>
}

function Workspace({ rows }: { rows: HistoryRow[] }) {
  const [params, setParams] = useSearchParams()
  const done = rows.filter(r => r.status === 'completed')
  const start = Number(params.get('t') ?? 0)
  const set = (k: string, v: string) => setParams(p => { const n = new URLSearchParams(p); n.set(k, v); return n }, { replace: true })
  if (done.length < 2) return <Empty title="Two completed analyses are needed">Analyse another recording in the Analysis Studio or Batch Analysis.</Empty>
  const ids = ['left', 'right'].map((k, i) => { const v = params.get(k); return done.some(r => r.id === v) ? v! : done[i].id })
  const [a, b] = ids.map(id => done.find(r => r.id === id)!)
  return <>
    <Card><div className="grid-3">{(['left', 'right'] as const).map((k, i) => <label key={k} className="field">{i ? 'Second' : 'First'} analysis<select value={ids[i]} onChange={e => set(k, e.target.value)}>{done.map(r => <option key={r.id} value={r.id}>{r.filename} · {r.id.slice(0, 8)}</option>)}</select></label>)}
      <label className="field">Excerpt start (seconds)<input type="number" min={0} value={start} onChange={e => set('t', e.target.value)} /></label></div></Card>
    {a.id === b.id && <PartialNotice>Both sides show the same analysis.</PartialNotice>}
    {a.model_sha256 !== b.model_sha256 && <PartialNotice>These analyses used different model files, so their outputs are not directly comparable.</PartialNotice>}
    <p className="notice">A difference in class counts between two recordings is not a clinical comparison. Lengths, patients and signal quality all differ.</p>
    <div className="grid-2">{ids.map((id, i) => <Side key={`${i}${id}`} id={id} start={start} />)}</div>
  </>
}

export default function Compare() {
  const q = useApi<HistoryRow[]>('/history')
  return <><PageHead title="ECG Comparison" sub="Two stored analyses side by side." /><WeaknessNote /><Load q={q} lines={8}>{rows => <Workspace rows={rows} />}</Load></>
}
