import { useState } from 'react'
import type { DataReport } from '../api/study'
import { useApi } from '../api/useApi'
import { Num } from '../components/Num'
import { Empty, Load, PartialNotice } from '../components/states'
import { Card, ClassBadge, PageHead, Table } from '../components/ui'
import type { WindowData } from '../components/WindowStages'
import { Trace } from '../components/waveform/Trace'
import type { ModelMetrics } from '../api/metrics'
import { CLASS_CODES, classFill, INCART_NOTICE } from '../theme/theme'

type SummaryData = {
  app_model: null | { dataset: ModelMetrics['metrics']['dataset']; preprocessing: Record<string, unknown> }; app_model_problem: string | null
  study: DataReport | null; study_problem: string | null
  minority_concentration: { split: string; class: string; total_beats: string; top_record: string; top_record_share: string }[]
}
type RecordRow = { record: string; split: string; lead: string; duration_seconds: number; annotated_beats: Record<string, number> }
const PARTS = ['train', 'validation', 'test'] as const

function Facts({ d }: { d: SummaryData }) {
  const app = d.app_model
  return <>
    {app ? <Card title={app.dataset.name}>
      <p><Num v={app.dataset.records_total} src="metrics#dataset.records_total" /> records, <Num v={app.dataset.records_used} src="metrics#dataset.records_used" /> used, <Num v={app.dataset.subject_groups_used} src="metrics#dataset.subject_groups_used" /> subjects.</p>
      {(['usable_windows', 'annotated_beats'] as const).map(kind => <Table key={kind} caption={kind === 'usable_windows' ? 'Usable windows (complete 2 s windows; what the model saw, before augmentation)' : 'Annotated beats (includes beats too close to a record edge to use)'}
        head={['Part', ...CLASS_CODES.map(c => <ClassBadge key={c} code={c} />)]} rows={PARTS.map(p => [p, ...CLASS_CODES.map(c => <Num key={c} v={app.dataset.class_counts[p][kind][c]} src={`metrics#dataset.class_counts.${p}.${kind}.${c}`} />)])} />)}
      <h3>Class imbalance, usable training windows</h3>
      <div className="probs">{CLASS_CODES.map(c => {
        const counts = app.dataset.class_counts.train.usable_windows
        const total = CLASS_CODES.reduce((a, k) => a + counts[k], 0)
        return <div key={c} className="prob-row"><ClassBadge code={c} /><div className="bar"><i style={{ width: `${counts[c] / total * 100}%`, background: classFill(c) }} /></div><span className="num"><Num v={counts[c] / total} fmt="pct1" src={`totalshare:metrics#dataset.class_counts.train.usable_windows.${c}`} /></span></div>
      })}</div>
    </Card> : <Empty title={d.app_model_problem ?? 'Not available'} />}
    <Card title="Rare classes come from few patients">
      {d.minority_concentration.length ? <Table head={['Part', 'Class', 'Annotated beats', 'Largest record', 'Its share']} rows={d.minority_concentration.map((r, i) => [r.split, r.class,
        <Num key="t" v={Number(r.total_beats)} src={`minority#${i}.total_beats`} />, <Num key="id" v={r.top_record} fmt="raw" src={`minority#${i}.top_record`} />, <Num key="s" v={Number(r.top_record_share)} fmt="pct1" src={`minority#${i}.top_record_share`} />])} /> : <Empty title="Concentration table not available." />}
    </Card>
    {d.study ? <Card title="Multi-database study data">
      <Table caption="All usable records per database" head={['Database', 'Lead', 'Source rate (Hz)', 'Records', 'Patients', ...CLASS_CODES]} rows={Object.entries(d.study.databases).map(([k, v]) => [k, <Num key="l" v={d.study!.lead[k]} fmt="raw" src={`datareport#lead.${k}`} />,
        <Num key="f" v={v.source_fs_hz} fmt="raw" src={`datareport#databases.${k}.source_fs_hz`} />, <Num key="r" v={v.records} src={`datareport#databases.${k}.records`} />, <Num key="p" v={v.patients} src={`datareport#databases.${k}.patients`} />,
        ...CLASS_CODES.map(c => <Num key={c} v={v.windows[c]} src={`datareport#databases.${k}.windows.${c}`} />)])} />
      <Table caption="Patient-wise study split" head={['Part', 'Records', 'Patients', ...CLASS_CODES]} rows={Object.entries(d.study.splits).map(([k, v]) => [k,
        <Num key="r" v={v.records} src={`datareport#splits.${k}.records`} />, <Num key="p" v={v.patients} src={`datareport#splits.${k}.patients`} />, ...CLASS_CODES.map(c => <Num key={c} v={v.windows[c]} src={`datareport#splits.${k}.windows.${c}`} />)])} />
      <p className="notice">{INCART_NOTICE}</p>
    </Card> : <Empty title={d.study_problem ?? 'Study data report not generated yet.'} />}
  </>
}

function Representatives() {
  const q = useApi<{ rule: string; examples: { label: string; record: string; sample: number; candidates: number; model_input: number[] }[] }>('/dataset/representatives')
  return <Card title="One real training window per class"><Load q={q} lines={6}>{d => <>
    <p className="muted">{d.rule}</p>
    <div className="grid-2">{d.examples.map(e => <div key={e.label}><ClassBadge code={e.label} long />
      <Trace values={e.model_input} fs={360} label={`Record ${e.record}, sample ${e.sample}`} unit="z-score" height={170} band={{ from: 270, to: 450, color: classFill(e.label) }} />
      <p className="small muted">Chosen from <Num v={e.candidates} src={`metrics#dataset.class_counts.train.usable_windows.${e.label}`} /> training windows.</p></div>)}</div>
  </>}</Load></Card>
}

type Segment = { raw: number[]; filtered: number[]; beats: { offset: number; label: string }[]; units: string; lead: string; sampling_rate: number; start_sample: number; duration_seconds: number; split: string }

function Browser({ records }: { records: RecordRow[] }) {
  const [record, setRecord] = useState(records[0].record)
  const [label, setLabel] = useState('N')
  const [index, setIndex] = useState(0)
  const [start, setStart] = useState(0)
  const row = records.find(r => r.record === record)!
  const count = row.annotated_beats[label]
  const segment = useApi<Segment>(`/dataset/records/${record}/segment?start=${start}&seconds=10`)
  const win = useApi<WindowData & { index: number; count: number }>(count ? `/dataset/records/${record}/window?class=${label}&index=${index}` : null)
  const max = Math.max(0, Math.floor(row.duration_seconds - 10))
  return <div data-testid="record-browser"><Card title="Browse a record" aside={<span className="tag">app model split: {row.split}</span>}>
    <div className="row">
      <label className="field">Record<select aria-label="Record" value={record} onChange={e => { setRecord(e.target.value); setIndex(0); setStart(0) }}>{records.map(r => <option key={r.record} value={r.record}>{r.record} · {r.split}</option>)}</select></label>
      <label className="field">Class<select aria-label="Class" value={label} onChange={e => { setLabel(e.target.value); setIndex(0) }}>{CLASS_CODES.map(c => <option key={c} value={c}>{c} ({row.annotated_beats[c]} annotated)</option>)}</select></label>
      <label className="field" style={{ flex: 1, minWidth: 200 }}>Start time: {start} s<input type="range" min={0} max={max} value={start} onChange={e => setStart(Number(e.target.value))} /></label>
    </div>
    <Load q={segment}>{s => <>
      <Trace values={s.raw} fs={s.sampling_rate} startSeconds={s.start_sample / s.sampling_rate} label={`Record ${record}, ${s.lead}, raw`} unit={s.units} height={180} />
      <p className="small">Reference beats in view: {s.beats.map((b, i) => <span key={i}><ClassBadge code={b.label} /> <span className="num">{((s.start_sample + b.offset) / s.sampling_rate).toFixed(1)}s</span> </span>)}</p>
    </>}</Load>
    <h3>Effect of preprocessing on one {label} window</h3>
    {!count ? <Empty title={`Record ${record} has no ${label} beats`} /> : <>
      <div className="row"><button type="button" disabled={index <= 0} onClick={() => setIndex(i => i - 1)}>Previous {label}</button><button type="button" disabled={!!win.data && index >= win.data.count - 1} onClick={() => setIndex(i => i + 1)}>Next {label}</button>{win.data && <span className="num">{win.data.index + 1} of {win.data.count}</span>}</div>
      <Load q={win}>{w => <div className="stack">
        <Trace values={w.raw} fs={360} label="Raw" unit={w.units ?? 'signal units'} height={170} />
        <Trace values={w.filtered} fs={360} label="Bandpass filtered" unit={w.units ?? 'signal units'} height={170} />
        <Trace values={w.model_input} fs={360} label="Standardized" unit="z-score" height={170} />
      </div>}</Load></>}
  </Card></div>
}

export default function Dataset() {
  const q = useApi<SummaryData>('/dataset/summary')
  const records = useApi<RecordRow[]>('/dataset/records')
  return <><PageHead title="Dataset Explorer" sub="Where the data come from, how imbalanced they are, and what real windows look like." />
    <PartialNotice>S and F are rare, and much of each comes from very few patients. This limits what any model can learn and what any score can show.</PartialNotice>
    <Load q={q} lines={10}>{d => <Facts d={d} />}</Load>
    <Representatives />
    <Load q={records} lines={6}>{r => r.length ? <Browser records={r} /> : <Empty title="No records available" />}</Load></>
}
