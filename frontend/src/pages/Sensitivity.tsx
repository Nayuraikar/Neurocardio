import { useState } from 'react'
import type { Analysis, Beat } from '../api/types'
import { MODEL_LABELS } from '../api/study'
import { useApi } from '../api/useApi'
import { Num } from '../components/Num'
import { Empty, ErrorState, Load } from '../components/states'
import { Card, ClassBadge, PageHead, Table } from '../components/ui'
import { WaveLink } from '../components/WaveLink'
import { WithWindow } from '../components/WithWindow'

type Point = { key: 'argmax' | 'thresholds'; thresholds: Record<string, number[]> | null; validation: Record<string, number> | null }
type Points = {
  available: boolean; message?: string; error?: boolean; primary: string; validation_message: string | null
  models: Record<string, { rule: string; points: Point[] }>; majority: { accuracy: number; macro_f1: number } | null
}
const NAMES = { argmax: 'Plain argmax', thresholds: 'Validation-frozen class thresholds' }

function Slider({ data }: { data: Points }) {
  const [model, setModel] = useState(data.primary)
  const [position, setPosition] = useState(0)
  const entry = data.models[model]
  // order the frozen points by validation S+F recall, so "right" always means higher recall when scores exist
  const points = [...entry.points].sort((a, b) => ((a.validation?.S_recall ?? 0) + (a.validation?.F_recall ?? 0)) - ((b.validation?.S_recall ?? 0) + (b.validation?.F_recall ?? 0)))
  const point = points[Math.min(position, points.length - 1)]
  const src = `operating-points#models.${model}.points.${entry.points.indexOf(point)}.validation`
  return <>
    <label className="field">Study model<select value={model} onChange={e => { setModel(e.target.value); setPosition(0) }}>{Object.keys(data.models).map(m => <option key={m} value={m}>{MODEL_LABELS[m] ?? m}{m === data.primary ? ' (primary)' : ''}</option>)}</select></label>
    {points.length < 2
      ? <p>Selection kept plain argmax for this model, so there is only one frozen operating point.</p>
      : <label className="field" style={{ marginTop: 12 }}>
        <span className="row between"><span>Lower S+F recall</span><span>Higher S+F recall (validation)</span></span>
        <input type="range" min={0} max={points.length - 1} step={1} value={position} onChange={e => setPosition(Number(e.target.value))} aria-valuetext={NAMES[point.key]} />
      </label>}
    <p><strong>{NAMES[point.key]}</strong></p>
    {point.thresholds && <p className="small muted">Per-class probability multipliers (N, S, V, F) by seed: {Object.entries(point.thresholds).map(([s, t]) => <span key={s} className="num">{s}: [{t.map(v => v.toFixed(2)).join(', ')}] </span>)}</p>}
    {point.validation && data.majority
      ? <Table caption="Validation data (optimistic: these thresholds and early stopping used validation data)" head={['Macro-F1', 'S recall', 'F recall', 'S precision', 'F precision', 'Accuracy', 'Majority-class accuracy', 'Majority-class macro-F1']}
        rows={[[...['macro_f1', 'S_recall', 'F_recall', 'S_precision', 'F_precision', 'accuracy'].map(k => <Num key={k} v={point.validation![k]} fmt="f3" src={`${src}.${k}`} />),
          <Num key="ma" v={data.majority.accuracy} fmt="f3" src="operating-points#majority.accuracy" />, <Num key="mf" v={data.majority.macro_f1} fmt="f3" src="operating-points#majority.macro_f1" />]]} />
      : <Empty title="No validation scores for these points">{data.validation_message}</Empty>}
    <p className="notice">These are the only operating points that selection froze. There is no separate “high-precision” threshold set, and no point is chosen on test data. S and F remain weak classes; check both recall and precision.</p>
  </>
}

function Queue({ analysis, beats }: { analysis: Analysis; beats: Beat[] }) {
  const [flaggedOnly, setFlaggedOnly] = useState(false)
  const ranked = beats.filter(b => b.review?.status !== 'reviewed' && (!flaggedOnly || b.prediction.review_flag))
    .map(b => { const sf = b.prediction.probabilities[1] + b.prediction.probabilities[3]; return { b, sf, uncertainty: 1 - b.prediction.margin, score: sf + 1 - b.prediction.margin } })
    .sort((x, y) => y.score - x.score)
  return <Card title="Priority review queue" aside={<label className="row"><input type="checkbox" checked={flaggedOnly} onChange={e => setFlaggedOnly(e.target.checked)} /> Flagged windows only</label>}>
    <p className="muted">Priority = P(S) + P(F) + (1 − top-2 margin). A sorting aid for this recording; it has not been evaluated as a triage method. Reviewed windows drop out.</p>
    {ranked.length ? <Table caption={`Top ${Math.min(50, ranked.length)} of ${ranked.length} unreviewed windows`} head={['Rank', 'Window', 'Time (s)', 'Model class', 'P(S)+P(F)', '1 − margin', 'Priority', 'Flag', '']}
      rows={ranked.slice(0, 50).map((r, i) => [i + 1, r.b.beat_index + 1, r.b.timestamp_seconds.toFixed(2), <ClassBadge key="c" code={r.b.prediction.predicted_class} />,
        r.sf.toFixed(3), r.uncertainty.toFixed(3), r.score.toFixed(3), r.b.prediction.review_flag ? 'Flagged' : '',
        <WaveLink key="l" analysisId={analysis.id} index={r.b.beat_index} />])} />
      : <Empty title="Nothing left in the queue" />}
  </Card>
}

export default function Sensitivity() {
  const q = useApi<Points>('/study/operating-points')
  return <>
    <PageHead title="Sensitivity & Review Queue" sub="Decision rules frozen on validation data, and which windows to look at first." />
    <Card title="Sensitivity (study models)"><Load q={q}>{d => d.available
      ? <><p className="muted">These rules belong to the study models trained for the multi-database study. They are <strong>not applied</strong> to your recording, which was classified by the app’s own model with plain argmax.</p><Slider data={d} /></>
      : d.error ? <ErrorState error={{ status: null, message: d.message ?? '' }} onRetry={q.reload} />
        : <><Empty title={d.message ?? 'Threshold selection not run yet.'}>The slider appears once <code>selection.json</code> is in <code>backend/artifacts/</code>.</Empty>
          <label className="field" style={{ marginTop: 12 }}><span className="row between"><span>Lower S+F recall</span><span>Higher S+F recall</span></span><input type="range" min={0} max={1} value={0} disabled readOnly aria-label="Sensitivity (unavailable)" /></label></>}</Load></Card>
    <WithWindow>{({ analysis, beats }) => <Queue analysis={analysis} beats={beats} />}</WithWindow>
  </>
}
