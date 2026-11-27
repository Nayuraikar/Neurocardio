import { useState } from 'react'
import type { ModelMetrics } from '../api/metrics'
import { score } from '../api/scoring'
import type { Analysis, Beat, RecordInfo } from '../api/types'
import { useApi } from '../api/useApi'
import { useBeats } from '../api/useBeats'
import { ConfusionMatrix } from '../components/ConfusionMatrix'
import { HeadlineMetrics } from '../components/HeadlineMetrics'
import { MapView } from '../components/MapView'
import { NeedsAnalysis } from '../components/NeedsAnalysis'
import { SplitNote } from '../components/SplitNote'
import { Empty, Load } from '../components/states'
import { Card, ClassBadge, PageHead, ProbBars, Table } from '../components/ui'
import { WaveLink } from '../components/WaveLink'
import { WeaknessNote } from '../components/WeaknessNote'
import type { WindowData } from '../components/WindowStages'
import { WindowTable } from '../components/WindowTable'
import { Trace } from '../components/waveform/Trace'
import { classFill, CLASS_CODES, XAI_NOTICE } from '../theme/theme'

type Reference = { available: boolean; message: string | null; labels: Record<string, string> }
type Filter = { kind: 'fn' | 'fp'; cls: string } | { kind: 'cell'; truth: string; predicted: string }

function Detail({ analysis, record, beat, truth }: { analysis: Analysis; record: RecordInfo; beat: Beat; truth: string }) {
  const q = useApi<WindowData>(`/recordings/${record.id}/window?sample=${beat.sample_index}`)
  const [method, setMethod] = useState<'gradcam' | 'ig' | null>(null)
  const predicted = beat.prediction.predicted_class
  return <Card title={`Window #${beat.beat_index + 1} at ${beat.timestamp_seconds.toFixed(2)} s`} aside={<WaveLink analysisId={analysis.id} index={beat.beat_index} label="Open in ECG Viewer" />}>
    <p>Reference <ClassBadge code={truth} long /> · model <ClassBadge code={predicted} long /></p>
    <Load q={q}>{w => <>
      <Trace values={w.model_input} fs={w.sampling_rate} startSeconds={w.start_sample / w.sampling_rate} label="Model input window" unit="unitless z-score" height={200} band={{ from: 270, to: 450, color: classFill(predicted) }} />
      <ProbBars probs={beat.prediction.probabilities} />
      <div className="row" style={{ marginTop: 10 }}>
        <button type="button" aria-pressed={method === 'ig'} onClick={() => setMethod('ig')}>Integrated Gradients for {predicted}</button>
        <button type="button" aria-pressed={method === 'gradcam'} onClick={() => setMethod('gradcam')}>Grad-CAM for {predicted}</button>
      </div>
      {method && <><MapView key={`${beat.id}${method}`} analysisId={analysis.id} beat={beat} cls={predicted} method={method} signal={w.model_input} /><p className="notice">{XAI_NOTICE} It does not show why the answer is wrong.</p></>}
    </>}</Load>
  </Card>
}

function Lab({ analysis, record, beats, reference }: { analysis: Analysis; record: RecordInfo; beats: Beat[]; reference: Reference }) {
  const [filter, setFilter] = useState<Filter>({ kind: 'fn', cls: 'S' })
  const [picked, setPicked] = useState<string | null>(null)
  const labelled = beats.filter(b => reference.labels[b.id])
  const s = score(labelled.map(b => ({ truth: reference.labels[b.id], predicted: b.prediction.predicted_class })))
  const shown = labelled.filter(b => {
    const t = reference.labels[b.id], p = b.prediction.predicted_class
    if (filter.kind === 'cell') return t === filter.truth && p === filter.predicted
    return filter.kind === 'fn' ? t === filter.cls && p !== filter.cls : p === filter.cls && t !== filter.cls
  })
  const beat = shown.find(b => b.id === picked)
  const title = filter.kind === 'cell' ? `Reference ${filter.truth}, model ${filter.predicted}` : filter.kind === 'fn' ? `False negatives: reference ${filter.cls}, model said otherwise` : `False positives: model ${filter.cls}, reference differs`
  const i = (c: string) => CLASS_CODES.indexOf(c as 'N')
  return <>
    <SplitNote record={record} />
    <p className="muted">This recording only: {s.n.toLocaleString()} windows with a reference label. One recording is one patient, so these scores say little about other patients.</p>
    <HeadlineMetrics macroF1={s.macroF1} sRecall={s.per[i('S')].recall} fRecall={s.per[i('F')].recall} accuracy={s.accuracy} baseline={s.baseline}
      sources={{ macroF1: 'recording#macro_f1', sRecall: 'recording#S.recall', fRecall: 'recording#F.recall', accuracy: 'recording#accuracy', baseline: 'recording#baseline' }} />
    <p className="small muted">Macro-F1 averages all four classes; a class absent from this recording counts as 0.</p>
    <div className="grid-2">
      <Card title="Errors by class">
        <Table head={['Class', 'Support', 'False negatives', 'False positives']} rows={s.per.map(p => [<ClassBadge key="c" code={p.code} />, p.support,
          <button type="button" key="fn" disabled={!p.fn} aria-pressed={filter.kind === 'fn' && filter.cls === p.code} onClick={() => setFilter({ kind: 'fn', cls: p.code })}>{p.fn} FN</button>,
          <button type="button" key="fp" disabled={!p.fp} aria-pressed={filter.kind === 'fp' && filter.cls === p.code} onClick={() => setFilter({ kind: 'fp', cls: p.code })}>{p.fp} FP</button>])} />
      </Card>
      <Card title="Confusion matrix (click a cell)"><ConfusionMatrix counts={s.matrix} source="recording#matrix" onSelect={(truth, predicted) => setFilter({ kind: 'cell', truth, predicted })} /></Card>
    </div>
    <Card title={`${title} (${shown.length})`}>
      {shown.length ? <WindowTable beats={shown} selectedId={picked} onSelect={b => setPicked(b.id)} /> : <Empty title="No windows in this group" />}
    </Card>
    {beat && <Detail key={beat.id} analysis={analysis} record={record} beat={beat} truth={reference.labels[beat.id]} />}
  </>
}

function Gate({ analysis, record }: { analysis: Analysis; record: RecordInfo }) {
  const beats = useBeats(analysis.id)
  const reference = useApi<Reference>(`/analyses/${analysis.id}/reference`)
  return <Load q={reference}>{r => !r.available
    ? <Empty title="No reference annotation">{r.message} Errors can only be identified for a recording uploaded with its .atr file.</Empty>
    : <Load q={beats}>{b => <Lab analysis={analysis} record={record} beats={b} reference={r} />}</Load>}</Load>
}

function HeldOut() {
  const q = useApi<ModelMetrics>('/model/metrics')
  return <Card title="For comparison: errors on the held-out test set"><Load q={q}>{m => !m.available ? <Empty title={m.message ?? 'Not available'} /> :
    <ConfusionMatrix counts={m.metrics.test.confusion_matrix.counts} source="metrics#test.confusion_matrix.counts" />}</Load></Card>
}

export default function ErrorsLab() {
  return <><PageHead title="Misclassification Lab" sub="Where the model disagrees with the reference annotation, window by window." />
    <WeaknessNote />
    <NeedsAnalysis>{(a, r) => <Gate key={a.id} analysis={a} record={r} />}</NeedsAnalysis>
    <HeldOut /></>
}
