import { useState } from 'react'
import type { Analysis, Beat, RecordInfo } from '../api/types'
import { useApi } from '../api/useApi'
import { MapView } from '../components/MapView'
import { explanationUrl as url, type Explanation } from '../api/explanations'
import { Empty, Load, Skeleton } from '../components/states'
import { Card, ClassBadge, KeyValues, PageHead } from '../components/ui'
import { WindowSelect } from '../components/WindowSelect'
import type { WindowData } from '../components/WindowStages'
import { WithWindow } from '../components/WithWindow'
import { AttributionPlot } from '../components/waveform/AttributionPlot'
import { XaiEvidence } from '../components/XaiEvidence'
import { CLASS_CODES, XAI_NOTICE } from '../theme/theme'

function Agreement({ analysisId, beat, cls }: { analysisId: string; beat: Beat; cls: string }) {
  const q = useApi<{ spearman_gradcam_vs_abs_ig: number | null }>(`/analyses/${analysisId}/explanations/compare?beat=${beat.id}&class=${cls}`)
  return <Load q={q}>{d => <p>Rank correlation between Grad-CAM and absolute IG on this window:{' '}
    <strong className="num">{d.spearman_gradcam_vs_abs_ig === null ? 'undefined (a map is constant)' : d.spearman_gradcam_vs_abs_ig.toFixed(3)}</strong>. Agreement between methods does not show that either is right.</p>}</Load>
}

/** Both maps must be stored before their agreement can be read, so compare mounts only after both have loaded. */
function Compare({ analysisId, beat, cls, signal }: { analysisId: string; beat: Beat; cls: string; signal: number[] }) {
  const cam = useApi<Explanation>(url(analysisId, beat.id, cls, 'gradcam'))
  const ig = useApi<Explanation>(url(analysisId, beat.id, cls, 'ig'))
  if (cam.error || ig.error) return <Load q={cam.error ? cam : ig}>{() => null}</Load>
  if (!cam.data || !ig.data) return <Skeleton lines={6} label="Computing both maps" />
  return <div className="stack">
    <AttributionPlot signal={signal} values={cam.data.attributions} signed={false} label={`Grad-CAM for class ${cls}`} />
    <AttributionPlot signal={signal} values={ig.data.attributions} signed label={`Signed Integrated Gradients for class ${cls}`} />
    <Agreement analysisId={analysisId} beat={beat} cls={cls} />
  </div>
}

function Studio({ analysis, record, beat }: { analysis: Analysis; record: RecordInfo; beat: Beat }) {
  const [tab, setTab] = useState<'ig' | 'gradcam' | 'compare'>('ig')
  const [absolute, setAbsolute] = useState(false)
  const [chosen, setChosen] = useState<{ beat: string; cls: string } | null>(null)
  const cls = chosen?.beat === beat.id ? chosen.cls : beat.prediction.predicted_class
  const window_ = useApi<WindowData>(`/recordings/${record.id}/window?sample=${beat.sample_index}`)
  const reference = useApi<{ available: boolean; message: string | null; labels: Record<string, string> }>(`/analyses/${analysis.id}/reference`)
  const truth = reference.data?.labels[beat.id]
  return <>
    <Card>
      <KeyValues rows={[
        ['Model class', <ClassBadge code={beat.prediction.predicted_class} long />],
        ['Softmax score', <span className="num">{(beat.prediction.probabilities[CLASS_CODES.indexOf(beat.prediction.predicted_class as 'N')] * 100).toFixed(2)}%</span>],
        ['Reference label', truth ? <><ClassBadge code={truth} long /> {truth !== beat.prediction.predicted_class && <strong>differs from the model</strong>}</> : reference.loading ? 'loading' : 'No reference annotation'],
        ['Explaining class', <select aria-label="Class to explain" value={cls} onChange={e => setChosen({ beat: beat.id, cls: e.target.value })}>{CLASS_CODES.map(c => <option key={c}>{c}</option>)}</select>],
      ]} />
      <p className="notice">{XAI_NOTICE} A 2-second window usually holds neighbouring beats, so a highlight can fall on a beat other than the one being classified.</p>
    </Card>
    <Card title="Attribution" aside={<div className="row" role="tablist" aria-label="Method">
      {([['ig', 'Integrated Gradients'], ['gradcam', 'Grad-CAM'], ['compare', 'Side by side']] as const).map(([k, l]) => <button type="button" role="tab" aria-selected={tab === k} aria-pressed={tab === k} key={k} onClick={() => setTab(k)}>{l}</button>)}</div>}>
      {tab === 'ig' && <label className="row"><input type="checkbox" checked={absolute} onChange={e => setAbsolute(e.target.checked)} /> Show absolute values (hides direction)</label>}
      <Load q={window_}>{w => tab === 'compare'
        ? <Compare analysisId={analysis.id} beat={beat} cls={cls} signal={w.model_input} />
        : w.model_input ? <MapView key={`${beat.id}${cls}${tab}`} analysisId={analysis.id} beat={beat} cls={cls} method={tab} signal={w.model_input} absolute={tab === 'ig' && absolute} /> : <Empty title="No window" />}</Load>
    </Card>
  </>
}

export default function XaiStudio() {
  return <><PageHead title="XAI Studio" sub="What in the input moved the model’s score for one window." />
    <WithWindow>{({ analysis, record, beats, beat }) => <><WindowSelect beats={beats} beat={beat} /><Studio analysis={analysis} record={record} beat={beat} /></>}</WithWindow>
    <XaiEvidence /></>
}
