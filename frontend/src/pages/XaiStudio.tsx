import { useState } from 'react'
import type { Analysis, Beat, RecordInfo } from '../api/types'
import { useApi } from '../api/useApi'
import { when } from '../components/format'
import { Empty, Load, Skeleton } from '../components/states'
import { Card, ClassBadge, KeyValues, PageHead, Table } from '../components/ui'
import { WindowSelect } from '../components/WindowSelect'
import type { WindowData } from '../components/WindowStages'
import { WithWindow } from '../components/WithWindow'
import { AttributionPlot, NEGATIVE, POSITIVE } from '../components/waveform/AttributionPlot'
import { XaiEvidence } from '../components/XaiEvidence'
import { CLASS_CODES, XAI_NOTICE } from '../theme/theme'

type Explanation = {
  id: string; method: 'gradcam' | 'ig'; class: string; attributions: number[]; cached: boolean; model_version: string; created_at: string
  predicted_class: string; reference_label: string | null
  metadata: { steps?: number; baseline?: string; score_difference?: number; attribution_sum?: number; completeness_error?: number; gradient_l2_norm?: number; target_layer?: string }
  top_regions: { start_sample: number; end_sample: number; start_ms: number; end_ms: number; share_of_total_absolute: number; signed_share: number; contains_centre_beat: boolean }[]
}
const NAMES = { gradcam: 'Grad-CAM', ig: 'Integrated Gradients' }
const url = (a: string, beat: string, cls: string, method: string) => `/analyses/${a}/explanations?beat=${beat}&class=${cls}&method=${method}`

function Completeness({ e }: { e: Explanation }) {
  const m = e.metadata
  if (m.completeness_error === undefined || m.score_difference === undefined || m.attribution_sum === undefined) return <p className="muted">Completeness applies to Integrated Gradients only.</p>
  const relative = Math.abs(m.score_difference) > 1e-9 ? Math.abs(m.completeness_error / m.score_difference) : null
  return <>
    <h3>Completeness for this window</h3>
    <KeyValues rows={[['Logit change from baseline', <span className="num">{m.score_difference.toFixed(4)}</span>], ['Sum of attributions', <span className="num">{m.attribution_sum.toFixed(4)}</span>],
      ['Absolute error', <span className="num">{Math.abs(m.completeness_error).toFixed(4)}</span>], ['Relative error', <span className="num">{relative === null ? 'undefined' : `${(relative * 100).toFixed(2)}%`}</span>],
      ['Steps', <span className="num">{m.steps}</span>], ['Baseline', m.baseline ?? '—']]} />
    {relative !== null && relative > 0.05 && <p className="state state-warn">The attributions do not add up to the score change within 5% for this window. Read this map with extra caution.</p>}
  </>
}

function MapView({ analysisId, beat, cls, method, signal, absolute = false }: { analysisId: string; beat: Beat; cls: string; method: 'gradcam' | 'ig'; signal: number[]; absolute?: boolean }) {
  const q = useApi<Explanation>(url(analysisId, beat.id, cls, method))
  if (q.loading && !q.data) return <><p className="muted">Computing {NAMES[method]}…</p><Skeleton lines={5} label={`Computing ${NAMES[method]}`} /></>
  return <Load q={q}>{e => {
    const signed = method === 'ig' && !absolute
    return <div className="stack">
      <AttributionPlot signal={signal} values={absolute ? e.attributions.map(Math.abs) : e.attributions} signed={signed} regions={e.top_regions}
        label={`${NAMES[method]} for class ${cls}${method === 'ig' ? (signed ? ' · signed' : ' · absolute') : ' · 0 to 1 per window'}`} />
      {method === 'ig' && <div className="legend"><span><i style={{ background: POSITIVE }} />positive: raises the {cls} logit</span>{signed && <span><i style={{ background: NEGATIVE }} />negative: lowers it</span>}</div>}
      <Table caption="Top regions within this window (offsets from the centre beat). Shares are of total absolute attribution." head={['Region', 'From (ms)', 'To (ms)', 'Share', 'Signed share', 'Includes centre beat']}
        rows={e.top_regions.map((r, i) => [`R${i + 1}`, r.start_ms.toFixed(0), r.end_ms.toFixed(0), `${(r.share_of_total_absolute * 100).toFixed(1)}%`, `${(r.signed_share * 100).toFixed(1)}%`, r.contains_centre_beat ? 'yes' : 'no'])} />
      {method === 'ig' ? <Completeness e={e} /> : <p className="small muted">Target layer {e.metadata.target_layer} · gradient norm <span className="num">{e.metadata.gradient_l2_norm?.toPrecision(3)}</span> · scaled to its own maximum, so heights cannot be compared between windows.</p>}
      <p className="small muted">Model {e.model_version} · computed {when(e.created_at)}{e.cached ? ' · stored result' : ''}</p>
    </div>
  }}</Load>
}

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
