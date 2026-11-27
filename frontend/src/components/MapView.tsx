import { explanationUrl, type Explanation } from '../api/explanations'
import type { Beat } from '../api/types'
import { useApi } from '../api/useApi'
import { when } from './format'
import { Load, Skeleton } from './states'
import { KeyValues, Table } from './ui'
import { AttributionPlot, NEGATIVE, POSITIVE } from './waveform/AttributionPlot'

const NAMES = { gradcam: 'Grad-CAM', ig: 'Integrated Gradients' }

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

export function MapView({ analysisId, beat, cls, method, signal, absolute = false }: { analysisId: string; beat: Beat; cls: string; method: 'gradcam' | 'ig'; signal: number[]; absolute?: boolean }) {
  const q = useApi<Explanation>(explanationUrl(analysisId, beat.id, cls, method))
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
