import { useEffect, useRef, useState } from 'react'
import { api, CLASS_NAMES, colors, effectiveClass, type Beat, type ClassCode } from './shared'

type Method = 'gradcam' | 'ig'
type Expl = { method: Method; class: string; attributions: number[]; metadata?: Record<string, unknown>; cached?: boolean }
const explCache = new Map<string, Expl>()
const ck = (a: string, b: string, c: string, m: Method) => `${a}:${b}:${c}:${m}`
const TIP = 'IG: deletion tests show the highlighted regions affect the prediction. Grad-CAM: low resolution; weaker in tests.'
const MAX_CANDIDATES = 12
const TOP_N = 8

function heat(t: number) {
  const k = Math.min(1, Math.max(0, t))
  const r = Math.round(138 + (43 - 138) * k), g = Math.round(154 + (58 - 154) * k), b = Math.round(91 + (30 - 91) * k)
  return { fill: `rgb(${r},${g},${b})`, opacity: 0.06 + 0.6 * k }
}

export default function ExplainDrawer({ analysisId, beat, beats, onClose, onJump, onHeat }: {
  analysisId: string; beat: Beat; beats: Beat[]; onClose: () => void; onJump: (b: Beat) => void; onHeat?: (h: { t: number; dur: number; values: number[] } | null) => void
}) {
  const cls = beat.prediction.predicted_class as ClassCode
  const [method, setMethod] = useState<Method>('ig')
  const [expl, setExpl] = useState<(Expl & { beatId: string }) | null>(null)
  const [loading, setLoading] = useState(false)
  const [signal, setSignal] = useState<number[]>([])
  const [err, setErr] = useState('')
  const [ranking, setRanking] = useState<{ beat: Beat; score: number }[]>([])
  const [rankBusy, setRankBusy] = useState(false)
  const closeRef = useRef<HTMLButtonElement | null>(null)
  const beatsRef = useRef(beats)
  beatsRef.current = beats

  useEffect(() => {
    closeRef.current?.focus()
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  useEffect(() => {
    let off = false
    setSignal([]); setErr('')
    api.get(`/analyses/${analysisId}/signal`, { params: { start_sample: Math.max(0, beat.sample_index - 360), count: 720, filtered: true } })
      .then(r => { if (!off) setSignal(r.data.amplitude) })
      .catch(e => { if (!off) setErr(e.response?.data?.detail || e.message) })
    return () => { off = true }
  }, [analysisId, beat.id, beat.sample_index])

  useEffect(() => {
    let off = false
    const key = ck(analysisId, beat.id, cls, method)
    const hit = explCache.get(key)
    if (hit) { setExpl({ ...hit, beatId: beat.id }); setLoading(false); setErr(''); return }
    setLoading(true)
    // keep the previous heatmap visible while loading, but only for the same beat
    setExpl(prev => (prev && prev.beatId === beat.id ? prev : null))
    api.get(`/analyses/${analysisId}/explanations`, { params: { beat: beat.id, class: cls, method } })
      .then(r => {
        if (off) return
        if (r.data.method !== method || r.data.class !== cls) { setErr('The API returned an explanation for a different method or class; not displayed.'); setLoading(false); return }
        explCache.set(key, r.data)
        setExpl({ ...r.data, beatId: beat.id }); setLoading(false)
      })
      .catch(e => { if (!off) { setErr(e.response?.data?.detail || e.message); setLoading(false) } })
    return () => { off = true }
  }, [analysisId, beat.id, cls, method])

  useEffect(() => {
    let off = false
    const ci = ['N', 'S', 'V', 'F'].indexOf(cls)
    const candidates = beatsRef.current.filter(b => b.prediction.predicted_class === cls)
      .sort((x, y) => y.prediction.probabilities[ci] - x.prediction.probabilities[ci]).slice(0, MAX_CANDIDATES)
    const scores = new Map<string, number>()
    const publish = () => {
      if (off) return
      setRanking(candidates.filter(b => scores.has(b.id)).map(b => ({ beat: b, score: scores.get(b.id)! }))
        .sort((x, y) => y.score - x.score).slice(0, TOP_N))
    }
    setRanking([]); setRankBusy(true)
    let next = 0
    const worker = async () => {
      while (!off && next < candidates.length) {
        const b = candidates[next++]
        const key = ck(analysisId, b.id, cls, method)
        try {
          let e = explCache.get(key)
          if (!e) {
            const r = await api.get(`/analyses/${analysisId}/explanations`, { params: { beat: b.id, class: cls, method } })
            e = r.data as Expl
            explCache.set(key, e)
          }
          const attrs = e.attributions
          scores.set(b.id, attrs.reduce((s, v) => s + (method === 'ig' ? Math.abs(v) : Math.max(0, v)), 0) / (attrs.length || 1))
          publish()
        } catch { /* beats near a recording edge cannot be explained; skip */ }
      }
    }
    Promise.all([worker(), worker(), worker()]).then(() => { if (!off) setRankBusy(false) })
    return () => { off = true }
  }, [analysisId, cls, method])

  const attrs = expl?.attributions ?? []
  useEffect(() => {
    onHeat?.(attrs.length ? { t: beat.timestamp_seconds - 1, dur: 2, values: attrs } : null)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [expl, beat.id])
  useEffect(() => () => onHeat?.(null), [])
  const maxAbs = attrs.reduce((m, v) => Math.max(m, Math.abs(v)), 1e-9)
  const mn = signal.length ? Math.min(...signal) : 0
  const mx = signal.length ? Math.max(...signal) : 1
  const CW = 720, CH = 240
  const sy = (v: number) => 10 + (1 - (v - mn) / (mx - mn || 1)) * (CH - 20)
  const trace = signal.map((v, i) => `${i},${sy(v).toFixed(1)}`).join(' ')
  const G = 6
  const bands = attrs.length ? Array.from({ length: Math.ceil(attrs.length / G) }, (_, i) => {
    const seg = attrs.slice(i * G, i * G + G)
    const v = seg.reduce((s, x) => s + Math.abs(x), 0) / seg.length / maxAbs
    return { x: i * G, ...heat(v) }
  }) : []
  const allZero = method === 'gradcam' && attrs.length > 0 && attrs.every(v => Math.abs(v) < 1e-6)
  const eff = effectiveClass(beat)
  const maxScore = ranking.reduce((m, r) => Math.max(m, r.score), 1e-9)
  const meta = expl?.metadata ?? {}

  return <>
    <div className="pf-backdrop" onClick={onClose} />
    <aside className="pf-drawer" role="dialog" aria-label="Explain selected beat">
      <header>
        <span className="pf-label">EXPLAIN</span>
        <button ref={closeRef} type="button" onClick={onClose} aria-label="Close explain drawer">CLOSE ✕</button>
      </header>
      <h2>Beat {beat.beat_index + 1} · {beat.timestamp_seconds.toFixed(2)} s</h2>
      <div className="pf-drawer-facts">
        <span><i>MODEL</i><b style={{ color: colors[cls] }}>{CLASS_NAMES[cls]}</b> · {(beat.prediction.confidence * 100).toFixed(1)}%</span>
        <span><i>REVIEWED</i>{beat.review?.reviewed_label ? <b style={{ color: colors[eff] }}>{CLASS_NAMES[eff as ClassCode]}</b> : <b>—</b>}</span>
      </div>
      <div className="pf-seg" role="group" aria-label="Attribution method">
        <button type="button" aria-pressed={method === 'ig'} className={method === 'ig' ? 'on' : ''} onClick={() => setMethod('ig')}>Integrated Gradients</button>
        <button type="button" aria-pressed={method === 'gradcam'} className={method === 'gradcam' ? 'on' : ''} onClick={() => setMethod('gradcam')}>Grad-CAM (coarse)</button>
        <span className="pf-meta" title={TIP} aria-label={TIP} style={{ marginLeft: 8, alignSelf: 'center' }}>ⓘ</span>
      </div>
      <p className="pf-meta">{TIP}</p>
      {err && <div className="error" role="alert">{err}</div>}
      <div className="pf-heat">
        {!signal.length || !expl ? <div className="pf-plot-empty">Loading attribution…</div> : <>
          {loading && <div className="pf-meta" role="status">Computing {method === 'ig' ? 'Integrated Gradients' : 'Grad-CAM'}… showing previous heatmap</div>}
          <svg viewBox={`0 0 ${CW} ${CH}`} role="img" aria-label="Beat trace with attribution heatmap behind it">
            {bands.map(bd => <rect key={bd.x} x={bd.x} y={0} width={G} height={CH} fill={bd.fill} fillOpacity={bd.opacity} />)}
            {[1, 2, 3].map(i => <line key={i} x1={0} x2={CW} y1={CH * i / 4} y2={CH * i / 4} stroke="#999" strokeOpacity=".5" strokeDasharray="1 4" />)}
            <line x1={360} x2={360} y1={0} y2={CH} stroke="#111" strokeDasharray="4 3" strokeOpacity=".6" />
            <polyline points={trace} fill="none" stroke="#111" strokeWidth="1.6" strokeLinejoin="round" />
            <text x={364} y={12} fontSize="10" fill="#111">R-PEAK</text>
          </svg></>}
      </div>
      <div className="pf-heat-legend"><span>LOW</span><i /><span>HIGH attribution</span><small>720 samples · 360 Hz{method === 'ig' ? ' · |attribution| shown' : ''}</small></div>
      {allZero && <div className="warning">No positive evidence for this class in this window (ReLU Grad-CAM).</div>}
      {expl && <p className="pf-meta">{method === 'gradcam'
        ? `Layer ${String(meta.target_layer ?? meta.layer ?? 'conv3')} · ${expl.cached ? 'cached' : 'computed'}`
        : `Baseline ${String(meta.baseline ?? 'zero')} · completeness error ${typeof meta.completeness_error === 'number' ? meta.completeness_error.toFixed(5) : '—'} · ${expl.cached ? 'cached' : 'computed'}`}</p>}

      <h3>Beats driving this decision</h3>
      <p className="pf-meta">Class {cls} · top {TOP_N} of the {MAX_CANDIDATES} highest-confidence {CLASS_NAMES[cls].toLowerCase()} beats, ranked by {method === 'ig' ? 'Integrated Gradients' : 'Grad-CAM'} attribution mass{rankBusy ? ' · computing…' : ''}.</p>
      <ol className="pf-rank">
        {ranking.map((r, i) => <li key={r.beat.id}>
          <button type="button" onClick={() => onJump(r.beat)} className={r.beat.id === beat.id ? 'on' : ''}>
            <span>{i + 1}</span><b>Beat {r.beat.beat_index + 1}</b><em>{r.beat.timestamp_seconds.toFixed(2)} s</em>
            <span className="pf-bar"><i style={{ width: `${r.score / maxScore * 100}%` }} /></span>
          </button>
        </li>)}
      </ol>
      {!ranking.length && !rankBusy && <p className="pf-meta">No rankable beats for this class.</p>}
      <p className="pf-meta">Experimental attribution; not clinical evidence.</p>
    </aside>
  </>
}
