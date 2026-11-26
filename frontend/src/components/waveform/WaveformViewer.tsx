import { useEffect, useRef } from 'react'
import { useApi } from '../../api/useApi'
import type { Beat, RecordInfo } from '../../api/types'
import { classFill, classText, CLASS_CODES, CLASSES } from '../../theme/theme'
import { ErrorState, Skeleton } from '../states'
import { decimatedPath, extent } from './path'
import { clampView, useChunk, type View } from './useChunk'

const W = 1000, H = 320, L = 52, R = 10, T = 22, B = 22
const HALF = 360 // model window is 720 samples, centred on the beat

function Navigator({ record, beats, view, onView, selected }: { record: RecordInfo; beats: Beat[]; view: View; onView: (v: View) => void; selected: Beat | null }) {
  const q = useApi<{ min: number[]; max: number[] }>(`/recordings/${record.id}/envelope`)
  const dur = record.duration_seconds
  const X = (t: number) => L + t / dur * (W - L - R)
  let path = ''
  if (q.data) {
    const [lo] = extent(q.data.min), [, hi] = extent(q.data.max)
    const n = q.data.min.length
    const Y = (v: number) => 4 + (1 - (v - lo) / (hi - lo)) * 32
    path = q.data.min.map((v, i) => `M${(L + i / n * (W - L - R)).toFixed(1)} ${Y(v).toFixed(1)}V${Y(q.data!.max[i]).toFixed(1)}`).join('')
  }
  const move = (clientX: number, el: Element) => {
    const r = el.getBoundingClientRect()
    const t = ((clientX - r.left) / r.width * W - L) / (W - L - R) * dur
    onView(clampView(t - view.span / 2, view.span, dur))
  }
  return <svg className="plot" viewBox={`0 0 ${W} 48`} role="slider" tabIndex={0} aria-label="Whole-recording navigator"
    aria-valuemin={0} aria-valuemax={Math.round(dur)} aria-valuenow={Math.round(view.a)}
    onKeyDown={e => { if (e.key === 'ArrowLeft') onView(clampView(view.a - view.span, view.span, dur)); if (e.key === 'ArrowRight') onView(clampView(view.a + view.span, view.span, dur)) }}
    onPointerDown={e => { e.currentTarget.setPointerCapture(e.pointerId); move(e.clientX, e.currentTarget) }}
    onPointerMove={e => { if (e.buttons === 1) move(e.clientX, e.currentTarget) }}>
    <path d={path} stroke="#666" strokeWidth="1" fill="none" />
    {beats.filter(b => b.prediction.predicted_class !== 'N').map(b => <rect key={b.id} x={X(b.timestamp_seconds)} y={40} width={1.4} height={7} fill={classFill(b.prediction.predicted_class)} />)}
    {selected && <line x1={X(selected.timestamp_seconds)} x2={X(selected.timestamp_seconds)} y1={0} y2={48} stroke="#000" />}
    <rect x={X(view.a)} y={1} width={Math.max(4, X(view.a + view.span) - X(view.a))} height={46} fill="var(--green)" fillOpacity=".14" stroke="var(--green)" />
  </svg>
}

export function WaveformViewer({ record, beats, filtered, view, onView, selected, onSelect, cursor }: {
  record: RecordInfo; beats: Beat[]; filtered: boolean; view: View; onView: (v: View) => void
  selected: Beat | null; onSelect: (b: Beat) => void; cursor?: number | null
}) {
  const fs = record.sampling_rate, dur = record.duration_seconds
  const { chunk, error, loading, retry } = useChunk(record.id, fs, view.a, filtered)
  const svg = useRef<SVGSVGElement>(null)
  const drag = useRef<{ x: number; a: number; moved: boolean } | null>(null)
  const latest = useRef({ view, onView })
  useEffect(() => { latest.current = { view, onView } })
  useEffect(() => {
    const el = svg.current
    if (!el) return
    const wheel = (e: WheelEvent) => {
      e.preventDefault()
      const { view: v, onView: set } = latest.current
      const r = el.getBoundingClientRect()
      const frac = Math.min(1, Math.max(0, ((e.clientX - r.left) / r.width * W - L) / (W - L - R)))
      const next = clampView(v.a, v.span * Math.pow(1.0015, e.deltaY), dur)
      set(clampView(v.a + frac * (v.span - next.span), next.span, dur))
    }
    el.addEventListener('wheel', wheel, { passive: false })
    return () => el.removeEventListener('wheel', wheel)
  }, [dur, chunk])

  if (!chunk) return error ? <ErrorState error={error} onRetry={retry} /> : <Skeleton lines={8} label="Loading signal" />

  const unit = record.units ?? 'signal units'
  const first = chunk.start_sample
  const i0 = Math.max(0, Math.floor(view.a * fs) - first)
  const i1 = Math.min(chunk.amplitude.length, Math.ceil((view.a + view.span) * fs) - first + 1)
  const values = chunk.amplitude.slice(i0, i1)
  const [lo, hi] = extent(values)
  const pad = (hi - lo) * 0.14
  const X = (t: number) => L + (t - view.a) / view.span * (W - L - R)
  const Y = (v: number) => T + (1 - (v - lo + pad) / (hi - lo + 2 * pad)) * (H - T - B)
  const amp = (sample: number) => chunk.amplitude[sample - first]
  const visible = beats.filter(b => b.timestamp_seconds >= view.a && b.timestamp_seconds <= view.a + view.span)
  const step = [0.2, 0.5, 1, 2, 5].find(s => view.span / s <= 11) ?? 5
  const ticks: number[] = []
  for (let t = Math.ceil(view.a / step) * step; t <= view.a + view.span; t += step) ticks.push(t)
  const grid: number[] = []
  for (let t = Math.ceil(view.a / 0.2) * 0.2; t <= view.a + view.span && grid.length < 120; t += 0.2) grid.push(t)
  const at = (clientX: number) => {
    const r = svg.current!.getBoundingClientRect()
    return view.a + ((clientX - r.left) / r.width * W - L) / (W - L - R) * view.span
  }
  const pan = (d: number) => onView(clampView(view.a + d, view.span, dur))
  const zoom = (f: number) => onView(clampView(view.a + view.span * (1 - f) / 2, view.span * f, dur))
  const step_ = (d: number) => {
    if (!beats.length) return
    const i = selected ? selected.beat_index + d : 0
    const b = beats[Math.min(beats.length - 1, Math.max(0, i))]
    onSelect(b)
  }
  const cursorSample = cursor == null ? null : Math.round(cursor * fs)

  return <div>
    <div className="plot-bar">
      <span className="num">{view.a.toFixed(2)}–{(view.a + view.span).toFixed(2)} s · {unit} · {filtered ? 'Bandpass filtered' : 'Raw'}{loading ? ' · loading…' : ''}</span>
      <span className="row">
        <button type="button" onClick={() => pan(-view.span * 0.8)}>◀ Pan</button>
        <button type="button" onClick={() => zoom(1.5)} aria-label="Zoom out">−</button>
        <button type="button" onClick={() => zoom(1 / 1.5)} aria-label="Zoom in">+</button>
        <button type="button" onClick={() => pan(view.span * 0.8)}>Pan ▶</button>
        <button type="button" onClick={() => step_(-1)}>Previous window</button>
        <button type="button" onClick={() => step_(1)}>Next window</button>
      </span>
    </div>
    {error && <ErrorState title="The next signal segment failed to load" error={error} onRetry={retry} />}
    <svg ref={svg} className="plot" viewBox={`0 0 ${W} ${H}`} tabIndex={0} role="img"
      aria-label="ECG waveform. Arrow keys pan, plus and minus zoom, N and P move between windows."
      onKeyDown={e => {
        if (e.key === 'ArrowLeft') pan(-view.span / 8); else if (e.key === 'ArrowRight') pan(view.span / 8)
        else if (e.key === '+' || e.key === '=') zoom(1 / 1.3); else if (e.key === '-') zoom(1.3)
        else if (e.key === 'n') step_(1); else if (e.key === 'p') step_(-1)
      }}
      onPointerDown={e => { drag.current = { x: e.clientX, a: view.a, moved: false } }}
      onPointerMove={e => {
        const d = drag.current
        if (!d || e.buttons !== 1) return
        if (Math.abs(e.clientX - d.x) > 3) d.moved = true
        if (d.moved) onView(clampView(d.a - (e.clientX - d.x) / svg.current!.getBoundingClientRect().width * W / (W - L - R) * view.span, view.span, dur))
      }}
      onPointerUp={e => {
        const d = drag.current
        drag.current = null
        if (!d || d.moved || !visible.length) return
        const t = at(e.clientX)
        onSelect(visible.reduce((best, b) => Math.abs(b.timestamp_seconds - t) < Math.abs(best.timestamp_seconds - t) ? b : best))
      }}>
      <defs><clipPath id="wave-clip"><rect x={L} y={T} width={W - L - R} height={H - T - B} /></clipPath></defs>
      {grid.map(t => <line key={t} x1={X(t)} x2={X(t)} y1={T} y2={H - B} stroke="var(--grid)" />)}
      {[0, 0.25, 0.5, 0.75, 1].map(f => <line key={f} x1={L} x2={W - R} y1={T + f * (H - T - B)} y2={T + f * (H - T - B)} stroke="var(--grid)" />)}
      <g clipPath="url(#wave-clip)">
        {selected && <rect x={X((selected.sample_index - HALF) / fs)} width={(2 * HALF / fs) / view.span * (W - L - R)} y={T} height={H - T - B}
          fill={classFill(selected.prediction.predicted_class)} opacity=".16" stroke={classFill(selected.prediction.predicted_class)} />}
        <path d={decimatedPath(values.length, i => X((first + i0 + i) / fs), i => Y(values[i]))} fill="none" stroke="var(--trace)" strokeWidth="1.4" />
        {visible.map(b => {
          const c = b.prediction.predicted_class, x = X(b.timestamp_seconds), v = amp(b.sample_index)
          const sel = selected?.id === b.id
          const label = sel || view.span <= 8 || (c !== 'N' && view.span <= 14)
          return <g key={b.id}>
            <line x1={x} x2={x} y1={v === undefined ? H - B - 14 : Y(v) - 6} y2={v === undefined ? H - B - 4 : Y(v) - 14} stroke={classFill(c)} strokeWidth={sel ? 3 : 2} />
            {label && <text x={x} y={H - B - 6} textAnchor="middle" fontSize={sel ? 12 : 10} fontWeight={sel ? 700 : 500} fill={classText(c)} stroke="var(--paper)" strokeWidth="3" paintOrder="stroke">{c} {Math.floor(b.prediction.confidence * 100)}%</text>}
            {b.review?.status === 'reviewed' && <text x={x} y={T + 12} textAnchor="middle" fontSize="11" fill={classText(b.review.reviewed_label ?? '')}>◆{b.review.reviewed_label}</text>}
          </g>
        })}
        {cursor != null && <line x1={X(cursor)} x2={X(cursor)} y1={T} y2={H - B} stroke="var(--green)" strokeWidth="2" data-testid="cursor" />}
      </g>
      {ticks.map(t => <text key={t} x={X(t)} y={H - 6} textAnchor="middle" fontSize="10" fill="#555">{t.toFixed(step < 1 ? 1 : 0)}s</text>)}
      <text x={L - 4} y={T + 8} textAnchor="end" fontSize="10" fill="#555">{hi.toFixed(2)}</text>
      <text x={L - 4} y={H - B} textAnchor="end" fontSize="10" fill="#555">{lo.toFixed(2)}</text>
      {cursorSample != null && amp(cursorSample) !== undefined && <text x={W - R} y={14} textAnchor="end" fontSize="11" fill="#333">t {cursor!.toFixed(2)} s · {amp(cursorSample).toFixed(3)} {unit}</text>}
    </svg>
    <Navigator record={record} beats={beats} view={view} onView={onView} selected={selected} />
    <div className="legend">
      {CLASS_CODES.map(c => <span key={c}><i style={{ background: CLASSES[c].fill }} />{c} {CLASSES[c].name}</span>)}
      <span>Marker and % = model class and softmax score</span><span>◆ = reviewer label</span><span>Shaded = the 2 s model window</span>
    </div>
  </div>
}
