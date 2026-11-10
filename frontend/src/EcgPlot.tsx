import { useEffect, useRef, useState } from 'react'
import { colors, effectiveClass, type Beat } from './shared'

const W = 1000, H = 300, L = 44, R = 12, T = 26, B = 24
const OH = 36 // overview strip height
const PAPER = '#FBF7EC', INK = '#1a1a1a'
export type Heat = { t: number; dur: number; values: number[] } // attribution window: start time, duration (s)
type Props = {
  times: number[]; amps: number[]; beats: Beat[]; selectedId?: string | null
  focus?: { t: number; k: number } | null
  heat?: Heat | null
  onSelect: (b: Beat) => void; onShift: (dir: -1 | 1) => void
}
const reduced = () => typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches

/** min/max per pixel column */
function decimate(times: number[], amps: number[], lo: number, hi: number, X: (t: number) => number, Y: (v: number) => number) {
  const cols = new Map<number, [number, number, number, number]>() // col -> [min,max,first,last]
  for (let i = lo; i <= hi; i += 1) {
    const c = Math.round(X(times[i]) * 2) / 2, v = amps[i]
    const e = cols.get(c)
    if (!e) cols.set(c, [v, v, v, v]); else { if (v < e[0]) e[0] = v; if (v > e[1]) e[1] = v; e[3] = v }
  }
  let d = ''
  cols.forEach(([mn, mx, f, l], c) => {
    const up = l >= f
    d += `${d ? 'L' : 'M'}${c} ${Y(up ? mn : mx).toFixed(1)}L${c} ${Y(up ? mx : mn).toFixed(1)}`
  })
  return d
}

export default function EcgPlot({ times, amps, beats, selectedId, focus, heat, onSelect, onShift }: Props) {
  const t0 = times[0] ?? 0
  const t1 = times[times.length - 1] ?? 1
  const [view, setView] = useState({ a: t0, b: Math.min(t1, t0 + 10) })
  const [hover, setHover] = useState<{ x: number; y: number } | null>(null)
  const svgRef = useRef<SVGSVGElement | null>(null)
  const ovRef = useRef<SVGSVGElement | null>(null)
  const drag = useRef<{ x: number; a: number; b: number; moved: boolean } | null>(null)
  const ovDrag = useRef<{ dx: number } | null>(null)
  const viewRef = useRef(view)
  viewRef.current = view
  const minSpan = 0.5
  const clampView = (a: number, b: number) => {
    const span = Math.min(Math.max(b - a, minSpan), Math.max(t1 - t0, minSpan))
    const na = Math.min(Math.max(a, t0), Math.max(t1 - span, t0))
    return { a: na, b: na + span }
  }

  useEffect(() => {
    setView(v => (v.b > v.a && v.a >= t0 - 1e-6 && v.b <= t1 + 1e-6) ? v : { a: t0, b: Math.min(t1, t0 + 10) })
  }, [t0, t1])
  useEffect(() => {
    if (!focus) return
    const span = viewRef.current.b - viewRef.current.a
    setView(clampView(focus.t - span / 2, focus.t + span / 2))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [focus?.k])
  useEffect(() => {
    const el = svgRef.current
    if (!el) return
    const onWheel = (event: WheelEvent) => {
      event.preventDefault()
      const rect = el.getBoundingClientRect()
      const frac = Math.min(1, Math.max(0, ((event.clientX - rect.left) / rect.width * W - L) / (W - L - R)))
      const { a, b } = viewRef.current
      const span = (b - a) * Math.pow(1.0015, event.deltaY)
      const anchor = a + frac * (b - a)
      setView(clampView(anchor - frac * span, anchor + (1 - frac) * span))
    }
    el.addEventListener('wheel', onWheel, { passive: false })
    return () => el.removeEventListener('wheel', onWheel)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [t0, t1])

  if (!times.length) return <div className="pf-plot-empty">No samples were returned from the backend.</div>

  const plotW = W - L - R, plotH = H - T - B
  const { a, b } = view
  const fs = times.length > 1 ? (times.length - 1) / (t1 - t0) : 360
  const lo = Math.max(0, Math.floor((a - t0) * fs)), hi = Math.min(times.length - 1, Math.ceil((b - t0) * fs))
  let mn = Infinity, mx = -Infinity
  for (let i = lo; i <= hi; i += 1) { const v = amps[i]; if (v < mn) mn = v; if (v > mx) mx = v }
  if (!Number.isFinite(mn)) { mn = 0; mx = 1 }
  const pad = (mx - mn || 1) * 0.15
  mn -= pad; mx += pad
  const X = (t: number) => L + (t - a) / (b - a) * plotW
  const Y = (v: number) => T + (1 - (v - mn) / (mx - mn)) * plotH
  const path = decimate(times, amps, lo, hi, X, Y)
  const ampAt = (t: number) => amps[Math.min(times.length - 1, Math.max(0, Math.round((t - t0) * fs)))]

  // grid: minor 0.04s / 0.1mV, major 0.2s / 0.5mV (amplitude units are assumed mV)
  const vLines: { t: number; major: boolean }[] = []
  const minorT = 0.04
  if ((b - a) / minorT <= 400) for (let k = Math.ceil(a / minorT); k * minorT <= b; k += 1) vLines.push({ t: k * minorT, major: k % 5 === 0 })
  else for (let k = Math.ceil(a / 0.2); k * 0.2 <= b; k += 1) vLines.push({ t: k * 0.2, major: true })
  const hLines: { v: number; major: boolean }[] = []
  const vStep = (mx - mn) / 0.1 <= 120 ? 0.1 : 0.5
  for (let k = Math.ceil(mn / vStep); k * vStep <= mx; k += 1) hLines.push({ v: k * vStep, major: Math.abs((k * vStep) / 0.5 - Math.round((k * vStep) / 0.5)) < 1e-6 })

  const steps = [0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]
  const step = steps.find(s => (b - a) / s <= 10) ?? 100
  const ticks: number[] = []
  for (let t = Math.ceil(a / step) * step; t <= b; t += step) ticks.push(t)

  const sorted = beats
  const visible = sorted.filter(x => x.timestamp_seconds >= a - 0.5 && x.timestamp_seconds <= b + 0.5)
  const inView = visible.filter(x => x.timestamp_seconds >= a && x.timestamp_seconds <= b)
  const labels = b - a <= 12
  const nearest = (t: number) => {
    let best: Beat | null = null, bd = Infinity
    for (const x of visible) { const d = Math.abs(x.timestamp_seconds - t); if (d < bd) { bd = d; best = x } }
    return best
  }
  const trans = reduced() ? undefined : 'transform 0.2s'

  const pan = (da: number) => setView(clampView(a + da, b + da))
  const zoom = (f: number) => { const c = (a + b) / 2, s = (b - a) * f; setView(clampView(c - s / 2, c + s / 2)) }

  // overview
  const OX = (t: number) => L + (t - t0) / ((t1 - t0) || 1) * plotW
  const ovStride = Math.max(1, Math.floor(times.length / plotW))
  let ovMn = Infinity, ovMx = -Infinity
  for (let i = 0; i < amps.length; i += ovStride) { if (amps[i] < ovMn) ovMn = amps[i]; if (amps[i] > ovMx) ovMx = amps[i] }
  const OY = (v: number) => 4 + (1 - (v - ovMn) / ((ovMx - ovMn) || 1)) * (OH - 12)
  const ovPath = decimate(times, amps, 0, times.length - 1, OX, OY)
  const ovT = (clientX: number) => {
    const r = ovRef.current!.getBoundingClientRect()
    return t0 + ((clientX - r.left) / r.width * W - L) / plotW * (t1 - t0)
  }

  const hv = hover && hover.x >= L && hover.x <= L + plotW ? hover : null
  const hT = hv ? a + (hv.x - L) / plotW * (b - a) : 0
  const hBeat = hv ? nearest(hT) : null
  const selBeat = selectedId ? beats.find(x => x.id === selectedId) : null
  const ptr = (e: React.PointerEvent | React.MouseEvent) => {
    const r = svgRef.current!.getBoundingClientRect()
    return { x: (e.clientX - r.left) / r.width * W, y: (e.clientY - r.top) / r.height * H }
  }
  const ttLeft = hv ? hv.x > W * 0.65 : false

  return <div className="pf-plot">
    <div className="pf-plot-bar">
      <span className="num">{a.toFixed(1)}–{b.toFixed(1)} s <em>· loaded {t0.toFixed(0)}–{t1.toFixed(0)} s</em></span>
      <span className="pf-plot-btns">
        <button type="button" onClick={() => onShift(-1)} aria-label="Load earlier window" title="Load earlier window">«</button>
        <button type="button" onClick={() => zoom(1.5)} aria-label="Zoom out" title="Zoom out">−</button>
        <button type="button" onClick={() => zoom(1 / 1.5)} aria-label="Zoom in" title="Zoom in">+</button>
        <button type="button" onClick={() => setView({ a: t0, b: t1 })} aria-label="Fit" title="Fit all">⤢</button>
        <button type="button" onClick={() => onShift(1)} aria-label="Load later window" title="Load later window">»</button>
      </span>
    </div>
    <svg ref={svgRef} viewBox={`0 0 ${W} ${H}`} className="pf-plot-svg" tabIndex={0} role="img"
      aria-label="ECG waveform on paper grid; abnormal beats shaded by class. Scroll to zoom, drag to pan, arrow keys to pan."
      onKeyDown={e => { if (e.key === 'ArrowLeft') pan(-(b - a) / 8); else if (e.key === 'ArrowRight') pan((b - a) / 8); else if (e.key === '+' || e.key === '=') zoom(1 / 1.3); else if (e.key === '-') zoom(1.3) }}
      onPointerDown={e => { drag.current = { x: e.clientX, a, b, moved: false } }}
      onPointerMove={e => {
        setHover(ptr(e))
        const d = drag.current
        if (!d || !svgRef.current) return
        const dx = e.clientX - d.x
        if (Math.abs(dx) > 3) d.moved = true
        if (!d.moved) return
        const dt = dx / svgRef.current.getBoundingClientRect().width * W / plotW * (d.b - d.a)
        setView(clampView(d.a - dt, d.b - dt))
      }}
      onPointerUp={e => {
        const moved = drag.current?.moved
        window.setTimeout(() => { drag.current = null }, 0)
        if (!moved && svgRef.current) {
          const p = ptr(e)
          if (p.x >= L && p.x <= L + plotW) {
            const t = a + (p.x - L) / plotW * (b - a), nb = nearest(t)
            if (nb && Math.abs(X(nb.timestamp_seconds) - p.x) < 24) onSelect(nb)
          }
        }
      }}
      onPointerLeave={() => { drag.current = null; setHover(null) }}>
      <defs>
        <clipPath id="pf-plot-clip"><rect x={L} y={T} width={plotW} height={plotH} /></clipPath>
        <linearGradient id="pf-heat-g" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stopColor="#2B3A1E" stopOpacity="0" /><stop offset="1" stopColor="#2B3A1E" stopOpacity="1" /></linearGradient>
      </defs>
      <rect x={L} y={T} width={plotW} height={plotH} fill={PAPER} />
      {vLines.map(g => <line key={g.t} x1={X(g.t)} x2={X(g.t)} y1={T} y2={T + plotH} stroke="#C0463A" strokeOpacity={g.major ? 0.22 : 0.08} strokeWidth={g.major ? 0.8 : 0.5} />)}
      {hLines.map(g => <line key={g.v} x1={L} x2={L + plotW} y1={Y(g.v)} y2={Y(g.v)} stroke="#C0463A" strokeOpacity={g.major ? 0.22 : 0.08} strokeWidth={g.major ? 0.8 : 0.5} />)}
      <g clipPath="url(#pf-plot-clip)">
        {heat && heat.values.length > 0 && (() => {
          const mxa = heat.values.reduce((m, v) => Math.max(m, Math.abs(v)), 1e-9), w = heat.dur / heat.values.length
          return heat.values.map((v, i) => {
            const k = Math.max(0, v) / mxa
            if (k < 0.02) return null
            const x = X(heat.t + i * w)
            return <rect key={i} x={x} y={T} width={Math.max(1, X(heat.t + (i + 1) * w) - x + 0.5)} height={plotH} fill={k > 0.6 ? '#2B3A1E' : '#B8892E'} opacity={0.08 + 0.4 * k} />
          })
        })()}
        {visible.map(beat => {
          const cls = effectiveClass(beat)
          if (cls === 'N' && beat.id !== selectedId) return null
          const sel = beat.id === selectedId, t = beat.timestamp_seconds
          return <rect key={beat.id} x={X(t - 0.25)} width={X(t + 0.25) - X(t - 0.25)} y={T} height={plotH} fill={colors[cls]} opacity={sel ? 0.34 : 0.18} stroke={sel ? colors[cls] : 'none'} strokeWidth={sel ? 1.5 : 0} style={{ transition: trans }} />
        })}
        <path d={path} fill="none" stroke={INK} strokeWidth="1.6" strokeLinejoin="round" strokeLinecap="round" />
        {/* RR intervals */}
        {visible.slice(1).map((bt, i) => {
          const p = visible[i], rr = bt.timestamp_seconds - p.timestamp_seconds
          if (!(rr > 0) || rr > 3) return null
          const xm = X((bt.timestamp_seconds + p.timestamp_seconds) / 2)
          if (X(bt.timestamp_seconds) - X(p.timestamp_seconds) < 54) return null
          return <g key={`rr${bt.id}`} className="pf-rr"><line x1={X(p.timestamp_seconds)} x2={X(bt.timestamp_seconds)} y1={T + 8} y2={T + 8} stroke="#777" strokeWidth=".6" />
            <text x={xm} y={T + 6} textAnchor="middle" fontSize="9" fill="#555">{rr.toFixed(2)}s · {Math.round(60 / rr)}</text></g>
        })}
        {hv && <line x1={hv.x} x2={hv.x} y1={T} y2={T + plotH} stroke="#1a1a1a" strokeOpacity=".4" strokeWidth=".8" />}
        {inView.map(beat => {
          const cls = effectiveClass(beat), sel = beat.id === selectedId
          const cx = X(beat.timestamp_seconds), cy = Y(ampAt(beat.timestamp_seconds))
          if (cls === 'N' && !sel) return <line key={beat.id} x1={cx} x2={cx} y1={cy + 6} y2={cy + 12} stroke={colors.N} strokeWidth="1.5" />
          const label = sel || labels ? `${cls} · ${Math.round(beat.prediction.confidence * 100)}%` : ''
          return <g key={beat.id} className="pf-peak">
            <line x1={cx} x2={cx} y1={cy + 6} y2={cy + 13} stroke={colors[cls]} strokeWidth={sel ? 2.5 : 1.5} />
            {label && <text x={cx} y={T + plotH - 6} textAnchor="middle" fontSize={sel ? 11 : 9.5} fontWeight={sel ? 700 : 500} fill={colors[cls]} stroke={PAPER} strokeWidth="3" paintOrder="stroke">{label}</text>}
          </g>
        })}
      </g>
      <rect x={L} y={T} width={plotW} height={plotH} fill="none" stroke="#C0463A" strokeOpacity=".35" strokeWidth=".8" />
      {ticks.map(t => <text key={t} x={X(t)} y={H - 8} textAnchor="middle" fontSize="10" fill="#555" style={{ fontVariantNumeric: 'tabular-nums' }}>{t.toFixed(step < 1 ? 1 : 0)}s</text>)}
      <text x={L - 6} y={T + 8} textAnchor="end" fontSize="9" fill="#666" style={{ fontVariantNumeric: 'tabular-nums' }}>{mx.toFixed(1)}</text>
      <text x={L - 6} y={T + plotH} textAnchor="end" fontSize="9" fill="#666" style={{ fontVariantNumeric: 'tabular-nums' }}>{mn.toFixed(1)}</text>
      <text x={L} y={12} fontSize="9" fill="#777">RR s · bpm</text>
      {hv && hBeat && (() => {
        const lines = [`t ${hT.toFixed(2)} s · ${ampAt(hT).toFixed(2)} mV`, `beat #${hBeat.beat_index + 1} · ${effectiveClass(hBeat)} · ${(hBeat.prediction.confidence * 100).toFixed(1)}%`]
        const w = 150, x = ttLeft ? hv.x - w - 10 : hv.x + 10, y = Math.min(Math.max(hv.y - 20, T + 4), T + plotH - 40)
        return <g pointerEvents="none"><rect x={x} y={y} width={w} height={34} fill="#1a1a1a" opacity=".92" />
          {lines.map((s, i) => <text key={i} x={x + 8} y={y + 14 + i * 13} fontSize="10" fill="#FAFAF6" style={{ fontVariantNumeric: 'tabular-nums' }}>{s}</text>)}</g>
      })()}
    </svg>
    <svg ref={ovRef} viewBox={`0 0 ${W} ${OH}`} className="pf-ov" role="slider" aria-label="Recording overview; drag the window to navigate"
      aria-valuemin={t0} aria-valuemax={t1} aria-valuenow={a}
      onPointerDown={e => {
        (e.currentTarget as Element).setPointerCapture(e.pointerId)
        const t = ovT(e.clientX), inside = t >= a && t <= b
        ovDrag.current = { dx: inside ? t - a : (b - a) / 2 }
        if (!inside) setView(clampView(t - (b - a) / 2, t + (b - a) / 2))
      }}
      onPointerMove={e => { if (ovDrag.current) { const na = ovT(e.clientX) - ovDrag.current.dx; setView(clampView(na, na + (b - a))) } }}
      onPointerUp={() => { ovDrag.current = null }}>
      <rect x={L} y={0} width={plotW} height={OH} fill={PAPER} />
      <path d={ovPath} fill="none" stroke="#555" strokeWidth=".8" />
      {beats.filter(x => effectiveClass(x) !== 'N').map(x => <line key={x.id} x1={OX(x.timestamp_seconds)} x2={OX(x.timestamp_seconds)} y1={OH - 8} y2={OH - 1} stroke={colors[effectiveClass(x)]} strokeWidth="1.5" />)}
      {selBeat && <line x1={OX(selBeat.timestamp_seconds)} x2={OX(selBeat.timestamp_seconds)} y1={0} y2={OH} stroke="#1a1a1a" strokeWidth="1" />}
      <rect x={OX(a)} y={0.5} width={Math.max(4, OX(b) - OX(a))} height={OH - 1} fill="#2B3A1E" fillOpacity=".12" stroke="#2B3A1E" strokeWidth="1" style={{ cursor: 'grab', transition: trans }} />
    </svg>
    <div className="pf-plot-legend">
      {(['N', 'S', 'V', 'F'] as const).map(c => <span key={c}><i style={{ background: colors[c] }} />{c}</span>)}
      {heat && <span className="pf-heatbar"><i />attribution low → high</span>}
      <span>Shaded band = non-normal beat · scroll to zoom · drag to pan · click a beat to explain</span>
    </div>
  </div>
}
