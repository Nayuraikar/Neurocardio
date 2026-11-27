import { useId } from 'react'
import { decimatedPath, extent } from './path'

/** A static waveform panel with labelled axes. `band` shades a sample range. */
export function Trace({ values, fs, label, unit, height = 150, startSeconds = 0, band, cursor }: {
  values: number[]; fs: number; label: string; unit: string; height?: number; startSeconds?: number
  band?: { from: number; to: number; color: string } | null; cursor?: number | null
}) {
  const id = useId()
  const W = 1000, L = 52, R = 10, T = 18, B = 18
  const [lo, hi] = extent(values)
  const pad = (hi - lo) * 0.08
  const X = (i: number) => L + i / Math.max(1, values.length - 1) * (W - L - R)
  const Y = (v: number) => T + (1 - (v - (lo - pad)) / (hi - lo + 2 * pad)) * (height - T - B)
  const seconds = values.length / fs
  return <svg className="plot" viewBox={`0 0 ${W} ${height}`} role="img" aria-labelledby={id}>
    <title id={id}>{`${label}, ${seconds.toFixed(1)} seconds, ${unit}`}</title>
    {band && <rect x={X(band.from)} width={Math.max(2, X(band.to) - X(band.from))} y={T} height={height - T - B} fill={band.color} opacity=".18" />}
    <path d={decimatedPath(values.length, X, i => Y(values[i]))} fill="none" stroke="var(--trace)" strokeWidth="1.3" />
    {cursor != null && cursor >= 0 && cursor < values.length && <line x1={X(cursor)} x2={X(cursor)} y1={T} y2={height - B} stroke="var(--green)" strokeWidth="1.5" />}
    <text x={L} y={12} fontSize="11" fill="#444">{label} · {unit}</text>
    <text x={L - 4} y={T + 8} fontSize="10" textAnchor="end" fill="#555">{hi.toFixed(2)}</text>
    <text x={L - 4} y={height - B} fontSize="10" textAnchor="end" fill="#555">{lo.toFixed(2)}</text>
    <text x={L} y={height - 5} fontSize="10" fill="#555">{startSeconds.toFixed(2)} s</text>
    <text x={W - R} y={height - 5} fontSize="10" textAnchor="end" fill="#555">{(startSeconds + seconds).toFixed(2)} s</text>
  </svg>
}
