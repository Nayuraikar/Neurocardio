import { decimatedPath, extent } from './path'

export const POSITIVE = '#1f5f8b'
export const NEGATIVE = '#8a4a12'

/** The model input with its attribution aligned underneath. Signed maps are drawn above and below zero. */
export function AttributionPlot({ signal, values, signed, label, regions = [] }: {
  signal: number[]; values: number[]; signed: boolean; label: string; regions?: { start_sample: number; end_sample: number }[]
}) {
  const W = 1000, L = 52, R = 10, S = 150, A = 110, H = S + A + 24
  const [lo, hi] = extent(signal)
  const X = (i: number) => L + i / (signal.length - 1) * (W - L - R)
  const Y = (v: number) => 18 + (1 - (v - lo) / (hi - lo)) * (S - 28)
  const scale = Math.max(...values.map(Math.abs)) || 1
  const zero = signed ? S + A / 2 : S + A - 4
  const AY = (v: number) => zero - v / scale * (signed ? A / 2 - 6 : A - 12)
  const bars = (positive: boolean) => values.map((v, i) => (positive ? v > 0 : v < 0)
    ? `M${X(i).toFixed(1)} ${zero}V${AY(v).toFixed(1)}` : '').join('')
  const allZero = values.every(v => v === 0)
  return <svg className="plot" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${label}: model input window with attribution below`}>
    {regions.map((r, i) => <g key={i}><rect x={X(r.start_sample)} width={Math.max(2, X(r.end_sample - 1) - X(r.start_sample))} y={18} height={S - 28} fill="#2b3a1e" opacity=".08" /><text x={X(r.start_sample) + 2} y={28} fontSize="10" fill="#2b3a1e">R{i + 1}</text></g>)}
    <line x1={X(360)} x2={X(360)} y1={18} y2={H - 20} stroke="#888" strokeDasharray="3 4" />
    <path d={decimatedPath(signal.length, X, i => Y(signal[i]))} fill="none" stroke="var(--trace)" strokeWidth="1.3" />
    <line x1={L} x2={W - R} y1={zero} y2={zero} stroke="#999" />
    <path d={bars(true)} stroke={POSITIVE} strokeWidth="1.3" fill="none" />
    {signed && <path d={bars(false)} stroke={NEGATIVE} strokeWidth="1.3" fill="none" />}
    <text x={L} y={12} fontSize="11" fill="#444">{label}</text>
    <text x={L - 4} y={S + 10} fontSize="10" textAnchor="end" fill="#555">{scale.toPrecision(2)}</text>
    {signed && <text x={L - 4} y={S + A - 2} fontSize="10" textAnchor="end" fill="#555">−{scale.toPrecision(2)}</text>}
    {allZero && <text x={W / 2} y={S + A / 2 - 8} fontSize="13" textAnchor="middle" fill="#8f301a">This map is all zero: the method produced no attribution for this window.</text>}
    <text x={L} y={H - 6} fontSize="10" fill="#555">−1000 ms</text>
    <text x={X(360)} y={H - 6} fontSize="10" textAnchor="middle" fill="#555">centre beat</text>
    <text x={W - R} y={H - 6} fontSize="10" textAnchor="end" fill="#555">+1000 ms</text>
  </svg>
}
