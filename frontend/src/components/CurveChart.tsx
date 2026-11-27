import { useId, type ReactNode } from 'react'

export type Series = { label: string; legend?: ReactNode; color: string; x: number[]; y: number[]; dash?: boolean }

export function CurveChart({ series, xLabel, yLabel, title, diagonal, yMax = 1, xDomain = [0, 1] }: {
  series: Series[]; xLabel: string; yLabel: string; title: string; diagonal?: boolean; yMax?: number; xDomain?: [number, number]
}) {
  const id = useId()
  const W = 360, H = 270, L = 40, R = 10, T = 12, B = 34
  const X = (v: number) => L + (v - xDomain[0]) / (xDomain[1] - xDomain[0] || 1) * (W - L - R)
  const Y = (v: number) => T + (1 - v / yMax) * (H - T - B)
  return <figure style={{ margin: 0 }}>
    <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-labelledby={id}>
      <title id={id}>{title}</title>
      {[0, 0.25, 0.5, 0.75, 1].map(f => <g key={f}>
        <line x1={L} x2={W - R} y1={Y(f * yMax)} y2={Y(f * yMax)} stroke="#e3e1d4" /><text x={L - 5} y={Y(f * yMax) + 3} textAnchor="end">{(f * yMax).toFixed(2)}</text>
        <text x={X(xDomain[0] + f * (xDomain[1] - xDomain[0]))} y={H - B + 13} textAnchor="middle">{(xDomain[0] + f * (xDomain[1] - xDomain[0])).toFixed(xDomain[1] > 2 ? 0 : 2)}</text></g>)}
      {diagonal && <line x1={X(0)} y1={Y(0)} x2={X(1)} y2={Y(1)} stroke="#999" strokeDasharray="4 4" />}
      {series.map(s => <polyline key={s.label} fill="none" stroke={s.color} strokeWidth="1.8" strokeDasharray={s.dash ? '5 3' : undefined}
        points={s.x.map((v, i) => `${X(v).toFixed(1)},${Y(Math.min(yMax, s.y[i])).toFixed(1)}`).join(' ')} />)}
      <text x={(L + W - R) / 2} y={H - 4} textAnchor="middle">{xLabel}</text>
      <text transform={`translate(10 ${(T + H - B) / 2}) rotate(-90)`} textAnchor="middle">{yLabel}</text>
    </svg>
    <figcaption className="legend"><strong>{title}</strong>{series.map(s => <span key={s.label}><i style={{ background: s.color }} />{s.legend ?? s.label}</span>)}</figcaption>
  </figure>
}
