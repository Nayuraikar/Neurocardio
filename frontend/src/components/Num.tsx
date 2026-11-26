export type Fmt = 'int' | 'f3' | 'f4' | 'pct1' | 'raw' | 'f2'

export function formatValue(v: unknown, fmt: Fmt): string {
  if (v === null || v === undefined) return '—'
  if (typeof v !== 'number') return String(v)
  if (fmt === 'int') return Math.round(v).toLocaleString('en-US')
  if (fmt === 'pct1') return `${(v * 100).toFixed(1)}%`
  if (fmt === 'f2') return v.toFixed(2)
  if (fmt === 'f3') return v.toFixed(3)
  if (fmt === 'f4') return v.toFixed(4)
  return String(v)
}

/** A displayed statistic with its source ("<artifact>#<path>"), so the checker can verify it. */
export function Num({ v, src, fmt = 'int' }: { v: unknown; src: string; fmt?: Fmt }) {
  return <span className="num" data-source={src} data-fmt={fmt}>{formatValue(v, fmt)}</span>
}
