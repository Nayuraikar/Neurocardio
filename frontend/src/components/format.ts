export const when = (iso: string | null | undefined) => iso ? new Date(/[zZ]|[+-]\d\d:?\d\d$/.test(iso) ? iso : `${iso}Z`).toLocaleString() : '—'
export const pct = (v: number, digits = 1) => `${(v * 100).toFixed(digits)}%`
export const secs = (s: number) => s >= 60 ? `${Math.floor(s / 60)} min ${(s % 60).toFixed(0)} s` : `${s.toFixed(1)} s`

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
