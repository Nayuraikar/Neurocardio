export const when = (iso: string | null | undefined) => iso ? new Date(/[zZ]|[+-]\d\d:?\d\d$/.test(iso) ? iso : `${iso}Z`).toLocaleString() : '—'
export const pct = (v: number, digits = 1) => `${(v * 100).toFixed(digits)}%`
export const secs = (s: number) => s >= 60 ? `${Math.floor(s / 60)} min ${(s % 60).toFixed(0)} s` : `${s.toFixed(1)} s`
