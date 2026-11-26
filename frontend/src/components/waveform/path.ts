/** Min/max decimation per half pixel so narrow R-peaks survive zooming out. */
export function decimatedPath(n: number, xAt: (i: number) => number, yAt: (i: number) => number): string {
  if (!n) return ''
  const cols = new Map<number, [number, number, number, number]>()
  for (let i = 0; i < n; i += 1) {
    const c = Math.round(xAt(i) * 2) / 2, y = yAt(i)
    const e = cols.get(c)
    if (!e) cols.set(c, [y, y, y, y])
    else { if (y < e[0]) e[0] = y; if (y > e[1]) e[1] = y; e[3] = y }
  }
  let d = ''
  cols.forEach(([lo, hi, first, last], c) => {
    const down = last >= first
    d += `${d ? 'L' : 'M'}${c} ${(down ? lo : hi).toFixed(1)}L${c} ${(down ? hi : lo).toFixed(1)}`
  })
  return d
}

export function extent(values: ArrayLike<number>): [number, number] {
  let lo = Infinity, hi = -Infinity
  for (let i = 0; i < values.length; i += 1) { const v = values[i]; if (v < lo) lo = v; if (v > hi) hi = v }
  if (!Number.isFinite(lo)) return [0, 1]
  return lo === hi ? [lo - 1, hi + 1] : [lo, hi]
}
