import type { Beat } from '../../api/types'

/** Index of the last beat at or before t (-1 if none). Beats are in time order. */
export function lastAtOrBefore(beats: Beat[], t: number): number {
  let lo = 0, hi = beats.length - 1, found = -1
  while (lo <= hi) {
    const mid = (lo + hi) >> 1
    if (beats[mid].timestamp_seconds <= t) { found = mid; lo = mid + 1 } else hi = mid - 1
  }
  return found
}

/** The window whose centre is nearest to t. */
export function windowAt(beats: Beat[], t: number): Beat | null {
  const i = lastAtOrBefore(beats, t)
  const a = beats[i], b = beats[i + 1]
  if (!a) return b ?? null
  return b && b.timestamp_seconds - t < t - a.timestamp_seconds ? b : a
}

export const nextAbnormal = (beats: Beat[], t: number) =>
  beats.find(b => b.timestamp_seconds > t + 0.05 && b.prediction.predicted_class !== 'N') ?? null
