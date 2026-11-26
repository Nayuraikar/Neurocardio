/** The last selection, kept in localStorage. It is only a convenience: ids are always re-checked with the API. */
export type Stored = { analysis: string | null; window: number }
const KEY = 'neurocardio.selection'
const listeners = new Set<() => void>()
let current: Stored = (() => {
  try { return { analysis: null, window: 0, ...JSON.parse(localStorage.getItem(KEY) ?? '{}') } } catch { return { analysis: null, window: 0 } }
})()

export const getSelection = () => current
export const subscribe = (fn: () => void) => { listeners.add(fn); return () => { listeners.delete(fn) } }
export function setSelection(next: Stored) {
  if (next.analysis === current.analysis && next.window === current.window) return
  current = next
  try { localStorage.setItem(KEY, JSON.stringify(next)) } catch { /* storage may be blocked */ }
  listeners.forEach(fn => fn())
}
