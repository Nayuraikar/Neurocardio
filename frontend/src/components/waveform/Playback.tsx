import { useEffect, useRef, useState } from 'react'
import type { Beat } from '../../api/types'
import { ClassBadge } from '../ui'
import { nextAbnormal, windowAt } from './playbackLogic'

const SPEEDS = [0.5, 1, 2, 4]

/** Replays a stored recording. It is not live monitoring: every class shown was computed beforehand. */
export function Playback({ duration, beats, cursor, onCursor, start, onStop }: {
  duration: number; beats: Beat[]; cursor: number | null; onCursor: (t: number) => void; start: number; onStop: (b: Beat | null) => void
}) {
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(1)
  const time = useRef(0)
  const move = useRef(onCursor)
  useEffect(() => { move.current = onCursor; if (cursor != null) time.current = cursor })
  useEffect(() => {
    if (!playing) return
    let frame = 0, last = performance.now()
    const tick = (now: number) => {
      const next = Math.min(duration, time.current + Math.min(now - last, 250) / 1000 * speed)
      last = now
      time.current = next
      move.current(next)
      if (next >= duration) setPlaying(false)
      else frame = requestAnimationFrame(tick)
    }
    frame = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(frame)
  }, [playing, speed, duration])

  const live = cursor == null ? null : windowAt(beats, cursor)
  const toggle = () => {
    if (playing) { setPlaying(false); onStop(live); return }
    if (cursor == null || cursor >= duration) { time.current = cursor == null ? start : 0; onCursor(time.current) }
    setPlaying(true)
  }
  const jump = () => {
    const b = nextAbnormal(beats, cursor ?? start - 0.1)
    if (b) { setPlaying(false); time.current = b.timestamp_seconds; onCursor(b.timestamp_seconds); onStop(b) }
  }
  return <div className="row between" style={{ marginTop: 10 }} aria-label="Playback controls">
    <div className="row">
      <button type="button" className="primary" onClick={toggle}>{playing ? 'Pause' : 'Play'}</button>
      <div className="row" role="group" aria-label="Playback speed">{SPEEDS.map(s => <button type="button" key={s} aria-pressed={speed === s} onClick={() => setSpeed(s)}>{s}x</button>)}</div>
      <button type="button" onClick={jump} disabled={!nextAbnormal(beats, cursor ?? start - 0.1)}>Next window not classed N</button>
    </div>
    <div className="row" data-testid="live-class">
      <span className="num">{cursor == null ? 'stopped' : `${cursor.toFixed(1)} s`}</span>
      {live && <>window #{live.beat_index + 1} <ClassBadge code={live.prediction.predicted_class} long /> <span className="num">{(Math.floor(live.prediction.confidence * 1000) / 10).toFixed(1)}%</span>{live.prediction.review_flag && <span className="tag moderate">flagged for review</span>}</>}
      <span className="small muted">Replay of stored results, not live monitoring.</span>
    </div>
  </div>
}
