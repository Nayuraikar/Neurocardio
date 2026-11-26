import type { Beat } from '../api/types'
import { useRecording } from '../context/RecordingContext'
import { ClassBadge } from './ui'

/** Steps through windows; the choice is shared by every workspace page. */
export function WindowSelect({ beats, beat }: { beats: Beat[]; beat: Beat }) {
  const { setWindow } = useRecording()
  const i = beat.beat_index
  const flagged = beats.find(b => b.beat_index > i && b.prediction.review_flag)
  return <div className="row">
    <button type="button" disabled={i <= 0} onClick={() => setWindow(i - 1)}>Previous</button>
    <label className="row">Window <input type="number" min={1} max={beats.length} value={i + 1} style={{ width: 90 }}
      onChange={e => { const n = Number(e.target.value); if (n >= 1 && n <= beats.length) setWindow(n - 1) }} /> of {beats.length}</label>
    <button type="button" disabled={i >= beats.length - 1} onClick={() => setWindow(i + 1)}>Next</button>
    <button type="button" disabled={!flagged} onClick={() => flagged && setWindow(flagged.beat_index)}>Next flagged</button>
    <span><span className="num">{beat.timestamp_seconds.toFixed(2)} s</span> · model class <ClassBadge code={beat.prediction.predicted_class} long /></span>
  </div>
}
