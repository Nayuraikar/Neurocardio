import { useEffect, useRef, useState } from 'react'
import type { Beat } from '../api/types'
import { ClassBadge } from './ui'

const ROW = 34, VISIBLE = 14

/** Virtualised so a 30-minute record (thousands of windows) stays responsive. */
export function WindowTable({ beats, selectedId, onSelect }: { beats: Beat[]; selectedId?: string | null; onSelect?: (b: Beat) => void }) {
  const [top, setTop] = useState(0)
  const box = useRef<HTMLDivElement>(null)
  const index = beats.findIndex(b => b.id === selectedId)
  useEffect(() => {
    const el = box.current
    if (!el || index < 0) return
    const y = index * ROW
    if (y < el.scrollTop || y > el.scrollTop + el.clientHeight - 2 * ROW) el.scrollTop = Math.max(0, y - 4 * ROW)
  }, [index])
  const from = Math.max(0, Math.floor(top / ROW) - 4), to = Math.min(beats.length, from + VISIBLE + 8)
  return <div className="virtual" ref={box} onScroll={e => setTop(e.currentTarget.scrollTop)}>
    <table>
      <thead><tr><th>#</th><th>Time (s)</th><th>Model class</th><th>Softmax</th><th>Flag</th><th>Reviewer</th></tr></thead>
      <tbody>
        {from > 0 && <tr aria-hidden style={{ height: from * ROW }}><td colSpan={6} /></tr>}
        {beats.slice(from, to).map(b => <tr key={b.id} style={{ height: ROW }} className={`${onSelect ? 'click' : ''} ${b.id === selectedId ? 'selected' : ''}`}
          tabIndex={onSelect ? 0 : undefined} onClick={() => onSelect?.(b)} onKeyDown={e => { if (e.key === 'Enter') onSelect?.(b) }}>
          <td className="num">{b.beat_index + 1}</td><td className="num">{b.timestamp_seconds.toFixed(2)}</td>
          <td><ClassBadge code={b.prediction.predicted_class} /></td><td className="num">{(Math.floor(b.prediction.confidence * 1000) / 10).toFixed(1)}%</td>
          <td>{b.prediction.review_flag ? 'Review' : ''}</td><td>{b.review?.status === 'reviewed' ? <ClassBadge code={b.review.reviewed_label ?? ''} /> : ''}</td>
        </tr>)}
        {to < beats.length && <tr aria-hidden style={{ height: (beats.length - to) * ROW }}><td colSpan={6} /></tr>}
      </tbody>
    </table>
  </div>
}
