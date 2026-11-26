import { CLASS_CODES, CLASSES } from '../theme/theme'
import { Num } from './Num'
import { ClassBadge } from './ui'

export function ClassDistribution({ counts, source }: { counts: Record<string, number>; source: string }) {
  const max = Math.max(1, ...CLASS_CODES.map(c => counts[c] ?? 0))
  return <div className="probs">
    {CLASS_CODES.map(c => <div key={c} className="prob-row" style={{ gridTemplateColumns: '34px 1fr 80px' }}>
      <ClassBadge code={c} />
      <div className="bar"><i style={{ width: `${(counts[c] ?? 0) / max * 100}%`, background: CLASSES[c].fill }} /></div>
      <span className="num" style={{ textAlign: 'right' }}><Num v={counts[c] ?? 0} src={`${source}.${c}`} /></span>
    </div>)}
  </div>
}
