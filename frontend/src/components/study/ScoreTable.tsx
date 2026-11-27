import type { Majority, Summary } from '../../api/study'
import { Num } from '../Num'
import { Table } from '../ui'
import { modelName } from './names'
import { Role } from './Role'

const HEAD = ['Model', 'Role', 'Macro-F1', 'S recall', 'F recall', 'S F1', 'F F1', 'S precision', 'F precision', 'V recall', 'Accuracy']
/** [path inside a summary, shows ±SD over seeds] */
const COLUMNS: [string, boolean][] = [['macro_f1', true], ['per_class.S.recall', true], ['per_class.F.recall', true], ['per_class.S.f1', false],
  ['per_class.F.f1', false], ['per_class.S.precision', false], ['per_class.F.precision', false], ['per_class.V.recall', false], ['accuracy', false]]
const pick = (o: unknown, path: string) => path.split('.').reduce<unknown>((x, k) => (x as Record<string, unknown>)[k], o)

/** One scored set. Rows keep the order given. The majority-class row is always in the same table as any accuracy. */
export function ScoreTable({ models, primary, rows, source, majority, majoritySource, caption }: {
  models: string[]; primary: string; rows: Record<string, Summary | undefined>; source: (m: string) => string; majority: Majority; majoritySource: string; caption?: string
}) {
  const row = (m: string) => {
    const s = rows[m]
    const head = [<span key="n">{modelName(m)}</span>, <Role key="r" model={m} primary={primary} />]
    if (!s) return [...head, <span key="x" className="muted">not scored on this set</span>, ...Array<string>(COLUMNS.length - 1).fill('')]
    return [...head, ...COLUMNS.map(([path, spread]) => {
      const stat = pick(s, path) as { mean: number; sd: number | null }
      return <span key={path}><Num v={stat.mean} fmt="f3" src={`${source(m)}.${path}.mean`} />
        {spread && stat.sd !== null && <span className="muted small"> ±<Num v={stat.sd} fmt="f3" src={`${source(m)}.${path}.sd`} /></span>}</span>
    })]
  }
  const baseline = [<strong key="n">Majority class (always N)</strong>, <span key="r" className="tag">Baseline</span>,
    ...COLUMNS.map(([path]) => <Num key={path} v={pick(majority, path)} fmt="f3" src={`${majoritySource}.${path}`} />)]
  // primary, then the deployed model (when scored) and the baseline, then everything secondary
  const first = models.filter(m => m === primary || m === 'deployed_model')
  return <Table caption={caption} head={HEAD} rows={[...first.map(row), baseline, ...models.filter(m => !first.includes(m)).map(row)]} />
}
