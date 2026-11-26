import type { ReactNode } from 'react'
import { classFill, classText, className_, CLASS_CODES } from '../theme/theme'

export function PageHead({ title, sub, actions }: { title: string; sub?: ReactNode; actions?: ReactNode }) {
  return <header className="page-head">
    <div><h1>{title}</h1>{sub && <p className="muted">{sub}</p>}</div>
    {actions && <div className="row">{actions}</div>}
  </header>
}

export function Card({ title, aside, children, className = '' }: { title?: ReactNode; aside?: ReactNode; children: ReactNode; className?: string }) {
  return <section className={`card ${className}`}>
    {(title || aside) && <div className="card-head">{title && <h2>{title}</h2>}{aside}</div>}
    {children}
  </section>
}

export function ClassBadge({ code, long }: { code: string; long?: boolean }) {
  return <span className="badge" style={{ color: classText(code), borderColor: classFill(code), background: `${classFill(code)}22` }} title={className_(code)}>
    {code}{long ? ` · ${className_(code)}` : ''}
  </span>
}

export function ProbBars({ probs }: { probs: number[] }) {
  return <div className="probs">
    {CLASS_CODES.map((c, i) => <div key={c} className="prob-row">
      <ClassBadge code={c} />
      <div className="bar"><i style={{ width: `${(probs[i] ?? 0) * 100}%`, background: classFill(c) }} /></div>
      <span className="num">{((probs[i] ?? 0) * 100).toFixed(2)}%</span>
    </div>)}
  </div>
}

export function KeyValues({ rows }: { rows: [string, ReactNode][] }) {
  return <dl className="kv">{rows.map(([k, v]) => <div key={k}><dt>{k}</dt><dd>{v}</dd></div>)}</dl>
}

export function Table({ head, rows, caption }: { head: ReactNode[]; rows: ReactNode[][]; caption?: string }) {
  return <div className="table-wrap"><table>
    {caption && <caption>{caption}</caption>}
    <thead><tr>{head.map((h, i) => <th key={i} scope="col">{h}</th>)}</tr></thead>
    <tbody>{rows.map((r, i) => <tr key={i}>{r.map((c, j) => <td key={j}>{c}</td>)}</tr>)}</tbody>
  </table></div>
}
