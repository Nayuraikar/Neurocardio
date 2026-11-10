import type { ReactNode } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { ArrowUpRight, HeartPulse } from 'lucide-react'

const items = [
  ['DASHBOARD', '/app/dashboard', ['/app/dashboard']],
  ['ANALYSIS', '/app/analysis', ['/app/analys', '/app/explainability', '/app/new']],
  ['MODEL', '/app/model', ['/app/model']],
  ['REPORTS', '/app/reports', ['/app/reports']],
] as const
const more = [['RECORDINGS', '/app/recordings'], ['HUMAN REVIEW', '/app/reviews'], ['NEW ANALYSIS', '/app/new'], ['SETTINGS', '/app/settings']] as const

export default function TopNav({ extra }: { extra?: ReactNode }) {
  const { pathname } = useLocation()
  return <header className="pf-top">
    <div className="pf-top-row">
      <Link to="/" className="pf-brand" aria-label="NeuroCardio home"><span className="pf-logo"><HeartPulse size={18} strokeWidth={1.6} /></span>NeuroCardio</Link>
      <nav className="pf-pill" aria-label="Primary">
        {items.map(([label, to, prefixes]) => <Link key={label} to={to} className={prefixes.some(p => pathname.startsWith(p)) ? 'active' : ''} aria-current={prefixes.some(p => pathname.startsWith(p)) ? 'page' : undefined}>{label}</Link>)}
      </nav>
      <Link className="pf-cta" to="/app/dashboard#upload">UPLOAD ECG <ArrowUpRight size={14} /></Link>
    </div>
    <nav className="pf-sub" aria-label="More views">
      {more.map(([label, to]) => <Link key={label} to={to}>{label}</Link>)}
      {extra}
    </nav>
  </header>
}
