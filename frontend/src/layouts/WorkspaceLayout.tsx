import { useState } from 'react'
import { Link, NavLink, Outlet, useLocation } from 'react-router-dom'
import { useRecording } from '../context/RecordingContext'
import { NOTICE } from '../theme/theme'
import { NAV, NEEDS_RECORDING, TITLES } from './nav'
import { BackendBanner, SystemStatus } from './StatusBar'

export default function WorkspaceLayout() {
  const { pathname } = useLocation()
  const rec = useRecording()
  const [open, setOpen] = useState(false)
  const base = `/${pathname.split('/')[1]}`
  const crumbs: [string, string | null][] = [['Overview', '/']]
  if (TITLES[base]) crumbs.push([TITLES[base], pathname === base ? null : base])
  if (pathname !== base) crumbs.push([decodeURIComponent(pathname.split('/').slice(2).join(' / ')), null])
  return <div className="workspace">
    <a className="skip" href="#main">Skip to content</a>
    <aside className={`sidebar ${open ? 'open' : ''}`}>
      <Link to="/" className="brand">NeuroCardio</Link>
      <nav aria-label="Workspace">
        {NAV.map(g => <div key={g.group}>
          <p className="nav-group">{g.group}</p>
          {g.items.map(([to, label]) => <NavLink key={to} to={NEEDS_RECORDING.includes(to) ? rec.link(to) : to} onClick={() => setOpen(false)}>{label}</NavLink>)}
        </div>)}
      </nav>
    </aside>
    <div className="workspace-body">
      <div className="topbar">
        <button type="button" className="menu-btn" onClick={() => setOpen(o => !o)} aria-expanded={open}>Menu</button>
        <ol className="crumbs" aria-label="Breadcrumb">{crumbs.map(([l, to], i) => <li key={i}>{to ? <Link to={to}>{l}</Link> : <span aria-current="page">{l}</span>}</li>)}</ol>
        <div className="topbar-rec">
          {rec.record && rec.analysis
            ? <><span title={rec.analysis.id}>{rec.record.filename} · {rec.record.lead} · {rec.analysis.status}</span><button type="button" onClick={() => rec.select(null)}>Clear</button></>
            : <span className="muted">No recording selected</span>}
        </div>
        <SystemStatus />
      </div>
      <main id="main" className="workspace-main"><BackendBanner /><Outlet /></main>
      <footer className="foot">{NOTICE}</footer>
    </div>
  </div>
}
