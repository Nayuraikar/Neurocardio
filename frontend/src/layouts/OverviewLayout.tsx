import { Link, Outlet } from 'react-router-dom'
import { NOTICE } from '../theme/theme'
import { BackendBanner, SystemStatus } from './StatusBar'

export default function OverviewLayout() {
  return <div className="overview-shell">
    <header className="overview-top">
      <Link to="/" className="brand">NeuroCardio</Link>
      <SystemStatus />
      <Link className="btn primary" to="/studio">Open workspace</Link>
    </header>
    <main className="overview-main" id="main"><BackendBanner /><Outlet /></main>
    <footer className="foot">{NOTICE}</footer>
  </div>
}
