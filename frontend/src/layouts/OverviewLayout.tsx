import { Link, Outlet, useLocation } from 'react-router-dom'
import { ErrorBoundary } from '../components/ErrorBoundary'
import { NOTICE } from '../theme/theme'
import { BackendBanner, SystemStatus } from './StatusBar'

export default function OverviewLayout() {
  const { pathname } = useLocation()
  return <div className="overview-shell">
    <header className="overview-top">
      <Link to="/" className="brand">NeuroCardio</Link>
      <SystemStatus />
      <Link className="btn primary" to="/studio">Open workspace</Link>
    </header>
    <main className="overview-main" id="main"><BackendBanner /><ErrorBoundary key={pathname}><Outlet /></ErrorBoundary></main>
    <footer className="foot">{NOTICE}</footer>
  </div>
}
