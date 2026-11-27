import { BrowserRouter, Route, Routes } from 'react-router-dom'
import { Empty } from './components/states'
import { RecordingProvider } from './context/RecordingContext'
import Dataset from './pages/Dataset'
import Demo from './pages/Demo'
import ErrorsLab from './pages/ErrorsLab'
import OverviewLayout from './layouts/OverviewLayout'
import WorkspaceLayout from './layouts/WorkspaceLayout'
import { NAV } from './layouts/nav'
import AnalysisStudio from './pages/AnalysisStudio'
import Challenge from './pages/Challenge'
import Comparison from './pages/Comparison'
import Confidence from './pages/Confidence'
import ModelCard from './pages/ModelCard'
import Overview from './pages/Overview'
import Performance, { MetricDetail } from './pages/Performance'
import Sensitivity from './pages/Sensitivity'
import SignalQuality from './pages/SignalQuality'
import ThreePanel from './pages/ThreePanel'
import XaiStudio from './pages/XaiStudio'
import Viewer from './pages/Viewer'

const Pending = ({ name }: { name: string }) => <Empty title={`${name}: not rebuilt yet`} />

const PAGES: Record<string, React.ReactNode> = {
  '/studio': <AnalysisStudio />,
  '/viewer': <Viewer />,
  '/panels': <ThreePanel />,
  '/quality': <SignalQuality />,
  '/confidence': <Confidence />,
  '/sensitivity': <Sensitivity />,
  '/xai': <XaiStudio />,
  '/errors': <ErrorsLab />,
  '/performance': <Performance />,
  '/comparison': <Comparison />,
  '/model-card': <ModelCard />,
  '/dataset': <Dataset />,
  '/demo': <Demo />,
  '/challenge': <Challenge />,
}

export default function App() {
  return <BrowserRouter><RecordingProvider><Routes>
    <Route element={<OverviewLayout />}><Route path="/" element={<Overview />} /></Route>
    <Route element={<WorkspaceLayout />}>
      {NAV.flatMap(g => g.items).map(([to, label]) => <Route key={to} path={to} element={PAGES[to] ?? <Pending name={label} />} />)}
      <Route path="/performance/:metric" element={<MetricDetail />} />
      <Route path="*" element={<Empty title="Page not found" />} />
    </Route>
  </Routes></RecordingProvider></BrowserRouter>
}
