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
import Batch from './pages/Batch'
import Challenge from './pages/Challenge'
import Compare from './pages/Compare'
import Comparison from './pages/Comparison'
import Confidence from './pages/Confidence'
import History from './pages/History'
import ModelCard from './pages/ModelCard'
import Overview from './pages/Overview'
import Performance, { MetricDetail } from './pages/Performance'
import Presentation from './pages/Presentation'
import ReportBuilder from './pages/ReportBuilder'
import ReviewWorkspace from './pages/ReviewWorkspace'
import Sensitivity from './pages/Sensitivity'
import SignalQuality from './pages/SignalQuality'
import ThreePanel from './pages/ThreePanel'
import XaiStudio from './pages/XaiStudio'
import Viewer from './pages/Viewer'

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
  '/history': <History />,
  '/batch': <Batch />,
  '/compare': <Compare />,
}

export default function App() {
  return <BrowserRouter><RecordingProvider><Routes>
    <Route path="/present" element={<Presentation />} />
    <Route element={<OverviewLayout />}><Route path="/" element={<Overview />} /></Route>
    <Route element={<WorkspaceLayout />}>
      {NAV.flatMap(g => g.items).filter(([to]) => to !== '/present').map(([to]) => <Route key={to} path={to} element={PAGES[to]} />)}
      <Route path="/performance/:metric" element={<MetricDetail />} />
      <Route path="/history/:id" element={<ReviewWorkspace />} />
      <Route path="/history/:id/report" element={<ReportBuilder />} />
      <Route path="*" element={<Empty title="Page not found" />} />
    </Route>
  </Routes></RecordingProvider></BrowserRouter>
}
