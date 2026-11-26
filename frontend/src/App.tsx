import { BrowserRouter, Route, Routes } from 'react-router-dom'
import { Empty } from './components/states'
import { RecordingProvider } from './context/RecordingContext'
import OverviewLayout from './layouts/OverviewLayout'
import WorkspaceLayout from './layouts/WorkspaceLayout'
import { NAV } from './layouts/nav'

const Pending = ({ name }: { name: string }) => <Empty title={`${name}: not rebuilt yet`} />

export default function App() {
  return <BrowserRouter><RecordingProvider><Routes>
    <Route element={<OverviewLayout />}><Route path="/" element={<Pending name="Overview" />} /></Route>
    <Route element={<WorkspaceLayout />}>
      {NAV.flatMap(g => g.items).map(([to, label]) => <Route key={to} path={to} element={<Pending name={label} />} />)}
      <Route path="*" element={<Empty title="Page not found" />} />
    </Route>
  </Routes></RecordingProvider></BrowserRouter>
}
