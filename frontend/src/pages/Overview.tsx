import { Link } from 'react-router-dom'
import { useApi } from '../api/useApi'
import type { ModelConfig } from '../api/types'
import { Num } from '../components/Num'
import { when } from '../components/format'
import { Empty, Load, ReviewReady } from '../components/states'
import { Card, Table } from '../components/ui'

type Overview = {
  recordings: number; completed_analyses: number; review_required: number; pending_review_windows: number
  first_review_analysis_id: string | null; model_version: string
  activity: { type: string; at: string; label: string; analysis_id: string | null }[]
}

export default function OverviewPage() {
  const q = useApi<Overview>('/overview')
  const config = useApi<ModelConfig>('/model/config')
  return <>
    <section className="hero">
      <h1>Interpretable AI-powered ECG analysis</h1>
      <p className="muted">For research use. The model classifies individual {config.data ? `${config.data.window_samples / config.data.sampling_rate_hz}-second ` : ''}windows. It does not diagnose a patient or a recording.</p>
      {config.data && <p className="small muted">Loaded model: <span data-source="config#architecture">{config.data.architecture}</span> ·
        input <span data-source="config#input_shape">{config.data.input_shape}</span> · <span data-source="config#normalization">{config.data.normalization}</span></p>}
    </section>
    <Load q={q}>{d => <>
      <div className="grid-4">
        <div className="stat"><b><Num v={d.recordings} src="db#recordings" /></b><span>Recordings stored</span></div>
        <div className="stat"><b><Num v={d.completed_analyses} src="db#completed_analyses" /></b><span>Completed analyses</span></div>
        <div className="stat"><b><Num v={d.review_required} src="db#review_required" /></b><span>Analyses needing review · <Num v={d.pending_review_windows} src="db#pending_review_windows" /> windows pending</span></div>
        <div className="stat"><b style={{ fontSize: 17 }} data-source="config#model_version">{d.model_version}</b><span>Model version</span></div>
      </div>
      {d.first_review_analysis_id && <ReviewReady pending={d.pending_review_windows} to={`/history/${d.first_review_analysis_id}`} />}
    </>}</Load>
    <div className="launcher">
      <Link className="launch feature" to="/studio"><h2>Analysis Studio</h2><p>Upload a recording, follow each processing step, and inspect every window the model classified.</p></Link>
      <Link className="launch" to="/dataset"><h2>Dataset Explorer</h2><p>Class counts, patient splits and real example windows.</p></Link>
      <Link className="launch" to="/performance"><h2>Model Performance</h2><p>Held-out results shown with the majority-class baseline and per-class weaknesses.</p></Link>
    </div>
    <Card title="Recent activity">
      <Load q={q}>{d => d.activity.length
        ? <Table head={['When', 'Event', 'Recording', '']} rows={d.activity.map(e => [when(e.at), e.type, e.label,
          e.analysis_id ? <Link key="l" to={`/viewer?analysis=${e.analysis_id}&window=0`}>Open</Link> : ''])} />
        : <Empty title="Nothing recorded yet">Upload a recording in the Analysis Studio.</Empty>}</Load>
    </Card>
  </>
}
