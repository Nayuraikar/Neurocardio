import { Link } from 'react-router-dom'
import { useApi } from '../api/useApi'
import type { Beat, RecordInfo } from '../api/types'
import { Load } from '../components/states'
import { Card, ClassBadge, PageHead, ProbBars } from '../components/ui'
import { WeaknessNote } from '../components/WeaknessNote'
import { WindowSelect } from '../components/WindowSelect'
import type { WindowData } from '../components/WindowStages'
import { WithWindow } from '../components/WithWindow'
import { Trace } from '../components/waveform/Trace'
import { useRecording } from '../context/RecordingContext'
import { classFill, SOFTMAX_NOTICE } from '../theme/theme'

function Panels({ record, beat }: { record: RecordInfo; beat: Beat }) {
  const q = useApi<WindowData>(`/recordings/${record.id}/window?sample=${beat.sample_index}`)
  const { link } = useRecording()
  const p = beat.prediction
  return <Load q={q} lines={8}>{w => {
    const start = w.start_sample / w.sampling_rate, unit = w.units ?? 'signal units'
    const band = { from: 360 - 90, to: 360 + 90, color: classFill(p.predicted_class) }
    return <div className="grid-3">
      <Card title="1 · Raw">
        <Trace values={w.raw} fs={w.sampling_rate} startSeconds={start} label="As recorded" unit={unit} height={300} />
        <p className="small muted">The stored samples, unchanged.</p>
      </Card>
      <Card title="2 · Processed">
        <Trace values={w.model_input} fs={w.sampling_rate} startSeconds={start} label="Model input" unit="unitless z-score" height={300} />
        <p className="small muted">Bandpass filtered on the whole record, cut to this window, then standardized on its own.</p>
      </Card>
      <Card title="3 · Model interpretation">
        <Trace values={w.model_input} fs={w.sampling_rate} startSeconds={start} label="Centre beat of the window" unit="z-score" height={170} band={band} />
        <p>Model class <ClassBadge code={p.predicted_class} long /></p>
        <ProbBars probs={p.probabilities} />
        <p>{p.review_flag ? <span className="tag moderate">Review recommended</span> : <span className="tag">Not flagged by the review rule</span>}</p>
        <Link className="btn" to={link('/xai', beat.beat_index)}>Show attribution</Link>
        <p className="notice">{SOFTMAX_NOTICE} The shading marks the beat the window is centred on; it is not an attribution.</p>
      </Card>
    </div>
  }}</Load>
}

export default function ThreePanel() {
  return <><PageHead title="Three-Panel View" sub="The same 2-second window: raw, processed, and the model’s output." />
    <WithWindow>{({ record, beats, beat }) => <><WindowSelect beats={beats} beat={beat} /><Panels record={record} beat={beat} /><WeaknessNote /></>}</WithWindow></>
}
