import { useApi } from '../api/useApi'
import type { Analysis, RecordInfo } from '../api/types'
import { NeedsAnalysis } from '../components/NeedsAnalysis'
import { Load } from '../components/states'
import { Card, PageHead, Table } from '../components/ui'

type Quality = {
  overall: string; rule: string; caveat: string; lead: string
  metrics: { key: string; label: string; definition: string; value: number; state: string; good_below: number; moderate_below: number }[]
  counts: Record<string, number>
}
const percent = (v: number) => `${(v * 100).toFixed(v > 0 && v < 0.001 ? 4 : 2)}%`
const State = ({ value }: { value: string }) => <span className={`tag ${value.toLowerCase()}`}>{value}</span>

function Panel({ analysis, record }: { analysis: Analysis; record: RecordInfo }) {
  const q = useApi<Quality>(`/recordings/${record.id}/quality?analysis_id=${analysis.id}`)
  return <Load q={q} lines={8}>{d => <>
    <Card title={<>Overall: <State value={d.overall} /></>} aside={<span className="muted">{record.filename} · {d.lead} · raw signal</span>}>
      <p className="muted">{d.rule}</p>
      <Table head={['Measurement', 'Value', 'State', 'Good below', 'Moderate below', 'How it is measured']} rows={d.metrics.map(m => [
        m.label, <span className="num" key="v">{percent(m.value)}</span>, <State key="s" value={m.state} />,
        m.key === 'missing_fraction' ? 'none' : percent(m.good_below), percent(m.moderate_below), <span key="d" className="small">{m.definition}</span>])} />
      <p className="notice">{d.caveat}</p>
    </Card>
    <Card title="Counts">
      <Table head={['Samples', 'Missing samples', 'Beat positions', 'Edge-crossing windows', 'Flat windows']}
        rows={[[d.counts.samples, d.counts.missing_samples, d.counts.beat_positions, d.counts.edge_windows, d.counts.flat_windows].map(n => n.toLocaleString())]} />
    </Card>
  </>}</Load>
}

export default function SignalQuality() {
  return <><PageHead title="Signal Quality" sub="Measured on the stored raw signal of the selected recording." />
    <NeedsAnalysis>{(a, r) => <Panel analysis={a} record={r} />}</NeedsAnalysis></>
}
