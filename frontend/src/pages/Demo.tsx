import { useState } from 'react'
import type { DemoList, DemoSample } from '../api/demo'
import { useApi } from '../api/useApi'
import { Load, PartialNotice } from '../components/states'
import { Card, ClassBadge, PageHead, ProbBars } from '../components/ui'
import { SourceSplit } from '../components/SourceSplit'
import { WeaknessNote } from '../components/WeaknessNote'
import { AttributionPlot } from '../components/waveform/AttributionPlot'
import { Trace } from '../components/waveform/Trace'
import { CLASS_CODES, classFill, SOFTMAX_NOTICE, XAI_NOTICE } from '../theme/theme'

function Attribution({ sample, cls }: { sample: DemoSample; cls: string }) {
  const q = useApi<{ attributions: number[]; top_regions: { start_sample: number; end_sample: number }[] }>(`/demo/samples/${sample.id}/gradcam?class=${cls}`)
  return <Load q={q}>{d => <><AttributionPlot signal={sample.model_input} values={d.attributions} signed={false} regions={d.top_regions} label={`Grad-CAM for class ${cls}`} /><p className="notice">{XAI_NOTICE}</p></>}</Load>
}

function View({ id }: { id: string }) {
  const q = useApi<DemoSample>(`/demo/samples/${id}`)
  const [explain, setExplain] = useState(false)
  return <Load q={q} lines={8}>{s => <div className="stack">
    <SourceSplit s={s} />
    <p>Reference label <ClassBadge code={s.label!} long /> · model class <ClassBadge code={s.prediction!.predicted_class} long /> {s.label !== s.prediction!.predicted_class && <strong>· the model disagrees with the reference</strong>}</p>
    <Trace values={s.model_input} fs={s.sampling_rate} label="Model input window" unit="unitless z-score" height={220} band={{ from: 270, to: 450, color: classFill(s.prediction!.predicted_class) }} />
    <ProbBars probs={s.prediction!.probabilities} />
    <p className="notice">{SOFTMAX_NOTICE}</p>
    {explain ? <Attribution sample={s} cls={s.prediction!.predicted_class} /> : <div><button type="button" onClick={() => setExplain(true)}>Show Grad-CAM</button></div>}
  </div>}</Load>
}

function Playground({ list }: { list: DemoList }) {
  const [label, setLabel] = useState('N')
  const [picked, setPicked] = useState<string | null>(null)
  const options = list.samples.filter(s => s.label === label)
  const id = options.some(s => s.id === picked) ? picked! : options[0]?.id
  return <>
    <p className="muted">{list.source}. Selection: {list.selection}.</p>
    <Card title="Pick a labelled window">
      <div className="row" role="group" aria-label="Reference class">{CLASS_CODES.map(c => <button type="button" key={c} aria-pressed={label === c} onClick={() => setLabel(c)}>{c}</button>)}
        <select aria-label="Sample" value={id} onChange={e => setPicked(e.target.value)}>{options.map(s => <option key={s.id} value={s.id}>record {s.record}, sample {s.sample} ({s.app_model_split})</option>)}</select></div>
    </Card>
    {id && <Card><View key={id} id={id} /></Card>}
  </>
}

export default function Demo() {
  const q = useApi<DemoList>('/demo/samples')
  return <><PageHead title="Demo Playground" sub="Real labelled windows from training and validation patients." />
    <PartialNotice>These patients were used to train or select the model, so it does <strong>better here than on new patients</strong>. This page is a demonstration, not an evaluation. No test patient appears.</PartialNotice>
    <WeaknessNote />
    <Load q={q}>{list => <Playground list={list} />}</Load></>
}
