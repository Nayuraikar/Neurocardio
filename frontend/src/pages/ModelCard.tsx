import { Link } from 'react-router-dom'
import type { ModelMetrics } from '../api/metrics'
import { MODEL_LABELS, RR_MODELS, type Study } from '../api/study'
import type { ModelConfig } from '../api/types'
import { useApi } from '../api/useApi'
import { Num } from '../components/Num'
import { Empty, Load, PartialNotice } from '../components/states'
import { LiveModelNotice } from '../components/study/LiveModelNotice'
import { StudySection } from '../components/study/StudySection'
import { Card, KeyValues, PageHead, Table } from '../components/ui'
import { WeaknessNote } from '../components/WeaknessNote'
import { CLASS_CODES, INCART_NOTICE, NOTICE } from '../theme/theme'

const DB: Record<string, string> = { mitdb: 'MIT-BIH Arrhythmia', svdb: 'MIT-BIH Supraventricular (SVDB)', incartdb: 'St Petersburg INCART' }

export default function ModelCard() {
  const config = useApi<ModelConfig>('/model/config')
  const metrics = useApi<ModelMetrics>('/model/metrics')
  const study = useApi<Study>('/study')
  return <><PageHead title="Model Card" sub="What the models were trained on, what they are for, and where they fail." />
    <LiveModelNotice study={study.data} />
    <Card title="Intended use"><p>{NOTICE}</p><p>For research and teaching on retrospective, single-lead recordings. Not for patient care, monitoring, screening or triage. Paced rhythms are out of scope.</p></Card>
    <Card title="The app’s model"><Load q={config}>{c => <KeyValues rows={[
      ['Architecture', <span data-source="config#architecture">{c.architecture}</span>], ['Input', <span data-source="config#input_shape">{c.input_shape}</span>],
      ['Normalization', <span data-source="config#normalization">{c.normalization}</span>], ['Lead', <span data-source="config#lead">{c.lead}</span>], ['Sampling rate', <><Num v={c.sampling_rate_hz} fmt="raw" src="config#sampling_rate_hz" /> Hz</>], ['Uses RR features', c.uses_rr_features ? 'yes' : 'no'],
      ['Version', <span data-source="config#model_version">{c.model_version}</span>], ['File SHA-256', <code data-source="config#model_sha256">{c.model_sha256}</code>]]} />}</Load></Card>
    <WeaknessNote />
    <Load q={metrics}>{m => !m.available ? <Empty title={m.message ?? 'Not available'} /> : <>
      <Card title="Training data of the app’s model: MIT-BIH only">
        <Table caption="Usable windows before augmentation, patient-wise split" head={['Part', 'Patients', ...CLASS_CODES]} rows={(['train', 'validation', 'test'] as const).map(p => [p,
          <Num key="p" v={m.metrics.split.patient_group_ids[p].length} src={`metrics#split.patient_group_ids.${p}.length`} />,
          ...CLASS_CODES.map(c => <Num key={c} v={m.metrics.dataset.class_counts[p].usable_windows[c]} src={`metrics#dataset.class_counts.${p}.usable_windows.${c}`} />)])} />
      </Card>
      <Card title="Single-patient dependence">
        <Table caption="Share of a class that comes from its single largest record" head={['Part', 'Class', 'Windows (annotated)', 'Largest record', 'Its share']}
          rows={m.minority_concentration.map((r, i) => [r.split, r.class, <Num key="t" v={r.total_beats} src={`minority#${i}.total_beats`} />, <Num key="id" v={r.top_record} fmt="raw" src={`minority#${i}.top_record`} />,
            <Num key="s" v={r.top_record_share} fmt="pct1" src={`minority#${i}.top_record_share`} />])} />
        {m.minority_concentration.filter(r => r.split === 'train' && r.class === 'F').map(r => <PartialNotice key="f">
          <strong>Record <Num v={r.top_record} fmt="raw" src={`minority#${m.minority_concentration.indexOf(r)}.top_record`} /> supplies <Num v={r.top_record_share} fmt="pct1" src={`minority#${m.minority_concentration.indexOf(r)}.top_record_share`} /> of the training F beats.</strong> What the model learned as “fusion” is largely one patient’s beats. Results on that record are not evidence of generalisation.</PartialNotice>)}
      </Card>
    </>}</Load>
    <Load q={study}>{s => {
      const d = s.data_report
      if (!d) return <Empty title="Study data report not generated yet.">{s.data_report_problem}</Empty>
      return <>
        <Card title="Training data of the study models, per database">
          <Table caption="Training part, patient-wise" head={['Database', 'Lead', 'Records', 'Patients', ...CLASS_CODES]} rows={Object.entries(d.splits.train.by_database ?? {}).map(([k, v]) => [DB[k] ?? k, <Num key="l" v={d.lead[k]} fmt="raw" src={`datareport#lead.${k}`} />,
            <Num key="r" v={v.records} src={`datareport#splits.train.by_database.${k}.records`} />, <Num key="p" v={v.patients} src={`datareport#splits.train.by_database.${k}.patients`} />,
            ...CLASS_CODES.map(c => <Num key={c} v={v.windows[c]} src={`datareport#splits.train.by_database.${k}.windows.${c}`} />)])} />
          <Table caption="Patient-wise parts of the study split" head={['Part', 'Records', 'Patients', 'Windows']} rows={Object.entries(d.splits).map(([k, v]) => [k,
            <Num key="r" v={v.records} src={`datareport#splits.${k}.records`} />, <Num key="p" v={v.patients} src={`datareport#splits.${k}.patients`} />, <Num key="t" v={v.total} src={`datareport#splits.${k}.total`} />])} />
          <p className="notice">{INCART_NOTICE} The SVDB record-to-patient mapping is assumed (one record, one patient) because none is published. These leads differ between databases.</p>
          {d.splits.train.concentration && <PartialNotice>In study training data, {(['S', 'F'] as const).map(c => <span key={c}> the largest single patient for {c} is <Num v={d.splits.train.concentration![c].top_patient} fmt="raw" src={`datareport#splits.train.concentration.${c}.top_patient`} /> with <Num v={d.splits.train.concentration![c].top_share} fmt="pct1" src={`datareport#splits.train.concentration.${c}.top_share`} /> of <Num v={d.splits.train.concentration![c].total} src={`datareport#splits.train.concentration.${c}.total`} />;</span>)} F beats come from only <Num v={d.splits.train.concentration.F.patients_with_any} src="datareport#splits.train.concentration.F.patients_with_any" /> patients.</PartialNotice>}
        </Card>
        <Card title="RR features are offline only"><p>These study models use RR-interval features: {RR_MODELS.map(x => MODEL_LABELS[x]).join('; ')}.</p>
          <p>The features use the <strong>following beat</strong> and the <strong>whole-record median</strong>, and the study computes them from reference beat positions. They cannot be computed in a live setting, and no beat-detector error is included.</p>
          <p><Link to="/comparison">Study results</Link></p></Card>
      </>
    }}</Load>
    <StudySection q={study} />
    <Card title="Other limits"><ul><li>Window-level results do not describe a patient or a recording.</li><li>XAI maps describe model behaviour, not clinical cause.</li><li>Softmax scores are not clinical certainty.</li><li>The held-out MIT-BIH test set had been scored for several variants before; see the split panel on Model Performance.</li></ul></Card>
  </>
}
