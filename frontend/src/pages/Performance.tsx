import { Link, useParams } from 'react-router-dom'
import type { ModelMetrics, SplitMetrics } from '../api/metrics'
import type { ModelConfig } from '../api/types'
import { useApi } from '../api/useApi'
import { ArchitectureDiagram } from '../components/ArchitectureDiagram'
import { ConfusionMatrix } from '../components/ConfusionMatrix'
import { CurveChart } from '../components/CurveChart'
import { HeadlineMetrics } from '../components/HeadlineMetrics'
import { Num } from '../components/Num'
import { SplitPanel } from '../components/SplitPanel'
import { Empty, Load, PartialNotice } from '../components/states'
import { Card, ClassBadge, KeyValues, PageHead, Table } from '../components/ui'
import { WeaknessNote } from '../components/WeaknessNote'
import { CLASS_CODES, CLASSES } from '../theme/theme'
import { METRIC_INFO } from './metricInfo'

const headline = (split: 'test' | 'validation', s: SplitMetrics) => <HeadlineMetrics macroF1={s.macro.f1} sRecall={s.per_class.S.recall} fRecall={s.per_class.F.recall}
  accuracy={s.accuracy} baseline={s.majority_class_baseline_accuracy} sources={{ macroF1: `metrics#${split}.macro.f1`, sRecall: `metrics#${split}.per_class.S.recall`,
    fRecall: `metrics#${split}.per_class.F.recall`, accuracy: `metrics#${split}.accuracy`, baseline: `metrics#${split}.majority_class_baseline_accuracy` }} />

const COLUMNS = ['precision', 'recall', 'specificity', 'f1', 'roc_auc', 'pr_auc'] as const

function PerClass({ split, s }: { split: 'test' | 'validation'; s: SplitMetrics }) {
  return <Table head={['Class', ...COLUMNS.map(c => <Link key={c} to={`/performance/${c}`}>{METRIC_INFO[c].name}</Link>), 'Support']} rows={[
    ...CLASS_CODES.map(c => [<ClassBadge key="c" code={c} long />, ...COLUMNS.map(k => <Num key={k} v={s.per_class[c][k]} fmt="f3" src={`metrics#${split}.per_class.${c}.${k}`} />),
      <Num key="s" v={s.per_class[c].support} src={`metrics#${split}.per_class.${c}.support`} />]),
    ...(['macro', 'weighted'] as const).map(avg => [<strong key="a">{avg === 'macro' ? 'Macro average' : 'Weighted average'}</strong>,
      ...COLUMNS.map(k => k === 'specificity' ? '—' : <Num key={k} v={s[avg][k]} fmt="f3" src={`metrics#${split}.${avg}.${k}`} />), '']),
  ]} />
}

function Curves({ s }: { s: SplitMetrics }) {
  return <div className="stack">
    <div style={{ maxWidth: 460 }}><CurveChart title="ROC, test" xLabel="False positive rate" yLabel="True positive rate" diagonal
      series={CLASS_CODES.map(c => ({ label: c, color: CLASSES[c].fill, x: s.roc_curves[c].fpr, y: s.roc_curves[c].tpr }))} /></div>
    <div className="grid-4">{CLASS_CODES.map(c => <CurveChart key={c} title={`Precision-recall ${c}, test`} xLabel="Recall" yLabel="Precision"
      series={[{ label: c, color: CLASSES[c].fill, x: s.pr_curves[c].recall, y: s.pr_curves[c].precision },
        { label: 'prevalence', legend: <>prevalence <Num v={s.per_class[c].prevalence} fmt="pct1" src={`metrics#test.per_class.${c}.prevalence`} /></>, color: '#888', x: [0, 1], y: [s.per_class[c].prevalence, s.per_class[c].prevalence], dash: true }]} />)}</div>
  </div>
}

function History({ m }: { m: ModelMetrics }) {
  const h = m.metrics.training.history, x = h.map(r => r.epoch), domain: [number, number] = [x[0], x[x.length - 1]]
  const lossMax = Math.ceil(Math.max(...h.flatMap(r => [r.loss, r.val_loss])))
  return <Card title="Training history of the app’s model">
    <div className="grid-3">
      <CurveChart title="Loss" xLabel="Epoch" yLabel="Loss" xDomain={domain} yMax={lossMax} series={[{ label: 'training', color: '#2b3a1e', x, y: h.map(r => r.loss) }, { label: 'validation', color: '#1f5f8b', x, y: h.map(r => r.val_loss) }]} />
      <CurveChart title="Validation accuracy" xLabel="Epoch" yLabel="Accuracy" xDomain={domain} series={[{ label: 'validation', color: '#1f5f8b', x, y: h.map(r => r.val_accuracy) },
        { label: 'validation majority-class baseline', color: '#888', dash: true, x: domain, y: [m.metrics.validation.majority_class_baseline_accuracy, m.metrics.validation.majority_class_baseline_accuracy] }]} />
      <CurveChart title="Validation macro-F1" xLabel="Epoch" yLabel="Macro-F1" xDomain={domain} series={[{ label: 'validation', color: '#1f5f8b', x, y: h.map(r => r.val_macro_f1) }]} />
    </div>
    <p className="notice">Training data were augmented, so training and validation curves are not on the same distribution. No training macro-F1 was recorded, and no verified majority-class baseline exists for the augmented training data, so training accuracy is not plotted.</p>
  </Card>
}

function Body({ m }: { m: ModelMetrics }) {
  const d = m.metrics, ci = d.test_bootstrap.ci95
  const config = useApi<ModelConfig>('/model/config')
  const op = d.review_flag_policy.operating_points[d.review_flag_policy.app_uses]
  return <>
    {m.stale.map(s => <PartialNotice key={s}>These metrics may be out of date: {s}.</PartialNotice>)}
    <Card title="Held-out test (MIT-BIH, window level)" aside={<span className="muted"><Num v={d.test.n_windows} src="metrics#test.n_windows" /> windows</span>}>
      {headline('test', d.test)}
      <p className="small muted">95% confidence intervals from a bootstrap over {d.test_bootstrap.unit}s (<Num v={d.test_bootstrap.resamples} src="metrics#test_bootstrap.resamples" /> resamples). Windows within a record are not independent.</p>
      <Table
        head={['Macro-F1', 'S recall', 'F recall', 'V recall']} rows={[['macro_f1', 'S_recall', 'F_recall', 'V_recall'].map(k => <span key={k}>
          <Num v={ci[k][0]} fmt="f3" src={`metrics#test_bootstrap.ci95.${k}.0`} /> to <Num v={ci[k][1]} fmt="f3" src={`metrics#test_bootstrap.ci95.${k}.1`} /></span>)]} />
    </Card>
    <Card title="Per class, test"><PerClass split="test" s={d.test} /></Card>
    <div className="grid-2">
      <Card title="Confusion matrix, test"><ConfusionMatrix counts={d.test.confusion_matrix.counts} source="metrics#test.confusion_matrix.counts" /></Card>
      <Card title="Calibration and review flag, test"><KeyValues rows={[
        ['ECE, raw softmax', <Num v={d.calibration.test_raw.ece_15} fmt="f4" src="metrics#calibration.test_raw.ece_15" />],
        ['ECE, temperature-scaled', <Num v={d.calibration.test_temperature_scaled.ece_15} fmt="f4" src="metrics#calibration.test_temperature_scaled.ece_15" />],
        ['Windows flagged', <Num v={op.test.review_rate} fmt="pct1" src={`metrics#review_flag_policy.operating_points.${d.review_flag_policy.app_uses}.test.review_rate`} />],
        ['Error rate among unflagged', <Num v={op.test.error_rate_on_unflagged} fmt="pct1" src={`metrics#review_flag_policy.operating_points.${d.review_flag_policy.app_uses}.test.error_rate_on_unflagged`} />],
      ]} /><p className="notice">The rule was chosen on validation data for a lower error rate than it reached here.</p></Card>
    </div>
    <Card title="ROC and precision-recall"><Curves s={d.test} /></Card>
    <Card title="Validation (used for model selection; not a final result)">{headline('validation', d.validation)}<PerClass split="validation" s={d.validation} /></Card>
    <History m={m} />
    <SplitPanel m={m} />
    <Card title="Model" aside={<Link to="/model-card">Model card</Link>}>
      <Load q={config}>{c => <><KeyValues rows={[
        ['Architecture', <span data-source="config#architecture">{c.architecture}</span>], ['Input', <span data-source="config#input_shape">{c.input_shape}</span>],
        ['Normalization', <span data-source="config#normalization">{c.normalization}</span>], ['Parameters', <Num v={c.parameter_count} src="config#parameter_count" />],
        ['Version', <span data-source="config#model_version">{c.model_version}</span>], ['Decision rule', d.training.decision_rule],
        ['Loss (as recorded, unverified)', d.training.loss_as_recorded_in_config], ['Class weighting and augmentation', 'recorded in config; unverified (no training code)'],
        ['Evaluated', <Num v={new Date(d.evaluated_at_utc).toLocaleString()} fmt="raw" src="datetime:metrics#evaluated_at_utc" />]]} />
        <h3>Layers of the loaded model</h3><ArchitectureDiagram config={c} /></>}</Load>
    </Card>
  </>
}

export default function Performance() {
  const q = useApi<ModelMetrics>('/model/metrics')
  return <><PageHead title="Model Performance" sub="The model this app runs. Window-level results on retrospective MIT-BIH data." />
    <WeaknessNote link={false} />
    <Load q={q} lines={10}>{m => m.available ? <Body m={m} /> : <Empty title={m.message ?? 'Not available'}>{m.problem}</Empty>}</Load></>
}

export function MetricDetail() {
  const { metric = '' } = useParams()
  const info = METRIC_INFO[metric]
  const q = useApi<ModelMetrics>('/model/metrics')
  if (!info) return <Empty title="Unknown metric"><Link to="/performance">Back to Model Performance</Link></Empty>
  const key = metric as (typeof COLUMNS)[number]
  return <><PageHead title={info.name} sub={info.definition} />
    <Card><p>{info.reading}</p></Card>
    <WeaknessNote />
    <Load q={q}>{m => !m.available ? <Empty title={m.message ?? 'Not available'} /> : <Card title="By class">
      <Table head={['Class', 'Test (held out)', 'Validation (selection)', 'Test support']} rows={CLASS_CODES.map(c => [<ClassBadge key="c" code={c} long />,
        <Num key="t" v={m.metrics.test.per_class[c][key]} fmt="f3" src={`metrics#test.per_class.${c}.${key}`} />,
        <Num key="v" v={m.metrics.validation.per_class[c][key]} fmt="f3" src={`metrics#validation.per_class.${c}.${key}`} />,
        <Num key="s" v={m.metrics.test.per_class[c].support} src={`metrics#test.per_class.${c}.support`} />])} /></Card>}</Load></>
}
