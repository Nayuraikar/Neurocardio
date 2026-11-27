import { MODEL_LABELS, RR_MODELS, type Majority, type Study, type Summary } from '../api/study'
import { useApi } from '../api/useApi'
import { Num } from '../components/Num'
import { Empty, ErrorState, Load, PartialNotice } from '../components/states'
import { Card, PageHead, Table } from '../components/ui'
import { INCART_NOTICE, REVIEW_BOUND_NOTICE } from '../theme/theme'

const SET_TITLES: Record<string, string> = {
  ds2_22_records: 'MIT-BIH DS2, all records', ds2_without_202: 'MIT-BIH DS2 without record 202 (the only DS2 set comparable with the app’s model)',
  incart_heldout: 'INCART, unseen patients',
}

/** One scored set. The majority-class row is always in the same table as any accuracy. */
function Scores({ rows, majority, source, majoritySource, models, primary }: {
  rows: Record<string, Summary | undefined>; majority: Majority; source: (m: string) => string; majoritySource: string; models: string[]; primary: string
}) {
  const cell = (m: string, path: string, v: number) => <Num v={v} fmt="f3" src={`${source(m)}.${path}`} />
  return <Table head={['Model', 'Macro-F1', 'S recall', 'F recall', 'S precision', 'F precision', 'V recall', 'Accuracy']} rows={[
    ...models.map(m => {
      const s = rows[m]
      const name = <span key="n">{MODEL_LABELS[m] ?? (m === 'deployed_model' ? 'App’s deployed model' : m)}{m === primary && <span className="tag"> primary</span>}</span>
      if (!s) return [name, <span key="x" className="muted">not scored on this set</span>, '', '', '', '', '', '']
      return [name, cell(m, 'macro_f1.mean', s.macro_f1.mean), cell(m, 'per_class.S.recall.mean', s.per_class.S.recall.mean), cell(m, 'per_class.F.recall.mean', s.per_class.F.recall.mean),
        cell(m, 'per_class.S.precision.mean', s.per_class.S.precision.mean), cell(m, 'per_class.F.precision.mean', s.per_class.F.precision.mean), cell(m, 'per_class.V.recall.mean', s.per_class.V.recall.mean), cell(m, 'accuracy.mean', s.accuracy.mean)]
    }),
    [<strong key="m">Majority class (always N)</strong>, <Num key="f" v={majority.macro_f1} fmt="f3" src={`${majoritySource}.macro_f1`} />, '0', '0', '—', '—', '0',
      <Num key="a" v={majority.accuracy} fmt="f3" src={`${majoritySource}.accuracy`} />],
  ]} />
}

function Validation({ study }: { study: Study }) {
  const a = study.validation
  if (!a.available || !a.data) return a.error ? <ErrorState title="Validation report rejected" error={{ status: null, message: a.message ?? '' }} /> : <Empty title={a.message ?? 'Study validation not run yet.'} />
  const d = a.data
  return <>
    <p className="muted">Validation data also drove early stopping and model selection, so these figures are optimistic. They are means over seeds.</p>
    <Scores models={study.models} primary={d.primary} majority={d.majority_class} majoritySource="study#validation.data.majority_class"
      rows={Object.fromEntries(Object.entries(d.results).map(([k, v]) => [k, v.chosen]))} source={m => `study#validation.data.results.${m}.chosen`} />
  </>
}

function Final({ study }: { study: Study }) {
  const a = study.final
  if (!a.available || !a.data) return a.error ? <ErrorState title="Final report rejected" error={{ status: null, message: a.message ?? '' }} /> : <Empty title={a.message ?? 'Final evaluation not run yet.'} />
  const d = a.data
  return <div className="stack">
    <p className="muted">DS2 had been scored before this study (earlier variants and an audit). This is one evaluation within this study, not a first look.</p>
    {Object.entries(d.sets).map(([set, entry]) => {
      const primary = entry[d.primary]
      const review = primary?.review
      return <div key={set}>
        <h3>{SET_TITLES[set] ?? set}</h3>
        {set.startsWith('incart') && <p className="notice">{INCART_NOTICE}</p>}
        {set === 'ds2_22_records' && <p className="notice">The app’s model used record 202 during training selection, so do not compare it on this set.</p>}
        <Scores models={[...study.models, 'deployed_model']} primary={d.primary} rows={entry} majority={entry.majority_class}
          majoritySource={`study#final.data.sets.${set}.majority_class`} source={m => `study#final.data.sets.${set}.${m}`} />
        {review && <>
          <Table caption="Selective review, primary model" head={['Review budget', 'Class', 'Model recall', 'After review (upper bound)', 'After random review']}
            rows={Object.entries(review).flatMap(([budget, st]) => (['S', 'F'] as const).filter(c => st[c]).map(c => {
              const base = `study#final.data.sets.${set}.${d.primary}.review.${budget.replace('.', '\\.')}.${c}`
              return [`${Number(budget) * 100}%`, c, <Num key="a" v={st[c]!.model_recall_all_beats} fmt="f3" src={`${base}.model_recall_all_beats`} />,
                <Num key="b" v={st[c]!.recall_after_perfect_review} fmt="f3" src={`${base}.recall_after_perfect_review`} />, <Num key="c" v={st[c]!.recall_after_random_review} fmt="f3" src={`${base}.recall_after_random_review`} />]
            }))} />
          <p className="notice">{REVIEW_BOUND_NOTICE}</p>
        </>}
      </div>
    })}
  </div>
}

export default function Comparison() {
  const q = useApi<Study>('/study')
  return <><PageHead title="Model Comparison Lab" sub="Study models trained on three databases. Lead with macro-F1 and S/F recall; validation and test are separate." />
    <PartialNotice>S and F are the hard classes in this problem. Compare their recall and precision before reading any accuracy.</PartialNotice>
    <Load q={q} lines={8}>{s => <>
      <Card title="Validation"><Validation study={s} /></Card>
      <Card title="Test"><Final study={s} /></Card>
      <Card title="What is compared"><Table head={['Model', 'ID', 'Inputs']} rows={s.models.map(m => [MODEL_LABELS[m] ?? m, <code key="c">{m}</code>,
        m === 'stacked_ensemble' ? 'Combines the other study models, including RR models (offline only)' : RR_MODELS.includes(m) ? 'RR-interval features: needs the following beat and the whole-record median (offline only)' : 'Waveform window only'])} />
        <p className="notice">The app does not run any study model. None of these scores describes the classifications shown on your recordings.</p></Card>
    </>}</Load></>
}
