import type { FinalData, Gain, GainKey, Study } from '../../api/study'
import { REVIEW_BOUND_NOTICE } from '../../theme/theme'
import { Num } from '../Num'
import { Table } from '../ui'
import { modelName, SHORT_SETS } from './names'

const METRICS: [GainKey, string, string][] = [['macro_f1', 'Macro-F1', 'macro_f1'], ['S_f1', 'S F1', 'per_class.S.f1'], ['F_f1', 'F F1', 'per_class.F.f1'],
  ['S_recall', 'S recall', 'per_class.S.recall'], ['F_recall', 'F recall', 'per_class.F.recall']]
/** The three outcomes, from the artifact's own flags. */
const verdict = (g: Gain) => g.meaningful ? 'improved' : g.gain < 0 ? 'lower; not a meaningful improvement' : 'did NOT meaningfully improve'
const at = (set: string) => `study#final.data.sets.${set}`
const pick = (o: unknown, path: string) => path.split('.').reduce<unknown>((x, k) => (x as Record<string, unknown>)[k], o)

/** Every sentence here is assembled from the study artifacts: verdicts from their flags, numbers from their fields. */
export function Findings({ study }: { study: Study }) {
  const d: FinalData | null = study.final.data
  if (!study.final.available || !d) return null
  const sets = Object.entries(d.sets), p = d.primary
  const headline = sets.filter(([, e]) => e.primary_vs_deployed.headline_comparison)
  const minority = METRICS.slice(1)
  const improvedMinority = headline.flatMap(([set, e]) => minority.filter(([k]) => e.primary_vs_deployed[k].meaningful).map(([, label]) => `${label} on ${SHORT_SETS[set] ?? set}`))
  const lower = headline.flatMap(([set, e]) => METRICS.filter(([k]) => e.primary_vs_deployed[k].gain < 0).map(([key, label]) => ({ set, key, label, gain: e.primary_vs_deployed[key].gain })))
  const macro = headline.map(([set, e]) => `${e.primary_vs_deployed.macro_f1.meaningful ? 'improved' : 'did not meaningfully improve'} on ${SHORT_SETS[set] ?? set}`)
  const validationRecall = study.validation.data?.results[p]?.chosen.per_class.S.recall.mean
  const concentration = study.data_report?.splits
  const share = (split: string, c: 'S' | 'F') => {
    const x = concentration?.[split]?.concentration?.[c]
    if (!x) return null
    const src = `datareport#splits.${split}.concentration.${c}`
    return <><Num v={x.top_patient} fmt="raw" src={`${src}.top_patient`} /> holds <Num v={x.top_share} fmt="pct1" src={`${src}.top_share`} /> of the <Num v={x.total} src={`${src}.total`} /> {c} beats</>
  }
  return <div className="stack" data-testid="findings">
    <ul>
      <li data-testid="finding-macro"><strong>Macro-F1 against the deployed model, under the rule declared before the test:</strong> {macro.join('; ')}.
        {' '}The comparison is not like for like: the study models trained on more data from more databases, so a gain mixes the effect of data with the effect of method.</li>
      <li data-testid="finding-minority"><strong>S and F:</strong>{' '}{improvedMinority.length
        ? <>meaningful improvement under the rule only for {improvedMinority.join(', ')}; every other S and F recall and F1 comparison did not meaningfully improve.</>
        : <>S and F recall and F1 did NOT meaningfully improve over the deployed model on {headline.map(([s]) => SHORT_SETS[s] ?? s).join(' or ')}.</>}
        {lower.map(x => <span key={`${x.set}.${x.key}`}> <strong>Lower than the deployed model:</strong> {x.label} on {SHORT_SETS[x.set] ?? x.set}, gain <Num v={x.gain} fmt="f3" src={`${at(x.set)}.primary_vs_deployed.${x.key}.gain`} />.</span>)}</li>
      <li data-testid="finding-f-recall"><strong>F recall of the primary model:</strong>{' '}
        {sets.map(([set, e], i) => {
          const ci = e[p].bootstrap_ci95_patients?.F_recall
          return <span key={set}>{i ? '; ' : ''}{SHORT_SETS[set] ?? set} <Num v={e[p].per_class.F.recall.mean} fmt="f3" src={`${at(set)}.${p}.per_class.F.recall.mean`} />
            {ci && <>, 95% CI <Num v={ci[0]} fmt="f3" src={`${at(set)}.${p}.bootstrap_ci95_patients.F_recall.0`} /> to <Num v={ci[1]} fmt="f3" src={`${at(set)}.${p}.bootstrap_ci95_patients.F_recall.1`} /> ({ci[0] <= 0 ? 'includes zero' : 'excludes zero'})</>}</span>
        })}.</li>
      {validationRecall !== undefined && <li data-testid="finding-validation-gap"><strong>Validation against test, S recall:</strong> validation <Num v={validationRecall} fmt="f3" src={`study#validation.data.results.${p}.chosen.per_class.S.recall.mean`} />
        {sets.map(([set, e]) => <span key={set}>; {SHORT_SETS[set] ?? set} <Num v={e[p].per_class.S.recall.mean} fmt="f3" src={`${at(set)}.${p}.per_class.S.recall.mean`} /> (validation {validationRecall > e[p].per_class.S.recall.mean ? 'overstated it' : 'did not overstate it'})</span>)}.</li>}
      {concentration && <li data-testid="finding-concentration"><strong>S and F depend on a few patients.</strong> DS2: {share('test_ds2', 'S')}; {share('test_ds2', 'F')}. INCART test: {share('test_incart', 'S')}; {share('test_incart', 'F')}. A pooled S or F score largely describes how the model does on those patients.</li>}
      <li><strong>Review:</strong> {REVIEW_BOUND_NOTICE} It is shown beside random review of the same share.</li>
      <li><strong>DS2 is not untouched:</strong> it had been scored before this study. This is one evaluation within the study, not a first look.</li>
    </ul>
    <Table caption={`${modelName(p)} (PRIMARY) against the deployed model. “Improved” needs all of: gain above twice the larger seed SD, patient-bootstrap CI of the gain above zero, and a positive gain with any one patient left out.`}
      head={['Test set', 'Metric', 'Primary', 'Deployed', 'Majority class', 'Gain', '95% CI of gain', 'Smallest gain, one patient left out', 'Verdict']}
      rows={headline.flatMap(([set, e]) => METRICS.map(([key, label, path]) => {
        const g = e.primary_vs_deployed[key], base = `${at(set)}.primary_vs_deployed.${key}`
        return [SHORT_SETS[set] ?? set, label,
          <Num key="p" v={pick(e[p], `${path}.mean`)} fmt="f3" src={`${at(set)}.${p}.${path}.mean`} />,
          <Num key="d" v={pick(e.deployed_model, `${path}.mean`)} fmt="f3" src={`${at(set)}.deployed_model.${path}.mean`} />,
          <Num key="m" v={pick(e.majority_class, path)} fmt="f3" src={`${at(set)}.majority_class.${path}`} />,
          <Num key="g" v={g.gain} fmt="f3" src={`${base}.gain`} />,
          <span key="c"><Num v={g.bootstrap_ci95[0]} fmt="f3" src={`${base}.bootstrap_ci95.0`} /> to <Num v={g.bootstrap_ci95[1]} fmt="f3" src={`${base}.bootstrap_ci95.1`} /></span>,
          <Num key="l" v={g.leave_one_record_out_min_gain} fmt="f3" src={`${base}.leave_one_record_out_min_gain`} />,
          verdict(g)]
      }))} />
  </div>
}
