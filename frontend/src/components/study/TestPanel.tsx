import type { FinalData, Study, TestSet } from '../../api/study'
import { INCART_NOTICE } from '../../theme/theme'
import { Num } from '../Num'
import { Empty, ErrorState, PartialNotice } from '../states'
import { Table } from '../ui'
import { modelName, ordered } from './names'
import { PRIMARY_NOTE, Role } from './Role'
import { ReviewTable } from './ReviewTable'
import { ScoreTable } from './ScoreTable'

function SetTitle({ set, entry, primary }: { set: string; entry: TestSet; primary: string }) {
  const n = <Num v={entry[primary].n_patients} src={`study#final.data.sets.${set}.${primary}.n_patients`} />
  if (set === 'ds2_22_records') return <>MIT-BIH DS2, all records ({n} records)</>
  if (set === 'ds2_without_202') return <>MIT-BIH DS2 without record 202 ({n} records): the fair comparison with the deployed model</>
  if (set === 'incart_heldout') return <>INCART held-out patients: {n} unseen patients</>
  return <>{set}</>
}

const INTERVALS = [['macro_f1', 'Macro-F1'], ['S_recall', 'S recall'], ['F_recall', 'F recall']] as const

function Intervals({ entry, primary, base }: { entry: TestSet; primary: string; base: string }) {
  const row = (m: string) => [modelName(m), ...INTERVALS.map(([k]) => {
    const ci = entry[m]?.bootstrap_ci95_patients?.[k]
    return ci ? <span key={k}><Num v={ci[0]} fmt="f3" src={`${base}.${m}.bootstrap_ci95_patients.${k}.0`} /> to <Num v={ci[1]} fmt="f3" src={`${base}.${m}.bootstrap_ci95_patients.${k}.1`} /></span> : '—'
  })]
  return <Table caption="95% CI from a bootstrap over patients" head={['Model', ...INTERVALS.map(([, label]) => label)]} rows={[row(primary), row('deployed_model'),
    ['Majority class (always N)', ...INTERVALS.map(() => 'no interval computed')]]} />
}

function Sets({ study, d, compact }: { study: Study; d: FinalData; compact: boolean }) {
  return <>{Object.entries(d.sets).map(([set, entry]) => {
    const base = `study#final.data.sets.${set}`
    return <div key={set} data-testid={`testset-${set}`} className="stack">
      <h3><SetTitle set={set} entry={entry} primary={d.primary} /></h3>
      {set.startsWith('incart') && <p className="notice">{INCART_NOTICE}</p>}
      {set === 'ds2_22_records' && <p className="notice">Not a fair comparison with the deployed model: record 202 drove its early stopping. Its row is shown for completeness.</p>}
      <ScoreTable models={compact ? [d.primary, 'deployed_model'] : [...ordered(study.models, d.primary), 'deployed_model']} primary={d.primary} rows={entry}
        majority={entry.majority_class} majoritySource={`${base}.majority_class`} source={m => `${base}.${m}`} />
      {!compact && <><Intervals entry={entry} primary={d.primary} base={base} /><ReviewTable entry={entry} primary={d.primary} base={base} /></>}
    </div>
  })}</>
}

export function TestPanel({ study, compact = false }: { study: Study; compact?: boolean }) {
  const a = study.final
  if (!a.available || !a.data) return a.error ? <ErrorState title="Final report rejected" error={{ status: null, message: a.message ?? '' }} /> : <Empty title={a.message ?? 'Final evaluation not run yet.'} />
  const d = a.data, p = study.provenance
  const failed = p && !(p.primary_matches && p.selection_hash_matches && p.selection_dated_before_test)
  return <div className="stack">
    <p className="muted">One evaluation within this study, not a first look: DS2 had been scored before this study (earlier variants and an audit).</p>
    <p><Role model={d.primary} primary={d.primary} /> {modelName(d.primary)}, {PRIMARY_NOTE}. All other study models are <Role model="" primary={d.primary} />. Rows are in a fixed order and are never ranked by test score.</p>
    {failed && <PartialNotice><strong>Provenance check failed.</strong> The files do not show that the primary model was frozen before the test: primary name {p.primary_matches ? 'matches' : 'differs between files'}; selection file hash {p.selection_hash_matches ? 'matches' : 'does not match'} the one in the final report; selection {p.selection_dated_before_test ? 'is' : 'is not'} dated before the test was opened.</PartialNotice>}
    {p && !failed && <p className="small muted">Checked now from the files: the same primary is named in all three; the selection file’s hash matches the one the final report recorded; its recorded date is earlier than the date the test was opened.</p>}
    <Sets study={study} d={d} compact={compact} />
  </div>
}
