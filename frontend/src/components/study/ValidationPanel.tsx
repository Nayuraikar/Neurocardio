import type { Study } from '../../api/study'
import { Empty, ErrorState } from '../states'
import { ordered } from './names'
import { PRIMARY_NOTE, Role } from './Role'
import { ScoreTable } from './ScoreTable'

/** compact: the primary model and the baseline only. */
export function ValidationPanel({ study, compact = false }: { study: Study; compact?: boolean }) {
  const a = study.validation
  if (!a.available || !a.data) return a.error ? <ErrorState title="Validation report rejected" error={{ status: null, message: a.message ?? '' }} /> : <Empty title={a.message ?? 'Study validation not run yet.'} />
  const d = a.data
  return <>
    <p className="muted">Study validation data, not a test result. It also drove early stopping and model selection, so these figures are optimistic. Means over training seeds. The deployed model was not scored on this data.</p>
    <p><Role model={d.primary} primary={d.primary} /> {PRIMARY_NOTE}. Rows are in a fixed order, not ranked.</p>
    <ScoreTable models={compact ? [d.primary] : ordered(study.models, d.primary)} primary={d.primary} majority={d.majority_class} majoritySource="study#validation.data.majority_class"
      rows={Object.fromEntries(Object.entries(d.results).map(([k, v]) => [k, v.chosen]))} source={m => `study#validation.data.results.${m}.chosen`} />
  </>
}
