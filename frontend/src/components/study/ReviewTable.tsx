import type { Review, TestSet } from '../../api/study'
import { REVIEW_BOUND_NOTICE } from '../../theme/theme'
import { Num } from '../Num'
import { Table } from '../ui'
import { modelName } from './names'

/** Recall after review, always beside random review of the same share, with the deployed model and the baseline. */
export function ReviewTable({ entry, primary, base }: { entry: TestSet; primary: string; base: string }) {
  const blocks: [string, string, string, Review | undefined][] = [
    [primary, 'review', 'ranked within this set', entry[primary]?.review],
    [primary, 'review_with_validation_cutoff', 'fixed on validation', entry[primary]?.review_with_validation_cutoff],
    ['deployed_model', 'review', 'ranked within this set', entry.deployed_model?.review],
  ]
  const rows = blocks.flatMap(([model, key, cutoff, review]) => Object.entries(review ?? {}).flatMap(([budget, st]) =>
    (['S', 'F'] as const).filter(c => st[c]).map(c => {
      const at = `${base}.${model}.${key}.${budget.replace('.', '\\.')}`
      return [`${Number(budget) * 100}%`, cutoff, <Num key="sh" v={st.reviewed_share} fmt="pct1" src={`${at}.reviewed_share`} />, modelName(model), c,
        <Num key="a" v={st[c]!.model_recall_all_beats} fmt="f3" src={`${at}.${c}.model_recall_all_beats`} />,
        <Num key="b" v={st[c]!.recall_after_perfect_review} fmt="f3" src={`${at}.${c}.recall_after_perfect_review`} />,
        <Num key="c" v={st[c]!.recall_after_random_review} fmt="f3" src={`${at}.${c}.recall_after_random_review`} />]
    })))
  if (!rows.length) return null
  const baseline = (['S', 'F'] as const).map(c => ['—', '—', '—', 'Majority class (always N)', c,
    <Num key="a" v={entry.majority_class.per_class[c].recall} fmt="f3" src={`${base}.majority_class.per_class.${c}.recall`} />,
    'not defined (no confidence score)', 'not defined'])
  return <>
    <Table caption="Selective review. Budget is the intended share of beats sent to a reviewer; “reviewed” is the share actually sent."
      head={['Budget', 'Cut-off', 'Reviewed', 'Model', 'Class', 'Model recall', 'After review (upper bound)', 'After random review']} rows={[...rows, ...baseline]} />
    <p className="notice">{REVIEW_BOUND_NOTICE}</p>
  </>
}
