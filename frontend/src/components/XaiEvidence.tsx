import { useApi } from '../api/useApi'
import { Num } from './Num'
import { Empty, Load, PartialNotice } from './states'
import { Card, KeyValues, Table } from './ui'

type Advantage = { mean: number; ci95: [number, number]; beats_random: boolean; worse_than_random: boolean }
type Corr = { mean_spearman: number; mean_abs_spearman: number; n_valid: number; n_degenerate: number }
type Evaluation = {
  available: boolean; message?: string; stale: string[]
  data: {
    settings: { ig_steps: number; n_sanity: number }
    sample: { n: number; strategy: string; true_label_agreement_by_predicted_class: Record<string, number> }
    ig_completeness: { steps_in_this_evaluation: number; steps_used_by_app: number; all_windows: Record<string, number> }
    faithfulness: { note: string; variants: Record<string, { better: string; advantage_over_random: Record<'gradcam' | 'ig', Advantage> }> }
    randomisation_sanity: { n_windows: number; ig_steps: number; full_reinitialisation: (Record<'gradcam' | 'ig_signed' | 'ig_absolute', Corr> & { seed: number })[] }
    interpretation_limit: string
  }
}
const VARIANTS: Record<string, string> = { deletion_zero: 'Deletion (zero)', deletion_blur: 'Deletion (blur)', deletion_interp: 'Deletion (interpolate)', insertion_zero: 'Insertion (zero)', insertion_blur: 'Insertion (blur)' }

export function XaiEvidence() {
  const q = useApi<Evaluation>('/model/xai-evaluation')
  return <Card title="How far these maps can be trusted (offline evaluation)"><Load q={q}>{e => {
    if (!e.available) return <Empty title={e.message ?? 'XAI evaluation has not been generated.'} />
    const d = e.data
    return <div className="stack">
      {e.stale.map(s => <PartialNotice key={s}>This evaluation may be out of date: {s}.</PartialNotice>)}
      <p>Sample: <Num v={d.sample.n} src="xai#sample.n" /> windows ({d.sample.strategy}). Share of those predictions that were correct:{' '}
        {Object.keys(d.sample.true_label_agreement_by_predicted_class).map(c => <span key={c}> {c} <Num v={d.sample.true_label_agreement_by_predicted_class[c]} fmt="pct1" src={`xai#sample.true_label_agreement_by_predicted_class.${c}`} /></span>)}.
        {' '}Per-class results therefore often describe wrong predictions.</p>
      <Table caption="Faithfulness: advantage in area-under-curve over smoothed random controls (positive is better than random), with 95% CI" head={['Test', 'Grad-CAM', 'Integrated Gradients']}
        rows={Object.entries(d.faithfulness.variants).map(([k, v]) => [VARIANTS[k] ?? k, ...(['gradcam', 'ig'] as const).map(m => {
          const a = v.advantage_over_random[m], base = `xai#faithfulness.variants.${k}.advantage_over_random.${m}`
          return <span key={m}><Num v={a.mean} fmt="f3" src={`${base}.mean`} /> [<Num v={a.ci95[0]} fmt="f3" src={`${base}.ci95.0`} />, <Num v={a.ci95[1]} fmt="f3" src={`${base}.ci95.1`} />] {a.beats_random ? 'beats random' : a.worse_than_random ? 'worse than random' : 'not different from random'}</span>
        })])} />
      <Table caption={`Weight randomisation: mean rank correlation with the trained model’s maps on ${d.randomisation_sanity.n_windows} windows (near 0 is expected of a method that depends on the weights). Small sample.`}
        head={['Re-initialisation seed', 'Grad-CAM', 'Signed IG', 'Absolute IG', 'Degenerate Grad-CAM maps']}
        rows={d.randomisation_sanity.full_reinitialisation.map((r, i) => [r.seed,
          ...(['gradcam', 'ig_signed', 'ig_absolute'] as const).map(m => <Num key={m} v={r[m].mean_spearman} fmt="f3" src={`xai#randomisation_sanity.full_reinitialisation.${i}.${m}.mean_spearman`} />),
          <Num key="d" v={r.gradcam.n_degenerate} src={`xai#randomisation_sanity.full_reinitialisation.${i}.gradcam.n_degenerate`} />])} />
      <KeyValues rows={[
        ['IG steps in this evaluation', <Num v={d.ig_completeness.steps_in_this_evaluation} src="xai#ig_completeness.steps_in_this_evaluation" />],
        ['IG steps used by this app', <Num v={d.ig_completeness.steps_used_by_app} src="xai#ig_completeness.steps_used_by_app" />],
        ['Median completeness error at evaluation steps', <Num v={d.ig_completeness.all_windows.relative_error_median} fmt="pct1" src="xai#ig_completeness.all_windows.relative_error_median" />],
        ['Windows above 5% error at evaluation steps', <Num v={d.ig_completeness.all_windows.fraction_above_5pct} fmt="pct1" src="xai#ig_completeness.all_windows.fraction_above_5pct" />],
      ]} />
      <p className="notice">The faithfulness and randomisation tests ranked samples using lower-step IG than this app uses, so they do not test the exact maps shown above. Note: {d.faithfulness.note}. {d.interpretation_limit}</p>
    </div>
  }}</Load></Card>
}
