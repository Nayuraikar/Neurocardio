import { Link } from 'react-router-dom'
import type { ModelMetrics } from '../api/metrics'
import { useApi } from '../api/useApi'
import { Num } from './Num'
import { PartialNotice } from './states'

/** S and F recall on the held-out test set, read from the metrics artifact. */
export function WeaknessNote({ link = true }: { link?: boolean }) {
  const q = useApi<ModelMetrics>('/model/metrics')
  if (!q.data?.available) return <PartialNotice>S and F are weak classes for this model. Its evaluation figures could not be loaded.</PartialNotice>
  const t = q.data.metrics.test
  return <PartialNotice>
    <strong>S and F weakness (held-out test):</strong>{' '}
    {(['S', 'F'] as const).map((c, i) => <span key={c}>{i ? '; ' : ''}{c} recall <Num v={t.per_class[c].recall} fmt="f3" src={`metrics#test.per_class.${c}.recall`} /> on <Num v={t.per_class[c].support} src={`metrics#test.per_class.${c}.support`} /> windows, precision <Num v={t.per_class[c].precision} fmt="f3" src={`metrics#test.per_class.${c}.precision`} /></span>)}.
    {' '}The model misses most S and F windows, so a low S or F count does not show that none are present.
    {link && <> <Link to="/performance">Details</Link></>}
  </PartialNotice>
}
