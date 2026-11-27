import { useState } from 'react'
import type { ModelMetrics } from '../api/metrics'
import { useApi } from '../api/useApi'
import type { Analysis, Beat } from '../api/types'
import { Num } from '../components/Num'
import { Load, PartialNotice } from '../components/states'
import { Card, ClassBadge, KeyValues, PageHead, ProbBars } from '../components/ui'
import { WindowSelect } from '../components/WindowSelect'
import { WindowTable } from '../components/WindowTable'
import { WithWindow } from '../components/WithWindow'
import { useRecording } from '../context/RecordingContext'
import { CLASS_CODES, SOFTMAX_NOTICE } from '../theme/theme'
import { temperatureScale, topTwo } from './temperature'

function Explorer({ analysis, beats, beat }: { analysis: Analysis; beats: Beat[]; beat: Beat }) {
  const [scaled, setScaled] = useState(false)
  const { setWindow } = useRecording()
  const metrics = useApi<ModelMetrics>('/model/metrics')
  const policy = analysis.summary.uncertainty_policy
  const hasPolicy = !!policy && policy.tau > 0
  const raw = beat.prediction.probabilities
  const probs = scaled && policy ? temperatureScale(raw, policy.temperature) : raw
  const t = topTwo(probs)
  const lowest = [...beats].sort((a, b) => a.prediction.margin - b.prediction.margin).slice(0, 100)
  return <>
    <WindowSelect beats={beats} beat={beat} />
    <div className="grid-2">
      <Card title="Class probabilities" aside={<div className="row" role="group" aria-label="Probability type">
        <button type="button" aria-pressed={!scaled} onClick={() => setScaled(false)}>Softmax</button>
        <button type="button" aria-pressed={scaled} disabled={!policy} onClick={() => setScaled(true)}>Temperature-scaled</button></div>}>
        <ProbBars probs={probs} />
        {scaled && policy && <p className="small muted">Temperature <span className="num">{policy.temperature}</span>, fitted on validation data only. Scaling changes the scores, never the predicted class.</p>}
        <p className="notice">{SOFTMAX_NOTICE}</p>
      </Card>
      <Card title="Uncertainty">
        <KeyValues rows={[
          ['Top class', <><ClassBadge code={CLASS_CODES[t.first]} /> <span className="num">{(t.top * 100).toFixed(2)}%</span></>],
          ['Second class', <><ClassBadge code={CLASS_CODES[t.second]} /> <span className="num">{(probs[t.second] * 100).toFixed(2)}%</span></>],
          ['Top-2 margin', <span className="num">{t.margin.toFixed(4)}</span>],
          ['Low-confidence flag', !hasPolicy ? 'No review rule was available' : beat.prediction.review_flag ? <span className="tag moderate">Flagged</span> : <span className="tag">Not flagged</span>],
          ['Review recommendation', beat.prediction.review_flag ? 'Send this window to a reviewer.' : 'The rule does not request review. This does not mean the class is correct.'],
        ]} />
        {hasPolicy && <p className="small muted">Rule used at analysis time ({policy!.name}): flag when the temperature-scaled top score is below <span className="num">{policy!.tau.toFixed(3)}</span>{policy!.margin > 0 && <> or the margin is below <span className="num">{policy!.margin}</span></>}. {policy!.label}.</p>}
      </Card>
    </div>
    <Load q={metrics}>{m => {
      if (!m.available) return <PartialNotice>{m.message}</PartialNotice>
      const p = m.metrics.review_flag_policy, op = p.operating_points[p.app_uses]
      return <Card title="How the review rule did on the held-out test set">
        <KeyValues rows={[
          ['Windows flagged', <Num v={op.test.review_rate} fmt="pct1" src={`metrics#review_flag_policy.operating_points.${p.app_uses}.test.review_rate`} />],
          ['Error rate among unflagged', <Num v={op.test.error_rate_on_unflagged} fmt="pct1" src={`metrics#review_flag_policy.operating_points.${p.app_uses}.test.error_rate_on_unflagged`} />],
          ['Share of all errors flagged', <Num v={op.test.fraction_of_all_errors_flagged} fmt="pct1" src={`metrics#review_flag_policy.operating_points.${p.app_uses}.test.fraction_of_all_errors_flagged`} />],
          ['True F windows flagged', <Num v={op.test.review_rate_by_true_class.F} fmt="pct1" src={`metrics#review_flag_policy.operating_points.${p.app_uses}.test.review_rate_by_true_class.F`} />],
          ['Calibration error (ECE), raw → scaled', <><Num v={m.metrics.calibration.test_raw.ece_15} fmt="f4" src="metrics#calibration.test_raw.ece_15" /> → <Num v={m.metrics.calibration.test_temperature_scaled.ece_15} fmt="f4" src="metrics#calibration.test_temperature_scaled.ece_15" /></>],
        ]} />
        <p className="notice">Unflagged windows still contain errors. The rule was set on validation data to a lower error target than it reached on test.</p>
      </Card>
    }}</Load>
    <Card title="Smallest top-2 margins in this recording (100 shown)"><WindowTable beats={lowest} selectedId={beat.id} onSelect={b => setWindow(b.beat_index)} /></Card>
  </>
}

export default function Confidence() {
  return <><PageHead title="Confidence & Uncertainty" sub="How sure the model score is for each window, and what the review rule does with it." />
    <WithWindow>{({ analysis, beats, beat }) => <Explorer analysis={analysis} beats={beats} beat={beat} />}</WithWindow></>
}
