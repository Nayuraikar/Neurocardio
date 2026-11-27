import { useCallback, useEffect, useState, type ReactNode } from 'react'
import { useNavigate } from 'react-router-dom'
import type { ModelMetrics } from '../api/metrics'
import type { Study } from '../api/study'
import type { ModelConfig } from '../api/types'
import { useApi } from '../api/useApi'
import { ClassDistribution } from '../components/ClassDistribution'
import { ErrorBoundary } from '../components/ErrorBoundary'
import { ConfusionMatrix } from '../components/ConfusionMatrix'
import { HeadlineMetrics } from '../components/HeadlineMetrics'
import { PipelineSteps } from '../components/PipelineSteps'
import { Empty, Load } from '../components/states'
import { WeaknessNote } from '../components/WeaknessNote'
import { WindowStages } from '../components/WindowStages'
import { useBeats } from '../api/useBeats'
import { useRecording } from '../context/RecordingContext'
import { NOTICE, XAI_NOTICE } from '../theme/theme'

function CurrentRecording() {
  const { analysis, record } = useRecording()
  const beats = useBeats(analysis?.status === 'completed' ? analysis.id : null)
  const { windowIndex } = useRecording()
  if (!analysis || !record || analysis.status !== 'completed') return <Empty title="No recording selected">Select an analysis in the workspace before presenting to include it here.</Empty>
  return <div className="stack"><p>{record.filename} · model class counts for its 2 s windows</p>
    <ClassDistribution counts={analysis.summary.class_counts ?? {}} source="analysis#summary.class_counts" />
    {record.model_split_by_name && record.model_split_by_name !== 'test' && <p className="state state-warn">This file name matches a {record.model_split_by_name} record of the model, so results here are optimistic.</p>}
    <Load q={beats}>{b => b.length ? <WindowStages record={record} sample={b[Math.min(windowIndex, b.length - 1)].sample_index} /> : <Empty title="No windows" />}</Load></div>
}

export default function Presentation() {
  const config = useApi<ModelConfig>('/model/config')
  const metrics = useApi<ModelMetrics>('/model/metrics')
  const study = useApi<Study>('/study')
  const [index, setIndex] = useState(0)
  const navigate = useNavigate()
  const m = metrics.data?.available ? metrics.data : null
  const withMetrics = (render: (x: ModelMetrics) => ReactNode) => <Load q={metrics}>{x => x.available ? <>{render(x)}</> : <Empty title={x.message ?? 'Model evaluation not available'} />}</Load>
  const slides: [string, ReactNode][] = [
    ['Interpretable AI-powered ECG analysis', <><p>{NOTICE}</p>{config.data && <p>{config.data.architecture} · input {config.data.input_shape} · {config.data.normalization} · classes {config.data.classes.join(', ')}</p>}</>],
    ['Processing', <Load q={config}>{c => <PipelineSteps pipeline={c.pipeline} status="idle" />}</Load>],
    ['Held-out test results', withMetrics(x => { const t = x.metrics.test; return <><HeadlineMetrics macroF1={t.macro.f1} sRecall={t.per_class.S.recall} fRecall={t.per_class.F.recall} accuracy={t.accuracy} baseline={t.majority_class_baseline_accuracy}
      sources={{ macroF1: 'metrics#test.macro.f1', sRecall: 'metrics#test.per_class.S.recall', fRecall: 'metrics#test.per_class.F.recall', accuracy: 'metrics#test.accuracy', baseline: 'metrics#test.majority_class_baseline_accuracy' }} /><p>MIT-BIH, window level, retrospective.</p></> })],
    ['Where it fails', <>{<WeaknessNote link={false} />}{m && <ConfusionMatrix counts={m.metrics.test.confusion_matrix.counts} source="metrics#test.confusion_matrix.counts" />}</>],
    ['One patient dominates fusion training', withMetrics(x => <>{x.minority_concentration.filter(r => r.split === 'train' && r.class === 'F').map(r => <p key="f">Record {r.top_record} supplies <strong>{(r.top_record_share * 100).toFixed(1)}%</strong> of the training F beats.</p>)}</>)],
    ['Current recording', <CurrentRecording />],
    ['Explanations', <><p>Integrated Gradients and Grad-CAM show which samples moved the model’s score.</p><p>{XAI_NOTICE}</p></>],
    ['Multi-database study', <Load q={study}>{s => <><p>{s.final.available ? 'A final evaluation report is available on the Model Comparison page.' : s.final.message}</p>{!s.validation.available && <p>{s.validation.message}</p>}<p>RR-feature models in the study are offline only.</p></>}</Load>],
    ['Limits', <ul><li>Window-level output; no diagnosis.</li><li>S and F recall are weak.</li><li>Retrospective, single-lead data.</li><li>Softmax is not clinical certainty.</li></ul>],
  ]
  const move = useCallback((d: number) => setIndex(i => Math.min(slides.length - 1, Math.max(0, i + d))), [slides.length])
  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      if (e.key === 'ArrowRight' || e.key === 'PageDown') move(1)
      else if (e.key === 'ArrowLeft' || e.key === 'PageUp') move(-1)
      else if (e.key === 'Escape') navigate('/')
    }
    window.addEventListener('keydown', key)
    return () => window.removeEventListener('keydown', key)
  }, [move, navigate])
  const [title, body] = slides[index]
  return <main className="present" aria-live="polite">
    <div className="row between" style={{ fontSize: 14 }}>
      <span className="muted">{index + 1} / {slides.length} · arrow keys move, Esc exits</span>
      <div className="row"><button type="button" disabled={index === 0} onClick={() => move(-1)}>Previous</button><button type="button" disabled={index === slides.length - 1} onClick={() => move(1)}>Next</button>
        <button type="button" onClick={() => void document.documentElement.requestFullscreen?.()}>Full screen</button><button type="button" onClick={() => navigate('/')}>Exit</button></div>
    </div>
    <h1>{title}</h1>
    <ErrorBoundary key={index}><div className="stack">{body}</div></ErrorBoundary>
    <p className="small muted" style={{ marginTop: 'auto' }}>{NOTICE}</p>
  </main>
}
