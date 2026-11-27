import type { ModelConfig } from '../api/types'

/** The real processing order, with the step the server reports. */
export function PipelineSteps({ pipeline, stage, status }: { pipeline: ModelConfig['pipeline']; stage?: string; status: 'idle' | 'running' | 'completed' | 'failed' }) {
  const current = pipeline.findIndex(p => p.key === stage)
  return <ol className="steps">{pipeline.map((p, i) => {
    const state = status === 'completed' ? 'done' : status === 'idle' ? '' : i < current ? 'done' : i === current ? (status === 'failed' ? 'failed' : 'active') : ''
    return <li key={p.key} className={state} aria-current={state === 'active' ? 'step' : undefined}>
      <b>{state === 'done' ? '✓' : state === 'failed' ? '!' : i + 1}</b>
      <div><strong>{p.label}</strong>{state === 'active' && ' · running'}{state === 'failed' && ' · failed here'}<div className="small muted">{p.detail}</div></div>
    </li>
  })}</ol>
}
