import type { ModelConfig } from '../api/types'

/** Generated from the layers of the loaded model. */
export function ArchitectureDiagram({ config }: { config: ModelConfig }) {
  return <ol className="arch" aria-label="Model layers in order">{config.layers.map(l => <li key={l.name}>
    <strong>{l.type}</strong><span className="small">{l.name}</span>
    <span className="num small">{l.output_shape.filter(d => d !== null).join('×') || '—'}</span>
    <span className="small muted">{l.params.toLocaleString()} params</span>
  </li>)}</ol>
}
