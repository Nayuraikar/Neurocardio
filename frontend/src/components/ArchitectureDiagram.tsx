import type { ModelConfig } from '../api/types'
import { Num } from './Num'

/** Generated from the layers of the loaded model. */
export function ArchitectureDiagram({ config }: { config: ModelConfig }) {
  return <ol className="arch" aria-label="Model layers in order">{config.layers.map((l, i) => <li key={l.name}>
    <strong data-source={`metrics#model.layers.${i}.type`} data-fmt="raw">{l.type}</strong><span className="small" data-source={`metrics#model.layers.${i}.name`} data-fmt="raw">{l.name}</span>
    <span className="num small" data-source={`shape:metrics#model.layers.${i}.output_shape`}>{l.output_shape.filter(d => d !== null).join('×') || '—'}</span>
    <span className="small muted"><Num v={l.params} src={`metrics#model.layers.${i}.params`} /> params</span>
  </li>)}</ol>
}
