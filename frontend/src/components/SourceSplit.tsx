import type { DemoMeta } from '../api/demo'

export function SourceSplit({ s }: { s: DemoMeta }) {
  return <span data-testid="source-split"><span className="tag">MIT-BIH record {s.record}</span> <span className="tag">app model: {s.app_model_split} patient</span> <span className="tag">study: {s.study_split} patient</span></span>
}
