import { useApi } from '../api/useApi'
import type { RecordInfo } from '../api/types'
import { Load } from './states'
import { Trace } from './waveform/Trace'

export type WindowData = {
  sample: number; start_sample: number; sampling_rate: number; units: string | null
  raw: number[]; filtered: number[]; model_input: number[]
  stats: Record<'raw' | 'filtered' | 'model_input', { mean: number; sd: number }>
}

/** One window at each real stage: raw signal, bandpass filtered, standardized model input. */
export function WindowStages({ record, sample }: { record: RecordInfo; sample: number }) {
  const q = useApi<WindowData>(`/recordings/${record.id}/window?sample=${sample}`)
  return <Load q={q}>{w => {
    const unit = w.units ?? 'signal units', start = w.start_sample / w.sampling_rate
    return <div className="stack">
      {([['raw', 'Raw signal', unit], ['filtered', 'Bandpass filtered', unit], ['model_input', 'Model input (standardized)', 'unitless z-score']] as const).map(([k, label, u]) => <div key={k}>
        <Trace values={w[k]} fs={w.sampling_rate} startSeconds={start} label={label} unit={u} />
        <div className="small muted num">mean {w.stats[k].mean.toFixed(3)} · SD {w.stats[k].sd.toFixed(3)} · {w[k].length} samples</div>
      </div>)}
    </div>
  }}</Load>
}
