import { Num } from './Num'

/** The only place accuracy is headlined: macro-F1 and S/F recall lead, and the majority-class baseline sits beside accuracy. */
export function HeadlineMetrics({ macroF1, sRecall, fRecall, accuracy, baseline, sources }: {
  macroF1: number | null; sRecall: number | null; fRecall: number | null; accuracy: number | null; baseline: number | null
  sources: { macroF1: string; sRecall: string; fRecall: string; accuracy: string; baseline: string }
}) {
  const recall = (v: number | null, src: string) => v === null ? <span>no windows</span> : <Num v={v} fmt="f3" src={src} />
  return <div className="headline" data-testid="headline">
    <div className="stat lead"><b><Num v={macroF1} fmt="f3" src={sources.macroF1} /></b><span>Macro-F1 (N, S, V, F weighted equally)</span></div>
    <div className="stat"><b>{recall(sRecall, sources.sRecall)}</b><span>S recall</span></div>
    <div className="stat"><b>{recall(fRecall, sources.fRecall)}</b><span>F recall</span></div>
    <div className="stat"><b><Num v={accuracy} fmt="pct1" src={sources.accuracy} /></b>
      <span>Accuracy. Always answering the majority class scores <strong><Num v={baseline} fmt="pct1" src={sources.baseline} /></strong>.</span></div>
  </div>
}
