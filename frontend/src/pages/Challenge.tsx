import { useState } from 'react'
import type { DemoMeta, DemoSample } from '../api/demo'
import { score } from '../api/scoring'
import { useApi } from '../api/useApi'
import { HeadlineMetrics } from '../components/HeadlineMetrics'
import { Load, PartialNotice } from '../components/states'
import { SourceSplit } from '../components/SourceSplit'
import { Card, ClassBadge, PageHead, Table } from '../components/ui'
import { Trace } from '../components/waveform/Trace'
import { CLASS_CODES, CLASSES } from '../theme/theme'

type Answer = { guess: string; sample: DemoSample }

function Question({ item, number, total, onNext }: { item: DemoMeta; number: number; total: number; onNext: (a: Answer) => void }) {
  const hidden = useApi<DemoSample>(`/demo/samples/${item.id}?reveal=false`)
  const [guess, setGuess] = useState<string | null>(null)
  const revealed = useApi<DemoSample>(guess ? `/demo/samples/${item.id}` : null)
  const r = revealed.data
  return <Card title={`Window ${number} of ${total}`}>
    <SourceSplit s={item} />
    <Load q={hidden} lines={6}>{s => <Trace values={s.model_input} fs={s.sampling_rate} label="Which class is the centre beat?" unit="unitless z-score" height={240} band={{ from: 270, to: 450, color: '#777777' }} />}</Load>
    <div className="row" style={{ marginTop: 10 }}>{CLASS_CODES.map(c => <button type="button" key={c} disabled={!!guess} aria-pressed={guess === c} onClick={() => setGuess(c)}>{c} · {CLASSES[c].name}</button>)}</div>
    {guess && <Load q={revealed}>{() => <>
      <p>Reference <ClassBadge code={r!.label!} long /> · you said <ClassBadge code={guess} /> · model said <ClassBadge code={r!.prediction!.predicted_class} /></p>
      <button type="button" className="primary" onClick={() => onNext({ guess, sample: r! })}>{number === total ? 'See round summary' : 'Next window'}</button>
    </>}</Load>}
  </Card>
}

function Summary({ answers, who, pick }: { answers: Answer[]; who: 'You' | 'Model'; pick: (a: Answer) => string }) {
  const s = score(answers.map(a => ({ truth: a.sample.label!, predicted: pick(a) })))
  const src = (k: string) => `challenge#${who}.${k}`
  return <Card title={who}><HeadlineMetrics macroF1={s.macroF1} sRecall={s.per[1].recall} fRecall={s.per[3].recall} accuracy={s.accuracy} baseline={s.baseline}
    sources={{ macroF1: src('macro_f1'), sRecall: src('s_recall'), fRecall: src('f_recall'), accuracy: src('accuracy'), baseline: src('baseline') }} /></Card>
}

function Round({ seed, onRestart }: { seed: number; onRestart: () => void }) {
  const q = useApi<{ items: DemoMeta[] }>(`/demo/challenge?per_class=2&seed=${seed}`)
  const [answers, setAnswers] = useState<Answer[]>([])
  return <Load q={q}>{({ items }) => answers.length < items.length
    ? <Question key={items[answers.length].id} item={items[answers.length]} number={answers.length + 1} total={items.length} onNext={a => setAnswers(x => [...x, a])} />
    : <>
      <PartialNotice>A tiny, class-balanced round from training and validation patients. These numbers say nothing about how you or the model would do on new patients.</PartialNotice>
      <Summary answers={answers} who="You" pick={a => a.guess} />
      <Summary answers={answers} who="Model" pick={a => a.sample.prediction!.predicted_class} />
      <Card title="Window by window"><Table head={['Source', 'Reference', 'You', 'Model']} rows={answers.map(a => [<SourceSplit key="s" s={a.sample} />, <ClassBadge key="r" code={a.sample.label!} />, <ClassBadge key="y" code={a.guess} />, <ClassBadge key="m" code={a.sample.prediction!.predicted_class} />])} /></Card>
      <div><button type="button" className="primary" onClick={onRestart}>New round</button></div>
    </>}</Load>
}

export default function Challenge() {
  const [seed, setSeed] = useState<number | null>(null)
  return <><PageHead title="Challenge Mode" sub="Label a window yourself, then compare with the reference and the model." />
    {seed === null
      ? <Card><p>Eight windows, two per class, from training and validation patients only. Each shows its source split.</p><button type="button" className="primary" onClick={() => setSeed(Math.floor(Math.random() * 100000))}>Start a round</button></Card>
      : <Round key={seed} seed={seed} onRestart={() => setSeed(Math.floor(Math.random() * 100000))} />}</>
}
