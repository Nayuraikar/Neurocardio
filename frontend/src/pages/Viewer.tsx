import { useState, type ComponentProps } from 'react'
import { Link } from 'react-router-dom'
import type { Analysis, Beat, RecordInfo } from '../api/types'
import { useBeats } from '../api/useBeats'
import { ClassDistribution } from '../components/ClassDistribution'
import { NeedsAnalysis } from '../components/NeedsAnalysis'
import { SplitNote } from '../components/SplitNote'
import { Empty, Load } from '../components/states'
import { Card, ClassBadge, PageHead, ProbBars } from '../components/ui'
import { WindowTable } from '../components/WindowTable'
import { Playback } from '../components/waveform/Playback'
import { clampView, type View } from '../components/waveform/useChunk'
import { WaveformViewer } from '../components/waveform/WaveformViewer'
import { useRecording } from '../context/RecordingContext'

/** Owns the playback cursor so that each animation frame re-renders only the waveform. */
function PlayingWaveform({ onStop, ...props }: Omit<ComponentProps<typeof WaveformViewer>, 'cursor'> & { onStop: (index: number) => void }) {
  const [cursor, setCursor] = useState<number | null>(null)
  const duration = props.record.duration_seconds
  const follow = (t: number) => {
    setCursor(t)
    const v = props.view
    if (t < v.a || t > v.a + v.span * 0.8) props.onView(clampView(t - v.span * 0.2, v.span, duration))
  }
  return <>
    <WaveformViewer {...props} cursor={cursor} />
    <Playback duration={duration} beats={props.beats} cursor={cursor} onCursor={follow} start={props.selected?.timestamp_seconds ?? 0}
      onStop={b => { if (b) onStop(b.beat_index) }} />
  </>
}

function Inner({ analysis, record, beats }: { analysis: Analysis; record: RecordInfo; beats: Beat[] }) {
  const { windowIndex, setWindow, link } = useRecording()
  const selected = beats[Math.min(windowIndex, beats.length - 1)] ?? null
  const centre = (t: number, span: number) => clampView(t - span / 2, span, record.duration_seconds)
  const [view, setView] = useState<View>(() => centre(selected?.timestamp_seconds ?? 0, 8))
  const [filtered, setFiltered] = useState(true)
  const select = (b: Beat) => {
    setWindow(b.beat_index)
    setView(v => b.timestamp_seconds < v.a + 1 || b.timestamp_seconds > v.a + v.span - 1 ? centre(b.timestamp_seconds, v.span) : v)
  }
  const jump = (b: Beat) => { setWindow(b.beat_index); setView(v => centre(b.timestamp_seconds, v.span)) }
  const ventricular = beats.filter(b => b.prediction.predicted_class === 'V')
  const pending = beats.filter(b => b.prediction.review_flag && b.review?.status !== 'reviewed')
  const list = (items: Beat[]) => items.length
    ? <div className="row" style={{ maxHeight: 130, overflow: 'auto' }}>{items.slice(0, 200).map(b => <button type="button" key={b.id} onClick={() => jump(b)}>#{b.beat_index + 1} · {b.timestamp_seconds.toFixed(1)} s</button>)}{items.length > 200 && <span className="muted">showing 200 of {items.length}</span>}</div>
    : <p className="muted">None.</p>
  return <>
    <SplitNote record={record} />
    <Card title={record.filename} aside={<div className="row" role="group" aria-label="Signal version">
      <button type="button" aria-pressed={!filtered} onClick={() => setFiltered(false)}>Raw</button>
      <button type="button" aria-pressed={filtered} onClick={() => setFiltered(true)}>Bandpass filtered</button></div>}>
      <PlayingWaveform record={record} beats={beats} filtered={filtered} view={view} onView={setView} selected={selected} onSelect={select} onStop={setWindow} />
    </Card>
    <div className="grid-3">
      <Card title="Selected window">
        {selected ? <>
          <p>#{selected.beat_index + 1} at <span className="num">{selected.timestamp_seconds.toFixed(2)} s</span> · model class <ClassBadge code={selected.prediction.predicted_class} long /></p>
          <ProbBars probs={selected.prediction.probabilities} />
          <div className="row"><Link className="btn primary" to={link('/xai', selected.beat_index)}>Explain this window</Link><Link className="btn" to={link('/confidence', selected.beat_index)}>Uncertainty</Link></div>
        </> : <Empty title="No window selected" />}
      </Card>
      <Card title="Class distribution"><ClassDistribution counts={analysis.summary.class_counts ?? {}} source="analysis#summary.class_counts" /></Card>
      <Card title={`Windows needing review (${pending.length})`}>{list(pending)}</Card>
    </div>
    <Card title={`V-class windows (${ventricular.length})`} aside={ventricular[0] && <Link to={link('/xai', ventricular[0].beat_index)}>Explain the first V window</Link>}>{list(ventricular)}</Card>
    <Card title="Window table"><WindowTable beats={beats} selectedId={selected?.id} onSelect={jump} /></Card>
  </>
}

export default function ViewerPage() {
  return <><PageHead title="ECG Viewer" sub="Scroll to zoom, drag to pan, click a beat to select its window." />
    <NeedsAnalysis>{(a, r) => <WithBeats analysis={a} record={r} />}</NeedsAnalysis></>
}

function WithBeats({ analysis, record }: { analysis: Analysis; record: RecordInfo }) {
  const q = useBeats(analysis.id)
  return <Load q={q} lines={10}>{b => b.length ? <Inner key={analysis.id} analysis={analysis} record={record} beats={b} /> : <Empty title="This analysis has no windows" />}</Load>
}
