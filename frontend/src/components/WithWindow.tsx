import type { ReactNode } from 'react'
import type { Analysis, Beat, RecordInfo } from '../api/types'
import { useBeats } from '../api/useBeats'
import { useRecording } from '../context/RecordingContext'
import { NeedsAnalysis } from './NeedsAnalysis'
import { Empty, Load } from './states'

type Args = { analysis: Analysis; record: RecordInfo; beats: Beat[]; beat: Beat }

function Inner({ analysis, record, children }: { analysis: Analysis; record: RecordInfo; children: (a: Args) => ReactNode }) {
  const q = useBeats(analysis.id)
  const { windowIndex } = useRecording()
  return <Load q={q} lines={8}>{beats => beats.length
    ? <>{children({ analysis, record, beats, beat: beats[Math.min(windowIndex, beats.length - 1)] })}</>
    : <Empty title="This analysis has no windows" />}</Load>
}

export function WithWindow({ children }: { children: (a: Args) => ReactNode }) {
  return <NeedsAnalysis>{(a, r) => <Inner analysis={a} record={r}>{children}</Inner>}</NeedsAnalysis>
}
