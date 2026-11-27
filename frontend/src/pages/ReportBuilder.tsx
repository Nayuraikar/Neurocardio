import { useState } from 'react'
import { useParams } from 'react-router-dom'
import { http, toApiError, type ApiError } from '../api/client'
import { when } from '../components/format'
import { Empty, ErrorState, Load, Skeleton, SuccessBanner } from '../components/states'
import { Card, KeyValues, PageHead } from '../components/ui'
import { useAnalysisWithRecord } from '../api/useAnalysisWithRecord'

const SECTIONS: [string, string, boolean][] = [
  ['recording', 'Recording information', true], ['preprocessing', 'Processing and model', true], ['waveform', 'Waveform excerpts', false],
  ['distribution', 'Class distribution', true], ['predictions', 'Window predictions', true], ['xai', 'Stored explanations', false],
  ['notes', 'Reviewer assessments and notes', true], ['performance', 'Model performance on held-out data', false], ['limitations', 'Limitations', false],
]

export default function ReportBuilder() {
  const { id = '' } = useParams()
  const { analysis, record } = useAnalysisWithRecord(id)
  const [chosen, setChosen] = useState<string[]>(SECTIONS.map(s => s[0]))
  const [busy, setBusy] = useState<string | null>(null)
  const [error, setError] = useState<ApiError | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [done, setDone] = useState<string | null>(null)
  const query = `sections=${chosen.join(',')}`
  const csvEmpty = !chosen.some(s => SECTIONS.find(x => x[0] === s)?.[2])

  const fetchFile = async (format: 'pdf' | 'csv', show: boolean) => {
    setBusy(format); setError(null); setDone(null)
    try {
      const blob = (await http.get(`/reports/${id}?format=${format}&${query}`, { responseType: 'blob' })).data as Blob
      const url = URL.createObjectURL(blob)
      if (show) { if (preview) URL.revokeObjectURL(preview); setPreview(url) }
      else {
        const a = document.createElement('a')
        a.href = url; a.download = `neurocardio-${id.slice(0, 8)}.${format}`; a.click()
        window.setTimeout(() => URL.revokeObjectURL(url), 1000)
        setDone(`${format.toUpperCase()} downloaded.`)
      }
    } catch (e) { setError(toApiError(e)) }
    finally { setBusy(null) }
  }
  const toggle = (key: string) => { setPreview(null); setChosen(c => c.includes(key) ? c.filter(k => k !== key) : [...c, key]) }

  return <><PageHead title="Report builder" sub="Choose the sections, preview the PDF, then export." />
    <Load q={analysis}>{a => a.status !== 'completed' ? <Empty title="Reports are available only for completed analyses" /> : <>
      <Card><KeyValues rows={[['Recording', record.data?.filename ?? '…'], ['Analysed', when(a.completed_at)], ['Model version', a.model_version], ['Windows', a.beat_count.toLocaleString()]]} /></Card>
      <Card title="Sections">
        <div className="grid-3">{SECTIONS.map(([key, label, csv]) => <label key={key} className="row"><input type="checkbox" checked={chosen.includes(key)} onChange={() => toggle(key)} /> {label}{!csv && <span className="small muted">PDF only</span>}</label>)}</div>
        <div className="row" style={{ marginTop: 14 }}>
          <button type="button" disabled={!chosen.length || !!busy} onClick={() => void fetchFile('pdf', true)}>Preview PDF</button>
          <button type="button" className="primary" disabled={!chosen.length || !!busy} onClick={() => void fetchFile('pdf', false)}>Download PDF</button>
          <button type="button" disabled={csvEmpty || !!busy} onClick={() => void fetchFile('csv', false)}>Download CSV</button>
        </div>
        {!chosen.length && <p className="muted">Choose at least one section.</p>}
        <p className="notice">Every report carries the model version and a generation time. Explanations appear only if they were already generated for this analysis. Long recordings make long prediction tables.</p>
        {busy && <Skeleton lines={2} label="Building report" />}
        {error && <ErrorState title="The report could not be built" error={{ ...error, message: error.status === null ? error.message : `The server answered HTTP ${error.status}.` }} />}
        {done && <SuccessBanner>{done}</SuccessBanner>}
      </Card>
      {preview && <Card title="Preview"><iframe title="Report preview" src={preview} style={{ width: '100%', height: '75vh', border: '1px solid var(--line)' }} /></Card>}
    </>}</Load></>
}
