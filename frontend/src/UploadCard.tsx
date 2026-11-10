import { useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Upload } from 'lucide-react'
import { api } from './shared'

export default function UploadCard() {
  const nav = useNavigate()
  const input = useRef<HTMLInputElement | null>(null)
  const [files, setFiles] = useState<File[]>([])
  const [lead, setLead] = useState('MLII')
  const [rate, setRate] = useState('360')
  const [busy, setBusy] = useState('')
  const [err, setErr] = useState('')
  const [drag, setDrag] = useState(false)
  const pick = (list: FileList | null) => { if (list?.length) { setFiles(Array.from(list)); setErr('') } }

  async function go() {
    if (!files.length) return
    setErr('')
    try {
      setBusy('Validating…')
      const f = new FormData()
      files.forEach(file => f.append('files', file))
      f.append('lead', lead)
      if (files.some(x => x.name.toLowerCase().endsWith('.csv'))) f.append('sampling_rate', rate)
      const up = await api.post('/ecg/upload', f)
      setBusy('Analysing beats…')
      const run = await api.post('/analyses', null, { params: { record_id: up.data.id } })
      nav(`/app/analyses/${run.data.id}`)
    } catch (e: any) {
      const d = e.response?.data?.detail
      setErr(typeof d === 'string' ? d : d?.message || e.message)
      setBusy('')
    }
  }

  return <article id="upload" className="pf-upload">
    <span className="pf-corner">UPLOAD</span>
    <h2>Upload an ECG</h2>
    <div className={`pf-drop${drag ? ' active' : ''}`} tabIndex={0} role="button" aria-label="Choose or drop ECG files"
      onClick={() => input.current?.click()}
      onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); input.current?.click() } }}
      onDragEnter={e => { e.preventDefault(); setDrag(true) }} onDragOver={e => e.preventDefault()}
      onDragLeave={() => setDrag(false)}
      onDrop={e => { e.preventDefault(); setDrag(false); pick(e.dataTransfer.files) }}>
      <Upload size={22} />
      <b>Drop WFDB or CSV files here</b>
      <small>.hea + .dat (+ optional .atr) · or a CSV signal file</small>
      <input ref={input} type="file" multiple accept=".csv,.hea,.dat,.atr" hidden onChange={e => pick(e.target.files)} />
    </div>
    {files.length > 0 && <ul className="pf-files">{files.map((f, i) => <li key={`${f.name}-${i}`}>{f.name}<small>{(f.size / 1024).toFixed(1)} KB</small></li>)}</ul>}
    <div className="pf-upload-fields">
      <label>LEAD / CSV COLUMN<input value={lead} onChange={e => setLead(e.target.value)} /></label>
      <label>SAMPLING (HZ)<input inputMode="decimal" value={rate} onChange={e => setRate(e.target.value)} /></label>
    </div>
    {err && <div className="error" role="alert">{err}</div>}
    <button type="button" className="pf-cta-btn" disabled={!files.length || !!busy} onClick={go}>{busy || 'UPLOAD & ANALYSE'}</button>
  </article>
}
