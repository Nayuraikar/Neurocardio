import { MODEL_LABELS, RR_MODELS, type Study } from '../api/study'
import { useApi } from '../api/useApi'
import { primaryOf } from '../components/study/names'
import { Role } from '../components/study/Role'
import { TestPanel } from '../components/study/TestPanel'
import { ValidationPanel } from '../components/study/ValidationPanel'
import { Load, PartialNotice } from '../components/states'
import { Card, PageHead, Table } from '../components/ui'

export default function Comparison() {
  const q = useApi<Study>('/study')
  return <><PageHead title="Model Comparison Lab" sub="Study models trained on three databases. Lead with macro-F1 and S/F recall; validation and test are separate." />
    <PartialNotice>S and F are the hard classes in this problem. Compare their recall and precision before reading any accuracy.</PartialNotice>
    <Load q={q} lines={8}>{s => <>
      <Card title="Validation"><ValidationPanel study={s} /></Card>
      <Card title="Test"><TestPanel study={s} /></Card>
      <Card title="What is compared"><Table head={['Model', 'Role', 'ID', 'Inputs']} rows={s.models.map(m => [MODEL_LABELS[m] ?? m,
        <Role key="r" model={m} primary={primaryOf(s) ?? ''} />, <code key="c">{m}</code>,
        m === 'stacked_ensemble' ? 'Combines the other study models, including RR models (offline only)' : RR_MODELS.includes(m) ? 'RR-interval features: needs the following beat and the whole-record median (offline only)' : 'Waveform window only'])} />
        <p className="notice">The app does not run any study model. None of these scores describes the classifications shown on your recordings.</p></Card>
    </>}</Load></>
}
