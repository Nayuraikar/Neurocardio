import { Link } from 'react-router-dom'

export function WaveLink({ analysisId, index, label = 'View' }: { analysisId: string; index: number; label?: string }) {
  return <Link to={`/viewer?analysis=${analysisId}&window=${index}`}>{label}</Link>
}
