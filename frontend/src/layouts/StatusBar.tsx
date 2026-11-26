import { useApi } from '../api/useApi'
import type { Status } from '../api/types'
import { ErrorState } from '../components/states'

export function SystemStatus() {
  const q = useApi<Status>('/status')
  const item = (label: string, ok: boolean | null) =>
    <span className={`dot ${ok === null ? '' : ok ? 'ok' : 'bad'}`}>{label}: {ok === null ? 'checking' : ok ? 'ready' : 'unavailable'}</span>
  return <div className="status-line" aria-label="System status">
    {item('API', q.error ? false : q.data ? true : null)}
    {item('Model', q.error ? false : q.data ? q.data.model_loaded : null)}
    {item('Database', q.error ? false : q.data ? q.data.database_ok : null)}
  </div>
}

export function BackendBanner() {
  const q = useApi<Status>('/status')
  return q.error ? <ErrorState title="Backend unavailable" error={q.error} onRetry={q.reload} /> : null
}
