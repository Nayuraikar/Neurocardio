import type { ReactNode } from 'react'
import { Link } from 'react-router-dom'
import type { ApiError } from '../../api/client'

export function Skeleton({ lines = 4, label = 'Loading' }: { lines?: number; label?: string }) {
  return <div className="skeleton" role="status" aria-label={label} data-state="loading">
    {Array.from({ length: lines }, (_, i) => <span key={i} style={{ width: `${95 - (i % 3) * 18}%` }} />)}
  </div>
}

export function Empty({ title, children }: { title: string; children?: ReactNode }) {
  return <div className="state state-empty" data-state="empty"><strong>{title}</strong>{children && <div>{children}</div>}</div>
}

export function ErrorState({ error, onRetry, title = 'This could not be loaded' }: { error: ApiError; onRetry?: () => void; title?: string }) {
  return <div className="state state-error" role="alert" data-state="error">
    <strong>{title}</strong><div>{error.message}</div>
    {onRetry && <button type="button" onClick={onRetry}>Retry</button>}
  </div>
}

export function PartialNotice({ children }: { children: ReactNode }) {
  return <div className="state state-warn" data-state="partial"><div>{children}</div></div>
}

export function SuccessBanner({ children }: { children: ReactNode }) {
  return <div className="state state-ok" role="status" data-state="success">{children}</div>
}

export function ReviewReady({ pending, to }: { pending: number; to: string }) {
  return <div className="state state-warn review-ready" data-state="review-ready">
    <strong>Ready for review</strong>
    <div>{pending} flagged window{pending === 1 ? '' : 's'} waiting for a reviewer.</div>
    <Link className="btn" to={to}>Open review</Link>
  </div>
}

/** One panel's loading / error handling, so one failure does not blank the page. */
export function Load<T>({ q, children, lines }: { q: { data: T | null; error: ApiError | null; loading: boolean; reload: () => void }; children: (data: T) => ReactNode; lines?: number }) {
  if (q.error) return <ErrorState error={q.error} onRetry={q.reload} />
  if (!q.data) return <Skeleton lines={lines} />
  return <>{children(q.data)}</>
}
