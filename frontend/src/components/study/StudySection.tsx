import type { Study } from '../../api/study'
import type { useApi } from '../../api/useApi'
import { ErrorBoundary } from '../ErrorBoundary'
import { Load } from '../states'
import { Card } from '../ui'
import { TestPanel } from './TestPanel'
import { ValidationPanel } from './ValidationPanel'

/** The study summary used on pages about the app's own model: primary only, validation and test in separate panels.
 * Its own error boundary keeps a malformed study file from hiding the rest of the page. */
export function StudySection({ q }: { q: ReturnType<typeof useApi<Study>> }) {
  return <ErrorBoundary><Load q={q}>{s => <>
    <Card title="Multi-database study: validation"><ValidationPanel study={s} compact /></Card>
    <Card title="Multi-database study: test"><TestPanel study={s} compact /></Card>
  </>}</Load></ErrorBoundary>
}
