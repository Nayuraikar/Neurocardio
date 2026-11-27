import { Component, type ReactNode } from 'react'

/** A rendering failure (for example an artifact with an unexpected shape) shows a message instead of a blank page. */
export class ErrorBoundary extends Component<{ children: ReactNode }, { message: string | null }> {
  state = { message: null as string | null }
  static getDerivedStateFromError(error: unknown) { return { message: error instanceof Error ? error.message : String(error) } }
  render() {
    if (this.state.message === null) return this.props.children
    return <div className="state state-error" role="alert" data-state="error">
      <strong>This page could not be displayed</strong><div>{this.state.message}</div>
      <button type="button" onClick={() => this.setState({ message: null })}>Retry</button>
    </div>
  }
}
