export const PRIMARY_NOTE = 'frozen on validation before the test was opened'

export function Role({ model, primary }: { model: string; primary: string }) {
  if (model === 'deployed_model') return <span className="tag">Reference</span>
  return model === primary ? <span className="tag good">PRIMARY</span> : <span className="tag">SECONDARY</span>
}
