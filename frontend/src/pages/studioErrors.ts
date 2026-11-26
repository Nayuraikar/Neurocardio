/** Turns a backend rejection into a title and a fix. The backend's own message is always shown underneath. */
const RULES: [RegExp, string, string][] = [
  [/paced/i, 'Paced recording', 'Paced rhythms are out of scope for this model. Choose a non-paced recording.'],
  [/alignment check failed/i, 'Annotation does not match the signal', 'Upload the .atr that belongs to this record, or leave it out.'],
  [/sampling.?rate|Hz/i, 'Unsupported or missing sampling rate', 'The model needs 360 Hz input. No resampling is applied.'],
  [/companion|matching \.dat|one \.hea|base name|record name|declares record/i, 'WFDB files are incomplete or mismatched', 'Upload one .hea and its .dat (and optionally its .atr) with the same base name.'],
  [/lead|column/i, 'Lead or column problem', 'Choose a lead or column that exists in the file.'],
  [/unsupported upload|accepts only/i, 'Unsupported file type', 'Upload a CSV, or WFDB .hea and .dat files.'],
  [/NaN|non-numeric|flatline|shorter|R peaks|malformed/i, 'Signal failed validation', 'Check the file contents: finite numbers, at least 2 s, with detectable beats.'],
  [/limit/i, 'File too large', 'Upload a shorter recording.'],
  [/reach the API/i, 'Backend unavailable', 'Start the backend and retry.'],
]
export function explain(message: string): { title: string; fix: string } {
  const hit = RULES.find(([re]) => re.test(message))
  return hit ? { title: hit[1], fix: hit[2] } : { title: 'Request failed', fix: 'Read the message below and retry.' }
}

export async function csvColumns(file: File): Promise<string[]> {
  const head = await file.slice(0, 4096).text()
  return head.split(/\r?\n/)[0].split(',').map(s => s.trim().replace(/^"|"$/g, '')).filter(Boolean)
}
