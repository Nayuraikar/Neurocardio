export type Job = { name: string; files: File[]; csv: boolean; problem: string | null }

/** One job per CSV file, and one per WFDB base name (.hea + .dat + optional .atr). */
export function groupFiles(files: File[]): Job[] {
  const jobs: Job[] = []
  const wfdb = new Map<string, File[]>()
  for (const f of files) {
    const dot = f.name.lastIndexOf('.'), ext = f.name.slice(dot + 1).toLowerCase()
    if (ext === 'csv') jobs.push({ name: f.name, files: [f], csv: true, problem: null })
    else if (['hea', 'dat', 'atr'].includes(ext)) wfdb.set(f.name.slice(0, dot), [...(wfdb.get(f.name.slice(0, dot)) ?? []), f])
    else jobs.push({ name: f.name, files: [f], csv: false, problem: 'Unsupported file type' })
  }
  for (const [stem, group] of wfdb) {
    const has = (e: string) => group.some(f => f.name.toLowerCase().endsWith(e))
    const missing = ['.hea', '.dat'].filter(e => !has(e))
    jobs.push({ name: stem, files: group, csv: false, problem: missing.length ? `Missing ${missing.join(' and ')}` : null })
  }
  return jobs.sort((a, b) => a.name.localeCompare(b.name))
}
