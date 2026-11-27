import { CLASS_CODES, classFill } from '../theme/theme'
import { Num } from './Num'

/** Rows are true classes, columns predicted. Shading is the share of the row. */
export function ConfusionMatrix({ counts, source, onSelect }: { counts: number[][]; source: string; onSelect?: (truth: string, predicted: string) => void }) {
  return <div className="table-wrap"><table className="matrix">
    <caption>Rows: reference class. Columns: model class. Percentages are of the row.</caption>
    <thead><tr><th>True \ model</th>{CLASS_CODES.map(c => <th key={c}>{c}</th>)}<th>Support</th></tr></thead>
    <tbody>{CLASS_CODES.map((t, i) => {
      const total = counts[i].reduce((a, b) => a + b, 0)
      return <tr key={t}><th scope="row">{t}</th>{counts[i].map((v, j) => {
        const share = total ? v / total : 0
        const body = <><Num v={v} src={`${source}.${i}.${j}`} /><div className="small muted">{(share * 100).toFixed(1)}%</div></>
        return <td key={j} style={{ background: `${classFill(i === j ? 'N' : 'V')}${Math.round(share * 150).toString(16).padStart(2, '0')}` }}>
          {onSelect && v > 0 ? <button type="button" onClick={() => onSelect(t, CLASS_CODES[j])} aria-label={`True ${t}, model ${CLASS_CODES[j]}: ${v} windows`}>{body}</button> : body}</td>
      })}<td>{total.toLocaleString()}</td></tr>
    })}</tbody>
  </table></div>
}
