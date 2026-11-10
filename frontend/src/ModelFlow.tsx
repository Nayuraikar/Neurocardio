import { useState } from 'react'

const BW = 130, BH = 72, GAP = 66, X0 = 20
const ROW_Y = [40, 250]
const blocks = [
  { t: 'Input', s: '720 × 1', tip: 'One R-peak-centred, z-scored ECG window (2 s at 360 Hz, single lead).' },
  { t: 'Conv1D + BN', s: '32 filters · k7', tip: 'Learns local morphology (QRS slope and width). Batch norm keeps training stable.' },
  { t: 'MaxPool', s: 'pool 2', tip: 'Halves temporal resolution and adds tolerance to small shifts.' },
  { t: 'Conv1D + BN', s: '64 filters · k5', tip: 'Combines first-level features into wider waveform motifs.' },
  { t: 'MaxPool', s: 'pool 2', tip: 'Second downsampling step before the final convolution.' },
  { t: 'Conv1D', s: '128 filters · k3', tip: 'Final feature map (conv3). Grad-CAM reads this layer.' },
  { t: 'BiLSTM', s: '64 / direction', tip: 'Reads the feature sequence forward and backward to capture beat context.' },
  { t: 'Dropout', s: 'rate 0.4', tip: 'Regularisation during training; inactive at inference.' },
  { t: 'Dense', s: '64 units', tip: 'Fully connected layer that mixes the sequence summary.' },
  { t: 'Dense + Softmax', s: '4 · N S V F', tip: 'Outputs probabilities for Normal, Supraventricular, Ventricular and Fusion beats.' },
]
const shapes = ['720×1', '720×32', '360×32', '360×64', '180×64', '180×128', '128', '128', '64']
const pos = (i: number) => ({ x: X0 + (i % 5) * (BW + GAP), y: ROW_Y[Math.floor(i / 5)] })

export default function ModelFlow({ acc, f1, auc, fromApi }: { acc: number; f1: number; auc: number; fromApi: boolean }) {
  const [tip, setTip] = useState<number | null>(null)
  const p = tip === null ? null : pos(tip)
  return <div className="pf-model-grid">
    <article className="panel pf-flowchart">
      <span className="pf-corner">ARCHITECTURE</span>
      <div className="panel-head"><h2>CNN–BiLSTM architecture</h2><span>Hover or focus a block</span></div>
      <div className="pf-flow-wrap">
        <svg viewBox="0 0 1000 360" role="img" aria-label="Model architecture flowchart">
          <defs><marker id="pf-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L8 4L0 8z" fill="#2B3A1E" /></marker></defs>
          {blocks.slice(0, 9).map((_, i) => {
            const a = pos(i), b = pos(i + 1)
            const wrap = i === 4
            const d = wrap ? `M${a.x + BW / 2} ${a.y + BH} V${(a.y + b.y + BH) / 2} H${b.x + BW / 2} V${b.y}` : `M${a.x + BW} ${a.y + BH / 2} H${b.x}`
            return <g key={i}>
              <path d={d} fill="none" stroke="#2B3A1E" strokeDasharray="4 3" markerEnd="url(#pf-arrow)" />
              <text x={wrap ? (a.x + b.x + BW) / 2 : a.x + BW + GAP / 2} y={wrap ? (a.y + b.y + BH) / 2 - 6 : a.y + BH / 2 - 7} textAnchor="middle" fontSize="10" fill="#555">{shapes[i]}</text>
            </g>
          })}
          {blocks.map((bl, i) => {
            const { x, y } = pos(i)
            const last = i === blocks.length - 1, first = i === 0
            return <g key={i} tabIndex={0} role="img" aria-label={`${bl.t} ${bl.s}. ${bl.tip}`} className="pf-block"
              onMouseEnter={() => setTip(i)} onMouseLeave={() => setTip(null)} onFocus={() => setTip(i)} onBlur={() => setTip(null)}>
              <rect x={x} y={y} width={BW} height={BH} fill={last ? '#E2E8C8' : first ? '#FAFAF6' : '#FAFAF6'} stroke="#999" strokeDasharray="1 3" strokeWidth="1.4" />
              <text x={x + 8} y={y + 14} fontSize="9" fill="#777" letterSpacing="1">{String(i + 1).padStart(2, '0')}</text>
              <text x={x + BW / 2} y={y + 36} textAnchor="middle" fontSize="13" fontWeight="600" fill="#111">{bl.t}</text>
              <text x={x + BW / 2} y={y + 55} textAnchor="middle" fontSize="11" fill="#444">{bl.s}</text>
            </g>
          })}
          <text x={pos(9).x + BW + 8} y={pos(9).y + BH / 2 + 4} fontSize="10" fill="#555">4</text>
          <text x={20} y={345} fontSize="10" fill="#777">Tensor shapes shown as (time × channels), batch omitted; assumes same padding.</text>
        </svg>
        {p && tip !== null && <div className="pf-tip" style={{ left: `${(p.x + BW / 2) / 10}%`, top: `${(p.y + BH + 8) / 3.6}%` }} role="tooltip"><b>{blocks[tip].t}</b>{blocks[tip].tip}</div>}
      </div>
    </article>
    <aside className="pf-side">
      <div><span className="pf-label">PREPROCESSING</span>
        <ul><li>0.5–40 Hz bandpass</li><li>R-peak-centred 720-sample window</li><li>Per-window z-score</li></ul></div>
      <div><span className="pf-label">PARAMETERS</span><div className="pf-big">142,980</div><small>trainable + non-trainable</small></div>
      <div><span className="pf-label">TEST METRICS</span>
        <dl><dt>Accuracy</dt><dd>{acc.toFixed(3)}</dd><dt>Macro F1</dt><dd>{f1.toFixed(3)}</dd><dt>Macro AUC</dt><dd>{auc.toFixed(3)}</dd></dl>
        <small>{fromApi ? 'From the model-info endpoint.' : 'Reference values; model-info endpoint unavailable.'} S and F recall are low.</small></div>
    </aside>
  </div>
}
