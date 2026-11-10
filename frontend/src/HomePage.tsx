import { Activity, ArrowDown, ArrowUpRight, BrainCircuit, HeartPulse, ScanLine, UserRoundCheck, Waves } from 'lucide-react'
import { Link } from 'react-router-dom'
import TopNav from './Nav'

const classes = [
  { code: 'N', title: 'Normal', detail: 'Normal beats and selected normal variants', color: 'mint' },
  { code: 'S', title: 'Supraventricular', detail: 'Supraventricular ectopic beats', color: 'blue' },
  { code: 'V', title: 'Ventricular', detail: 'Ventricular ectopic beats', color: 'blush' },
  { code: 'F', title: 'Fusion', detail: 'Fusion of ventricular and normal beats', color: 'lavender' },
]

const features = [
  { number: '01', title: 'ECG analysis', text: 'Bring a local recording into view. Inspect its waveform, confirm the lead and sampling rate, then run a retrospective beat-level analysis.', icon: Waves, color: 'blue' },
  { number: '02', title: 'Beat classification', text: 'Explore model predictions across four AAMI beat classes, with class probabilities and review flags attached to each beat.', icon: ScanLine, color: 'blush' },
  { number: '03', title: 'Explainable AI', text: 'Inspect Grad-CAM or Integrated Gradients attribution on the same ECG window used for the selected beat prediction.', icon: BrainCircuit, color: 'lavender' },
  { number: '04', title: 'Human review', text: 'Record a separate human label and note. A review is kept distinct from the original prediction so the model output remains intact.', icon: UserRoundCheck, color: 'mint' },
]

function HeroTrace() {
  return (
    <svg className="hero-trace" viewBox="0 0 760 250" role="img" aria-labelledby="trace-title trace-description">
      <title id="trace-title">Illustrative ECG waveform</title>
      <desc id="trace-description">A decorative ECG trace accompanying the introduction to NeuroCardio. It is not patient data.</desc>
      <defs>
        <pattern id="hero-grid-small" width="18" height="18" patternUnits="userSpaceOnUse">
          <path d="M18 0H0V18" fill="none" stroke="#343448" strokeOpacity=".07" strokeWidth="1" />
        </pattern>
        <pattern id="hero-grid-large" width="90" height="90" patternUnits="userSpaceOnUse">
          <rect width="90" height="90" fill="url(#hero-grid-small)" />
          <path d="M90 0H0V90" fill="none" stroke="#343448" strokeOpacity=".11" strokeWidth="1" />
        </pattern>
      </defs>
      <rect width="760" height="250" fill="url(#hero-grid-large)" />
      <path d="M0 128H124l16-1 13 2 15-2 14 1 11-2 9 2 8-1 12 1 8-2 7 3 6-1 9-2 7 3 10-1 10-2 12 2 11-1 10-2 10 2 10-1 12 2 10-1 12-1 12 2 15-1 17 2 17-3 10 2 8-1 9 2 11-1 11 2 13-2 17 1 11-2 10 1 14-1 13 2 14-2 15 1 17-2 10 1 16-1 16 2h80" fill="none" stroke="#a9b5a5" strokeWidth="2" strokeDasharray="3 7" />
      <path className="hero-trace-line" d="M0 128h122l20-1 14 2 16-2 15 1 13-2 10 2 11-1 13 1 8-2 7 3 6-1 9-2 7 3 11-1 10-2 12 2 11-1 10-2 11 2 9-1 13 2 11-1 10-1 12 2h20l17 2 16-3 11 2 8-1 9 2 11-1 11 2 13-2 18 1 10-2 11 1 13-1 14 2 14-2 16 1 16-2 11 1 16-1 18 2h73" fill="none" stroke="#526a5c" strokeWidth="3" strokeLinejoin="round" />
      <path className="hero-beat" d="M245 129h30l12-2 13 1 13-4 9 2 15-1 12 2 16-1 15-3 10 2 9-1 12 2 14-1 9-2 8 3 5-1 4-49 5 97 5-33 10-13 12 2 16-1 13 2 14-1 20 2h36" fill="none" stroke="#a85d55" strokeWidth="3.5" strokeLinejoin="round" strokeLinecap="round" />
      <circle cx="453" cy="99" r="5" fill="#a85d55" />
      <text x="461" y="91" fill="#6f5a53" fontSize="10" letterSpacing="2">R-PEAK</text>
      <text x="22" y="30" fill="#737766" fontSize="9" letterSpacing="2">ILLUSTRATIVE TRACE · NOT PATIENT DATA</text>
      <text x="22" y="230" fill="#737766" fontSize="9" letterSpacing="1.5">A SIGNAL, READ ONE BEAT AT A TIME</text>
      <text x="704" y="230" fill="#737766" fontSize="9" textAnchor="end">SCHEMATIC</text>
    </svg>
  )
}

function MethodDiagram() {
  return (
    <div className="method-diagram" aria-label="CNN-BiLSTM analysis flow">
      <div className="method-node method-input"><span>INPUT</span><Waves size={22} /><b>ECG beat window</b><small>720 samples · 360 Hz</small></div>
      <div className="method-connector"><i /><span>features</span></div>
      <div className="method-node method-cnn"><span>FEATURE LEARNING</span><div className="method-bars" aria-hidden="true"><i /><i /><i /><i /><i /></div><b>CNN</b><small>Local morphology</small></div>
      <div className="method-connector"><i /><span>sequence</span></div>
      <div className="method-node method-bilstm"><span>CONTEXT</span><BrainCircuit size={22} /><b>BiLSTM</b><small>Bidirectional sequence</small></div>
      <div className="method-connector"><i /><span>scores</span></div>
      <div className="method-node method-output"><span>OUTPUT</span><div className="method-classes"><i>N</i><i>S</i><i>V</i><i>F</i></div><b>Beat classes</b><small>Probabilities · flags</small></div>
    </div>
  )
}

export default function HomePage() {
  return (
    <main className="home">
      <TopNav extra={<><a href="#approach">APPROACH</a><a href="#method">METHOD</a><a href="#dataset">RESEARCH DATA</a></>} />

      <section className="home-hero">
        <div className="hero-copy">
          <div className="home-kicker"><span /> A LOCAL ECG RESEARCH WORKSPACE</div>
          <h1>From Raw ECG<br />to Clear Decisions</h1>
          <p>NeuroCardio brings ECG signals, beat-level machine learning and human interpretation into one thoughtful research space.</p>
          <div className="hero-actions">
            <Link to="/app" className="button-ink">Enter the workspace <ArrowUpRight size={16} /></Link>
            <a className="text-link" href="#approach">Explore the approach <ArrowDown size={14} /></a>
          </div>
          <div className="hero-footnote"><span>01 / 04</span><i /> Built for investigation, not diagnosis</div>
        </div>
        <div className="pf-flow" aria-label="Pipeline: signal, preprocess, CNN-BiLSTM, explain">
          <span className="pf-corner">PIPELINE</span>
          <svg className="pf-flow-lines" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
            <path d="M24 22H76M78 26V74M76 78H24M22 74V26" /><path d="M24 24L44 44M76 24L56 44M76 76L56 56M24 76L44 56" />
          </svg>
          <div className="pf-node n1"><i>01</i>SIGNAL<small>Raw ECG · 360 Hz</small></div>
          <div className="pf-node n2"><i>02</i>PREPROCESS<small>Filter · window · z-score</small></div>
          <div className="pf-node ai">AI<small>CNN-BiLSTM</small></div>
          <div className="pf-node n4"><i>04</i>EXPLAIN<small>Grad-CAM · IG</small></div>
          <div className="pf-node n3"><i>03</i>CNN-BiLSTM<small>N · S · V · F</small></div>
          <div className="pf-flow-trace"><HeroTrace /></div>
        </div>
      </section>

      <section className="intro-band" id="approach">
        <div className="section-marker"><span>01</span><i /> AN OPENING NOTE</div>
        <div className="intro-copy">
          <h2>Medicine has its measures.<br /><em>Research begins with a better question.</em></h2>
          <p>NeuroCardio is an exploratory research prototype for studying ECG beat classification. It lets researchers examine the signal alongside model predictions, inspect attribution, and preserve human review as its own record.</p>
        </div>
        <div className="intro-aside"><Activity size={21} /><span>One recording.<br />Many ways to look.</span></div>
      </section>

      <section className="feature-section">
        <div className="section-heading">
          <div className="section-marker"><span>02</span><i /> FOUR WAYS IN</div>
          <h2>From waveform<br />to considered review.</h2>
          <p>Each view is a different lens on the same research question.</p>
        </div>
        <div className="feature-grid">
          {features.map(({ number, title, text, icon: Icon, color }) => (
            <article className={`feature feature-${color}`} key={number}>
              <div className="feature-top"><span>{number}</span><Icon size={20} strokeWidth={1.6} /></div>
              <h3>{title}</h3>
              <p>{text}</p>
              <div className="feature-rule"><i /></div>
            </article>
          ))}
        </div>
      </section>

      <section className="method-section" id="method">
        <div className="method-heading">
          <div className="section-marker"><span>03</span><i /> UNDER THE HOOD</div>
          <h2>A sequence, not<br />a black box.</h2>
          <p>The research model combines convolutional feature learning with bidirectional sequence context to classify individual beat windows.</p>
          <p className="method-note">The diagram describes the model family. Predictions and attribution in the workspace come from the selected local model and stored signal data.</p>
        </div>
        <div className="method-plate">
          <div className="plate-label"><span>RESEARCH METHOD</span><span>01—04</span></div>
          <MethodDiagram />
          <div className="method-caption"><span>INPUT WINDOW CENTRED ON A DETECTED R-PEAK</span><span>ILLUSTRATIVE MODEL FLOW</span></div>
        </div>
      </section>

      <section className="dataset-section" id="dataset">
        <div className="dataset-top">
          <div>
            <div className="section-marker"><span>04</span><i /> RESEARCH MATERIAL</div>
            <h2>Grounded in a<br /><em>shared benchmark.</em></h2>
          </div>
          <div className="dataset-note">
            <span className="dataset-seal">MIT<br /><b>BIH</b></span>
            <p>The MIT-BIH Arrhythmia Database is a research benchmark of annotated ambulatory ECG recordings. NeuroCardio's model work uses the AAMI beat-class framework; benchmark performance should not be read as clinical validation.</p>
          </div>
        </div>
        <div className="class-list">
          {classes.map(({ code, title, detail, color }, index) => (
            <article className={`class-entry class-${color}`} key={code}>
              <span className="class-index">0{index + 1}</span><b className="class-code">{code}</b>
              <div><h3>{title}</h3><p>{detail}</p></div>
              <span className="class-mark" aria-hidden="true">↗</span>
            </article>
          ))}
        </div>
      </section>

      <section className="limits-section">
        <div className="limits-number">!</div>
        <div className="limits-copy">
          <div className="section-marker"><span>05</span><i /> READ WITH CARE</div>
          <h2>Useful to study.<br /><em>Not a clinical answer.</em></h2>
          <p>NeuroCardio is a research prototype for retrospective beat-level analysis. It is not a medical device and is not intended for diagnosis, treatment, monitoring or care decisions. Predictions can be wrong, class performance may vary, and attribution methods do not establish clinical reasoning. Keep a qualified human in the loop.</p>
        </div>
        <Link className="limits-link" to="/app">Explore the research workspace <ArrowUpRight size={16} /></Link>
      </section>

      <section className="closing-cta">
        <div className="closing-orbit" aria-hidden="true" />
        <span className="closing-kicker">THE NEXT QUESTION IS YOURS</span>
        <h2>Start with the signal.</h2>
        <p>Explore a local recording, inspect a prediction, and make space for human review.</p>
        <Link to="/app" className="button-ink">Enter the workspace <ArrowUpRight size={16} /></Link>
        <div className="closing-trace"><Waves size={94} strokeWidth={.7} /></div>
      </section>

      <footer className="home-footer">
        <Link to="/" className="home-brand">
          <span className="home-brand-mark"><HeartPulse size={20} strokeWidth={1.6} /></span>
          <span>NeuroCardio<small>THE RHYTHM RESEARCH DESK</small></span>
        </Link>
        <p>A research prototype for looking closer at the rhythm.</p>
        <a href="#top" onClick={event => { event.preventDefault(); window.scrollTo({ top: 0, behavior: 'smooth' }) }}>Back to top ↑</a>
        <small>© NeuroCardio · Research use only</small>
      </footer>
    </main>
  )
}
