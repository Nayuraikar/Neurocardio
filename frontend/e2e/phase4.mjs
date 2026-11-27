// Phase 4 checks. Starts its own API and UI on spare ports with a temporary data directory (never your database).
//   npm run e2e            -> writes docs/phase4-test-report.md and exits 1 if anything fails
import { chromium } from '@playwright/test'
import { execFileSync, spawn } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..')
const PY = path.join(ROOT, 'backend/.venv/bin/python')
const MITDB = path.join(ROOT, 'results/mitdb')
const ART = path.join(ROOT, 'backend/artifacts')
/** Where the committed study reports live, relative to an artifacts directory. */
const STUDY = { selection: 'analysis/multidb/selection.json', validation: 'analysis/multidb/validation_report.json', final: 'analysis/multidb_final/final_report.json' }
const API_PORT = 8791, UI_PORT = 5191
const API = `http://127.0.0.1:${API_PORT}`, UI = `http://localhost:${UI_PORT}`
const TMP = fs.mkdtempSync(path.join(os.tmpdir(), 'nc-e2e-'))
const results = []
const record = (group, name, ok, detail = '') => { results.push({ group, name, ok, detail }); console.log(`${ok ? 'PASS' : 'FAIL'} [${group}] ${name}${detail ? ' — ' + detail : ''}`) }
const step = async (group, name, fn) => { try { const d = await fn(); record(group, name, true, d ?? '') } catch (e) { record(group, name, false, String(e.message ?? e).split('\n')[0].slice(0, 300)) } }
const expect = (cond, message) => { if (!cond) throw new Error(message) }
const sleep = ms => new Promise(r => setTimeout(r, ms))

// ---------------------------------------------------------------- servers
let apiProc = null, uiProc = null
async function waitFor(url, up = true, tries = 150) {
  for (let i = 0; i < tries; i++) {
    const ok = await fetch(url).then(r => r.ok, () => false)
    if (ok === up) return
    await sleep(500)
  }
  throw new Error(`timeout waiting for ${url} to be ${up ? 'up' : 'down'}`)
}
async function startApi(dataDir, artifactsDir = ART) {
  apiProc = spawn(PY, ['-m', 'uvicorn', 'app.main:app', '--port', String(API_PORT)], {
    cwd: path.join(ROOT, 'backend'), stdio: ['ignore', fs.openSync(path.join(TMP, 'api.log'), 'a'), fs.openSync(path.join(TMP, 'api.log'), 'a')],
    env: { ...process.env, NEUROCARDIO_DATA_DIR: dataDir, NEUROCARDIO_ARTIFACTS_DIR: artifactsDir },
  })
  await waitFor(`${API}/api/status`)
}
async function stopApi() { if (apiProc) { apiProc.kill('SIGTERM'); apiProc = null; await waitFor(`${API}/api/status`, false) } }
async function startUi() {
  uiProc = spawn(path.join(ROOT, 'frontend/node_modules/.bin/vite'), ['--port', String(UI_PORT), '--strictPort'], {
    cwd: path.join(ROOT, 'frontend'), stdio: 'ignore', env: { ...process.env, NEUROCARDIO_API: API },
  })
  await waitFor(UI)
}

const form = (names, extra = {}) => {
  const f = new FormData()
  for (const n of names) f.append('files', new Blob([fs.readFileSync(path.join(MITDB, n))]), n)
  for (const [k, v] of Object.entries(extra)) f.append(k, v)
  return f
}
const api = async (p, init) => { const r = await fetch(API + p, init); const t = await r.text(); let body; try { body = JSON.parse(t) } catch { body = t } return { status: r.status, body } }

// ---------------------------------------------------------------- independent truth
const json = f => JSON.parse(fs.readFileSync(f, 'utf8'))
const csv = f => { const [h, ...rows] = fs.readFileSync(f, 'utf8').trim().split(/\r?\n/).map(l => l.split(',')); return rows.map(r => Object.fromEntries(h.map((k, i) => [k, r[i]]))) }
const walk = (obj, p) => p.split(/(?<!\\)\./).map(s => s.replace(/\\\./g, '.')).reduce((o, k) => {
  if (o === undefined || o === null || !(k in Object(o))) throw new Error(`path segment "${k}" missing`)
  return o[k]
}, obj)

function truthFor(src, ctx) {
  const seedkey = /^seedkey:operating-points#models\.([a-z0-9_]+)\.points\.1\.thresholds\.(\d+)$/.exec(src)
  if (seedkey) { const frozen = json(path.join(ctx.artifacts, STUDY.selection)).candidates[seedkey[1]].frozen; if (!(seedkey[2] in frozen)) throw new Error('unknown seed'); return seedkey[2] }
  const prefix = /^(rowshare|sum|totalshare|shape|spaced|commas|datetime):(.*)$/.exec(src)
  if (prefix) {
    const [, op, rest] = prefix
    if (op === 'rowshare') { const m = /^(.*)\.(\d+)\.(\d+)$/.exec(rest); const row = truthFor(`${m[1]}.${m[2]}`, ctx); const t = row.reduce((a, b) => a + b, 0); return t ? row[+m[3]] / t : 0 }
    if (op === 'sum') return truthFor(rest, ctx).reduce((a, b) => a + b, 0)
    if (op === 'totalshare') { const m = /^(.*)\.([NSVF])$/.exec(rest); const o = truthFor(m[1], ctx); return o[m[2]] / Object.values(o).reduce((a, b) => a + b, 0) }
    const v = truthFor(rest, ctx)
    if (op === 'shape') return v.filter(d => d !== null).join('×') || '—'
    if (op === 'spaced') return v.join(' ')
    if (op === 'commas') return v.join(', ')
    if (op === 'datetime') return ctx.localTime(v)
  }
  const config = json(path.join(ROOT, 'backend/models/preprocessing_config.json'))
  if (src === 'windowseconds') return String(config.window_size / config.sampling_frequency)
  const [file, p] = src.split('#')
  const metrics = () => json(path.join(ctx.artifacts, 'metrics.json'))
  if (file === 'metrics') {
    if (p === 'variant_test_scores.length') return csv(path.join(ROOT, 'results/experiment_table_test.csv')).length
    return walk(metrics(), p)
  }
  if (file === 'datareport') return walk(json(path.join(ctx.artifacts, 'multidb_data_report.json')), p)
  if (file === 'minority') { const v = walk(csv(path.join(ROOT, 'results/minority_concentration_by_split.csv')), p); return /top_record$/.test(p) ? v : Number(v) }
  if (file === 'db') return walk(ctx.db(), p)
  if (file === 'config') {
    const model = metrics().model, types = model.layers.map(l => l.type)
    return {
      architecture: [types.includes('Conv1D') && 'CNN', types.includes('Bidirectional') && 'BiLSTM'].filter(Boolean).join('-'),
      lead: config.lead, sampling_rate_hz: config.sampling_frequency,
      model_sha256: fs.readFileSync(path.join(ROOT, 'backend/models/neurocardio_final.keras.sha256'), 'utf8').split(/\s+/)[0],
      input_shape: model.input_shape.filter(d => d !== null).join('×'),
      normalization: { per_window_zscore: 'per-window z-score' }[config.normalization],
      model_version: config.selected_model, parameter_count: model.parameter_count,
    }[p]
  }
  if (file === 'study') {
    const m = /^(validation|final)\.data\.(.*)$/.exec(p)
    return walk(json(path.join(ctx.artifacts, STUDY[m[1]])), m[2])
  }
  if (file === 'operating-points') {
    // rebuilt from the report files themselves, not from the API: point 0 is argmax, point 1 the frozen thresholds
    const validation = json(path.join(ctx.artifacts, STUDY.validation)), selection = json(path.join(ctx.artifacts, STUDY.selection))
    if (p.startsWith('majority.')) return walk(validation.majority_class, p.slice(9))
    const point = /^models\.([a-z0-9_]+)\.points\.([01])\.(validation|thresholds)\.(.*)$/.exec(p)
    if (!point) throw new Error(`unknown operating point "${p}"`)
    const [, name, index, kind, rest] = point
    if (kind === 'thresholds') { const [seed, i] = rest.split('.'); return selection.candidates[name].frozen[seed].thresholds[+i] }
    const summary = validation.results[name][index === '0' ? 'argmax' : 'chosen']
    const [cls, measure] = rest.split('_')
    return rest === 'macro_f1' || rest === 'accuracy' ? summary[rest].mean : summary.per_class[cls][measure].mean
  }
  throw new Error(`unknown source "${src}"`)
}
const format = (v, fmt) => typeof v !== 'number' ? String(v)
  : fmt === 'int' ? Math.round(v).toLocaleString('en-US') : fmt === 'pct1' ? `${(v * 100).toFixed(1)}%`
    : fmt === 'f3' ? v.toFixed(3) : fmt === 'f4' ? v.toFixed(4) : fmt === 'f2' ? v.toFixed(2) : String(v)

/** Digits outside a sourced element must be one of these, each with a stated reason. */
const ALLOWED_TEXT = [
  [/Macro-F1|macro-F1|\bF1\b/g, 'metric name'],
  [/\bde Chazal DS2 without record 202\b/g, 'name of a published partition (checked by the API against the split file)'],
  [/\bDS2\b|\btest_ds2\b|\bds2\b/g, 'data-set name'],
  [/ResNet1D|resnet1d/g, 'model name'],
  [/\bSHA-256\b/g, 'hash algorithm name'],
  [/95% confidence intervals|95% CI/g, 'method constant (artifact key ci95)'],
  [/\b2 s windows?\b/g, 'window length constant (config)'],
  [/\brecord 202\b/g, 'record identifier in a fixed caveat (it is a validation record in the model config)'],
]
const ALLOWED_AREAS = [
  ['svg', 'chart axis ticks and waveform axis labels'],
  ['[data-testid="activity"] table', 'timestamps and file names of stored events (not statistics)'],
  ['[data-testid="record-browser"]', 'per-record browser, checked separately against the WFDB annotations'],
]

async function checkNumbers(page, ctx, name, url, ready, group = 'numbers', tamper = null, mustFailWith = null, scope = 'main') {
  const wrap = mustFailWith ? async fn => { try { await fn() } catch (e) { expect(String(e.message).includes(mustFailWith), `failed for another reason: ${e.message}`); return `caught: ${mustFailWith}` } throw new Error('the checker did not notice') } : fn => fn()
  await step(group, name, () => wrap(async () => {
    await page.goto(UI + url); await page.getByText(ready).first().waitFor({ timeout: 60000 })
    await page.waitForLoadState('networkidle'); await page.waitForTimeout(500)
    expect(await page.locator('[data-state="loading"], [data-state="error"]').count() === 0, 'page still loading or showing an error')
    if (tamper) await page.evaluate(tamper)
    const items = await page.$$eval(`${scope} [data-source]`, els => els.map(e => ({ src: e.getAttribute('data-source'), fmt: e.getAttribute('data-fmt') || 'raw', text: e.textContent.trim() })))
    const times = {}
    for (const i of items.filter(x => x.src.startsWith('datetime:'))) { const iso = truthFor(i.src.slice(9), ctx); times[iso] = await page.evaluate(v => new Date(v).toLocaleString(), iso) }
    ctx.localTime = iso => times[iso]
    const bad = []
    for (const i of items) {
      try { const want = format(truthFor(i.src, ctx), i.fmt); if (want !== i.text) bad.push(`${i.src}: shows "${i.text}", source says "${want}"`) }
      catch (e) { bad.push(`${i.src}: ${e.message}`) }
    }
    expect(items.length > 0 || !name.includes('study'), 'no sourced values found')
    const loose = await page.evaluate(({ areas, texts, scope }) => {
      const root = document.querySelector(scope), out = []
      const w = document.createTreeWalker(root, NodeFilter.SHOW_TEXT)
      // join adjacent text nodes of one element so phrases split by React still match
      const seen = new Set()
      for (let n = w.nextNode(); n; n = w.nextNode()) {
        const el = n.parentElement
        if (seen.has(el)) continue
        seen.add(el)
        if (el.closest('[data-source]') || areas.some(a => el.closest(a))) continue
        let text = [...el.childNodes].filter(c => c.nodeType === 3).map(c => c.textContent).join('')
        for (const [source, flags] of texts) text = text.replace(new RegExp(source, flags), '')
        if (/\d/.test(text)) out.push(text.trim().slice(0, 90))
      }
      return out
    }, { areas: ALLOWED_AREAS.map(a => a[0]), texts: ALLOWED_TEXT.map(([r]) => [r.source, r.flags]), scope })
    const budgets = loose.filter(t => t === '5%' || t === '10%')  // review budgets are artifact keys
    const rest = loose.filter(t => !budgets.includes(t))
    expect(!bad.length, `${bad.length} mismatches, e.g. ${bad.slice(0, 3).join(' | ')}`)
    expect(!rest.length, `${rest.length} unsourced numbers, e.g. ${rest.slice(0, 6).join(' | ')}`)
    // accuracy must sit in a block that also shows the majority-class figure
    const orphan = await page.$$eval('main [data-source]', els => els.filter(e => /\.accuracy(\.mean)?$/.test(e.getAttribute('data-source')) && !/majority/.test(e.getAttribute('data-source')))
      .filter(e => !(e.closest('[data-testid="headline"], table')?.querySelector('[data-source*="majority"]'))).map(e => e.getAttribute('data-source')))
    expect(!orphan.length, `accuracy without a baseline beside it: ${orphan.slice(0, 3).join(', ')}`)
    return `${items.length} sourced values match; 0 unsourced numbers`
  }))
}

// ---------------------------------------------------------------- run
const browser = await chromium.launch()
const page = await browser.newPage({ viewport: { width: 1360, height: 900 }, locale: 'en-US' })
const pageErrors = []
page.on('pageerror', e => pageErrors.push(e.message))
const dataDir = path.join(TMP, 'data')
const dbFile = path.join(dataDir, 'neurocardio.sqlite3')
const ctx = { artifacts: ART, db: () => JSON.parse(execFileSync(PY, [path.join(ROOT, 'frontend/e2e/truth.py'), ROOT, dbFile, 'db'], { encoding: 'utf8' })) }

try {
  await startApi(dataDir); await startUi()
  let analysisId = null

  // ---- demo flow through the UI
  await step('flow', 'empty overview and history', async () => {
    await page.goto(UI); await page.getByText('Nothing recorded yet').waitFor()
    await page.goto(`${UI}/history`); await page.getByText('No analyses yet').waitFor()
  })
  await step('flow', 'upload record 101 in Analysis Studio', async () => {
    await page.goto(`${UI}/studio`)
    await page.getByTestId('file-input').setInputFiles(['101.hea', '101.dat', '101.atr'].map(f => path.join(MITDB, f)))
    await page.getByRole('button', { name: 'Upload and validate' }).click()
    await page.getByText('The file passed upload validation.').waitFor({ timeout: 90000 })
  })
  await step('flow', 'process and reach results', async () => {
    await page.getByRole('button', { name: 'Analyse recording' }).click()
    await page.getByText('Analysis complete.').waitFor({ timeout: 180000 })
    analysisId = new URL(page.url()).searchParams.get('analysis'); expect(analysisId, 'no analysis id in the URL')
    await page.getByText('V-class windows present:').waitFor()
    expect(await page.getByText('S and F weakness').count() > 0, 'no S/F weakness note on results')
  })
  if (!analysisId) throw new Error('no analysis; the remaining flow cannot run')
  const q = `?analysis=${analysisId}&window=4`
  await step('flow', 'selection restored after leaving the URL', async () => {
    await page.goto(`${UI}/quality`); await page.getByText('Overall:').waitFor({ timeout: 60000 })
  })
  await step('flow', 'ECG viewer and playback', async () => {
    await page.goto(`${UI}/viewer${q}`); await page.getByText('Selected window').waitFor({ timeout: 60000 })
    await page.getByRole('button', { name: '4x' }).click(); await page.getByRole('button', { name: 'Play', exact: true }).click(); await page.waitForTimeout(1500)
    await page.getByRole('button', { name: 'Pause' }).click()
    const text = await page.getByTestId('live-class').innerText(); expect(/window #\d+/.test(text), `no live window: ${text}`)
    await page.getByRole('button', { name: '0.5x' }).click()
    return text.split('\n')[0]
  })
  await step('flow', 'three-panel view', async () => { await page.goto(`${UI}/panels${q}`); await page.getByText('3 · Model interpretation').waitFor({ timeout: 60000 }) })
  await step('flow', 'confidence explorer, temperature toggle', async () => {
    await page.goto(`${UI}/confidence${q}`); await page.getByRole('button', { name: 'Temperature-scaled' }).click()
    await page.getByText('fitted on validation data only').waitFor(); await page.getByText('It is not clinical certainty.').first().waitFor()
  })
  await step('flow', 'sensitivity slider (real artifacts) and priority queue', async () => {
    await page.goto(`${UI}/sensitivity${q}`); await page.getByText('Priority review queue').waitFor({ timeout: 60000 })
    // the primary model kept argmax, so it has one frozen point and no slider
    await page.getByText('Selection kept plain argmax for this model').waitFor({ timeout: 60000 })
    expect(await page.getByRole('slider').count() === 0, 'a slider is shown for a model with one operating point')
  })
  await step('flow', 'XAI: Integrated Gradients, completeness, Grad-CAM comparison', async () => {
    await page.goto(`${UI}/xai${q}`); await page.getByText('Completeness for this window').waitFor({ timeout: 240000 })
    await page.getByRole('tab', { name: 'Side by side' }).click(); await page.getByText('Rank correlation between').waitFor({ timeout: 240000 })
  })
  await step('flow', 'misclassification lab', async () => { await page.goto(`${UI}/errors${q}`); await page.getByText('Errors by class').waitFor({ timeout: 60000 }) })
  await step('flow', 'save a review without changing the prediction', async () => {
    const before = (await api(`/api/analyses/${analysisId}/beats?limit=10000`)).body
    await page.goto(`${UI}/history/${analysisId}`)
    await page.getByRole('checkbox', { name: 'Show every window' }).check()
    await page.getByRole('group', { name: 'Reviewer label' }).getByRole('button', { name: /^S ·/ }).click()
    await page.getByRole('button', { name: 'Save assessment' }).click(); await page.getByText(/Saved reviewer label S/).waitFor()
    await page.getByRole('button', { name: 'Mark analysis reviewed' }).click(); await page.getByText('Analysis marked reviewed.').waitFor()
    const after = (await api(`/api/analyses/${analysisId}/beats?limit=10000`)).body
    expect(JSON.stringify(before.map(b => b.prediction)) === JSON.stringify(after.map(b => b.prediction)), 'predictions changed')
    expect(after.some(b => b.review?.reviewed_label === 'S'), 'review not stored')
  })
  await step('flow', 'report: preview, PDF and CSV download', async () => {
    await page.goto(`${UI}/history/${analysisId}/report`)
    await page.getByRole('button', { name: 'Preview PDF' }).click(); await page.locator('iframe[title="Report preview"]').waitFor({ timeout: 120000 })
    const sizes = []
    for (const kind of ['PDF', 'CSV']) {
      const [d] = await Promise.all([page.waitForEvent('download', { timeout: 120000 }), page.getByRole('button', { name: `Download ${kind}` }).click()])
      const f = path.join(TMP, `r.${kind}`); await d.saveAs(f); const buf = fs.readFileSync(f)
      expect(kind === 'PDF' ? buf.subarray(0, 4).toString() === '%PDF' : buf.toString().includes('# model_version:'), `bad ${kind}`)
      sizes.push(`${kind} ${buf.length} bytes`)
    }
    return sizes.join(', ')
  })
  await step('flow', 'demo playground shows source splits', async () => {
    await page.goto(`${UI}/demo`); await page.getByTestId('source-split').first().waitFor({ timeout: 60000 })
    expect(/(train|validation) patient/.test(await page.getByTestId('source-split').first().innerText()), 'no split label')
  })
  await step('flow', 'challenge round of 8', async () => {
    await page.goto(`${UI}/challenge`); await page.getByRole('button', { name: 'Start a round' }).click()
    for (let i = 1; i <= 8; i++) {
      await page.getByText(`Window ${i} of 8`).waitFor(); expect(/(train|validation) patient/.test(await page.getByTestId('source-split').innerText()), 'no split label')
      await page.getByRole('button', { name: /^N ·/ }).click(); await page.getByRole('button', { name: i === 8 ? 'See round summary' : 'Next window' }).click()
    }
    await page.getByText('Window by window').waitFor()
    expect(await page.getByTestId('headline').count() === 2, 'scores missing the headline block')
  })
  await step('flow', 'batch with one good and one paced recording', async () => {
    await page.goto(`${UI}/batch`)
    await page.getByTestId('batch-input').setInputFiles(['115.hea', '115.dat', '115.atr', '217.hea', '217.dat', '217.atr'].map(f => path.join(MITDB, f)))
    await page.getByRole('button', { name: 'Analyse 2 recordings' }).click(); await page.getByText('Partial result:').waitFor({ timeout: 180000 })
  })
  await step('flow', 'comparison workspace and presentation', async () => {
    await page.goto(`${UI}/compare`); await page.getByText('Windows by model class').first().waitFor({ timeout: 60000 })
    await page.goto(`${UI}/present`); await page.locator('.present h1').waitFor()
    for (let i = 0; i < 8; i++) await page.keyboard.press('ArrowRight')
    expect((await page.locator('.present h1').innerText()) === 'Limits', 'did not reach the last slide')
  })
  await step('flow', 'every route renders at 390 px without horizontal scroll', async () => {
    await page.setViewportSize({ width: 390, height: 800 })
    const routes = ['/', '/studio', '/viewer', '/panels', '/quality', '/confidence', '/sensitivity', '/xai', '/errors', '/performance', '/performance/recall', '/comparison', '/model-card', '/dataset', '/demo', '/challenge', '/history', `/history/${analysisId}`, `/history/${analysisId}/report`, '/batch', '/compare', '/present', '/nope']
    const wide = []
    for (const r of routes) { await page.goto(UI + r + (r.includes('analysis') ? '' : '')); await page.waitForLoadState('networkidle'); const o = await page.evaluate(() => document.documentElement.scrollWidth - innerWidth); if (o > 0) wide.push(`${r} +${o}px`) }
    await page.setViewportSize({ width: 1360, height: 900 })
    expect(!wide.length, wide.join(', ')); return `${routes.length} routes`
  })

  // ---- numbers, against the real committed artifacts
  await checkNumbers(page, ctx, 'Overview', '/', 'Recordings stored')
  await checkNumbers(page, ctx, 'Dataset Explorer', '/dataset', 'Chosen from')
  await checkNumbers(page, ctx, 'Model Performance (app model and real study)', '/performance', 'Layers of the loaded model')
  await checkNumbers(page, ctx, 'Model Comparison Lab (real study)', '/comparison', 'What is compared')
  await checkNumbers(page, ctx, 'Model Card (app model and real study)', '/model-card', 'Other limits')
  await checkNumbers(page, ctx, 'Sensitivity, primary model (real study)', '/sensitivity', 'Selection kept plain argmax', 'numbers', null, null, '.sensitivity-study')
  await step('numbers', 'sensitivity slider moves between the two real frozen points of every model that has them', async () => {
    const selection = json(path.join(ART, STUDY.selection))
    const two = Object.entries(selection.candidates).filter(([, c]) => c.rule === 'thresholds').map(([m]) => m)
    await page.goto(`${UI}/sensitivity`); const model = page.getByRole('combobox', { name: 'Study model' }); await model.waitFor()
    const label = page.locator('.sensitivity-study strong', { hasText: /^(Plain argmax|Validation-frozen class thresholds)$/ })
    const checked = []
    for (const name of two) {
      await model.selectOption(name)
      const recall = () => page.$$eval('.sensitivity-study [data-source$=".S_recall"], .sensitivity-study [data-source$=".F_recall"]', els => els.reduce((a, e) => a + Number(e.textContent), 0))
      const slider = page.getByRole('slider')
      const lowLabel = await label.innerText(), low = await recall(); await slider.fill('1'); const highLabel = await label.innerText(), high = await recall()
      expect(lowLabel !== highLabel && low <= high, `${name}: ${lowLabel} ${low} -> ${highLabel} ${high}`)
      for (const position of ['0', '1']) {
        await slider.fill(position)
        const items = await page.$$eval('.sensitivity-study [data-source]', els => els.map(e => ({ src: e.getAttribute('data-source'), fmt: e.getAttribute('data-fmt'), text: e.textContent.trim() })))
        const bad = items.filter(i => format(truthFor(i.src, ctx), i.fmt) !== i.text)
        expect(!bad.length, `${name}: ${bad.slice(0, 3).map(i => `${i.src} shows ${i.text}`).join(' | ')}`)
      }
      checked.push(name)
    }
    expect(checked.length > 0, 'no model has two operating points')
    return checked.join(', ')
  })
  const control = (name, tamper, reason, url = '/performance', ready = 'Layers of the loaded model') => checkNumbers(page, ctx, `negative control: ${name}`, url, ready, 'numbers', tamper, reason)
  await control('a changed value is detected', () => { document.querySelector('main [data-source="metrics#test.macro.f1"]').textContent = '0.999' }, 'mismatches')
  await control('an unsourced number is detected', () => { document.querySelector('main h1').append(' 12345') }, 'unsourced numbers')
  await control('accuracy without a baseline is detected', () => { document.querySelectorAll('main [data-source="metrics#test.majority_class_baseline_accuracy"]').forEach(e => { e.setAttribute('data-source', 'metrics#test.accuracy'); e.textContent = document.querySelector('main [data-source="metrics#test.accuracy"]').textContent; e.setAttribute('data-fmt', 'pct1') }) }, 'without a baseline')
  await control('a changed study test score is detected', () => { document.querySelector('main [data-source="study#final.data.sets.ds2_without_202.gbm.per_class.S.recall.mean"]').textContent = '0.656' }, 'mismatches', '/comparison', 'What is compared')

  // ---- how the real study is presented
  const finalReport = json(path.join(ART, STUDY.final)), PRIMARY = finalReport.primary
  const FIXED = ['cnn_bilstm', 'rr_fusion', 'rr_fusion_attn', 'rr_fusion_attn_shead', 'cnn', 'lstm', 'resnet1d', 'gbm', 'stacked_ensemble']
  const tables = testid => page.$$eval(`[data-testid="${testid}"] table`, ts => ts.map(t => ({ caption: t.caption?.textContent ?? '', head: [...t.tHead.rows[0].cells].map(c => c.textContent), rows: [...t.tBodies[0].rows].map(r => [...r.cells].map(c => c.textContent)) })))
  for (const [url, name, secondaries] of [['/comparison', 'Comparison Lab', true], ['/performance', 'Model Performance', false], ['/model-card', 'Model Card', false]]) {
    await step('study', `${name}: every test table has the deployed model and the majority class; primary first, never ranked by score`, async () => {
      await page.goto(UI + url); await page.getByTestId('testset-incart_heldout').waitFor({ timeout: 60000 })
      let count = 0
      for (const set of Object.keys(finalReport.sets)) {
        const found = await tables(`testset-${set}`)
        expect(found.length === (secondaries ? 3 : 1), `${set}: ${found.length} tables`)
        for (const t of found) {
          const text = t.rows.map(r => r.join(' ')).join('\n')
          expect(/Deployed CNN-BiLSTM/.test(text) && /Majority class/.test(text), `${set}: a table lacks the deployed model or the majority class (${t.caption.slice(0, 40)})`)
          if (t.head.includes('Accuracy')) {
            expect(t.head.indexOf('Macro-F1') === 2 && t.head[3] === 'S recall' && t.head[4] === 'F recall' && t.head.at(-1) === 'Accuracy', `column order: ${t.head.join(', ')}`)
            expect(t.rows[0][1] === 'PRIMARY' && t.rows[1][1] === 'Reference' && t.rows[2][1] === 'Baseline', `first rows: ${t.rows.slice(0, 3).map(r => r[1]).join(', ')}`)
            const rest = t.rows.slice(3)
            expect(rest.length === (secondaries ? FIXED.length - 1 : 0) && rest.every(r => r[1] === 'SECONDARY'), `${set}: ${rest.length} secondary rows`)
            const labels = { cnn_bilstm: 'CNN-BiLSTM', rr_fusion: 'CNN-BiLSTM + RR fusion', rr_fusion_attn: 'RR fusion + attention', rr_fusion_attn_shead: 'RR fusion + attention + S head', cnn: 'CNN only', lstm: 'LSTM only', resnet1d: 'ResNet1D', gbm: 'Gradient-boosted trees (GBM)', stacked_ensemble: 'Stacked ensemble' }
            if (secondaries) expect(rest.map(r => r[0]).join('|') === FIXED.filter(m => m !== PRIMARY).map(m => labels[m]).join('|'), `order: ${rest.map(r => r[0]).join(', ')}`)
          }
          count++
        }
      }
      const main = await page.locator('main').innerText()
      expect(/frozen on validation before the test was opened/.test(main), 'primary is not described as frozen before the test')
      expect(/unseen patients/.test(main), 'INCART wording missing')
      expect(await page.getByTestId('testset-ds2_without_202').textContent().then(t => /fair comparison with the deployed model/.test(t)), '21-record set not marked as the fair comparison')
      expect(await page.getByRole('heading', { name: /validation$/i }).count() >= 1, 'no separate validation panel')
      return `${count} test tables over ${Object.keys(finalReport.sets).length} sets`
    })
  }
  for (const [url, name] of [['/performance', 'Model Performance'], ['/model-card', 'Model Card'], ['/comparison', 'Comparison Lab']]) {
    await step('study', `${name}: says the live model is the deployed CNN-BiLSTM and the primary is not served or applied`, async () => {
      await page.goto(UI + url); await page.getByTestId('live-model-notice').getByText('The API cannot serve').waitFor({ timeout: 60000 })
      const text = await page.getByTestId('live-model-notice').textContent()
      for (const part of ['is the deployed CNN-BiLSTM', 'not the multi-database study’s primary model (Gradient-boosted trees (GBM))', 'applied to uploaded recordings', 'The API cannot serve Gradient-boosted trees (GBM)']) expect(text.includes(part), `notice lacks "${part}"`)
    })
  }
  await step('study', 'findings state what the final report’s flags say', async () => {
    await page.goto(`${UI}/comparison`); const box = page.getByTestId('findings'); await box.waitFor({ timeout: 60000 })
    const gains = Object.values(finalReport.sets).map(s => s.primary_vs_deployed).filter(g => g.headline_comparison)
    expect(gains.length === 2 && gains.every(g => g.macro_f1.meaningful && ['S_f1', 'F_f1', 'S_recall', 'F_recall'].every(k => !g[k].meaningful)), 'the report flags are not what this check expects')
    const macro = await page.getByTestId('finding-macro').innerText(), minority = await page.getByTestId('finding-minority').innerText()
    expect(/improved on DS2 without record 202; improved on INCART unseen patients/.test(macro), macro)
    expect(/did NOT meaningfully improve/.test(minority), minority)
    expect(minority.includes('Lower than the deployed model: S recall on INCART unseen patients'), minority)
    expect((await page.getByTestId('finding-f-recall').innerText()).includes('includes zero'), 'F recall interval wording')
    expect((await page.getByTestId('finding-validation-gap').innerText()).includes('validation overstated it'), 'validation gap wording')
    const conc = await page.getByTestId('finding-concentration').innerText()
    expect(conc.includes('mitdb:232') && conc.includes('mitdb:213'), conc)
    expect((await box.innerText()).includes('upper bound') && (await box.innerText()).includes('had been scored before this study'), 'review or DS2 disclosure missing')
  })

  await step('numbers', 'Dataset record browser matches WFDB annotations (record 100)', async () => {
    const t = JSON.parse(execFileSync(PY, [path.join(ROOT, 'frontend/e2e/truth.py'), ROOT, dbFile, 'record'], { encoding: 'utf8' }))
    await page.goto(`${UI}/dataset`); await page.getByText(/\d+ of \d+/).first().waitFor({ timeout: 60000 })
    const options = await page.getByRole('combobox', { name: 'Class' }).locator('option').allInnerTexts()
    for (const c of 'NSVF') expect(options.includes(`${c} (${t.counts[c]} annotated)`), `count for ${c}: ${options.join(', ')}`)
    await page.getByText(`1 of ${t.complete_N}`).waitFor()
  })
  await step('numbers', 'representative examples all come from training records', async () => {
    const split = json(path.join(ROOT, 'backend/models/preprocessing_config.json')).split
    const reps = (await api('/api/dataset/representatives')).body.examples
    expect(reps.every(e => split.train_records.includes(e.record)), 'an example is not from a training record')
  })

  // ---- error paths (API)
  const apiCase = (name, run, status, text) => step('errors', name, async () => {
    const r = await run(); const body = JSON.stringify(r.body)
    expect(r.status === status, `HTTP ${r.status}: ${body.slice(0, 160)}`); if (text) expect(body.includes(text), `message lacks "${text}": ${body.slice(0, 160)}`)
    return `HTTP ${r.status}`
  })
  const upload = (names, extra) => () => api('/api/upload', { method: 'POST', body: form(names, extra) })
  await apiCase('paced recording rejected', upload(['217.hea', '217.dat', '217.atr']), 422, 'paced')
  await apiCase('.hea without .dat', upload(['101.hea']), 422, '.dat')
  await apiCase('unsupported file type', () => { const f = new FormData(); f.append('files', new Blob(['x']), 'a.txt'); return api('/api/upload', { method: 'POST', body: f }) }, 415, 'unsupported')
  const csvCase = (text, extra) => () => { const f = new FormData(); f.append('files', new Blob([text]), 'a.csv'); for (const [k, v] of Object.entries(extra)) f.append(k, v); return api('/api/upload', { method: 'POST', body: f }) }
  await apiCase('CSV without sampling rate', csvCase('MLII\n1\n2\n', { lead: 'MLII' }), 422, 'sampling_rate')
  await apiCase('CSV at the wrong rate', csvCase('MLII\n' + '1\n2\n'.repeat(500), { lead: 'MLII', sampling_rate: '250' }), 422, '360')
  await apiCase('unknown analysis', () => api('/api/analyses/nope'), 404)
  await apiCase('unknown recording', () => api('/api/recordings/nope/signal'), 404)
  await apiCase('bad report section', () => api(`/api/reports/${analysisId}?sections=zzz`), 422, 'zzz')
  await apiCase('analyse unknown recording', () => api('/api/analyze?record_id=nope', { method: 'POST' }), 404)

  // ---- error paths (UI)
  await step('errors', 'UI shows a specific upload error', async () => {
    await page.goto(`${UI}/studio`); await page.getByRole('button', { name: 'New upload' }).click({ timeout: 3000 }).catch(() => {})
    await page.getByTestId('file-input').setInputFiles(['217.hea', '217.dat', '217.atr'].map(f => path.join(MITDB, f)))
    await page.getByRole('button', { name: 'Upload and validate' }).click(); await page.getByText('Paced recording', { exact: true }).waitFor({ timeout: 60000 })
  })
  await step('errors', 'UI explains a deleted analysis id', async () => { await page.goto(`${UI}/viewer?analysis=does-not-exist`); await page.getByText('The selected analysis no longer exists.').waitFor() })
  await step('errors', 'partial failure: one panel fails, the page stays', async () => {
    await page.route('**/api/overview', r => r.fulfill({ status: 500, contentType: 'application/json', body: '{"detail":"forced failure"}' }))
    await page.goto(UI); await page.getByText('forced failure').first().waitFor(); await page.getByRole('heading', { name: 'Analysis Studio' }).waitFor()
    await page.unroute('**/api/overview'); await page.getByRole('button', { name: 'Retry' }).first().click(); await page.getByText('Recordings stored').waitFor()
  })
  await step('errors', 'missing stored signal file gives a clear error', async () => {
    const rec = (await api(`/api/analyses/${analysisId}`)).body.record_id
    const file = path.join(dataDir, 'records', rec, 'signal.npy'); fs.renameSync(file, file + '.bak')
    try { const r = await api(`/api/recordings/${rec}/signal`); expect(r.status === 410, `HTTP ${r.status}`) } finally { fs.renameSync(file + '.bak', file) }
  })
  await stopApi()
  await step('errors', 'backend down: banner and status, no crash', async () => {
    await page.goto(UI); await page.getByText('Backend unavailable').waitFor({ timeout: 30000 }); await page.getByText('API: unavailable').waitFor()
  })
  await step('errors', 'interrupted analysis is marked failed after restart', async () => {
    execFileSync(PY, ['-c', `import sqlite3;c=sqlite3.connect(${JSON.stringify(dbFile)});c.execute("update analyses set status='running' where id=?",(${JSON.stringify(analysisId)},));c.commit()`])
    await startApi(dataDir)
    const a = (await api(`/api/analyses/${analysisId}`)).body; expect(a.status === 'failed' && a.error_message.includes('interrupted'), JSON.stringify(a).slice(0, 200)); await stopApi()
  })

  // ---- artifact states: empty, corrupt, smoke, wrong shape, stale
  const artDir = (name, files) => { const d = path.join(TMP, name); fs.mkdirSync(d, { recursive: true }); for (const [f, v] of Object.entries(files)) { fs.mkdirSync(path.dirname(path.join(d, f)), { recursive: true }); fs.writeFileSync(path.join(d, f), typeof v === 'string' ? v : JSON.stringify(v)) } return d }
  const withArtifacts = async (name, files, fn) => { await startApi(path.join(TMP, 'data2'), artDir(name, files)); try { await fn() } finally { await stopApi() } }
  await withArtifacts('none', {}, async () => {
    await step('errors', 'no artifacts: honest empty states', async () => {
      await page.goto(`${UI}/performance`); await page.getByText('Model evaluation has not been generated.').waitFor()
      await page.goto(`${UI}/comparison`); await page.getByText('Final evaluation not run yet.').waitFor()
      await page.goto(`${UI}/dataset`); await page.getByText('Study data report not generated yet.').waitFor()
      await page.goto(`${UI}/demo`); await page.getByText('Demo samples have not been built.').waitFor()
    })
  })
  const real = kind => json(path.join(ART, STUDY[kind]))
  await withArtifacts('broken', { 'metrics.json': '{oops', [STUDY.final]: { ...real('final'), smoke: true }, [STUDY.selection]: '[' }, async () => {
    await step('errors', 'corrupt metrics and smoke final report are refused', async () => {
      await page.goto(`${UI}/performance`); await page.getByText('could not be read').first().waitFor()
      await page.goto(`${UI}/comparison`); await page.getByText('smoke run').waitFor()
      await page.goto(`${UI}/sensitivity`); await page.getByText('selection.json could not be read').waitFor()
    })
  })
  await withArtifacts('shape', { [STUDY.validation]: { smoke: false, primary: 'cnn' } }, async () => {
    await step('errors', 'wrong-shaped artifact shows an error, not a blank page', async () => {
      await page.goto(`${UI}/comparison`); await page.getByText('This page could not be displayed').waitFor(); await page.getByRole('link', { name: 'Model Card' }).waitFor()
    })
  })
  const stale = json(path.join(ART, 'metrics.json')); stale.model.sha256 = 'different'
  await withArtifacts('stale', { 'metrics.json': stale }, async () => {
    await step('errors', 'stale metrics are flagged', async () => { await page.goto(`${UI}/performance`); await page.getByText('may be out of date').waitFor() })
  })

  // ---- perturbed COPIES of the real reports, in a temporary directory: the wording and warnings must follow the files
  const changed = real('final')
  changed.sets.ds2_without_202.primary_vs_deployed.macro_f1.meaningful = false
  changed.sets.incart_heldout.primary_vs_deployed.S_f1.meaningful = true
  changed.sets.ds2_without_202.gbm.per_class.F.recall.mean = 0.5
  const selectionChanged = { ...real('selection'), note: 'edited after the test' }
  const dir = artDir('perturbed', { [STUDY.final]: changed, [STUDY.validation]: real('validation'), [STUDY.selection]: selectionChanged,
    'metrics.json': json(path.join(ART, 'metrics.json')), 'multidb_data_report.json': json(path.join(ART, 'multidb_data_report.json')) })
  await startApi(path.join(TMP, 'data3'), dir)
  await checkNumbers(page, { ...ctx, artifacts: dir }, 'perturbed copy: numbers follow the files', '/comparison', 'What is compared', 'study')
  await step('study', 'perturbed copy: findings wording follows the changed flags', async () => {
    await page.goto(`${UI}/comparison`); await page.getByTestId('findings').waitFor({ timeout: 60000 })
    const macro = await page.getByTestId('finding-macro').innerText(), minority = await page.getByTestId('finding-minority').innerText()
    expect(macro.includes('did not meaningfully improve on DS2 without record 202'), macro)
    expect(minority.includes('meaningful improvement under the rule only for S F1 on INCART unseen patients'), minority)
  })
  await step('study', 'perturbed copy: a selection file edited after the test raises the provenance warning', async () => {
    await page.goto(`${UI}/comparison`); await page.getByText('Provenance check failed.').waitFor({ timeout: 60000 })
  })
  await stopApi()
  record('flow', 'no uncaught page errors during the run', pageErrors.length === 0, pageErrors.slice(0, 3).join(' | '))
} catch (e) {
  record('run', 'checks aborted', false, String(e.message ?? e).slice(0, 300))
} finally {
  await browser.close(); apiProc?.kill(); uiProc?.kill()
}

// ---------------------------------------------------------------- placeholder grep
const sources = dir => fs.readdirSync(dir, { recursive: true }).map(String).filter(f => /\.(tsx?|py|css|html|json)$/.test(f) && !f.includes('__pycache__')).map(f => path.join(dir, f))
const files = [...sources(path.join(ROOT, 'frontend/src')), ...sources(path.join(ROOT, 'backend/app')), path.join(ROOT, 'frontend/index.html')]
const grep = (name, regex, skip = () => false) => {
  const hits = files.flatMap(f => fs.readFileSync(f, 'utf8').split('\n').map((l, i) => regex.test(l) && !skip(l) ? `${path.relative(ROOT, f)}:${i + 1}: ${l.trim().slice(0, 100)}` : null).filter(Boolean))
  record('grep', name, hits.length === 0, hits.slice(0, 8).join(' ¦ '))
}
grep('no placeholder or unfinished-work text', /not rebuilt|\bTODO\b|\bFIXME\b|lorem|coming soon|\bTBD\b|\bmock|dummy|fake|hard-?coded/i)
grep('"placeholder" only as an HTML input hint', /placeholder/i, l => /placeholder=("|\{)/.test(l))
grep('no known study statistics typed into source', /\b0\.(656|174|188|555|560|620|483|472|457|529)|\b75\.2\s?%|\b93\.[03]\s?%/)
grep('no known statistics typed into source', /0\.906|0\.887|0\.473|90\.6\s?%|88\.7|0\.097|0\.057|0\.917|142[,_]?980|47[,_]?518|98\.9|\b2[,_]?953\b/)
grep('no overclaiming wording', /clinical(ly)?[- ]grade|diagnoses |detects arrhythmia|real-time monitoring|unseen database(?!:)/i, l => /not an unseen database/.test(l))

// ---------------------------------------------------------------- report
const failed = results.filter(r => !r.ok)
const sha = execFileSync('git', ['-C', ROOT, 'rev-parse', '--short', 'HEAD'], { encoding: 'utf8' }).trim()
const titles = { flow: 'End-to-end demo flow', numbers: 'Numbers against artifacts', study: 'How the real study results are presented', errors: 'Error paths', grep: 'Placeholder and wording grep', run: 'Run' }
const md = ['# Phase 4 automated checks', '', `Generated by \`npm run e2e\` at ${new Date().toISOString()}, on top of commit ${sha}. This file is overwritten on each run.`, '',
  `**${results.length - failed.length} passed, ${failed.length} failed.**`, '',
  ...Object.entries(titles).flatMap(([g, t]) => { const rows = results.filter(r => r.group === g); return rows.length ? [`## ${t}`, '', '| Result | Check | Detail |', '|---|---|---|', ...rows.map(r => `| ${r.ok ? 'PASS' : '**FAIL**'} | ${r.name} | ${r.detail.replace(/\|/g, '/')} |`), ''] : [] }),
  '## How numbers are checked', '', 'Each statistic on screen carries a `data-source` attribute. The checker resolves it from the files on disk (and, for stored counts, with its own SQL), formats it the same way and compares the text. Any other digit on the page fails the check unless it is in this allow-list:', '',
  ...ALLOWED_AREAS.map(([a, why]) => `- area \`${a}\`: ${why}`), ...ALLOWED_TEXT.map(([r, why]) => `- text \`${r.source}\`: ${why}`), '- cells `5%` and `10%` in the study review table: review budgets (artifact keys)', '',
  'Study numbers are compared with the committed reports in `backend/artifacts/analysis/`. A match shows that the page displays what the file says. It does not show that the evaluation behind the file is correct; that was not re-run.', '',
  'The "perturbed copy" checks edit copies of those reports in a temporary directory. The committed files are not modified.', '',
  '`npm run e2e` rewrites this file. Notes added by hand belong in the section it appends from `frontend/e2e/report-notes.md`.', '',
  ...(fs.existsSync(path.join(ROOT, 'frontend/e2e/report-notes.md')) ? [fs.readFileSync(path.join(ROOT, 'frontend/e2e/report-notes.md'), 'utf8')] : [])]
fs.writeFileSync(path.join(ROOT, 'docs/phase4-test-report.md'), md.join('\n'))
fs.rmSync(TMP, { recursive: true, force: true })
console.log(`\n${results.length - failed.length} passed, ${failed.length} failed`)
process.exit(failed.length ? 1 : 0)
