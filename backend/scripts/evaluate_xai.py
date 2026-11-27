"""XAI audit: Grad-CAM check, IG completeness, weight randomisation, deletion/insertion.
Resumable and observable. Needs results/test_examples_for_xai.npz (gitignored).
Run: backend/.venv/bin/python -u backend/scripts/evaluate_xai.py --n-per-class 30
"""
import argparse, json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorflow as tf
import keras
from scipy.stats import spearmanr
from scipy.ndimage import uniform_filter1d, gaussian_filter1d

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'backend'))
from app.ml.inference import ECGClassifier            # noqa: E402
from app.ml.explainability import ModelExplainer      # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--n-per-class', type=int, default=30)
ap.add_argument('--ig-steps', type=int, default=50)
ap.add_argument('--check-steps', type=int, default=200)
ap.add_argument('--n-check', type=int, default=5)
ap.add_argument('--n-sanity', type=int, default=10)
ap.add_argument('--seed', type=int, default=20240611)
ap.add_argument('--tag', default='', help='suffix for output files, e.g. _smoke')
args = ap.parse_args()

C = list('NSVF')
ART = ROOT / 'backend/artifacts'
PARTIAL = ART / f'xai_eval_partial{args.tag}.json'
FINAL = ART / f'xai_evaluation{args.tag}.json'
T0 = time.time()


def log(msg):
    print(f'[{datetime.now():%H:%M:%S}] {msg}', flush=True)


def save(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)          # atomic: a stop never leaves a half-written file


# ------------------------------------------------------------------ model + data
log('loading model')
clf = ECGClassifier()
model = clf.model
gpus = tf.config.list_physical_devices('GPU')
log(f'TF {tf.__version__} | physical devices: {[d.device_type for d in tf.config.list_physical_devices()]} '
    f'| tensor device: {model(tf.zeros((1, 720, 1)), training=False).device}')

REF = ROOT / 'results/test_examples_for_xai.npz'
if not REF.exists():
    sys.exit(f'{REF} not found (gitignored, >100MB). XAI evaluation needs it.')
ART.mkdir(parents=True, exist_ok=True)
z = np.load(REF)
pred, true = z['pred'].astype(int), z['y'].astype(int)
X = z['X'].astype(np.float32)
X = X if X.ndim == 3 else X[..., None]

# stratified by PREDICTED class (explanations target the predicted class); seeded per class
sample = {c: np.sort(np.random.default_rng([args.seed, k]).choice(
    np.flatnonzero(pred == k), min(args.n_per_class, int((pred == k).sum())), replace=False))
    for k, c in enumerate(C)}
N_TOTAL = sum(len(v) for v in sample.values())

# ------------------------------------------------------------------ compiled kernels
TRACES = {}
SIG_X = tf.TensorSpec([None, 720, 1], tf.float32)
SIG_T = tf.TensorSpec([], tf.int32)


class Kernels:
    """One set of tf.functions per model object. Weights can change in place without retracing."""
    def __init__(self, m, name):
        grad_model = keras.Model(m.inputs, [m.get_layer('conv3').output, m.get_layer('logits').output])
        logit_model = keras.Model(m.inputs, m.get_layer('logits').output)

        def count(fn):
            TRACES[f'{name}.{fn}'] = TRACES.get(f'{name}.{fn}', 0) + 1   # runs only while tracing

        @tf.function(input_signature=[SIG_X])
        def logits(x):
            count('logits')
            return logit_model(x, training=False)

        @tf.function(input_signature=[SIG_X])
        def probs(x):
            count('probs')
            return m(x, training=False)

        @tf.function(input_signature=[SIG_X, SIG_T])
        def input_grads(x, t):
            count('input_grads')
            with tf.GradientTape() as tape:
                tape.watch(x)
                obj = tf.reduce_sum(logit_model(x, training=False)[:, t])
            return tape.gradient(obj, x)

        @tf.function(input_signature=[SIG_X, SIG_T])
        def gradcam(x, t):
            count('gradcam')
            with tf.GradientTape() as tape:
                conv, lg = grad_model(x, training=False)
                obj = tf.reduce_sum(lg[:, t])
            g = tape.gradient(obj, conv)
            cam = tf.nn.relu(tf.reduce_sum(conv * tf.reduce_mean(g, 1, keepdims=True), -1))
            up = tf.squeeze(tf.image.resize(cam[..., None, None], (720, 1), method='bilinear'), [-1, -2])
            mx = tf.reduce_max(up, 1, keepdims=True)
            return tf.where(mx > 0, up / tf.maximum(mx, 1e-7), tf.zeros_like(up)), conv, g

        self.logits, self.probs, self.input_grads, self.gradcam = logits, probs, input_grads, gradcam

    def ig(self, w, t, steps):
        """All interpolation points of one window in a single model call (zero baseline, trapezoid)."""
        alphas = tf.linspace(0.0, 1.0, steps + 1)[:, None, None]
        g = self.input_grads(alphas * w[None], t).numpy().astype(np.float64)
        avg = (g.sum(0) - 0.5 * (g[0] + g[-1])) / steps
        attr = (w * avg)[:, 0]
        ends = self.logits(tf.constant(np.stack([np.zeros_like(w), w]))).numpy()[:, int(t)]
        delta = float(ends[1] - ends[0])
        return attr.astype(np.float32), float(attr.sum()), delta


K = Kernels(model, 'trained')

# ------------------------------------------------------------------ faithfulness setup
FR = np.linspace(0, 1, 21)                     # 20 perturbation steps
N_RANDOM = 3
ORDERS = ['gradcam', 'ig', 'gradcam_lerf', 'ig_lerf'] + [f'random_{i}' for i in range(N_RANDOM)]
VARIANTS = [('deletion', 'zero'), ('deletion', 'blur'), ('deletion', 'interp'), ('insertion', 'zero'), ('insertion', 'blur')]


def smooth(a):
    return uniform_filter1d(np.asarray(a, np.float64), 15, mode='nearest')


def perturbations(x, orders):
    """-> (len(VARIANTS), len(ORDERS), len(FR), 720) float32"""
    blur = gaussian_filter1d(x, 10, mode='nearest').astype(np.float32)
    zero = np.zeros_like(x)
    ranks = np.argsort(np.stack([orders[o] for o in ORDERS]), axis=1)        # rank of each sample per order
    masks = ranks[:, None, :] < np.round(FR * 720).astype(int)[None, :, None]  # top-k selected
    out = []
    t = np.arange(720)
    for mode, repl in VARIANTS:
        fill = zero if repl == 'zero' else blur
        if mode == 'insertion':
            out.append(np.where(masks, x, fill))
        elif repl == 'interp':
            v = np.zeros(masks.shape, np.float32)
            for i in range(masks.shape[0]):
                for j in range(masks.shape[1]):
                    keep = ~masks[i, j]
                    v[i, j] = np.interp(t, t[keep], x[keep]) if keep.any() else 0
            out.append(v)
        else:
            out.append(np.where(masks, fill, x))
    return np.stack(out).astype(np.float32)


def explain_window(row, cls_idx):
    w = X[row]
    t = tf.constant(cls_idx, tf.int32)
    cam, conv, grad = K.gradcam(tf.constant(w[None]), t)
    cam, conv, grad = cam.numpy()[0], conv.numpy()[0], grad.numpy()[0]
    # independent numpy re-implementation of Grad-CAM from raw activations and gradients
    raw = np.maximum((conv * grad.mean(0, keepdims=True)).sum(-1), 0)
    src = np.clip((np.arange(720) + .5) * len(raw) / 720 - .5, 0, len(raw) - 1)
    up = np.interp(src, np.arange(len(raw)), raw)
    up = up / up.max() if up.max() > 0 else np.zeros(720)
    attr, total, delta = K.ig(w, t, args.ig_steps)

    rng = np.random.default_rng([args.seed, int(row)])
    tie = rng.uniform(0, 1e-12, 720)
    sc, si = smooth(cam) + tie, smooth(attr) + tie
    # order[o][i] = deletion priority position (ascending sort key)
    def order_of(key):
        return np.argsort(key, kind='stable')
    orders = {'gradcam': order_of(-sc), 'ig': order_of(-si), 'gradcam_lerf': order_of(sc), 'ig_lerf': order_of(si)}
    for i in range(N_RANDOM):
        orders[f'random_{i}'] = order_of(-smooth(rng.normal(size=720)))
    batch = perturbations(w[:, 0], orders)
    p = K.probs(tf.constant(batch.reshape(-1, 720, 1))).numpy()[:, cls_idx].reshape(batch.shape[:3])
    curves = {}
    for v, (mode, repl) in enumerate(VARIANTS):
        c = {o: p[v, j] for j, o in enumerate(ORDERS)}
        c['random'] = np.mean([c.pop(f'random_{i}') for i in range(N_RANDOM)], axis=0)
        curves[f'{mode}_{repl}'] = {k: [round(float(a), 6) for a in val] for k, val in c.items()}
    return {'row': int(row), 'record': str(z['record'][row]), 'position': int(z['position'][row]),
            'predicted': C[cls_idx], 'true': C[true[row]],
            'gradcam_reimpl_max_abs_diff': float(np.abs(up - cam).max()), 'gradcam_all_zero': bool(cam.max() == 0),
            'ig_attr_sum': total, 'ig_score_delta': delta, 'curves': curves}


# ------------------------------------------------------------------ resume
fingerprint = {'model_sha256': clf.sha256, 'n_per_class': args.n_per_class, 'ig_steps': args.ig_steps,
               'check_steps': args.check_steps, 'n_check': args.n_check, 'n_sanity': args.n_sanity,
               'seed': args.seed, 'script_version': 3}
state = {'fingerprint': fingerprint, 'parts': {}}
if PARTIAL.exists():
    old = json.loads(PARTIAL.read_text())
    if old.get('fingerprint') == fingerprint:
        state = old
        log(f'resuming; finished parts: {list(state["parts"])}')
    else:
        log('partial file has different settings; starting fresh')

# ------------------------------------------------------------------ part 1: per-window, saved per class
done = sum(len(sample[c]) for c in C if f'class_{c}' in state['parts'])
timed, first_secs = [], None
for k, c in enumerate(C):
    if f'class_{c}' in state['parts']:
        log(f'class {c}: already done, skipping')
        continue
    rows = []
    for row in sample[c]:
        s = time.time()
        rows.append(explain_window(row, k))
        dt = time.time() - s
        done += 1
        if first_secs is None:
            first_secs = dt        # includes tf.function tracing
        else:
            timed.append(dt)
        rate = np.mean(timed) if timed else dt
        log(f'window {done}/{N_TOTAL} (class {c}) {dt:.2f}s | elapsed {time.time() - T0:.0f}s | ETA windows {rate * (N_TOTAL - done):.0f}s')
    state['parts'][f'class_{c}'] = rows
    save(PARTIAL, state)
    log(f'class {c} saved -> {PARTIAL.name}')
if first_secs is not None:
    log(f'TIMING first window (with tracing) {first_secs:.2f}s; steady-state mean {np.mean(timed) if timed else float("nan"):.2f}s over {len(timed)} windows')

# fixed class-balanced subsets, interleaved N,S,V,F,N,S,...
inter = [(sample[c][i], k) for i in range(args.n_per_class) for k, c in enumerate(C) if i < len(sample[c])]

# ------------------------------------------------------------------ part 2: completeness + app-path check
if 'completeness' not in state['parts']:
    s = time.time()
    out = {'check_windows': []}
    for i, (row, k) in enumerate(inter[:args.n_check]):
        t = tf.constant(k, tf.int32)
        rec = {'row': int(row), 'predicted': C[k]}
        for steps in (args.ig_steps, args.check_steps, 512):
            _, total, delta = K.ig(X[row], t, steps)
            rec[f'steps_{steps}'] = {'attr_sum': total, 'score_delta': delta, 'abs_error': abs(total - delta),
                                     'relative_error': abs(total - delta) / max(abs(delta), 1e-12)}
        out['check_windows'].append(rec)
        log(f'completeness {i + 1}/{args.n_check}: rel err @{args.ig_steps}={rec[f"steps_{args.ig_steps}"]["relative_error"]:.4f} '
            f'@{args.check_steps}={rec[f"steps_{args.check_steps}"]["relative_error"]:.4f} @512={rec["steps_512"]["relative_error"]:.4f}')
    # the audited code is the app's ModelExplainer (512 steps for IG): compare it with these kernels
    app = []
    for row, k in inter[:2]:
        a50 = ModelExplainer(model, ig_steps=args.ig_steps).integrated_gradients(X[row][None], k)
        a512 = ModelExplainer(model, ig_steps=512).integrated_gradients(X[row][None], k)
        mine, _, _ = K.ig(X[row], tf.constant(k, tf.int32), args.ig_steps)
        cam_app = ModelExplainer(model).gradcam(X[row][None], k)[0]
        cam_mine = K.gradcam(tf.constant(X[row][None]), tf.constant(k, tf.int32))[0].numpy()[0]
        r = {'row': int(row), 'ig_max_abs_diff_same_steps': float(np.abs(a50['attributions'][0] - mine).max()),
             'ig_max_abs_attr': float(np.abs(mine).max()),
             'gradcam_max_abs_diff': float(np.abs(cam_app - cam_mine).max()),
             'app_512_relative_completeness_error': float(abs(a512['completeness_error'][0]) / max(abs(a512['score_difference'][0]), 1e-12))}
        app.append(r)
        log(f'app-path check: {r}')
    out['app_implementation_check'] = app
    state['parts']['completeness'] = out
    save(PARTIAL, state)
    log(f'completeness part saved ({time.time() - s:.0f}s)')

# ------------------------------------------------------------------ part 3: weight randomisation sanity
if 'sanity' not in state['parts']:
    s = time.time()
    sub = inter[:args.n_sanity]

    def maps(kern):
        cams, igs = [], []
        for row, k in sub:
            t = tf.constant(k, tf.int32)
            cams.append(kern.gradcam(tf.constant(X[row][None]), t)[0].numpy()[0])
            igs.append(kern.ig(X[row], t, args.ig_steps)[0])
        return np.stack(cams), np.stack(igs)

    def rho(a, b):
        v = [float(spearmanr(p, q)[0]) for p, q in zip(a, b) if np.ptp(p) > 0 and np.ptp(q) > 0]
        return {'mean_spearman': float(np.mean(v)) if v else None,
                'mean_abs_spearman': float(np.mean(np.abs(v))) if v else None,
                'n_valid': len(v), 'n_degenerate': len(a) - len(v)}

    cam0, ig0 = maps(K)
    inp = np.abs(np.stack([X[row][:, 0] for row, _ in sub]))

    def compare(cm, im):
        return {'gradcam': rho(cam0, cm), 'ig_signed': rho(ig0, im), 'ig_absolute': rho(np.abs(ig0), np.abs(im))}

    def random_weights(seed):
        keras.utils.set_random_seed(seed)
        m = keras.models.clone_model(model)
        m(tf.zeros((1, 720, 1)))
        return m

    probe = keras.models.clone_model(model)
    probe(tf.zeros((1, 720, 1)))
    KR = Kernels(probe, 'randomised')
    full = []
    for seed in (1, 2, 3):
        probe.set_weights(random_weights(seed).get_weights())
        cm, im = maps(KR)
        r = {'seed': seed, **compare(cm, im), 'random_model_ig_abs_vs_input_abs': rho(np.abs(im), inp)}
        full.append(r)
        log(f'sanity full seed {seed}: gradcam {r["gradcam"]} | ig_abs {r["ig_absolute"]}')
    rm = random_weights(11)
    probe.set_weights(model.get_weights())
    casc, through = [], ['logits', 'dense_8', 'bilstm', 'conv3', 'batch_normalization_19', 'conv2', 'batch_normalization_18', 'conv1']
    for name in through:
        probe.get_layer(name).set_weights(rm.get_layer(name).get_weights())
        cm, im = maps(KR)
        r = {'randomised_through': name, **compare(cm, im)}
        casc.append(r)
        log(f'sanity cascade {name}: gradcam {r["gradcam"]["mean_spearman"]} | ig_signed {r["ig_signed"]["mean_spearman"]} | ig_abs {r["ig_absolute"]["mean_spearman"]}')
    state['parts']['sanity'] = {'n_windows': len(sub), 'ig_steps': args.ig_steps,
                                'full_reinitialisation': full, 'cascading_top_down': casc}
    save(PARTIAL, state)
    log(f'sanity part saved ({time.time() - s:.0f}s)')

# ------------------------------------------------------------------ aggregate
rows = [r for c in C for r in state['parts'][f'class_{c}']]
n = len(rows)
cls = np.array([r['predicted'] for r in rows])
rel = np.array([abs(r['ig_attr_sum'] - r['ig_score_delta']) / max(abs(r['ig_score_delta']), 1e-12) for r in rows])
err = np.array([abs(r['ig_attr_sum'] - r['ig_score_delta']) for r in rows])


def summary(e, r):
    return {'n': int(len(r)), 'abs_error_median': float(np.median(e)), 'abs_error_max': float(e.max()),
            'relative_error_median': float(np.median(r)), 'relative_error_p95': float(np.percentile(r, 95)),
            'relative_error_max': float(r.max()), 'fraction_above_5pct': float((r > .05).mean())}


brng = np.random.default_rng(1)


def ci(d):
    b = d[brng.integers(0, len(d), (2000, len(d)))].mean(1)
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


variants = {}
for mode, repl in VARIANTS:
    key = f'{mode}_{repl}'
    cur = {o: np.array([r['curves'][key][o] for r in rows]) for o in rows[0]['curves'][key]}
    auc = {o: np.trapezoid(v, FR, axis=1) for o, v in cur.items()}
    sign = 1 if mode == 'insertion' else -1
    out = {'better': 'higher AUC' if mode == 'insertion' else 'lower AUC',
           'curves': {o: v.mean(0).round(5).tolist() for o, v in cur.items()},
           'auc': {o: float(a.mean()) for o, a in auc.items()},
           'advantage_over_random': {}, 'advantage_by_predicted_class': {}, 'morf_vs_lerf': {}}
    for m_ in ('gradcam', 'ig'):
        adv = sign * (auc[m_] - auc['random'])
        lo, hi = ci(adv)
        out['advantage_over_random'][m_] = {'mean': float(adv.mean()), 'ci95': [lo, hi], 'beats_random': lo > 0, 'worse_than_random': hi < 0}
        out['advantage_by_predicted_class'][m_] = {c: float(adv[cls == c].mean()) for c in C}
        ml = sign * (auc[m_] - auc[m_ + '_lerf'])
        out['morf_vs_lerf'][m_] = {'advantage_mean': float(ml.mean()), 'ci95': ci(ml)}
    variants[key] = out

diffs = [r['gradcam_reimpl_max_abs_diff'] for r in rows]
art = {
    'schema_version': 2, 'evaluated_at_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
    'model_sha256': clf.sha256, 'settings': vars(args),
    'device': {'physical_devices': [d.device_type for d in tf.config.list_physical_devices()], 'gpu_used': bool(gpus)},
    'sample': {'strategy': f'{args.n_per_class} test windows per PREDICTED class', 'n': n, 'seed': args.seed,
               'true_label_agreement_by_predicted_class': {c: float(np.mean([r['true'] == c for r in rows if r['predicted'] == c])) for c in C},
               'source': 'results/test_examples_for_xai.npz'},
    'gradcam': {'target_layer': 'conv3', 'target_layer_output_shape': list(model.get_layer('conv3').output.shape),
                'gradient_of': 'pre-softmax logit of target class', 'upsampling': 'bilinear (half-pixel centres) to 720 samples, per-window max normalisation',
                'independent_reimplementation_max_abs_diff': max(diffs),
                'all_zero_maps': {c: int(sum(r['gradcam_all_zero'] for r in rows if r['predicted'] == c)) for c in C}},
    'ig_completeness': {'baseline': 'all-zero z-scored window', 'output_attributed': 'logit', 'rule': 'trapezoidal',
                        'steps_in_this_evaluation': args.ig_steps, 'steps_used_by_app': 512,
                        'all_windows': summary(err, rel),
                        'by_predicted_class': {c: summary(err[cls == c], rel[cls == c]) for c in C},
                        **state['parts']['completeness']},
    'randomisation_sanity': state['parts']['sanity'],
    'faithfulness': {'target': 'softmax probability of predicted class', 'steps': 20, 'fractions': FR.round(2).tolist(),
                     'smoothing_samples': 15, 'random_control': f'{N_RANDOM} random orders from identically smoothed noise',
                     'note': 'perturbed windows are off the training distribution and are not re-z-scored', 'variants': variants},
    'interpretation_limit': 'These tests describe model behaviour under perturbation. They do not show physiological or causal relevance.',
}
save(FINAL, art)

for key, v in variants.items():
    log(f'{key}: auc {({o: round(a, 3) for o, a in v["auc"].items()})}')
    log(f'   advantage over random { {m_: (round(a["mean"], 4), [round(q, 4) for q in a["ci95"]]) for m_, a in v["advantage_over_random"].items()} }')
log(f'completeness (all windows, {args.ig_steps} steps): {art["ig_completeness"]["all_windows"]}')
log(f'gradcam reimplementation max diff {max(diffs):.2e}')
log(f'tf.function trace counts (expect 1 each): {TRACES}')
log(f'DONE in {time.time() - T0:.0f}s -> {FINAL}')
