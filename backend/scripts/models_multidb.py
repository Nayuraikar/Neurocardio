"""Model definitions for the multi-database study.

rr_fusion versus rr_context (train.py), checked layer by layer:
  rr_context  BiLSTM -> Dropout -> concat(5 raw standardised RR)             -> Dense 64   143,300 parameters
  rr_fusion   BiLSTM -> concat(Dense16 ReLU of RR) -> Dropout                -> Dense 64   144,100 parameters
rr_fusion is the layout of the original notebook's results/bilstm_rr_sqrtw.keras. The two give the classifier the same
information and differ only by one small layer and the dropout position, so they are near-duplicates: rr_context is
NOT trained in this study.
"""
from __future__ import annotations

import numpy as np

CLASSES = ["N", "S", "V", "F"]
NEURAL = ["cnn_bilstm", "rr_fusion", "rr_fusion_attn", "rr_fusion_attn_shead", "cnn", "lstm", "resnet1d"]
ALL_MODELS = NEURAL + ["gbm"]
USES_RR = {"rr_fusion", "rr_fusion_attn", "rr_fusion_attn_shead", "gbm"}
S_HEAD = {"rr_fusion_attn_shead"}
N_RR = 5


def _backbone(layers, inp):
    """The deployed CNN front end, unchanged."""
    h = layers.Conv1D(32, 7, padding="same", activation="relu", name="conv1")(inp)
    h = layers.MaxPooling1D(2)(layers.BatchNormalization()(h))
    h = layers.Conv1D(64, 5, padding="same", activation="relu", name="conv2")(h)
    h = layers.MaxPooling1D(2)(layers.BatchNormalization()(h))
    return layers.Conv1D(128, 3, padding="same", activation="relu", name="conv3")(h)


def _head(layers, h, name="probs"):
    h = layers.Dense(64, activation="relu")(h)
    return layers.Softmax(name=name)(layers.Dense(4, name="logits")(h))


def build(name: str):
    """-> keras.Model. Output is (batch, 4) probabilities; with an S head it is (batch, 5): 4 probabilities + P(S)."""
    import keras
    from keras import layers
    inp = keras.Input((720, 1), name="ecg_input")
    rr_in = keras.Input((N_RR,), name="rr_input") if name in USES_RR else None

    if name == "cnn_bilstm":                                   # the deployed architecture, no RR: ablation control
        h = layers.Bidirectional(layers.LSTM(64), name="bilstm")(_backbone(layers, inp))
        out = _head(layers, layers.Dropout(0.4)(h))
    elif name in ("rr_fusion", "rr_fusion_attn", "rr_fusion_attn_shead"):
        seq = _backbone(layers, inp)
        if name == "rr_fusion":
            h = layers.Bidirectional(layers.LSTM(64), name="bilstm")(seq)
        else:                                                  # additive attention over the 180 BiLSTM steps
            seq = layers.Bidirectional(layers.LSTM(64, return_sequences=True), name="bilstm")(seq)
            score = layers.Dense(1, use_bias=False, name="attn_score")(layers.Dense(64, activation="tanh", name="attn_hidden")(seq))
            weights = layers.Softmax(axis=1, name="attn_weights")(score)
            h = layers.Flatten()(layers.Dot(axes=1, name="attn_pool")([weights, seq]))
        rr = layers.Dense(16, activation="relu", name="rr_dense")(rr_in)
        fused = layers.Dropout(0.4)(layers.Concatenate(name="fuse")([h, rr]))
        out = _head(layers, fused, name="probs")
        if name in S_HEAD:                                     # auxiliary S-versus-rest head on the shared features
            s = layers.Dense(32, activation="relu", name="s_hidden")(fused)
            s = layers.Dense(1, activation="sigmoid", name="s_prob")(s)
            out = layers.Concatenate(name="probs_and_s")([out, s])
    elif name == "cnn":
        h = layers.MaxPooling1D(20)(_backbone(layers, inp))    # keeps coarse position: the target beat is the central one
        out = _head(layers, layers.Dropout(0.4)(layers.Flatten()(h)))
    elif name == "lstm":
        h = layers.Reshape((90, 8))(inp)                       # 90 steps of 8 samples; no convolution
        h = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(h)
        h = layers.Bidirectional(layers.LSTM(64))(h)
        out = _head(layers, layers.Dropout(0.4)(h))
    elif name == "resnet1d":
        h = layers.Conv1D(32, 7, strides=2, padding="same", use_bias=False)(inp)
        h = layers.ReLU()(layers.BatchNormalization()(h))
        for filters in (32, 64, 128, 128):
            short = layers.BatchNormalization()(layers.Conv1D(filters, 1, strides=2, padding="same", use_bias=False)(h))
            r = layers.Conv1D(filters, 5, strides=2, padding="same", use_bias=False)(h)
            r = layers.ReLU()(layers.BatchNormalization()(r))
            r = layers.BatchNormalization()(layers.Conv1D(filters, 5, padding="same", use_bias=False)(r))
            h = layers.ReLU()(layers.Add()([short, r]))
        h = layers.Conv1D(32, 1, activation="relu")(h)         # then flatten, again to keep position
        out = _head(layers, layers.Dropout(0.4)(layers.Flatten()(h)))
    else:
        raise ValueError(name)
    return keras.Model([inp, rr_in] if rr_in is not None else inp, out, name=name)


def s_head_loss(class_w: np.ndarray, s_pos_weight: float, lam: float):
    """sqrt-weighted 4-class cross-entropy + lam * weighted binary cross-entropy for S versus rest."""
    import keras
    ops = keras.ops
    cw = ops.convert_to_tensor(class_w.astype(np.float32))

    def loss(y_true, out):
        y = ops.cast(ops.reshape(y_true, (-1,)), "int32")
        p = ops.clip(out[:, :4], 1e-7, 1.0)
        ce = -ops.take(cw, y) * ops.log(ops.sum(p * ops.one_hot(y, 4), axis=1))
        s = ops.clip(out[:, 4], 1e-7, 1 - 1e-7)
        t = ops.cast(ops.equal(y, 1), "float32")
        return ce + lam * -(s_pos_weight * t * ops.log(s) + (1 - t) * ops.log(1 - s))
    return loss


# ------------------------------------------------------------------ handcrafted features for the boosted-tree baseline
def handcrafted_features(X: np.ndarray, rr: np.ndarray, fs: int = 360, chunk: int = 20000) -> np.ndarray:
    """X: (n,720) z-scored windows centred on the beat; rr: (n,5) raw RR features. Deterministic, no fitting."""
    import pywt
    ms = lambda t: int(round(t * fs / 1000))
    c = 360
    out = []
    for i in range(0, len(X), chunk):
        w = X[i:i + chunk].astype(np.float64)
        feats = [rr[i:i + chunk].astype(np.float64), (rr[i:i + chunk, 0] / np.maximum(rr[i:i + chunk, 1], 1e-3))[:, None]]
        regions = {"p": w[:, c - ms(250):c - ms(80)], "qrs": w[:, c - ms(80):c + ms(80)], "t": w[:, c + ms(80):c + ms(400)], "all": w}
        for seg in regions.values():
            mu, sd = seg.mean(1), seg.std(1) + 1e-8
            z = (seg - mu[:, None]) / sd[:, None]
            feats.append(np.stack([mu, sd, seg.min(1), seg.max(1), np.ptp(seg, axis=1), np.abs(np.diff(seg, axis=1)).max(1),
                                   (seg ** 2).mean(1), (z ** 3).mean(1), (z ** 4).mean(1)], axis=1))
        qrs = w[:, c - ms(100):c + ms(100)]
        base = np.median(w[:, c - ms(250):c + ms(400)], axis=1, keepdims=True)
        dev = np.abs(qrs - base)
        feats.append(np.stack([(dev > 0.5 * dev.max(1, keepdims=True)).sum(1), (dev > 0.2 * dev.max(1, keepdims=True)).sum(1),
                               qrs.argmax(1), qrs.argmin(1)], axis=1).astype(np.float64))
        feats.append(w[:, c - 90:c + 131:5])                   # downsampled beat shape, -250 to +360 ms
        coeffs = pywt.wavedec(w[:, c - 128:c + 128], "db4", level=4, axis=1)
        feats.append(coeffs[0])
        feats.append(np.stack([f(d) for d in coeffs[1:] for f in (lambda d: (d ** 2).sum(1), lambda d: np.abs(d).max(1))], axis=1))
        out.append(np.concatenate(feats, axis=1).astype(np.float32))
    return np.concatenate(out)
