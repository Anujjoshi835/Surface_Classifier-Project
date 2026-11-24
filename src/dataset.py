"""Turn recordings into window-level feature matrices."""
import numpy as np
from features import extract, to_vector, FEATURE_NAMES
from segment import active_span

# window length is a recording artefact, not physics, so it is excluded
DROP = {"n_eff"}
MODEL_FEATURES = [f for f in FEATURE_NAMES if f not in DROP]
KEEP_IDX = np.array([FEATURE_NAMES.index(f) for f in MODEL_FEATURES])

WIN = 56
HOP = 8
MIN_SEG = 32


def _bounds(n, win, hop):
    if n <= win:
        return [(0, n)] if n >= MIN_SEG else []
    starts = list(range(0, n - win + 1, hop))
    if starts[-1] + win < n:
        starts.append(n - win)
    return [(s, s + win) for s in starts]


def featurise(raw, win=WIN, hop=HOP, use_active=True):
    """Return (X, spans, span_active). X has one row per window."""
    raw = np.asarray(raw, dtype=float)
    if use_active:
        lo, hi = active_span(raw)
    else:
        lo, hi = 0, raw.size
    seg = raw[lo:hi]
    rows, spans = [], []
    for a, b in _bounds(seg.size, win, hop):
        w = seg[a:b]
        fin = np.isfinite(w).sum()
        if fin < 16:
            f = {k: np.nan for k in FEATURE_NAMES}
            f["ovf_frac"] = float((~np.isfinite(w)).mean())
            f["ovf_run_max"] = 1.0
            v = to_vector(f)
        else:
            v = to_vector(extract(w))
        rows.append(v[KEEP_IDX])
        spans.append((lo + a, lo + b))
    if not rows:
        return np.zeros((0, len(MODEL_FEATURES))), [], (lo, hi)
    return np.vstack(rows), spans, (lo, hi)


def build(recs, win=WIN, hop=HOP):
    X, y, g, names = [], [], [], []
    for r in recs:
        W, _, _ = featurise(r["x"], win, hop)
        if W.shape[0] == 0:
            continue
        X.append(W)
        y += [1 if r["label"] == "rough" else 0] * W.shape[0]
        g += [sample_id(r["name"])] * W.shape[0]
        names += [r["name"]] * W.shape[0]
    return np.vstack(X), np.array(y), np.array(g), np.array(names)


def sample_id(name):
    u = name.upper()
    for suf in ("_ROUGH", "_SMOOTH", "ROUGH", "SMOOTH"):
        if u.endswith(suf):
            return u[: -len(suf)]
    return u
