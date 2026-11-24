"""The production pipeline: segmentation, feature set and augmentation.

The configuration here is fixed in advance on physical grounds rather than
searched, because searching over this little data just picks noise. Every
feature is scale free (computed on log10 R) and independent of how long the
operator happened to record for.
"""
import numpy as np
from features import extract
from segment import active_span, log_signal

# Deliberately excluded: anything that encodes recording length or operator
# timing (n_eff, span_frac, settle_ratio), and features that need more points
# than a crop provides (hurst).
MODEL_FEATURES = [
    "jitter_p90",      # 90th pct of |step| in log R  -- jump size
    "jitter_med",      # median |step|
    "std_log",         # spread of log R
    "range_dec",       # decades spanned
    "spike_rate",      # fraction of outlier steps
    "spec_slope",      # spectral slope: flatter means more fine structure
    "band_frac_hi",    # power in the top quarter of the band
    "hf_ratio",        # power above half Nyquist
    "spec_entropy",    # spectral flatness
    "ac1_diff",        # lag-1 autocorrelation of the steps
    "turn_rate",       # turning point density
    "perm_entropy",    # ordinal complexity
    "wavelet_ratio",   # short-scale vs long-scale fluctuation
    "ovf_frac",        # fraction of contact-loss readings
    "ovf_run_max",     # longest contact-loss run
    "exc_height",      # height of the loading excursion, in decades
]

FEATURE_LABEL = {
    "jitter_p90": "Step jump size (p90)",
    "jitter_med": "Step jump size (median)",
    "std_log": "Spread of log resistance",
    "range_dec": "Dynamic range (decades)",
    "spike_rate": "Outlier step rate",
    "spec_slope": "Spectral slope",
    "band_frac_hi": "High-band power fraction",
    "hf_ratio": "High-frequency power fraction",
    "spec_entropy": "Spectral flatness",
    "ac1_diff": "Step autocorrelation (lag 1)",
    "turn_rate": "Turning point density",
    "perm_entropy": "Ordinal complexity",
    "wavelet_ratio": "Fine vs coarse fluctuation",
    "ovf_frac": "Contact-loss fraction",
    "ovf_run_max": "Longest contact loss",
    "exc_height": "Loading excursion (decades)",
}

MIN_CROP = 48
SEG_MODE = "onset"


def _exc_height(raw):
    y, _ = log_signal(raw)
    if y is None or y.size < 8:
        return np.nan
    b = float(np.median(y[: max(4, min(12, y.size // 4))]))
    return float(np.percentile(y, 97) - b)


def feature_dict(seg, raw_full=None):
    f = extract(seg)
    f["exc_height"] = _exc_height(raw_full if raw_full is not None else seg)
    return f


def vector(seg, raw_full=None):
    f = feature_dict(seg, raw_full)
    return np.array([f.get(k, np.nan) for k in MODEL_FEATURES], float)


def crops(n, n_crops, rng, min_frac=0.55):
    """Deterministic-ish set of contiguous sub-stretches of length n."""
    if n <= MIN_CROP:
        return [(0, n)]
    out = [(0, n)]
    for _ in range(n_crops - 1):
        L = int(round(n * rng.uniform(min_frac, 1.0)))
        L = max(MIN_CROP, min(n, L))
        s = rng.integers(0, n - L + 1) if n > L else 0
        out.append((int(s), int(s + L)))
    return out


def eval_crops(n, k=9, min_frac=0.6):
    """Fixed crop set used at prediction time, so results are reproducible."""
    if n <= MIN_CROP:
        return [(0, n)]
    out = [(0, n)]
    for frac in (0.85, 0.7, 0.6):
        L = max(MIN_CROP, int(round(n * frac)))
        if L >= n:
            continue
        for s in (0, (n - L) // 2, n - L):
            out.append((s, s + L))
    seen, uniq = set(), []
    for a, b in out:
        if (a, b) not in seen:
            seen.add((a, b))
            uniq.append((a, b))
    return uniq[:k]


def recording_matrix(raw, train=False, n_crops=24, seed=0):
    """Feature rows for one recording: one per crop of its active stretch."""
    raw = np.asarray(raw, dtype=float)
    lo, hi = active_span(raw, mode=SEG_MODE)
    seg = raw[lo:hi]
    if seg.size < 24:
        seg, lo, hi = raw, 0, raw.size
    spans = (crops(seg.size, n_crops, np.random.default_rng(seed))
             if train else eval_crops(seg.size))
    rows, abs_spans = [], []
    for a, b in spans:
        rows.append(vector(seg[a:b], raw))
        abs_spans.append((lo + a, lo + b))
    return np.vstack(rows), abs_spans, (lo, hi)


def build(recs, train=True, n_crops=24):
    X, y, g, nm = [], [], [], []
    for i, r in enumerate(recs):
        M, _, _ = recording_matrix(r["x"], train=train, n_crops=n_crops, seed=i)
        X.append(M)
        lab = 1 if r["label"] == "rough" else 0
        y += [lab] * M.shape[0]
        g += [sample_id(r["name"])] * M.shape[0]
        nm += [r["name"]] * M.shape[0]
    return np.vstack(X), np.array(y), np.array(g), np.array(nm)


def sample_id(name):
    u = name.upper()
    for suf in ("_ROUGH", "_SMOOTH", "ROUGH", "SMOOTH"):
        if u.endswith(suf):
            return u[: -len(suf)]
    return u
