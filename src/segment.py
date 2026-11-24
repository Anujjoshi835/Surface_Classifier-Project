"""Locate the loaded (active) part of a resistance trace.

Every recording in this study has the same shape: a quiet rest baseline, a
jump when the probe is loaded against the surface, an oscillating scan, then
a decay back towards a settled plateau. The features that carry surface
information live in the scan, so the scan has to be located first.
"""
import numpy as np

BASE_N = 12          # readings used for the rest baseline
RISE_FRAC = 0.30     # fraction of the full excursion that counts as "loaded"


def log_signal(raw):
    """log10 of the readings with overflow interpolated. Returns (y, bad)."""
    raw = np.asarray(raw, dtype=float)
    bad = ~np.isfinite(raw) | (raw <= 0)
    y = np.full(raw.size, np.nan)
    y[~bad] = np.log10(raw[~bad])
    ok = np.isfinite(y)
    if ok.sum() == 0:
        return None, bad
    if ok.sum() < y.size:
        idx = np.arange(y.size)
        y[~ok] = np.interp(idx[~ok], idx[ok], y[ok])
    return y, bad


def moving_std(y, w):
    n = y.size
    w = max(3, min(int(w), n))
    c1 = np.concatenate([[0.0], np.cumsum(y)])
    c2 = np.concatenate([[0.0], np.cumsum(y * y)])
    s1, s2 = c1[w:] - c1[:-w], c2[w:] - c2[:-w]
    s = np.sqrt(np.maximum(s2 / w - (s1 / w) ** 2, 0.0))
    pad = n - s.size
    left = pad // 2
    return np.concatenate([np.full(left, s[0]), s, np.full(pad - left, s[-1])])


def baseline(y, nb=BASE_N):
    return float(np.median(y[:max(4, min(nb, y.size // 4))]))


def active_span(raw, mode="level", min_len=40):
    """Index bounds (lo, hi) of the loaded part of the recording."""
    raw = np.asarray(raw, dtype=float)
    n = raw.size
    y, bad = log_signal(raw)
    if y is None or n < min_len:
        return 0, n

    if mode == "full":
        return 0, n

    b = baseline(y)
    top = float(np.percentile(y, 97))
    amp = top - b

    if mode == "onset":
        # from the loading jump to the end of the recording
        if amp <= 0.1:
            return 0, n
        hot = (y > b + RISE_FRAC * amp) | bad
        idx = np.flatnonzero(hot)
        if idx.size == 0:
            return 0, n
        lo = max(0, int(idx[0]) - 2)
        return lo, n

    # mode == "level": loading jump to the last loaded reading
    if amp > 0.1:
        hot = (y > b + RISE_FRAC * amp) | bad
        idx = np.flatnonzero(hot)
        if idx.size >= 8:
            lo, hi = max(0, int(idx[0]) - 2), min(n, int(idx[-1]) + 3)
            if hi - lo >= min_len:
                return lo, hi

    s = moving_std(y, 7)
    k = min(max(min_len, n // 2), n)
    c = np.concatenate([[0.0], np.cumsum(s)])
    tot = c[k:] - c[:-k]
    lo = int(np.argmax(tot)) if tot.size else 0
    return lo, min(n, lo + k)
