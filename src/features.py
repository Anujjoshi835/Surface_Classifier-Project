"""Scale-free features of a resistance vs reading-number trace.

Everything is computed on log10(R) so that recordings with wildly different
base resistance (1e3 to 1e9 ohm) are comparable. Overflow codes are treated
as missing and interpolated for the shape features, while their rate is kept
as a feature of its own.
"""
import math
from functools import cmp_to_key
import numpy as np

FEATURE_NAMES = [
    "ovf_frac", "ovf_run_max",
    "range_dec", "iqr_log", "std_log", "mad_log",
    "jitter_med", "jitter_mad", "jitter_p90", "jitter_rel",
    "spike_rate", "spike_rate_big", "diff_kurt", "diff_skew_abs",
    "turn_rate", "zcr", "ac1_diff", "ac1_sig", "ac5_sig",
    "hf_ratio", "mf_ratio", "spec_slope", "spec_entropy",
    "perm_entropy", "hurst",
    "rough_rms_s3", "rough_rms_s9", "wavelet_ratio",
    "band_frac_hi", "settle_ratio", "excursion_frac", "n_eff",
]


def _interp_nan(x):
    x = np.asarray(x, dtype=float).copy()
    ok = np.isfinite(x)
    if ok.sum() == 0:
        return None
    if ok.sum() < x.size:
        idx = np.arange(x.size)
        x[~ok] = np.interp(idx[~ok], idx[ok], x[ok])
    return x


def _max_run(mask):
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def _moving_std(y, w):
    n = y.size
    if n < w + 1:
        return np.array([np.std(y)])
    c1 = np.concatenate([[0.0], np.cumsum(y)])
    c2 = np.concatenate([[0.0], np.cumsum(y * y)])
    s1 = c1[w:] - c1[:-w]
    s2 = c2[w:] - c2[:-w]
    var = np.maximum(s2 / w - (s1 / w) ** 2, 0.0)
    return np.sqrt(var)


def _perm_entropy(y, m=4):
    """Ordinal (permutation) entropy.

    Values that differ only by rounding noise are treated as ties and broken
    by position. Long runs of interpolated readings are exactly flat in
    principle, and without the tolerance their ordering flips on differences
    of order 1e-17, which makes the result depend on summation order.
    """
    n = y.size - m + 1
    if n < 10:
        return np.nan
    scale = float(np.max(np.abs(y))) if y.size else 0.0
    eps = 1e-9 * (scale if scale > 0 else 1.0)
    counts = {}
    for i in range(n):
        w = y[i:i + m]

        def cmp(a, b):
            d = w[a] - w[b]
            if abs(d) <= eps:
                return a - b
            return -1 if d < 0 else 1

        pat = tuple(sorted(range(m), key=cmp_to_key(cmp)))
        counts[pat] = counts.get(pat, 0) + 1
    p = np.array(list(counts.values()), dtype=float) / n
    return float(-(p * np.log(p)).sum() / np.log(math.factorial(m)))


def _hurst(y):
    y = y - y.mean()
    n = y.size
    if n < 32:
        return np.nan
    scales, vals = [], []
    s = 4
    while s <= n // 4:
        k = n // s
        seg = y[:k * s].reshape(k, s)
        r = (np.maximum.accumulate(seg - seg.mean(1, keepdims=True), axis=1).max(1)
             - np.minimum.accumulate(seg - seg.mean(1, keepdims=True), axis=1).min(1))
        sd = seg.std(1)
        ok = sd > 1e-12
        if ok.sum() >= 2:
            scales.append(s)
            vals.append(np.mean(r[ok] / sd[ok]))
        s *= 2
    if len(scales) < 3:
        return np.nan
    a = np.polyfit(np.log(scales), np.log(vals), 1)[0]
    return float(a)


def extract(raw, return_extras=False):
    """raw: 1-D array of resistance readings, NaN where overflow/invalid."""
    raw = np.asarray(raw, dtype=float)
    n_raw = raw.size
    bad = ~np.isfinite(raw) | (raw <= 0)
    f = {name: np.nan for name in FEATURE_NAMES}
    f["ovf_frac"] = float(bad.mean()) if n_raw else np.nan
    f["ovf_run_max"] = float(_max_run(bad) / n_raw) if n_raw else np.nan

    y = np.where(bad, np.nan, np.log10(np.where(bad, 1.0, raw)))
    y = _interp_nan(y)
    if y is None or y.size < 24:
        return (f, {}) if return_extras else f
    n = y.size
    f["n_eff"] = float(n)

    q75, q25 = np.percentile(y, [75, 25])
    f["range_dec"] = float(y.max() - y.min())
    f["iqr_log"] = float(q75 - q25)
    f["std_log"] = float(y.std())
    f["mad_log"] = float(np.median(np.abs(y - np.median(y))))

    d = np.diff(y)
    ad = np.abs(d)
    f["jitter_med"] = float(np.median(ad))
    f["jitter_mad"] = float(np.median(np.abs(ad - np.median(ad))))
    f["jitter_p90"] = float(np.percentile(ad, 90))
    scale = f["iqr_log"] if f["iqr_log"] > 1e-9 else 1e-9
    f["jitter_rel"] = float(f["jitter_med"] / scale)

    mad_d = np.median(np.abs(d - np.median(d))) * 1.4826
    mad_d = mad_d if mad_d > 1e-12 else 1e-12
    f["spike_rate"] = float((ad > 3 * mad_d).mean())
    f["spike_rate_big"] = float((ad > 8 * mad_d).mean())
    dz = (d - d.mean()) / (d.std() + 1e-12)
    f["diff_kurt"] = float(np.mean(dz ** 4) - 3.0)
    f["diff_skew_abs"] = float(abs(np.mean(dz ** 3)))

    turns = int(np.sum(np.sign(d[1:]) * np.sign(d[:-1]) < 0))
    f["turn_rate"] = float(turns / max(1, d.size - 1))

    # detrended signal: remove slow drift with a long moving average
    w = max(5, min(31, n // 6))
    if w % 2 == 0:
        w += 1
    pad = np.pad(y, (w // 2, w // 2), mode="edge")
    trend = np.convolve(pad, np.ones(w) / w, mode="valid")[:n]
    r = y - trend
    f["zcr"] = float(np.mean(np.sign(r[1:]) * np.sign(r[:-1]) < 0))

    def ac(v, lag):
        v = v - v.mean()
        den = np.sum(v * v)
        return float(np.sum(v[lag:] * v[:-lag]) / den) if den > 1e-15 else np.nan

    f["ac1_diff"] = ac(d, 1)
    f["ac1_sig"] = ac(y, 1)
    f["ac5_sig"] = ac(y, 5) if n > 10 else np.nan

    # spectrum of the detrended log signal
    seg = r - r.mean()
    win = np.hanning(seg.size)
    P = np.abs(np.fft.rfft(seg * win)) ** 2
    P = P[1:]
    tot = P.sum()
    if tot > 1e-20 and P.size > 8:
        p = P / tot
        k = P.size
        f["hf_ratio"] = float(p[int(0.5 * k):].sum())
        f["mf_ratio"] = float(p[int(0.25 * k):int(0.5 * k)].sum())
        f["band_frac_hi"] = float(p[int(0.75 * k):].sum())
        pe = p[p > 1e-15]
        f["spec_entropy"] = float(-(pe * np.log(pe)).sum() / np.log(k))
        fr = np.arange(1, k + 1, dtype=float)
        m = P > 1e-20
        if m.sum() > 8:
            f["spec_slope"] = float(np.polyfit(np.log(fr[m]), np.log(P[m]), 1)[0])

    f["perm_entropy"] = _perm_entropy(r, 4)
    f["hurst"] = _hurst(y)

    s3 = _moving_std(r, 3)
    s9 = _moving_std(r, 9)
    f["rough_rms_s3"] = float(np.median(s3))
    f["rough_rms_s9"] = float(np.median(s9))
    f["wavelet_ratio"] = float(np.median(s3) / (np.median(s9) + 1e-12))

    # how much of the trace sits well above its own quiet baseline
    base = np.median(y[:max(5, n // 20)])
    f["excursion_frac"] = float(np.mean(y > base + 0.3))
    h1 = np.median(np.abs(np.diff(y[: n // 2])))
    h2 = np.median(np.abs(np.diff(y[n // 2:])))
    f["settle_ratio"] = float(h2 / (h1 + 1e-12))

    if return_extras:
        return f, {"y": y, "resid": r, "trend": trend, "diff": d, "bad": bad}
    return f


def to_vector(f):
    return np.array([f[k] for k in FEATURE_NAMES], dtype=float)
