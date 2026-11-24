"""Synthetic recording generator, calibrated to the real measurements.

The real set is only 21 recordings, several of them partly spoiled by
contact loss, so it is too small to train on comfortably. This module builds
extra traces with the same structure the instrument actually produces: a
quiet rest baseline, a loading jump, a scan whose fluctuation spectrum
depends on surface finish, then a decay to a settled plateau.

Class parameters are fitted from the real recordings rather than invented.
Synthetic data is used to pad training and to stress test; accuracy is only
ever reported on real recordings.
"""
import numpy as np
from segment import log_signal, active_span

OVERFLOW = 9.9e37


def fit_params(recs):
    """Measure the generator's parameters from real recordings, per class."""
    p = {"rough": [], "smooth": []}
    for r in recs:
        if r["label"] not in p:
            continue
        x = np.asarray(r["x"], float)
        y, bad = log_signal(x)
        if y is None:
            continue
        lo, hi = active_span(x, mode="level")
        seg = y[lo:hi]
        if seg.size < 32:
            continue
        base = float(np.median(y[: max(4, min(12, y.size // 4))]))
        # The noise colour has to be measured on the scan itself with only a
        # straight line removed. Measuring it on an already high-passed
        # residual and then generating from that slope produces noise that is
        # high-passed twice, which comes out far too blue.
        t = np.arange(seg.size, dtype=float)
        lin = np.polyval(np.polyfit(t, seg, 1), t)
        fluct = seg - lin
        psd = np.abs(np.fft.rfft(fluct * np.hanning(fluct.size))) ** 2
        psd = psd[1:]
        fr = np.arange(1, psd.size + 1, dtype=float)
        m = psd > 1e-20
        slope = float(np.polyfit(np.log(fr[m]), np.log(psd[m]), 1)[0]) if m.sum() > 8 else -1.0
        w = 9
        pad = np.pad(seg, (w // 2, w // 2), mode="edge")
        trend = np.convolve(pad, np.ones(w) / w, "valid")[: seg.size]
        p[r["label"]].append({
            "base": base,
            "exc": float(np.percentile(y, 97) - base),
            "sigma": float(fluct.std()),
            "hp_sigma": float((seg - trend).std()),
            "slope": float(np.clip(slope, -3.0, 0.5)),
            "range": float(y.max() - y.min()),
            "ovf": float(bad.mean()),
            "n": int(x.size),
            "act": float((hi - lo) / max(1, x.size)),
            "tail": float(np.median(y[-12:]) - base),
        })
    return p


def _colored_noise(n, slope, rng):
    """Noise with power spectrum f**slope."""
    k = n // 2 + 1
    f = np.arange(k, dtype=float)
    f[0] = 1.0
    amp = f ** (slope / 2.0)
    amp[0] = 0.0
    ph = rng.uniform(0, 2 * np.pi, k)
    spec = amp * (np.cos(ph) + 1j * np.sin(ph))
    y = np.fft.irfft(spec, n)
    s = y.std()
    return y / s if s > 1e-12 else y


def _jitter(v, rel, rng, lo=None, hi=None):
    out = v * (1.0 + rel * rng.normal())
    if lo is not None:
        out = max(lo, out)
    if hi is not None:
        out = min(hi, out)
    return out


def _hp_std(v, w=9):
    pad = np.pad(v, (w // 2, w // 2), mode="edge")
    trend = np.convolve(pad, np.ones(w) / w, "valid")[: v.size]
    return float((v - trend).std())


def generate(label, params, rng, n=None):
    """One synthetic recording, returned as resistance with overflow codes."""
    pool = params[label]
    q = pool[rng.integers(len(pool))]
    n = int(n or _jitter(q["n"], 0.15, rng, 140, 400))
    # the base level is shifted freely: the model must be scale free, and a
    # generator that only ever reproduces the recorded decades would never
    # test that
    base = q["base"] + rng.uniform(-1.5, 1.5)
    exc = _jitter(q["exc"], 0.15, rng, 0.25, 4.0)
    sigma = _jitter(q["hp_sigma"], 0.15, rng, 0.02, 0.8)
    slope = q["slope"] + rng.normal(0, 0.15)
    ovf_rate = max(0.0, q["ovf"] * rng.uniform(0.3, 1.4))
    tail = q["tail"] * rng.uniform(0.6, 1.2)

    n_rest = int(rng.integers(8, 26))
    n_act = int(round(n * _jitter(q["act"], 0.18, rng, 0.35, 0.95)))
    n_act = max(48, min(n_act, n - n_rest - 8))
    n_tail = max(4, n - n_rest - n_act)

    # slow envelope of the scan: rise, undulating plateau, decay
    t = np.linspace(0, 1, n_act)
    env = np.ones(n_act)
    rise = max(2, int(0.04 * n_act))
    env[:rise] = np.linspace(0.15, 1.0, rise)
    dec_start = int(rng.uniform(0.55, 0.9) * n_act)
    env[dec_start:] = np.linspace(1.0, rng.uniform(0.05, 0.35), n_act - dec_start)
    n_hump = int(rng.integers(2, 6))
    slow = np.zeros(n_act)
    for _ in range(n_hump):
        c, w = rng.uniform(0.05, 0.95), rng.uniform(0.04, 0.18)
        slow += rng.uniform(-1, 1) * np.exp(-0.5 * ((t - c) / w) ** 2)
    if np.abs(slow).max() > 1e-9:
        slow = slow / np.abs(slow).max()

    scan = base + exc * env * (0.80 + 0.30 * slow)
    # scale the noise by its own high-passed spread, because that is the
    # quantity the features actually measure; scaling by total spread would
    # under-fill the fine structure for a steep (pink) slope
    noise = _colored_noise(n_act, slope, rng)
    hs = _hp_std(noise)
    scan = scan + noise * (sigma / hs if hs > 1e-9 else sigma)

    rest = base + rng.normal(0, max(0.004, sigma * 0.06), n_rest)
    tail_lvl = base + tail
    tl = np.linspace(scan[-1], tail_lvl, n_tail)
    tl = tl + rng.normal(0, max(0.004, sigma * 0.10), n_tail)

    y = np.concatenate([rest, scan, tl])
    R = np.power(10.0, y)

    # contact loss: short bursts of open circuit during the scan
    if ovf_rate > 1e-4:
        n_bad = int(round(ovf_rate * n))
        placed = 0
        guard = 0
        while placed < n_bad and guard < 200:
            guard += 1
            L = int(rng.integers(1, max(2, int(0.08 * n))))
            s = int(rng.integers(n_rest, max(n_rest + 1, n_rest + n_act - L)))
            R[s:s + L] = OVERFLOW
            placed += L
    R = np.where(np.isfinite(R) & (R > 0), R, OVERFLOW)
    return R


def make_recording(label, params, rng):
    """(x with NaN for contact loss, raw R with overflow codes)."""
    R = generate(label, params, rng)
    x = np.where(R >= 1e30, np.nan, R)
    return x, R


def make_set(params, rng, n_each=200):
    out = []
    for lab in ("rough", "smooth"):
        for i in range(n_each):
            x, _ = make_recording(lab, params, rng)
            out.append({"name": f"SYN_{lab}_{i:03d}", "label": lab, "x": x})
    return out
