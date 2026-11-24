"""Cut a recording into overlapping windows and featurise each one."""
import numpy as np
from features import extract, to_vector, FEATURE_NAMES

WIN = 64
HOP = 16
MIN_WIN = 48


def window_bounds(n, win=WIN, hop=HOP):
    if n < MIN_WIN:
        return [(0, n)] if n >= 24 else []
    if n < win:
        return [(0, n)]
    starts = list(range(0, n - win + 1, hop))
    if starts[-1] + win < n:
        starts.append(n - win)
    return [(s, s + win) for s in starts]


def featurise_windows(raw, win=WIN, hop=HOP):
    raw = np.asarray(raw, dtype=float)
    rows, spans = [], []
    for a, b in window_bounds(raw.size, win, hop):
        seg = raw[a:b]
        if np.isfinite(seg).sum() < 16:
            # a window that is almost entirely overflow still carries
            # information, so keep it with the shape features blanked out
            f = {k: np.nan for k in FEATURE_NAMES}
            f["ovf_frac"] = float((~np.isfinite(seg)).mean())
            f["ovf_run_max"] = 1.0
            f["n_eff"] = float(np.isfinite(seg).sum())
            rows.append(to_vector(f))
            spans.append((a, b))
            continue
        rows.append(to_vector(extract(seg)))
        spans.append((a, b))
    if not rows:
        return np.zeros((0, len(FEATURE_NAMES))), []
    return np.vstack(rows), spans
