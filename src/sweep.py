"""Honest sweep: how well can rough vs smooth actually be told apart?

Validation is leave-one-sample-out, where a sample is the physical specimen
(S1..S8, T5, T8, T10). A specimen's rough and smooth recordings are always
held out together, so the model never sees the test specimen in any form.
"""
import sys, warnings, itertools
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, "src")

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB

from loader import load_all
from features import extract, FEATURE_NAMES
from segment import active_span, log_signal
from dataset import sample_id


def trace_features(raw, mode):
    lo, hi = active_span(raw, mode=mode)
    seg = raw[lo:hi]
    f = extract(seg)
    f["span_frac"] = (hi - lo) / max(1, raw.size)
    y, bad = log_signal(raw)
    if y is not None:
        b = float(np.median(y[:12]))
        f["exc_height"] = float(np.percentile(y, 97) - b)
        f["end_offset"] = float(np.median(y[-12:]) - b)
    else:
        f["exc_height"] = f["end_offset"] = np.nan
    return f


NAMES = FEATURE_NAMES + ["span_frac", "exc_height", "end_offset"]
DROP = {"n_eff"}

SETS = {
    "all": [n for n in NAMES if n not in DROP],
    "phys": ["rough_rms_s9", "jitter_med", "jitter_p90", "hf_ratio", "wavelet_ratio",
             "spike_rate", "turn_rate", "ac1_diff", "range_dec", "ovf_frac",
             "span_frac", "exc_height", "settle_ratio"],
    "lean": ["rough_rms_s9", "jitter_p90", "wavelet_ratio", "spike_rate",
             "ovf_frac", "span_frac", "range_dec"],
    "shape": ["rough_rms_s3", "rough_rms_s9", "wavelet_ratio", "hf_ratio", "mf_ratio",
              "band_frac_hi", "spec_slope", "spec_entropy", "perm_entropy",
              "turn_rate", "zcr", "ac1_diff", "jitter_rel"],
}

MODELS = {
    "lr0.1": Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler()),
                       ("c", LogisticRegression(C=0.1, max_iter=5000))]),
    "lr0.5": Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler()),
                       ("c", LogisticRegression(C=0.5, max_iter=5000))]),
    "lrL1": Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler()),
                      ("c", LogisticRegression(C=0.4, penalty="l1", solver="liblinear", max_iter=5000))]),
    "nb": Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler()),
                    ("c", GaussianNB())]),
    "rf": Pipeline([("i", SimpleImputer(strategy="median")),
                    ("c", RandomForestClassifier(n_estimators=800, min_samples_leaf=2,
                                                 max_features=0.5, random_state=0))]),
    "et": Pipeline([("i", SimpleImputer(strategy="median")),
                    ("c", ExtraTreesClassifier(n_estimators=800, min_samples_leaf=2,
                                               max_features=0.5, random_state=0))]),
    "gb": Pipeline([("i", SimpleImputer(strategy="median")),
                    ("c", GradientBoostingClassifier(n_estimators=150, learning_rate=0.06,
                                                     max_depth=2, random_state=0))]),
}


def main():
    recs = [r for r in load_all() if r["label"]]
    cache = {}
    for mode in ("level", "onset", "full"):
        rows = []
        for r in recs:
            f = trace_features(r["x"], mode)
            rows.append([f.get(k, np.nan) for k in NAMES])
        cache[mode] = np.array(rows, float)
    y = np.array([1 if r["label"] == "rough" else 0 for r in recs])
    g = np.array([sample_id(r["name"]) for r in recs])
    nm = np.array([r["name"] for r in recs])
    np.save("out/trace_y.npy", y); np.save("out/trace_g.npy", g); np.save("out/trace_nm.npy", nm)
    for k, v in cache.items():
        np.save(f"out/trace_X_{k}.npy", v)

    S = np.array([n.upper().startswith("S") for n in nm])
    results = []
    for mode, setname, mdl in itertools.product(cache, SETS, MODELS):
        idx = [NAMES.index(f) for f in SETS[setname]]
        X = cache[mode][:, idx]
        for scope, mask in (("all21", np.ones(len(y), bool)), ("S15", S)):
            Xs, ys, gs, ns = X[mask], y[mask], g[mask], nm[mask]
            pred = np.zeros(len(ys))
            for gi in sorted(set(gs)):
                te = gs == gi
                if len(set(ys[~te])) < 2:
                    continue
                m = clone(MODELS[mdl]).fit(Xs[~te], ys[~te])
                pred[te] = m.predict_proba(Xs[te])[:, 1]
            acc = np.mean((pred > .5) == ys.astype(bool))
            results.append((acc, scope, mode, setname, mdl, pred.copy(), ns, ys))

    for scope in ("all21", "S15"):
        print(f"\n===== {scope} : top 12 by leave-one-sample-out accuracy =====")
        sub = sorted([r for r in results if r[1] == scope], key=lambda t: -t[0])[:12]
        for acc, _, mode, setname, mdl, pred, ns, ys in sub:
            wrong = [n for n, p, t in zip(ns, pred, ys) if (p > .5) != bool(t)]
            print(f"  {acc*100:5.1f}%  seg={mode:6s} feats={setname:6s} model={mdl:6s} "
                  f"miss={len(wrong)} {','.join(w[:9] for w in wrong)}")


if __name__ == "__main__":
    main()
