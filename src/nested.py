"""Nested leave-one-sample-out: the inner loop picks the configuration, the
outer loop scores it. Nothing about the held-out specimen touches model or
configuration choice, so the number it reports is not cherry-picked."""
import sys, warnings, itertools
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, "src")

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from loader import load_all
from sweep import trace_features, NAMES, SETS
from dataset import sample_id

MODELS = {
    "lr0.1": lambda: Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler()),
                               ("c", LogisticRegression(C=0.1, max_iter=5000))]),
    "lr0.5": lambda: Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler()),
                               ("c", LogisticRegression(C=0.5, max_iter=5000))]),
    "lrL1": lambda: Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler()),
                              ("c", LogisticRegression(C=0.4, penalty="l1", solver="liblinear", max_iter=5000))]),
    "nb": lambda: Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler()),
                            ("c", GaussianNB())]),
    "rf": lambda: Pipeline([("i", SimpleImputer(strategy="median")),
                            ("c", RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                                         max_features=0.5, random_state=0, n_jobs=-1))]),
    "et": lambda: Pipeline([("i", SimpleImputer(strategy="median")),
                            ("c", ExtraTreesClassifier(n_estimators=300, min_samples_leaf=2,
                                                       max_features=0.5, random_state=0, n_jobs=-1))]),
    "gb": lambda: Pipeline([("i", SimpleImputer(strategy="median")),
                            ("c", GradientBoostingClassifier(n_estimators=150, learning_rate=0.06,
                                                             max_depth=2, random_state=0))]),
}
CONFIGS = [(m, s, md) for m in ("level", "onset", "full") for s in SETS for md in MODELS]


def logo_acc(X, y, g, cfgmodel):
    pred = np.zeros(len(y))
    for gi in sorted(set(g)):
        te = g == gi
        if len(set(y[~te])) < 2:
            pred[te] = 0.5
            continue
        m = clone(cfgmodel()).fit(X[~te], y[~te])
        pred[te] = m.predict_proba(X[te])[:, 1]
    return float(np.mean((pred > .5) == y.astype(bool))), pred


def main(scope="S15"):
    recs = [r for r in load_all() if r["label"]]
    nm = np.array([r["name"] for r in recs])
    y = np.array([1 if r["label"] == "rough" else 0 for r in recs])
    g = np.array([sample_id(r["name"]) for r in recs])
    feats = {}
    for mode in ("level", "onset", "full"):
        feats[mode] = np.array([[trace_features(r["x"], mode).get(k, np.nan) for k in NAMES]
                                for r in recs], float)
    mask = np.array([n.upper().startswith("S") for n in nm]) if scope == "S15" else np.ones(len(y), bool)
    y, g, nm = y[mask], g[mask], nm[mask]
    feats = {k: v[mask] for k, v in feats.items()}

    groups = sorted(set(g))
    outer_pred, chosen = np.zeros(len(y)), []
    for gi in groups:
        te = g == gi
        best, bestcfg = -1, None
        for mode, sname, mdl in CONFIGS:
            idx = [NAMES.index(f) for f in SETS[sname]]
            Xi, yi, gi2 = feats[mode][~te][:, idx], y[~te], g[~te]
            a, _ = logo_acc(Xi, yi, gi2, MODELS[mdl])
            if a > best:
                best, bestcfg = a, (mode, sname, mdl)
        mode, sname, mdl = bestcfg
        idx = [NAMES.index(f) for f in SETS[sname]]
        m = clone(MODELS[mdl]()).fit(feats[mode][~te][:, idx], y[~te])
        outer_pred[te] = m.predict_proba(feats[mode][te][:, idx])[:, 1]
        chosen.append(bestcfg)
        print(f"  fold {gi:6s}: inner-best {best*100:5.1f}% {bestcfg}  -> "
              + ", ".join(f"{n}={p:.2f}" for n, p in zip(nm[te], outer_pred[te])))

    acc = np.mean((outer_pred > .5) == y.astype(bool))
    wrong = [n for n, p, t in zip(nm, outer_pred, y) if (p > .5) != bool(t)]
    print(f"\n{scope} NESTED leave-one-sample-out accuracy: {acc*100:.1f}% "
          f"({len(y)-len(wrong)}/{len(y)})   misses: {', '.join(wrong) or 'none'}")
    from collections import Counter
    print("configs chosen by the inner loop:", Counter(chosen).most_common())
    return acc


if __name__ == "__main__":
    for sc in ("S15", "all21"):
        print(f"\n===== {sc} =====")
        main(sc)
