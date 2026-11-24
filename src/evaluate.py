"""Leave-one-specimen-out evaluation of the fixed pipeline.

No configuration search happens here, so the accuracy is not inflated by
picking the best of many options after seeing the answers.
"""
import sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, "src")

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier

from loader import load_all
from pipeline import build, MODEL_FEATURES, sample_id


def make(kind):
    pre = [("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]
    if kind == "lr":
        return Pipeline(pre + [("c", LogisticRegression(C=0.15, max_iter=6000))])
    if kind == "lr_strong":
        return Pipeline(pre + [("c", LogisticRegression(C=0.05, max_iter=6000))])
    if kind == "gb":
        return Pipeline(pre + [("c", GradientBoostingClassifier(
            n_estimators=120, learning_rate=0.05, max_depth=2,
            subsample=0.85, random_state=0))])
    if kind == "et":
        return Pipeline(pre + [("c", ExtraTreesClassifier(
            n_estimators=500, min_samples_leaf=8, max_features=0.5,
            random_state=0, n_jobs=-1))])
    raise ValueError(kind)


def logo(recs, kind, n_crops, tag):
    Xtr, ytr, gtr, ntr = build(recs, train=True, n_crops=n_crops)
    Xte, yte, gte, nte = build(recs, train=False)
    names = [r["name"] for r in recs]
    truth = {r["name"]: (1 if r["label"] == "rough" else 0) for r in recs}
    prob = {}
    for gi in sorted(set(gtr)):
        tr = gtr != gi
        if len(set(ytr[tr])) < 2:
            continue
        m = clone(make(kind)).fit(Xtr[tr], ytr[tr])
        sel = gte == gi
        p = m.predict_proba(Xte[sel])[:, 1]
        for n in set(nte[sel]):
            prob[n] = float(np.mean(p[nte[sel] == n]))
    ok = [n for n in names if n in prob]
    acc = np.mean([(prob[n] > .5) == bool(truth[n]) for n in ok])
    wrong = [n for n in ok if (prob[n] > .5) != bool(truth[n])]
    # mean margin: how decisive the correct calls are
    marg = np.mean([abs(prob[n] - .5) for n in ok]) * 2
    print(f"{tag:26s} {kind:9s} crops={n_crops:3d}  acc {acc*100:5.1f}% "
          f"({len(ok)-len(wrong)}/{len(ok)})  conf {marg:.2f}  miss: "
          f"{', '.join(wrong) or 'none'}")
    return acc, prob, truth


if __name__ == "__main__":
    allrecs = [r for r in load_all() if r["label"]]
    S = [r for r in allrecs if r["name"].upper().startswith("S")]
    T = [r for r in allrecs if not r["name"].upper().startswith("S")]
    print(f"features: {len(MODEL_FEATURES)}\n")
    for kind in ("lr_strong", "lr", "gb", "et"):
        for nc in (1, 12, 24, 40):
            logo(S, kind, nc, "S-series (15 clean)")
        print()
    print()
    for kind in ("lr_strong", "lr", "gb"):
        for nc in (24,):
            logo(allrecs, kind, nc, "all 21 labelled")
