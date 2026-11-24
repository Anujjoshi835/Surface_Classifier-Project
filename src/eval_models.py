"""Leave-one-sample-out comparison of candidate models at recording level."""
import sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, "src")

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC

from loader import load_all
from dataset import build, MODEL_FEATURES


def models():
    def pipe(clf, qt=False):
        steps = [("imp", SimpleImputer(strategy="median"))]
        steps.append(("qt", QuantileTransformer(n_quantiles=80, output_distribution="normal"))
                     if qt else ("sc", StandardScaler()))
        steps.append(("clf", clf))
        return Pipeline(steps)

    return {
        "logreg_L2": pipe(LogisticRegression(C=0.3, max_iter=4000)),
        "logreg_L1": pipe(LogisticRegression(C=0.5, penalty="l1", solver="liblinear", max_iter=4000)),
        "logreg_qt": pipe(LogisticRegression(C=0.5, max_iter=4000), qt=True),
        "svm_rbf": pipe(SVC(C=2.0, gamma="scale", probability=True), qt=True),
        "rf": pipe(RandomForestClassifier(n_estimators=500, min_samples_leaf=4,
                                          max_features="sqrt", random_state=0)),
        "extra": pipe(ExtraTreesClassifier(n_estimators=600, min_samples_leaf=3,
                                           max_features="sqrt", random_state=0)),
        "gbdt": pipe(GradientBoostingClassifier(n_estimators=250, learning_rate=0.05,
                                                max_depth=3, subsample=0.8, random_state=0)),
        "hgb": pipe(HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                                   max_leaf_nodes=8, l2_regularization=1.0,
                                                   random_state=0)),
    }


def run(X, y, g, nm, subset=None, tag=""):
    if subset is not None:
        X, y, g, nm = X[subset], y[subset], g[subset], nm[subset]
    groups = sorted(set(g))
    recs = sorted(set(nm))
    truth = {n: int(y[nm == n][0]) for n in recs}
    out = {}
    for mname, mk in models().items():
        prob = {}
        for gi in groups:
            te = g == gi
            tr = ~te
            if len(set(y[tr])) < 2:
                continue
            m = mk
            from sklearn.base import clone
            m = clone(m)
            m.fit(X[tr], y[tr])
            p = m.predict_proba(X[te])[:, 1]
            for n in set(nm[te]):
                prob[n] = float(np.mean(p[nm[te] == n]))
        ok = [n for n in prob]
        acc = np.mean([(prob[n] > .5) == bool(truth[n]) for n in ok])
        wrong = [n for n in ok if (prob[n] > .5) != bool(truth[n])]
        out[mname] = (acc, prob, wrong)
        print(f"{tag}{mname:12s} rec-acc {acc*100:5.1f}%  ({len(ok)-len(wrong)}/{len(ok)})"
              f"   misses: {', '.join(wrong) if wrong else 'none'}")
    return out


if __name__ == "__main__":
    recs = [r for r in load_all() if r["label"]]
    X, y, g, nm = build(recs)
    keep = [i for i, f in enumerate(MODEL_FEATURES) if f != "hurst"]
    X = X[:, keep]
    print(f"features: {len(keep)}   windows: {X.shape[0]}\n")
    print("=== ALL 21 recordings, leave-one-sample-out ===")
    run(X, y, g, nm)
    S = np.array([n.upper().startswith("S") for n in nm])
    print("\n=== S-series only (15 clean recordings) ===")
    run(X, y, g, nm, subset=S)
