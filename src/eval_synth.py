"""Does synthetic padding actually help on real recordings?

The generator is refitted inside every fold using only the training
specimens, so the held-out specimen never influences the synthetic data
either directly or through the fitted parameters.
"""
import sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, "src")

from sklearn.base import clone
from loader import load_all
from pipeline import build, recording_matrix, sample_id
from evaluate import make
from synth import fit_params, make_recording


def run(recs, kind="lr", n_syn=0, w_syn=0.3, seed=0, tag=""):
    names = [r["name"] for r in recs]
    truth = {r["name"]: (1 if r["label"] == "rough" else 0) for r in recs}
    gid = {r["name"]: sample_id(r["name"]) for r in recs}
    groups = sorted(set(gid.values()))
    prob = {}
    for gi in groups:
        tr_recs = [r for r in recs if gid[r["name"]] != gi]
        te_recs = [r for r in recs if gid[r["name"]] == gi]
        if len({r["label"] for r in tr_recs}) < 2:
            continue
        X, y, _, _ = build(tr_recs, train=True, n_crops=24)
        w = np.ones(len(y))
        if n_syn:
            rng = np.random.default_rng(seed + abs(hash(gi)) % 10000)
            P = fit_params(tr_recs)
            syn = []
            for lab in ("rough", "smooth"):
                for i in range(n_syn):
                    x, _ = make_recording(lab, P, rng)
                    syn.append({"name": f"SYN{i}", "label": lab, "x": x})
            Xs, ys, _, _ = build(syn, train=True, n_crops=3)
            X = np.vstack([X, Xs])
            y = np.concatenate([y, ys])
            w = np.concatenate([w, np.full(len(ys), w_syn)])
        m = clone(make(kind))
        m.fit(X, y, **{f"{m.steps[-1][0]}__sample_weight": w})
        for r in te_recs:
            M, _, _ = recording_matrix(r["x"], train=False)
            prob[r["name"]] = float(np.mean(m.predict_proba(M)[:, 1]))
    ok = [n for n in names if n in prob]
    acc = np.mean([(prob[n] > .5) == bool(truth[n]) for n in ok])
    wrong = [n for n in ok if (prob[n] > .5) != bool(truth[n])]
    print(f"{tag:14s} {kind:4s} n_syn={n_syn:4d} w={w_syn:.2f}  acc {acc*100:5.1f}% "
          f"({len(ok)-len(wrong)}/{len(ok)})  miss: {', '.join(wrong) or 'none'}")
    return acc, prob


if __name__ == "__main__":
    allr = [r for r in load_all() if r["label"]]
    usable = [r for r in allr if np.mean(~np.isfinite(r["x"])) <= 0.25]
    print(f"usable recordings: {len(usable)}\n")
    for kind in ("lr", "lr_strong"):
        run(usable, kind, 0, 0, tag="real only")
        for ns in (30, 80, 150):
            for wv in (0.15, 0.4):
                run(usable, kind, ns, wv, tag="real+synth")
        print()
