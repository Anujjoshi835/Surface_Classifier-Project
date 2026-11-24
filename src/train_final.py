"""Train the shipped model and export everything the web page needs."""
import sys, os, json, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
os.makedirs("out", exist_ok=True)

from sklearn.base import clone
from loader import load_all
from pipeline import (build, recording_matrix, MODEL_FEATURES, FEATURE_LABEL,
                      sample_id)
from evaluate import make
from segment import active_span
from synth import fit_params, make_recording

MAX_OVF = 0.25
KIND = "lr"
N_CROPS = 24
N_SYN = 120        # synthetic recordings per class
W_SYN = 0.20       # their weight relative to a real crop
SEED = 11


def usable(recs):
    return [r for r in recs if np.mean(~np.isfinite(r["x"])) <= MAX_OVF]


def fit_one(train_recs, seed=SEED, use_synth=True):
    """Fit the pipeline on a set of recordings, optionally padded with
    synthetic traces generated from those same recordings only."""
    X, y, _, _ = build(train_recs, train=True, n_crops=N_CROPS)
    w = np.ones(len(y))
    n_real = X.shape[0]
    if use_synth and N_SYN:
        rng = np.random.default_rng(seed)
        P = fit_params(train_recs)
        syn = []
        for lab in ("rough", "smooth"):
            for i in range(N_SYN):
                x, _ = make_recording(lab, P, rng)
                syn.append({"name": f"SYN_{lab}_{i}", "label": lab, "x": x})
        Xs, ys, _, _ = build(syn, train=True, n_crops=3)
        X = np.vstack([X, Xs])
        y = np.concatenate([y, ys])
        w = np.concatenate([w, np.full(len(ys), W_SYN)])
    m = clone(make(KIND))
    m.fit(X, y, **{f"{m.steps[-1][0]}__sample_weight": w})
    return m, X, y, n_real


def predict(model, raw):
    M, spans, span = recording_matrix(raw, train=False)
    p = model.predict_proba(M)[:, 1]
    return float(np.mean(p)), p, spans, span


def main():
    allr = [r for r in load_all() if r["label"]]
    train = usable(allr)
    print(f"training on {len(train)} of {len(allr)} labelled recordings "
          f"(dropped >{MAX_OVF:.0%} contact loss: "
          f"{[r['name'] for r in allr if r not in train]})")

    model, X, y, n_real = fit_one(train)
    print(f"training rows: {n_real} real crops + {X.shape[0]-n_real} synthetic "
          f"(weight {W_SYN})")

    # ---------- leave-one-specimen-out, the number we report ----------
    gid = {r["name"]: sample_id(r["name"]) for r in train}
    results = {}
    for use_syn in (False, True):
        probs, truth = {}, {}
        for gi in sorted(set(gid.values())):
            tr = [r for r in train if gid[r["name"]] != gi]
            te = [r for r in train if gid[r["name"]] == gi]
            if len({r["label"] for r in tr}) < 2:
                continue
            # the generator is refitted inside the fold, so the held-out
            # specimen never reaches the model even through synthetic data
            m, _, _, _ = fit_one(tr, seed=SEED + len(tr), use_synth=use_syn)
            for r in te:
                probs[r["name"]] = predict(m, r["x"])[0]
                truth[r["name"]] = 1 if r["label"] == "rough" else 0
        ok = sorted(probs)
        a = float(np.mean([(probs[n] > .5) == bool(truth[n]) for n in ok]))
        results["synth" if use_syn else "real"] = (a, dict(probs), dict(truth))
        print(f"\nleave-one-specimen-out, {'real + synthetic' if use_syn else 'real only'}: "
              f"{a*100:.1f}% ({sum((probs[n]>.5)==bool(truth[n]) for n in ok)}/{len(ok)})")
        for n in ok:
            mark = "ok " if (probs[n] > .5) == bool(truth[n]) else "MISS"
            print(f"   {mark} {n:12s} truth={'rough' if truth[n] else 'smooth':6s} P(rough)={probs[n]:.2f}")
    acc, probs, truth = results["synth"]
    acc_real = results["real"][0]
    ok = sorted(probs)

    # ---------- scale invariance check ----------
    print("\nscale invariance (same trace, resistance multiplied):")
    worst = 0.0
    for r in train[:6]:
        base = predict(model, r["x"])[0]
        for k in (0.01, 100.0):
            p = predict(model, r["x"] * k)[0]
            worst = max(worst, abs(p - base))
        print(f"   {r['name']:12s} x1={base:.4f}  x0.01={predict(model, r['x']*0.01)[0]:.4f}"
              f"  x100={predict(model, r['x']*100)[0]:.4f}")
    print(f"   largest shift from rescaling: {worst:.2e}")

    # ---------- synthetic stress test ----------
    P = fit_params(train)
    rng = np.random.default_rng(99)
    hit = 0
    n_s = 300
    for i in range(n_s):
        lab = "rough" if i % 2 == 0 else "smooth"
        x, _ = make_recording(lab, P, rng)
        p = predict(model, x)[0]
        hit += (p > .5) == (lab == "rough")
    print(f"\nsynthetic stress test ({n_s} generated traces spanning "
          f"~6 decades of resistance): {hit/n_s*100:.1f}% correct")

    # ---------- export ----------
    imp = model.named_steps["i"].statistics_
    sc = model.named_steps["s"]
    clf = model.named_steps["c"]
    # reference bands come from the REAL recordings only, so what the page
    # shows the user is measured data, not generated data
    Xr, yr, _, _ = build(train, train=True, n_crops=N_CROPS)
    ref = {}
    for i, f in enumerate(MODEL_FEATURES):
        col = Xr[:, i][np.isfinite(Xr[:, i])]
        ref[f] = {
            "rough": [float(np.percentile(Xr[(yr == 1), i][np.isfinite(Xr[(yr == 1), i])], q))
                      for q in (10, 50, 90)],
            "smooth": [float(np.percentile(Xr[(yr == 0), i][np.isfinite(Xr[(yr == 0), i])], q))
                       for q in (10, 50, 90)],
            "lo": float(np.percentile(col, 1)),
            "hi": float(np.percentile(col, 99)),
        }

    bundle = {
        "version": 1,
        "features": MODEL_FEATURES,
        "labels": {f: FEATURE_LABEL[f] for f in MODEL_FEATURES},
        "impute": [float(v) for v in imp],
        "mean": [float(v) for v in sc.mean_],
        "scale": [float(v) for v in sc.scale_],
        "coef": [float(v) for v in clf.coef_[0]],
        "intercept": float(clf.intercept_[0]),
        "reference": ref,
        "training": {
            "n_recordings": len(train),
            "n_rough": int(sum(1 for r in train if r["label"] == "rough")),
            "n_smooth": int(sum(1 for r in train if r["label"] == "smooth")),
            "n_rows": int(X.shape[0]),
            "n_real_rows": int(n_real),
            "n_synth": int(N_SYN),
            "loso_accuracy": acc,
            "loso_accuracy_real_only": acc_real,
            "loso_detail": {n: {"p": probs[n], "truth": int(truth[n])} for n in ok},
            "dropped": [r["name"] for r in allr if r not in train],
            "max_overflow": MAX_OVF,
        },
    }
    with open("out/model.json", "w") as fh:
        json.dump(bundle, fh, indent=1)
    print(f"\nwrote out/model.json  ({len(MODEL_FEATURES)} features)")

    # feature weights, for the write-up
    order = np.argsort(-np.abs(clf.coef_[0]))
    print("\nmodel weights (standardised, positive pushes towards rough):")
    for i in order:
        print(f"   {MODEL_FEATURES[i]:14s} {clf.coef_[0][i]:+.3f}   {FEATURE_LABEL[MODEL_FEATURES[i]]}")

    # per-file predictions for the JS cross-check
    out = {}
    for r in [x for x in load_all()]:
        p, pc, spans, span = predict(model, r["x"])
        f0 = recording_matrix(r["x"], train=False)[0]
        out[r["name"]] = {"p": p, "label": r["label"], "span": list(span),
                          "features": [None if not np.isfinite(v) else float(v)
                                       for v in f0[0]]}
    with open("out/python_predictions.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote out/python_predictions.json for the JS cross-check")


if __name__ == "__main__":
    main()
