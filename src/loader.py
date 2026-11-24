"""Parse the Keithley-style resistance logs into clean arrays."""
import os, re, csv
import numpy as np

OVERFLOW = 9.9e37
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def parse_csv(path):
    """Return (readings, meta). Overflow codes become np.nan."""
    vals = []
    meta = {"start_date": None, "start_time": None, "n_raw": 0}
    with open(path, "r", newline="") as fh:
        rows = list(csv.reader(fh))
    for row in rows:
        if not row:
            continue
        c0 = row[0].strip()
        if c0.lower().startswith("start date"):
            meta["start_date"] = row[1].strip() if len(row) > 1 else None
            if len(row) > 3:
                meta["start_time"] = row[3].strip()
            continue
        if c0.lower().startswith("reading"):
            continue
        if len(row) < 2:
            continue
        try:
            idx = int(float(c0))
        except ValueError:
            continue
        txt = row[1].strip()
        if not txt:
            continue
        try:
            v = float(txt)
        except ValueError:
            continue
        vals.append(v)
    arr = np.array(vals, dtype=float)
    meta["n_raw"] = arr.size
    bad = (arr >= 1e30) | (~np.isfinite(arr)) | (arr <= 0)
    arr = arr.copy()
    arr[bad] = np.nan
    meta["n_overflow"] = int(bad.sum())
    return arr, meta


def label_from_name(name):
    u = name.upper()
    if "ROUGH" in u:
        return "rough"
    if "SMOOTH" in u:
        return "smooth"
    return None


def load_all(data_dir=DATA_DIR):
    out = []
    for fn in sorted(os.listdir(data_dir)):
        if not fn.lower().endswith(".csv"):
            continue
        arr, meta = parse_csv(os.path.join(data_dir, fn))
        out.append({
            "file": fn,
            "name": os.path.splitext(fn)[0],
            "label": label_from_name(fn),
            "x": arr,
            "meta": meta,
        })
    return out


if __name__ == "__main__":
    recs = load_all()
    print(f"{'file':16s} {'label':7s} {'n':>5s} {'ovf':>5s} {'ovf%':>6s} "
          f"{'min':>10s} {'median':>10s} {'max':>10s} {'decades':>8s}")
    for r in recs:
        x = r["x"]
        g = x[np.isfinite(x)]
        pct = 100.0 * r["meta"]["n_overflow"] / max(1, r["meta"]["n_raw"])
        if g.size:
            dec = np.log10(g.max() / g.min())
            print(f"{r['name']:16s} {str(r['label']):7s} {r['meta']['n_raw']:5d} "
                  f"{r['meta']['n_overflow']:5d} {pct:6.1f} "
                  f"{g.min():10.3g} {np.median(g):10.3g} {g.max():10.3g} {dec:8.2f}")
        else:
            print(f"{r['name']:16s} {str(r['label']):7s} {r['meta']['n_raw']:5d} ALL OVERFLOW")
