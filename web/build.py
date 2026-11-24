"""Assemble the single-file page: markup + styles + engine + model + samples."""
import json, os, pathlib

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent

SAMPLES = [
    ("S3_ROUGH.csv", "Rough, run S3", "measured 8 Oct, ~1e8 ohm"),
    ("S3_SMOOTH.csv", "Smooth, run S3", "measured 9 Oct, ~1e8 ohm"),
    ("T5Rough.csv", "Rough, run T5", "measured 9 Aug, ~1e5 ohm"),
    ("T5Smooth.csv", "Smooth, run T5", "measured 9 Aug, ~1e5 ohm"),
]


def main():
    page = (HERE / "page.html").read_text()
    engine = (HERE / "engine.js").read_text()
    app = (HERE / "app.js").read_text()
    model = json.loads((ROOT / "out" / "model.json").read_text())

    samples = []
    for fn, title, note in SAMPLES:
        samples.append({
            "file": fn, "title": title, "note": note,
            "csv": (ROOT / "data" / fn).read_text(),
        })

    # the engine attaches itself to window.SurfaceEngine when not in node
    engine = engine.replace(
        'typeof module !== "undefined" && module.exports ? module.exports : window',
        "window")

    out = [
        page,
        "<script>\n", engine, "\n</script>\n",
        "<script>\nvar MODEL = ", json.dumps(model, separators=(",", ":")), ";\n",
        "var SAMPLES = ", json.dumps(samples, separators=(",", ":")), ";\n</script>\n",
        "<script>\n", app, "\n</script>\n",
    ]
    body = "".join(out)
    dest = ROOT / "out" / "index.html"
    dest.write_text(body)
    print(f"wrote {dest}  {dest.stat().st_size/1024:.1f} KB")

    # A standalone copy that opens correctly by double-clicking. index.html is
    # built for a host that supplies the document skeleton and a small reset;
    # without them a browser falls into quirks mode, so this version carries
    # its own.
    local = ROOT / "out" / "surface_classifier_local.html"
    local.write_text(
        "<!doctype html>\n<html lang=\"en\">\n<head>\n"
        "<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1,viewport-fit=cover\">\n"
        "<style>\n"
        "html{color-scheme:light}\n"
        "body{margin:0;font:14px system-ui,-apple-system,sans-serif}\n"
        "img{max-width:100%}\n"
        "[hidden]{display:none!important}\n"
        "</style>\n"
        + body
        + "\n</body>\n</html>\n"
    )
    print(f"wrote {local}  {local.stat().st_size/1024:.1f} KB   (open this one locally)")


if __name__ == "__main__":
    main()
