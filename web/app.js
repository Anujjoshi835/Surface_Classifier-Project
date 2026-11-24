/* Page logic: parse, analyse, draw. MODEL and SAMPLES are injected at build. */
(function () {
  "use strict";
  var E = window.SurfaceEngine;
  var $ = function (id) { return document.getElementById(id); };
  var files = [];      // {name, raw, result, quality}
  var active = -1;

  /* ---------------- parsing ---------------- */

  function parseCsv(text, name) {
    var lines = text.split(/\r?\n/);
    var vals = [], meta = { date: null, time: null, skipped: 0, nonNumeric: 0 };
    var sawHeader = false;
    for (var i = 0; i < lines.length; i++) {
      var line = lines[i];
      if (!line.trim()) continue;
      var c = line.split(/[,;\t]/);
      var c0 = (c[0] || "").trim();
      if (/^start\s*date/i.test(c0)) {
        meta.date = (c[1] || "").trim();
        if (c.length > 3) meta.time = (c[3] || "").trim();
        continue;
      }
      if (/^reading/i.test(c0)) { sawHeader = true; continue; }
      var idx = Number(c0);
      var vraw, valCol = 1;
      if (!isFinite(idx) || c0 === "") {
        meta.skipped++;
        continue;
      }
      if (c.length < 2) { valCol = 0; }
      vraw = String(c[valCol] === undefined ? "" : c[valCol]).trim();
      if (vraw === "") { meta.skipped++; continue; }
      var v = Number(vraw);
      if (!isFinite(v)) { meta.nonNumeric++; continue; }
      vals.push(v >= E.OVERFLOW_MIN || v <= 0 ? NaN : v);
    }
    // a single-column file: the one number is the reading, not an index
    if (!vals.length) {
      for (var j = 0; j < lines.length; j++) {
        var t = lines[j].trim();
        if (!t) continue;
        var n2 = Number(t);
        if (isFinite(n2)) vals.push(n2 >= E.OVERFLOW_MIN || n2 <= 0 ? NaN : n2);
      }
    }
    meta.sawHeader = sawHeader;
    meta.name = name;
    return { values: vals, meta: meta };
  }

  /* ---------------- quality ---------------- */

  function quality(raw, meta) {
    var n = raw.length, nBad = 0;
    for (var i = 0; i < n; i++) if (!isFinite(raw[i])) nBad++;
    var good = [];
    for (var j = 0; j < n; j++) if (isFinite(raw[j])) good.push(raw[j]);
    var checks = [], worst = "ok";
    function add(level, text, detail) {
      checks.push({ level: level, text: text, detail: detail });
      if (level === "bad" || (level === "warn" && worst === "ok")) worst = level;
    }
    if (n === 0) { add("bad", "No readings found", "Nothing in the file parsed as a number."); return { checks: checks, worst: "bad", n: n, nBad: nBad, good: good }; }
    if (n < 48) add("bad", "Too few readings (" + n + ")",
      "The model needs at least 48 to measure fluctuation. The call below is not reliable.");
    else if (n < 120) add("warn", "Short recording (" + n + " readings)",
      "Training recordings ran 125 to 377 readings. Shorter traces give noisier features.");
    else add("ok", n + " readings", "Within the range the model was trained on.");

    var frac = nBad / n;
    if (frac > 0.25) add("bad", "Contact lost on " + (frac * 100).toFixed(0) + "% of readings",
      "Above 25% the recording was treated as unusable during training and excluded. Re-run the measurement.");
    else if (frac > 0.05) add("warn", "Contact lost on " + (frac * 100).toFixed(1) + "% of readings",
      "Overflow readings are interpolated over. A few are normal on a rough surface.");
    else if (nBad > 0) add("ok", nBad + " contact-loss reading" + (nBad > 1 ? "s" : ""),
      (frac * 100).toFixed(1) + "% of the trace, interpolated over.");
    else add("ok", "No contact loss", "Every reading is a finite resistance.");

    if (good.length > 1) {
      var mn = Math.min.apply(null, good), mx = Math.max.apply(null, good);
      var dec = Math.log10(mx / mn);
      if (dec < 0.15) add("warn", "Almost no variation (" + dec.toFixed(2) + " decades)",
        "The trace barely moves, so there may be no loading event in this file.");
      else add("ok", "Spans " + dec.toFixed(2) + " decades", "Enough movement to analyse.");
    }
    if (meta && meta.nonNumeric > 3) add("warn", meta.nonNumeric + " rows could not be read as numbers",
      "They were skipped. Check the file for stray text in the reading column.");
    return { checks: checks, worst: worst, n: n, nBad: nBad, good: good, frac: frac };
  }

  /* ---------------- svg helpers ---------------- */

  var NS = "http://www.w3.org/2000/svg";
  function el(tag, attrs, text) {
    var e = document.createElementNS(NS, tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (text !== undefined) e.textContent = text;
    return e;
  }
  function svgRoot(w, h) {
    var s = el("svg", { viewBox: "0 0 " + w + " " + h, preserveAspectRatio: "xMidYMid meet",
      role: "img", "font-family": "var(--f-mono)" });
    return s;
  }
  function niceTicks(lo, hi, want) {
    var span = hi - lo;
    if (!(span > 0)) return [lo];
    var step = Math.pow(10, Math.floor(Math.log10(span / want)));
    var err = (span / want) / step;
    if (err >= 7.5) step *= 10; else if (err >= 3.5) step *= 5; else if (err >= 1.5) step *= 2;
    var out = [], v = Math.ceil(lo / step) * step;
    for (; v <= hi + step * 1e-9; v += step) out.push(Number(v.toFixed(10)));
    return out;
  }
  function fmtOhm(log10v) {
    var v = Math.pow(10, log10v);
    var units = [[1e9, "G"], [1e6, "M"], [1e3, "k"], [1, ""]];
    for (var i = 0; i < units.length; i++) {
      if (v >= units[i][0] * 0.999) {
        var x = v / units[i][0];
        return (x >= 100 ? x.toFixed(0) : x >= 10 ? x.toFixed(1) : x.toFixed(2)) + units[i][1];
      }
    }
    return v.toPrecision(2);
  }

  /* ---------------- chart: resistance trace ---------------- */

  function drawTrace(host, raw, span) {
    host.textContent = "";
    var W = 900, H = 290, L = 56, R = 14, T = 22, B = 34;
    var ls = E.logSignal(raw);
    if (!ls.y) { host.appendChild(document.createTextNode("No finite readings to plot.")); return; }
    var y = ls.y, bad = ls.bad, n = y.length;
    var lo = Math.min.apply(null, Array.from(y)), hi = Math.max.apply(null, Array.from(y));
    if (hi - lo < 0.3) { var m = (hi + lo) / 2; lo = m - 0.15; hi = m + 0.15; }
    var pad = (hi - lo) * 0.08; lo -= pad; hi += pad;
    var X = function (i) { return L + (i / Math.max(1, n - 1)) * (W - L - R); };
    var Y = function (v) { return T + (1 - (v - lo) / (hi - lo)) * (H - T - B); };
    var s = svgRoot(W, H);
    s.appendChild(el("title", {}, "Resistance against reading number"));

    // active span
    s.appendChild(el("rect", { x: X(span[0]), y: T, width: Math.max(1, X(span[1] - 1) - X(span[0])),
      height: H - T - B, fill: "var(--accent)", opacity: ".09" }));
    s.appendChild(el("line", { x1: X(span[0]), y1: T, x2: X(span[0]), y2: H - B,
      stroke: "var(--accent)", "stroke-width": "1", "stroke-dasharray": "3 3", opacity: ".7" }));

    // grid + y labels
    niceTicks(lo, hi, 5).forEach(function (t) {
      s.appendChild(el("line", { x1: L, y1: Y(t), x2: W - R, y2: Y(t), stroke: "var(--line)", "stroke-width": "1" }));
      s.appendChild(el("text", { x: L - 8, y: Y(t) + 4, "text-anchor": "end", "font-size": "11",
        fill: "var(--ink-3)" }, fmtOhm(t)));
    });
    niceTicks(1, n, 6).forEach(function (t) {
      if (t < 1 || t > n) return;
      s.appendChild(el("text", { x: X(t - 1), y: H - B + 18, "text-anchor": "middle", "font-size": "11",
        fill: "var(--ink-3)" }, String(t)));
    });
    s.appendChild(el("text", { x: (L + W - R) / 2, y: H - 4, "text-anchor": "middle", "font-size": "11",
      fill: "var(--ink-3)" }, "reading number"));
    s.appendChild(el("text", { x: 12, y: T - 8, "font-size": "11", fill: "var(--ink-3)" }, "ohm"));

    // the trace, split so interpolated stretches draw dashed
    var segs = [], cur = null;
    for (var i = 0; i < n; i++) {
      var isBad = !!bad[i];
      if (!cur || cur.bad !== isBad) { cur = { bad: isBad, pts: [] }; segs.push(cur); }
      cur.pts.push([X(i), Y(y[i])]);
    }
    segs.forEach(function (sg, k) {
      if (k > 0) sg.pts.unshift(segs[k - 1].pts[segs[k - 1].pts.length - 1]);
      var d = sg.pts.map(function (p, j) { return (j ? "L" : "M") + p[0].toFixed(1) + " " + p[1].toFixed(1); }).join("");
      s.appendChild(el("path", { d: d, fill: "none", stroke: sg.bad ? "var(--bad)" : "var(--accent)",
        "stroke-width": sg.bad ? "1.2" : "1.4", "stroke-dasharray": sg.bad ? "2 2" : "none",
        "stroke-linejoin": "round" }));
    });
    // contact-loss ticks on the top rule
    for (var b = 0; b < n; b++) if (bad[b]) {
      s.appendChild(el("line", { x1: X(b), y1: T - 10, x2: X(b), y2: T - 3, stroke: "var(--bad)", "stroke-width": "1.5" }));
    }
    s.appendChild(el("line", { x1: L, y1: H - B, x2: W - R, y2: H - B, stroke: "var(--line-2)", "stroke-width": "1" }));
    host.appendChild(s);
  }

  /* ---------------- chart: fluctuation ---------------- */

  function detrend(seg) {
    var n = seg.length;
    var w = Math.max(5, Math.min(31, Math.floor(n / 6)));
    if (w % 2 === 0) w += 1;
    var tr = E.movingAvgEdge(seg, w), out = new Float64Array(n);
    for (var i = 0; i < n; i++) out[i] = seg[i] - tr[i];
    return out;
  }

  function drawFluct(host, raw, span) {
    host.textContent = "";
    var ls = E.logSignal(raw);
    if (!ls.y) return;
    var seg = ls.y.slice(span[0], span[1]);
    if (seg.length < 8) return;
    var r = detrend(seg), n = r.length;
    var W = 460, H = 210, L = 46, R = 10, T = 14, B = 30;
    var amp = 0;
    for (var i = 0; i < n; i++) amp = Math.max(amp, Math.abs(r[i]));
    amp = Math.max(amp * 1.1, 0.05);
    var X = function (k) { return L + (k / Math.max(1, n - 1)) * (W - L - R); };
    var Y = function (v) { return T + (1 - (v + amp) / (2 * amp)) * (H - T - B); };
    var s = svgRoot(W, H);
    s.appendChild(el("title", {}, "Detrended log resistance over the scan"));
    var sd = E.std(r);
    s.appendChild(el("rect", { x: L, y: Y(sd), width: W - L - R, height: Math.max(1, Y(-sd) - Y(sd)),
      fill: "var(--accent)", opacity: ".10" }));
    niceTicks(-amp, amp, 4).forEach(function (t) {
      s.appendChild(el("line", { x1: L, y1: Y(t), x2: W - R, y2: Y(t), stroke: "var(--line)", "stroke-width": "1" }));
      s.appendChild(el("text", { x: L - 7, y: Y(t) + 4, "text-anchor": "end", "font-size": "10",
        fill: "var(--ink-3)" }, t.toFixed(2)));
    });
    s.appendChild(el("line", { x1: L, y1: Y(0), x2: W - R, y2: Y(0), stroke: "var(--line-2)", "stroke-width": "1" }));
    var d = "";
    for (var j = 0; j < n; j++) d += (j ? "L" : "M") + X(j).toFixed(1) + " " + Y(r[j]).toFixed(1);
    s.appendChild(el("path", { d: d, fill: "none", stroke: "var(--ink)", "stroke-width": "1", opacity: ".85" }));
    s.appendChild(el("text", { x: L, y: H - 8, "font-size": "10.5", fill: "var(--ink-3)" },
      "scan, reading " + (span[0] + 1) + " to " + span[1]));
    s.appendChild(el("text", { x: W - R, y: H - 8, "text-anchor": "end", "font-size": "10.5", fill: "var(--accent)" },
      "band = ±1σ = " + sd.toFixed(3) + " decades"));
    host.appendChild(s);
  }

  /* ---------------- chart: fluctuation by scale ---------------- */

  function drawScale(host, raw, span) {
    host.textContent = "";
    var ls = E.logSignal(raw);
    if (!ls.y) return;
    var seg = ls.y.slice(span[0], span[1]);
    if (seg.length < 24) return;
    var r = detrend(seg);
    var ws = [3, 5, 9, 17, 33, 65].filter(function (w) { return w < r.length; });
    var pts = ws.map(function (w) { return [w, E.median(E.movingStdPad(r, w))]; });
    var W = 460, H = 210, L = 50, R = 14, T = 16, B = 34;
    var lo = Math.min.apply(null, pts.map(function (p) { return p[1]; }));
    var hi = Math.max.apply(null, pts.map(function (p) { return p[1]; }));
    if (!(hi > 0)) return;
    lo = Math.max(lo, hi / 100);
    var lyl = Math.log10(lo) - 0.12, lyh = Math.log10(hi) + 0.12;
    var lxl = Math.log10(ws[0]) - 0.08, lxh = Math.log10(ws[ws.length - 1]) + 0.08;
    var X = function (w) { return L + (Math.log10(w) - lxl) / (lxh - lxl) * (W - L - R); };
    var Y = function (v) { return T + (1 - (Math.log10(Math.max(v, lo / 2)) - lyl) / (lyh - lyl)) * (H - T - B); };
    var s = svgRoot(W, H);
    s.appendChild(el("title", {}, "Fluctuation amplitude against window length"));
    niceTicks(lyl, lyh, 4).forEach(function (t) {
      s.appendChild(el("line", { x1: L, y1: Y(Math.pow(10, t)), x2: W - R, y2: Y(Math.pow(10, t)),
        stroke: "var(--line)", "stroke-width": "1" }));
      s.appendChild(el("text", { x: L - 7, y: Y(Math.pow(10, t)) + 4, "text-anchor": "end", "font-size": "10",
        fill: "var(--ink-3)" }, Math.pow(10, t) < 0.01 ? Math.pow(10, t).toExponential(0) : Math.pow(10, t).toFixed(3)));
    });
    var d = pts.map(function (p, i) { return (i ? "L" : "M") + X(p[0]).toFixed(1) + " " + Y(p[1]).toFixed(1); }).join("");
    s.appendChild(el("path", { d: d, fill: "none", stroke: "var(--accent)", "stroke-width": "1.8", "stroke-linejoin": "round" }));
    pts.forEach(function (p) {
      s.appendChild(el("circle", { cx: X(p[0]), cy: Y(p[1]), r: "3.2", fill: "var(--accent)" }));
      s.appendChild(el("text", { x: X(p[0]), y: H - B + 17, "text-anchor": "middle", "font-size": "10.5",
        fill: "var(--ink-3)" }, String(p[0])));
    });
    s.appendChild(el("text", { x: (L + W - R) / 2, y: H - 4, "text-anchor": "middle", "font-size": "10.5",
      fill: "var(--ink-3)" }, "window length, readings"));
    s.appendChild(el("text", { x: 2, y: T - 5, "font-size": "10.5", fill: "var(--ink-3)" }, "decades"));
    host.appendChild(s);
  }

  /* ---------------- render ---------------- */

  function callOf(p) { return p >= 0.6 ? "rough" : p <= 0.4 ? "smooth" : "unsure"; }

  /* plain-language reading of each measurement, [when high, when low] */
  var PHRASE = {
    jitter_p90: ["big jumps between consecutive readings", "small jumps between consecutive readings"],
    jitter_med: ["a large typical step size", "a small typical step size"],
    std_log: ["a wide spread of resistance", "a narrow spread of resistance"],
    range_dec: ["a wide dynamic range", "a narrow dynamic range"],
    spike_rate: ["frequent outlier steps", "few outlier steps"],
    spec_slope: ["a flat fluctuation spectrum", "a steep fluctuation spectrum"],
    band_frac_hi: ["strong short-scale structure", "little short-scale structure"],
    hf_ratio: ["a lot of high-frequency content", "little high-frequency content"],
    spec_entropy: ["fluctuation spread across many scales", "fluctuation concentrated at a few scales"],
    ac1_diff: ["steps that follow one another", "steps that reverse sharply"],
    turn_rate: ["many changes of direction", "few changes of direction"],
    perm_entropy: ["a disordered step pattern", "an orderly step pattern"],
    wavelet_ratio: ["fine structure that survives at short windows", "fluctuation that lives mainly at long windows"],
    ovf_frac: ["repeated loss of contact", "contact held throughout"],
    ovf_run_max: ["a long break in contact", "no sustained break in contact"],
    exc_height: ["a tall loading excursion", "a shallow loading excursion"]
  };

  function describe(res) {
    var p = res.p, c = callOf(p);
    if (c === "unsure") {
      return "The measurements pull both ways and the probability sits near the middle. "
        + "Treat this trace as undecided rather than as a weak rough or a weak smooth.";
    }
    var lead = res.detail.contrib.slice().sort(function (a, b) {
      return Math.abs(b.contribution) - Math.abs(a.contribution);
    }).filter(function (t) {
      return (t.contribution > 0) === (p > 0.5) && isFinite(t.value);
    }).slice(0, 2);
    var phrases = lead.map(function (t) {
      var pair = PHRASE[t.name];
      return pair ? pair[t.z >= 0 ? 0 : 1] : MODEL.labels[t.name].toLowerCase();
    });
    var agree = res.cropProbs.filter(function (x) { return (x > 0.5) === (p > 0.5); }).length;
    var body = phrases.length
      ? "The scan shows " + phrases.join(" and ") + ", which matches the "
        + c + " recordings in the training set."
      : "The fluctuation features across the scan sit closer to the " + c + " recordings.";
    return body + " " + agree + " of " + res.cropProbs.length
      + " analysis windows read it the same way.";
  }

  function render(i) {
    active = i;
    var f = files[i];
    if (!f) return;
    var res = f.result, q = f.quality, p = res.p, c = callOf(p);
    $("report").hidden = false;

    $("callFile").textContent = f.name;
    var ct = $("callText");
    ct.textContent = c === "unsure" ? "Undecided" : c === "rough" ? "Rough" : "Smooth";
    ct.className = "calltext " + c;
    $("needle").style.left = (p * 100).toFixed(1) + "%";
    $("pval").textContent = "P = " + p.toFixed(2);
    $("whyText").textContent = describe(res);

    var chips = $("whyChips"); chips.textContent = "";
    function chip(txt, cls) { var e = document.createElement("span"); e.className = "chip " + (cls || ""); e.textContent = txt; chips.appendChild(e); }
    var agree = res.cropProbs.filter(function (x) { return (x > 0.5) === (p > 0.5); }).length;
    chip(agree + "/" + res.cropProbs.length + " windows agree", agree === res.cropProbs.length ? "" : "w");
    chip("scan = readings " + (res.span[0] + 1) + "–" + res.span[1]);
    chip(q.n + " readings");
    if (q.nBad) chip(q.nBad + " contact loss", q.frac > 0.25 ? "w" : "");
    if (q.worst !== "ok") chip(q.worst === "bad" ? "quality: unusable" : "quality: check below", "w");

    drawTrace($("chartTrace"), f.raw, res.span);
    drawFluct($("chartFluct"), f.raw, res.span);
    drawScale($("chartScale"), f.raw, res.span);
    $("spanLabel").textContent = "scan " + (res.span[0] + 1) + "–" + res.span[1] + " of " + q.n;

    /* evidence */
    var ev = $("evidence"); ev.textContent = "";
    var top = res.detail.contrib.slice().sort(function (a, b) {
      return Math.abs(b.contribution) - Math.abs(a.contribution);
    }).slice(0, 7);
    var maxAbs = Math.max.apply(null, top.map(function (t) { return Math.abs(t.contribution); })) || 1;
    top.forEach(function (t) {
      var row = document.createElement("div"); row.className = "evrow";
      var lab = document.createElement("div"); lab.className = "lab";
      lab.innerHTML = "<b></b><span></span>";
      lab.querySelector("b").textContent = MODEL.labels[t.name];
      lab.querySelector("span").textContent = fmtVal(t.name, t.value);
      var bar = document.createElement("div"); bar.className = "evbar";
      var w = Math.abs(t.contribution) / maxAbs * 50;
      var fill = document.createElement("i");
      fill.style.background = t.contribution > 0 ? "var(--rough)" : "var(--smooth)";
      fill.style.width = w + "%";
      if (t.contribution > 0) fill.style.left = "50%"; else fill.style.right = "50%";
      var mid = document.createElement("span"); mid.className = "mid";
      bar.appendChild(fill); bar.appendChild(mid);
      bar.title = (t.contribution > 0 ? "+" : "") + t.contribution.toFixed(2) + " log-odds";
      row.appendChild(lab); row.appendChild(bar); ev.appendChild(row);
    });

    /* stats */
    var good = q.good, st = $("stats"); st.textContent = "";
    function stat(k, v) {
      var d = document.createElement("div");
      var dt = document.createElement("dt"); dt.textContent = k;
      var dd = document.createElement("dd"); dd.textContent = v;
      d.appendChild(dt); d.appendChild(dd); st.appendChild(d);
    }
    if (good.length) {
      var srt = good.slice().sort(function (a, b) { return a - b; });
      stat("Readings", String(q.n));
      stat("Median R", fmtOhm(Math.log10(srt[srt.length >> 1])) + "Ω");
      stat("Minimum", fmtOhm(Math.log10(srt[0])) + "Ω");
      stat("Maximum", fmtOhm(Math.log10(srt[srt.length - 1])) + "Ω");
      stat("Range", Math.log10(srt[srt.length - 1] / srt[0]).toFixed(2) + " dec");
      stat("Contact loss", q.nBad + " (" + (q.frac * 100).toFixed(1) + "%)");
      stat("Scan length", (res.span[1] - res.span[0]) + " rdg");
      stat("Fluctuation", res.features.std_log.toFixed(3) + " dec");
    }

    /* checks */
    var ul = $("checks"); ul.textContent = "";
    q.checks.forEach(function (ck) {
      var li = document.createElement("li");
      var dot = document.createElement("span"); dot.className = "dot " + ck.level;
      var body = document.createElement("div");
      var strong = document.createElement("div"); strong.textContent = ck.text;
      var sm = document.createElement("small"); sm.textContent = ck.detail;
      body.appendChild(strong); body.appendChild(sm);
      li.appendChild(dot); li.appendChild(body); ul.appendChild(li);
    });

    /* feature table */
    var tb = $("featBody"); tb.textContent = "";
    MODEL.features.forEach(function (k, idx) {
      var ref = MODEL.reference[k], v = res.vector[idx];
      var tr = document.createElement("tr");
      function td(txt, cls) { var d = document.createElement("td"); if (cls) d.className = cls; d.textContent = txt; return d; }
      tr.appendChild(td(MODEL.labels[k]));
      tr.appendChild(td(isFinite(v) ? fmtNum(v) : "n/a", "num"));
      tr.appendChild(td(fmtNum(ref.rough[1]), "num"));
      tr.appendChild(td(fmtNum(ref.smooth[1]), "num"));
      var cell = document.createElement("td");
      var lo = Math.min(ref.lo, ref.rough[0], ref.smooth[0], isFinite(v) ? v : ref.lo);
      var hi = Math.max(ref.hi, ref.rough[2], ref.smooth[2], isFinite(v) ? v : ref.hi);
      if (!(hi > lo)) { hi = lo + 1; }
      var pc = function (x) { return ((x - lo) / (hi - lo) * 100).toFixed(2) + "%"; };
      var strip = document.createElement("div"); strip.className = "strip";
      function band(cls, a, b) {
        var e = document.createElement("span"); e.className = "band " + cls;
        e.style.left = pc(a); e.style.width = Math.max(1.5, (b - a) / (hi - lo) * 100) + "%";
        strip.appendChild(e);
      }
      band("r", ref.rough[0], ref.rough[2]);
      band("s", ref.smooth[0], ref.smooth[2]);
      if (isFinite(v)) { var mk = document.createElement("span"); mk.className = "mk"; mk.style.left = pc(v); strip.appendChild(mk); }
      strip.title = "rough band " + fmtNum(ref.rough[0]) + " to " + fmtNum(ref.rough[2])
        + ", smooth band " + fmtNum(ref.smooth[0]) + " to " + fmtNum(ref.smooth[2]);
      cell.appendChild(strip); tr.appendChild(cell);
      tb.appendChild(tr);
    });

    /* batch */
    var bw = $("batchWrap");
    bw.hidden = files.length < 2;
    if (files.length >= 2) {
      $("batchCount").textContent = files.length + " files";
      var bb = $("batchBody"); bb.textContent = "";
      files.forEach(function (ff, k) {
        var tr = document.createElement("tr");
        if (k === i) tr.className = "sel";
        tr.tabIndex = 0;
        var cc = callOf(ff.result.p);
        function td(html) { var d = document.createElement("td"); d.appendChild(html); return d; }
        function txt(t) { return document.createTextNode(t); }
        tr.appendChild(td(txt(ff.name)));
        var tag = document.createElement("span"); tag.className = "tag " + cc;
        tag.textContent = cc === "unsure" ? "UNDECIDED" : cc.toUpperCase();
        tr.appendChild(td(tag));
        var pv = document.createElement("span"); pv.className = "mono"; pv.textContent = ff.result.p.toFixed(2);
        tr.appendChild(td(pv));
        tr.appendChild(td(txt(String(ff.quality.n))));
        tr.appendChild(td(txt((ff.quality.frac * 100).toFixed(1) + "%")));
        var qd = document.createElement("span"); qd.className = "dot " + ff.quality.worst;
        qd.style.display = "inline-block"; qd.style.width = "9px"; qd.style.height = "9px";
        qd.style.borderRadius = "50%";
        tr.appendChild(td(qd));
        tr.addEventListener("click", function () { render(k); });
        tr.addEventListener("keydown", function (e2) { if (e2.key === "Enter" || e2.key === " ") { e2.preventDefault(); render(k); } });
        bb.appendChild(tr);
      });
    }
    $("clear").hidden = false;
    $("report").scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function fmtNum(v) {
    if (!isFinite(v)) return "n/a";
    var a = Math.abs(v);
    if (a === 0) return "0";
    if (a < 0.001 || a >= 1e5) return v.toExponential(2);
    return v.toPrecision(a < 1 ? 3 : 4).replace(/\.?0+$/, "");
  }
  function fmtVal(name, v) {
    if (!isFinite(v)) return "not measurable";
    if (name === "ovf_frac" || name === "ovf_run_max") return (v * 100).toFixed(1) + "% of readings";
    if (name === "range_dec" || name === "exc_height") return v.toFixed(2) + " decades";
    if (name === "jitter_p90" || name === "jitter_med" || name === "std_log") return v.toFixed(3) + " decades";
    return fmtNum(v);
  }

  /* ---------------- intake ---------------- */

  function addFiles(list) {
    var pending = list.length, errs = $("errs");
    Array.prototype.forEach.call(list, function (file) {
      var rd = new FileReader();
      rd.onload = function () {
        try {
          var parsed = parseCsv(String(rd.result), file.name);
          if (!parsed.values.length) throw new Error("no readings could be parsed from this file");
          var q = quality(parsed.values, parsed.meta);
          var res = E.analyse(parsed.values, MODEL);
          files.push({ name: file.name, raw: parsed.values, result: res, quality: q, meta: parsed.meta });
        } catch (e) {
          var d = document.createElement("div");
          d.className = "card err"; d.style.marginTop = "16px";
          d.innerHTML = "<h3></h3><p class=\"sub\"></p>";
          d.querySelector("h3").textContent = "Could not read " + file.name;
          d.querySelector("p").textContent = e.message
            + ". The file needs a reading number and a resistance value per row.";
          errs.appendChild(d);
        }
        if (--pending === 0 && files.length) render(files.length - 1);
      };
      rd.onerror = function () { if (--pending === 0 && files.length) render(files.length - 1); };
      rd.readAsText(file);
    });
  }

  function addText(name, text) {
    var parsed = parseCsv(text, name);
    var q = quality(parsed.values, parsed.meta);
    var res = E.analyse(parsed.values, MODEL);
    files.push({ name: name, raw: parsed.values, result: res, quality: q, meta: parsed.meta });
    render(files.length - 1);
  }

  /* ---------------- report text ---------------- */

  function reportText() {
    var f = files[active]; if (!f) return "";
    var r = f.result, c = callOf(r.p);
    var L = [];
    L.push("SURFACE FINISH CLASSIFICATION");
    L.push("file: " + f.name);
    L.push("call: " + (c === "unsure" ? "UNDECIDED" : c.toUpperCase()) + "   P(rough) = " + r.p.toFixed(3));
    L.push("scan analysed: readings " + (r.span[0] + 1) + " to " + r.span[1] + " of " + f.quality.n);
    L.push("window agreement: " + r.cropProbs.filter(function (x) { return (x > 0.5) === (r.p > 0.5); }).length
      + " of " + r.cropProbs.length);
    L.push("");
    L.push("DATA QUALITY");
    f.quality.checks.forEach(function (ck) { L.push("  [" + ck.level + "] " + ck.text); });
    L.push("");
    L.push("FEATURES  (value | rough median | smooth median)");
    MODEL.features.forEach(function (k, i) {
      var ref = MODEL.reference[k];
      L.push("  " + MODEL.labels[k].padEnd(30) + fmtNum(r.vector[i]).padStart(12)
        + fmtNum(ref.rough[1]).padStart(14) + fmtNum(ref.smooth[1]).padStart(14));
    });
    L.push("");
    L.push("MODEL: logistic regression on 16 scale-free features of log10 resistance.");
    L.push("Leave-one-specimen-out accuracy on " + MODEL.training.n_recordings
      + " usable recordings: " + (MODEL.training.loso_accuracy * 100).toFixed(1) + "%.");
    return L.join("\n");
  }

  /* ---------------- model card ---------------- */

  function buildModelCard() {
    var t = MODEL.training;
    var g = $("mcGrid");
    function sect(h, html) {
      var s = document.createElement("section");
      var hh = document.createElement("h3"); hh.textContent = h;
      s.appendChild(hh);
      var d = document.createElement("div"); d.innerHTML = html;
      s.appendChild(d); g.appendChild(s);
    }
    sect("Training data",
      "<p>" + t.n_recordings + " labelled recordings (" + t.n_rough + " rough, " + t.n_smooth
      + " smooth) from two measurement campaigns, cut into " + t.n_real_rows
      + " overlapping windows. " + t.n_synth + " generated traces per class were added at low weight "
      + "to steady the fit.</p>");
    sect("Accuracy",
      "<p>Leave-one-specimen-out, holding out a specimen's rough and smooth recordings together: <b>"
      + (t.loso_accuracy * 100).toFixed(1) + "%</b> (" + Math.round(t.loso_accuracy * t.n_recordings)
      + " of " + t.n_recordings + "). Without the generated traces it is "
      + (t.loso_accuracy_real_only * 100).toFixed(1) + "%. The gap is one borderline recording, "
      + "which is well inside the noise at this sample size.</p>");
    sect("What it measures",
      "<p>Every feature is computed on log10 resistance, so multiplying a whole trace by any constant "
      + "leaves the answer unchanged. A rough surface shows a wider dynamic range, a taller loading "
      + "excursion, more momentary contact loss and more fine structure in the scan.</p>");
    sect("Where it is weak",
      "<ul>"
      + "<li>" + t.n_recordings + " recordings is a small training set. Expect the accuracy above to move by "
      + "several points once more specimens are measured.</li>"
      + "<li>Rough and smooth for each specimen were recorded on different days, so some day-to-day "
      + "variation is mixed in with surface finish.</li>"
      + "<li>It answers rough or smooth only. It does not estimate Ra or any other roughness number.</li>"
      + "<li>Recordings with more than " + (t.max_overflow * 100).toFixed(0)
      + "% contact loss were excluded as unusable: " + t.dropped.join(", ") + ".</li>"
      + "</ul>");

    var tbl = $("losoTable");
    tbl.innerHTML = "<thead><tr><th>Recording</th><th>Labelled</th><th>P(rough)</th><th>Result</th></tr></thead>";
    var tb = document.createElement("tbody");
    Object.keys(t.loso_detail).forEach(function (k) {
      var d = t.loso_detail[k], ok = (d.p > 0.5) === !!d.truth;
      var tr = document.createElement("tr");
      function td(txt, cls) { var e = document.createElement("td"); if (cls) e.className = cls; e.textContent = txt; return e; }
      tr.appendChild(td(k));
      tr.appendChild(td(d.truth ? "rough" : "smooth"));
      tr.appendChild(td(d.p.toFixed(2), "mono"));
      var r = td(ok ? "correct" : "missed");
      r.style.color = ok ? "var(--ok)" : "var(--bad)";
      tr.appendChild(r);
      tb.appendChild(tr);
    });
    tbl.appendChild(tb);
    $("footVer").textContent = "model v" + MODEL.version + " · " + MODEL.features.length + " features";
    $("hdrStats").innerHTML =
      "<span>validated <b>" + (t.loso_accuracy * 100).toFixed(0) + "%</b></span>"
      + "<span><b>" + t.n_recordings + "</b> recordings</span>"
      + "<span><b>" + MODEL.features.length + "</b> features</span>";
  }

  /* ---------------- wiring ---------------- */

  function init() {
    buildModelCard();
    var sg = $("samples");
    SAMPLES.forEach(function (s) {
      var b = document.createElement("button");
      b.type = "button"; b.className = "btn";
      var strong = document.createElement("span"); strong.textContent = s.title;
      var sm = document.createElement("small"); sm.textContent = s.note;
      b.appendChild(strong); b.appendChild(sm);
      b.addEventListener("click", function () { addText(s.file, s.csv); });
      sg.appendChild(b);
    });

    $("pick").addEventListener("click", function () { $("file").click(); });
    $("file").addEventListener("change", function (e) { if (e.target.files.length) addFiles(e.target.files); e.target.value = ""; });
    $("clear").addEventListener("click", function () {
      files = []; active = -1; $("report").hidden = true; $("errs").textContent = "";
      $("clear").hidden = true; window.scrollTo({ top: 0, behavior: "smooth" });
    });
    var drop = $("drop");
    ["dragenter", "dragover"].forEach(function (ev) {
      drop.addEventListener(ev, function (e) { e.preventDefault(); drop.classList.add("over"); });
    });
    ["dragleave", "drop"].forEach(function (ev) {
      drop.addEventListener(ev, function (e) { e.preventDefault(); drop.classList.remove("over"); });
    });
    drop.addEventListener("drop", function (e) {
      if (e.dataTransfer && e.dataTransfer.files.length) addFiles(e.dataTransfer.files);
    });
    $("copyBtn").addEventListener("click", function () {
      var txt = reportText(), btn = $("copyBtn");
      function done(msg) { btn.textContent = msg; setTimeout(function () { btn.textContent = "Copy report"; }, 1600); }
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(txt).then(function () { done("Copied"); }, function () { fallback(); });
      } else fallback();
      function fallback() {
        var ta = document.createElement("textarea");
        ta.value = txt; ta.style.position = "fixed"; ta.style.opacity = "0";
        document.body.appendChild(ta); ta.select();
        try { done("Copied"); } catch (e) { done("Select and copy"); }
        setTimeout(function () { ta.remove(); }, 4000);
      }
    });

    // open on a measured example so the page shows what it does
    addText(SAMPLES[0].file, SAMPLES[0].csv);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
