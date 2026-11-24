/* Surface classifier engine.
   A line-for-line port of the Python feature pipeline, so the page computes
   exactly what the model was trained on. Verified against the Python
   implementation on every labelled recording. */
(function (root) {
  "use strict";

  var OVERFLOW_MIN = 1e30;

  /* ---------- small numeric helpers, matching numpy semantics ---------- */

  function median(a) {
    if (!a.length) return NaN;
    var s = Float64Array.from(a).sort();
    var n = s.length, h = n >> 1;
    return n % 2 ? s[h] : (s[h - 1] + s[h]) / 2;
  }

  function percentile(a, q) {            // numpy's default linear method
    if (!a.length) return NaN;
    var s = Float64Array.from(a).sort();
    var idx = (q / 100) * (s.length - 1);
    var lo = Math.floor(idx), hi = Math.ceil(idx);
    if (lo === hi) return s[lo];
    return s[lo] + (idx - lo) * (s[hi] - s[lo]);
  }

  function mean(a) {
    var t = 0;
    for (var i = 0; i < a.length; i++) t += a[i];
    return t / a.length;
  }

  function std(a) {
    if (!a.length) return NaN;
    var m = mean(a), t = 0;
    for (var i = 0; i < a.length; i++) { var d = a[i] - m; t += d * d; }
    return Math.sqrt(t / a.length);
  }

  function diff(a) {
    var o = new Float64Array(Math.max(0, a.length - 1));
    for (var i = 1; i < a.length; i++) o[i - 1] = a[i] - a[i - 1];
    return o;
  }

  function absArr(a) {
    var o = new Float64Array(a.length);
    for (var i = 0; i < a.length; i++) o[i] = Math.abs(a[i]);
    return o;
  }

  function polyfit1(x, y) {            // least squares slope + intercept
    var n = x.length, sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (var i = 0; i < n; i++) { sx += x[i]; sy += y[i]; sxx += x[i] * x[i]; sxy += x[i] * y[i]; }
    var den = n * sxx - sx * sx;
    if (Math.abs(den) < 1e-300) return [0, sy / n];
    var m = (n * sxy - sx * sy) / den;
    return [m, (sy - m * sx) / n];
  }

  /* real FFT power spectrum, |rfft(x)|^2, length n/2+1 */
  function rfftPower(x) {
    var n = x.length, k = (n >> 1) + 1, out = new Float64Array(k);
    for (var f = 0; f < k; f++) {
      var re = 0, im = 0, w = (-2 * Math.PI * f) / n;
      for (var t = 0; t < n; t++) {
        var a = w * t;
        re += x[t] * Math.cos(a);
        im += x[t] * Math.sin(a);
      }
      out[f] = re * re + im * im;
    }
    return out;
  }

  function hanning(M) {               // np.hanning
    var o = new Float64Array(M);
    if (M === 1) { o[0] = 1; return o; }
    for (var i = 0; i < M; i++) o[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (M - 1));
    return o;
  }

  function factorial(n) { var r = 1; for (var i = 2; i <= n; i++) r *= i; return r; }

  /* Python's round(): halves go to the even neighbour, unlike Math.round
     which always rounds a half upwards. The crop lengths are derived with it,
     so getting this wrong silently shifts the analysis windows. */
  function pyRound(x) {
    var f = Math.floor(x), d = x - f;
    if (d > 0.5) return f + 1;
    if (d < 0.5) return f;
    return f % 2 === 0 ? f : f + 1;
  }

  /* ---------- signal preparation ---------- */

  /* log10 of readings; contact-loss readings are interpolated over but
     reported separately in `bad`. */
  function logSignal(raw) {
    var n = raw.length, y = new Float64Array(n), bad = new Uint8Array(n), nok = 0;
    for (var i = 0; i < n; i++) {
      var v = raw[i];
      var isBad = !isFinite(v) || v <= 0 || v >= OVERFLOW_MIN;
      bad[i] = isBad ? 1 : 0;
      if (!isBad) { y[i] = Math.log10(v); nok++; } else y[i] = NaN;
    }
    if (nok === 0) return { y: null, bad: bad };
    if (nok < n) {
      var idx = [], val = [];
      for (var j = 0; j < n; j++) if (!bad[j]) { idx.push(j); val.push(y[j]); }
      for (var k = 0; k < n; k++) {
        if (!bad[k]) continue;
        if (k <= idx[0]) { y[k] = val[0]; continue; }
        if (k >= idx[idx.length - 1]) { y[k] = val[val.length - 1]; continue; }
        var a = 0, b = idx.length - 1;
        while (b - a > 1) { var m = (a + b) >> 1; if (idx[m] <= k) a = m; else b = m; }
        var f = (k - idx[a]) / (idx[b] - idx[a]);
        y[k] = val[a] + f * (val[b] - val[a]);
      }
    }
    return { y: y, bad: bad };
  }

  /* moving standard deviation, unpadded (length n-w+1), as in features.py */
  function movingStd(y, w) {
    var n = y.length;
    if (n < w + 1) return new Float64Array([std(y)]);
    var c1 = new Float64Array(n + 1), c2 = new Float64Array(n + 1);
    for (var i = 0; i < n; i++) { c1[i + 1] = c1[i] + y[i]; c2[i + 1] = c2[i] + y[i] * y[i]; }
    var out = new Float64Array(n - w + 1);
    for (var j = 0; j + w <= n; j++) {
      var s1 = c1[j + w] - c1[j], s2 = c2[j + w] - c2[j];
      out[j] = Math.sqrt(Math.max(s2 / w - (s1 / w) * (s1 / w), 0));
    }
    return out;
  }

  /* padded moving std (length n), as in segment.py */
  function movingStdPad(y, w) {
    var n = y.length;
    w = Math.max(3, Math.min(w | 0, n));
    var c1 = new Float64Array(n + 1), c2 = new Float64Array(n + 1);
    for (var i = 0; i < n; i++) { c1[i + 1] = c1[i] + y[i]; c2[i + 1] = c2[i] + y[i] * y[i]; }
    var m = n - w + 1, s = new Float64Array(m);
    for (var j = 0; j < m; j++) {
      var s1 = c1[j + w] - c1[j], s2 = c2[j + w] - c2[j];
      s[j] = Math.sqrt(Math.max(s2 / w - (s1 / w) * (s1 / w), 0));
    }
    var pad = n - m, left = pad >> 1, o = new Float64Array(n);
    for (var a = 0; a < left; a++) o[a] = s[0];
    for (var b = 0; b < m; b++) o[left + b] = s[b];
    for (var c = left + m; c < n; c++) o[c] = s[m - 1];
    return o;
  }

  /* edge-padded moving average, matching the detrend in features.py */
  function movingAvgEdge(y, w) {
    var n = y.length, half = w >> 1, L = n + 2 * half;
    var pad = new Float64Array(L);
    for (var i = 0; i < L; i++) {
      var k = i - half;
      pad[i] = y[k < 0 ? 0 : (k >= n ? n - 1 : k)];
    }
    // summed window by window rather than with a running total: a running
    // total drifts over long flat stretches and that drift is enough to flip
    // tie ordering in the ordinal-pattern feature
    var out = new Float64Array(L - w + 1);
    for (var t = 0; t + w <= L; t++) {
      var s = 0;
      for (var j = 0; j < w; j++) s += pad[t + j];
      out[t] = s / w;
    }
    return out.subarray(0, n);
  }

  /* Ordinal (permutation) entropy. Values that differ only by rounding noise
     count as ties and are broken by position, so the result does not depend
     on the order terms happened to be summed in. */
  function permEntropy(y, m) {
    var n = y.length - m + 1;
    if (n < 10) return NaN;
    var scale = 0;
    for (var s0 = 0; s0 < y.length; s0++) { var av = Math.abs(y[s0]); if (av > scale) scale = av; }
    var eps = 1e-9 * (scale > 0 ? scale : 1);
    var counts = Object.create(null), order = new Array(m);
    for (var i = 0; i < n; i++) {
      for (var a = 0; a < m; a++) order[a] = a;
      (function (base) {
        order.sort(function (p, q) {
          var d = y[base + p] - y[base + q];
          if (Math.abs(d) <= eps) return p - q;
          return d < 0 ? -1 : 1;
        });
      })(i);
      var key = order.join(",");
      counts[key] = (counts[key] || 0) + 1;
    }
    var h = 0;
    for (var k in counts) { var p = counts[k] / n; h -= p * Math.log(p); }
    return h / Math.log(factorial(m));
  }

  /* ---------- the 16 model features ---------- */

  var FEATURES = ["jitter_p90", "jitter_med", "std_log", "range_dec", "spike_rate",
    "spec_slope", "band_frac_hi", "hf_ratio", "spec_entropy", "ac1_diff",
    "turn_rate", "perm_entropy", "wavelet_ratio", "ovf_frac", "ovf_run_max",
    "exc_height"];

  function maxRun(bad) {
    var best = 0, cur = 0;
    for (var i = 0; i < bad.length; i++) { cur = bad[i] ? cur + 1 : 0; if (cur > best) best = cur; }
    return best;
  }

  function excHeight(raw) {
    var ls = logSignal(raw);
    if (!ls.y || ls.y.length < 8) return NaN;
    var y = ls.y, nb = Math.max(4, Math.min(12, Math.floor(y.length / 4)));
    var b = median(y.subarray(0, nb));
    return percentile(y, 97) - b;
  }

  /* raw: a window of readings. fullRaw: the whole recording (for exc_height) */
  function extract(raw, fullRaw) {
    var f = {};
    FEATURES.forEach(function (k) { f[k] = NaN; });
    var nRaw = raw.length;
    var ls = logSignal(raw);
    var bad = ls.bad, nBad = 0;
    for (var i = 0; i < nRaw; i++) nBad += bad[i];
    f.ovf_frac = nRaw ? nBad / nRaw : NaN;
    f.ovf_run_max = nRaw ? maxRun(bad) / nRaw : NaN;
    f.exc_height = excHeight(fullRaw || raw);
    if (!ls.y || ls.y.length < 24) return f;

    var y = ls.y, n = y.length;
    f.std_log = std(y);
    f.range_dec = Math.max.apply(null, Array.from(y)) - Math.min.apply(null, Array.from(y));

    var d = diff(y), ad = absArr(d);
    f.jitter_med = median(ad);
    f.jitter_p90 = percentile(ad, 90);

    var medD = median(d);
    var devs = new Float64Array(d.length);
    for (var q = 0; q < d.length; q++) devs[q] = Math.abs(d[q] - medD);
    var madD = median(devs) * 1.4826;
    if (!(madD > 1e-12)) madD = 1e-12;
    var c3 = 0;
    for (var r = 0; r < ad.length; r++) if (ad[r] > 3 * madD) c3++;
    f.spike_rate = c3 / ad.length;

    var turns = 0;
    for (var t = 1; t < d.length; t++) if (Math.sign(d[t]) * Math.sign(d[t - 1]) < 0) turns++;
    f.turn_rate = turns / Math.max(1, d.length - 1);

    // detrend with an edge-padded moving average, window n/6 forced odd
    var w = Math.max(5, Math.min(31, Math.floor(n / 6)));
    if (w % 2 === 0) w += 1;
    var trend = movingAvgEdge(y, w);
    var res = new Float64Array(n);
    for (var u = 0; u < n; u++) res[u] = y[u] - trend[u];

    var dm = mean(d), den = 0, num = 0;
    for (var v = 0; v < d.length; v++) den += (d[v] - dm) * (d[v] - dm);
    for (var v2 = 1; v2 < d.length; v2++) num += (d[v2] - dm) * (d[v2 - 1] - dm);
    f.ac1_diff = den > 1e-15 ? num / den : NaN;

    // spectrum of the detrended signal
    var rm = mean(res), win = hanning(n), sig = new Float64Array(n);
    for (var s0 = 0; s0 < n; s0++) sig[s0] = (res[s0] - rm) * win[s0];
    var P = rfftPower(sig).subarray(1);
    var tot = 0;
    for (var p0 = 0; p0 < P.length; p0++) tot += P[p0];
    if (tot > 1e-20 && P.length > 8) {
      var k = P.length, hiS = 0, bandS = 0, ent = 0;
      for (var p1 = Math.floor(0.5 * k); p1 < k; p1++) hiS += P[p1] / tot;
      for (var p2 = Math.floor(0.75 * k); p2 < k; p2++) bandS += P[p2] / tot;
      for (var p3 = 0; p3 < k; p3++) {
        var pp = P[p3] / tot;
        if (pp > 1e-15) ent -= pp * Math.log(pp);
      }
      f.hf_ratio = hiS;
      f.band_frac_hi = bandS;
      f.spec_entropy = ent / Math.log(k);
      var lx = [], ly = [];
      for (var p4 = 0; p4 < k; p4++) if (P[p4] > 1e-20) { lx.push(Math.log(p4 + 1)); ly.push(Math.log(P[p4])); }
      if (lx.length > 8) f.spec_slope = polyfit1(lx, ly)[0];
    }

    f.perm_entropy = permEntropy(res, 4);
    var s3 = movingStd(res, 3), s9 = movingStd(res, 9);
    f.wavelet_ratio = median(s3) / (median(s9) + 1e-12);
    return f;
  }

  /* ---------- segmentation (onset mode) ---------- */

  function activeSpan(raw, minLen) {
    minLen = minLen || 40;
    var n = raw.length, ls = logSignal(raw);
    if (!ls.y || n < minLen) return [0, n];
    var y = ls.y, bad = ls.bad;
    var nb = Math.max(4, Math.min(12, Math.floor(y.length / 4)));
    var b = median(y.subarray(0, nb));
    var top = percentile(y, 97), amp = top - b;
    if (amp <= 0.1) return [0, n];
    var thr = b + 0.30 * amp, first = -1;
    for (var i = 0; i < n; i++) if (y[i] > thr || bad[i]) { first = i; break; }
    if (first < 0) return [0, n];
    return [Math.max(0, first - 2), n];
  }

  /* ---------- crops used at prediction time ---------- */

  var MIN_CROP = 48;

  function evalCrops(n) {
    if (n <= MIN_CROP) return [[0, n]];
    var out = [[0, n]];
    [0.85, 0.7, 0.6].forEach(function (frac) {
      var L = Math.max(MIN_CROP, pyRound(n * frac));
      if (L >= n) return;
      [0, Math.floor((n - L) / 2), n - L].forEach(function (s) { out.push([s, s + L]); });
    });
    var seen = {}, uniq = [];
    out.forEach(function (ab) {
      var key = ab[0] + ":" + ab[1];
      if (!seen[key]) { seen[key] = 1; uniq.push(ab); }
    });
    return uniq.slice(0, 9);
  }

  /* ---------- model ---------- */

  function score(model, fvec) {
    var z = model.intercept;
    var contrib = [];
    for (var i = 0; i < model.features.length; i++) {
      var v = fvec[i];
      if (!isFinite(v)) v = model.impute[i];
      var sv = (v - model.mean[i]) / model.scale[i];
      var c = sv * model.coef[i];
      z += c;
      contrib.push({ name: model.features[i], value: fvec[i], z: sv, contribution: c });
    }
    return { p: 1 / (1 + Math.exp(-z)), logit: z, contrib: contrib };
  }

  function analyse(raw, model) {
    var span = activeSpan(raw);
    var seg = raw.slice(span[0], span[1]);
    if (seg.length < 24) { seg = raw.slice(); span = [0, raw.length]; }
    var crops = evalCrops(seg.length);
    var ps = [], first = null, spans = [];
    crops.forEach(function (ab) {
      var f = extract(seg.slice(ab[0], ab[1]), raw);
      var vec = model.features.map(function (k) { return f[k]; });
      var s = score(model, vec);
      ps.push(s.p);
      spans.push([span[0] + ab[0], span[0] + ab[1]]);
      if (!first) first = { f: f, vec: vec, s: s };
    });
    var p = ps.reduce(function (a, b) { return a + b; }, 0) / ps.length;
    return {
      p: p, cropProbs: ps, cropSpans: spans, span: span,
      features: first.f, vector: first.vec, detail: first.s
    };
  }

  root.SurfaceEngine = {
    FEATURES: FEATURES, logSignal: logSignal, extract: extract,
    activeSpan: activeSpan, evalCrops: evalCrops, analyse: analyse,
    score: score, median: median, percentile: percentile, std: std,
    movingAvgEdge: movingAvgEdge, movingStdPad: movingStdPad, diff: diff,
    OVERFLOW_MIN: OVERFLOW_MIN
  };
})(typeof module !== "undefined" && module.exports ? module.exports : window);
