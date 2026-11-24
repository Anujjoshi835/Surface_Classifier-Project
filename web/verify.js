/* Cross-check: run the shipped JS engine on every recording and compare the
   features and the predicted probability against the Python implementation. */
const fs = require("fs");
const path = require("path");
const E = require("./engine.js").SurfaceEngine;

const ROOT = path.join(__dirname, "..");
const model = JSON.parse(fs.readFileSync(path.join(ROOT, "out/model.json"), "utf8"));
const ref = JSON.parse(fs.readFileSync(path.join(ROOT, "out/python_predictions.json"), "utf8"));

function parseCsv(text) {
  const out = [];
  for (const line of text.split(/\r?\n/)) {
    if (!line.trim()) continue;
    const c = line.split(",");
    const idx = Number(c[0]);
    if (!Number.isFinite(idx) || c.length < 2) continue;
    const v = Number(String(c[1]).trim());
    if (!Number.isFinite(v)) continue;
    out.push(v >= 1e30 || v <= 0 ? NaN : v);
  }
  return out;
}

let worstFeat = 0, worstP = 0, worstFeatName = "", worstFile = "", nFiles = 0;
const rows = [];
for (const fn of fs.readdirSync(path.join(ROOT, "data")).sort()) {
  if (!fn.endsWith(".csv")) continue;
  const name = fn.replace(/\.csv$/, "");
  if (!ref[name]) continue;
  const raw = parseCsv(fs.readFileSync(path.join(ROOT, "data", fn), "utf8"));
  const r = E.analyse(raw, model);
  const exp = ref[name];
  let fmax = 0, fname = "";
  model.features.forEach((k, i) => {
    const a = r.vector[i], b = exp.features[i];
    if (b === null || !Number.isFinite(a)) {
      if ((b === null) !== !Number.isFinite(a)) { fmax = Infinity; fname = k + " (nan mismatch)"; }
      return;
    }
    const d = Math.abs(a - b) / Math.max(1e-9, Math.abs(b));
    if (d > fmax) { fmax = d; fname = k; }
  });
  const dp = Math.abs(r.p - exp.p);
  rows.push([name, exp.p, r.p, dp, fmax, fname,
    exp.span[0] === r.span[0] && exp.span[1] === r.span[1]]);
  if (dp > worstP) { worstP = dp; worstFile = name; }
  if (fmax > worstFeat) { worstFeat = fmax; worstFeatName = fname; }
  nFiles++;
}

console.log("file          py P(rough)  js P(rough)     dP   worst feat rel err  span");
for (const [n, pe, pj, dp, fm, fn2, spanOk] of rows) {
  const flag = dp > 1e-6 || fm > 1e-6 ? "  <-- MISMATCH" : "";
  console.log(
    `${n.padEnd(12)} ${pe.toFixed(6).padStart(11)} ${pj.toFixed(6).padStart(12)} ` +
    `${dp.toExponential(1).padStart(8)} ${fm.toExponential(1).padStart(9)} ${fn2.padEnd(14)} ` +
    `${spanOk ? "ok" : "DIFF"}${flag}`);
}
console.log(`\n${nFiles} files checked`);
console.log(`largest probability difference : ${worstP.toExponential(2)}  (${worstFile})`);
console.log(`largest feature relative error : ${worstFeat.toExponential(2)}  (${worstFeatName})`);
process.exit(worstP < 1e-6 && worstFeat < 1e-6 ? 0 : 1);
