/* ML Interview Playbook — interactive demos. Mount with <div class="widget" data-demo="name"></div> */
(function () {
  const D = {};
  const rnd = (() => { let s = 7; return () => { s = (s * 16807) % 2147483647; return (s - 1) / 2147483646; }; })();
  const gauss = () => { let u = 0, v = 0; while (!u) u = rnd(); v = rnd(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
  const fmt = (x, d = 3) => (isFinite(x) ? Number(x).toFixed(d) : "∞");
  const erf = x => { const t = 1 / (1 + 0.3275911 * Math.abs(x)); const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x); return x >= 0 ? y : -y; };
  const Phi = z => 0.5 * (1 + erf(z / Math.SQRT2));
  const PhiInv = p => { // Acklam approximation
    const a = [-39.6968302866538, 220.946098424521, -275.928510446969, 138.357751867269, -30.6647980661472, 2.50662827745924];
    const b = [-54.4760987982241, 161.585836858041, -155.698979859887, 66.8013118877197, -13.2806815528857];
    const c = [-0.00778489400243029, -0.322396458041136, -2.40075827716184, -2.54973253934373, 4.37466414146497, 2.93816398269878];
    const d = [0.00778469570904146, 0.32246712907004, 2.445134137143, 3.75440866190742];
    const pl = 0.02425; let q, r;
    if (p < pl) { q = Math.sqrt(-2 * Math.log(p)); return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
    if (p > 1 - pl) { q = Math.sqrt(-2 * Math.log(1 - p)); return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
    q = p - 0.5; r = q * q; return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
  };

  function frame(c, xr, yr, opt = {}) {
    const { ctx, w, h, col } = c;
    const p = { l: 38, r: 12, t: 12, b: 26, ...(opt.pad || {}) };
    const X = x => p.l + (x - xr[0]) / (xr[1] - xr[0]) * (w - p.l - p.r);
    const Y = y => h - p.b - (y - yr[0]) / (yr[1] - yr[0]) * (h - p.t - p.b);
    const iX = px => xr[0] + (px - p.l) / (w - p.l - p.r) * (xr[1] - xr[0]);
    const iY = py => yr[0] + (h - p.b - py) / (h - p.t - p.b) * (yr[1] - yr[0]);
    ctx.clearRect(0, 0, w, h);
    ctx.strokeStyle = col("--line"); ctx.lineWidth = 1; ctx.fillStyle = col("--muted"); ctx.font = "11px ui-monospace,monospace";
    ctx.beginPath(); ctx.moveTo(p.l, p.t); ctx.lineTo(p.l, h - p.b); ctx.lineTo(w - p.r, h - p.b); ctx.stroke();
    if (opt.xl) { ctx.textAlign = "right"; ctx.fillText(opt.xl, w - p.r, h - 6); }
    if (opt.yl) { ctx.textAlign = "left"; ctx.fillText(opt.yl, p.l + 4, p.t + 10); }
    if (opt.ticks !== false) {
      ctx.textAlign = "right";
      [yr[0], (yr[0] + yr[1]) / 2, yr[1]].forEach(v => ctx.fillText(+v.toFixed(2), p.l - 4, Y(v) + 4));
      ctx.textAlign = "center";
      [xr[0], (xr[0] + xr[1]) / 2, xr[1]].forEach(v => ctx.fillText(+v.toFixed(2), X(v), h - p.b + 14));
    }
    return { X, Y, iX, iY, p };
  }
  function dot(ctx, x, y, r, fill, stroke) { ctx.beginPath(); ctx.arc(x, y, r, 0, 7); if (fill) { ctx.fillStyle = fill; ctx.fill(); } if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = 1.5; ctx.stroke(); } }
  function line(ctx, pts, color, wd = 2, dash) { ctx.beginPath(); ctx.setLineDash(dash || []); pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y))); ctx.strokeStyle = color; ctx.lineWidth = wd; ctx.stroke(); ctx.setLineDash([]); }
  function slider(id, label, min, max, step, val) { return `<div class="control"><label>${label} <b id="${id}v">${val}</b></label><input type="range" id="${id}" min="${min}" max="${max}" step="${step}" value="${val}"></div>`; }
  function seg(id, opts, on) { return `<div class="seg" id="${id}">${opts.map(o => `<button data-v="${o[0]}" class="${o[0] === on ? "on" : ""}">${o[1]}</button>`).join("")}</div>`; }
  function wireSeg(root, id, cb) { const s = root.querySelector("#" + id); s.addEventListener("click", e => { const b = e.target.closest("button"); if (!b) return; s.querySelectorAll("button").forEach(x => x.classList.remove("on")); b.classList.add("on"); cb(b.dataset.v); }); }
  function uid() { return "d" + Math.random().toString(36).slice(2, 8); }
  function solve(A, b) { // Gaussian elimination, A n×n
    const n = b.length; const M = A.map((r, i) => [...r, b[i]]);
    for (let i = 0; i < n; i++) {
      let mx = i; for (let k = i + 1; k < n; k++) if (Math.abs(M[k][i]) > Math.abs(M[mx][i])) mx = k;
      [M[i], M[mx]] = [M[mx], M[i]];
      for (let k = i + 1; k < n; k++) { const f = M[k][i] / M[i][i]; for (let j = i; j <= n; j++) M[k][j] -= f * M[i][j]; }
    }
    const x = Array(n).fill(0);
    for (let i = n - 1; i >= 0; i--) { let s = M[i][n]; for (let j = i + 1; j < n; j++) s -= M[i][j] * x[j]; x[i] = s / M[i][i]; }
    return x;
  }

  // 1. Gradient descent ------------------------------------------------------
  D.gd = root => {
    const u = uid();
    root.innerHTML += `<div class="controls">${slider(u + "lr", "learning rate η", 0.01, 1.1, 0.01, 0.1)}${slider(u + "w0", "start w₀", -4, 9, 0.5, -3)}${slider(u + "n", "iterations", 1, 50, 1, 15)}</div><canvas height="260"></canvas><div class="readout"></div>`;
    const cv = root.querySelector("canvas"), ro = root.querySelector(".readout");
    const f = w => (w - 3) ** 2 + 1, g = w => 2 * (w - 3);
    function draw() {
      const lr = +root.querySelector("#" + u + "lr").value, w0 = +root.querySelector("#" + u + "w0").value, n = +root.querySelector("#" + u + "n").value;
      ["lr", "w0", "n"].forEach(k => root.querySelector("#" + u + k + "v").textContent = root.querySelector("#" + u + k).value);
      const c = MLP_canvas(cv, 260); const F = frame(c, [-5, 11], [0, 70], { xl: "w", yl: "loss L(w) = (w−3)² + 1" });
      const pts = []; for (let x = -5; x <= 11; x += 0.1) pts.push([F.X(x), F.Y(Math.min(f(x), 70))]);
      line(c.ctx, pts, c.col("--muted"), 2);
      let w = w0; const path = [w];
      for (let i = 0; i < n; i++) { w = w - lr * g(w); path.push(w); if (Math.abs(w) > 1e6) break; }
      const vis = path.map(x => [F.X(Math.max(-5, Math.min(11, x))), F.Y(Math.min(f(x), 70))]);
      line(c.ctx, vis, c.col("--coral"), 1.5, [4, 3]);
      vis.forEach(([x, y], i) => dot(c.ctx, x, y, i === vis.length - 1 ? 6 : 3.5, i === vis.length - 1 ? c.col("--accent") : c.col("--coral")));
      const last = path[path.length - 1];
      const verdict = lr > 1 ? "💥 diverging: |1 − 2η| > 1, every step overshoots further" : lr === 1 ? "↔ oscillating forever between two points" : lr > 0.5 ? "〰 converging but zig-zagging across the minimum" : lr < 0.05 ? "🐢 converging very slowly: η too small" : "✅ smooth convergence";
      ro.innerHTML = `after ${path.length - 1} steps: w = ${fmt(last)}, loss = ${fmt(f(last))} (minimum 1.000 at w = 3)<br>${verdict}`;
    }
    root.querySelectorAll("input").forEach(i => i.addEventListener("input", draw)); window.addEventListener("resize", draw); window.addEventListener("mlp-theme", draw); draw();
  };

  // 2. Linear regression assumptions -----------------------------------------
  D.linfit = root => {
    const u = uid();
    root.innerHTML += `<div style="margin:6px 0 10px">${seg(u + "s", [["ok", "✅ Assumptions hold"], ["hetero", "Heteroscedastic"], ["nonlin", "Non-linear"], ["outlier", "Outlier / leverage"], ["auto", "Autocorrelated errors"]], "ok")} <button class="btn small" id="${u}re">↻ New sample</button></div>
      <div class="demo-grid"><div><div class="small">Data + OLS fit</div><canvas height="240"></canvas></div><div><div class="small" id="${u}rt">Residuals vs fitted</div><canvas height="240"></canvas></div></div><div class="readout"></div>`;
    const [c1, c2] = root.querySelectorAll("canvas"), ro = root.querySelector(".readout");
    let mode = "ok", data = [];
    const MSG = {
      ok: "Residuals form a structureless horizontal band around 0 with constant spread → linearity, homoscedasticity and independence look fine. Coefficients and their standard errors can be trusted.",
      hetero: "Funnel shape: residual spread grows with the fitted value → <b>heteroscedasticity</b>. Coefficients stay unbiased, but standard errors / p-values / CIs are wrong. Fixes: log or Box-Cox transform y, weighted least squares, or robust (HC3) standard errors. Test: Breusch–Pagan or White.",
      nonlin: "U-shaped residuals → the <b>linearity</b> assumption is violated, so predictions are biased in a systematic way. Fixes: add polynomial / spline terms or interactions, transform features, or use a non-linear model.",
      outlier: "One high-leverage point drags the whole line → OLS minimises <i>squared</i> error so it is very outlier-sensitive. Diagnose with Cook's distance / leverage; fix with investigation, robust regression (Huber), or capping.",
      auto: "Residuals plotted in time order drift in long runs → <b>autocorrelated errors</b> (independence violated, typical in time series). Standard errors are too small. Test: Durbin–Watson (≈2 is good). Fixes: add lag features, use time-series models or Newey–West SEs.",
    };
    function gen() {
      data = []; let e = 0;
      for (let i = 0; i < 70; i++) {
        const x = rnd() * 10; let y;
        if (mode === "hetero") y = 2 + 1.5 * x + gauss() * (0.2 + 0.55 * x);
        else if (mode === "nonlin") y = 1 + 0.35 * (x - 5) ** 2 + 0.6 * x + gauss() * 0.8;
        else if (mode === "auto") { e = 0.92 * e + gauss() * 0.9; const xx = i / 7; data.push([xx, 2 + 1.5 * xx + e * 1.4]); continue; }
        else y = 2 + 1.5 * x + gauss() * 1.4;
        data.push([x, y]);
      }
      if (mode === "outlier") data.push([9.8, -4]);
    }
    function draw() {
      const n = data.length, mx = data.reduce((s, d) => s + d[0], 0) / n, my = data.reduce((s, d) => s + d[1], 0) / n;
      let sxy = 0, sxx = 0; data.forEach(([x, y]) => { sxy += (x - mx) * (y - my); sxx += (x - mx) ** 2; });
      const b1 = sxy / sxx, b0 = my - b1 * mx;
      const res = data.map(([x, y]) => y - (b0 + b1 * x));
      const sst = data.reduce((s, d) => s + (d[1] - my) ** 2, 0), sse = res.reduce((s, r) => s + r * r, 0);
      const ys = data.map(d => d[1]); const lo = Math.min(...ys, 0) - 1, hi = Math.max(...ys) + 1;
      const k1 = MLP_canvas(c1, 240); const F = frame(k1, [0, 10], [lo, hi], { xl: "x", yl: "y" });
      data.forEach(([x, y]) => dot(k1.ctx, F.X(x), F.Y(y), 3.2, k1.col("--accent")));
      line(k1.ctx, [[F.X(0), F.Y(b0)], [F.X(10), F.Y(b0 + 10 * b1)]], k1.col("--coral"), 2.5);
      const k2 = MLP_canvas(c2, 240);
      const auto = mode === "auto";
      const xs = auto ? data.map((_, i) => i) : data.map(([x]) => b0 + b1 * x);
      const rm = Math.max(...res.map(Math.abs)) * 1.1;
      const G = frame(k2, [Math.min(...xs), Math.max(...xs)], [-rm, rm], { xl: auto ? "observation order (time)" : "fitted ŷ", yl: "residual" });
      line(k2.ctx, [[G.X(Math.min(...xs)), G.Y(0)], [G.X(Math.max(...xs)), G.Y(0)]], k2.col("--muted"), 1, [5, 4]);
      if (auto) line(k2.ctx, xs.map((x, i) => [G.X(x), G.Y(res[i])]), k2.col("--line"), 1);
      xs.forEach((x, i) => dot(k2.ctx, G.X(x), G.Y(res[i]), 3.2, k2.col("--coral")));
      let dw = 0; for (let i = 1; i < n; i++) dw += (res[i] - res[i - 1]) ** 2; dw /= sse;
      root.querySelector("#" + u + "rt").textContent = auto ? "Residuals vs time order" : "Residuals vs fitted";
      ro.innerHTML = `ŷ = ${fmt(b0, 2)} + ${fmt(b1, 2)}·x &nbsp; R² = ${fmt(1 - sse / sst)} &nbsp; Durbin–Watson = ${fmt(dw, 2)}<br><span style="font-family:var(--sans)">${MSG[mode]}</span>`;
    }
    wireSeg(root, u + "s", v => { mode = v; gen(); draw(); });
    root.querySelector("#" + u + "re").addEventListener("click", () => { gen(); draw(); });
    window.addEventListener("resize", draw); window.addEventListener("mlp-theme", draw); gen(); draw();
  };

  // 3. Bias–variance via polynomial degree -----------------------------------
  D.biasvar = root => {
    const u = uid();
    root.innerHTML += `<div class="controls">${slider(u + "d", "polynomial degree", 1, 15, 1, 3)}${slider(u + "n", "training points", 8, 80, 1, 18)}${slider(u + "l", "ridge λ (log₁₀)", -8, 1, 0.5, -8)}</div><div class="demo-grid"><canvas height="250"></canvas><canvas height="250"></canvas></div><div class="readout"></div>`;
    const [c1, c2] = root.querySelectorAll("canvas"), ro = root.querySelector(".readout");
    const truth = x => Math.sin(Math.PI * x);
    const test = []; for (let i = 0; i < 200; i++) { const x = -1 + 2 * rnd(); test.push([x, truth(x) + gauss() * 0.25]); }
    let pool = []; for (let i = 0; i < 80; i++) { const x = -1 + 2 * rnd(); pool.push([x, truth(x) + gauss() * 0.25]); }
    function fit(tr, d, lam) {
      const A = Array.from({ length: d + 1 }, () => Array(d + 1).fill(0)), b = Array(d + 1).fill(0);
      tr.forEach(([x, y]) => { const p = []; for (let j = 0; j <= d; j++) p.push(x ** j); for (let i = 0; i <= d; i++) { b[i] += p[i] * y; for (let j = 0; j <= d; j++) A[i][j] += p[i] * p[j]; } });
      for (let i = 1; i <= d; i++) A[i][i] += lam;
      A[0][0] += 1e-10;
      const w = solve(A, b); return x => w.reduce((s, c, j) => s + c * x ** j, 0);
    }
    const mse = (f, S) => S.reduce((s, [x, y]) => s + (f(x) - y) ** 2, 0) / S.length;
    function draw() {
      const d = +root.querySelector("#" + u + "d").value, n = +root.querySelector("#" + u + "n").value, ll = +root.querySelector("#" + u + "l").value;
      ["d", "n", "l"].forEach(k => root.querySelector("#" + u + k + "v").textContent = root.querySelector("#" + u + k).value);
      const tr = pool.slice(0, n), lam = 10 ** ll;
      const f = fit(tr, d, lam);
      const k1 = MLP_canvas(c1, 250); const F = frame(k1, [-1, 1], [-2, 2], { xl: "x", yl: "y" });
      test.forEach(([x, y]) => dot(k1.ctx, F.X(x), F.Y(y), 1.6, k1.col("--line")));
      const tp = [], fp = []; for (let x = -1; x <= 1.0001; x += 0.01) { tp.push([F.X(x), F.Y(truth(x))]); fp.push([F.X(x), F.Y(Math.max(-2, Math.min(2, f(x))))]); }
      line(k1.ctx, tp, k1.col("--green"), 1.5, [5, 4]); line(k1.ctx, fp, k1.col("--coral"), 2.5);
      tr.forEach(([x, y]) => dot(k1.ctx, F.X(x), F.Y(y), 3.5, k1.col("--accent")));
      const curve = []; for (let k = 1; k <= 15; k++) { const g = fit(tr, k, lam); curve.push([k, mse(g, tr), mse(g, test)]); }
      const ymax = Math.min(1.2, Math.max(...curve.map(c => c[2])) * 1.05);
      const k2 = MLP_canvas(c2, 250); const G = frame(k2, [1, 15], [0, ymax], { xl: "degree (complexity)", yl: "MSE" });
      line(k2.ctx, curve.map(c => [G.X(c[0]), G.Y(Math.min(c[1], ymax))]), k2.col("--accent"), 2);
      line(k2.ctx, curve.map(c => [G.X(c[0]), G.Y(Math.min(c[2], ymax))]), k2.col("--coral"), 2);
      line(k2.ctx, [[G.X(d), G.Y(0)], [G.X(d), G.Y(ymax)]], k2.col("--muted"), 1, [4, 4]);
      k2.ctx.fillStyle = k2.col("--accent"); k2.ctx.fillText("train", G.X(13), G.Y(curve[12][1]) - 6); k2.ctx.fillStyle = k2.col("--coral"); k2.ctx.fillText("test", G.X(13), G.Y(Math.min(curve[12][2], ymax)) - 6);
      const trE = mse(f, tr), teE = mse(f, test);
      const v = teE > trE * 2.5 && d > 4 ? "🔺 <b>High variance / overfitting</b>: train error tiny, test error large. Fix: lower degree, more data, or regularise (raise λ)." : d <= 2 ? "🔻 <b>High bias / underfitting</b>: both errors high, curve too rigid for sin(πx). Fix: more complexity / features." : "✅ <b>Good trade-off</b>: test error near the noise floor (σ² = 0.0625).";
      ro.innerHTML = `train MSE = ${fmt(trE)} · test MSE = ${fmt(teE)} · irreducible noise ≈ 0.063<br><span style="font-family:var(--sans)">${v}</span>`;
    }
    root.querySelectorAll("input").forEach(i => i.addEventListener("input", draw)); window.addEventListener("resize", draw); window.addEventListener("mlp-theme", draw); draw();
  };

  // 4. Threshold / confusion matrix / ROC ------------------------------------
  D.threshold = root => {
    const u = uid();
    root.innerHTML += `<div class="controls">${slider(u + "t", "decision threshold", 0.01, 0.99, 0.01, 0.5)}${slider(u + "s", "class separation (d′)", 0.2, 4, 0.1, 1.6)}${slider(u + "p", "positive rate (prevalence)", 0.01, 0.5, 0.01, 0.1)}</div>
      <div class="demo-grid"><canvas height="230"></canvas><canvas height="230"></canvas></div>
      <div style="display:flex;gap:22px;flex-wrap:wrap;align-items:flex-start;margin-top:10px"><div class="cmatrix" id="${u}cm"></div><div class="readout" style="flex:1;min-width:240px" id="${u}ro"></div></div>`;
    const [c1, c2] = root.querySelectorAll("canvas");
    function draw() {
      const t = +root.querySelector("#" + u + "t").value, dp = +root.querySelector("#" + u + "s").value, pr = +root.querySelector("#" + u + "p").value;
      ["t", "s", "p"].forEach(k => root.querySelector("#" + u + k + "v").textContent = root.querySelector("#" + u + k).value);
      // score = sigmoid(z); negatives z~N(-d/2,1), positives z~N(d/2,1)
      const logit = p => Math.log(p / (1 - p));
      const N = 10000, P = Math.round(N * pr), Ng = N - P;
      const zt = logit(t);
      const tpr = 1 - Phi(zt - dp / 2), fpr = 1 - Phi(zt + dp / 2);
      const TP = Math.round(P * tpr), FN = P - TP, FP = Math.round(Ng * fpr), TN = Ng - FP;
      const prec = TP / Math.max(1, TP + FP), rec = TP / Math.max(1, P), f1 = 2 * prec * rec / Math.max(1e-9, prec + rec), acc = (TP + TN) / N;
      const k1 = MLP_canvas(c1, 230); const F = frame(k1, [-5, 5], [0, 0.42], { xl: "model score (logit)", yl: "density", ticks: false });
      const pdf = (x, m) => Math.exp(-((x - m) ** 2) / 2) / Math.sqrt(2 * Math.PI);
      const npts = [], ppts = []; for (let x = -5; x <= 5; x += 0.05) { npts.push([F.X(x), F.Y(pdf(x, -dp / 2) * (1 - pr) / 0.5 * 0.5)]); ppts.push([F.X(x), F.Y(pdf(x, dp / 2) * pr / 0.5 * 0.5 + 0)]); }
      line(k1.ctx, npts, k1.col("--teal"), 2); line(k1.ctx, ppts, k1.col("--coral"), 2);
      line(k1.ctx, [[F.X(zt), F.Y(0)], [F.X(zt), F.Y(0.42)]], k1.col("--ink"), 1.5, [5, 4]);
      k1.ctx.fillStyle = k1.col("--teal"); k1.ctx.fillText("negatives", F.X(-4.6), F.Y(0.38)); k1.ctx.fillStyle = k1.col("--coral"); k1.ctx.fillText("positives", F.X(2.6), F.Y(0.38));
      const k2 = MLP_canvas(c2, 230); const G = frame(k2, [0, 1], [0, 1], { xl: "FPR", yl: "TPR (recall)" });
      const roc = []; for (let z = -8; z <= 8; z += 0.05) roc.push([G.X(1 - Phi(z + dp / 2)), G.Y(1 - Phi(z - dp / 2))]);
      line(k2.ctx, roc, k2.col("--accent"), 2.5); line(k2.ctx, [[G.X(0), G.Y(0)], [G.X(1), G.Y(1)]], k2.col("--line"), 1, [4, 4]);
      dot(k2.ctx, G.X(fpr), G.Y(tpr), 6, k2.col("--coral"));
      const auc = Phi(dp / Math.SQRT2);
      root.querySelector("#" + u + "cm").innerHTML = `<div class="h"></div><div class="h">pred +</div><div class="h">pred −</div><div class="h">actual +</div><div class="tp">TP<br><b>${TP}</b></div><div class="fn">FN<br><b>${FN}</b></div><div class="h">actual −</div><div class="fp">FP<br><b>${FP}</b></div><div class="tn">TN<br><b>${TN}</b></div>`;
      root.querySelector("#" + u + "ro").innerHTML = `accuracy ${fmt(acc)} · precision ${fmt(prec)} · recall ${fmt(rec)}<br>F1 ${fmt(f1)} · FPR ${fmt(fpr)} · specificity ${fmt(1 - fpr)}<br>ROC-AUC ${fmt(auc)} (threshold-free)<br><span style="font-family:var(--sans);font-size:13.5px">${pr < 0.06 && acc > 0.9 ? "⚠️ Rare positives: accuracy looks great even when recall is poor — use precision/recall, PR-AUC." : "Raise the threshold → precision ↑ recall ↓. Lower it → recall ↑, more false alarms."}</span>`;
    }
    root.querySelectorAll("input").forEach(i => i.addEventListener("input", draw)); window.addEventListener("resize", draw); window.addEventListener("mlp-theme", draw); draw();
  };

  // 5. kNN decision boundary --------------------------------------------------
  D.knn = root => {
    const u = uid();
    root.innerHTML += `<div class="controls">${slider(u + "k", "k (neighbours)", 1, 31, 2, 1)}<div class="control"><label>weights</label>${seg(u + "w", [["uniform", "uniform"], ["distance", "distance"]], "uniform")}</div></div><canvas height="300"></canvas><div class="readout">Click to add a blue point, Shift+click for orange. Small k → jagged islands (high variance). Large k → smooth, eventually just the majority class (high bias).</div>`;
    const cv = root.querySelector("canvas"); let W = "uniform";
    const pts = []; for (let i = 0; i < 70; i++) { const c = i % 2; const a = rnd() * Math.PI; const r = 1 + gauss() * 0.18; pts.push(c ? [1 - Math.cos(a) * r, 0.5 - Math.sin(a) * r + 0.5, 1] : [Math.cos(a) * r, Math.sin(a) * r - 0.2, 0]); }
    for (let i = 0; i < 6; i++) pts.push([rnd() * 3 - 1, rnd() * 2 - 0.8, rnd() < 0.5 ? 1 : 0]);
    function draw() {
      const k = +root.querySelector("#" + u + "k").value; root.querySelector("#" + u + "kv").textContent = k;
      const c = MLP_canvas(cv, 300); const F = frame(c, [-1.6, 2.6], [-1.1, 1.6], { ticks: false });
      const cA = c.col("--accent"), cB = c.col("--amber"); const step = 7;
      for (let px = F.p.l; px < c.w - F.p.r; px += step) for (let py = F.p.t; py < c.h - F.p.b; py += step) {
        const x = F.iX(px + step / 2), y = F.iY(py + step / 2);
        const ds = pts.map(p => [Math.hypot(p[0] - x, p[1] - y), p[2]]).sort((a, b) => a[0] - b[0]).slice(0, k);
        let v = 0, tot = 0; ds.forEach(([d, l]) => { const wgt = W === "distance" ? 1 / (d + 1e-6) : 1; v += wgt * l; tot += wgt; });
        c.ctx.globalAlpha = 0.18; c.ctx.fillStyle = v / tot > 0.5 ? cB : cA; c.ctx.fillRect(px, py, step, step); c.ctx.globalAlpha = 1;
      }
      pts.forEach(p => dot(c.ctx, F.X(p[0]), F.Y(p[1]), 4, p[2] ? cB : cA, c.col("--panel")));
      cv._F = F;
    }
    cv.addEventListener("click", e => { const r = cv.getBoundingClientRect(); const F = cv._F; pts.push([F.iX(e.clientX - r.left), F.iY(e.clientY - r.top), e.shiftKey ? 1 : 0]); draw(); });
    wireSeg(root, u + "w", v => { W = v; draw(); });
    root.querySelector("input").addEventListener("input", draw); window.addEventListener("resize", draw); window.addEventListener("mlp-theme", draw); draw();
  };

  // 6. K-means step by step ----------------------------------------------------
  D.kmeans = root => {
    const u = uid();
    root.innerHTML += `<div class="controls">${slider(u + "k", "K", 2, 7, 1, 3)}<div class="control"><label>init</label>${seg(u + "i", [["pp", "k-means++"], ["rand", "random"]], "pp")}</div></div>
      <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:8px"><button class="btn small" data-a="init">1 · Initialise</button><button class="btn small" data-a="assign">2 · Assign points</button><button class="btn small" data-a="update">3 · Move centroids</button><button class="btn small primary" data-a="run">▶ Run to convergence</button><button class="btn small" data-a="data">New data</button></div><canvas height="300"></canvas><div class="readout"></div>`;
    const cv = root.querySelector("canvas"), ro = root.querySelector(".readout");
    let pts = [], C = [], lab = [], init = "pp", it = 0;
    const COL = ["--accent", "--coral", "--green", "--amber", "--teal", "#a855f7", "#64748b"];
    function data() { pts = []; const centers = [[-2, 1.5], [2, 2], [0, -1.8], [2.6, -1], [-2.4, -1.2]]; centers.forEach(([cx, cy]) => { for (let i = 0; i < 40; i++) pts.push([cx + gauss() * 0.65, cy + gauss() * 0.65]); }); lab = pts.map(() => -1); C = []; it = 0; }
    const d2 = (a, b) => (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2;
    function doInit() {
      const k = +root.querySelector("#" + u + "k").value; it = 0; lab = pts.map(() => -1);
      if (init === "rand") { C = []; while (C.length < k) C.push([...pts[Math.floor(rnd() * pts.length)]]); }
      else { C = [[...pts[Math.floor(rnd() * pts.length)]]]; while (C.length < k) { const D2 = pts.map(p => Math.min(...C.map(c => d2(p, c)))); let r = rnd() * D2.reduce((a, b) => a + b, 0); let i = 0; while ((r -= D2[i]) > 0) i++; C.push([...pts[i]]); } }
    }
    function assign() { let ch = 0; lab = pts.map((p, j) => { let b = 0; C.forEach((c, i) => { if (d2(p, c) < d2(p, C[b])) b = i; }); if (b !== lab[j]) ch++; return b; }); return ch; }
    function update() { C = C.map((c, i) => { const m = pts.filter((_, j) => lab[j] === i); return m.length ? [m.reduce((s, p) => s + p[0], 0) / m.length, m.reduce((s, p) => s + p[1], 0) / m.length] : c; }); it++; }
    function inertia() { return pts.reduce((s, p, j) => s + (lab[j] >= 0 ? d2(p, C[lab[j]]) : 0), 0); }
    function draw(msg) {
      root.querySelector("#" + u + "kv").textContent = root.querySelector("#" + u + "k").value;
      const c = MLP_canvas(cv, 300); const F = frame(c, [-4.5, 4.8], [-3.8, 3.8], { ticks: false });
      const col = i => (COL[i].startsWith("--") ? c.col(COL[i]) : COL[i]);
      pts.forEach((p, j) => dot(c.ctx, F.X(p[0]), F.Y(p[1]), 3.4, lab[j] >= 0 ? col(lab[j]) : c.col("--muted")));
      C.forEach((p, i) => { c.ctx.save(); c.ctx.translate(F.X(p[0]), F.Y(p[1])); c.ctx.rotate(Math.PI / 4); c.ctx.fillStyle = col(i); c.ctx.strokeStyle = c.col("--ink"); c.ctx.lineWidth = 2; c.ctx.fillRect(-7, -7, 14, 14); c.ctx.strokeRect(-7, -7, 14, 14); c.ctx.restore(); });
      ro.innerHTML = (msg || "") + (C.length ? `<br>iteration ${it} · inertia (WCSS) = ${lab[0] >= 0 ? fmt(inertia(), 1) : "—"}` : "Press Initialise.");
    }
    root.addEventListener("click", e => {
      const a = e.target.dataset && e.target.dataset.a; if (!a) return;
      if (a === "data") { data(); draw("New data with 5 true blobs."); }
      if (a === "init") { doInit(); draw(init === "pp" ? "k-means++: first centroid random, each next one sampled ∝ squared distance to the nearest chosen centroid → spread-out starts." : "Random init: K random points — can start two centroids in one blob → poor local minimum."); }
      if (a === "assign") { if (!C.length) doInit(); const ch = assign(); draw(`Assignment step: each point joins its nearest centroid (${ch} points changed).`); }
      if (a === "update") { if (lab[0] < 0) assign(); update(); draw("Update step: each centroid moves to the mean of its points. Inertia can only go down."); }
      if (a === "run") { if (!C.length) doInit(); let n = 0; while (n++ < 100) { const ch = assign(); update(); if (!ch && n > 1) break; } assign(); draw(`Converged after ${it} iterations (no point changed cluster). Try random init a few times to see different local minima.`); }
    });
    wireSeg(root, u + "i", v => { init = v; });
    root.querySelector("input").addEventListener("input", () => { C = []; lab = pts.map(() => -1); draw(); });
    window.addEventListener("resize", () => draw()); window.addEventListener("mlp-theme", () => draw()); data(); draw();
  };

  // 7. Gini / entropy split calculator -------------------------------------------
  D.gini = root => {
    const u = uid();
    root.innerHTML += `<div class="controls">${slider(u + "a", "left node: class 1", 0, 50, 1, 30)}${slider(u + "b", "left node: class 0", 0, 50, 1, 5)}${slider(u + "c", "right node: class 1", 0, 50, 1, 10)}${slider(u + "d", "right node: class 0", 0, 50, 1, 35)}</div><div class="readout"></div>`;
    const ro = root.querySelector(".readout");
    const gini = (p, n) => { const t = p + n; return t ? 1 - (p / t) ** 2 - (n / t) ** 2 : 0; };
    const ent = (p, n) => { const t = p + n; let e = 0; [p, n].forEach(x => { if (x) e -= (x / t) * Math.log2(x / t); }); return e; };
    function draw() {
      const [a, b, c, d] = ["a", "b", "c", "d"].map(k => { const v = +root.querySelector("#" + u + k).value; root.querySelector("#" + u + k + "v").textContent = v; return v; });
      const nl = a + b, nr = c + d, n = nl + nr || 1;
      const gp = gini(a + c, b + d), gl = gini(a, b), gr = gini(c, d), gw = (nl * gl + nr * gr) / n;
      const ep = ent(a + c, b + d), el = ent(a, b), er = ent(c, d), ew = (nl * el + nr * er) / n;
      const bar = v => `<span style="display:inline-block;width:${Math.round(v * 120)}px;height:8px;background:var(--accent);border-radius:4px;vertical-align:middle"></span>`;
      ro.innerHTML = `parent: [${a + c} | ${b + d}] → Gini ${fmt(gp)} ${bar(gp)} · entropy ${fmt(ep)}<br>left: [${a} | ${b}] → Gini ${fmt(gl)} · entropy ${fmt(el)}<br>right: [${c} | ${d}] → Gini ${fmt(gr)} · entropy ${fmt(er)}<br>weighted children: Gini ${fmt(gw)} · entropy ${fmt(ew)}<br><b>Gini decrease = ${fmt(gp - gw)} · Information gain = ${fmt(ep - ew)}</b><br><span style="font-family:var(--sans);font-size:13.5px">The tree tries every feature/threshold and keeps the split with the largest decrease. Pure nodes have Gini 0; a 50/50 node has Gini 0.5 and entropy 1.</span>`;
    }
    root.querySelectorAll("input").forEach(i => i.addEventListener("input", draw)); draw();
  };

  // 8. Ridge vs Lasso coefficient paths -----------------------------------------
  D.regpath = root => {
    const u = uid();
    root.innerHTML += `<div style="margin:6px 0">${seg(u + "m", [["lasso", "Lasso (L1)"], ["ridge", "Ridge (L2)"], ["enet", "Elastic Net (50/50)"]], "lasso")}</div><div class="controls">${slider(u + "a", "α (log₁₀)", -3, 1.5, 0.05, -1)}</div><canvas height="270"></canvas><div class="readout"></div>`;
    const cv = root.querySelector("canvas"), ro = root.querySelector(".readout");
    // synthetic standardized data: 6 features, x2 ≈ x1 (correlated), x5,x6 irrelevant
    const n = 120, X = [], y = [], TRUE = [3, 0, -2, 1.2, 0, 0];
    for (let i = 0; i < n; i++) { const z = [gauss(), 0, gauss(), gauss(), gauss(), gauss()]; z[1] = 0.9 * z[0] + 0.44 * gauss(); X.push(z); y.push(z.reduce((s, v, j) => s + v * TRUE[j], 0) + gauss() * 1.2); }
    const p = 6; const mu = Array(p).fill(0).map((_, j) => X.reduce((s, r) => s + r[j], 0) / n);
    const sd = Array(p).fill(0).map((_, j) => Math.sqrt(X.reduce((s, r) => s + (r[j] - mu[j]) ** 2, 0) / n));
    const Xs = X.map(r => r.map((v, j) => (v - mu[j]) / sd[j])); const ym = y.reduce((a, b) => a + b, 0) / n; const yc = y.map(v => v - ym);
    function cd(alpha, l1r) { // sklearn objective: 1/(2n)||y-Xw||² + α·l1r·|w|₁ + α(1-l1r)/2·||w||²
      const w = Array(p).fill(0);
      for (let it = 0; it < 300; it++) for (let j = 0; j < p; j++) {
        let rho = 0; for (let i = 0; i < n; i++) { let r = yc[i]; for (let k = 0; k < p; k++) if (k !== j) r -= Xs[i][k] * w[k]; rho += Xs[i][j] * r; }
        rho /= n; const zj = 1 + alpha * (1 - l1r); const t = alpha * l1r;
        w[j] = Math.sign(rho) * Math.max(Math.abs(rho) - t, 0) / zj;
      }
      return w;
    }
    let mode = "lasso"; const cache = {};
    function path(m) { if (cache[m]) return cache[m]; const l1r = m === "lasso" ? 1 : m === "ridge" ? 0 : 0.5; const out = []; for (let la = -3; la <= 1.5001; la += 0.1) out.push([la, cd(10 ** la, l1r)]); return (cache[m] = out); }
    const COLS = ["--accent", "--coral", "--green", "--amber", "--teal", "--muted"];
    function draw() {
      const la = +root.querySelector("#" + u + "a").value; root.querySelector("#" + u + "av").textContent = (10 ** la).toPrecision(2);
      const P = path(mode); const c = MLP_canvas(cv, 270); const F = frame(c, [-3, 1.5], [-2.5, 3.6], { xl: "log₁₀ α  (more regularisation →)", yl: "standardised coefficient" });
      line(c.ctx, [[F.X(-3), F.Y(0)], [F.X(1.5), F.Y(0)]], c.col("--line"), 1);
      for (let j = 0; j < p; j++) line(c.ctx, P.map(([a, w]) => [F.X(a), F.Y(w[j])]), c.col(COLS[j]), 2);
      line(c.ctx, [[F.X(la), F.Y(-2.5)], [F.X(la), F.Y(3.6)]], c.col("--ink"), 1, [4, 4]);
      const w = cd(10 ** la, mode === "lasso" ? 1 : mode === "ridge" ? 0 : 0.5);
      ro.innerHTML = w.map((v, j) => `<span style="color:var(${COLS[j]})">β${j + 1}=${fmt(v, 2)}</span>`).join(" · ") + ` &nbsp; non-zero: ${w.filter(v => Math.abs(v) > 1e-6).length}/6<br><span style="font-family:var(--sans);font-size:13.5px">True β = [3, 0, −2, 1.2, 0, 0]; x₂ is a near-copy of x₁. ${mode === "lasso" ? "Lasso drives coefficients exactly to 0 (feature selection) and tends to keep only one of the correlated pair." : mode === "ridge" ? "Ridge shrinks every coefficient smoothly toward 0 but never exactly to 0, and shares weight between correlated x₁/x₂." : "Elastic Net mixes both: sparse like Lasso, but keeps correlated features together like Ridge."}</span>`;
    }
    wireSeg(root, u + "m", v => { mode = v; draw(); });
    root.querySelector("input").addEventListener("input", draw); window.addEventListener("resize", draw); window.addEventListener("mlp-theme", draw); draw();
  };

  // 9. Logistic sigmoid & log-loss ----------------------------------------------
  D.sigmoid = root => {
    const u = uid();
    root.innerHTML += `<div class="controls">${slider(u + "w", "weight w", -4, 6, 0.1, 1.5)}${slider(u + "b", "bias b", -8, 8, 0.1, -3)}</div><canvas height="250"></canvas><div class="readout"></div>`;
    const cv = root.querySelector("canvas"), ro = root.querySelector(".readout");
    const D0 = [0.5, 1, 1.3, 1.8, 2.2, 2.5, 3.1, 3.4, 4.6], D1 = [2.8, 3.6, 4.1, 4.4, 5, 5.5, 6.1, 6.8, 7.5];
    const s = z => 1 / (1 + Math.exp(-z));
    function draw() {
      const w = +root.querySelector("#" + u + "w").value, b = +root.querySelector("#" + u + "b").value;
      root.querySelector("#" + u + "wv").textContent = w; root.querySelector("#" + u + "bv").textContent = b;
      const c = MLP_canvas(cv, 250); const F = frame(c, [0, 8], [-0.05, 1.05], { xl: "x (e.g. hours studied)", yl: "P(y = 1 | x)" });
      const pts = []; for (let x = 0; x <= 8; x += 0.05) pts.push([F.X(x), F.Y(s(w * x + b))]);
      line(c.ctx, [[F.X(0), F.Y(0.5)], [F.X(8), F.Y(0.5)]], c.col("--line"), 1, [4, 4]);
      line(c.ctx, pts, c.col("--accent"), 2.5);
      let ll = 0, ok = 0;
      D0.forEach(x => { const p = s(w * x + b); ll -= Math.log(Math.max(1e-12, 1 - p)); if (p < 0.5) ok++; dot(c.ctx, F.X(x), F.Y(0), 5, c.col("--teal")); line(c.ctx, [[F.X(x), F.Y(0)], [F.X(x), F.Y(p)]], c.col("--teal"), 1, [2, 3]); });
      D1.forEach(x => { const p = s(w * x + b); ll -= Math.log(Math.max(1e-12, p)); if (p >= 0.5) ok++; dot(c.ctx, F.X(x), F.Y(1), 5, c.col("--coral")); line(c.ctx, [[F.X(x), F.Y(1)], [F.X(x), F.Y(p)]], c.col("--coral"), 1, [2, 3]); });
      const nn = D0.length + D1.length;
      ro.innerHTML = `decision boundary x = −b/w = ${w ? fmt(-b / w, 2) : "none"} · mean log-loss = ${fmt(ll / nn)} · accuracy = ${ok}/${nn}<br><span style="font-family:var(--sans);font-size:13.5px">Dotted lines show each point's error. Log-loss punishes confident mistakes hard (−log p → ∞ as p → 0). A one-unit increase in x multiplies the odds by e<sup>w</sup> = ${fmt(Math.exp(w), 2)}.</span>`;
    }
    root.querySelectorAll("input").forEach(i => i.addEventListener("input", draw)); window.addEventListener("resize", draw); window.addEventListener("mlp-theme", draw); draw();
  };

  // 10. Gradient boosting step by step -------------------------------------------
  D.boost = root => {
    const u = uid();
    root.innerHTML += `<div class="controls">${slider(u + "lr", "learning rate (shrinkage)", 0.05, 1, 0.05, 0.3)}${slider(u + "d", "tree depth", 1, 3, 1, 1)}</div><div style="display:flex;gap:8px;margin-bottom:8px"><button class="btn small primary" data-a="add">+ Add 1 tree</button><button class="btn small" data-a="add10">+ Add 10 trees</button><button class="btn small" data-a="reset">Reset</button></div><div class="demo-grid"><div><div class="small">Ensemble prediction F(x)</div><canvas height="240"></canvas></div><div><div class="small">Current residuals (what the next tree fits)</div><canvas height="240"></canvas></div></div><div class="readout"></div>`;
    const [c1, c2] = root.querySelectorAll("canvas"), ro = root.querySelector(".readout");
    const n = 60, X = [], Y = []; for (let i = 0; i < n; i++) { const x = i / (n - 1) * 6; X.push(x); Y.push(Math.sin(x) * 2 + 0.3 * x + gauss() * 0.35); }
    let trees = [], F0 = Y.reduce((a, b) => a + b, 0) / n;
    function fitTree(idx, r, depth) {
      if (depth === 0 || idx.length < 4) { const m = idx.reduce((s, i) => s + r[i], 0) / idx.length; return { leaf: m }; }
      let best = null;
      for (let k = 2; k < idx.length - 1; k++) {
        const L = idx.slice(0, k), R = idx.slice(k);
        const ml = L.reduce((s, i) => s + r[i], 0) / L.length, mr = R.reduce((s, i) => s + r[i], 0) / R.length;
        const sse = L.reduce((s, i) => s + (r[i] - ml) ** 2, 0) + R.reduce((s, i) => s + (r[i] - mr) ** 2, 0);
        if (!best || sse < best.sse) best = { sse, k, t: (X[idx[k - 1]] + X[idx[k]]) / 2 };
      }
      return { t: best.t, l: fitTree(idx.slice(0, best.k), r, depth - 1), r: fitTree(idx.slice(best.k), r, depth - 1) };
    }
    const predT = (tr, x) => (tr.leaf !== undefined ? tr.leaf : x < tr.t ? predT(tr.l, x) : predT(tr.r, x));
    const pred = x => F0 + trees.reduce((s, [tr, lr]) => s + lr * predT(tr, x), 0);
    function add(k) { const lr = +root.querySelector("#" + u + "lr").value, d = +root.querySelector("#" + u + "d").value; for (let j = 0; j < k; j++) { const r = Y.map((y, i) => y - pred(X[i])); trees.push([fitTree([...Array(n).keys()], r, d), lr]); } draw(); }
    function draw() {
      ["lr", "d"].forEach(k => root.querySelector("#" + u + k + "v").textContent = root.querySelector("#" + u + k).value);
      const k1 = MLP_canvas(c1, 240); const F = frame(k1, [0, 6], [-3, 4.5], { xl: "x", yl: "y" });
      X.forEach((x, i) => dot(k1.ctx, F.X(x), F.Y(Y[i]), 3, k1.col("--accent")));
      const pl = []; for (let x = 0; x <= 6; x += 0.02) pl.push([F.X(x), F.Y(pred(x))]); line(k1.ctx, pl, k1.col("--coral"), 2.5);
      const r = Y.map((y, i) => y - pred(X[i]));
      const k2 = MLP_canvas(c2, 240); const G = frame(k2, [0, 6], [-3, 3], { xl: "x", yl: "residual y − F(x)" });
      line(k2.ctx, [[G.X(0), G.Y(0)], [G.X(6), G.Y(0)]], k2.col("--line"), 1, [4, 4]);
      X.forEach((x, i) => dot(k2.ctx, G.X(x), G.Y(Math.max(-3, Math.min(3, r[i]))), 3, k2.col("--coral")));
      const mse = r.reduce((s, v) => s + v * v, 0) / n;
      ro.innerHTML = `trees: ${trees.length} · training MSE: ${fmt(mse)}<br><span style="font-family:var(--sans);font-size:13.5px">F₀ = mean(y). Each new tree is fit to the residuals (= negative gradient of squared loss) and added with weight η. Small η needs more trees but generalises better; with η = 1 and many deep trees the curve starts chasing noise.</span>`;
    }
    root.addEventListener("click", e => { const a = e.target.dataset && e.target.dataset.a; if (a === "add") add(1); if (a === "add10") add(10); if (a === "reset") { trees = []; draw(); } });
    root.querySelectorAll("input").forEach(i => i.addEventListener("input", draw)); window.addEventListener("resize", draw); window.addEventListener("mlp-theme", draw); draw();
  };

  // 11. PCA ---------------------------------------------------------------------
  D.pca = root => {
    const u = uid();
    root.innerHTML += `<div class="controls">${slider(u + "r", "correlation between x₁ and x₂", -0.98, 0.98, 0.01, 0.8)}${slider(u + "s", "x₂ scale (try without standardising!)", 0.2, 5, 0.1, 1)}</div><label class="small"><input type="checkbox" id="${u}p"> project onto PC1 (1-D compression)</label><canvas height="300"></canvas><div class="readout"></div>`;
    const cv = root.querySelector("canvas"), ro = root.querySelector(".readout");
    const Z = []; for (let i = 0; i < 180; i++) Z.push([gauss(), gauss()]);
    function draw() {
      const r = +root.querySelector("#" + u + "r").value, sc = +root.querySelector("#" + u + "s").value, proj = root.querySelector("#" + u + "p").checked;
      root.querySelector("#" + u + "rv").textContent = r; root.querySelector("#" + u + "sv").textContent = sc;
      const P = Z.map(([a, b]) => [a, (r * a + Math.sqrt(1 - r * r) * b) * sc]);
      const m = [P.reduce((s, p) => s + p[0], 0) / P.length, P.reduce((s, p) => s + p[1], 0) / P.length];
      let sxx = 0, syy = 0, sxy = 0; P.forEach(([x, y]) => { sxx += (x - m[0]) ** 2; syy += (y - m[1]) ** 2; sxy += (x - m[0]) * (y - m[1]); });
      sxx /= P.length - 1; syy /= P.length - 1; sxy /= P.length - 1;
      const tr = sxx + syy, det = sxx * syy - sxy * sxy, l1 = tr / 2 + Math.sqrt(tr * tr / 4 - det), l2 = tr / 2 - Math.sqrt(tr * tr / 4 - det);
      let v1 = Math.abs(sxy) > 1e-9 ? [l1 - syy, sxy] : sxx >= syy ? [1, 0] : [0, 1]; const nv = Math.hypot(...v1); v1 = [v1[0] / nv, v1[1] / nv]; const v2 = [-v1[1], v1[0]];
      const R = Math.max(4, 3.2 * sc);
      const c = MLP_canvas(cv, 300); const F = frame(c, [-R * 1.3, R * 1.3], [-R, R], { xl: "x₁", yl: "x₂", ticks: false });
      P.forEach(p => { const t = (p[0] - m[0]) * v1[0] + (p[1] - m[1]) * v1[1]; const q = proj ? [m[0] + t * v1[0], m[1] + t * v1[1]] : p; if (proj) line(c.ctx, [[F.X(p[0]), F.Y(p[1])], [F.X(q[0]), F.Y(q[1])]], c.col("--line"), 1); dot(c.ctx, F.X(q[0]), F.Y(q[1]), 2.8, c.col("--accent")); });
      const arrow = (v, l, colr, lab) => { const e = [m[0] + v[0] * 2 * Math.sqrt(l), m[1] + v[1] * 2 * Math.sqrt(l)]; line(c.ctx, [[F.X(m[0]), F.Y(m[1])], [F.X(e[0]), F.Y(e[1])]], colr, 3.5); c.ctx.fillStyle = colr; c.ctx.font = "bold 12px sans-serif"; c.ctx.fillText(lab, F.X(e[0]) + 6, F.Y(e[1])); };
      arrow(v1, l1, c.col("--coral"), "PC1"); arrow(v2, l2, c.col("--green"), "PC2");
      ro.innerHTML = `eigenvalues λ₁ = ${fmt(l1, 2)}, λ₂ = ${fmt(l2, 2)} · explained variance: PC1 ${fmt(100 * l1 / tr, 1)}%, PC2 ${fmt(100 * l2 / tr, 1)}%<br>PC1 direction = [${fmt(v1[0], 2)}, ${fmt(v1[1], 2)}] (the loadings)<br><span style="font-family:var(--sans);font-size:13.5px">PCA finds the orthogonal directions of maximum variance (eigenvectors of the covariance matrix). Strong correlation → PC1 alone keeps most information. Inflate x₂'s scale and PC1 swings toward x₂ even with zero correlation — that's why you standardise first.</span>`;
    }
    root.querySelectorAll("input").forEach(i => i.addEventListener("input", draw)); window.addEventListener("resize", draw); window.addEventListener("mlp-theme", draw); draw();
  };

  // 12. Cross-validation split schemes (HTML blocks) -----------------------------
  D.cvsplit = root => {
    const u = uid();
    root.innerHTML += `<div style="margin:6px 0 12px">${seg(u + "m", [["kfold", "K-Fold"], ["strat", "Stratified K-Fold"], ["group", "Group K-Fold"], ["ts", "TimeSeriesSplit"], ["loo", "Leave-One-Out"], ["nested", "Nested CV"]], "kfold")}</div><div id="${u}v"></div><div class="readout" id="${u}t" style="font-family:var(--sans)"></div>`;
    const N = 20, cls = Array.from({ length: N }, (_, i) => (i % 5 === 0 ? 1 : 0)), grp = Array.from({ length: N }, (_, i) => Math.floor(i / 4));
    const cell = (bg, txt, bd) => `<div style="flex:1;height:26px;border-radius:4px;background:${bg};display:grid;place-items:center;font:10px ui-monospace,monospace;color:#fff;${bd || ""}">${txt || ""}</div>`;
    const row = (lab, cells) => `<div style="display:flex;gap:3px;align-items:center;margin:4px 0"><div style="width:74px;font-size:12px;color:var(--muted)">${lab}</div>${cells}</div>`;
    const TXT = {
      kfold: "Split into K equal folds; each fold is the validation set once. Final score = mean ± std of K scores. Shuffle first unless the data are ordered in time.",
      strat: "Like K-Fold, but every fold keeps the class ratio (here 1 positive in 5). Default for classification, essential with imbalance — otherwise some folds may contain almost no positives.",
      group: "Rows from the same group (patient, customer, store — colours/labels G0–G4) never appear in both train and validation. Prevents leakage when one entity has many rows.",
      ts: "Expanding window: always train on the past and validate on the future. Never shuffle time-series. Add a gap between train and test if features use lookback windows.",
      loo: "K = n: each single row is the validation set once. Nearly unbiased but very high variance and n model fits — only for tiny datasets.",
      nested: "Outer loop estimates generalisation; an inner CV inside each outer training fold does hyperparameter tuning. Gives an unbiased estimate of the whole “tune + fit” procedure.",
    };
    function draw(m) {
      const C = getComputedStyle(document.documentElement); const tr = C.getPropertyValue("--accent"), va = C.getPropertyValue("--coral"), off = C.getPropertyValue("--line"), te = C.getPropertyValue("--amber");
      let h = "";
      if (m === "kfold" || m === "strat") {
        const order = m === "strat" ? [...Array(N).keys()].sort((a, b) => (a % 5) - (b % 5) || a - b) : [...Array(N).keys()];
        const fold = Array(N); order.forEach((idx, k) => (fold[idx] = m === "strat" ? k % 5 : Math.floor(k / 4)));
        h += row("class", cls.map(c => cell(c ? "#6b21a8" : "#94a3b8", c ? "1" : "0")).join(""));
        for (let f = 0; f < 5; f++) h += row("split " + (f + 1), fold.map(fo => cell(fo === f ? va : tr)).join(""));
      } else if (m === "group") {
        h += row("group", grp.map(g => cell(["#0e7490", "#7c3aed", "#b45309", "#15803d", "#be123c"][g], "G" + g)).join(""));
        for (let f = 0; f < 5; f++) h += row("split " + (f + 1), grp.map(g => cell(g === f ? va : tr)).join(""));
      } else if (m === "ts") {
        for (let f = 0; f < 5; f++) h += row("split " + (f + 1), Array.from({ length: N }, (_, i) => cell(i < 5 + f * 3 ? tr : i < 8 + f * 3 ? va : off)).join(""));
        h += row("", `<div style="flex:1;font-size:11px;color:var(--muted)">time →</div>`);
      } else if (m === "loo") {
        for (let f = 0; f < 6; f++) h += row("split " + (f + 1), Array.from({ length: N }, (_, i) => cell(i === f ? va : tr)).join(""));
        h += row("…", `<div style="font-size:12px;color:var(--muted)">… ${N} splits total</div>`);
      } else {
        for (let f = 0; f < 4; f++) {
          h += row("outer " + (f + 1), Array.from({ length: N }, (_, i) => cell(Math.floor(i / 5) === f ? te : tr)).join(""));
          if (f === 0) for (let g = 0; g < 3; g++) h += row("  inner " + (g + 1), Array.from({ length: N }, (_, i) => (Math.floor(i / 5) === 0 ? cell("transparent") : cell(Math.floor((i - 5) / 5) === g ? va : tr, "", "opacity:.75"))).join(""));
        }
      }
      h += `<div class="small" style="margin-top:6px"><span style="color:${tr}">■</span> train <span style="color:${va}">■</span> validation ${m === "nested" ? `<span style="color:${te}">■</span> outer test` : ""} ${m === "ts" ? `<span style="color:${off}">■</span> unused (future)` : ""}</div>`;
      root.querySelector("#" + u + "v").innerHTML = h; root.querySelector("#" + u + "t").innerHTML = TXT[m];
    }
    let cur = "kfold"; wireSeg(root, u + "m", v => { cur = v; draw(v); }); window.addEventListener("mlp-theme", () => draw(cur)); draw(cur);
  };

  // 13. Linear SVM margin with C ---------------------------------------------------
  D.svm = root => {
    const u = uid();
    root.innerHTML += `<div class="controls">${slider(u + "c", "C (log₁₀) — penalty on margin violations", -2, 2, 0.1, 0)}</div><canvas height="300"></canvas><div class="readout"></div>`;
    const cv = root.querySelector("canvas"), ro = root.querySelector(".readout");
    const P = []; for (let i = 0; i < 40; i++) { P.push([gauss() * 0.9 - 1.5, gauss() * 0.9 + 1, -1]); P.push([gauss() * 0.9 + 1.5, gauss() * 0.9 - 1, 1]); }
    P.push([0.8, 0.9, -1]); P.push([-0.6, -0.5, 1]);
    function train(C) { // full-batch subgradient on 0.5||w||² + C Σ hinge
      let w = [0, 0], b = 0;
      for (let t = 1; t <= 3000; t++) {
        const lr = 1 / (t * 0.5 + 50); let gw = [w[0], w[1]], gb = 0;
        P.forEach(([x, y, l]) => { if (l * (w[0] * x + w[1] * y + b) < 1) { gw[0] -= C * l * x; gw[1] -= C * l * y; gb -= C * l; } });
        w = [w[0] - lr * gw[0], w[1] - lr * gw[1]]; b -= lr * gb;
      }
      return [w, b];
    }
    function draw() {
      const lc = +root.querySelector("#" + u + "c").value, C = 10 ** lc; root.querySelector("#" + u + "cv").textContent = C.toPrecision(2);
      const [w, b] = train(C);
      const c = MLP_canvas(cv, 300); const F = frame(c, [-4.5, 4.5], [-3.5, 3.5], { ticks: false });
      const yAt = (x, off) => (off - b - w[0] * x) / w[1];
      [[0, "--ink", []], [1, "--muted", [5, 4]], [-1, "--muted", [5, 4]]].forEach(([o, colr, dsh]) => line(c.ctx, [[F.X(-4.5), F.Y(yAt(-4.5, o))], [F.X(4.5), F.Y(yAt(4.5, o))]], c.col(colr), o ? 1.5 : 2.5, dsh));
      let sv = 0, viol = 0;
      P.forEach(([x, y, l]) => { const m = l * (w[0] * x + w[1] * y + b); const isSV = m <= 1.0001; if (isSV) sv++; if (m < 1) viol++; dot(c.ctx, F.X(x), F.Y(y), isSV ? 5.5 : 3.8, l > 0 ? c.col("--coral") : c.col("--accent"), isSV ? c.col("--ink") : null); });
      const width = 2 / Math.hypot(w[0], w[1]);
      ro.innerHTML = `margin width 2/‖w‖ = ${fmt(width, 2)} · support vectors (circled) ≈ ${sv} · points inside margin / misclassified: ${viol}<br><span style="font-family:var(--sans);font-size:13.5px">${C < 0.3 ? "Small C → wide, soft margin: tolerates violations, lower variance (more regularised)." : C > 10 ? "Large C → narrow, hard-ish margin: tries to classify every point, risk of overfitting outliers." : "Medium C → balance between margin width and training errors."} Only the circled points determine the boundary.</span>`;
    }
    root.querySelector("input").addEventListener("input", draw); window.addEventListener("resize", draw); window.addEventListener("mlp-theme", draw); draw();
  };

  // 14. Drift: PSI & KS -------------------------------------------------------------
  D.drift = root => {
    const u = uid();
    root.innerHTML += `<div class="controls">${slider(u + "m", "mean shift in production", -2, 2, 0.05, 0.4)}${slider(u + "s", "std multiplier in production", 0.5, 2.5, 0.05, 1)}</div><canvas height="230"></canvas><div class="readout"></div>`;
    const cv = root.querySelector("canvas"), ro = root.querySelector(".readout");
    function draw() {
      const ms = +root.querySelector("#" + u + "m").value, ss = +root.querySelector("#" + u + "s").value;
      root.querySelector("#" + u + "mv").textContent = ms; root.querySelector("#" + u + "sv").textContent = ss;
      const edges = [-Infinity]; for (let i = 1; i < 10; i++) edges.push(PhiInv(i / 10)); edges.push(Infinity); // deciles of reference
      let psi = 0; const exp = [], act = [];
      for (let i = 0; i < 10; i++) { const e = 0.1; const a = Math.max(1e-4, Phi((edges[i + 1] - ms) / ss) - Phi((edges[i] - ms) / ss)); exp.push(e); act.push(a); psi += (a - e) * Math.log(a / e); }
      let ks = 0; for (let x = -6; x <= 6; x += 0.01) ks = Math.max(ks, Math.abs(Phi(x) - Phi((x - ms) / ss)));
      const c = MLP_canvas(cv, 230); const F = frame(c, [0, 10], [0, Math.max(0.35, ...act) * 1.1], { xl: "reference deciles (bins)", yl: "share of rows", ticks: false });
      const bw = (F.X(1) - F.X(0)) * 0.38;
      for (let i = 0; i < 10; i++) { c.ctx.fillStyle = c.col("--accent"); c.ctx.fillRect(F.X(i + 0.5) - bw, F.Y(exp[i]), bw, F.Y(0) - F.Y(exp[i])); c.ctx.fillStyle = c.col("--coral"); c.ctx.fillRect(F.X(i + 0.5), F.Y(act[i]), bw, F.Y(0) - F.Y(act[i])); }
      const lvl = psi < 0.1 ? "🟢 stable (< 0.1)" : psi < 0.25 ? "🟠 moderate shift (0.1–0.25): investigate" : "🔴 significant shift (> 0.25): likely retrain / alert";
      ro.innerHTML = `PSI = Σ (actual − expected)·ln(actual / expected) = <b>${fmt(psi)}</b> → ${lvl}<br>KS statistic (max CDF gap) = ${fmt(ks)}<br><span style="font-family:var(--sans);font-size:13.5px">Blue = training (reference) distribution, 10% per decile by construction; coral = production. Drift in inputs doesn't always hurt accuracy — confirm with performance metrics once labels arrive.</span>`;
    }
    root.querySelectorAll("input").forEach(i => i.addEventListener("input", draw)); window.addEventListener("resize", draw); window.addEventListener("mlp-theme", draw); draw();
  };

  // 15. A/B sample size -------------------------------------------------------------
  D.power = root => {
    const u = uid();
    root.innerHTML += `<div class="controls">${slider(u + "p", "baseline conversion rate (%)", 1, 50, 0.5, 10)}${slider(u + "m", "minimum detectable effect (relative %)", 1, 50, 1, 10)}${slider(u + "a", "significance α (two-sided)", 0.01, 0.1, 0.01, 0.05)}${slider(u + "w", "power 1 − β", 0.5, 0.99, 0.01, 0.8)}</div><div class="readout"></div>`;
    const ro = root.querySelector(".readout");
    function draw() {
      const p1 = +root.querySelector("#" + u + "p").value / 100, mde = +root.querySelector("#" + u + "m").value / 100, a = +root.querySelector("#" + u + "a").value, pw = +root.querySelector("#" + u + "w").value;
      ["p", "m", "a", "w"].forEach(k => root.querySelector("#" + u + k + "v").textContent = root.querySelector("#" + u + k).value);
      const p2 = p1 * (1 + mde), za = PhiInv(1 - a / 2), zb = PhiInv(pw), pb = (p1 + p2) / 2;
      const n = Math.ceil((za * Math.sqrt(2 * pb * (1 - pb)) + zb * Math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2 / (p2 - p1) ** 2);
      ro.innerHTML = `detect ${fmt(p1 * 100, 1)}% → ${fmt(p2 * 100, 2)}% (absolute +${fmt((p2 - p1) * 100, 2)} pp)<br>z<sub>α/2</sub> = ${fmt(za, 2)}, z<sub>β</sub> = ${fmt(zb, 2)}<br><b>≈ ${n.toLocaleString()} users per group</b> (${(2 * n).toLocaleString()} total)<br><span style="font-family:var(--sans);font-size:13.5px">Halving the MDE roughly quadruples n (n ∝ 1/δ²). Lower baselines and higher power also need more traffic. Fix n in advance — don't stop when p &lt; 0.05 first appears (peeking inflates false positives).</span>`;
    }
    root.querySelectorAll("input").forEach(i => i.addEventListener("input", draw)); draw();
  };

  document.addEventListener("DOMContentLoaded", () => {
    document.querySelectorAll("[data-demo]").forEach(el => {
      const f = D[el.dataset.demo];
      if (f) { try { f(el); } catch (e) { el.insertAdjacentHTML("beforeend", `<div class="small">Demo failed to load: ${e.message}</div>`); } }
    });
  });
})();
