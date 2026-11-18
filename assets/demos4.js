/* ML Interview Playbook — tree-ensemble and forecasting demos */
(function () {
  const H = window.MLP_D; if (!H) return;
  const { D, frame, dot, line, slider, seg, wireSeg, uid, fmt } = H;
  const lcg = seed => { let s = seed; return () => { s = (s * 16807) % 2147483647; return (s - 1) / 2147483646; }; };
  const normal = r => { let u = 0; while (!u) u = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r()); };
  const val = (root, id) => +root.querySelector("#" + id).value;
  const show = (root, ids, u) => ids.forEach(k => { const e = root.querySelector("#" + u + k + "v"); if (e) e.textContent = root.querySelector("#" + u + k).value; });
  const live = (root, draw) => { root.querySelectorAll("input").forEach(i => i.addEventListener("input", draw)); window.addEventListener("resize", draw); window.addEventListener("mlp-theme", draw); draw(); };
  const sig = z => 1 / (1 + Math.exp(-z));
  const mean = a => a.reduce((s, v) => s + v, 0) / (a.length || 1);

  // ---------- a small CART regression tree (squared error; on 0/1 targets this is Gini) ----------
  // opts: depth, minLeaf, mtry (features tried per split), random (Extra-Trees thresholds), rnd
  function fitTree(X, t, idx, opts, d = 0) {
    const m = mean(idx.map(i => t[i]));
    if (d >= opts.depth || idx.length < 2 * opts.minLeaf) return { leaf: m, idx };
    const p = X[0].length; let feats = [...Array(p).keys()];
    if (opts.mtry && opts.mtry < p) { feats = feats.sort(() => opts.rnd() - 0.5).slice(0, opts.mtry); }
    let best = null;
    for (const f of feats) {
      const ord = idx.slice().sort((a, b) => X[a][f] - X[b][f]);
      const n = ord.length; let sL = 0, qL = 0; const sT = ord.reduce((s, i) => s + t[i], 0), qT = ord.reduce((s, i) => s + t[i] * t[i], 0);
      if (opts.random) {
        const lo = X[ord[0]][f], hi = X[ord[n - 1]][f]; if (hi <= lo) continue;
        const thr = lo + (hi - lo) * opts.rnd(); let k = 0; for (; k < n && X[ord[k]][f] <= thr; k++) { sL += t[ord[k]]; qL += t[ord[k]] ** 2; }
        if (k < opts.minLeaf || n - k < opts.minLeaf) continue;
        const sse = qL - sL * sL / k + (qT - qL) - (sT - sL) ** 2 / (n - k);
        if (!best || sse < best.sse) best = { sse, f, thr, k, ord };
        continue;
      }
      for (let k = 1; k < n; k++) {
        const v = t[ord[k - 1]]; sL += v; qL += v * v;
        if (k < opts.minLeaf || n - k < opts.minLeaf || X[ord[k - 1]][f] === X[ord[k]][f]) continue;
        const sse = qL - sL * sL / k + (qT - qL) - (sT - sL) ** 2 / (n - k);
        if (!best || sse < best.sse - 1e-12) best = { sse, f, thr: (X[ord[k - 1]][f] + X[ord[k]][f]) / 2, k, ord };
      }
    }
    if (!best) return { leaf: m, idx };
    return { f: best.f, thr: best.thr, l: fitTree(X, t, best.ord.slice(0, best.k), opts, d + 1), r: fitTree(X, t, best.ord.slice(best.k), opts, d + 1) };
  }
  const predT = (n, x) => (n.leaf !== undefined ? n.leaf : x[n.f] <= n.thr ? predT(n.l, x) : predT(n.r, x));
  const leaves = (n, out = []) => { if (n.leaf !== undefined) out.push(n); else { leaves(n.l, out); leaves(n.r, out); } return out; };
  const boot = (n, rnd) => Array.from({ length: n }, () => Math.floor(rnd() * n));

  // ---------- 1. one dataset, five tree models ----------
  D.ensembles = root => {
    const u = uid();
    root.innerHTML += `<div style="margin:6px 0 10px">${seg(u + "m", [["tree", "One tree"], ["bag", "Bagging"], ["rf", "Random Forest"], ["et", "Extra Trees"], ["gb", "Gradient Boosting"]], "tree")}</div>
      <div class="controls">${slider(u + "d", "max_depth", 1, 12, 1, 6)}${slider(u + "n", "number of trees", 1, 100, 1, 1)}${slider(u + "lr", "learning rate (boosting only)", 0.05, 1, 0.05, 0.3)}</div>
      <canvas height="330"></canvas><div class="readout"></div>`;
    const cv = root.querySelector("canvas"), ro = root.querySelector(".readout");
    const mk = (n, seed) => { const r = lcg(seed), X = [], y = []; for (let i = 0; i < n; i++) { const c = i % 2, a = r() * Math.PI; let x = c ? 1 - Math.cos(a) : Math.cos(a), z = c ? 0.45 - Math.sin(a) : Math.sin(a); x += normal(r) * 0.18; z += normal(r) * 0.18; X.push([x, z]); y.push(r() < 0.08 ? 1 - c : c); } return [X, y]; };
    const [X, y] = mk(220, 3), [Xt, yt] = mk(1200, 9);
    let mode = "tree", model = null;
    function train() {
      const d = val(root, u + "d"), nT = mode === "tree" ? 1 : val(root, u + "n"), rnd = lcg(42), all = [...Array(X.length).keys()];
      if (mode === "gb") {
        const p0 = mean(y), F0 = Math.log(p0 / (1 - p0)), lr = val(root, u + "lr"), F = X.map(() => F0), trees = [];
        for (let m = 0; m < nT; m++) {
          const r = y.map((v, i) => v - sig(F[i])), tr = fitTree(X, r, all, { depth: Math.min(d, 4), minLeaf: 3, rnd });
          leaves(tr).forEach(L => { let a = 0, b = 0; L.idx.forEach(i => { a += r[i]; const p = sig(F[i]); b += p * (1 - p); }); L.leaf = a / (b + 1e-9); });
          X.forEach((x, i) => (F[i] += lr * predT(tr, x))); trees.push(tr);
        }
        return x => sig(F0 + lr * trees.reduce((s, tr) => s + predT(tr, x), 0));
      }
      const trees = [];
      for (let b = 0; b < nT; b++) {
        const idx = mode === "tree" || mode === "et" ? all : boot(X.length, rnd);
        trees.push(fitTree(X, y, idx, { depth: d, minLeaf: 1, rnd, mtry: mode === "rf" || mode === "et" ? 1 : 0, random: mode === "et" }));
      }
      return x => trees.reduce((s, tr) => s + predT(tr, x), 0) / trees.length;
    }
    const acc = (f, A, b) => A.reduce((s, x, i) => s + ((f(x) >= 0.5 ? 1 : 0) === b[i] ? 1 : 0), 0) / A.length;
    const NOTE = {
      tree: "A single tree cuts the plane into axis-aligned rectangles. Deep → it carves islands around noisy points (high variance); shallow → blocky and wrong near the curves (high bias).",
      bag: "Bagging trains each deep tree on a bootstrap sample and averages them: the islands that only one tree believed in get voted away. Bias stays about the same, variance drops.",
      rf: "Random Forest = bagging + each split may only look at a random subset of features (here 1 of 2). Trees disagree more, so averaging cancels more noise. With only two features that restriction also costs accuracy — the gain over bagging shows up when there are many, correlated features.",
      et: "Extra Trees also pick split thresholds at random. Even more diversity, smoother boundary, and much faster training — slightly more bias per tree.",
      gb: "Boosting adds shallow trees one after another, each fixing the current mistakes. Few trees → underfits; many trees with a high learning rate → starts chasing the noisy points again.",
    };
    function draw() {
      show(root, ["d", "n", "lr"], u);
      const f = model;
      const c = MLP_canvas(cv, 330), F = frame(c, [-1.6, 2.6], [-1.2, 1.7], { ticks: false });
      const step = 6, cA = c.col("--accent"), cB = c.col("--amber");
      for (let px = F.p.l; px < c.w - F.p.r; px += step) for (let py = F.p.t; py < c.h - F.p.b; py += step) {
        const p = f([F.iX(px + step / 2), F.iY(py + step / 2)]);
        c.ctx.globalAlpha = 0.08 + 0.32 * Math.abs(p - 0.5) * 2; c.ctx.fillStyle = p >= 0.5 ? cB : cA; c.ctx.fillRect(px, py, step, step);
      }
      c.ctx.globalAlpha = 1;
      X.forEach((x, i) => dot(c.ctx, F.X(x[0]), F.Y(x[1]), 3.6, y[i] ? cB : cA, c.col("--panel")));
      const tr = acc(f, X, y), te = acc(f, Xt, yt);
      ro.innerHTML = `train accuracy ${fmt(tr * 100, 1)}% · test accuracy <b>${fmt(te * 100, 1)}%</b> · gap ${fmt((tr - te) * 100, 1)} pts (8% of labels are deliberately flipped, so ~92% is the ceiling)<br><span style="font-family:var(--sans);font-size:13.5px">${NOTE[mode]}</span>`;
    }
    const retrain = () => { show(root, ["d", "n", "lr"], u); model = train(); draw(); };
    wireSeg(root, u + "m", v => { mode = v; if (v !== "tree" && val(root, u + "n") === 1) root.querySelector("#" + u + "n").value = v === "gb" ? 60 : 40; if (v === "gb") root.querySelector("#" + u + "d").value = 2; retrain(); });
    root.querySelectorAll("input").forEach(i => i.addEventListener("change", retrain));
    root.querySelectorAll("input").forEach(i => i.addEventListener("input", () => show(root, ["d", "n", "lr"], u)));
    window.addEventListener("resize", draw); window.addEventListener("mlp-theme", draw); retrain();
  };

  // ---------- 2. trees cannot extrapolate ----------
  D.extrap = root => {
    const u = uid();
    root.innerHTML += `<div style="margin:6px 0 10px">${seg(u + "m", [["tree", "Decision tree"], ["rf", "Random Forest"], ["gb", "Gradient Boosting"], ["lin", "Linear regression"], ["diff", "Linear trend + trees on residual"]], "tree")}</div><canvas height="280"></canvas><div class="readout"></div>`;
    const cv = root.querySelector("canvas"), ro = root.querySelector(".readout");
    const r0 = lcg(4), xs = Array.from({ length: 80 }, (_, i) => (i / 79) * 6), ys = xs.map(x => 2 + 1.5 * x + Math.sin(2 * x) * 0.8 + normal(r0) * 0.6);
    const X = xs.map(x => [x]), truth = x => 2 + 1.5 * x + Math.sin(2 * x) * 0.8;
    let mode = "tree";
    function model() {
      const all = [...X.keys()], rnd = lcg(1);
      if (mode === "lin") { const mx = mean(xs), my = mean(ys); let a = 0, b = 0; xs.forEach((x, i) => { a += (x - mx) * (ys[i] - my); b += (x - mx) ** 2; }); const s = a / b; return x => my + s * (x - mx); }
      if (mode === "tree") { const t = fitTree(X, ys, all, { depth: 6, minLeaf: 2 }); return x => predT(t, [x]); }
      if (mode === "rf") { const T = Array.from({ length: 40 }, () => fitTree(X, ys, boot(80, rnd), { depth: 8, minLeaf: 2 })); return x => mean(T.map(t => predT(t, [x]))); }
      if (mode === "gb") { const F0 = mean(ys), F = ys.map(() => F0), T = []; for (let m = 0; m < 80; m++) { const r = ys.map((v, i) => v - F[i]), t = fitTree(X, r, all, { depth: 2, minLeaf: 2 }); X.forEach((x, i) => (F[i] += 0.1 * predT(t, x))); T.push(t); } return x => F0 + 0.1 * T.reduce((s, t) => s + predT(t, [x]), 0); }
      // hybrid: linear trend + forest on the residual, using the position within the cycle as the feature
      const mx = mean(xs), my = mean(ys); let sa = 0, sb = 0; xs.forEach((x, i) => { sa += (x - mx) * (ys[i] - my); sb += (x - mx) ** 2; }); const sl = sa / sb, tr = x => my + sl * (x - mx);
      const ph = x => [((x % Math.PI) + Math.PI) % Math.PI], res = ys.map((v, i) => v - tr(xs[i])), Xp = xs.map(ph);
      const T = Array.from({ length: 40 }, () => fitTree(Xp, res, boot(80, rnd), { depth: 4, minLeaf: 3 }));
      return x => tr(x) + mean(T.map(t => predT(t, ph(x))));
    }
    function draw() {
      const f = model(), c = MLP_canvas(cv, 280), F = frame(c, [0, 10], [0, 20], { xl: "time", yl: "value" });
      c.ctx.fillStyle = c.col("--amber-soft"); c.ctx.fillRect(F.X(6), F.p.t, F.X(10) - F.X(6), c.h - F.p.t - F.p.b);
      c.ctx.fillStyle = c.col("--muted"); c.ctx.fillText("future (no training data)", F.X(6.2), F.p.t + 14);
      const tp = [], fp = []; for (let x = 0; x <= 10.0001; x += 0.05) { tp.push([F.X(x), F.Y(truth(x))]); fp.push([F.X(x), F.Y(Math.max(0, Math.min(20, f(x))))]); }
      line(c.ctx, tp, c.col("--green"), 1.5, [5, 4]); xs.forEach((x, i) => dot(c.ctx, F.X(x), F.Y(ys[i]), 2.6, c.col("--accent"))); line(c.ctx, fp, c.col("--coral"), 2.5);
      const p10 = f(10);
      const N = { tree: "Each leaf predicts an average of training targets, so beyond the last training point the prediction is frozen at the last leaf's value.", rf: "Averaging trees doesn't help: every tree is flat in the future, so the average is flat too.", gb: "Same for boosting — a sum of step functions, each flat beyond the data.", lin: "A straight line extrapolates the trend (but misses the wiggles). Extrapolation is only as good as the assumption behind it.", diff: "The fix forecasters use: let a simple model carry the <b>trend</b> (it can extrapolate), and let the trees learn only the leftover pattern from features that repeat — here the position within the cycle, like month-of-year. Modelling the change per step (Δy) or the ratio to a recent level works the same way." };
      ro.innerHTML = `prediction at time 10 = <b>${fmt(p10, 2)}</b> · true value ≈ ${fmt(truth(10), 2)}<br><span style="font-family:var(--sans);font-size:13.5px">${N[mode]}</span>`;
    }
    wireSeg(root, u + "m", v => { mode = v; draw(); }); window.addEventListener("resize", draw); window.addEventListener("mlp-theme", draw); draw();
  };

  // ---------- 3. bagging reduces variance ----------
  D.bagvar = root => {
    const u = uid();
    root.innerHTML += `<div class="controls">${slider(u + "b", "number of bootstrap trees (B)", 1, 60, 1, 1)}${slider(u + "d", "tree depth", 1, 12, 1, 8)}</div><div style="display:flex;gap:8px;margin-bottom:8px"><button class="btn small" data-a="new">↻ New training sample</button><label class="small"><input type="checkbox" id="${u}s" checked> show individual trees</label></div><canvas height="290"></canvas><div class="readout"></div>`;
    const cv = root.querySelector("canvas"), ro = root.querySelector(".readout");
    const f = x => Math.sin(1.4 * x) * 1.6; let seed = 1, xs, ys;
    const gen = () => { const r = lcg(seed); xs = Array.from({ length: 60 }, () => r() * 6).sort((a, b) => a - b); ys = xs.map(x => f(x) + normal(r) * 0.55); };
    const grid = Array.from({ length: 121 }, (_, i) => i * 0.05);
    function draw() {
      show(root, ["b", "d"], u);
      const B = val(root, u + "b"), d = val(root, u + "d"), X = xs.map(x => [x]), rnd = lcg(7 + seed);
      const T = Array.from({ length: B }, () => fitTree(X, ys, B === 1 ? [...X.keys()] : boot(60, rnd), { depth: d, minLeaf: 1 }));
      const preds = T.map(t => grid.map(x => predT(t, [x]))), avg = grid.map((_, j) => mean(preds.map(p => p[j])));
      const c = MLP_canvas(cv, 290), F = frame(c, [0, 6], [-3, 3], { xl: "x", yl: "y" });
      if (root.querySelector("#" + u + "s").checked && B > 1) { c.ctx.globalAlpha = Math.max(0.08, 0.6 / Math.sqrt(B)); preds.forEach(p => line(c.ctx, grid.map((x, j) => [F.X(x), F.Y(p[j])]), c.col("--accent"), 1)); c.ctx.globalAlpha = 1; }
      line(c.ctx, grid.map(x => [F.X(x), F.Y(f(x))]), c.col("--green"), 1.5, [5, 4]);
      xs.forEach((x, i) => dot(c.ctx, F.X(x), F.Y(ys[i]), 2.8, c.col("--muted")));
      line(c.ctx, grid.map((x, j) => [F.X(x), F.Y(avg[j])]), c.col("--coral"), 2.6);
      const mse = mean(grid.map((x, j) => (avg[j] - f(x)) ** 2)), spread = B > 1 ? mean(grid.map((_, j) => { const m = avg[j]; return mean(preds.map(p => (p[j] - m) ** 2)); })) : 0;
      ro.innerHTML = `error of the ${B > 1 ? "bagged average" : "single tree"} vs the true curve: <b>${fmt(mse, 3)}</b>${B > 1 ? ` · average disagreement between trees: ${fmt(spread, 3)}` : ""}<br><span style="font-family:var(--sans);font-size:13.5px">${B === 1 ? "A deep tree on one sample follows every noisy point. Press “New training sample” a few times: the red curve jumps around — that jumpiness is variance." : "Each thin line is one deep tree trained on a bootstrap sample; each is noisy in its own way. Their average (red) is far smoother and closer to the truth (dashed). Press “New training sample”: the average now barely moves."}</span>`;
    }
    root.querySelector("[data-a=new]").addEventListener("click", () => { seed++; gen(); draw(); });
    gen(); live(root, draw);
  };

  // ---------- 4. exponential smoothing: SES, Holt, Holt-Winters ----------
  D.esdemo = root => {
    const u = uid();
    root.innerHTML += `<div style="margin:6px 0 10px">${seg(u + "m", [["ses", "SES (level)"], ["holt", "Holt (+ trend)"], ["hw", "Holt-Winters (+ season)"]], "hw")} <button class="btn small" data-a="opt">⚙ Fit α, β, γ automatically</button></div>
      <div class="controls">${slider(u + "a", "α level", 0.01, 1, 0.01, 0.3)}${slider(u + "b", "β trend", 0, 1, 0.01, 0.1)}${slider(u + "g", "γ season", 0, 1, 0.01, 0.2)}</div><canvas height="290"></canvas><div class="readout"></div>`;
    const cv = root.querySelector("canvas"), ro = root.querySelector(".readout");
    const r0 = lcg(12), N = 84, S = 12, y = Array.from({ length: N }, (_, t) => 100 + 1.2 * t + 14 * Math.sin(2 * Math.PI * t / S) + 6 * Math.cos(4 * Math.PI * t / S) + normal(r0) * 4);
    const nTr = 72; let mode = "hw";
    function run(a, b, g) {
      const tr = y.slice(0, nTr); let L = mean(tr.slice(0, S)), T = mode === "ses" ? 0 : (mean(tr.slice(S, 2 * S)) - mean(tr.slice(0, S))) / S;
      const s = tr.slice(0, S).map(v => (mode === "hw" ? v - L : 0)), fit = [];
      for (let t = 0; t < nTr; t++) {
        const st = s[t % S], yh = L + T + st; fit.push(t === 0 ? NaN : yh);
        const Lp = L; L = a * (tr[t] - st) + (1 - a) * (L + T);
        if (mode !== "ses") T = b * (L - Lp) + (1 - b) * T;
        if (mode === "hw") s[t % S] = g * (tr[t] - L) + (1 - g) * st;
      }
      const fc = Array.from({ length: N - nTr }, (_, h) => L + (h + 1) * T + s[(nTr + h) % S]);
      const sse = fit.slice(S).reduce((q, v, i) => q + (tr[i + S] - v) ** 2, 0);
      return { fit, fc, sse, mae: mean(fc.map((v, h) => Math.abs(v - y[nTr + h]))) };
    }
    function draw() {
      show(root, ["a", "b", "g"], u);
      const R = run(val(root, u + "a"), val(root, u + "b"), val(root, u + "g"));
      const c = MLP_canvas(cv, 290), F = frame(c, [0, N], [60, 220], { xl: "month", yl: "sales" });
      c.ctx.fillStyle = c.col("--amber-soft"); c.ctx.fillRect(F.X(nTr), F.p.t, F.X(N) - F.X(nTr), c.h - F.p.t - F.p.b);
      line(c.ctx, y.map((v, t) => [F.X(t), F.Y(v)]), c.col("--muted"), 1.3);
      line(c.ctx, R.fit.map((v, t) => [F.X(t), F.Y(v)]).filter(p => isFinite(p[1])), c.col("--accent"), 2);
      line(c.ctx, R.fc.map((v, h) => [F.X(nTr + h), F.Y(v)]), c.col("--coral"), 2.6);
      const naive = mean(y.slice(nTr).map((v, h) => Math.abs(v - y[nTr - S + h])));
      const N2 = { ses: "SES keeps only a level: its forecast is a flat line, so it can't follow either the trend or the seasonal swings.", holt: "Holt adds a trend: the forecast now rises, but it still ignores the yearly pattern.", hw: "Holt-Winters adds a seasonal index per month: the forecast repeats the learned pattern on top of the trend." };
      ro.innerHTML = `in-sample SSE ${fmt(R.sse, 0)} · <b>holdout MAE (last 12 months) ${fmt(R.mae, 2)}</b> · seasonal-naive MAE ${fmt(naive, 2)}<br><span style="font-family:var(--sans);font-size:13.5px">${N2[mode]} Higher α/β/γ = react faster to recent data (and to noise); lower = smoother, slower to adapt.</span>`;
      root.querySelector("#" + u + "b").disabled = mode === "ses"; root.querySelector("#" + u + "g").disabled = mode !== "hw";
    }
    root.querySelector("[data-a=opt]").addEventListener("click", () => {
      let best = null; const G = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9];
      G.forEach(a => (mode === "ses" ? [0] : [0, 0.02, 0.05, 0.1, 0.2]).forEach(b => (mode === "hw" ? [0, 0.05, 0.1, 0.2, 0.4] : [0]).forEach(g => { const s = run(a, b, g).sse; if (!best || s < best[0]) best = [s, a, b, g]; })));
      root.querySelector("#" + u + "a").value = best[1]; root.querySelector("#" + u + "b").value = best[2]; root.querySelector("#" + u + "g").value = best[3]; draw();
    });
    wireSeg(root, u + "m", v => { mode = v; draw(); }); live(root, draw);
  };

  // ---------- 5. ARIMA simulator with ACF and PACF ----------
  function acf(x, K) { const n = x.length, m = mean(x); let c0 = 0; x.forEach(v => (c0 += (v - m) ** 2)); return Array.from({ length: K + 1 }, (_, k) => { let s = 0; for (let t = k; t < n; t++) s += (x[t] - m) * (x[t - k] - m); return s / c0; }); }
  function pacf(r, K) { const out = [1], phi = []; for (let k = 1; k <= K; k++) { let num = r[k], den = 1; for (let j = 1; j < k; j++) { num -= phi[k - 1][j] * r[k - j]; den -= phi[k - 1][j] * r[j]; } const pkk = num / den, row = Array(k + 1).fill(0); for (let j = 1; j < k; j++) row[j] = phi[k - 1][j] - pkk * phi[k - 1][k - j]; row[k] = pkk; phi[k] = row; out.push(pkk); } return out; }
  D.arimasim = root => {
    const u = uid();
    root.innerHTML += `<div style="margin:6px 0 10px">${seg(u + "p", [["ar1", "AR(1)"], ["ar2", "AR(2)"], ["ma1", "MA(1)"], ["arma", "ARMA(1,1)"], ["rw", "Random walk"], ["sar", "Seasonal AR (lag 12)"]], "ar1")} <button class="btn small" data-a="seed">↻ New noise</button></div>
      <div class="controls">${slider(u + "p1", "φ₁", -0.95, 0.95, 0.05, 0.7)}${slider(u + "p2", "φ₂", -0.95, 0.95, 0.05, 0)}${slider(u + "t1", "θ₁", -0.95, 0.95, 0.05, 0)}${slider(u + "s", "Φ₁₂ (seasonal AR)", 0, 0.95, 0.05, 0)}<div class="control"><label>differencing</label><label class="small"><input type="checkbox" id="${u}d"> integrate once (d = 1)</label></div></div>
      <canvas height="200"></canvas><div class="demo-grid"><div><div class="small">ACF — autocorrelation</div><canvas height="190"></canvas></div><div><div class="small">PACF — partial autocorrelation</div><canvas height="190"></canvas></div></div><div class="readout"></div>`;
    const [c1, c2, c3] = root.querySelectorAll("canvas"), ro = root.querySelector(".readout"); let seed = 3;
    const P = { ar1: [0.7, 0, 0, 0, 0], ar2: [0.5, 0.35, 0, 0, 0], ma1: [0, 0, 0.8, 0, 0], arma: [0.6, 0, 0.5, 0, 0], rw: [0, 0, 0, 0, 1], sar: [0.3, 0, 0, 0.7, 0] };
    function bars(cv, vals, n, lab) {
      const c = MLP_canvas(cv, 190), F = frame(c, [0, vals.length], [-1, 1], { ticks: false, xl: "lag" }), band = 1.96 / Math.sqrt(n);
      c.ctx.fillStyle = c.col("--accent-soft"); c.ctx.fillRect(F.X(0), F.Y(band), F.X(vals.length) - F.X(0), F.Y(-band) - F.Y(band));
      line(c.ctx, [[F.X(0), F.Y(0)], [F.X(vals.length), F.Y(0)]], c.col("--line"), 1);
      vals.forEach((v, k) => { if (k === 0) return; const sig = Math.abs(v) > band; c.ctx.fillStyle = sig ? c.col("--coral") : c.col("--muted"); const x = F.X(k) - 3; c.ctx.fillRect(x, Math.min(F.Y(v), F.Y(0)), 6, Math.abs(F.Y(v) - F.Y(0))); });
      c.ctx.fillStyle = c.col("--muted"); [1, 6, 12, 18, 24].forEach(k => { if (k < vals.length) c.ctx.fillText(k, F.X(k) - 3, c.h - 8); });
    }
    function draw() {
      show(root, ["p1", "p2", "t1", "s"], u);
      const p1 = val(root, u + "p1"), p2 = val(root, u + "p2"), t1 = val(root, u + "t1"), S1 = val(root, u + "s"), d = root.querySelector("#" + u + "d").checked;
      const r = lcg(seed), n = 300, e = Array.from({ length: n + 100 }, () => normal(r)), x = Array(n + 100).fill(0);
      for (let t = 13; t < n + 100; t++) x[t] = p1 * x[t - 1] + p2 * x[t - 2] + S1 * x[t - 12] - p1 * S1 * x[t - 13] + e[t] + t1 * e[t - 1];
      let s = x.slice(100); if (d) { let c = 0; s = s.map(v => (c += v)); }
      const c = MLP_canvas(c1, 200), mn = Math.min(...s), mx = Math.max(...s), F = frame(c, [0, n], [mn - 0.5, mx + 0.5], { xl: "time", yl: "series" });
      line(c.ctx, s.map((v, t) => [F.X(t), F.Y(v)]), c.col("--accent"), 1.3);
      const A = acf(s, 26), PA = pacf(A, 26); bars(c2, A, n); bars(c3, PA, n);
      const statOK = Math.abs(p2) < 1 && p2 + p1 < 1 && p2 - p1 < 1 && S1 < 1 && !d;
      let hint;
      if (d) hint = "A unit root: the ACF decays very slowly from near 1 — the signature of a non-stationary series. Difference it (untick the box) before choosing p and q.";
      else if (!statOK) hint = "These AR coefficients are outside the stationary region (φ₁ + φ₂ &lt; 1, φ₂ − φ₁ &lt; 1, |φ₂| &lt; 1): shocks never die out and the series explodes or wanders.";
      else if (S1 > 0.2) hint = "Seasonal AR: spikes at lags 12, 24 in the ACF and a spike at lag 12 in the PACF → add a seasonal term, SARIMA(p,d,q)(1,0,0)₁₂.";
      else if (t1 !== 0 && p1 === 0 && p2 === 0) hint = "Pure MA(q): the ACF cuts off after lag q, while the PACF tails off. Read q from the ACF.";
      else if (t1 !== 0) hint = "ARMA: both ACF and PACF tail off gradually — neither cuts off cleanly, so pick orders by AIC rather than by eye.";
      else hint = `Pure AR(p): the PACF cuts off after lag ${p2 !== 0 ? 2 : 1} while the ACF decays gradually. Read p from the PACF.`;
      ro.innerHTML = `${statOK ? "✅ stationary" : "⚠️ non-stationary"} · lag-1 ACF ${fmt(A[1], 2)} · lag-2 PACF ${fmt(PA[2], 2)} · coral bars are outside the ±1.96/√n band (statistically significant)<br><span style="font-family:var(--sans);font-size:13.5px">${hint}</span>`;
    }
    wireSeg(root, u + "p", v => { const q = P[v]; ["p1", "p2", "t1", "s"].forEach((k, i) => (root.querySelector("#" + u + k).value = q[i])); root.querySelector("#" + u + "d").checked = !!q[4]; draw(); });
    root.querySelector("[data-a=seed]").addEventListener("click", () => { seed++; draw(); });
    root.querySelector("#" + u + "d").addEventListener("change", draw); live(root, draw);
  };

  // ---------- 6. forecast fan chart: uncertainty grows with horizon ----------
  D.fanchart = root => {
    const u = uid();
    root.innerHTML += `<div style="margin:6px 0 10px">${seg(u + "m", [["rw", "Random walk"], ["ar", "AR(1)"], ["drift", "Random walk + drift"], ["ets", "Holt (trend) model"]], "rw")}</div><div class="controls">${slider(u + "p", "φ (AR only)", 0, 0.98, 0.02, 0.7)}${slider(u + "s", "noise σ", 0.2, 3, 0.1, 1)}</div><canvas height="280"></canvas><div class="readout"></div>`;
    const cv = root.querySelector("canvas"), ro = root.querySelector(".readout"); let mode = "rw";
    function draw() {
      show(root, ["p", "s"], u);
      const phi = val(root, u + "p"), sg = val(root, u + "s"), r = lcg(8), n = 60, Hh = 40, x = [10];
      for (let t = 1; t < n; t++) x.push(mode === "ar" ? 10 + phi * (x[t - 1] - 10) + sg * normal(r) : x[t - 1] + (mode === "drift" || mode === "ets" ? 0.15 : 0) + sg * normal(r));
      const last = x[n - 1], mu = [], sd = [];
      for (let h = 1; h <= Hh; h++) {
        if (mode === "rw") { mu.push(last); sd.push(sg * Math.sqrt(h)); }
        else if (mode === "ar") { mu.push(10 + phi ** h * (last - 10)); sd.push(sg * Math.sqrt((1 - phi ** (2 * h)) / (1 - phi * phi + 1e-12))); }
        else if (mode === "drift") { mu.push(last + 0.15 * h); sd.push(sg * Math.sqrt(h * (1 + h / n))); }
        else { const a = 0.5, b = 0.1; mu.push(last + 0.15 * h); let v = 1; for (let j = 1; j < h; j++) v += (a + j * a * b) ** 2; sd.push(sg * Math.sqrt(v)); }
      }
      const lo = Math.min(...x, ...mu.map((m, i) => m - 2 * sd[i])), hi = Math.max(...x, ...mu.map((m, i) => m + 2 * sd[i]));
      const c = MLP_canvas(cv, 280), F = frame(c, [0, n + Hh], [lo - 1, hi + 1], { xl: "time", yl: "value" });
      const band = (k, col, al) => { c.ctx.globalAlpha = al; c.ctx.fillStyle = col; c.ctx.beginPath(); c.ctx.moveTo(F.X(n - 1), F.Y(last)); mu.forEach((m, i) => c.ctx.lineTo(F.X(n + i), F.Y(m + k * sd[i]))); for (let i = Hh - 1; i >= 0; i--) c.ctx.lineTo(F.X(n + i), F.Y(mu[i] - k * sd[i])); c.ctx.closePath(); c.ctx.fill(); c.ctx.globalAlpha = 1; };
      band(1.96, c.col("--accent"), 0.15); band(1.28, c.col("--accent"), 0.22);
      line(c.ctx, x.map((v, t) => [F.X(t), F.Y(v)]), c.col("--ink"), 1.5); line(c.ctx, [[F.X(n - 1), F.Y(last)]].concat(mu.map((m, i) => [F.X(n + i), F.Y(m)])), c.col("--coral"), 2.4);
      const N = { rw: "Random walk: the best forecast is the last value, and the 95% interval widens like √h without limit — the future is genuinely unknowable far out.", ar: "Stationary AR(1): the forecast decays back to the long-run mean and the interval stops widening once h is large — it converges to the series' own spread.", drift: "With drift the centre follows the average past change; the interval still widens like √h.", ets: "Trend models widen faster than √h because the trend estimate itself is uncertain — a small slope error grows with every step." };
      ro.innerHTML = `95% interval half-width: h = 1 → ±${fmt(1.96 * sd[0], 2)} · h = 10 → ±${fmt(1.96 * sd[9], 2)} · h = 40 → ±${fmt(1.96 * sd[39], 2)}<br><span style="font-family:var(--sans);font-size:13.5px">${N[mode]} Shaded: 80% and 95% prediction intervals.</span>`;
      root.querySelector("#" + u + "p").disabled = mode !== "ar";
    }
    wireSeg(root, u + "m", v => { mode = v; draw(); }); live(root, draw);
  };

  // ---------- 7. backtesting (rolling-origin evaluation) ----------
  D.backtest = root => {
    const u = uid();
    root.innerHTML += `<div style="margin:6px 0 10px">${seg(u + "m", [["exp", "Expanding window"], ["slide", "Sliding window"]], "exp")}</div><div class="controls">${slider(u + "i", "initial training length", 12, 70, 1, 36)}${slider(u + "h", "forecast horizon", 1, 24, 1, 6)}${slider(u + "s", "step between origins", 1, 24, 1, 6)}${slider(u + "g", "gap (data delay)", 0, 6, 1, 0)}</div><canvas height="260"></canvas><div class="readout"></div>`;
    const cv = root.querySelector("canvas"), ro = root.querySelector(".readout"); let mode = "exp";
    function draw() {
      show(root, ["i", "h", "s", "g"], u);
      const N = 96, I = val(root, u + "i"), Hh = val(root, u + "h"), St = val(root, u + "s"), G = val(root, u + "g"), folds = [];
      for (let o = I; o + G + Hh <= N; o += St) folds.push(o);
      const c = MLP_canvas(cv, 260), rows = Math.max(1, folds.length), rh = Math.min(22, (c.h - 40) / rows), x0 = 70, w = (c.w - x0 - 12) / N;
      c.ctx.font = "11px ui-monospace,monospace";
      folds.slice(0, 40).forEach((o, k) => {
        const y = 10 + k * rh, s = mode === "slide" ? o - I : 0;
        c.ctx.fillStyle = c.col("--muted"); c.ctx.fillText("origin " + o, 4, y + rh * 0.7);
        c.ctx.fillStyle = c.col("--line"); c.ctx.fillRect(x0, y + 2, N * w, rh - 4);
        c.ctx.fillStyle = c.col("--accent"); c.ctx.fillRect(x0 + s * w, y + 2, (o - s) * w, rh - 4);
        if (G) { c.ctx.fillStyle = c.col("--amber"); c.ctx.fillRect(x0 + o * w, y + 2, G * w, rh - 4); }
        c.ctx.fillStyle = c.col("--coral"); c.ctx.fillRect(x0 + (o + G) * w, y + 2, Hh * w, rh - 4);
      });
      c.ctx.fillStyle = c.col("--muted"); c.ctx.fillText("time →", x0, c.h - 6);
      if (!folds.length) { ro.innerHTML = `<b>No backtest windows fit.</b> <span style="font-family:var(--sans);font-size:13.5px">The initial training length + gap + horizon is longer than the 96 periods of history. Shorten one of them — in practice this is the “not enough history to validate a long horizon” problem.</span>`; return; }
      ro.innerHTML = `${folds.length} backtest windows × ${Hh} steps = ${folds.length * Hh} forecast errors to average · <span style="color:var(--accent)">■</span> train <span style="color:var(--coral)">■</span> forecast ${G ? '<span style="color:var(--amber)">■</span> gap' : ""}<br><span style="font-family:var(--sans);font-size:13.5px">${mode === "exp" ? "Expanding: every refit uses all history so far — mimics retraining on everything you have." : "Sliding: fixed-length window — mimics a model that forgets old regimes, and lets you test whether old data hurts."} Set the horizon to the one the business uses, the step to how often you'll retrain, and the gap to how late the data arrive. Report error per horizon, not just the average.</span>`;
    }
    wireSeg(root, u + "m", v => { mode = v; draw(); }); live(root, draw);
  };

  // ---------- 8. supervised windowing for ML / LSTM ----------
  D.windowing = root => {
    const u = uid();
    root.innerHTML += `<div class="controls">${slider(u + "l", "lookback window L", 2, 12, 1, 5)}${slider(u + "h", "horizon H", 1, 6, 1, 1)}${slider(u + "k", "highlight sample #", 0, 20, 1, 0)}</div><div id="${u}g" style="overflow-x:auto"></div><div class="readout" id="${u}r"></div>`;
    const vals = [12, 15, 14, 18, 21, 19, 24, 27, 25, 29, 33, 31, 36, 39, 37, 42, 45, 43, 48, 51, 49, 54, 57, 55, 60, 63, 61, 66];
    function draw() {
      show(root, ["l", "h", "k"], u);
      const L = val(root, u + "l"), Hh = val(root, u + "h"), nS = vals.length - L - Hh + 1, ks = root.querySelector("#" + u + "k"); ks.max = nS - 1; const k = Math.min(val(root, u + "k"), nS - 1);
      const cell = (v, i) => { const inX = i >= k && i < k + L, inY = i >= k + L && i < k + L + Hh; return `<div style="min-width:34px;text-align:center;padding:8px 2px;border-radius:6px;font:12px ui-monospace,monospace;background:${inX ? "var(--accent)" : inY ? "var(--coral)" : "var(--bg)"};color:${inX || inY ? "#fff" : "var(--ink)"}">${v}<br><span style="font-size:9px;opacity:.7">t${i}</span></div>`; };
      root.querySelector("#" + u + "g").innerHTML = `<div style="display:flex;gap:3px">${vals.map(cell).join("")}</div>`;
      root.querySelector("#" + u + "r").innerHTML = `sample ${k}: X = [${vals.slice(k, k + L).join(", ")}] → y = [${vals.slice(k + L, k + L + Hh).join(", ")}]<br>${vals.length} time steps → <b>${nS} training samples</b>, each with ${L} inputs and ${Hh} target${Hh > 1 ? "s" : ""}.<br><span style="font-family:var(--sans);font-size:13.5px">This is how any ML or deep model (MLP, LSTM, TCN, transformer) sees a series: a table of sliding windows. Longer L = more context but fewer samples; split the windows by time (never shuffle across the split) and scale using training data only.</span>`;
    }
    live(root, draw);
  };

  // ---------- 9. LSTM cell intuition vs vanilla RNN ----------
  D.lstmcell = root => {
    const u = uid();
    root.innerHTML += `<div class="controls">${slider(u + "f", "forget gate f (keep memory)", 0, 1, 0.01, 0.95)}${slider(u + "i", "input gate i (write new info)", 0, 1, 0.01, 0.8)}${slider(u + "w", "vanilla RNN recurrent weight w", 0, 1.5, 0.01, 0.8)}</div><label class="small"><input type="checkbox" id="${u}r" checked> a “reset” signal at t = 30 closes the forget gate for one step</label>
      <div class="demo-grid"><div><div class="small">Response to one spike of input at t = 5</div><canvas height="230"></canvas></div><div><div class="small">How much gradient survives going back k steps (log scale)</div><canvas height="230"></canvas></div></div><div class="readout"></div>`;
    const [c1, c2] = root.querySelectorAll("canvas"), ro = root.querySelector(".readout");
    function draw() {
      show(root, ["f", "i", "w"], u);
      const f = val(root, u + "f"), ig = val(root, u + "i"), w = val(root, u + "w"), rs = root.querySelector("#" + u + "r").checked, T = 50;
      let c = 0, h = 0; const C = [], Hh = [];
      for (let t = 0; t < T; t++) { const x = t === 5 ? 1 : 0, ft = rs && t === 30 ? 0 : f; c = ft * c + ig * Math.tanh(2 * x); C.push(c); h = Math.tanh(w * h + 1.5 * x); Hh.push(h); }
      const k1 = MLP_canvas(c1, 230), F = frame(k1, [0, T], [-0.05, 1.05], { xl: "time step", yl: "memory" });
      line(k1.ctx, [[F.X(5), F.Y(0)], [F.X(5), F.Y(1)]], k1.col("--line"), 1, [3, 3]);
      line(k1.ctx, C.map((v, t) => [F.X(t), F.Y(v / Math.max(1e-9, ig * Math.tanh(2)))]), k1.col("--accent"), 2.5);
      line(k1.ctx, Hh.map((v, t) => [F.X(t), F.Y(v / Math.tanh(1.5))]), k1.col("--coral"), 2);
      k1.ctx.fillStyle = k1.col("--accent"); k1.ctx.fillText("LSTM cell state c", F.X(31), F.Y(0.95)); k1.ctx.fillStyle = k1.col("--coral"); k1.ctx.fillText("vanilla RNN state h", F.X(31), F.Y(0.85));
      const k2 = MLP_canvas(c2, 230), G = frame(k2, [0, 40], [-12, 0.5], { xl: "steps back (k)", yl: "log₁₀ gradient" });
      const rnn = [], lst = []; for (let k = 0; k <= 40; k++) { rnn.push([G.X(k), G.Y(Math.max(-12, k * Math.log10(Math.max(1e-12, w * 0.65))))]); lst.push([G.X(k), G.Y(Math.max(-12, k * Math.log10(Math.max(1e-12, f))))]); }
      line(k2.ctx, lst, k2.col("--accent"), 2.5); line(k2.ctx, rnn, k2.col("--coral"), 2);
      const r20 = (w * 0.65) ** 20, l20 = f ** 20;
      ro.innerHTML = `after 20 steps the gradient keeps ≈ ${r20.toExponential(1)} of its size in the RNN vs ≈ ${l20.toExponential(1)} along the LSTM cell<br><span style="font-family:var(--sans);font-size:13.5px">The vanilla RNN multiplies its memory by w·tanh′ at every step: below 1 the signal (and the gradient) vanishes within a few steps, above 1 it explodes. The LSTM's cell state is updated <b>additively</b> — c<sub>t</sub> = f·c<sub>t−1</sub> + i·g — so with f near 1 information and gradients flow across long gaps. Because f is computed from the input, the network can also <b>decide</b> to forget (the reset at t = 30).</span>`;
    }
    root.querySelector("#" + u + "r").addEventListener("change", draw); live(root, draw);
  };

  // ---------- 10. intermittent demand: SES vs Croston vs TSB ----------
  D.croston = root => {
    const u = uid();
    root.innerHTML += `<div class="controls">${slider(u + "a", "smoothing α", 0.05, 0.6, 0.01, 0.15)}${slider(u + "p", "chance of a sale each week", 0.1, 0.9, 0.05, 0.3)}</div><canvas height="270"></canvas><div class="readout"></div>`;
    const cv = root.querySelector("canvas"), ro = root.querySelector(".readout");
    function draw() {
      show(root, ["a", "p"], u);
      const a = val(root, u + "a"), pr = val(root, u + "p"), r = lcg(21), T = 60;
      const y = Array.from({ length: T }, () => (r() < pr ? 1 + Math.floor(r() * 6) : 0)), trueRate = pr * 3.5;
      const ses = [], cro = [], tsb = []; let s = y[0] || 1, z = 3.5, q = 1 / pr, cnt = 1, pz = 3.5, pp = pr;
      for (let t = 0; t < T; t++) {
        ses.push(s); cro.push(z / q); tsb.push(pp * pz);
        s = a * y[t] + (1 - a) * s;
        if (y[t] > 0) { z = a * y[t] + (1 - a) * z; q = a * cnt + (1 - a) * q; cnt = 1; pz = a * y[t] + (1 - a) * pz; pp = a + (1 - a) * pp; } else { cnt++; pp = (1 - a) * pp; }
      }
      const c = MLP_canvas(cv, 270), F = frame(c, [0, T], [0, 7], { xl: "week", yl: "units" }), bw = (F.X(1) - F.X(0)) * 0.6;
      y.forEach((v, t) => { if (v) { c.ctx.fillStyle = c.col("--line"); c.ctx.fillRect(F.X(t) - bw / 2, F.Y(v), bw, F.Y(0) - F.Y(v)); } });
      line(c.ctx, [[F.X(0), F.Y(trueRate)], [F.X(T), F.Y(trueRate)]], c.col("--green"), 1.5, [5, 4]);
      line(c.ctx, ses.map((v, t) => [F.X(t), F.Y(v)]), c.col("--coral"), 1.8); line(c.ctx, cro.map((v, t) => [F.X(t), F.Y(v)]), c.col("--accent"), 2.4); line(c.ctx, tsb.map((v, t) => [F.X(t), F.Y(v)]), c.col("--amber"), 1.8);
      c.ctx.fillStyle = c.col("--coral"); c.ctx.fillText("SES", c.w - 110, 18); c.ctx.fillStyle = c.col("--accent"); c.ctx.fillText("Croston", c.w - 80, 18); c.ctx.fillStyle = c.col("--amber"); c.ctx.fillText("TSB", c.w - 30, 18);
      const zeros = y.filter(v => !v).length, sd = v => Math.sqrt(mean(v.slice(10).map(x => (x - mean(v.slice(10))) ** 2)));
      ro.innerHTML = `${zeros} of ${T} weeks have zero sales · true average demand ${fmt(trueRate, 2)}/week (dashed) · forecast wobble (SD): SES ${fmt(sd(ses), 2)}, Croston ${fmt(sd(cro), 2)}, TSB ${fmt(sd(tsb), 2)}<br><span style="font-family:var(--sans);font-size:13.5px">SES jumps after every sale and decays during gaps, so it overshoots right after demand and undershoots before it. Croston smooths the <b>size</b> of non-zero demands and the <b>interval</b> between them separately and divides — a steadier rate (it's slightly biased high; the SBA variant multiplies by 1 − α/2). TSB smooths the <b>probability</b> of a sale instead of the interval, so it can decay towards zero for a product that's dying.</span>`;
    }
    live(root, draw);
  };
})();
