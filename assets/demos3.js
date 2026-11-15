/* ML Interview Playbook — statistics demos: covariance/correlation/cosine, vector similarity, Monte Carlo */
(function () {
  const H = window.MLP_D; if (!H) return;
  const { D, frame, dot, line, slider, seg, wireSeg, uid, fmt } = H;
  const lcg = seed => { let s = seed; return () => { s = (s * 16807) % 2147483647; return (s - 1) / 2147483646; }; };
  const normal = r => { let u = 0; while (!u) u = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r()); };
  const val = (root, id) => +root.querySelector("#" + id).value;
  const show = (root, ids, u) => ids.forEach(k => { const e = root.querySelector("#" + u + k + "v"); if (e) e.textContent = root.querySelector("#" + u + k).value; });
  const redraw = (root, draw) => { root.querySelectorAll("input").forEach(i => i.addEventListener("input", draw)); window.addEventListener("resize", draw); window.addEventListener("mlp-theme", draw); draw(); };
  const mean = a => a.reduce((s, v) => s + v, 0) / a.length;
  const ranks = a => { const idx = a.map((v, i) => [v, i]).sort((p, q) => p[0] - q[0]); const r = Array(a.length); let i = 0; while (i < idx.length) { let j = i; while (j + 1 < idx.length && idx[j + 1][0] === idx[i][0]) j++; for (let k = i; k <= j; k++) r[idx[k][1]] = (i + j) / 2 + 1; i = j + 1; } return r; };
  function stats(X, Y) {
    const n = X.length, mx = mean(X), my = mean(Y);
    let sxx = 0, syy = 0, sxy = 0, dot0 = 0, nx = 0, ny = 0;
    for (let i = 0; i < n; i++) { const a = X[i] - mx, b = Y[i] - my; sxx += a * a; syy += b * b; sxy += a * b; dot0 += X[i] * Y[i]; nx += X[i] * X[i]; ny += Y[i] * Y[i]; }
    const r = sxy / Math.sqrt(sxx * syy);
    const rx = ranks(X), ry = ranks(Y), mr = mean(rx); let a = 0, b = 0, c = 0; for (let i = 0; i < n; i++) { a += (rx[i] - mr) * (ry[i] - mr); b += (rx[i] - mr) ** 2; c += (ry[i] - mr) ** 2; }
    return { n, mx, my, vx: sxx / (n - 1), vy: syy / (n - 1), cov: sxy / (n - 1), r, rho: a / Math.sqrt(b * c), cos: dot0 / Math.sqrt(nx * ny), byx: sxy / sxx, bxy: sxy / syy };
  }

  // 1. Covariance, correlation, cosine and regression on one scatter -------------------
  D.assoc = root => {
    const u = uid();
    root.innerHTML += `<div style="margin:6px 0 10px">${seg(u + "s", [["lin", "Linear"], ["out", "+ one outlier"], ["curve", "U-shaped (y = x²)"], ["none", "No relation"]], "lin")}</div>
      <div class="controls">${slider(u + "r", "strength of linear link", -1, 1, 0.05, 0.7)}${slider(u + "k", "rescale x (e.g. hours → minutes)", 1, 60, 1, 1)}${slider(u + "o", "add a constant to y", 0, 50, 1, 0)}</div>
      <canvas height="320"></canvas><div class="readout"></div>`;
    const cv = root.querySelector("canvas"), ro = root.querySelector(".readout");
    let mode = "lin"; const base = (() => { const r = lcg(11); return Array.from({ length: 70 }, () => [normal(r), normal(r)]); })();
    function data() {
      const rr = val(root, u + "r"), k = val(root, u + "k"), off = val(root, u + "o");
      let P = base.map(([a, b]) => {
        if (mode === "curve") return [a, a * a * 0.9 + b * 0.15];
        if (mode === "none") return [a, b];
        return [a, rr * a + Math.sqrt(Math.max(0, 1 - rr * rr)) * b];
      });
      if (mode === "out") P = P.concat([[3.2, -3.4 * Math.sign(rr || 1)]]);
      return P.map(([a, b]) => [(a + 3) * k, b * 1.5 + 5 + off]);
    }
    function draw() {
      show(root, ["r", "k", "o"], u);
      const P = data(), X = P.map(p => p[0]), Y = P.map(p => p[1]), s = stats(X, Y);
      const pad = (a) => { const lo = Math.min(...a), hi = Math.max(...a), d = (hi - lo) * 0.08 || 1; return [lo - d, hi + d]; }; const xs = pad(X), ys = pad(Y);
      const c = MLP_canvas(cv, 320), F = frame(c, xs, ys, { xl: "x", yl: "y" });
      line(c.ctx, [[F.X(s.mx), F.Y(ys[0])], [F.X(s.mx), F.Y(ys[1])]], c.col("--line"), 1, [3, 3]);
      line(c.ctx, [[F.X(xs[0]), F.Y(s.my)], [F.X(xs[1]), F.Y(s.my)]], c.col("--line"), 1, [3, 3]);
      P.forEach(([x, y]) => dot(c.ctx, F.X(x), F.Y(y), 4, (x - s.mx) * (y - s.my) >= 0 ? c.col("--teal") : c.col("--coral")));
      const a = s.my - s.byx * s.mx; line(c.ctx, [[F.X(xs[0]), F.Y(a + s.byx * xs[0])], [F.X(xs[1]), F.Y(a + s.byx * xs[1])]], c.col("--accent"), 2.5);
      if (Math.abs(s.bxy) > 1e-9) { const yA = ys[0], yB = ys[1]; line(c.ctx, [[F.X(s.mx + s.bxy * (yA - s.my)), F.Y(yA)], [F.X(s.mx + s.bxy * (yB - s.my)), F.Y(yB)]], c.col("--green"), 1.5, [6, 4]); }
      c.ctx.textAlign = "right"; c.ctx.fillStyle = c.col("--accent"); c.ctx.fillText("— regression of y on x", c.w - 16, c.h - 44);
      c.ctx.fillStyle = c.col("--green"); c.ctx.fillText("- - regression of x on y", c.w - 16, c.h - 30); c.ctx.textAlign = "left";
      let note;
      if (mode === "curve") note = "y is completely determined by x, yet Pearson r is close to 0: correlation only measures <b>linear</b> association. Zero correlation does not mean independence.";
      else if (mode === "out") note = `One extreme point shifts Pearson r (uses raw values) but barely moves Spearman ρ (uses ranks). Covariance and both regression slopes move with it.`;
      else if (val(root, u + "k") > 1) note = "Rescaling x multiplied the variance of x by k² and the covariance by k, but correlation and cosine did not change: <b>covariance has units, correlation is unit-free</b>. The slope shrank by 1/k.";
      else if (val(root, u + "o") > 0) note = "Adding a constant to y changes cosine similarity (it measures angle from the origin, without centring) but leaves covariance, correlation and the slope unchanged. Pearson r <b>is</b> the cosine of the <b>centred</b> vectors.";
      else note = "Teal points have (x − x̄)(y − ȳ) &gt; 0 and push covariance up; coral points push it down. Two regression lines exist because predicting y from x is a different question from predicting x from y — they coincide only when |r| = 1.";
      ro.innerHTML = `var(x) = ${fmt(s.vx, 2)} · var(y) = ${fmt(s.vy, 2)} · <b>cov(x, y) = ${fmt(s.cov, 2)}</b><br>Pearson r = <b>${fmt(s.r)}</b> · Spearman ρ = ${fmt(s.rho)} · r² = ${fmt(s.r * s.r)} · cosine (raw) = ${fmt(s.cos)}<br>slope of y on x = ${fmt(s.byx, 3)} (= r·s<sub>y</sub>/s<sub>x</sub>) · slope of x on y = ${fmt(s.bxy, 3)} · product of slopes = r² = ${fmt(s.byx * s.bxy)}<br><span style="font-family:var(--sans);font-size:13.5px">${note}</span>`;
    }
    wireSeg(root, u + "s", v => { mode = v; draw(); }); redraw(root, draw);
  };

  // 2. Cosine similarity vs Euclidean distance between two vectors ---------------------
  D.cosvec = root => {
    const u = uid();
    root.innerHTML += `<div class="controls">${slider(u + "a", "angle of A (°)", 0, 180, 1, 20)}${slider(u + "la", "length of A", 0.5, 5, 0.1, 3)}${slider(u + "b", "angle of B (°)", 0, 180, 1, 50)}${slider(u + "lb", "length of B", 0.5, 5, 0.1, 1.5)}</div><canvas height="300"></canvas><div class="readout"></div>`;
    const cv = root.querySelector("canvas"), ro = root.querySelector(".readout");
    function arrow(c, F, x, y, col, lab) {
      line(c.ctx, [[F.X(0), F.Y(0)], [F.X(x), F.Y(y)]], col, 3);
      const ang = Math.atan2(F.Y(y) - F.Y(0), F.X(x) - F.X(0)); c.ctx.beginPath(); c.ctx.moveTo(F.X(x), F.Y(y));
      c.ctx.lineTo(F.X(x) - 11 * Math.cos(ang - 0.4), F.Y(y) - 11 * Math.sin(ang - 0.4)); c.ctx.lineTo(F.X(x) - 11 * Math.cos(ang + 0.4), F.Y(y) - 11 * Math.sin(ang + 0.4)); c.ctx.closePath(); c.ctx.fillStyle = col; c.ctx.fill();
      c.ctx.font = "bold 13px sans-serif"; c.ctx.fillText(lab, F.X(x) + 6, F.Y(y) - 6);
    }
    function draw() {
      show(root, ["a", "la", "b", "lb"], u);
      const ta = val(root, u + "a") * Math.PI / 180, tb = val(root, u + "b") * Math.PI / 180, la = val(root, u + "la"), lb = val(root, u + "lb");
      const A = [la * Math.cos(ta), la * Math.sin(ta)], B = [lb * Math.cos(tb), lb * Math.sin(tb)];
      const c = MLP_canvas(cv, 300), F = frame(c, [-5.5, 5.5], [-0.3, 5.5], { ticks: false, xl: "feature 1", yl: "feature 2" });
      line(c.ctx, [[F.X(A[0]), F.Y(A[1])], [F.X(B[0]), F.Y(B[1])]], c.col("--muted"), 1.5, [5, 4]);
      arrow(c, F, A[0], A[1], c.col("--accent"), "A"); arrow(c, F, B[0], B[1], c.col("--coral"), "B");
      const dotp = A[0] * B[0] + A[1] * B[1], cos = dotp / (la * lb), eu = Math.hypot(A[0] - B[0], A[1] - B[1]);
      const ang = Math.abs(val(root, u + "a") - val(root, u + "b"));
      ro.innerHTML = `A·B = ${fmt(dotp, 2)} · ‖A‖ = ${fmt(la, 2)} · ‖B‖ = ${fmt(lb, 2)}<br>angle = ${ang}° → <b>cosine similarity = ${fmt(cos)}</b> · Euclidean distance (dashed) = <b>${fmt(eu, 2)}</b><br><span style="font-family:var(--sans);font-size:13.5px">Change only a <b>length</b>: cosine stays the same, Euclidean distance changes. That is why cosine suits text and embeddings — a long document and a short one on the same topic point the same way. Cosine: 1 same direction, 0 unrelated (90°), −1 opposite.</span>`;
    }
    redraw(root, draw);
  };

  // 3. Monte Carlo estimate of π with convergence ---------------------------------------
  D.montecarlo = root => {
    const u = uid();
    root.innerHTML += `<div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:8px"><button class="btn small primary" data-a="100">+100 darts</button><button class="btn small" data-a="1000">+1,000</button><button class="btn small" data-a="10000">+10,000</button><button class="btn small" data-a="reset">Reset</button></div>
      <div class="demo-grid"><div><div class="small">Darts thrown at a 1×1 square</div><canvas height="260"></canvas></div><div><div class="small">Estimate of π as darts accumulate (shaded: ±2 standard errors)</div><canvas height="260"></canvas></div></div><div class="readout"></div>`;
    const [c1, c2] = root.querySelectorAll("canvas"), ro = root.querySelector(".readout");
    let rnd, pts, hist, inside;
    const reset = () => { rnd = lcg(7); pts = []; hist = []; inside = 0; };
    function add(k) { for (let i = 0; i < k; i++) { const x = rnd(), y = rnd(), h = x * x + y * y <= 1; if (h) inside++; if (pts.length < 4000) pts.push([x, y, h]); const n = (hist.n || 0) + 1; hist.n = n; if (n < 50 || n % Math.max(1, Math.floor(n / 200)) === 0) hist.push([n, 4 * inside / n]); } }
    function draw() {
      const n = hist.n || 0;
      const k1 = MLP_canvas(c1, 260), F = frame(k1, [0, 1], [0, 1], { ticks: false });
      const arc = []; for (let t = 0; t <= Math.PI / 2 + 1e-9; t += 0.02) arc.push([F.X(Math.cos(t)), F.Y(Math.sin(t))]); line(k1.ctx, arc, k1.col("--ink"), 1.5);
      pts.forEach(([x, y, h]) => dot(k1.ctx, F.X(x), F.Y(y), 1.8, h ? k1.col("--teal") : k1.col("--coral")));
      const k2 = MLP_canvas(c2, 260), N = Math.max(10, n), G = frame(k2, [0, Math.log10(N)], [2.4, 3.9], { xl: "log₁₀(darts)", yl: "estimate" });
      const band = []; for (let e = 0; e <= Math.log10(N) + 1e-9; e += 0.05) { const m = 10 ** e, se = 4 * Math.sqrt((Math.PI / 4) * (1 - Math.PI / 4) / m); band.push([e, se]); }
      k2.ctx.globalAlpha = 0.15; k2.ctx.fillStyle = k2.col("--accent"); k2.ctx.beginPath(); band.forEach(([e, se], i) => (i ? k2.ctx.lineTo : k2.ctx.moveTo).call(k2.ctx, G.X(e), G.Y(Math.min(3.9, Math.PI + 2 * se)))); band.slice().reverse().forEach(([e, se]) => k2.ctx.lineTo(G.X(e), G.Y(Math.max(2.4, Math.PI - 2 * se)))); k2.ctx.fill(); k2.ctx.globalAlpha = 1;
      line(k2.ctx, [[G.X(0), G.Y(Math.PI)], [G.X(Math.log10(N)), G.Y(Math.PI)]], k2.col("--green"), 1.5, [5, 4]);
      if (hist.length) line(k2.ctx, hist.map(([m, e]) => [G.X(Math.log10(m)), G.Y(Math.max(2.4, Math.min(3.9, e)))]), k2.col("--coral"), 2);
      const est = n ? 4 * inside / n : NaN, se = n ? 4 * Math.sqrt((inside / n) * (1 - inside / n) / n) : NaN;
      ro.innerHTML = n ? `darts: ${n.toLocaleString()} · inside the quarter circle: ${inside.toLocaleString()} → π ≈ 4 × ${fmt(inside / n, 4)} = <b>${fmt(est, 4)}</b> (true 3.1416) · standard error ≈ ${fmt(se, 4)}<br><span style="font-family:var(--sans);font-size:13.5px">The share of darts inside the quarter circle estimates its area, π/4. Monte Carlo error shrinks like 1/√n: 100× more darts for one more correct digit. This is simulation — the rule (uniform darts) is known and we compute its consequences; no data are learned from.</span>` : "Throw some darts.";
    }
    root.addEventListener("click", e => { const a = e.target.dataset && e.target.dataset.a; if (!a) return; if (a === "reset") reset(); else add(+a); draw(); });
    reset(); window.addEventListener("resize", draw); window.addEventListener("mlp-theme", draw); draw();
  };

  // 4. Regression to the mean ------------------------------------------------------------
  D.regmean = root => {
    const u = uid();
    root.innerHTML += `<div class="controls">${slider(u + "s", "share of score that is true skill (test reliability)", 0.1, 0.95, 0.05, 0.5)}${slider(u + "t", "select the top % on test 1", 5, 50, 5, 10)}</div><canvas height="280"></canvas><div class="readout"></div>`;
    const cv = root.querySelector("canvas"), ro = root.querySelector(".readout");
    const base = (() => { const r = lcg(5); return Array.from({ length: 400 }, () => [normal(r), normal(r), normal(r)]); })();
    function draw() {
      show(root, ["s", "t"], u);
      const rel = val(root, u + "s"), top = val(root, u + "t") / 100, a = Math.sqrt(rel), b = Math.sqrt(1 - rel);
      const P = base.map(([sk, e1, e2]) => [50 + 10 * (a * sk + b * e1), 50 + 10 * (a * sk + b * e2)]);
      const cut = P.map(p => p[0]).sort((x, y) => y - x)[Math.max(0, Math.floor(P.length * top) - 1)];
      const sel = P.filter(p => p[0] >= cut), m1 = mean(sel.map(p => p[0])), m2 = mean(sel.map(p => p[1]));
      const c = MLP_canvas(cv, 280), F = frame(c, [15, 85], [15, 85], { xl: "test 1 score", yl: "test 2 score" });
      line(c.ctx, [[F.X(15), F.Y(15)], [F.X(85), F.Y(85)]], c.col("--line"), 1, [4, 4]);
      P.forEach(([x, y]) => dot(c.ctx, F.X(x), F.Y(y), 2.6, x >= cut ? c.col("--coral") : c.col("--muted")));
      line(c.ctx, [[F.X(cut), F.Y(15)], [F.X(cut), F.Y(85)]], c.col("--coral"), 1, [3, 3]);
      ro.innerHTML = `selected (coral): average ${fmt(m1, 1)} on test 1 → <b>${fmt(m2, 1)}</b> on test 2 (population mean 50). Expected: 50 + reliability × (${fmt(m1, 1)} − 50) = ${fmt(50 + rel * (m1 - 50), 1)}<br><span style="font-family:var(--sans);font-size:13.5px">Nobody got worse — the top scorers were partly lucky on test 1 and the luck doesn't repeat. The lower the reliability (more noise), the stronger the pull back toward the mean. Any “intervention on the worst performers” will look effective for the same reason unless you keep a control group.</span>`;
    }
    redraw(root, draw);
  };
})();
