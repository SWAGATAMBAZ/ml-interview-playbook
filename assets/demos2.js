/* ML Interview Playbook — second set of interactive demos (needs demos.js loaded first) */
(function () {
  const H = window.MLP_D; if (!H) return;
  const { D, frame, dot, line, slider, seg, wireSeg, uid, gauss, fmt } = H;
  const lcg = seed => { let s = seed; return () => { s = (s * 16807) % 2147483647; return (s - 1) / 2147483646; }; };
  const val = (root, id) => +root.querySelector("#" + id).value;
  const show = (root, ids, u) => ids.forEach(k => { const e = root.querySelector("#" + u + k + "v"); if (e) e.textContent = root.querySelector("#" + u + k).value; });
  const redraw = (root, draw) => { root.querySelectorAll("input").forEach(i => i.addEventListener("input", draw)); window.addEventListener("resize", draw); window.addEventListener("mlp-theme", draw); draw(); };
  const sig = z => 1 / (1 + Math.exp(-z));

  // 1. OLS: squares of residuals ------------------------------------------------
  D.olsfit = root => {
    const u = uid();
    root.innerHTML += `<div class="controls">${slider(u + "m", "your slope", -1, 3, 0.05, 0.2)}${slider(u + "b", "your intercept", -2, 8, 0.1, 3)}</div><div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:8px"><button class="btn small primary" data-a="best">Snap to best fit (OLS)</button><button class="btn small" data-a="out">Add an outlier</button><button class="btn small" data-a="reset">Reset points</button></div><canvas height="300"></canvas><div class="readout"></div>`;
    const cv = root.querySelector("canvas"), ro = root.querySelector(".readout");
    let pts; const init = () => { pts = [[1, 2], [2, 3], [3, 5], [4, 4], [5, 6]]; }; init();
    const ols = () => { const n = pts.length, mx = pts.reduce((s, p) => s + p[0], 0) / n, my = pts.reduce((s, p) => s + p[1], 0) / n; let a = 0, b = 0; pts.forEach(([x, y]) => { a += (x - mx) * (y - my); b += (x - mx) ** 2; }); const m = a / b; return [m, my - m * mx]; };
    const sse = (m, b) => pts.reduce((s, [x, y]) => s + (y - m * x - b) ** 2, 0);
    function draw() {
      const m = val(root, u + "m"), b = val(root, u + "b"); show(root, ["m", "b"], u);
      const c = MLP_canvas(cv, 300), F = frame(c, [0, 7], [0, 9], { xl: "x", yl: "y" });
      pts.forEach(([x, y]) => { const yh = m * x + b, side = Math.abs(F.Y(y) - F.Y(yh)); c.ctx.globalAlpha = 0.22; c.ctx.fillStyle = c.col("--coral"); c.ctx.fillRect(F.X(x), Math.min(F.Y(y), F.Y(yh)), side, side); c.ctx.globalAlpha = 1; line(c.ctx, [[F.X(x), F.Y(y)], [F.X(x), F.Y(yh)]], c.col("--coral"), 1.5); });
      line(c.ctx, [[F.X(0), F.Y(b)], [F.X(7), F.Y(m * 7 + b)]], c.col("--accent"), 2.5);
      const [bm, bb] = ols(); line(c.ctx, [[F.X(0), F.Y(bb)], [F.X(7), F.Y(bm * 7 + bb)]], c.col("--green"), 1.5, [5, 4]);
      pts.forEach(([x, y]) => dot(c.ctx, F.X(x), F.Y(y), 5, c.col("--ink")));
      cv._F = F;
      ro.innerHTML = `your line: ŷ = ${fmt(b, 2)} + ${fmt(m, 2)}·x → sum of squared errors = <b>${fmt(sse(m, b), 2)}</b><br>best possible (dashed green): ŷ = ${fmt(bb, 2)} + ${fmt(bm, 2)}·x → SSE = <b>${fmt(sse(bm, bb), 2)}</b><br><span style="font-family:var(--sans);font-size:13.5px">Each pink square's area is one squared residual; OLS is the line that makes the total pink area as small as possible. Click the chart to add points; add an outlier and see how hard one big square pulls the line.</span>`;
    }
    cv.addEventListener("click", e => { const r = cv.getBoundingClientRect(), F = cv._F; pts.push([F.iX(e.clientX - r.left), F.iY(e.clientY - r.top)]); draw(); });
    root.addEventListener("click", e => { const a = e.target.dataset && e.target.dataset.a; if (!a) return; if (a === "reset") init(); if (a === "out") pts.push([6.5, 0.5]); if (a === "best") { const [m, b] = ols(); root.querySelector("#" + u + "m").value = Math.max(-1, Math.min(3, m)); root.querySelector("#" + u + "b").value = Math.max(-2, Math.min(8, b)); } draw(); });
    redraw(root, draw);
  };

  // 2. Loss function shapes -------------------------------------------------------
  D.losses = root => {
    const u = uid();
    root.innerHTML += `<div style="margin:6px 0 10px">${seg(u + "s", [["reg", "Regression losses"], ["cls", "Classification losses"]], "reg")}</div><canvas height="270"></canvas><div class="readout" style="font-family:var(--sans)"></div>`;
    const cv = root.querySelector("canvas"), ro = root.querySelector(".readout"); let mode = "reg";
    function draw() {
      const c = MLP_canvas(cv, 270);
      const sets = mode === "reg" ? [["squared (MSE)", "--coral", e => e * e], ["absolute (MAE)", "--accent", e => Math.abs(e)], ["Huber δ=1", "--green", e => Math.abs(e) <= 1 ? 0.5 * e * e : Math.abs(e) - 0.5]]
        : [["0-1 loss", "--muted", m => m > 0 ? 0 : 1], ["log-loss", "--accent", m => Math.log(1 + Math.exp(-m))], ["hinge (SVM)", "--green", m => Math.max(0, 1 - m)], ["exponential (AdaBoost)", "--coral", m => Math.exp(-m)]];
      const F = frame(c, [-3, 3], [0, mode === "reg" ? 6 : 4], { xl: mode === "reg" ? "error  y − ŷ" : "margin  y·f(x)   (>0 = correct)", yl: "loss" });
      sets.forEach(([name, colr, f], i) => { const p = []; for (let x = -3; x <= 3.001; x += 0.02) p.push([F.X(x), F.Y(Math.min(f(x), mode === "reg" ? 6 : 4))]); line(c.ctx, p, c.col(colr), 2.2); c.ctx.fillStyle = c.col(colr); c.ctx.font = "12px sans-serif"; c.ctx.textAlign = "left"; c.ctx.fillText("■ " + name, F.X(-0.9), F.Y((mode === "reg" ? 5.6 : 3.75) - i * (mode === "reg" ? 0.45 : 0.3))); });
      ro.innerHTML = mode === "reg" ? "Squared loss grows quadratically, so one outlier with error 3 costs 9× an error of 1 — it predicts the <b>mean</b>. Absolute loss grows linearly — it predicts the <b>median</b> and shrugs off outliers. Huber is squared near zero (smooth) and linear in the tails (robust)."
        : "All three smooth curves are convex stand-ins for the 0-1 loss that we can't optimise. Log-loss never reaches zero (keeps pushing probabilities); hinge is exactly zero once the margin exceeds 1 (only support vectors matter); exponential explodes for confident mistakes (why AdaBoost hates label noise).";
    }
    wireSeg(root, u + "s", v => { mode = v; draw(); }); redraw(root, draw);
  };

  // 3. 2-D optimiser playground -----------------------------------------------------
  D.opt2d = root => {
    const u = uid();
    root.innerHTML += `<div style="margin:6px 0 10px">${seg(u + "o", [["gd", "Gradient descent"], ["mom", "+ Momentum 0.9"], ["adam", "Adam"]], "gd")}</div>
      <div class="controls">${slider(u + "lr", "learning rate (× 1/steepest curvature)", 0.02, 2.2, 0.02, 0.3)}${slider(u + "k", "condition number (unscaled features →)", 1, 60, 1, 20)}${slider(u + "s", "gradient noise (small batch →)", 0, 3, 0.1, 0)}</div>
      <label class="small"><input type="checkbox" id="${u}d"> decay the learning rate over time (η / (1 + 0.05·t))</label><canvas height="300"></canvas><div class="readout"></div>`;
    const cv = root.querySelector("canvas"), ro = root.querySelector(".readout"); let opt = "gd";
    const Z = []; for (let i = 0; i < 400; i++) Z.push(gauss());
    function draw() {
      const lrm = val(root, u + "lr"), k = val(root, u + "k"), ns = val(root, u + "s"), decay = root.querySelector("#" + u + "d").checked; show(root, ["lr", "k", "s"], u);
      const a = k, b = 1, c = MLP_canvas(cv, 300), F = frame(c, [-2.2, 2.2], [-2.6, 2.6], { ticks: false, xl: "w₁ (steep direction)", yl: "w₂ (flat direction)" });
      const L = (x, y) => 0.5 * (a * x * x + b * y * y);
      [0.05, 0.25, 0.7, 1.5, 2.8].forEach(lv => { const p = []; for (let t = 0; t <= 6.3; t += 0.05) p.push([F.X(Math.sqrt(2 * lv / a) * Math.cos(t)), F.Y(Math.sqrt(2 * lv / b) * Math.sin(t))]); line(c.ctx, p, c.col("--line"), 1); });
      let x = 0.35 * Math.sqrt(20 / a) + 0.0, y = 2.3; x = Math.min(2, Math.max(x, 0.3)); let vx = 0, vy = 0, mx = 0, my = 0, sx = 0, sy = 0; const path = [[x, y]]; let diverged = false;
      for (let t = 1; t <= 150; t++) {
        let lr = lrm / a; if (decay) lr /= 1 + 0.05 * t;
        const gx = a * x + ns * Z[2 * t] * Math.sqrt(a) * 0.35, gy = b * y + ns * Z[2 * t + 1] * 0.35;
        if (opt === "gd") { x -= lr * gx; y -= lr * gy; }
        else if (opt === "mom") { vx = 0.9 * vx + gx; vy = 0.9 * vy + gy; x -= lr * vx; y -= lr * vy; }
        else { const al = lrm * 0.12 / (decay ? 1 + 0.05 * t : 1); mx = 0.9 * mx + 0.1 * gx; my = 0.9 * my + 0.1 * gy; sx = 0.999 * sx + 0.001 * gx * gx; sy = 0.999 * sy + 0.001 * gy * gy; x -= al * (mx / (1 - 0.9 ** t)) / (Math.sqrt(sx / (1 - 0.999 ** t)) + 1e-8); y -= al * (my / (1 - 0.9 ** t)) / (Math.sqrt(sy / (1 - 0.999 ** t)) + 1e-8); }
        if (!isFinite(x) || Math.abs(x) > 50 || Math.abs(y) > 50) { diverged = true; break; }
        path.push([x, y]);
      }
      const cl = v => Math.max(-3, Math.min(3, v));
      line(c.ctx, path.map(p => [F.X(cl(p[0])), F.Y(cl(p[1]))]), c.col("--coral"), 1.4);
      path.forEach((p, i) => { if (i % 3 === 0 || i === path.length - 1) dot(c.ctx, F.X(cl(p[0])), F.Y(cl(p[1])), i === path.length - 1 ? 5 : 2, i === path.length - 1 ? c.col("--accent") : c.col("--coral")); });
      dot(c.ctx, F.X(0), F.Y(0), 4, c.col("--green"));
      const last = path[path.length - 1], tail = path.slice(-30), wob = Math.sqrt(tail.reduce((s, p) => s + L(p[0], p[1]), 0) / tail.length), fin = L(last[0], last[1]);
      let msg;
      if (diverged) msg = "💥 <b>Diverged.</b> The step is larger than 2 / (steepest curvature), so every step overshoots further in the steep direction.";
      else if (ns > 0 && wob > 0.02 && !decay) msg = "🌀 <b>Roaming around the minimum without settling.</b> The learning rate is fine for the curvature, but noisy mini-batch gradients keep kicking the weights. A constant η leaves a noise floor ∝ η × gradient variance. Fix: decay the learning rate, use a larger batch, or average the weights.";
      else if (opt === "gd" && lrm > 1) msg = "〰 <b>Zig-zagging across the steep direction</b> while crawling along the flat one. Classic ill-conditioning: scale the features, add momentum, or use an adaptive optimiser.";
      else if (fin > 0.01) msg = `🐢 <b>Still far from the minimum after 150 steps</b> (loss ${fmt(fin, 3)}). The step is limited by the steep direction, so the flat direction (curvature ${k}× smaller) barely moves. Fix the conditioning (standardise features) rather than raising η.`;
      else msg = "✅ <b>Converged.</b>";
      ro.innerHTML = `final loss ${fmt(fin, 4)} · average loss over the last 30 steps ${fmt(wob * wob, 4)}<br><span style="font-family:var(--sans);font-size:13.5px">${msg}</span>`;
    }
    wireSeg(root, u + "o", v => { opt = v; draw(); }); redraw(root, draw);
  };

  // 4. L1 vs L2 geometry --------------------------------------------------------------
  D.l1l2 = root => {
    const u = uid();
    root.innerHTML += `<div style="margin:6px 0 10px">${seg(u + "p", [["l1", "L1 (Lasso) — diamond"], ["l2", "L2 (Ridge) — circle"]], "l1")}</div><div class="controls">${slider(u + "a", "OLS solution β̂₁", -3, 3, 0.1, 2.2)}${slider(u + "b", "OLS solution β̂₂", -3, 3, 0.1, 0.9)}${slider(u + "t", "budget t (smaller = stronger penalty)", 0.2, 3, 0.05, 1.2)}${slider(u + "r", "feature correlation", -0.8, 0.8, 0.05, 0)}</div><canvas height="330"></canvas><div class="readout"></div>`;
    const cv = root.querySelector("canvas"), ro = root.querySelector(".readout"); let pen = "l1";
    function draw() {
      const a = val(root, u + "a"), b = val(root, u + "b"), t = val(root, u + "t"), r = val(root, u + "r"); show(root, ["a", "b", "t", "r"], u);
      const c = MLP_canvas(cv, 330), hx = 3.3 * (c.w - 50) / (330 - 38), F = frame(c, [-hx, hx], [-3.3, 3.3], { xl: "β₁", yl: "β₂", ticks: false });
      const loss = (x, y) => { const dx = x - a, dy = y - b; return dx * dx + 2 * r * dx * dy + dy * dy; };
      const X0 = F.X(0), Y0 = F.Y(0); line(c.ctx, [[F.X(-9), Y0], [F.X(9), Y0]], c.col("--line"), 1); line(c.ctx, [[X0, F.Y(-9)], [X0, F.Y(9)]], c.col("--line"), 1);
      const bd = []; for (let k = 0; k <= 720; k++) { const th = k / 720 * 2 * Math.PI, cx = Math.cos(th), sy = Math.sin(th); const s = pen === "l1" ? t / (Math.abs(cx) + Math.abs(sy)) : t; bd.push([s * cx, s * sy]); }
      c.ctx.beginPath(); bd.forEach(([x, y], i) => i ? c.ctx.lineTo(F.X(x), F.Y(y)) : c.ctx.moveTo(F.X(x), F.Y(y))); c.ctx.closePath(); c.ctx.globalAlpha = 0.25; c.ctx.fillStyle = c.col("--accent"); c.ctx.fill(); c.ctx.globalAlpha = 1; c.ctx.strokeStyle = c.col("--accent"); c.ctx.lineWidth = 2; c.ctx.stroke();
      const inside = pen === "l1" ? Math.abs(a) + Math.abs(b) <= t : Math.hypot(a, b) <= t;
      let sol = [a, b]; if (!inside) { let best = Infinity; bd.forEach(p => { const l = loss(p[0], p[1]); if (l < best) { best = l; sol = p; } }); }
      const cs = loss(sol[0], sol[1]);
      [cs, cs * 2.2 + 0.15, cs * 4 + 0.5].forEach((lv, i) => { if (lv <= 0) return; const p = []; for (let k = 0; k <= 200; k++) { const th = k / 200 * 2 * Math.PI, dx = Math.cos(th), dy = Math.sin(th), q = dx * dx + 2 * r * dx * dy + dy * dy, rad = Math.sqrt(lv / q); p.push([F.X(a + rad * dx), F.Y(b + rad * dy)]); } line(c.ctx, p, c.col("--coral"), i ? 1 : 2, i ? [4, 4] : null); });
      dot(c.ctx, F.X(a), F.Y(b), 5, c.col("--coral")); c.ctx.fillStyle = c.col("--coral"); c.ctx.font = "12px sans-serif"; c.ctx.fillText("OLS β̂", F.X(a) + 8, F.Y(b) - 6);
      dot(c.ctx, F.X(sol[0]), F.Y(sol[1]), 6, c.col("--green"), c.col("--ink"));
      const z = v => Math.abs(v) < 0.012 ? 0 : v, s1 = z(sol[0]), s2 = z(sol[1]), zeros = (s1 === 0) + (s2 === 0);
      ro.innerHTML = `penalised solution (green): β₁ = ${fmt(s1, 2)}, β₂ = ${fmt(s2, 2)} ${zeros ? "→ <b>" + zeros + " coefficient exactly 0</b>" : ""}<br><span style="font-family:var(--sans);font-size:13.5px">${inside ? "The OLS solution already fits inside the budget, so the penalty changes nothing." : pen === "l1" ? "The red error contour grows from the OLS point until it first touches the blue diamond. Diamonds have corners on the axes, and corners stick out — so the first touch is very often a corner, where one coefficient is exactly zero. Shrink the budget and watch β₂ snap to 0." : "The circle has no corners, so the contour touches it at a generic point: both coefficients shrink toward zero but neither becomes exactly zero."}</span>`;
    }
    wireSeg(root, u + "p", v => { pen = v; draw(); }); redraw(root, draw);
  };

  // 5. Logistic regression trained by gradient descent in 2-D ---------------------------
  D.logit2d = root => {
    const u = uid();
    root.innerHTML += `<div class="controls">${slider(u + "lr", "learning rate", 0.05, 3, 0.05, 0.5)}</div><div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:8px"><button class="btn small primary" data-a="1">1 gradient step</button><button class="btn small" data-a="20">20 steps</button><button class="btn small" data-a="300">300 steps</button><button class="btn small" data-a="r">Reset weights</button><button class="btn small" data-a="sep">Toggle: overlapping ↔ perfectly separable</button></div><canvas height="300"></canvas><div class="readout"></div>`;
    const cv = root.querySelector("canvas"), ro = root.querySelector(".readout"); let w = [0, 0], b = 0, steps = 0, sep = false, P = [];
    function data() { const r = lcg(11); const g = () => { let a = 0; for (let i = 0; i < 6; i++) a += r(); return (a - 3) * 1.4; }; P = []; const d = sep ? 1.9 : 0.9; for (let i = 0; i < 50; i++) { P.push([-d + g() * 0.75, -d * 0.6 + g() * 0.75, 0]); P.push([d + g() * 0.75, d * 0.6 + g() * 0.75, 1]); } }
    const lossf = () => P.reduce((s, [x, y, l]) => { const p = Math.min(1 - 1e-12, Math.max(1e-12, sig(w[0] * x + w[1] * y + b))); return s - (l ? Math.log(p) : Math.log(1 - p)); }, 0) / P.length;
    function step(n) { const lr = val(root, u + "lr"); for (let k = 0; k < n; k++) { let g0 = 0, g1 = 0, gb = 0; P.forEach(([x, y, l]) => { const e = sig(w[0] * x + w[1] * y + b) - l; g0 += e * x; g1 += e * y; gb += e; }); w[0] -= lr * g0 / P.length; w[1] -= lr * g1 / P.length; b -= lr * gb / P.length; steps++; } }
    function draw() {
      show(root, ["lr"], u); const c = MLP_canvas(cv, 300), F = frame(c, [-4.5, 4.5], [-3.6, 3.6], { ticks: false, xl: "x₁", yl: "x₂" }), st = 8;
      for (let px = F.p.l; px < c.w - F.p.r; px += st) for (let py = F.p.t; py < c.h - F.p.b; py += st) { const p = sig(w[0] * F.iX(px + st / 2) + w[1] * F.iY(py + st / 2) + b); c.ctx.globalAlpha = Math.abs(p - 0.5) * 0.55; c.ctx.fillStyle = p > 0.5 ? c.col("--coral") : c.col("--accent"); c.ctx.fillRect(px, py, st, st); } c.ctx.globalAlpha = 1;
      if (Math.abs(w[1]) > 1e-6 || Math.abs(w[0]) > 1e-6) { const pts = Math.abs(w[1]) > Math.abs(w[0]) ? [[-4.5, (-b + 4.5 * w[0]) / w[1]], [4.5, (-b - 4.5 * w[0]) / w[1]]] : [[(-b + 3.6 * w[1]) / w[0], -3.6], [(-b - 3.6 * w[1]) / w[0], 3.6]]; line(c.ctx, pts.map(p => [F.X(p[0]), F.Y(Math.max(-3.6, Math.min(3.6, p[1])))]), c.col("--ink"), 2); }
      P.forEach(([x, y, l]) => dot(c.ctx, F.X(x), F.Y(y), 3.6, l ? c.col("--coral") : c.col("--accent"), c.col("--panel")));
      const acc = P.filter(([x, y, l]) => (sig(w[0] * x + w[1] * y + b) >= 0.5) == l).length / P.length, nw = Math.hypot(w[0], w[1]);
      ro.innerHTML = `step ${steps}: w = [${fmt(w[0], 2)}, ${fmt(w[1], 2)}], b = ${fmt(b, 2)} · ‖w‖ = ${fmt(nw, 2)} · log-loss = <b>${fmt(lossf(), 4)}</b> · accuracy = ${fmt(acc * 100, 0)}%<br><span style="font-family:var(--sans);font-size:13.5px">${steps === 0 ? "All weights are 0, so every prediction is 0.5 and the loss is ln 2 = 0.693. Press a step button: each step moves w by −η × average of (p − y)·x." : sep ? "With perfectly separable data the loss keeps falling and ‖w‖ keeps growing forever — the sigmoid gets steeper and steeper. There is no finite optimum: this is why unregularised logistic regression “fails to converge” on separable data, and why L2 regularisation fixes it." : "The black line is the decision boundary w·x + b = 0; colour intensity is confidence. With overlapping classes the loss levels off at a finite minimum and ‖w‖ stabilises."}</span>`;
    }
    root.addEventListener("click", e => { const a = e.target.dataset && e.target.dataset.a; if (!a) return; if (a === "r") { w = [0, 0]; b = 0; steps = 0; } else if (a === "sep") { sep = !sep; data(); w = [0, 0]; b = 0; steps = 0; } else step(+a); draw(); });
    data(); redraw(root, draw);
  };

  // 6. Decision tree / random forest decision regions -------------------------------------
  D.treeviz = root => {
    const u = uid(), forest = root.dataset.mode === "forest";
    root.innerHTML += `<div class="controls">${slider(u + "d", "max_depth", 1, 10, 1, forest ? 10 : 3)}${slider(u + "l", "min_samples_leaf", 1, 25, 1, 1)}${forest ? slider(u + "n", "n_estimators (trees)", 1, 60, 1, 1) : ""}</div>${forest ? `<label class="small"><input type="checkbox" id="${u}f" checked> random feature at each split (max_features = 1 of 2) + bootstrap rows</label>` : ""}<canvas height="320"></canvas><div class="readout"></div>`;
    const cv = root.querySelector("canvas"), ro = root.querySelector(".readout");
    const r0 = lcg(5), g0 = () => { let a = 0; for (let i = 0; i < 6; i++) a += r0(); return (a - 3) * 1.41; };
    const mk = n => { const A = [], B = []; for (let i = 0; i < n; i++) { const c = i % 2, t = r0() * Math.PI; let x = c ? 1 - Math.cos(t) : Math.cos(t), y = c ? 0.5 - Math.sin(t) : Math.sin(t); x += g0() * 0.2; y += g0() * 0.2; A.push([x, y]); B.push(r0() < 0.06 ? 1 - c : c); } return [A, B]; };
    const [X, Y] = mk(180), [Xt, Yt] = mk(500);
    function build(idx, depth, o, rr) {
      const n = idx.length; let n1 = 0; idx.forEach(i => n1 += Y[i]);
      if (depth >= o.depth || n < 2 * o.leaf || n1 === 0 || n1 === n) return { p: n1 / n, n };
      let best = null; const feats = o.rf ? [rr() < 0.5 ? 0 : 1] : [0, 1];
      for (const f of feats) { const s = idx.slice().sort((a, b) => X[a][f] - X[b][f]); let l1 = 0;
        for (let k = 1; k < n; k++) { l1 += Y[s[k - 1]]; if (X[s[k]][f] === X[s[k - 1]][f] || k < o.leaf || n - k < o.leaf) continue; const r1 = n1 - l1, pl = l1 / k, pr = r1 / (n - k), g = k * 2 * pl * (1 - pl) + (n - k) * 2 * pr * (1 - pr); if (!best || g < best.g) best = { g, f, t: (X[s[k]][f] + X[s[k - 1]][f]) / 2, s, k }; } }
      if (!best) return { p: n1 / n, n };
      return { f: best.f, t: best.t, l: build(best.s.slice(0, best.k), depth + 1, o, rr), r: build(best.s.slice(best.k), depth + 1, o, rr) };
    }
    const pred = (t, x) => { while (t.p === undefined) t = x[t.f] <= t.t ? t.l : t.r; return t.p; };
    const leaves = t => t.p !== undefined ? 1 : leaves(t.l) + leaves(t.r);
    function draw() {
      const depth = val(root, u + "d"), leaf = val(root, u + "l"), nt = forest ? val(root, u + "n") : 1, rf = forest && root.querySelector("#" + u + "f").checked; show(root, ["d", "l", "n"], u);
      const rr = lcg(99), trees = [], all = X.map((_, i) => i);
      for (let k = 0; k < nt; k++) { const idx = forest && rf ? all.map(() => Math.floor(rr() * X.length)) : all; trees.push(build(idx, 0, { depth, leaf, rf }, rr)); }
      const P = x => trees.reduce((s, t) => s + pred(t, x), 0) / trees.length;
      const c = MLP_canvas(cv, 320), F = frame(c, [-1.9, 2.9], [-1.5, 2.0], { ticks: false, xl: "x₁", yl: "x₂" }), st = 6;
      for (let px = F.p.l; px < c.w - F.p.r; px += st) for (let py = F.p.t; py < c.h - F.p.b; py += st) { const p = P([F.iX(px + st / 2), F.iY(py + st / 2)]); c.ctx.globalAlpha = 0.1 + Math.abs(p - 0.5) * 0.45; c.ctx.fillStyle = p > 0.5 ? c.col("--amber") : c.col("--accent"); c.ctx.fillRect(px, py, st, st); } c.ctx.globalAlpha = 1;
      X.forEach((x, i) => dot(c.ctx, F.X(x[0]), F.Y(x[1]), 3.4, Y[i] ? c.col("--amber") : c.col("--accent"), c.col("--panel")));
      const acc = (A, B) => A.filter((x, i) => (P(x) > 0.5 ? 1 : 0) === B[i]).length / A.length, tr = acc(X, Y), te = acc(Xt, Yt);
      const verdict = tr - te > 0.07 ? "🔺 Large train–test gap: the model is memorising noise (those tiny islands around mislabelled points). " + (forest ? "Add trees — averaging many different overfit trees smooths the islands away." : "Reduce max_depth or raise min_samples_leaf.") : te < 0.86 ? "🔻 Underfitting: too few splits to follow the two moons." : "✅ Good generalisation.";
      ro.innerHTML = `${forest ? nt + " tree(s)" : leaves(trees[0]) + " leaves"} · train accuracy ${fmt(tr * 100, 1)}% · test accuracy <b>${fmt(te * 100, 1)}%</b> (500 fresh points; ~6% of labels are flipped noise)<br><span style="font-family:var(--sans);font-size:13.5px">${verdict} Every boundary is made of horizontal and vertical pieces because each split tests one feature against one threshold.</span>`;
    }
    redraw(root, draw);
  };

  // 7. AdaBoost round by round ---------------------------------------------------------------
  D.adaboost = root => {
    root.innerHTML += `<div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:8px"><button class="btn small primary" data-a="1">+ 1 round</button><button class="btn small" data-a="10">+ 10 rounds</button><button class="btn small" data-a="r">Reset</button></div><canvas height="300"></canvas><div class="readout"></div>`;
    const cv = root.querySelector("canvas"), ro = root.querySelector(".readout"), r0 = lcg(21), P = [];
    for (let i = 0; i < 70; i++) { const x = r0() * 4 - 2, y = r0() * 4 - 2; P.push([x, y, (x * 0.9 + y + (r0() - 0.5) * 0.5 > 0.1) ? 1 : -1]); }
    let w, st, last;
    const reset = () => { w = P.map(() => 1 / P.length); st = []; last = null; }; reset();
    function round() {
      let best = null;
      for (let f = 0; f < 2; f++) { const vals = [...new Set(P.map(p => p[f]))].sort((a, b) => a - b); for (let k = 0; k < vals.length - 1; k++) { const t = (vals[k] + vals[k + 1]) / 2; for (const s of [1, -1]) { let e = 0; P.forEach((p, i) => { if ((p[f] > t ? s : -s) !== p[2]) e += w[i]; }); if (!best || e < best.e) best = { e, f, t, s }; } } }
      const eps = Math.max(best.e, 1e-9), al = 0.5 * Math.log((1 - eps) / eps); let z = 0;
      w = w.map((wi, i) => { const h = P[i][best.f] > best.t ? best.s : -best.s, v = wi * Math.exp(-al * P[i][2] * h); z += v; return v; }).map(v => v / z);
      st.push({ ...best, al }); last = { eps, al, f: best.f, t: best.t };
    }
    const Fx = (x, y) => st.reduce((s, h) => s + h.al * ((h.f ? y : x) > h.t ? h.s : -h.s), 0);
    function draw() {
      const c = MLP_canvas(cv, 300), F = frame(c, [-2.2, 2.2], [-2.2, 2.2], { ticks: false, xl: "x₁", yl: "x₂" }), q = 7;
      if (st.length) for (let px = F.p.l; px < c.w - F.p.r; px += q) for (let py = F.p.t; py < c.h - F.p.b; py += q) { c.ctx.globalAlpha = 0.16; c.ctx.fillStyle = Fx(F.iX(px + q / 2), F.iY(py + q / 2)) > 0 ? c.col("--coral") : c.col("--accent"); c.ctx.fillRect(px, py, q, q); } c.ctx.globalAlpha = 1;
      if (last) { const pts = last.f ? [[-2.2, last.t], [2.2, last.t]] : [[last.t, -2.2], [last.t, 2.2]]; line(c.ctx, pts.map(p => [F.X(p[0]), F.Y(p[1])]), c.col("--ink"), 2, [6, 4]); }
      P.forEach((p, i) => dot(c.ctx, F.X(p[0]), F.Y(p[1]), 2.5 + Math.min(14, Math.sqrt(w[i] * P.length) * 2.6), p[2] > 0 ? c.col("--coral") : c.col("--accent"), c.col("--panel")));
      const err = st.length ? P.filter(p => (Fx(p[0], p[1]) > 0 ? 1 : -1) !== p[2]).length / P.length : 0.5;
      ro.innerHTML = st.length ? `round ${st.length}: stump “${last.f ? "x₂" : "x₁"} > ${fmt(last.t, 2)}” · weighted error ε = ${fmt(last.eps, 3)} · vote weight α = ½·ln((1−ε)/ε) = ${fmt(last.al, 3)} · ensemble training error = <b>${fmt(err * 100, 1)}%</b><br><span style="font-family:var(--sans);font-size:13.5px">Dot size = sample weight. Points the last stump got wrong have grown, so the next stump is forced to deal with them. The shaded region is the weighted vote of all stumps: a staircase approximating the diagonal boundary no single stump can draw.</span>` : `<span style="font-family:var(--sans)">All 70 points start with equal weight 1/70. Press “+ 1 round” to fit the first decision stump (a one-split tree).</span>`;
    }
    root.addEventListener("click", e => { const a = e.target.dataset && e.target.dataset.a; if (!a) return; if (a === "r") reset(); else for (let i = 0; i < +a; i++) round(); draw(); });
    redraw(root, draw);
  };

  // 8. Naive Bayes spam calculator -------------------------------------------------------------
  D.nbcalc = root => {
    const u = uid(), W = [["free", 30, 6], ["win", 24, 3], ["urgent", 20, 9], ["crypto", 12, 0], ["meeting", 2, 36], ["project", 4, 30], ["invoice", 0, 15]], NS = 40, NH = 60;
    root.innerHTML += `<div class="small" style="margin-bottom:6px">Training set: 40 spam and 60 normal (“ham”) emails. Tick the words that appear in the new email:</div><div id="${u}w" style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px">${W.map((x, i) => `<label class="btn small" style="cursor:pointer"><input type="checkbox" data-i="${i}" ${i === 0 || i === 2 ? "checked" : ""}> ${x[0]}</label>`).join("")}</div><div class="controls">${slider(u + "a", "Laplace smoothing α", 0, 2, 0.1, 1)}${slider(u + "p", "prior P(spam)", 0.05, 0.95, 0.05, 0.4)}</div><div class="table-wrap"><table id="${u}t"></table></div><div class="readout"></div>`;
    const ro = root.querySelector(".readout");
    function draw() {
      const al = val(root, u + "a"), pr = val(root, u + "p"); show(root, ["a", "p"], u);
      const on = [...root.querySelectorAll("#" + u + "w input")].map(i => i.checked); let ls = Math.log(pr), lh = Math.log(1 - pr), zs = false, zh = false;
      let rows = `<tr><th>word</th><th>in email?</th><th>spam emails with it</th><th>ham emails with it</th><th>P(evidence | spam)</th><th>P(evidence | ham)</th><th>pushes toward</th></tr>`;
      W.forEach(([wd, cs, ch], i) => { let ps = (cs + al) / (NS + 2 * al), ph = (ch + al) / (NH + 2 * al); if (!on[i]) { ps = 1 - ps; ph = 1 - ph; } if (ps <= 0) zs = true; else ls += Math.log(ps); if (ph <= 0) zh = true; else lh += Math.log(ph);
        rows += `<tr><td><code>${wd}</code></td><td>${on[i] ? "yes" : "no"}</td><td>${cs}/40</td><td>${ch}/60</td><td>${fmt(ps, 3)}</td><td>${fmt(ph, 3)}</td><td>${ps === ph ? "—" : ps > ph ? "spam ×" + (ph > 0 ? fmt(ps / ph, 1) : "∞") : "ham ×" + (ps > 0 ? fmt(ph / ps, 1) : "∞")}</td></tr>`; });
      root.querySelector("#" + u + "t").innerHTML = rows;
      const s = zs ? 0 : Math.exp(ls), h = zh ? 0 : Math.exp(lh), post = s + h > 0 ? s / (s + h) : NaN;
      const bar = `<div style="height:16px;border-radius:8px;background:var(--accent);overflow:hidden;margin:6px 0"><div style="height:100%;width:${isNaN(post) ? 0 : post * 100}%;background:var(--coral)"></div></div>`;
      ro.innerHTML = `score(spam) = prior × ∏ likelihoods = ${s.toExponential(2)} · score(ham) = ${h.toExponential(2)}<br>P(spam | email) = ${isNaN(post) ? "undefined (0 / 0)" : "<b>" + fmt(post * 100, 1) + "%</b>"}${bar}<span style="font-family:var(--sans);font-size:13.5px">${zs || zh ? "⚠️ <b>Zero-frequency problem:</b> with α = 0, a word never seen in one class gives probability exactly 0 and wipes out all the other evidence for that class. Raise α above 0 to fix it." : "Each row multiplies the running score. Set α to 0 and tick “crypto” or “invoice” to see why smoothing is needed."}</span>`;
    }
    redraw(root, draw);
  };

  // 9. Kernel trick: lift 1-D to 2-D ---------------------------------------------------------------
  D.kernel = root => {
    const u = uid();
    root.innerHTML += `<div class="controls">${slider(u + "t", "threshold on the new feature x²", 0, 6, 0.05, 1.7)}</div><div class="demo-grid"><div><div class="small">Original 1-D data: no single cut-point separates the classes</div><canvas height="230"></canvas></div><div><div class="small">After adding the feature x²: a straight line separates them</div><canvas height="230"></canvas></div></div><div class="readout"></div>`;
    const [c1, c2] = root.querySelectorAll("canvas"), ro = root.querySelector(".readout"), r0 = lcg(8), P = [];
    for (let i = 0; i < 46; i++) { const x = r0() * 5 - 2.5; if (Math.abs(Math.abs(x) - 1.3) < 0.18) continue; P.push([x, Math.abs(x) < 1.3 ? 0 : 1]); }
    function draw() {
      const t = val(root, u + "t"); show(root, ["t"], u);
      const a = MLP_canvas(c1, 230), F = frame(a, [-2.7, 2.7], [-1, 1], { ticks: false, xl: "x" });
      line(a.ctx, [[F.X(-2.7), F.Y(0)], [F.X(2.7), F.Y(0)]], a.col("--line"), 1);
      const rt = Math.sqrt(t); [-rt, rt].forEach(v => line(a.ctx, [[F.X(v), F.Y(-0.5)], [F.X(v), F.Y(0.5)]], a.col("--ink"), 1.5, [4, 4]));
      P.forEach(([x, l]) => dot(a.ctx, F.X(x), F.Y(0), 5, l ? a.col("--coral") : a.col("--accent"), a.col("--panel")));
      const b = MLP_canvas(c2, 230), G = frame(b, [-2.7, 2.7], [0, 6.5], { xl: "x", yl: "x²" });
      const pp = []; for (let x = -2.6; x <= 2.6; x += 0.05) pp.push([G.X(x), G.Y(x * x)]); line(b.ctx, pp, b.col("--line"), 1);
      line(b.ctx, [[G.X(-2.7), G.Y(t)], [G.X(2.7), G.Y(t)]], b.col("--ink"), 2);
      P.forEach(([x, l]) => dot(b.ctx, G.X(x), G.Y(x * x), 5, l ? b.col("--coral") : b.col("--accent"), b.col("--panel")));
      const ok = P.filter(([x, l]) => (x * x > t ? 1 : 0) === l).length;
      ro.innerHTML = `rule “x² > ${fmt(t, 2)}” classifies ${ok}/${P.length} points correctly${ok === P.length ? " ✅" : ""}<br><span style="font-family:var(--sans);font-size:13.5px">A flat cut in the lifted space (right) becomes two cut-points — a non-linear rule — back in the original space (left, dashed). A kernel SVM does this lifting implicitly: K(x, z) = (x·z)² gives the same dot products as explicitly computing squared features, and the RBF kernel corresponds to infinitely many such features.</span>`;
    }
    redraw(root, draw);
  };

  // 10. DBSCAN -------------------------------------------------------------------------------------
  D.dbscan = root => {
    const u = uid();
    root.innerHTML += `<div class="controls">${slider(u + "e", "eps (neighbourhood radius)", 0.05, 0.8, 0.01, 0.22)}${slider(u + "m", "min_samples", 2, 15, 1, 5)}</div><canvas height="320"></canvas><div class="readout"></div>`;
    const cv = root.querySelector("canvas"), ro = root.querySelector(".readout"), r0 = lcg(3), g0 = () => { let a = 0; for (let i = 0; i < 6; i++) a += r0(); return (a - 3) * 1.41; }, P = [];
    for (let i = 0; i < 180; i++) { const c = i % 2, t = r0() * Math.PI; P.push(c ? [1 - Math.cos(t) + g0() * 0.07, 0.45 - Math.sin(t) + g0() * 0.07] : [Math.cos(t) + g0() * 0.07, Math.sin(t) + g0() * 0.07]); }
    for (let i = 0; i < 30; i++) P.push([2.3 + g0() * 0.13, 1.3 + g0() * 0.13]);
    for (let i = 0; i < 14; i++) P.push([r0() * 4.6 - 1.5, r0() * 3.2 - 1.1]);
    const CL = ["--accent", "--coral", "--green", "--amber", "--teal", "#a855f7", "#64748b", "#db2777"];
    function draw() {
      const eps = val(root, u + "e"), ms = val(root, u + "m"); show(root, ["e", "m"], u);
      const n = P.length, nb = P.map(p => { const o = []; P.forEach((q, j) => { if (Math.hypot(p[0] - q[0], p[1] - q[1]) <= eps) o.push(j); }); return o; }), core = nb.map(o => o.length >= ms), lab = Array(n).fill(-1); let k = 0;
      for (let i = 0; i < n; i++) { if (lab[i] !== -1 || !core[i]) continue; const st = [i]; lab[i] = k; while (st.length) { const j = st.pop(); if (!core[j]) continue; nb[j].forEach(q => { if (lab[q] === -1) { lab[q] = k; st.push(q); } }); } k++; }
      const c = MLP_canvas(cv, 320), F = frame(c, [-1.7, 3.2], [-1.3, 2.2], { ticks: false }), colr = i => { const v = CL[i % CL.length]; return v.startsWith("--") ? c.col(v) : v; };
      P.forEach((p, i) => { if (lab[i] < 0) { c.ctx.strokeStyle = c.col("--muted"); c.ctx.lineWidth = 1.5; const x = F.X(p[0]), y = F.Y(p[1]); c.ctx.beginPath(); c.ctx.moveTo(x - 4, y - 4); c.ctx.lineTo(x + 4, y + 4); c.ctx.moveTo(x + 4, y - 4); c.ctx.lineTo(x - 4, y + 4); c.ctx.stroke(); } else dot(c.ctx, F.X(p[0]), F.Y(p[1]), core[i] ? 4.5 : 2.6, colr(lab[i]), core[i] ? c.col("--panel") : null); });
      c.ctx.beginPath(); c.ctx.arc(F.X(-1.2), F.Y(1.8), Math.abs(F.X(eps) - F.X(0)), 0, 7); c.ctx.strokeStyle = c.col("--ink"); c.ctx.setLineDash([4, 3]); c.ctx.stroke(); c.ctx.setLineDash([]); c.ctx.fillStyle = c.col("--muted"); c.ctx.font = "11px sans-serif"; c.ctx.fillText("eps", F.X(-1.2) - 9, F.Y(1.8) + 4);
      const noise = lab.filter(l => l < 0).length;
      ro.innerHTML = `clusters found: <b>${k}</b> · core points: ${core.filter(Boolean).length} (large dots) · border points: ${n - noise - core.filter(Boolean).length} (small dots) · noise: ${noise} (×)<br><span style="font-family:var(--sans);font-size:13.5px">${k === 3 ? "✅ Two moons and the small blob recovered, with scattered points flagged as noise — something K-means can't do." : k < 3 ? "eps too large (or min_samples too small): neighbourhoods bridge the gaps and separate clusters merge." : "eps too small (or min_samples too large): dense regions fragment and many points become noise."} A point is <b>core</b> if at least min_samples points (itself included) lie within eps.</span>`;
    }
    redraw(root, draw);
  };

  // 11. Time series: components, differencing, ACF ----------------------------------------------------
  D.tsdemo = root => {
    const u = uid();
    root.innerHTML += `<div class="controls">${slider(u + "t", "trend (per step)", 0, 0.3, 0.01, 0.12)}${slider(u + "s", "seasonal amplitude (period 12)", 0, 6, 0.2, 3)}${slider(u + "p", "AR(1) coefficient φ of the noise", 0, 0.95, 0.05, 0.3)}</div><label class="small"><input type="checkbox" id="${u}d1"> first difference (y<sub>t</sub> − y<sub>t−1</sub>)</label> &nbsp; <label class="small"><input type="checkbox" id="${u}d12"> seasonal difference (y<sub>t</sub> − y<sub>t−12</sub>)</label><div class="demo-grid" style="margin-top:8px"><div><div class="small">Series</div><canvas height="230"></canvas></div><div><div class="small">Autocorrelation function (ACF), lags 1–24</div><canvas height="230"></canvas></div></div><div class="readout"></div>`;
    const [c1, c2] = root.querySelectorAll("canvas"), ro = root.querySelector(".readout"), Z = []; for (let i = 0; i < 180; i++) Z.push(gauss());
    function draw() {
      const tr = val(root, u + "t"), am = val(root, u + "s"), ph = val(root, u + "p"), d1 = root.querySelector("#" + u + "d1").checked, d12 = root.querySelector("#" + u + "d12").checked; show(root, ["t", "s", "p"], u);
      let e = 0, y = []; for (let t = 0; t < 168; t++) { e = ph * e + Z[t]; y.push(10 + tr * t + am * Math.sin(2 * Math.PI * t / 12) + e); }
      if (d12) y = y.slice(12).map((v, i) => v - y[i]); if (d1) y = y.slice(1).map((v, i) => v - y[i]);
      const n = y.length, m = y.reduce((a, b) => a + b, 0) / n, v0 = y.reduce((s, v) => s + (v - m) ** 2, 0), acf = []; for (let k = 1; k <= 24; k++) { let s = 0; for (let t = k; t < n; t++) s += (y[t] - m) * (y[t - k] - m); acf.push(s / v0); }
      const a = MLP_canvas(c1, 230), lo = Math.min(...y), hi = Math.max(...y), F = frame(a, [0, n], [lo - 0.5, hi + 0.5], { xl: "time" }); line(a.ctx, y.map((v, i) => [F.X(i), F.Y(v)]), a.col("--accent"), 1.6);
      const b = MLP_canvas(c2, 230), G = frame(b, [0, 25], [-1, 1], { xl: "lag" }), band = 1.96 / Math.sqrt(n);
      b.ctx.globalAlpha = 0.18; b.ctx.fillStyle = b.col("--accent"); b.ctx.fillRect(G.X(0), G.Y(band), G.X(25) - G.X(0), G.Y(-band) - G.Y(band)); b.ctx.globalAlpha = 1;
      line(b.ctx, [[G.X(0), G.Y(0)], [G.X(25), G.Y(0)]], b.col("--muted"), 1);
      acf.forEach((r, i) => line(b.ctx, [[G.X(i + 1), G.Y(0)], [G.X(i + 1), G.Y(r)]], Math.abs(r) > band ? b.col("--coral") : b.col("--muted"), 4));
      const big = acf.filter(r => Math.abs(r) > band).length, slow = acf.slice(0, 10).every(r => r > 0.5);
      ro.innerHTML = `${big} of 24 lags fall outside the ±1.96/√n band (shaded) · lag-1 ACF = ${fmt(acf[0], 2)} · lag-12 ACF = ${fmt(acf[11], 2)}<br><span style="font-family:var(--sans);font-size:13.5px">${slow ? "ACF decays very slowly and stays high for many lags → <b>trend / non-stationary</b>. Tick “first difference”." : Math.abs(acf[11]) > 0.4 && Math.abs(acf[11]) > Math.abs(acf[5]) ? "Spikes at lags 12 and 24 → <b>seasonality with period 12</b>. Tick “seasonal difference” (or add seasonal terms)." : big <= 3 ? "ACF is inside the band almost everywhere → looks like <b>stationary, near-white noise</b>: nothing left to model." : "Short-memory structure in the first few lags → a low-order AR/MA model is appropriate; read p from the PACF cut-off and q from the ACF cut-off."}</span>`;
    }
    redraw(root, draw);
  };

  // 12. XGBoost gain calculator ---------------------------------------------------------------------------
  D.gaincalc = root => {
    const u = uid();
    root.innerHTML += `<div class="controls">${slider(u + "gl", "G_L (sum of gradients, left)", -30, 30, 1, 20)}${slider(u + "hl", "H_L (sum of hessians, left)", 0.5, 20, 0.5, 2)}${slider(u + "gr", "G_R (right)", -30, 30, 1, -20)}${slider(u + "hr", "H_R (right)", 0.5, 20, 0.5, 2)}${slider(u + "la", "λ (reg_lambda)", 0, 20, 0.5, 1)}${slider(u + "ga", "γ (min_split_loss)", 0, 150, 1, 0)}</div><div class="readout"></div>`;
    const ro = root.querySelector(".readout");
    function draw() {
      const [gl, hl, gr, hr, la, ga] = ["gl", "hl", "gr", "hr", "la", "ga"].map(k => val(root, u + k)); show(root, ["gl", "hl", "gr", "hr", "la", "ga"], u);
      const s = (g, h) => g * g / (h + la), gain = 0.5 * (s(gl, hl) + s(gr, hr) - s(gl + gr, hl + hr)) - ga;
      ro.innerHTML = `leaf weights w* = −G/(H+λ): left = ${fmt(-gl / (hl + la), 3)}, right = ${fmt(-gr / (hr + la), 3)}, no-split parent = ${fmt(-(gl + gr) / (hl + hr + la), 3)}<br>similarity scores G²/(H+λ): left ${fmt(s(gl, hl), 2)} + right ${fmt(s(gr, hr), 2)} − parent ${fmt(s(gl + gr, hl + hr), 2)}<br>Gain = ½ × (${fmt(s(gl, hl) + s(gr, hr) - s(gl + gr, hl + hr), 2)}) − γ = <b>${fmt(gain, 2)}</b> → ${gain > 0 ? "✅ keep the split" : "✂️ prune the split (gain ≤ 0)"}<br><span style="font-family:var(--sans);font-size:13.5px">Gradients with opposite signs on the two sides give a big gain (the children want different corrections). Raising λ shrinks the leaf weights and the gain; raising γ demands a minimum improvement per split. For squared error, H is simply the number of rows in the leaf — that's what min_child_weight limits.</span>`;
    }
    redraw(root, draw);
  };

  // 13. Loss-curve doctor ----------------------------------------------------------------------------------
  D.losscurves = root => {
    const u = uid(), N = []; for (let i = 0; i < 210; i++) N.push(gauss());
    const PAT = {
      healthy: ["✅ Healthy", e => 0.25 + 1.8 * Math.exp(-e / 14), e => 0.33 + 1.8 * Math.exp(-e / 14), "Both curves fall quickly, then flatten, with a small stable gap. Stop when validation flattens; nothing to fix."],
      overfit: ["Overfitting", e => 0.04 + 2 * Math.exp(-e / 14), e => 0.3 + 1.8 * Math.exp(-e / 11) + 0.011 * Math.max(0, e - 22), "Training loss keeps falling while validation loss turns upward: the model has started memorising. <b>Fix:</b> early stopping at the validation minimum, more data / augmentation, stronger regularisation (L2, dropout, shallower trees, lower learning rate with fewer rounds), fewer features."],
      underfit: ["Underfitting", e => 1.15 + 0.8 * Math.exp(-e / 7), e => 1.2 + 0.8 * Math.exp(-e / 7), "Both curves plateau high and close together: the model can't even fit the training data. <b>Fix:</b> more capacity (deeper trees, more features/interactions, bigger network), less regularisation, better features, train longer, check for a bug by overfitting one tiny batch."],
      diverge: ["LR too high: diverging", e => 2 + 0.02 * Math.exp(e / 16) * (1 + 0.5 * Math.abs(N[e])), null, "Loss increases (often to inf/NaN). Each step overshoots the minimum by more than the previous one. <b>Fix:</b> lower the learning rate (÷3 to ÷10), add warm-up, clip gradients, standardise inputs/targets."],
      oscillate: ["Oscillating / roaming", e => 0.75 + 1.1 * Math.exp(-e / 9) + 0.22 * Math.sin(e * 1.4) + 0.12 * N[e], e => 0.85 + 1.1 * Math.exp(-e / 9) + 0.2 * Math.sin(e * 1.4 + 0.5) + 0.1 * N[e + 100], "Loss drops, then bounces around a level without settling. Causes: learning rate too high for the sharpest direction (unscaled features), a constant learning rate with noisy mini-batches, very small batches, momentum too high, or noisy labels. <b>Fix:</b> standardise features, learning-rate decay/schedule, bigger batch, gradient clipping, lower momentum."],
      slow: ["LR too low", e => 2 - 0.0065 * e, e => 2.06 - 0.0062 * e, "Loss decreases almost linearly and is nowhere near flat at the end: the steps are tiny. <b>Fix:</b> raise the learning rate (run an LR range test), use an adaptive optimiser, scale the features, train longer."],
      noisy: ["Very noisy", e => 0.3 + 1.7 * Math.exp(-e / 14) + 0.2 * N[e], e => 0.4 + 1.7 * Math.exp(-e / 14) + 0.07 * N[e + 100], "The trend is fine but the training curve is jagged: the batch size is small, so each step's gradient is a noisy estimate. Usually harmless. <b>Fix if needed:</b> larger batch, smooth the plot (moving average), lower the learning rate late in training."],
      plateau: ["Plateau, then drop", e => e < 38 ? 1.7 + 0.02 * N[e] : 0.35 + 1.35 * Math.exp(-(e - 38) / 8), e => e < 38 ? 1.75 + 0.02 * N[e + 100] : 0.45 + 1.3 * Math.exp(-(e - 38) / 8), "A long flat stretch followed by a sudden improvement: the optimiser was crossing a saddle point / flat region, or a learning-rate schedule just kicked in. Don't stop too early. <b>Help it:</b> momentum/Adam, better initialisation, normalisation layers, learning-rate warm-up."],
      valbelow: ["Validation below training", e => 0.55 + 1.6 * Math.exp(-e / 14), e => 0.36 + 1.6 * Math.exp(-e / 14), "Not necessarily a bug. Regularisation (dropout, augmentation) is active only in training; training loss is averaged during the epoch while validation is measured at its end; or the validation set is easier / too small / leaks from training. <b>Check:</b> evaluate the training set in eval mode, inspect the split, look for duplicates across sets."],
      nan: ["Sudden spike / NaN", e => e < 61 ? 0.3 + 1.7 * Math.exp(-e / 14) : e === 61 ? 3.4 : NaN, e => e < 61 ? 0.4 + 1.7 * Math.exp(-e / 14) : e === 61 ? 3.4 : NaN, "Training was fine, then the loss exploded. Causes: exploding gradients, a bad batch (NaN/inf or extreme values), log(0) or division by zero in the loss, learning-rate too high after warm-up, mixed-precision overflow. <b>Fix:</b> gradient clipping, validate/clean inputs, add epsilons, lower the LR, find the offending batch."],
      saw: ["Periodic saw-tooth", e => 0.4 + 1.5 * Math.exp(-e / 14) + 0.3 * ((e % 10) / 10), e => 0.55 + 1.5 * Math.exp(-e / 14), "The training loss repeats a pattern every epoch: the data aren't shuffled (batches are ordered by class, time or source), so the model keeps adapting to one chunk and forgetting the previous one. <b>Fix:</b> shuffle every epoch; check the data loader."],
    };
    root.innerHTML += `<div style="margin:6px 0 10px">${seg(u + "s", Object.keys(PAT).map(k => [k, PAT[k][0]]), "oscillate")}</div><canvas height="270"></canvas><div class="readout" style="font-family:var(--sans)"></div>`;
    const cv = root.querySelector("canvas"), ro = root.querySelector(".readout"); let cur = "oscillate";
    root.querySelector(".seg").style.flexWrap = "wrap";
    function draw() {
      const [, tr, va, txt] = PAT[cur], c = MLP_canvas(cv, 270), F = frame(c, [0, 100], [0, 3.5], { xl: "epoch", yl: "loss" });
      const plot = (f, colr, name, yy) => { if (!f) return; const p = []; for (let e = 0; e <= 100; e++) { const v = f(e); if (isNaN(v)) break; p.push([F.X(e), F.Y(Math.max(0, Math.min(3.5, v)))]); } line(c.ctx, p, c.col(colr), 2.2); c.ctx.fillStyle = c.col(colr); c.ctx.font = "12px sans-serif"; c.ctx.textAlign = "left"; c.ctx.fillText("■ " + name, F.X(70), F.Y(yy)); };
      plot(tr, "--accent", "training loss", 3.25); plot(va, "--coral", "validation loss", 2.95);
      ro.innerHTML = `<b>${PAT[cur][0].replace("✅ ", "")}.</b> ${txt}`;
    }
    wireSeg(root, u + "s", v => { cur = v; draw(); }); redraw(root, draw);
  };

  // 14. Hierarchical clustering dendrogram (1-D points) ---------------------------------------------------------
  D.dendro = root => {
    const u = uid(), X = [1, 2, 5, 6, 11, 12.5, 18];
    root.innerHTML += `<div style="margin:6px 0 10px">${seg(u + "l", [["single", "Single linkage"], ["complete", "Complete linkage"], ["average", "Average linkage"]], "single")}</div><div class="controls">${slider(u + "c", "cut height", 0.2, 18, 0.1, 4)}</div><canvas height="290"></canvas><div class="readout"></div>`;
    const cv = root.querySelector("canvas"), ro = root.querySelector(".readout"); let link = "single";
    function draw() {
      const cut = val(root, u + "c"); show(root, ["c"], u);
      let cl = X.map((x, i) => ({ m: [i], x, h: 0 })); const merges = [];
      const dist = (a, b) => { const ds = []; a.m.forEach(i => b.m.forEach(j => ds.push(Math.abs(X[i] - X[j])))); return link === "single" ? Math.min(...ds) : link === "complete" ? Math.max(...ds) : ds.reduce((s, v) => s + v, 0) / ds.length; };
      while (cl.length > 1) { let bi = 0, bj = 1, bd = Infinity; for (let i = 0; i < cl.length; i++) for (let j = i + 1; j < cl.length; j++) { const d = dist(cl[i], cl[j]); if (d < bd) { bd = d; bi = i; bj = j; } } const a = cl[bi], b = cl[bj], nw = { m: [...a.m, ...b.m], x: (a.x + b.x) / 2, h: bd }; merges.push([a, b, nw]); cl = cl.filter((_, k) => k !== bi && k !== bj); cl.push(nw); cl.sort((p, q) => p.x - q.x); }
      const c = MLP_canvas(cv, 290), top = Math.max(18, merges[merges.length - 1][2].h * 1.1), F = frame(c, [0, 19], [0, top], { xl: "points on a line", yl: "merge distance" });
      merges.forEach(([a, b, nw]) => { const colr = nw.h <= cut ? c.col("--accent") : c.col("--muted"); line(c.ctx, [[F.X(a.x), F.Y(a.h)], [F.X(a.x), F.Y(nw.h)], [F.X(b.x), F.Y(nw.h)], [F.X(b.x), F.Y(b.h)]], colr, 2); });
      line(c.ctx, [[F.X(0), F.Y(cut)], [F.X(19), F.Y(cut)]], c.col("--coral"), 1.5, [6, 4]);
      X.forEach(x => dot(c.ctx, F.X(x), F.Y(0), 5, c.col("--ink")));
      const k = 1 + merges.filter(m => m[2].h > cut).length;
      ro.innerHTML = `cutting at height ${fmt(cut, 1)} gives <b>${k} cluster${k > 1 ? "s" : ""}</b> · merge heights: ${merges.map(m => fmt(m[2].h, 1)).join(" → ")}<br><span style="font-family:var(--sans);font-size:13.5px">Read it bottom-up: the closest pair merges first; each horizontal bar is drawn at the distance at which two groups were joined. Long vertical stretches with no merges are natural places to cut. Switch linkage: single uses the closest pair between groups (chains easily), complete the farthest pair (compact groups), average the mean of all pairs.</span>`;
    }
    wireSeg(root, u + "l", v => { link = v; draw(); }); redraw(root, draw);
  };
})();
