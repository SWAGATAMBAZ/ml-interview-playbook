/* ML Interview Playbook — shared behaviour: sidebar, progress, search, maths, code, Q&A */
(function () {
  const PAGES = [
    { id: "index", file: "index.html", title: "Start here & study plan", group: "Start" },
    { id: "fundamentals", file: "fundamentals.html", title: "ML fundamentals", group: "Foundations" },
    { id: "data-prep", file: "data-prep.html", title: "Data prep & feature engineering", group: "Foundations" },
    { id: "metrics", file: "metrics.html", title: "Evaluation metrics", group: "Foundations" },
    { id: "validation", file: "validation.html", title: "Cross-validation & tuning", group: "Foundations" },
    { id: "performance", file: "performance.html", title: "Improving models & explainability", group: "Foundations" },
    { id: "statistics", file: "statistics.html", title: "Statistics & A/B testing", group: "Foundations" },
    { id: "linear-regression", file: "linear-regression.html", title: "Linear regression", group: "Supervised models" },
    { id: "regularization", file: "regularization.html", title: "Ridge, Lasso & Elastic Net", group: "Supervised models" },
    { id: "logistic-regression", file: "logistic-regression.html", title: "Logistic regression", group: "Supervised models" },
    { id: "knn", file: "knn.html", title: "k-Nearest Neighbours", group: "Supervised models" },
    { id: "naive-bayes", file: "naive-bayes.html", title: "Naive Bayes", group: "Supervised models" },
    { id: "svm", file: "svm.html", title: "Support Vector Machines", group: "Supervised models" },
    { id: "decision-trees", file: "decision-trees.html", title: "Decision trees", group: "Supervised models" },
    { id: "random-forest", file: "random-forest.html", title: "Bagging & Random Forest", group: "Supervised models" },
    { id: "boosting", file: "boosting.html", title: "AdaBoost & Gradient Boosting", group: "Supervised models" },
    { id: "xgboost", file: "xgboost.html", title: "XGBoost, LightGBM & CatBoost", group: "Supervised models" },
    { id: "kmeans", file: "kmeans.html", title: "K-Means clustering", group: "Unsupervised" },
    { id: "clustering", file: "clustering.html", title: "Hierarchical, DBSCAN, GMM & anomalies", group: "Unsupervised" },
    { id: "dim-reduction", file: "dim-reduction.html", title: "PCA & dimensionality reduction", group: "Unsupervised" },
    { id: "time-series", file: "time-series.html", title: "Time-series forecasting", group: "Applied" },
    { id: "mlops", file: "mlops.html", title: "MLOps & production ML", group: "Production" },
    { id: "system-design", file: "system-design.html", title: "ML system design", group: "Production" },
    { id: "labs", file: "labs.html", title: "Coding drills (Python)", group: "Practice" },
    { id: "playground", file: "playground.html", title: "Python playground", group: "Practice" },
    { id: "bank", file: "bank.html", title: "Question bank & mock interview", group: "Practice" },
    { id: "cheatsheet", file: "cheatsheet.html", title: "Cheat sheet & model chooser", group: "Practice" },
  ];
  window.MLP_PAGES = PAGES;

  const store = {
    get(k, d) { try { const v = localStorage.getItem(k); return v === null ? d : JSON.parse(v); } catch (e) { return d; } },
    set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) {} },
  };
  window.MLP_STORE = store;

  const theme = store.get("mlp-theme", null) || (window.matchMedia && matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  document.documentElement.setAttribute("data-theme", theme);

  let current = "index";
  const esc = s => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  window.MLP_esc = esc;

  function buildSidebar() {
    const aside = document.createElement("aside");
    aside.className = "sidebar";
    const done = store.get("mlp-done", {});
    let html = `<a class="brand" href="index.html"><span class="logo">ƒ</span>ML Playbook</a>
      <div class="brand-sub">Machine-learning interviews, end to end</div>
      <button class="btn small search-btn" id="searchBtn"><span>🔎 Search everything</span><span class="kbd">/</span></button>
      <div class="progress-wrap"><div class="progress-bar"><span id="pbar"></span></div><div class="progress-label" id="plabel"></div></div>
      <nav class="nav">`;
    let g = "";
    PAGES.forEach((p, i) => {
      if (p.group !== g) { g = p.group; html += `<div class="nav-group">${g}</div>`; }
      html += `<a href="${p.file}" class="${p.id === current ? "active" : ""} ${done[p.id] ? "done" : ""}" data-id="${p.id}"><span class="num"><span>${String(i).padStart(2, "0")}</span></span>${p.title}</a>`;
    });
    html += `</nav><div class="side-tools"><button class="btn small" id="themeBtn">◐ Theme</button><button class="btn small" id="resetBtn">Reset progress</button></div>
      <div class="small" style="margin:14px 6px 0">Sister sites: <a href="https://swagatambaz.github.io/deep-learning-playbook/">Deep Learning</a> · <a href="https://swagatambaz.github.io/sql-interview-lab/">SQL</a> · <a href="https://swagatambaz.github.io/dsa-playbook/">DSA</a></div>`;
    aside.innerHTML = html;
    return aside;
  }

  function updateProgress() {
    const d = store.get("mlp-done", {});
    const n = PAGES.filter(p => d[p.id]).length;
    const pct = Math.round((n / PAGES.length) * 100);
    const labs = Object.values(store.get("mlp-labs", {})).filter(Boolean).length;
    const known = Object.values(store.get("mlp-known", {})).filter(Boolean).length;
    const bar = document.getElementById("pbar"); if (bar) bar.style.width = pct + "%";
    const lab = document.getElementById("plabel");
    if (lab) lab.textContent = `${n}/${PAGES.length} topics · ${labs} labs passed · ${known} Qs known`;
    document.querySelectorAll(".nav a").forEach(a => a.classList.toggle("done", !!d[a.dataset.id]));
    document.querySelectorAll("[data-step]").forEach(s => s.classList.toggle("done", !!d[s.dataset.step]));
  }
  window.MLP_updateProgress = updateProgress;

  // ---- tiny Python highlighter for static snippets
  const KW = new Set("def class return if elif else for while in not and or import from as with try except finally raise lambda yield None True False pass break continue is assert global nonlocal async await".split(" "));
  const BUILTIN = new Set("self print len range super np pd plt int float str list dict tuple zip enumerate sum max min abs round sorted isinstance".split(" "));
  function highlight(src) {
    const re = /("""[\s\S]*?"""|'''[\s\S]*?'''|"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*'|#[^\n]*|\b\d+\.?\d*(?:e-?\d+)?\b|\b[A-Za-z_][A-Za-z0-9_]*\b)/g;
    let out = "", last = 0, m;
    while ((m = re.exec(src))) {
      out += esc(src.slice(last, m.index));
      const t = m[0];
      let cls = null;
      if (t[0] === "#") cls = "c";
      else if (t[0] === '"' || t[0] === "'") cls = "s";
      else if (/^\d/.test(t)) cls = "n";
      else if (KW.has(t)) cls = "k";
      else if (BUILTIN.has(t)) cls = "b";
      else if (src[re.lastIndex] === "(") cls = "f";
      out += cls ? `<span class="tok-${cls}">${esc(t)}</span>` : esc(t);
      last = re.lastIndex;
    }
    return out + esc(src.slice(last));
  }
  function dedent(s) {
    s = s.replace(/^\n+|\s+$/g, "");
    const lines = s.split("\n");
    const ind = Math.min(...lines.filter(l => l.trim()).map(l => l.match(/^ */)[0].length));
    return lines.map(l => l.slice(ind)).join("\n");
  }
  window.MLP_dedent = dedent;
  function copyBtn(btn, getText) {
    btn.addEventListener("click", () => {
      const ok = () => { btn.textContent = "Copied ✓"; setTimeout(() => (btn.textContent = "Copy"), 1400); };
      const t = getText();
      if (navigator.clipboard) navigator.clipboard.writeText(t).then(ok, ok); else ok();
    });
  }
  function buildCode() {
    document.querySelectorAll('script[type="text/plain"].snippet').forEach(sc => {
      const src = dedent(sc.textContent);
      const wrap = document.createElement("div");
      wrap.className = "code";
      wrap.innerHTML = `<div class="code-head"><span>${esc(sc.dataset.title || "python")}</span><button class="copy">Copy</button></div><pre><code>${sc.dataset.lang === "text" ? esc(src) : highlight(src)}</code></pre>`;
      copyBtn(wrap.querySelector(".copy"), () => src);
      sc.replaceWith(wrap);
    });
  }

  function buildTabs() {
    document.querySelectorAll("[data-tabs]").forEach(root => {
      const btns = root.querySelectorAll(":scope > .tabs button");
      const panels = root.querySelectorAll(":scope > .tab-panel");
      btns.forEach((b, i) => b.addEventListener("click", () => {
        btns.forEach(x => x.classList.remove("on")); panels.forEach(x => x.classList.remove("on"));
        b.classList.add("on"); panels[i].classList.add("on");
        window.dispatchEvent(new Event("resize"));
      }));
    });
  }

  // ---- Q&A: level badges, "I know this", anchors
  const LV = { 1: "basic", 2: "core", 3: "advanced" };
  function buildQA() {
    const known = store.get("mlp-known", {});
    document.querySelectorAll("details.qa").forEach(d => {
      const id = d.dataset.id;
      if (id) d.id = "q-" + id;
      const s = d.querySelector("summary");
      if (s && !s.querySelector(".qtext")) {
        s.innerHTML = `<span class="qtext">${s.innerHTML}</span>` + (d.dataset.l ? `<span class="lvl l${d.dataset.l}">${LV[d.dataset.l]}</span>` : "");
      }
      if (id) {
        const ans = d.querySelector(".ans");
        const k = document.createElement("label");
        k.className = "known";
        k.innerHTML = `<input type="checkbox"> I can answer this confidently`;
        const cb = k.querySelector("input");
        cb.checked = !!known[id];
        d.classList.toggle("is-known", cb.checked);
        cb.addEventListener("change", () => {
          const kn = store.get("mlp-known", {}); kn[id] = cb.checked; store.set("mlp-known", kn);
          d.classList.toggle("is-known", cb.checked); updateProgress(); paintQATools();
        });
        (ans || d).appendChild(k);
      }
    });
    // tools above each qa-block
    document.querySelectorAll(".qa-block").forEach(b => {
      const t = document.createElement("div");
      t.className = "qa-tools";
      t.innerHTML = `<button class="btn small" data-act="open">Expand all</button><button class="btn small" data-act="close">Collapse all</button><button class="btn small" data-act="unknown">Show only not-yet-known</button><span class="qa-count"></span>`;
      b.prepend(t);
      t.addEventListener("click", e => {
        const act = e.target.dataset.act; if (!act) return;
        const qs = b.querySelectorAll("details.qa");
        if (act === "open") qs.forEach(q => (q.open = true));
        if (act === "close") qs.forEach(q => (q.open = false));
        if (act === "unknown") {
          const on = e.target.classList.toggle("primary");
          qs.forEach(q => (q.style.display = on && q.classList.contains("is-known") ? "none" : ""));
        }
      });
    });
    paintQATools();
  }
  function paintQATools() {
    document.querySelectorAll(".qa-block").forEach(b => {
      const qs = b.querySelectorAll("details.qa");
      const k = b.querySelectorAll("details.qa.is-known").length;
      const c = b.querySelector(".qa-count"); if (c) c.textContent = `${k} / ${qs.length} known`;
    });
  }

  function openHash() {
    const h = decodeURIComponent(location.hash.slice(1));
    if (!h) return;
    const el = document.getElementById(h);
    if (!el) return;
    if (el.tagName === "DETAILS") el.open = true;
    const par = el.closest(".tab-panel");
    if (par && !par.classList.contains("on")) {
      const root = par.parentElement; const idx = [...root.querySelectorAll(":scope > .tab-panel")].indexOf(par);
      const b = root.querySelectorAll(":scope > .tabs button")[idx]; if (b) b.click();
    }
    setTimeout(() => el.scrollIntoView({ block: "start" }), 60);
  }

  function buildPager() {
    const main = document.querySelector("main");
    const i = PAGES.findIndex(p => p.id === current);
    const row = document.createElement("div");
    row.className = "done-row";
    row.innerHTML = `<button class="btn primary" id="doneBtn"></button><span style="color:var(--muted);font-size:14px">Tick this off once you can explain the 🎯 points out loud, pass the labs, and answer most questions without peeking.</span>`;
    main.appendChild(row);
    const btn = row.querySelector("#doneBtn");
    const paint = () => { const d = store.get("mlp-done", {}); btn.textContent = d[current] ? "✓ Done — click to undo" : "Mark this topic as done"; };
    btn.addEventListener("click", () => { const d = store.get("mlp-done", {}); d[current] = !d[current]; store.set("mlp-done", d); paint(); updateProgress(); });
    paint();
    const pager = document.createElement("div");
    pager.className = "pager";
    const prev = PAGES[i - 1], next = PAGES[i + 1];
    pager.innerHTML = (prev ? `<a href="${prev.file}"><small>← Previous</small>${prev.title}</a>` : "<span></span>") +
      (next ? `<a class="next" href="${next.file}"><small>Next →</small>${next.title}</a>` : "");
    main.appendChild(pager);
  }

  // ---- maths via KaTeX (graceful if offline)
  function loadCSS(href) { const l = document.createElement("link"); l.rel = "stylesheet"; l.href = href; document.head.appendChild(l); }
  function loadJS(src) { return new Promise((res, rej) => { const s = document.createElement("script"); s.src = src; s.onload = res; s.onerror = rej; document.head.appendChild(s); }); }
  window.MLP_loadJS = loadJS; window.MLP_loadCSS = loadCSS;
  const KTX = "assets/vendor/katex/";
  function renderMath(root) {
    if (!/\\\(|\$\$|\\\[/.test((root || document.body).innerHTML)) return;
    const go = () => window.renderMathInElement && window.renderMathInElement(root || document.body, {
      delimiters: [{ left: "$$", right: "$$", display: true }, { left: "\\[", right: "\\]", display: true }, { left: "\\(", right: "\\)", display: false }],
      ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code"], throwOnError: false,
    });
    if (window.renderMathInElement) return go();
    if (!window.__ktx) {
      loadCSS(KTX + "katex.min.css");
      window.__ktx = loadJS(KTX + "katex.min.js").then(() => loadJS(KTX + "contrib/auto-render.min.js"));
    }
    window.__ktx.then(go).catch(() => {});
  }
  window.MLP_renderMath = renderMath;

  // ---- search
  let searchData = null;
  function openSearch() {
    let m = document.getElementById("searchModal");
    if (!m) {
      m = document.createElement("div"); m.id = "searchModal"; m.className = "search-modal";
      m.innerHTML = `<div class="search-box"><input id="searchInput" placeholder="Search topics, questions, labs… e.g. heteroscedasticity, SMOTE, gamma" autocomplete="off"><div class="search-results" id="searchResults"></div></div>`;
      document.body.appendChild(m);
      m.addEventListener("click", e => { if (e.target === m) m.classList.remove("on"); });
      const inp = m.querySelector("input");
      let sel = 0;
      inp.addEventListener("input", () => { sel = 0; runSearch(inp.value, sel); });
      inp.addEventListener("keydown", e => {
        const items = m.querySelectorAll(".search-results a");
        if (e.key === "ArrowDown") { sel = Math.min(sel + 1, items.length - 1); e.preventDefault(); }
        else if (e.key === "ArrowUp") { sel = Math.max(sel - 1, 0); e.preventDefault(); }
        else if (e.key === "Enter" && items[sel]) { location.href = items[sel].href; m.classList.remove("on"); return; }
        else if (e.key === "Escape") { m.classList.remove("on"); return; }
        else return;
        items.forEach((a, i) => a.classList.toggle("sel", i === sel));
        if (items[sel]) items[sel].scrollIntoView({ block: "nearest" });
      });
    }
    m.classList.add("on");
    const inp = m.querySelector("input"); inp.focus(); inp.select();
    if (!searchData) {
      if (window.MLP_SEARCH) { searchData = window.MLP_SEARCH; runSearch(inp.value, 0); }
      else loadJS("assets/search-index.js").then(() => { searchData = window.MLP_SEARCH || []; runSearch(inp.value, 0); });
    }
  }
  function runSearch(q, sel) {
    const box = document.getElementById("searchResults"); if (!box || !searchData) return;
    q = q.trim().toLowerCase();
    if (!q) { box.innerHTML = `<div style="padding:14px 18px" class="small">Type to search ${searchData.length} sections, questions and labs.</div>`; return; }
    const terms = q.split(/\s+/);
    const scored = [];
    for (const it of searchData) {
      const hay = (it.t + " " + (it.x || "")).toLowerCase();
      if (!terms.every(t => hay.includes(t))) continue;
      let s = 0; const tl = it.t.toLowerCase();
      terms.forEach(t => { if (tl.includes(t)) s += 3; });
      if (tl.startsWith(terms[0])) s += 2;
      s += { page: 3, section: 2, lab: 1, question: 1 }[it.k] || 0;
      scored.push([s, it]);
    }
    scored.sort((a, b) => b[0] - a[0]);
    box.innerHTML = scored.slice(0, 40).map(([, it], i) => `<a class="${i === sel ? "sel" : ""}" href="${it.p}${it.h ? "#" + it.h : ""}"><span class="kind">${it.k}</span>${esc(it.t)}<small>${esc(it.g)}</small></a>`).join("") || `<div style="padding:14px 18px" class="small">No matches.</div>`;
  }

  document.addEventListener("DOMContentLoaded", () => {
    current = document.body.dataset.page || "index";
    const main = document.querySelector("main");
    const layout = document.createElement("div");
    layout.className = "layout";
    const side = buildSidebar();
    main.parentNode.insertBefore(layout, main);
    layout.appendChild(side);
    layout.appendChild(main);
    const mt = document.createElement("button");
    mt.className = "btn menu-toggle"; mt.textContent = "☰ Topics";
    mt.addEventListener("click", () => side.classList.toggle("open"));
    document.body.appendChild(mt);
    document.getElementById("themeBtn").addEventListener("click", () => {
      const t = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", t); store.set("mlp-theme", t);
      window.dispatchEvent(new Event("mlp-theme"));
    });
    document.getElementById("resetBtn").addEventListener("click", () => {
      if (!confirm("Reset topic ticks, lab passes and known questions?")) return;
      ["mlp-done", "mlp-labs", "mlp-known"].forEach(k => store.set(k, {})); location.reload();
    });
    document.getElementById("searchBtn").addEventListener("click", openSearch);
    document.addEventListener("keydown", e => {
      const tag = (e.target.tagName || "").toLowerCase();
      const typing = tag === "input" || tag === "textarea" || e.target.isContentEditable || e.target.closest(".CodeMirror");
      if ((e.key === "/" && !typing) || ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k")) { e.preventDefault(); openSearch(); }
    });
    buildCode();
    buildTabs();
    buildQA();
    if (current !== "index" && !document.body.dataset.nopager) buildPager();
    updateProgress();
    renderMath(document.body);
    openHash();
    window.addEventListener("hashchange", openHash);
  });

  // canvas helper for demos
  window.MLP_canvas = function (canvas, h) {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth || 600;
    canvas.width = w * dpr; canvas.height = h * dpr; canvas.style.height = h + "px";
    const ctx = canvas.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const cs = getComputedStyle(document.documentElement);
    const col = n => cs.getPropertyValue(n).trim();
    return { ctx, w, h, col };
  };
})();
