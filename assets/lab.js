/* ML Interview Playbook — in-browser Python labs (Pyodide in a Web Worker) */
(function () {
  const CM = "assets/vendor/codemirror/";
  const store = () => window.MLP_STORE;
  let worker = null, workerReady = false, busy = null, seq = 0;
  const handlers = {};

  function toast(msg) {
    let t = document.getElementById("pyStatus");
    if (!t) { t = document.createElement("div"); t.id = "pyStatus"; t.className = "py-status"; document.body.appendChild(t); }
    if (msg) { t.textContent = msg; t.classList.add("on"); } else t.classList.remove("on");
  }

  function getWorker() {
    if (worker) return worker;
    toast("⏳ Starting Python in your browser… (first time downloads ~20 MB, then it's cached)");
    const pb = /^(localhost|127\.0\.0\.1)$/.test(location.hostname) ? (location.search.match(/[?&]pybase=([^&]+)/) || [])[1] : null; // local testing only
    worker = new Worker("assets/lab-worker.js" + (pb ? "?pybase=" + pb : ""));
    worker.onmessage = e => {
      const m = e.data;
      if (m.type === "ready") { workerReady = true; toast(null); return; }
      if (m.type === "fatal") {
        toast(null);
        // let the pending run report the failure, then discard this worker so the next Run retries
        const w = worker; worker = null; workerReady = false;
        setTimeout(() => w && w.terminate(), 2000);
        return;
      }
      const h = handlers[m.id]; if (h) h(m);
    };
    worker.onerror = e => { e.preventDefault && e.preventDefault(); toast(null); };
    return worker;
  }
  function stopWorker() {
    if (worker) worker.terminate();
    worker = null; workerReady = false;
    if (busy) { const h = handlers[busy]; if (h) h({ type: "error", text: "⏹ Stopped. Python was restarted (variables cleared)." }); }
    busy = null; toast(null);
  }

  // public API: run(code, {check, nsKey, onOut}) -> Promise<result>
  function run(code, opts = {}) {
    return new Promise(resolve => {
      const id = "r" + (++seq);
      busy = id;
      handlers[id] = m => {
        if (m.type === "out" || m.type === "err" || m.type === "sys") { opts.onOut && opts.onOut(m); return; }
        delete handlers[id]; busy = null; toast(null);
        resolve(m);
      };
      if (!workerReady) toast("⏳ Starting Python in your browser… (first time downloads ~20 MB, then it's cached)");
      getWorker().postMessage({ id, code, check: opts.check || null, nsKey: opts.nsKey || null });
    });
  }
  window.MLP_py = { run, stop: stopWorker };

  // ---- editor
  let cmPromise = null;
  function loadCM() {
    if (!cmPromise) {
      window.MLP_loadCSS(CM + "codemirror.css");
      cmPromise = window.MLP_loadJS(CM + "codemirror.min.js")
        .then(() => window.MLP_loadJS(CM + "python.min.js"))
        .then(() => window.MLP_loadJS(CM + "matchbrackets.min.js"))
        .catch(() => null);
    }
    return cmPromise;
  }
  function makeEditor(host, value, onRun) {
    const ta = document.createElement("textarea");
    ta.value = value; ta.spellcheck = false;
    host.appendChild(ta);
    const api = {
      get: () => ta.value,
      set: v => { ta.value = v; },
      cm: null,
    };
    ta.addEventListener("keydown", e => {
      if (e.key === "Tab") { e.preventDefault(); const s = ta.selectionStart; ta.setRangeText("    ", s, ta.selectionEnd, "end"); }
      if (e.key === "Enter" && (e.ctrlKey || e.metaKey || e.shiftKey)) { e.preventDefault(); onRun(); }
    });
    loadCM().then(() => {
      if (!window.CodeMirror) return;
      const cm = window.CodeMirror.fromTextArea(ta, {
        mode: "python", lineNumbers: true, indentUnit: 4, matchBrackets: true, viewportMargin: Infinity,
        theme: "mlp", extraKeys: {
          "Ctrl-Enter": onRun, "Cmd-Enter": onRun, "Shift-Enter": onRun,
          Tab: c => c.somethingSelected() ? c.indentSelection("add") : c.replaceSelection("    ", "end"),
        },
      });
      api.get = () => cm.getValue(); api.set = v => cm.setValue(v); api.cm = cm;
      cm.on("change", () => api.onChange && api.onChange());
    });
    ta.addEventListener("input", () => api.onChange && api.onChange());
    return api;
  }

  function part(lab, name) { const el = lab.querySelector(`[data-part="${name}"]`); return el ? window.MLP_dedent(el.textContent) : ""; }

  function append(out, text, cls) {
    const span = document.createElement("span");
    if (cls) span.className = cls;
    span.textContent = text.endsWith("\n") ? text : text + "\n";
    out.appendChild(span);
    out.scrollTop = out.scrollHeight;
  }

  function buildLab(lab, n) {
    const id = lab.dataset.id;
    const starter = part(lab, "starter");
    const solution = part(lab, "solution");
    const check = part(lab, "check");
    const setup = part(lab, "setup");
    const hints = [...lab.querySelectorAll('[data-part="hint"]')].map(h => h.innerHTML);
    const task = lab.querySelector(".lab-task");
    lab.querySelectorAll("script[data-part], [data-part='hint']").forEach(s => s.remove());

    const saved = store().get("mlp-code", {})[id];
    const passed = !!store().get("mlp-labs", {})[id];
    const head = document.createElement("div");
    head.className = "lab-head";
    head.innerHTML = `<span class="badge">${lab.dataset.kind || "LAB"} ${n}</span><span class="lab-name">${window.MLP_esc(lab.dataset.title || "")}</span><span class="status ${passed ? "pass" : ""}">${passed ? "✓ passed" : check ? "not passed yet" : "free run"}</span>`;
    lab.prepend(head);
    if (lab.id === "") lab.id = "lab-" + id;

    if (hints.length) {
      const hd = document.createElement("div");
      hd.className = "lab-hints";
      hd.innerHTML = hints.map((h, i) => `<details><summary>Hint ${i + 1}</summary><div>${h}</div></details>`).join("");
      (task || head).after(hd);
    }
    const edHost = document.createElement("div"); edHost.className = "lab-editor";
    const bar = document.createElement("div"); bar.className = "lab-bar";
    const out = document.createElement("div"); out.className = "lab-out";
    lab.append(edHost, bar, out);

    const doRun = () => go(false);
    const ed = makeEditor(edHost, saved != null ? saved : starter, doRun);
    let saveT;
    ed.onChange = () => { clearTimeout(saveT); saveT = setTimeout(() => { const c = store().get("mlp-code", {}); c[id] = ed.get(); store().set("mlp-code", c); }, 400); };

    bar.innerHTML = `<button class="btn primary small" data-a="run">▶ Run</button>` +
      (check ? `<button class="btn small" data-a="check">✓ Check my answer</button>` : "") +
      (solution ? `<button class="btn small" data-a="sol">💡 Show solution</button>` : "") +
      `<button class="btn small" data-a="reset">↺ Reset</button><button class="btn small" data-a="stop" style="display:none">⏹ Stop</button><span class="hint">Ctrl/⌘ + Enter to run</span>`;
    const btns = {}; bar.querySelectorAll("button").forEach(b => (btns[b.dataset.a] = b));
    let mine = null;

    async function go(withCheck) {
      if (busy) { append(out, "Another cell is running — wait or press Stop.", "sys"); return; }
      out.innerHTML = "";
      Object.values(btns).forEach(b => b.disabled = true);
      btns.stop.style.display = ""; btns.stop.disabled = false;
      btns.run.textContent = "… running";
      const code = (setup ? setup + "\n" : "") + ed.get();
      const res = await run(code, {
        check: withCheck ? check : null,
        nsKey: lab.dataset.ns || null,
        onOut: m => append(out, m.text, m.type === "err" ? "err" : m.type === "sys" ? "sys" : null),
      });
      Object.values(btns).forEach(b => b.disabled = false);
      btns.stop.style.display = "none"; btns.run.textContent = "▶ Run";
      (res.figs || []).forEach(b64 => { const img = new Image(); img.src = "data:image/png;base64," + b64; out.appendChild(img); });
      if (res.type === "done") {
        if (withCheck) {
          append(out, "✅ All checks passed! Nice work.", "ok");
          const L = store().get("mlp-labs", {}); L[id] = true; store().set("mlp-labs", L);
          const st = head.querySelector(".status"); st.textContent = "✓ passed"; st.classList.add("pass");
          window.MLP_updateProgress && window.MLP_updateProgress();
        } else {
          if (!out.textContent.trim() && !(res.figs || []).length) append(out, "(ran without printing anything)", "sys");
          append(out, `⏱ ${res.ms} ms` + (check ? " — press “Check my answer” to grade it" : ""), "sys");
        }
      } else {
        if (res.isAssert) append(out, "❌ Not quite: " + res.text, "err");
        else append(out, (res.stage === "check" ? "❌ Your code ran, but the checker hit an error (a variable may be missing or the wrong type):\n" : "") + res.text, "err");
      }
    }
    btns.run.addEventListener("click", () => go(false));
    if (btns.check) btns.check.addEventListener("click", () => go(true));
    btns.stop.addEventListener("click", stopWorker);
    btns.reset.addEventListener("click", () => {
      if (!confirm("Reset this editor to the starter code?")) return;
      ed.set(starter); const c = store().get("mlp-code", {}); delete c[id]; store().set("mlp-code", c); out.innerHTML = "";
      mine = null; if (btns.sol) btns.sol.textContent = "💡 Show solution";
    });
    if (btns.sol) btns.sol.addEventListener("click", () => {
      if (mine === null) { mine = ed.get(); ed.set(solution); btns.sol.textContent = "↩ Back to my code"; }
      else { ed.set(mine); mine = null; btns.sol.textContent = "💡 Show solution"; }
    });
  }

  document.addEventListener("DOMContentLoaded", () => {
    const labs = document.querySelectorAll(".lab[data-id]");
    if (!labs.length) return;
    labs.forEach((l, i) => buildLab(l, i + 1));
    // warm up Python when the first lab scrolls into view
    if ("IntersectionObserver" in window) {
      const io = new IntersectionObserver(es => { if (es.some(x => x.isIntersecting)) { getWorker(); io.disconnect(); } }, { rootMargin: "300px" });
      io.observe(labs[0]);
    }
  });
})();
