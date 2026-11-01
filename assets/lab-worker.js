/* Runs Python (Pyodide) off the main thread so the page never freezes. */
const PYODIDE_VERSION = "0.29.5";
const LOCAL = /^(localhost|127\.0\.0\.1)$/.test(self.location.hostname);
const BASE = (LOCAL && self.location.search.match(/[?&]pybase=([^&]+)/) ? decodeURIComponent(self.location.search.match(/[?&]pybase=([^&]+)/)[1]) : `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`);
let py = null, curId = null;
const keep = {}; // persistent namespaces (playground)

const SETUP = `
import os, sys, warnings
os.environ["MPLBACKEND"] = "AGG"
warnings.filterwarnings("ignore", message=".*non-interactive.*")
warnings.filterwarnings("ignore", category=FutureWarning)
def __mlp_figs():
    if "matplotlib.pyplot" not in sys.modules:
        return []
    import io, base64
    import matplotlib.pyplot as plt
    out = []
    for n in plt.get_fignums():
        f = plt.figure(n)
        b = io.BytesIO()
        f.savefig(b, format="png", dpi=90, bbox_inches="tight")
        out.append(base64.b64encode(b.getvalue()).decode())
    plt.close("all")
    return out
`;

async function init() {
  importScripts(BASE + "pyodide.js");
  py = await loadPyodide({ indexURL: BASE });
  py.setStdout({ batched: s => postMessage({ type: "out", id: curId, text: s }) });
  py.setStderr({ batched: s => postMessage({ type: "err", id: curId, text: s }) });
  await py.runPythonAsync(SETUP);
  postMessage({ type: "ready" });
}
const ready = init().catch(e => { postMessage({ type: "fatal", text: String(e && e.message || e) }); throw e; });

function cleanTrace(msg) {
  const lines = String(msg).split("\n");
  // keep the part of the traceback that refers to user code
  let start = lines.findIndex(l => l.includes('File "<exec>"'));
  if (start < 0) start = Math.max(0, lines.length - 6);
  return lines.slice(start).join("\n").trim();
}

async function runChunk(code, ns) {
  await py.loadPackagesFromImports(code, { messageCallback: m => postMessage({ type: "sys", id: curId, text: m }) });
  await py.runPythonAsync(code, { globals: ns });
}

onmessage = async e => {
  const { id, code, check, nsKey } = e.data;
  try { await ready; } catch (err) {
    postMessage({ type: "error", id, stage: "load", isAssert: false, text: "Python could not be loaded (the runtime is fetched from cdn.jsdelivr.net). Check your internet connection or ad-blocker, then press Run again.", figs: [] });
    return;
  }
  curId = id;
  let ns;
  if (nsKey) { if (!keep[nsKey]) keep[nsKey] = py.globals.get("dict")(); ns = keep[nsKey]; }
  else ns = py.globals.get("dict")();
  const t0 = performance.now();
  let stage = "code";
  try {
    await runChunk(code, ns);
    if (check) { stage = "check"; await runChunk(check, ns); }
    const figs = py.globals.get("__mlp_figs")().toJs();
    postMessage({ type: "done", id, figs: Array.from(figs), ms: Math.round(performance.now() - t0), checked: !!check });
  } catch (err) {
    let figs = [];
    try { figs = Array.from(py.globals.get("__mlp_figs")().toJs()); } catch (_) {}
    const isAssert = err && err.type === "AssertionError";
    let text = cleanTrace(err && err.message || err);
    if (isAssert) { const m = text.match(/AssertionError:?\s*([\s\S]*)$/); text = (m && m[1].trim()) || "A check failed."; }
    postMessage({ type: "error", id, stage, isAssert, text, figs });
  } finally {
    if (!nsKey && ns && ns.destroy) { try { ns.destroy(); } catch (_) {} }
  }
};
