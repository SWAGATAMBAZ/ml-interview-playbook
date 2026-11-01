#!/usr/bin/env python3
"""Build helper for the ML Interview Playbook.

  python3 build.py            -> assign question ids, write assets/qbank.js, search-index.js, labs-index.js
  python3 build.py --test     -> also execute every lab (solution must pass its check; starter must not)
  python3 build.py --test lr- -> only labs whose id starts with the prefix
"""
import hashlib, html, json, os, re, subprocess, sys, tempfile, textwrap

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = os.environ.get("LAB_PY", os.path.join(ROOT, "..", "labenv", "bin", "python"))

pages_js = open(os.path.join(ROOT, "assets", "app.js"), encoding="utf-8").read()
PAGES = re.findall(r'\{ id: "([^"]+)", file: "([^"]+)", title: "([^"]+)", group: "([^"]+)" \}', pages_js)

QA_RE = re.compile(r'<details class="qa"([^>]*)>\s*<summary>(.*?)</summary>\s*<div class="ans">(.*?)</div></details>', re.S)
TAG_RE = re.compile(r"<[^>]+>")


def text(s):
    return re.sub(r"\s+", " ", html.unescape(TAG_RE.sub(" ", s))).strip()


def dedent(s):
    s = s.strip("\n").rstrip()
    return textwrap.dedent(s)


def main():
    qbank, search, labs = [], [], []
    for pid, fname, ptitle, group in PAGES:
        path = os.path.join(ROOT, fname)
        if not os.path.exists(path):
            print("  (missing page)", fname)
            continue
        src = open(path, encoding="utf-8").read()
        search.append({"k": "page", "t": ptitle, "p": fname, "g": group})

        # 1) stable ids for questions
        def add_id(m):
            attrs, q, a = m.group(1), m.group(2), m.group(3)
            if "data-id=" not in attrs:
                h = hashlib.md5(text(q).encode()).hexdigest()[:7]
                attrs = f' data-id="{pid}-{h}"' + attrs
            return f'<details class="qa"{attrs}><summary>{q}</summary><div class="ans">{a}</div></details>'
        new = QA_RE.sub(add_id, src)
        if new != src:
            open(path, "w", encoding="utf-8").write(new)
            src = new

        for m in QA_RE.finditer(src):
            attrs, q, a = m.groups()
            qid = re.search(r'data-id="([^"]+)"', attrs).group(1)
            lv = re.search(r'data-l="(\d)"', attrs)
            a_clean = re.sub(r"<label class=\"known\".*?</label>", "", a)
            qbank.append({"id": qid, "p": fname, "g": ptitle, "grp": group, "l": int(lv.group(1)) if lv else 2, "q": q.strip(), "a": a_clean.strip()})
            search.append({"k": "question", "t": text(q), "p": fname, "h": "q-" + qid, "g": ptitle, "x": text(a)[:260]})

        for m in re.finditer(r'<h([23]) id="([^"]+)"[^>]*>(.*?)</h\1>', src, re.S):
            search.append({"k": "section", "t": text(m.group(3)), "p": fname, "h": m.group(2), "g": ptitle})

        # labs
        for m in re.finditer(r'<div class="lab" data-id="([^"]+)"([^>]*)>(.*?)(?=<div class="lab" data-id=|<h2|</main>)', src, re.S):
            lid, attrs, body = m.groups()
            title = re.search(r'data-title="([^"]*)"', attrs)
            title = html.unescape(title.group(1)) if title else lid
            parts = {k: dedent(v) for k, v in re.findall(r'<script type="text/x-python" data-part="(\w+)">(.*?)</script>', body, re.S)}
            nsm = re.search(r'data-ns="([^"]+)"', attrs)
            labs.append({"id": lid, "t": title, "p": fname, "g": ptitle, "parts": parts, "ns": nsm.group(1) if nsm else None})
            search.append({"k": "lab", "t": title, "p": fname, "h": "lab-" + lid, "g": ptitle})

    with open(os.path.join(ROOT, "assets", "qbank.js"), "w", encoding="utf-8") as f:
        f.write("window.MLP_QBANK=" + json.dumps(qbank, ensure_ascii=False) + ";\n")
    with open(os.path.join(ROOT, "assets", "search-index.js"), "w", encoding="utf-8") as f:
        f.write("window.MLP_SEARCH=" + json.dumps(search, ensure_ascii=False) + ";\n")
    with open(os.path.join(ROOT, "assets", "labs-index.js"), "w", encoding="utf-8") as f:
        f.write("window.MLP_LABS=" + json.dumps([{k: l[k] for k in ("id", "t", "p", "g")} for l in labs], ensure_ascii=False) + ";\n")
    ids = [l["id"] for l in labs]
    dup = {i for i in ids if ids.count(i) > 1}
    print(f"pages={len(PAGES)} questions={len(qbank)} labs={len(labs)} search={len(search)}" + (f"  DUPLICATE LAB IDS: {dup}" if dup else ""))
    return labs


def run_py(code, timeout=240):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
        fn = f.name
    try:
        r = subprocess.run([PY, fn], capture_output=True, text=True, timeout=timeout, env={**os.environ, "MPLBACKEND": "Agg", "PYTHONWARNINGS": "ignore"})
        return r.returncode, (r.stdout + r.stderr)
    except subprocess.TimeoutExpired:
        return -9, "TIMEOUT"
    finally:
        os.unlink(fn)


def test(labs, prefix=""):
    bad = 0
    acc = {}
    for l in labs:
        if prefix and not l["id"].startswith(prefix):
            continue
        p = l["parts"]
        setup = p.get("setup", "")
        sol, chk, st = p.get("solution", ""), p.get("check", ""), p.get("starter", "")
        try:
            compile(setup + "\n" + st, l["id"] + ":starter", "exec")
        except SyntaxError as e:
            print(f"✗ {l['id']}: starter has a syntax error: {e}"); bad += 1; continue
        if not sol:
            pre = acc.get(l["ns"], "") if l["ns"] else ""
            if l["ns"]:
                acc[l["ns"]] = pre + "\n" + setup + "\n" + st
            code, out = run_py(pre + "\n" + setup + "\n" + st)
            status = "✓" if code == 0 else "✗"
            if code: bad += 1
            print(f"{status} {l['id']} (free run) {out.strip().splitlines()[-1] if out.strip() and code else ''}")
            continue
        code, out = run_py(setup + "\n" + sol + "\n" + chk)
        if code != 0:
            bad += 1
            print(f"✗ {l['id']}: SOLUTION FAILED\n" + "\n".join(out.strip().splitlines()[-8:]))
            continue
        msg = f"✓ {l['id']}"
        if chk:
            c2, o2 = run_py(setup + "\n" + st + "\n" + chk, timeout=120)
            if c2 == 0:
                msg += "   ⚠ starter ALSO passes the check"
                bad += 1
        print(msg)
    print("problems:", bad)
    return bad


if __name__ == "__main__":
    labs = main()
    if "--test" in sys.argv:
        i = sys.argv.index("--test")
        pre = sys.argv[i + 1] if len(sys.argv) > i + 1 else ""
        sys.exit(1 if test(labs, pre) else 0)
