from __future__ import annotations

import random
import re

# Questions are generated from the repo's own AST + code graph, so you can point this at any
# python repo and get a few hundred questions without labeling anything by hand.
#
# caveat (also in the README): the ground truth for the callers/callees/imports/inherits
# questions comes from the same graph the retriever uses, so those numbers say "does retrieval
# surface what static analysis found", not "is static analysis right". the docstring and
# definition questions are independent of the graph. for real claims use a hand labeled set.

CALLER_TEMPLATES = [
    "Which functions call `{n}`?",
    "Where is `{n}` called from?",
    "What uses `{n}`?",
]
CALLEE_TEMPLATES = [
    "What does `{n}` call?",
    "Which functions does `{n}` call?",
]
IMPORT_TEMPLATES = [
    "Which files import `{n}`?",
    "What imports `{n}`?",
]
INHERIT_TEMPLATES = [
    "Which classes extend `{n}`?",
    "What are the subclasses of `{n}`?",
]
DEF_TEMPLATES = [
    "Where is `{n}` defined?",
    "Show me the implementation of `{n}`.",
]


def _is_test(path: str) -> bool:
    low = path.lower()
    return "/tests/" in low or low.startswith("tests/") or "/test_" in low or low.rsplit("/", 1)[-1].startswith("test_")


def _first_sentence(doc: str) -> str:
    doc = " ".join(doc.split())
    m = re.match(r"(.+?[.!?])(\s|$)", doc)
    return (m.group(1) if m else doc)[:200]


def _unique_name(graph, sym) -> bool:
    return len([i for i in graph.by_name[sym.name] if graph.symbols[i].kind == sym.kind]) == 1


def generate(index, n: int = 120, seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    g = index.graph
    syms = [s for s in g.symbols.values() if not _is_test(s.path) and not s.name.startswith("__")]
    syms.sort(key=lambda s: s.id)

    pools: dict[str, list[dict]] = {k: [] for k in ("docstring", "definition", "callers", "callees", "imports", "inherits")}

    for s in syms:
        if s.doc:
            first = _first_sentence(s.doc)
            if len(first.split()) >= 5:
                pools["docstring"].append({
                    "type": "docstring",
                    "question": f"Where is the {s.kind} that does the following: {first}",
                    "relevant": [{"path": s.path, "qualname": s.qualname}],
                })
        if _unique_name(g, s) and len(s.name) >= 4:
            pools["definition"].append({
                "type": "definition",
                "question": rng.choice(DEF_TEMPLATES).format(n=s.name),
                "relevant": [{"path": s.path, "qualname": s.qualname}],
            })
        if _unique_name(g, s) and len(s.name) >= 4 and s.kind != "class":
            callers = [c for c in g.callers(s.id) if not _is_test(g.symbols[c].path)]
            if 1 <= len(callers) <= 6:
                pools["callers"].append({
                    "type": "callers",
                    "question": rng.choice(CALLER_TEMPLATES).format(n=s.name),
                    "relevant": [{"path": g.symbols[c].path, "qualname": g.symbols[c].qualname} for c in callers],
                })
            callees = g.callees(s.id)
            if 2 <= len(callees) <= 6:
                pools["callees"].append({
                    "type": "callees",
                    "question": rng.choice(CALLEE_TEMPLATES).format(n=s.name),
                    "relevant": [{"path": g.symbols[c].path, "qualname": g.symbols[c].qualname} for c in callees],
                })
        if s.kind == "class" and _unique_name(g, s):
            subs = [c for c in g.subclasses(s.id) if not _is_test(g.symbols[c].path)]
            if subs:
                pools["inherits"].append({
                    "type": "inherits",
                    "question": rng.choice(INHERIT_TEMPLATES).format(n=s.name),
                    "relevant": [{"path": g.symbols[c].path, "qualname": g.symbols[c].qualname} for c in subs],
                })

    for path in sorted(g.files):
        if _is_test(path) or g.files[path].language != "python":
            continue
        importers = [p for p in g.importers(path) if not _is_test(p)]
        stem = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        if stem != "__init__" and 2 <= len(importers) <= 8:
            pools["imports"].append({
                "type": "imports",
                "question": rng.choice(IMPORT_TEMPLATES).format(n=path),
                "relevant": [{"path": p} for p in importers],
            })

    for p in pools.values():
        rng.shuffle(p)

    weights = {"docstring": 0.30, "definition": 0.15, "callers": 0.20, "callees": 0.15, "imports": 0.10, "inherits": 0.10}
    out, leftovers = [], []
    for kind, w in weights.items():
        take = int(round(n * w))
        out += pools[kind][:take]
        leftovers += pools[kind][take:]
    rng.shuffle(leftovers)
    out += leftovers[: max(0, n - len(out))]
    out = out[:n]
    for i, item in enumerate(out):
        item["id"] = f"q{i:03d}"
    return out
