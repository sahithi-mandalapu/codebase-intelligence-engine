from __future__ import annotations

import posixpath
from collections import defaultdict

import networkx as nx

from .models import FileInfo, Symbol

# method names that are almost always stdlib / builtin calls. if a call looks like
# `something.get()` we don't want to link it to some random get() in the repo
COMMON_METHODS = {
    "get", "set", "add", "append", "extend", "pop", "remove", "update", "items", "keys", "values",
    "join", "split", "strip", "format", "read", "write", "open", "close", "copy", "clear", "sort",
    "insert", "replace", "lower", "upper", "startswith", "endswith", "encode", "decode", "next",
    "dump", "dumps", "load", "loads", "exists", "mkdir", "info", "debug", "warning", "error",
    "then", "catch", "map", "filter", "reduce", "push", "slice", "includes", "forEach", "count",
    "index", "find", "send", "flush", "seek", "tell", "run", "start", "stop", "wait", "emit",
}

EDGE_KINDS = ("calls", "imports", "inherits")


def fid(path: str) -> str:
    return "file:" + path


class CodeGraph:
    """files + symbols as nodes, edges are contains / imports / calls / inherits.

    built from the parsed FileInfo objects, so it's cheap to rebuild after an incremental update.
    """

    def __init__(self):
        self.g = nx.MultiDiGraph()
        self.symbols: dict[str, Symbol] = {}
        self.files: dict[str, FileInfo] = {}
        self.by_name: dict[str, list[str]] = defaultdict(list)
        self._mod_exact: dict[str, str] = {}
        self._mod_suffix: dict[str, set] = defaultdict(set)
        self._flow = None

    # ------------------------------------------------------------------ build

    def build(self, files: dict[str, FileInfo]) -> "CodeGraph":
        self.__init__()
        self.files = files
        for path, f in files.items():
            self.g.add_node(fid(path), type="file", path=path, language=f.language)
            for s in f.symbols:
                self.symbols[s.id] = s
                self.by_name[s.name].append(s.id)
                self.g.add_node(s.id, type="symbol", kind=s.kind)
                self.g.add_edge(fid(path), s.id, key="contains")
        self._index_modules()
        for f in files.values():
            self._link_imports(f)
        for f in files.values():
            for s in f.symbols:
                if s.kind == "class":
                    self._link_inherits(f, s)
                else:
                    self._link_calls(f, s)
        return self

    def _index_modules(self):
        for path, f in self.files.items():
            if f.language != "python":
                continue
            dotted = path[:-3].replace("/", ".")
            if dotted.endswith(".__init__"):
                dotted = dotted[: -len(".__init__")]
            keys = {dotted}
            if dotted.startswith("src."):
                keys.add(dotted[4:])
            for k in keys:
                self._mod_exact.setdefault(k, path)
            parts = dotted.split(".")
            for i in range(1, len(parts) - 0):
                suffix = ".".join(parts[i:])
                if "." in suffix:  # only suffixes with 2+ parts, single names are way too ambiguous
                    self._mod_suffix[suffix].add(path)

    def resolve_module(self, dotted: str):
        if not dotted:
            return None
        if dotted in self._mod_exact:
            return self._mod_exact[dotted]
        cands = self._mod_suffix.get(dotted)
        if cands and len(cands) == 1:
            return next(iter(cands))
        return None

    def _resolve_js(self, spec: str, from_path: str):
        if not spec.startswith("."):
            return None
        base = posixpath.normpath(posixpath.join(posixpath.dirname(from_path), spec))
        exts = ["", ".js", ".jsx", ".ts", ".tsx", ".mjs", "/index.js", "/index.ts", "/index.tsx"]
        for e in exts:
            if base + e in self.files:
                return base + e
        return None

    def _link_imports(self, f: FileInfo):
        targets = set()
        for m in f.imports:
            if f.language == "python":
                p = self.resolve_module(m)
            elif f.language in ("javascript", "typescript"):
                p = self._resolve_js(m, f.path)
            else:
                p = None
            if p and p != f.path:
                targets.add(p)
        # `from pkg import mod` also lists `pkg` itself, drop the package __init__ if we also hit a child module
        for p in list(targets):
            if p.endswith("/__init__.py"):
                pkg_dir = p[: -len("__init__.py")]
                if any(q != p and q.startswith(pkg_dir) for q in targets):
                    targets.discard(p)
        for p in sorted(targets):
            self.g.add_edge(fid(f.path), fid(p), key="imports")

    # --------------------------------------------------------------- resolving

    def _symbol_in(self, path: str, qualname: str):
        sid = f"{path}::{qualname}"
        return sid if sid in self.symbols else None

    def _unique(self, name: str, kinds):
        ids = [i for i in self.by_name.get(name, []) if self.symbols[i].kind in kinds]
        return ids[0] if len(ids) == 1 else None

    def _unique_attr(self, attr: str):
        if attr in COMMON_METHODS or len(attr) < 4:
            return None
        return self._unique(attr, ("method", "function"))

    def _resolve_class(self, name: str, f: FileInfo):
        last = name.split(".")[-1]
        cand = self._symbol_in(f.path, last)
        if cand and self.symbols[cand].kind == "class":
            return cand
        imp = f.import_names.get(name.split(".")[0])
        if imp:
            mod, _, orig = imp.partition(":")
            p = self.resolve_module(mod)
            if p:
                cand = self._symbol_in(p, orig or last)
                if cand and self.symbols[cand].kind == "class":
                    return cand
        return self._unique(last, ("class",))

    def _link_inherits(self, f: FileInfo, s: Symbol):
        for b in s.bases:
            t = self._resolve_class(b, f)
            if t and t != s.id:
                self.g.add_edge(s.id, t, key="inherits")

    def _bases_of(self, sid):
        return [v for _, v, k in self.g.out_edges(sid, keys=True) if k == "inherits"]

    def _method_via_bases(self, cls_id, attr, depth=0):
        if depth > 5 or cls_id is None:
            return None
        for b in self._bases_of(cls_id):
            bs = self.symbols[b]
            cand = self._symbol_in(bs.path, f"{bs.qualname}.{attr}")
            if cand:
                return cand
            cand = self._method_via_bases(b, attr, depth + 1)
            if cand:
                return cand
        return None

    def _resolve_call(self, f: FileInfo, s: Symbol, c: str):
        if "." not in c:
            cand = self._symbol_in(f.path, c)
            if cand:
                return cand
            imp = f.import_names.get(c)
            if imp:
                mod, _, orig = imp.partition(":")
                p = self.resolve_module(mod)
                if p and orig:
                    return self._symbol_in(p, orig)
                return None  # external library
            return self._unique(c, ("function", "class"))

        root, _, attr = c.partition(".")
        if root == "":
            return self._unique_attr(attr)
        if root in ("self", "cls"):
            if "." in s.qualname:
                cls_q = s.qualname.rsplit(".", 1)[0]
                cand = self._symbol_in(f.path, f"{cls_q}.{attr}")
                if cand:
                    return cand
                cand = self._method_via_bases(self._symbol_in(f.path, cls_q), attr)
                if cand:
                    return cand
            return self._unique_attr(attr)
        imp = f.import_names.get(root)
        if imp:
            mod, _, orig = imp.partition(":")
            if orig:  # `from pkg import utils` where utils is a submodule, then utils.func()
                sub = self.resolve_module(f"{mod}.{orig}")
                if sub:
                    return self._symbol_in(sub, attr)
            p = self.resolve_module(mod)
            if p:
                if not orig:  # `import x` / `import x as y`, then y.func()
                    return self._symbol_in(p, attr)
                return self._symbol_in(p, f"{orig}.{attr}")  # `from x import Cls`, then Cls.method()
            return None  # external library, ignore
        cand = self._symbol_in(f.path, f"{root}.{attr}")
        if cand:
            return cand
        return self._unique_attr(attr)

    def _link_calls(self, f: FileInfo, s: Symbol):
        for c in s.calls:
            t = self._resolve_call(f, s, c)
            if t and t != s.id:
                self.g.add_edge(s.id, t, key="calls")

    # ------------------------------------------------------------------ queries

    def _nb(self, node, kind, direction):
        if node not in self.g:
            return []
        if direction == "out":
            vs = [v for _, v, k in self.g.out_edges(node, keys=True) if k == kind]
        else:
            vs = [u for u, _, k in self.g.in_edges(node, keys=True) if k == kind]
        return sorted(set(vs))

    def callers(self, sid): return self._nb(sid, "calls", "in")
    def callees(self, sid): return self._nb(sid, "calls", "out")
    def subclasses(self, sid): return self._nb(sid, "inherits", "in")
    def bases(self, sid): return self._nb(sid, "inherits", "out")
    def imports(self, path): return [p[5:] for p in self._nb(fid(path), "imports", "out")]
    def importers(self, path): return [p[5:] for p in self._nb(fid(path), "imports", "in")]

    def find_symbols(self, query: str) -> list[str]:
        q = query.strip().strip("`").rstrip("()")
        if q in self.symbols:
            return [q]
        if q in self.by_name:
            return sorted(self.by_name[q])
        hits = [i for i, s in self.symbols.items() if s.qualname == q or s.qualname.endswith("." + q)]
        if hits:
            return sorted(hits)
        # "module.func" style, e.g. indexer.Index
        if "." in q:
            tail = q.split(".")[-1]
            head = q.split(".")[-2]
            hits = [i for i in self.by_name.get(tail, []) if head in self.symbols[i].path.lower()]
            if hits:
                return sorted(hits)
        low = q.lower()
        return sorted(i for n, ids in self.by_name.items() if n.lower() == low for i in ids)

    def find_files(self, query: str) -> list[str]:
        q = query.strip().strip("`")
        if q in self.files:
            return [q]
        hits = [p for p in self.files if p.endswith("/" + q)]
        if hits:
            return sorted(hits)
        # bare module name like "parser" -> only if unambiguous
        stems = [p for p in self.files if p.rsplit("/", 1)[-1].rsplit(".", 1)[0] == q]
        return sorted(stems) if len(stems) == 1 else []

    def neighbors(self, sid: str) -> dict:
        return {
            "callers": self.callers(sid),
            "callees": self.callees(sid),
            "bases": self.bases(sid),
            "subclasses": self.subclasses(sid),
        }

    def _flow_graph(self):
        if self._flow is None:
            fg = nx.DiGraph()
            for u, v, k in self.g.edges(keys=True):
                if k in EDGE_KINDS:
                    fg.add_edge(u, v, kind=k)
            self._flow = fg
        return self._flow

    def trace(self, src: str, dst: str):
        """shortest dependency path between two symbols / files. returns None if nothing found"""
        a = self._resolve_node(src)
        b = self._resolve_node(dst)
        if not a or not b:
            return None
        fg = self._flow_graph()
        if a not in fg or b not in fg:
            return None
        try:
            path = nx.shortest_path(fg, a, b)
            directed = True
        except nx.NetworkXNoPath:
            try:
                path = nx.shortest_path(fg.to_undirected(), a, b)
                directed = False
            except nx.NetworkXNoPath:
                return None
        edges = []
        for u, v in zip(path, path[1:]):
            if fg.has_edge(u, v):
                edges.append(fg[u][v]["kind"])
            else:
                edges.append(fg[v][u]["kind"] + " (reverse)")
        return {"path": path, "edges": edges, "directed": directed}

    def _resolve_node(self, text: str):
        syms = self.find_symbols(text)
        if syms:
            return syms[0]
        files = self.find_files(text)
        if files:
            return fid(files[0])
        return None

    def stats(self) -> dict:
        kinds = defaultdict(int)
        for _, _, k in self.g.edges(keys=True):
            kinds[k] += 1
        return {
            "files": len(self.files),
            "symbols": len(self.symbols),
            "edges": dict(kinds),
        }
