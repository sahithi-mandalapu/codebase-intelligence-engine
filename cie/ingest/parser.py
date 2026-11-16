from __future__ import annotations

import ast
import os
import re

from ..models import Chunk, FileInfo, Symbol, file_hash, short_hash

MAX_CHUNK_LINES = 90
WINDOW = 60
OVERLAP = 10


def parse_file(path: str, language: str, text: str) -> FileInfo:
    sha = file_hash(text)
    info = None
    if language == "python":
        info = _parse_python(path, text, sha)
    elif language in ("javascript", "typescript"):
        info = _parse_js(path, language, text, sha)
    elif language == "markdown":
        info = _parse_markdown(path, text, sha)
    if info is None:  # syntax error, or a language we only do line windows for
        info = _parse_generic(path, language, text, sha)
    info.chunks.sort(key=lambda c: (c.start, c.end))
    return info


# ---------------------------------------------------------------- chunk helper

def _make_chunks(path, language, lines, nums, kind, name, qualname, size, overlap=0):
    """nums = sorted 1-indexed line numbers to put into chunks (windowed if too long)"""
    out = []
    if not nums:
        return out
    step = max(1, size - overlap)
    i, part = 0, 0
    while i < len(nums):
        win = nums[i:i + size]
        text = "\n".join(lines[n - 1] for n in win)
        if text.strip():
            # id doesn't include line numbers on purpose, so unchanged code that just
            # moved down the file keeps its embedding
            cid = short_hash(path, kind, qualname or name, str(part), text)
            out.append(Chunk(id=cid, path=path, start=win[0], end=win[-1], text=text,
                             kind=kind, name=name, qualname=qualname, language=language))
        part += 1
        if i + size >= len(nums):
            break
        i += step
    return out


# ---------------------------------------------------------------------- python

def _dotted(node) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        left = _dotted(node.value)
        return f"{left}.{node.attr}" if left else node.attr
    if isinstance(node, ast.Subscript):  # Generic[T]
        return _dotted(node.value)
    return ""


def _py_calls(fn) -> list[str]:
    out = set()
    for n in ast.walk(fn):
        if not isinstance(n, ast.Call):
            continue
        f = n.func
        if isinstance(f, ast.Name):
            out.add(f.id)
        elif isinstance(f, ast.Attribute):
            if isinstance(f.value, ast.Name):
                out.add(f"{f.value.id}.{f.attr}")
            else:
                out.add(f".{f.attr}")  # something like self.a.b() or foo().bar()
    return sorted(out)


def _py_imports(tree, path):
    pkg = path.split("/")[:-1]
    mods, names = [], {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                mods.append(a.name)
                if a.asname:
                    names[a.asname] = f"{a.name}:"
                else:
                    top = a.name.split(".")[0]
                    names[top] = f"{top}:"
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                keep = len(pkg) - (node.level - 1)
                base = pkg[:keep] if keep > 0 else []
                mod = ".".join(base + (node.module.split(".") if node.module else []))
            else:
                mod = node.module or ""
            if mod:
                mods.append(mod)
            for a in node.names:
                if a.name == "*":
                    continue
                names[a.asname or a.name] = f"{mod}:{a.name}"
                mods.append(f"{mod}.{a.name}" if mod else a.name)  # might be a submodule
    return sorted(set(mods)), names


def _parse_python(path, text, sha):
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError):
        return None
    lines = text.splitlines()
    info = FileInfo(path=path, language="python", sha=sha)
    info.imports, info.import_names = _py_imports(tree, path)
    covered = set()

    def visit(body, scope, in_class):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                is_class = isinstance(node, ast.ClassDef)
                qual = ".".join(scope + [node.name])
                kind = "class" if is_class else ("method" if in_class else "function")
                start = min([node.lineno] + [d.lineno for d in node.decorator_list])
                end = node.end_lineno or node.lineno
                sym = Symbol(id=f"{path}::{qual}", path=path, name=node.name, qualname=qual,
                             kind=kind, start=start, end=end, doc=ast.get_docstring(node) or "")
                chunk_end = end
                if is_class:
                    sym.bases = [b for b in (_dotted(x) for x in node.bases) if b]
                    inner = [
                        min([n.lineno] + [d.lineno for d in n.decorator_list])
                        for n in node.body
                        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                    ]
                    if inner:  # class chunk = just the header/attrs, methods get their own chunks
                        chunk_end = max(start, min(inner) - 1)
                    visit(node.body, scope + [node.name], True)
                else:
                    sym.calls = _py_calls(node)
                info.symbols.append(sym)
                info.chunks += _make_chunks(path, "python", lines, list(range(start, chunk_end + 1)),
                                            kind, node.name, qual, MAX_CHUNK_LINES)
                covered.update(range(start, end + 1))
            elif isinstance(node, (ast.If, ast.For, ast.While, ast.With, ast.AsyncWith, ast.AsyncFor)):
                visit(node.body, scope, in_class)
                visit(getattr(node, "orelse", []), scope, in_class)
            elif isinstance(node, ast.Try):
                visit(node.body, scope, in_class)
                for h in node.handlers:
                    visit(h.body, scope, in_class)
                visit(node.orelse, scope, in_class)
                visit(node.finalbody, scope, in_class)

    visit(tree.body, [], False)

    rest = [i for i in range(1, len(lines) + 1) if i not in covered and lines[i - 1].strip()]
    stem = os.path.splitext(os.path.basename(path))[0]
    info.chunks += _make_chunks(path, "python", lines, rest, "module", stem, "", WINDOW)
    return info


# ------------------------------------------------------------------ js / ts
# regex based, so it's best effort. good enough for typical code, not for weird formatting.
# TODO: swap for tree-sitter

_JS_CLASS = re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+([A-Za-z_$][\w$]*)(?:\s+extends\s+([\w$.]+))?")
_JS_FUNC = re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s*\*?\s*([A-Za-z_$][\w$]*)")
_JS_ARROW = re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*(?::[^=]+)?=\s*(?:async\s*)?(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*(?::\s*[^=]+)?=>")
_JS_METHOD = re.compile(r"^\s+(?:static\s+)?(?:async\s+)?(?:get\s+|set\s+)?\*?\s*([A-Za-z_$][\w$]*)\s*\([^)]*\)\s*(?::\s*[^{]+)?\{")
_JS_IMPORT = re.compile(r"""(?:import\s+(?:[^'"]*?\s+from\s+)?|require\(\s*)['"]([^'"]+)['"]""")
_JS_KW = {"if", "for", "while", "switch", "catch", "function", "return", "else", "do", "try", "with", "super", "new", "typeof", "await"}


def _brace_end(lines, i, limit=600):
    depth, seen = 0, False
    for j in range(i, min(len(lines), i + limit)):
        for ch in lines[j]:
            if ch == "{":
                depth += 1
                seen = True
            elif ch == "}":
                depth -= 1
        if seen and depth <= 0:
            return j
        if not seen and lines[j].rstrip().endswith(";"):
            return j
    return i


def _js_calls(body: str) -> list[str]:
    out = set()
    for m in re.finditer(r"(?<![\w$.])([A-Za-z_$][\w$]*)\s*\(", body):
        if m.group(1) not in _JS_KW and m.group(1) not in ("require", "import"):
            out.add(m.group(1))
    for m in re.finditer(r"\bthis\.([A-Za-z_$][\w$]*)\s*\(", body):
        out.add("self." + m.group(1))
    for m in re.finditer(r"(?<!this)\.([A-Za-z_$][\w$]*)\s*\(", body):
        out.add("." + m.group(1))
    return sorted(out)


def _parse_js(path, language, text, sha):
    lines = text.splitlines()
    info = FileInfo(path=path, language=language, sha=sha)
    info.imports = sorted(set(_JS_IMPORT.findall(text)))
    covered = set()
    n = len(lines)
    i = 0
    while i < n:
        line = lines[i]
        m = _JS_CLASS.match(line)
        kind = None
        if m:
            kind, name, base = "class", m.group(1), m.group(2)
        else:
            m = _JS_FUNC.match(line) or _JS_ARROW.match(line)
            if m:
                kind, name, base = "function", m.group(1), None
        if kind is None:
            i += 1
            continue

        end = _brace_end(lines, i)  # 0-indexed
        start1, end1 = i + 1, end + 1
        sym = Symbol(id=f"{path}::{name}", path=path, name=name, qualname=name, kind=kind, start=start1, end=end1)
        chunk_end = end1
        if kind == "class":
            sym.bases = [base] if base else []
            methods = []
            j = i + 1
            while j <= end:
                mm = _JS_METHOD.match(lines[j])
                if mm and mm.group(1) not in _JS_KW:
                    mend = min(_brace_end(lines, j), end)
                    methods.append((mm.group(1), j, mend))
                    j = mend + 1
                else:
                    j += 1
            if methods:
                chunk_end = methods[0][1]  # 1-indexed line right before the first method
            for mname, ms, me in methods:
                q = f"{name}.{mname}"
                body = "\n".join(lines[ms:me + 1])
                msym = Symbol(id=f"{path}::{q}", path=path, name=mname, qualname=q, kind="method",
                              start=ms + 1, end=me + 1, calls=_js_calls(body))
                info.symbols.append(msym)
                info.chunks += _make_chunks(path, language, lines, list(range(ms + 1, me + 2)),
                                            "method", mname, q, MAX_CHUNK_LINES)
        else:
            sym.calls = _js_calls("\n".join(lines[i:end + 1]))
        info.symbols.append(sym)
        info.chunks += _make_chunks(path, language, lines, list(range(start1, max(start1, chunk_end) + 1)),
                                    kind, name, name, MAX_CHUNK_LINES)
        covered.update(range(start1, end1 + 1))
        i = end + 1

    rest = [k for k in range(1, n + 1) if k not in covered and lines[k - 1].strip()]
    stem = os.path.splitext(os.path.basename(path))[0]
    info.chunks += _make_chunks(path, language, lines, rest, "module", stem, "", WINDOW)
    return info


# ------------------------------------------------------------ markdown / other

def _parse_markdown(path, text, sha):
    lines = text.splitlines()
    info = FileInfo(path=path, language="markdown", sha=sha)
    starts = [i + 1 for i, ln in enumerate(lines) if re.match(r"^#{1,6}\s+\S", ln)]
    if not starts or starts[0] != 1:
        starts = [1] + starts
    starts.append(len(lines) + 1)
    for a, b in zip(starts, starts[1:]):
        heading = lines[a - 1].lstrip("# ").strip() if lines and lines[a - 1].startswith("#") else os.path.basename(path)
        info.chunks += _make_chunks(path, "markdown", lines, list(range(a, b)), "doc", heading, "", WINDOW)
    return info


def _parse_generic(path, language, text, sha):
    lines = text.splitlines()
    info = FileInfo(path=path, language=language, sha=sha)
    stem = os.path.splitext(os.path.basename(path))[0]
    info.chunks += _make_chunks(path, language, lines, list(range(1, len(lines) + 1)),
                                "window", stem, "", WINDOW, OVERLAP)
    return info
