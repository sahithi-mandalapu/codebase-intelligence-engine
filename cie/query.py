from __future__ import annotations

import re
from dataclasses import dataclass, field

# things that look like code identifiers even without backticks
_CODEISH = re.compile(
    r"[\w./-]+\.(?:py|js|ts|tsx|jsx|go|java|md)\b"                       # file names
    r"|\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+\b"                              # a.b.c
    r"|\b[a-z]+_[A-Za-z0-9_]+\b|\b_[A-Za-z0-9_]+\b"                      # snake_case / _private
    r"|\b[a-z]+[A-Z][A-Za-z0-9]*\b|\b[A-Z][a-z0-9]+[A-Z][A-Za-z0-9]*\b"  # camelCase / PascalCase
)
_BACKTICK = re.compile(r"`([^`\n]+)`")
_PLAIN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

_WEAK_STOP = {
    "what", "which", "where", "when", "does", "this", "that", "with", "from", "have", "show", "work",
    "works", "code", "file", "files", "class", "function", "method", "call", "calls", "used", "uses",
    "import", "imports", "inherit", "extend", "extends", "defined", "about", "into", "than", "then",
}

# forward = "what does X call/import/extend", reverse = "what calls/imports/extends X"
_FWD = {
    "calls": [r"\b(?:does|do|did)\s+[\w`.:'\"/-]+\s+(?:call|invoke|use)\b", r"\bcallees?\b", r"\bcalls what\b"],
    "imports": [r"\b(?:does|do)\s+[\w`./-]+\s+import\b", r"\bdependenc(?:y|ies)\b", r"\b(?:does|do)\s+[\w`./-]+\s+depend on\b"],
    "inherits": [r"\b(?:does|do)\s+[\w`.:'\"-]+\s+(?:extend|inherit)", r"\b(?:base|parent|super)\s*class(?:es)?\s+of\b"],
}
_REV = {
    "calls": [r"\b(?:who|what|which|where)\b[^?.]*\b(?:calls?|invokes?|invoked|uses?|used|called)\b", r"\bcallers?\b"],
    "imports": [r"\b(?:who|what|which)\b[^?.]*\bimports?\b", r"\bimported by\b", r"\bwhat depends on\b", r"\bdependents?\b"],
    "inherits": [r"\bsub-?class(?:es)?\b", r"\b(?:who|what|which)\b[^?.]*\b(?:extends?|inherits?|derives?)\b", r"\bchild classes\b"],
}


@dataclass
class QueryPlan:
    idents: list[str] = field(default_factory=list)       # strong identifiers the user mentioned
    symbols: list[str] = field(default_factory=list)      # resolved symbol ids
    files: list[str] = field(default_factory=list)        # resolved file paths
    relations: list[tuple[str, str]] = field(default_factory=list)  # (calls|imports|inherits, in|out)

    @property
    def structural(self) -> bool:
        return bool(self.relations and (self.symbols or self.files))


def detect_relations(q: str) -> list[tuple[str, str]]:
    ql = q.lower()
    out = []
    for rel in ("calls", "imports", "inherits"):
        if any(re.search(p, ql) for p in _FWD[rel]):
            out.append((rel, "out"))
        elif any(re.search(p, ql) for p in _REV[rel]):
            out.append((rel, "in"))
    return out


def extract_idents(q: str) -> list[str]:
    seen, out = set(), []
    for m in _BACKTICK.finditer(q):
        s = m.group(1).strip().rstrip("()")
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    for m in _CODEISH.finditer(q):
        s = m.group(0).strip(".")
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def analyze_query(query: str, graph) -> QueryPlan:
    plan = QueryPlan(relations=detect_relations(query))
    plan.idents = extract_idents(query)

    for ident in plan.idents:
        files = graph.find_files(ident) if ("/" in ident or "." in ident) else []
        syms = graph.find_symbols(ident)
        # a file name like "parser.py" shouldn't also match a symbol called "parser"
        if files and ident.rsplit(".", 1)[-1] in ("py", "js", "ts", "tsx", "jsx", "go", "java", "md"):
            syms = []
        for p in files[:3]:
            if p not in plan.files:
                plan.files.append(p)
        for s in syms[:5]:
            if s not in plan.symbols:
                plan.symbols.append(s)

    # plain words only count if they exactly match a (rare) symbol name, e.g. "where is parse_file used"
    # is covered above, but "what does sync do" needs this
    if not plan.symbols:
        for w in _PLAIN.findall(query):
            lw = w.lower()
            if len(w) < 4 or lw in _WEAK_STOP:
                continue
            ids = graph.by_name.get(w, [])
            if 0 < len(ids) <= 2:
                for s in ids:
                    if s not in plan.symbols:
                        plan.symbols.append(s)
    return plan
