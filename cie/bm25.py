from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CAMEL1 = re.compile(r"([a-z0-9])([A-Z])")
_CAMEL2 = re.compile(r"([A-Z]+)([A-Z][a-z])")

STOP = {
    "the", "a", "an", "of", "to", "in", "is", "it", "and", "or", "for", "on", "with", "as", "at",
    "by", "be", "this", "that", "are", "was", "from", "how", "what", "which", "who", "does", "do",
    "where", "why", "when", "self", "cls", "def", "return", "import", "none", "true", "false",
    "if", "else", "not", "can", "my", "me", "all", "any", "use", "used", "uses",
}


def tokenize(text: str) -> list[str]:
    """code aware tokenizer: keeps the full identifier AND its snake/camel pieces.

    parse_file -> parse_file, parse, file     (so "parse" and "file" both hit)
    """
    out = []
    for word in _WORD.findall(text):
        low = word.lower()
        spaced = _CAMEL2.sub(r"\1 \2", _CAMEL1.sub(r"\1 \2", word)).replace("_", " ")
        parts = [p.lower() for p in spaced.split() if p]
        if len(parts) > 1:
            out.append(low)
        out.extend(parts)
    return [t for t in out if len(t) > 1 and t not in STOP]


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.post: dict[str, dict[str, int]] = defaultdict(dict)  # term -> {doc: tf}
        self.dl: dict[str, int] = {}
        self.total_len = 0

    def __contains__(self, doc_id):
        return doc_id in self.dl

    def __len__(self):
        return len(self.dl)

    def add(self, doc_id: str, text: str):
        if doc_id in self.dl:
            self.remove(doc_id)
        toks = tokenize(text)
        self.dl[doc_id] = len(toks)
        self.total_len += len(toks)
        for t, c in Counter(toks).items():
            self.post[t][doc_id] = c

    def remove(self, doc_id: str):
        if doc_id not in self.dl:
            return
        self.total_len -= self.dl.pop(doc_id)
        for t in list(self.post):
            if doc_id in self.post[t]:
                del self.post[t][doc_id]
                if not self.post[t]:
                    del self.post[t]

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        n = len(self.dl)
        if n == 0:
            return []
        avgdl = self.total_len / n or 1.0
        scores: dict[str, float] = defaultdict(float)
        for t in set(tokenize(query)):
            docs = self.post.get(t)
            if not docs:
                continue
            idf = math.log(1 + (n - len(docs) + 0.5) / (len(docs) + 0.5))
            for d, tf in docs.items():
                denom = tf + self.k1 * (1 - self.b + self.b * self.dl[d] / avgdl)
                scores[d] += idf * tf * (self.k1 + 1) / denom
        ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        return ranked[:k]
