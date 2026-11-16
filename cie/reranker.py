from __future__ import annotations

import math

from .bm25 import tokenize
from .models import Chunk


class Hit:
    def __init__(self, chunk: Chunk, score: float, sources: list[str]):
        self.chunk = chunk
        self.score = score
        self.sources = sources

    def to_dict(self) -> dict:
        c = self.chunk
        return {
            "path": c.path, "start": c.start, "end": c.end, "kind": c.kind,
            "name": c.qualname or c.name, "score": round(self.score, 4), "sources": self.sources,
        }


def _structural(h: Hit) -> float:
    # came out of the code graph because of a relation in the question (callers of X etc)
    return 1.0 if "graph:rel" in h.sources else 0.0


class LexicalReranker:
    """cheap, deterministic reranker. idf-weighted query coverage + name/path match + a bit of the
    original fused rank. not as good as a cross-encoder but needs no model download."""

    name = "lexical"

    def rerank(self, query: str, hits: list[Hit], plan=None) -> list[Hit]:
        qt = set(tokenize(query))
        if not hits or not qt:
            return hits
        docs = [set(tokenize(h.chunk.search_text())) for h in hits]
        n = len(hits)
        w = {t: math.log(1 + n / (1 + sum(1 for d in docs if t in d))) for t in qt}
        wsum = sum(w.values()) or 1.0
        strong = [s.lower().split(".")[-1] for s in (plan.idents if plan else [])]
        # "what does X call" / "who imports Y": the name in the question is the *anchor*, not the
        # answer, so matching it by name would just promote the anchor's own chunk. in that case
        # lean on the graph evidence instead of lexical overlap
        relational = bool(plan and plan.structural)
        out = []
        for rank, (h, d) in enumerate(zip(hits, docs)):
            cov = sum(w[t] for t in qt if t in d) / wsum
            name_toks = set(tokenize(f"{h.chunk.qualname} {h.chunk.name} {h.chunk.path}"))
            name_hit = len(qt & name_toks) / len(qt)
            body = h.chunk.text.lower()
            ident = 1.0 if any(s and (s in body or s in h.chunk.qualname.lower()) for s in strong) else 0.0
            prior = 1.0 - rank / n
            if relational:
                score = 0.25 * cov + 0.05 * name_hit + 0.40 * _structural(h) + 0.30 * prior
            else:
                score = 0.40 * cov + 0.20 * name_hit + 0.10 * ident + 0.15 * _structural(h) + 0.25 * prior
            out.append(Hit(h.chunk, score, h.sources))
        out.sort(key=lambda x: -x.score)
        return out


class CrossEncoderReranker:
    name = "cross"

    def __init__(self, model: str = "BAAI/bge-reranker-base"):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model)

    def rerank(self, query: str, hits: list[Hit], plan=None) -> list[Hit]:
        if not hits:
            return hits
        raw = self.model.predict([(query, h.chunk.search_text()[:2000]) for h in hits])
        out = []
        for h, s in zip(hits, raw):
            prob = 1 / (1 + math.exp(-float(s)))
            out.append(Hit(h.chunk, prob + 0.15 * _structural(h), h.sources))
        out.sort(key=lambda x: -x.score)
        return out


def get_reranker(name: str = "lexical"):
    if name in ("none", "", None):
        return None
    if name == "lexical":
        return LexicalReranker()
    if name.startswith("cross"):
        model = name.split(":", 1)[1] if ":" in name else "BAAI/bge-reranker-base"
        return CrossEncoderReranker(model)
    raise ValueError(f"unknown reranker '{name}' (use lexical, cross or none)")
