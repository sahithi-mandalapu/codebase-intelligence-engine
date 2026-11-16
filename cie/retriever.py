from __future__ import annotations

from .query import QueryPlan, analyze_query
from .reranker import Hit

RRF_K = 60


def fuse(lists):
    """weighted reciprocal rank fusion. lists = [(weight, [chunk_ids], label), ...]"""
    scores: dict[str, float] = {}
    srcs: dict[str, list[str]] = {}
    for weight, ids, label in lists:
        for rank, cid in enumerate(ids):
            scores[cid] = scores.get(cid, 0.0) + weight / (RRF_K + rank + 1)
            s = srcs.setdefault(cid, [])
            if label not in s:
                s.append(label)
    order = sorted(scores, key=lambda c: (-scores[c], c))
    return order, scores, srcs


class HybridRetriever:
    """dense + bm25 (rrf) -> graph evidence -> rerank.

    graph evidence comes in three flavours:
      graph:def     chunks for symbols/files the question names
      graph:rel     chunks reached through a relation in the question ("who calls X")
      graph:expand  callers/callees/bases of the top results, low weight, adds context
    """

    def __init__(self, index, reranker=None, dense_k: int = 40, sparse_k: int = 40, rerank_depth: int = 30):
        self.index = index
        self.reranker = reranker
        self.dense_k, self.sparse_k, self.rerank_depth = dense_k, sparse_k, rerank_depth

    def retrieve(self, query: str, k: int = 8, use_dense: bool = True, use_sparse: bool = True,
                 use_graph: bool = True, use_rerank: bool = True, plan: QueryPlan | None = None) -> list[Hit]:
        idx = self.index
        lists = []
        if use_dense:
            lists.append((1.0, [c for c, _ in idx.search_dense(query, self.dense_k)], "dense"))
        if use_sparse:
            lists.append((1.0, [c for c, _ in idx.search_sparse(query, self.sparse_k)], "bm25"))

        if plan is None:
            plan = analyze_query(query, idx.graph)

        if use_graph:
            defs = self._definition_chunks(plan)
            if defs:
                lists.append((1.5, defs, "graph:def"))
            rel = self._relation_chunks(plan)
            if rel:
                lists.append((2.0, rel, "graph:rel"))

        order, scores, srcs = fuse(lists)

        if use_graph and order:
            exp = self._expand(order[:5])
            if exp:
                lists.append((0.5, exp, "graph:expand"))
                order, scores, srcs = fuse(lists)

        depth = self.rerank_depth if (use_rerank and self.reranker) else k
        hits = [Hit(idx.chunks[c], scores[c], srcs[c]) for c in order[:max(depth, k)] if c in idx.chunks]
        if use_rerank and self.reranker:
            hits = self.reranker.rerank(query, hits, plan)
        return hits[:k]

    # ----------------------------------------------------------------- graph

    def _definition_chunks(self, plan: QueryPlan) -> list[str]:
        out = []
        for sid in plan.symbols:
            c = self.index.symbol_chunk(sid)
            if c and c not in out:
                out.append(c)
        for p in plan.files:
            c = self.index.file_chunk(p)
            if c and c not in out:
                out.append(c)
        return out

    def _relation_chunks(self, plan: QueryPlan) -> list[str]:
        if not plan.relations:
            return []
        g, idx = self.index.graph, self.index
        out: list[str] = []

        def add(cid):
            if cid and cid not in out:
                out.append(cid)

        for rel, direction in plan.relations:
            if rel == "imports":
                paths = list(plan.files) + [g.symbols[s].path for s in plan.symbols]
                for p in paths:
                    nbrs = g.imports(p) if direction == "out" else g.importers(p)
                    for q in nbrs:
                        add(idx.file_chunk(q))
            else:
                for sid in plan.symbols:
                    if rel == "calls":
                        nbrs = g.callees(sid) if direction == "out" else g.callers(sid)
                    else:
                        nbrs = g.bases(sid) if direction == "out" else g.subclasses(sid)
                    for n in nbrs:
                        add(idx.symbol_chunk(n))
        return out[:25]

    def _expand(self, top_ids: list[str]) -> list[str]:
        g, idx = self.index.graph, self.index
        out: list[str] = []
        for cid in top_ids:
            c = idx.chunks[cid]
            sid = f"{c.path}::{c.qualname}" if c.qualname else None
            if not sid or sid not in g.symbols:
                continue
            nb = g.neighbors(sid)
            for key in ("callees", "callers", "bases"):
                for n in nb[key][:3]:
                    t = idx.symbol_chunk(n)
                    if t and t not in out:
                        out.append(t)
        return out
