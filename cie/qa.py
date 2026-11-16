from __future__ import annotations

import os
import re
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field

from .cache import get_cache, make_key
from .config import Settings
from .embeddings import get_embedder
from .indexer import Index
from .ingest import clone
from .llm import get_llm
from .query import QueryPlan, analyze_query
from .reranker import get_reranker
from .retriever import HybridRetriever

MAX_CONTEXT_CHARS = 24_000


@dataclass
class Answer:
    question: str
    answer: str
    citations: list[dict] = field(default_factory=list)
    context: list[dict] = field(default_factory=list)
    graph_notes: str = ""
    cached: bool = False
    timings_ms: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _short(ids, limit=8):
    names = [i.split("::", 1)[-1] if "::" in i else i for i in ids]
    extra = f" (+{len(names) - limit} more)" if len(names) > limit else ""
    return ", ".join(f"`{n}`" for n in names[:limit]) + extra


def graph_notes(index: Index, plan: QueryPlan) -> str:
    """short text summary of what the code graph knows about the things the question mentions"""
    g = index.graph
    lines = []
    for sid in plan.symbols[:3]:
        s = g.symbols[sid]
        lines.append(f"- `{s.qualname}` is a {s.kind} in {s.path}:{s.start}-{s.end}")
        nb = g.neighbors(sid)
        for key, label in (("callers", "called by"), ("callees", "calls"), ("bases", "extends"),
                           ("subclasses", "extended by")):
            if nb[key]:
                lines.append(f"  - {label}: {_short(nb[key])}")
    for p in plan.files[:2]:
        lines.append(f"- file {p}")
        if g.imports(p):
            lines.append(f"  - imports: {_short(g.imports(p))}")
        if g.importers(p):
            lines.append(f"  - imported by: {_short(g.importers(p))}")
    return "\n".join(lines)


class Engine:
    """ties everything together: index + retriever + llm + cache. one engine per repo."""

    def __init__(self, settings: Settings | None = None, name: str = "default"):
        self.settings = settings or Settings()
        self.name = name
        self.lock = threading.RLock()
        data_dir = os.path.join(self.settings.data_dir, name)
        self.embedder = get_embedder(self.settings.embedder)
        self.index = Index(data_dir, self.embedder, store_kind=self.settings.vector_store,
                           qdrant_url=self.settings.qdrant_url, collection=f"cie_{name}")
        self.retriever = HybridRetriever(self.index, get_reranker(self.settings.reranker))
        self.llm = get_llm(self.settings.llm)
        self.cache = get_cache(self.settings)

    # ---------------------------------------------------------------- indexing

    def _resolve_source(self, source: str) -> str:
        if re.match(r"^(https?://|git@)", source):
            slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", source.rstrip("/").removesuffix(".git").split("://")[-1])
            return clone(source, os.path.join(self.settings.data_dir, "repos", slug))
        return source

    def index_repo(self, source: str) -> dict:
        path = self._resolve_source(source)
        with self.lock:
            return self.index.sync(path)

    def update(self) -> dict:
        if not self.index.repo_root:
            raise RuntimeError("nothing indexed yet, run index first")
        with self.lock:
            return self.index.sync(self.index.repo_root)

    def pull_and_update(self) -> dict:
        """what the github webhook calls: pull the latest commit, then re-sync"""
        root = self.index.repo_root
        if not root:
            raise RuntimeError("nothing indexed yet")
        if os.path.isdir(os.path.join(root, ".git")):
            subprocess.run(["git", "-C", root, "pull", "--ff-only"], capture_output=True, timeout=300)
        return self.update()

    # --------------------------------------------------------------------- ask

    def ask(self, question: str, k: int | None = None, use_graph: bool = True, use_rerank: bool = True,
            use_cache: bool = True) -> Answer:
        k = k or self.settings.top_k
        t0 = time.perf_counter()
        key = make_key(question.strip().lower(), k, use_graph, use_rerank, self.index.version, self.llm.name)
        if use_cache:
            hit = self.cache.get(key)
            if hit is not None:
                ans = Answer(**hit)
                ans.cached = True
                ans.timings_ms = {"total": round((time.perf_counter() - t0) * 1000, 2)}
                return ans

        with self.lock:
            plan = analyze_query(question, self.index.graph)
            t1 = time.perf_counter()
            hits = self.retriever.retrieve(question, k=k, use_graph=use_graph, use_rerank=use_rerank, plan=plan)
            t2 = time.perf_counter()
            notes = graph_notes(self.index, plan) if use_graph else ""

        # keep the prompt under budget
        used, kept = 0, []
        for h in hits:
            used += len(h.chunk.text)
            if used > MAX_CONTEXT_CHARS and kept:
                break
            kept.append(h)

        text = self.llm.answer(question, kept, notes)
        t3 = time.perf_counter()

        refs = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text) if 0 < int(n) <= len(kept)})
        cited = [kept[n - 1] for n in refs] or kept[:3]
        ans = Answer(
            question=question,
            answer=text,
            citations=[h.to_dict() for h in cited],
            context=[h.to_dict() for h in kept],
            graph_notes=notes,
            timings_ms={
                "retrieve": round((t2 - t1) * 1000, 2),
                "llm": round((t3 - t2) * 1000, 2),
                "total": round((t3 - t0) * 1000, 2),
            },
        )
        if use_cache:
            self.cache.set(key, ans.to_dict())
        return ans
