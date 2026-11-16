from __future__ import annotations

import json
import os
import time
from collections import defaultdict

from .bm25 import BM25
from .graph import CodeGraph
from .ingest import parse_file, walk
from .models import Chunk, FileInfo, file_hash, short_hash
from .vectorstore import make_store


class Index:
    """holds everything for one repo: parsed files, chunks, graph, bm25 and vectors.

    sync() is the only thing that changes it. first run indexes everything, later runs only
    re-parse files whose sha changed and only re-embed chunks we haven't seen before.
    """

    def __init__(self, data_dir: str, embedder, store_kind: str = "numpy", qdrant_url: str = "",
                 collection: str = "cie"):
        self.data_dir = data_dir
        self.embedder = embedder
        self.repo_root: str | None = None
        self.files: dict[str, FileInfo] = {}
        self.chunks: dict[str, Chunk] = {}
        self.graph = CodeGraph()
        self.bm25 = BM25()
        self.store = make_store(store_kind, embedder.dim, qdrant_url, collection)
        self._by_symbol: dict[str, list[str]] = defaultdict(list)
        self._load()

    # ------------------------------------------------------------- persistence

    @property
    def state_path(self) -> str:
        return os.path.join(self.data_dir, "state.json")

    def _load(self):
        if not os.path.exists(self.state_path):
            return
        try:
            with open(self.state_path) as f:
                state = json.load(f)
        except (OSError, json.JSONDecodeError):
            return
        if state.get("embedder") != self.embedder.name:
            # vectors from a different model are useless, start clean
            return
        self.repo_root = state.get("repo")
        self.files = {p: FileInfo.from_dict(d) for p, d in state["files"].items()}
        self.store.load(self.data_dir)
        self._rebuild_derived()

    def save(self):
        os.makedirs(self.data_dir, exist_ok=True)
        state = {
            "version": self.version,
            "repo": self.repo_root,
            "embedder": self.embedder.name,
            "files": {p: f.to_dict() for p, f in self.files.items()},
        }
        tmp = self.state_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f)
        os.replace(tmp, self.state_path)
        self.store.save(self.data_dir)

    # ----------------------------------------------------------------- helpers

    @property
    def version(self) -> str:
        return short_hash(*[f"{p}:{self.files[p].sha}" for p in sorted(self.files)])

    def _rebuild_derived(self):
        self.chunks = {}
        for f in self.files.values():
            for c in f.chunks:
                self.chunks[c.id] = c
        self.bm25 = BM25()
        for cid, c in self.chunks.items():
            self.bm25.add(cid, c.search_text())
        self._refresh_graph()

    def _refresh_graph(self):
        self.graph = CodeGraph().build(self.files)
        self._by_symbol = defaultdict(list)
        for c in self.chunks.values():
            if c.qualname:
                self._by_symbol[f"{c.path}::{c.qualname}"].append(c.id)

    def symbol_chunk(self, sid: str):
        ids = self._by_symbol.get(sid)
        if not ids:
            return None
        return sorted(ids, key=lambda i: self.chunks[i].start)[0]

    def file_chunk(self, path: str):
        f = self.files.get(path)
        if not f or not f.chunks:
            return None
        for c in f.chunks:
            if c.kind == "module":
                return c.id
        return f.chunks[0].id

    # ------------------------------------------------------------------- sync

    def sync(self, repo: str) -> dict:
        t0 = time.time()
        repo = os.path.abspath(repo)
        if not os.path.isdir(repo):
            raise FileNotFoundError(f"not a directory: {repo}")
        self.repo_root = repo

        seen, parsed = set(), []
        added = changed = 0
        for rel, lang, text in walk(repo):
            seen.add(rel)
            old = self.files.get(rel)
            if old is not None and old.sha == file_hash(text):
                continue
            parsed.append(parse_file(rel, lang, text))
            if old is None:
                added += 1
            else:
                changed += 1
        removed = [p for p in self.files if p not in seen]

        # chunk ids that belong to files we're about to replace / drop
        drop = set()
        for p in removed + [i.path for i in parsed if i.path in self.files]:
            drop |= {c.id for c in self.files[p].chunks}
        new_ids = {c.id for i in parsed for c in i.chunks}
        drop -= new_ids  # same id = same text, keep the vector

        for p in removed:
            del self.files[p]
        for info in parsed:
            self.files[info.path] = info

        for cid in drop:
            self.chunks.pop(cid, None)
            self.bm25.remove(cid)
        self.store.delete(list(drop))

        for info in parsed:
            for c in info.chunks:
                is_new = c.id not in self.bm25
                self.chunks[c.id] = c  # always overwrite so line numbers stay fresh
                if is_new:
                    self.bm25.add(c.id, c.search_text())

        # embed whatever the store doesn't have yet (also repairs a half-written store)
        have = self.store.existing(list(self.chunks))
        todo = [cid for cid in self.chunks if cid not in have]
        for i in range(0, len(todo), 64):
            batch = todo[i:i + 64]
            vecs = self.embedder.embed_docs([self.chunks[c].search_text() for c in batch])
            self.store.upsert(batch, vecs)

        self._refresh_graph()
        self.save()
        return {
            "added": added,
            "changed": changed,
            "removed": len(removed),
            "files": len(self.files),
            "chunks": len(self.chunks),
            "embedded": len(todo),
            "reused": len(self.chunks) - len(todo),
            "seconds": round(time.time() - t0, 2),
        }

    # ----------------------------------------------------------------- search

    def search_dense(self, query: str, k: int = 30):
        res = self.store.search(self.embedder.embed_query(query), k)
        return [(cid, s) for cid, s in res if cid in self.chunks]

    def search_sparse(self, query: str, k: int = 30):
        return [(cid, s) for cid, s in self.bm25.search(query, k) if cid in self.chunks]

    def stats(self) -> dict:
        langs = defaultdict(int)
        for f in self.files.values():
            langs[f.language] += 1
        return {
            "repo": self.repo_root,
            "version": self.version,
            "files": len(self.files),
            "chunks": len(self.chunks),
            "languages": dict(langs),
            "graph": self.graph.stats(),
            "embedder": self.embedder.name,
        }
