from __future__ import annotations

import math
import os
import zlib
from collections import Counter

import numpy as np

from .bm25 import tokenize


def _norm(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=-1, keepdims=True)
    n[n == 0] = 1.0
    return (m / n).astype(np.float32)


class HashEmbedder:
    """feature hashing over identifier-aware tokens + bigrams.

    NOT a neural model. it's here so the whole pipeline (and the tests / CI) run offline with
    zero downloads. for real semantic search use CIE_EMBEDDER=st (bge) or openai.
    """

    def __init__(self, dim: int = 512):
        self.dim = dim
        self.name = f"hash{dim}"

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        toks = tokenize(text)
        feats = Counter(toks)
        bigrams = Counter(f"{a}|{b}" for a, b in zip(toks, toks[1:]))
        for table, w in ((feats, 1.0), (bigrams, 0.5)):
            for f, c in table.items():
                h = zlib.crc32(f.encode())
                sign = 1.0 if (h >> 16) & 1 else -1.0
                v[h % self.dim] += sign * w * (1.0 + math.log(c))
        return v

    def embed_docs(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _norm(np.stack([self._vec(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_docs([text])[0]


class SentenceTransformerEmbedder:
    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # bge wants this on queries

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5"):
        from sentence_transformers import SentenceTransformer  # lazy, it's a heavy import

        self.model_name = model
        self.model = SentenceTransformer(model)
        self.dim = int(self.model.get_sentence_embedding_dimension())
        self.name = f"st:{model}"

    def embed_docs(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = self.model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
        return np.asarray(out, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        prefix = self.QUERY_PREFIX if "bge" in self.model_name.lower() else ""
        return self.embed_docs([prefix + text])[0]


class OpenAIEmbedder:
    def __init__(self, model: str = "text-embedding-3-small"):
        from openai import OpenAI

        self.client = OpenAI()
        self.model = model
        self.dim = 1536 if "small" in model else 3072
        self.name = f"openai:{model}"

    def embed_docs(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        rows = []
        for i in range(0, len(texts), 96):
            batch = [t[:8000] for t in texts[i:i + 96]]
            resp = self.client.embeddings.create(model=self.model, input=batch)
            rows.extend(d.embedding for d in resp.data)
        return _norm(np.array(rows, dtype=np.float32))

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_docs([text])[0]


def get_embedder(name: str = "hash"):
    name = (name or "hash").strip()
    if name == "hash":
        return HashEmbedder()
    if name.startswith(("st", "bge")):
        model = name.split(":", 1)[1] if ":" in name else os.getenv("CIE_ST_MODEL", "BAAI/bge-small-en-v1.5")
        return SentenceTransformerEmbedder(model)
    if name.startswith("openai"):
        model = name.split(":", 1)[1] if ":" in name else "text-embedding-3-small"
        return OpenAIEmbedder(model)
    raise ValueError(f"unknown embedder '{name}' (use hash, st, st:<model> or openai)")
