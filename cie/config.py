from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass
class Settings:
    # everything can be overridden with env vars so docker-compose is easy
    data_dir: str = field(default_factory=lambda: _env("CIE_DATA_DIR", ".cie_data"))
    embedder: str = field(default_factory=lambda: _env("CIE_EMBEDDER", "hash"))  # hash | st | st:<model> | openai
    reranker: str = field(default_factory=lambda: _env("CIE_RERANKER", "lexical"))  # lexical | cross | none
    llm: str = field(default_factory=lambda: _env("CIE_LLM", "none"))  # none | anthropic | openai
    vector_store: str = field(default_factory=lambda: _env("CIE_VECTOR_STORE", "numpy"))  # numpy | qdrant
    qdrant_url: str = field(default_factory=lambda: _env("CIE_QDRANT_URL", ""))
    redis_url: str = field(default_factory=lambda: _env("CIE_REDIS_URL", ""))
    cache_ttl: int = field(default_factory=lambda: int(_env("CIE_CACHE_TTL", "3600")))
    top_k: int = 8
