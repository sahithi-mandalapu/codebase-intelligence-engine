import numpy as np

from cie.bm25 import BM25, tokenize
from cie.embeddings import HashEmbedder


def test_tokenize_splits_identifiers_but_keeps_full_name():
    toks = tokenize("parse_file HttpRequest getUserID")
    assert {"parse_file", "parse", "file", "httprequest", "http", "request", "getuserid", "get", "user", "id"} <= set(toks)


def test_tokenize_drops_stopwords():
    assert "the" not in tokenize("the quick fox") and "quick" in tokenize("the quick fox")


def test_bm25_ranks_and_removes():
    b = BM25()
    b.add("a", "def parse_config(path): read the yaml config file")
    b.add("b", "def render_page(html): build html output")
    b.add("c", "unrelated things about networking sockets")
    assert b.search("yaml config", 3)[0][0] == "a"
    assert b.search("html render", 3)[0][0] == "b"
    b.remove("a")
    assert all(d != "a" for d, _ in b.search("yaml config", 3))
    assert len(b) == 2


def test_bm25_readd_replaces():
    b = BM25()
    b.add("a", "alpha beta")
    b.add("a", "gamma delta")
    assert b.search("alpha", 3) == [] and b.search("gamma", 3)[0][0] == "a"


def test_hash_embedder_is_normalized_and_deterministic():
    e = HashEmbedder()
    m = e.embed_docs(["def foo(): pass", "def foo(): pass", "completely different words here"])
    assert np.allclose(np.linalg.norm(m, axis=1), 1.0, atol=1e-5)
    assert np.allclose(m[0], m[1])
    assert float(m[0] @ m[1]) > float(m[0] @ m[2])


def test_hash_embedder_empty():
    assert HashEmbedder().embed_docs([]).shape == (0, 512)
