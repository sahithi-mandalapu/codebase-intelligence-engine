import json

import numpy as np
import pytest

from cie.cache import RedisCache, TTLCache, make_key
from cie.vectorstore import NumpyStore


def _unit(*vals):
    v = np.array(vals, dtype=np.float32)
    return v / np.linalg.norm(v)


def _store_contract(store):
    store.upsert(["a", "b", "c"], np.stack([_unit(1, 0, 0), _unit(0, 1, 0), _unit(0, 0, 1)]))
    assert store.search(_unit(1, 0.1, 0), 2)[0][0] == "a"
    assert store.existing(["a", "z"]) == {"a"}
    store.upsert(["a"], np.stack([_unit(0, 0, 1)]))  # overwrite
    assert {i for i, _ in store.search(_unit(0, 0, 1), 2)} == {"a", "c"}
    store.delete(["a"])
    assert store.existing(["a", "b", "c"]) == {"b", "c"}
    assert len(store) == 2


def test_numpy_store():
    _store_contract(NumpyStore(3))


def test_numpy_store_save_load(tmp_path):
    s = NumpyStore(3)
    s.upsert(["a", "b"], np.stack([_unit(1, 0, 0), _unit(0, 1, 0)]))
    s.save(str(tmp_path))
    t = NumpyStore(3)
    t.load(str(tmp_path))
    assert t.ids == ["a", "b"] and t.search(_unit(0, 1, 0), 1)[0][0] == "b"


def test_numpy_store_ignores_wrong_dim_on_load(tmp_path):
    s = NumpyStore(3)
    s.upsert(["a"], np.stack([_unit(1, 0, 0)]))
    s.save(str(tmp_path))
    t = NumpyStore(5)
    t.load(str(tmp_path))
    assert len(t) == 0


def test_qdrant_store_in_memory():
    pytest.importorskip("qdrant_client")
    from cie.vectorstore import QdrantStore

    _store_contract(QdrantStore(3, collection="t"))


def test_ttl_cache_expiry_and_lru(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr("cie.cache.time.time", lambda: now[0])
    c = TTLCache(max_items=2, ttl=10)
    c.set("a", 1)
    c.set("b", 2)
    assert c.get("a") == 1
    c.set("c", 3)  # evicts b (least recently used)
    assert c.get("b") is None and c.get("c") == 3
    now[0] += 11
    assert c.get("a") is None
    assert c.stats()["hits"] == 2


def test_make_key_is_stable_and_order_sensitive():
    assert make_key("a", 1) == make_key("a", 1)
    assert make_key("a", 1) != make_key(1, "a")


def test_redis_cache_with_fake_redis(monkeypatch):
    fakeredis = pytest.importorskip("fakeredis")
    import redis

    server = fakeredis.FakeServer()
    monkeypatch.setattr(redis.Redis, "from_url", staticmethod(lambda url: fakeredis.FakeRedis(server=server)))
    c = RedisCache("redis://whatever", ttl=60)
    assert c.get("k") is None
    c.set("k", {"x": [1, 2]})
    assert c.get("k") == {"x": [1, 2]}
    c.clear()
    assert c.get("k") is None


def test_redis_cache_survives_redis_being_down(monkeypatch):
    pytest.importorskip("redis")
    import redis

    class Broken:
        def get(self, *a):
            raise ConnectionError

        def set(self, *a, **kw):
            raise ConnectionError

    monkeypatch.setattr(redis.Redis, "from_url", staticmethod(lambda url: Broken()))
    c = RedisCache("redis://nope")
    c.set("k", 1)
    assert c.get("k") is None
    json.dumps(c.stats())
