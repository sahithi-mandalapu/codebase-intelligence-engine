from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict


def make_key(*parts) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode())
        h.update(b"\x00")
    return h.hexdigest()


class TTLCache:
    def __init__(self, max_items: int = 512, ttl: int = 3600):
        self.max_items, self.ttl = max_items, ttl
        self._d: OrderedDict = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key):
        item = self._d.get(key)
        if item is None:
            self.misses += 1
            return None
        ts, val = item
        if time.time() - ts > self.ttl:
            del self._d[key]
            self.misses += 1
            return None
        self._d.move_to_end(key)
        self.hits += 1
        return val

    def set(self, key, val):
        self._d[key] = (time.time(), val)
        self._d.move_to_end(key)
        while len(self._d) > self.max_items:
            self._d.popitem(last=False)

    def clear(self):
        self._d.clear()

    def stats(self):
        return {"backend": "memory", "items": len(self._d), "hits": self.hits, "misses": self.misses}


class RedisCache:
    """same interface as TTLCache but shared between workers. if redis is down we just act like a miss."""

    def __init__(self, url: str, ttl: int = 3600, prefix: str = "cie:"):
        import redis

        self.r = redis.Redis.from_url(url)
        self.ttl, self.prefix = ttl, prefix
        self.hits = 0
        self.misses = 0

    def get(self, key):
        try:
            raw = self.r.get(self.prefix + key)
        except Exception:
            raw = None
        if raw is None:
            self.misses += 1
            return None
        self.hits += 1
        return json.loads(raw)

    def set(self, key, val):
        try:
            self.r.set(self.prefix + key, json.dumps(val), ex=self.ttl)
        except Exception:
            pass

    def clear(self):
        try:
            for k in self.r.scan_iter(self.prefix + "*"):
                self.r.delete(k)
        except Exception:
            pass

    def stats(self):
        return {"backend": "redis", "hits": self.hits, "misses": self.misses}


def get_cache(settings):
    if settings.redis_url:
        return RedisCache(settings.redis_url, ttl=settings.cache_ttl)
    return TTLCache(ttl=settings.cache_ttl)
