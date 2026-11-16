from __future__ import annotations

import json
import os
import uuid

import numpy as np


class NumpyStore:
    """brute force cosine search. vectors are already normalized so it's just a dot product.
    fine up to ~100k chunks, which is way more than most repos."""

    def __init__(self, dim: int):
        self.dim = dim
        self.ids: list[str] = []
        self.mat = np.zeros((0, dim), dtype=np.float32)
        self.pos: dict[str, int] = {}

    def __len__(self):
        return len(self.ids)

    def existing(self, ids) -> set:
        return {i for i in ids if i in self.pos}

    def upsert(self, ids: list[str], vecs: np.ndarray):
        new_ids, new_rows = [], []
        for i, v in zip(ids, vecs):
            if i in self.pos:
                self.mat[self.pos[i]] = v
            else:
                new_ids.append(i)
                new_rows.append(v)
        if new_ids:
            start = len(self.ids)
            self.ids.extend(new_ids)
            self.mat = np.vstack([self.mat, np.asarray(new_rows, dtype=np.float32)])
            for off, i in enumerate(new_ids):
                self.pos[i] = start + off

    def delete(self, ids):
        drop = {i for i in ids if i in self.pos}
        if not drop:
            return
        keep = [n for n, i in enumerate(self.ids) if i not in drop]
        self.mat = self.mat[keep]
        self.ids = [self.ids[n] for n in keep]
        self.pos = {i: n for n, i in enumerate(self.ids)}

    def search(self, vec: np.ndarray, k: int = 10) -> list[tuple[str, float]]:
        if not self.ids:
            return []
        sims = self.mat @ vec
        k = min(k, len(self.ids))
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [(self.ids[i], float(sims[i])) for i in top]

    def save(self, folder: str):
        os.makedirs(folder, exist_ok=True)
        np.save(os.path.join(folder, "vectors.npy"), self.mat)
        with open(os.path.join(folder, "vector_ids.json"), "w") as f:
            json.dump(self.ids, f)

    def load(self, folder: str):
        vp, ip = os.path.join(folder, "vectors.npy"), os.path.join(folder, "vector_ids.json")
        if not (os.path.exists(vp) and os.path.exists(ip)):
            return
        mat = np.load(vp)
        with open(ip) as f:
            ids = json.load(f)
        if mat.ndim == 2 and mat.shape[1] == self.dim and len(ids) == mat.shape[0]:
            self.mat, self.ids = mat.astype(np.float32), ids
            self.pos = {i: n for n, i in enumerate(ids)}


class QdrantStore:
    """same interface, backed by qdrant. pass a url for a real server, no url = in-memory."""

    def __init__(self, dim: int, url: str = "", collection: str = "cie"):
        from qdrant_client import QdrantClient, models

        self._m = models
        self.dim = dim
        self.collection = collection
        self.client = QdrantClient(url=url) if url else QdrantClient(":memory:")
        existing = {c.name for c in self.client.get_collections().collections}
        if collection not in existing:
            self.client.create_collection(
                collection_name=collection,
                vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
            )

    @staticmethod
    def _pid(cid: str) -> str:
        # qdrant ids have to be ints or uuids
        return str(uuid.uuid5(uuid.NAMESPACE_URL, cid))

    def __len__(self):
        return self.client.count(self.collection, exact=True).count

    def existing(self, ids) -> set:
        ids = list(ids)
        found = set()
        for i in range(0, len(ids), 256):
            batch = ids[i:i + 256]
            recs = self.client.retrieve(self.collection, ids=[self._pid(c) for c in batch],
                                        with_payload=True, with_vectors=False)
            found.update(r.payload["cid"] for r in recs)
        return found

    def upsert(self, ids, vecs):
        pts = [
            self._m.PointStruct(id=self._pid(c), vector=np.asarray(v, dtype=float).tolist(), payload={"cid": c})
            for c, v in zip(ids, vecs)
        ]
        for i in range(0, len(pts), 128):
            self.client.upsert(self.collection, points=pts[i:i + 128])

    def delete(self, ids):
        ids = list(ids)
        if not ids:
            return
        self.client.delete(self.collection,
                           points_selector=self._m.PointIdsList(points=[self._pid(c) for c in ids]))

    def search(self, vec, k=10):
        res = self.client.query_points(self.collection, query=np.asarray(vec, dtype=float).tolist(),
                                       limit=k, with_payload=True)
        return [(p.payload["cid"], float(p.score)) for p in res.points]

    def save(self, folder: str):  # server keeps its own data
        pass

    def load(self, folder: str):
        pass


def make_store(kind: str, dim: int, url: str = "", collection: str = "cie"):
    if kind == "qdrant":
        return QdrantStore(dim, url=url, collection=collection)
    return NumpyStore(dim)
