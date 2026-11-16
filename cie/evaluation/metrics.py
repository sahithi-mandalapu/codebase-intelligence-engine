from __future__ import annotations


def is_relevant(chunk, rel: dict) -> bool:
    """rel = {"path": ..., "qualname": ... (optional)}. no qualname means any chunk in that file counts.
    a class counts its own methods too."""
    if chunk.path != rel["path"]:
        return False
    q = rel.get("qualname")
    if not q:
        return True
    return chunk.qualname == q or chunk.qualname.startswith(q + ".")


def score_query(chunks, relevant: list[dict], ks=(1, 5, 10)) -> dict:
    """chunks = ranked list of retrieved chunks. returns recall@k for each k plus reciprocal rank"""
    first_rank = []
    for rel in relevant:
        r = next((i + 1 for i, c in enumerate(chunks) if is_relevant(c, rel)), None)
        first_rank.append(r)
    best = min((r for r in first_rank if r), default=None)
    out = {"mrr": (1.0 / best) if best else 0.0}
    for k in ks:
        out[f"recall@{k}"] = sum(1 for r in first_rank if r and r <= k) / max(1, len(relevant))
    return out


def mean(rows: list[dict], key: str) -> float:
    return sum(r[key] for r in rows) / len(rows) if rows else 0.0
