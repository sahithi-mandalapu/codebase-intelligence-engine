from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
import time
from collections import defaultdict

from ..embeddings import get_embedder
from ..indexer import Index
from ..ingest import clone
from ..reranker import get_reranker
from ..retriever import HybridRetriever
from .dataset import generate
from .metrics import mean, score_query

KS = (1, 5, 10)

# the ablation: each row adds one thing on top of the previous
CONFIGS = [
    ("dense only", dict(use_dense=True, use_sparse=False, use_graph=False, use_rerank=False)),
    ("bm25 only", dict(use_dense=False, use_sparse=True, use_graph=False, use_rerank=False)),
    ("hybrid (dense+bm25)", dict(use_dense=True, use_sparse=True, use_graph=False, use_rerank=False)),
    ("hybrid + graph", dict(use_dense=True, use_sparse=True, use_graph=True, use_rerank=False)),
    ("hybrid + graph + rerank", dict(use_dense=True, use_sparse=True, use_graph=True, use_rerank=True)),
]


def run_config(retriever, dataset, config) -> dict:
    rows, lat = [], []
    for item in dataset:
        t = time.perf_counter()
        hits = retriever.retrieve(item["question"], k=max(KS), **config)
        lat.append((time.perf_counter() - t) * 1000)
        row = score_query([h.chunk for h in hits], item["relevant"], KS)
        row["type"] = item["type"]
        rows.append(row)
    keys = [f"recall@{k}" for k in KS] + ["mrr"]
    summary = {k: round(mean(rows, k), 3) for k in keys}
    summary["latency_ms"] = round(sum(lat) / len(lat), 1) if lat else 0.0
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)
    summary["by_type"] = {
        t: {k: round(mean(rs, k), 3) for k in keys} | {"n": len(rs)} for t, rs in sorted(by_type.items())
    }
    return summary


def evaluate_repo(repo: str, name: str, n: int, embedder: str, reranker: str, workdir: str, dataset_path=None) -> dict:
    index = Index(os.path.join(workdir, name), get_embedder(embedder))
    sync = index.sync(repo)
    if dataset_path:
        with open(dataset_path) as f:
            dataset = json.load(f)
    else:
        dataset = generate(index, n=n)
    retriever = HybridRetriever(index, get_reranker(reranker))
    results = {label: run_config(retriever, dataset, cfg) for label, cfg in CONFIGS}
    return {
        "repo": name,
        "embedder": embedder,
        "reranker": reranker,
        "index": {"files": sync["files"], "chunks": sync["chunks"], "symbols": len(index.graph.symbols),
                  "index_seconds": sync["seconds"]},
        "n_questions": len(dataset),
        "results": results,
        "dataset": dataset,
    }


def to_markdown(rep: dict) -> str:
    lines = [
        f"### {rep['repo']}",
        "",
        f"{rep['index']['files']} files, {rep['index']['chunks']} chunks, {rep['index']['symbols']} symbols, "
        f"{rep['n_questions']} questions. embedder `{rep['embedder']}`, reranker `{rep['reranker']}`.",
        "",
        "| config | Recall@1 | Recall@5 | Recall@10 | MRR | ms/query |",
        "|---|---|---|---|---|---|",
    ]
    for label, r in rep["results"].items():
        lines.append(f"| {label} | {r['recall@1']:.3f} | {r['recall@5']:.3f} | {r['recall@10']:.3f} | "
                     f"{r['mrr']:.3f} | {r['latency_ms']} |")
    full = rep["results"][CONFIGS[-1][0]]
    base = rep["results"]["hybrid (dense+bm25)"]
    lines += ["", "per question type (hybrid vs full pipeline, Recall@5):", "",
              "| type | n | hybrid | full |", "|---|---|---|---|"]
    for t, r in full["by_type"].items():
        lines.append(f"| {t} | {r['n']} | {base['by_type'][t]['recall@5']:.3f} | {r['recall@5']:.3f} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description="retrieval eval + ablation")
    ap.add_argument("--repo", action="append", required=True, help="local path or git url, repeatable")
    ap.add_argument("--n", type=int, default=120, help="questions per repo")
    ap.add_argument("--embedder", default="hash")
    ap.add_argument("--reranker", default="lexical")
    ap.add_argument("--dataset", default=None, help="use a labeled json file instead of generating (one repo only)")
    ap.add_argument("--out", default="eval/results")
    ap.add_argument("--save-datasets", default="eval/datasets")
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    work = tempfile.mkdtemp(prefix="cie_eval_")
    md_all = []
    try:
        for src in args.repo:
            if re.match(r"^(https?://|git@)", src):
                name = src.rstrip("/").removesuffix(".git").split("/")[-1]
                path = clone(src, os.path.join(work, "clones", name))
            else:
                path = src
                name = os.path.basename(os.path.abspath(src))
            rep = evaluate_repo(path, name, args.n, args.embedder, args.reranker, work, args.dataset)
            os.makedirs(args.save_datasets, exist_ok=True)
            if not args.dataset:
                with open(os.path.join(args.save_datasets, f"{name}.json"), "w") as f:
                    json.dump(rep["dataset"], f, indent=1)
            slim = {k: v for k, v in rep.items() if k != "dataset"}
            with open(os.path.join(args.out, f"{name}.json"), "w") as f:
                json.dump(slim, f, indent=2)
            md = to_markdown(rep)
            md_all.append(md)
            print(md)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write("\n".join(md_all))


if __name__ == "__main__":
    main()
