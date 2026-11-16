# Codebase Intelligence Engine

Ask questions about a GitHub repo and get answers with file + line citations.

I wanted to see how much better a code Q&A tool gets if it understands code *structure* instead of just chopping files into text chunks. Plain RAG is bad at questions like "what calls `X`?" or "which files import `Y`?" because the answer isn't textually similar to the question. So this builds a code graph from the AST and mixes it into retrieval.

## How it works

```
repo -> walker -> parser (AST) -> chunks (one per function / method / class)
                              \-> code graph (calls, imports, inherits)
chunks -> embeddings -> vector store
chunks -> BM25 index

question -> query analysis (finds symbol names + "who calls / imports / extends" intent)
         -> dense search + BM25 + graph lookups
         -> reciprocal rank fusion
         -> rerank
         -> LLM answer with [1] [2] citations
```

- **Chunking** is by symbol, not by fixed size. Python uses the `ast` module. JS/TS uses regexes (best effort, see limitations). Other languages fall back to line windows.
- **Code graph** (networkx): file and symbol nodes, edges for `calls`, `imports`, `inherits`. Call resolution handles `self.method()`, imported names, `module.func()`, and methods inherited from base classes.
- **Hybrid retrieval**: dense vectors + BM25 (with a tokenizer that splits `snake_case` / `camelCase`), merged with weighted reciprocal rank fusion. Graph evidence gets added as extra ranked lists.
- **Reranking**: a lexical reranker by default (no downloads), or a cross-encoder (`bge-reranker`) if you install `sentence-transformers`.
- **Incremental indexing**: files are hashed. On an update only changed files are re-parsed and only chunks with new content get re-embedded. A GitHub push webhook triggers this.
- **Caching**: in-memory LRU/TTL cache, or Redis. Cache keys include the index version so edits invalidate stale answers.
- **Vector stores**: numpy (default) or Qdrant.
- **LLMs**: Anthropic or OpenAI. With no key it just returns the retrieved snippets, which is enough to try everything.

## Quick start

```bash
git clone <your-repo-url>
cd codebase-intelligence-engine
pip install -e ".[dev]"

# index any local folder or a git url
python -m cie index /path/to/some/repo
python -m cie ask "Which functions call \`parse_file\`?"
python -m cie symbol Index.sync
python -m cie deps cie/retriever.py
python -m cie trace HybridRetriever.retrieve BM25.search

# run the API
python -m cie serve
```

Or with Docker (API + Qdrant + Redis):

```bash
docker compose up --build
curl -X POST localhost:8000/index -H 'content-type: application/json' \
  -d '{"source": "https://github.com/pallets/click"}'
curl -X POST localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question": "Which classes extend `Parameter`?"}'
```

### Endpoints

| route | what it does |
|---|---|
| `POST /index` | index a local path or git url |
| `POST /index/update` | re-index only what changed |
| `POST /ask` | answer a question with citations |
| `GET /symbol?name=` | definition, callers, callees, bases, subclasses |
| `GET /deps?path=` | what a file imports and who imports it |
| `GET /trace?src=&dst=` | shortest dependency path between two symbols or files |
| `GET /stats` | index + cache stats |
| `POST /webhook/github` | push event -> pull + incremental update (HMAC verified if `GITHUB_WEBHOOK_SECRET` is set) |

### Config (env vars)

| var | default | options |
|---|---|---|
| `CIE_EMBEDDER` | `hash` | `hash`, `st`, `st:<model>`, `openai` |
| `CIE_RERANKER` | `lexical` | `lexical`, `cross`, `none` |
| `CIE_LLM` | `none` | `none`, `anthropic`, `openai` |
| `CIE_VECTOR_STORE` | `numpy` | `numpy`, `qdrant` |
| `CIE_QDRANT_URL` | | e.g. `http://localhost:6333` |
| `CIE_REDIS_URL` | | e.g. `redis://localhost:6379/0` |

**Important:** the default `hash` embedder is a feature-hashing baseline, not a neural model. It's there so everything runs offline and in CI. For real semantic search use `CIE_EMBEDDER=st` (BGE) or `openai`. All the numbers below were produced with `hash`, so they're a floor for the dense side, not a showcase of it. I haven't benchmarked the neural embedders.

## Evaluation

`cie/evaluation` generates a question set from a repo's own AST + graph (callers, callees, imports, subclasses, definitions, docstring lookups), then measures Recall@1/5/10 and MRR for an ablation where each row adds one component.

```bash
python -m cie.evaluation.runner --repo /path/to/repo --n 120
```

Results with the offline `hash` embedder + lexical reranker, 120 questions per repo (full output in `eval/results/`, the question sets are in `eval/datasets/`):

| repo | config | Recall@5 | MRR |
|---|---|---|---|
| requests | hybrid (dense + BM25) | 0.611 | 0.496 |
| requests | + graph | 0.916 | 0.725 |
| requests | + graph + rerank | 0.978 | 0.948 |
| click | hybrid (dense + BM25) | 0.628 | 0.506 |
| click | + graph | 0.946 | 0.746 |
| click | + graph + rerank | 0.966 | 0.947 |
| flask | hybrid (dense + BM25) | 0.643 | 0.548 |
| flask | + graph | 0.928 | 0.774 |
| flask | + graph + rerank | 0.977 | 0.955 |

### Read this before quoting the numbers

- **Partly circular.** For the callers / callees / imports / inherits questions, the "correct" answer comes from the same code graph the retriever uses. Those rows show that the graph feeds retrieval correctly, not that static analysis is right or that this beats a human-labeled benchmark. The docstring and definition questions don't depend on the graph. Hybrid retrieval already scores high on those, and the graph barely helps there (flask docstring Recall@5 even dips from 1.000 to 0.944).
- **Retrieval only.** This measures whether the right chunks come back, not whether the LLM's final answer is good. I haven't built answer-quality evaluation (RAGAS etc.) yet.
- **Questions are templated**, so they're cleaner than what a person would type.
- The lexical reranker was tuned while looking at this repo's own eval, then run unchanged on the other three. The relational-query weighting was a fix after I saw it hurting MRR on callee questions.

## Limitations

- Python is parsed properly. JS/TS is regex-based and will miss odd formatting. Everything else is just line windows. Tree-sitter would be the real fix.
- Call resolution is static and name-based. Dynamic dispatch, decorators that rewrap functions, and `getattr` tricks are invisible to it. Method calls on unknown receivers only link when the method name is unique in the repo.
- Brute-force numpy search is fine for repos up to tens of thousands of chunks. Use Qdrant past that.
- No auth on the API apart from the webhook signature.

## Layout

```
cie/
  ingest/       walker + AST/regex parsers
  graph.py      code graph and call/import/inheritance resolution
  bm25.py       code-aware BM25
  embeddings.py hash / sentence-transformers / OpenAI
  vectorstore.py numpy + Qdrant
  indexer.py    incremental indexing + persistence
  query.py      query analysis (symbols, relation intent)
  retriever.py  hybrid retrieval + rank fusion + graph evidence
  reranker.py   lexical + cross-encoder
  llm.py        Anthropic / OpenAI / offline fallback
  qa.py         ties it together, caching
  api.py        FastAPI app
  evaluation/   dataset generation, metrics, ablation runner
tests/          69 tests
eval/           generated question sets + results
```

## Tests

```bash
pytest
```

## Ideas for later

- tree-sitter parsing for more languages
- run the neural embedder + cross-encoder benchmark and compare
- hand-label a set of questions so the eval isn't circular
- answer-level evaluation (faithfulness / citation accuracy)
- `git diff` aware "what changed and what does it affect" questions using the graph
