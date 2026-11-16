### requests

66 files, 896 chunks, 696 symbols, 120 questions. embedder `hash`, reranker `lexical`.

| config | Recall@1 | Recall@5 | Recall@10 | MRR | ms/query |
|---|---|---|---|---|---|
| dense only | 0.328 | 0.586 | 0.673 | 0.497 | 0.5 |
| bm25 only | 0.386 | 0.676 | 0.780 | 0.575 | 0.7 |
| hybrid (dense+bm25) | 0.331 | 0.611 | 0.714 | 0.496 | 0.8 |
| hybrid + graph | 0.442 | 0.916 | 0.991 | 0.725 | 0.9 |
| hybrid + graph + rerank | 0.683 | 0.978 | 0.997 | 0.948 | 7.0 |

per question type (hybrid vs full pipeline, Recall@5):

| type | n | hybrid | full |
|---|---|---|---|
| callees | 18 | 0.104 | 0.991 |
| callers | 24 | 0.569 | 0.958 |
| definition | 18 | 0.889 | 1.000 |
| docstring | 36 | 0.944 | 1.000 |
| imports | 12 | 0.156 | 0.935 |
| inherits | 12 | 0.489 | 0.944 |

### click

134 files, 1932 chunks, 1389 symbols, 120 questions. embedder `hash`, reranker `lexical`.

| config | Recall@1 | Recall@5 | Recall@10 | MRR | ms/query |
|---|---|---|---|---|---|
| dense only | 0.328 | 0.546 | 0.651 | 0.463 | 0.5 |
| bm25 only | 0.358 | 0.707 | 0.737 | 0.526 | 0.8 |
| hybrid (dense+bm25) | 0.363 | 0.628 | 0.709 | 0.506 | 1.3 |
| hybrid + graph | 0.437 | 0.946 | 0.985 | 0.746 | 1.5 |
| hybrid + graph + rerank | 0.692 | 0.966 | 0.995 | 0.947 | 6.2 |

per question type (hybrid vs full pipeline, Recall@5):

| type | n | hybrid | full |
|---|---|---|---|
| callees | 18 | 0.028 | 0.991 |
| callers | 24 | 0.750 | 1.000 |
| definition | 18 | 0.889 | 0.944 |
| docstring | 37 | 0.973 | 0.973 |
| imports | 11 | 0.076 | 0.902 |
| inherits | 12 | 0.334 | 0.928 |

### flask

168 files, 1322 chunks, 902 symbols, 120 questions. embedder `hash`, reranker `lexical`.

| config | Recall@1 | Recall@5 | Recall@10 | MRR | ms/query |
|---|---|---|---|---|---|
| dense only | 0.400 | 0.587 | 0.663 | 0.510 | 0.4 |
| bm25 only | 0.412 | 0.649 | 0.720 | 0.546 | 0.7 |
| hybrid (dense+bm25) | 0.429 | 0.643 | 0.720 | 0.548 | 1.0 |
| hybrid + graph | 0.514 | 0.928 | 1.000 | 0.774 | 1.2 |
| hybrid + graph + rerank | 0.727 | 0.977 | 0.992 | 0.955 | 9.4 |

per question type (hybrid vs full pipeline, Recall@5):

| type | n | hybrid | full |
|---|---|---|---|
| callees | 18 | 0.000 | 1.000 |
| callers | 24 | 0.667 | 1.000 |
| definition | 18 | 0.944 | 1.000 |
| docstring | 36 | 1.000 | 0.944 |
| imports | 12 | 0.231 | 0.965 |
| inherits | 12 | 0.452 | 0.969 |
