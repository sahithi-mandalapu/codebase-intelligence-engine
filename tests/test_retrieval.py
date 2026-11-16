from cie.evaluation.dataset import generate
from cie.evaluation.metrics import is_relevant, score_query
from cie.query import analyze_query, detect_relations, extract_idents
from cie.reranker import LexicalReranker
from cie.retriever import HybridRetriever, fuse


def test_detect_relations():
    assert detect_relations("Which functions call `bark`?") == [("calls", "in")]
    assert detect_relations("What does `make_dog` call?") == [("calls", "out")]
    assert detect_relations("Which files import pkg/models.py?") == [("imports", "in")]
    assert detect_relations("What does service.py import?") == [("imports", "out")]
    assert detect_relations("Which classes extend `Animal`?") == [("inherits", "in")]
    assert detect_relations("what are the subclasses of Animal") == [("inherits", "in")]
    assert detect_relations("how does caching work") == []


def test_extract_idents():
    got = extract_idents("Why does `make_dog` use Dog.speak and parse_file in cie/graph.py?")
    assert "make_dog" in got and "Dog.speak" in got and "parse_file" in got and "cie/graph.py" in got


def test_analyze_query_resolves_symbols_and_files(index):
    plan = analyze_query("Which functions call `bark`?", index.graph)
    assert plan.symbols == ["pkg/models.py::bark"] and plan.structural
    plan = analyze_query("What does service.py import?", index.graph)
    assert plan.files == ["pkg/service.py"] and plan.relations == [("imports", "out")]


def test_plain_word_matches_rare_symbol_names(index):
    plan = analyze_query("what does greet do", index.graph)
    assert plan.symbols == ["pkg/service.py::greet"]


def test_fuse_is_weighted_rrf():
    order, scores, srcs = fuse([(1.0, ["a", "b"], "x"), (2.0, ["b", "c"], "y")])
    assert order[0] == "b" and srcs["b"] == ["x", "y"]


def _top_names(hits, n=5):
    return [h.chunk.qualname for h in hits[:n]]


def test_graph_finds_callers_that_text_search_misses(index):
    r = HybridRetriever(index, LexicalReranker())
    q = "Which functions call `bark`?"
    with_graph = _top_names(r.retrieve(q, k=5, use_graph=True))
    assert "greet" in with_graph and "Dog.speak" in with_graph
    h = r.retrieve(q, k=5, use_graph=True)
    assert any("graph:rel" in x.sources for x in h)


def test_imports_question_returns_importing_file(index):
    r = HybridRetriever(index, LexicalReranker())
    hits = r.retrieve("Which files import pkg/models.py?", k=3)
    assert hits[0].chunk.path == "pkg/service.py"


def test_each_stage_can_be_switched_off(index):
    r = HybridRetriever(index, LexicalReranker())
    q = "register an animal"
    assert r.retrieve(q, k=3, use_sparse=False, use_graph=False, use_rerank=False)
    assert r.retrieve(q, k=3, use_dense=False, use_graph=False, use_rerank=False)
    hits = r.retrieve(q, k=3, use_graph=False, use_rerank=False)
    assert all("graph" not in s for h in hits for s in h.sources)


def test_semantic_style_question(index):
    r = HybridRetriever(index, LexicalReranker())
    hits = r.retrieve("create a dog and register it", k=3)
    assert hits[0].chunk.qualname == "make_dog"


def test_is_relevant_and_score_query(index):
    chunks = [index.chunks[index.symbol_chunk(s)] for s in
              ("pkg/utils.py::register", "pkg/models.py::bark", "pkg/models.py::Dog")]
    assert is_relevant(chunks[1], {"path": "pkg/models.py", "qualname": "bark"})
    assert is_relevant(chunks[2], {"path": "pkg/models.py"})
    assert not is_relevant(chunks[0], {"path": "pkg/models.py"})
    s = score_query(chunks, [{"path": "pkg/models.py", "qualname": "bark"}, {"path": "nope.py"}], ks=(1, 2))
    assert s["mrr"] == 0.5 and s["recall@1"] == 0.0 and s["recall@2"] == 0.5


def test_class_relevance_includes_its_methods(index):
    method_chunk = index.chunks[index.symbol_chunk("pkg/models.py::Dog.speak")]
    assert is_relevant(method_chunk, {"path": "pkg/models.py", "qualname": "Dog"})
    assert not is_relevant(method_chunk, {"path": "pkg/models.py", "qualname": "Do"})


def test_dataset_generation_is_deterministic_and_well_formed(index):
    a = generate(index, n=20, seed=1)
    b = generate(index, n=20, seed=1)
    assert a == b and a
    for item in a:
        assert {"id", "type", "question", "relevant"} <= set(item)
        assert item["relevant"] and all("path" in r for r in item["relevant"])
    types = {i["type"] for i in a}
    assert {"definition", "callers"} <= types
