from cie.embeddings import HashEmbedder
from cie.indexer import Index


def test_first_sync_indexes_everything(sample_repo, tmp_path):
    idx = Index(str(tmp_path / "d"), HashEmbedder())
    s = idx.sync(str(sample_repo))
    assert s["added"] == 7 and s["changed"] == 0 and s["removed"] == 0
    assert s["embedded"] == s["chunks"] > 10 and s["reused"] == 0


def test_second_sync_is_a_noop(index, sample_repo):
    s = index.sync(str(sample_repo))
    assert (s["added"], s["changed"], s["removed"], s["embedded"]) == (0, 0, 0, 0)
    assert s["reused"] == s["chunks"]


def test_edit_only_reembeds_new_chunks(index, sample_repo):
    before = index.sync(str(sample_repo))["chunks"]
    svc = sample_repo / "pkg" / "service.py"
    svc.write_text(svc.read_text() + "\n\ndef shout(name):\n    return bark(name).upper()\n")
    s = index.sync(str(sample_repo))
    assert s["changed"] == 1 and s["added"] == 0
    assert 1 <= s["embedded"] <= 3  # the new function (+ maybe the module chunk), not the whole repo
    assert s["reused"] >= before - 2
    assert "pkg/service.py::shout" in index.graph.callees("pkg/service.py::shout") or \
        "pkg/models.py::bark" in index.graph.callees("pkg/service.py::shout")


def test_moving_code_reuses_embeddings(index, sample_repo):
    util = sample_repo / "pkg" / "utils.py"
    util.write_text("# a comment that shifts everything down\n\n" + util.read_text())
    s = index.sync(str(sample_repo))
    assert s["changed"] == 1
    assert s["embedded"] <= 1  # register() has identical text so its vector is reused
    reg = index.chunks[index.symbol_chunk("pkg/utils.py::register")]
    assert reg.start > 3  # but the line numbers did get updated


def test_deleted_file_is_dropped_everywhere(index, sample_repo):
    (sample_repo / "pkg" / "utils.py").unlink()
    s = index.sync(str(sample_repo))
    assert s["removed"] == 1
    assert not any(c.path == "pkg/utils.py" for c in index.chunks.values())
    assert "pkg/utils.py::register" not in index.graph.symbols
    assert all(index.chunks[c].path != "pkg/utils.py" for c, _ in index.search_dense("register animal", 20))
    assert all(index.chunks[c].path != "pkg/utils.py" for c, _ in index.search_sparse("register animal", 20))


def test_persistence_roundtrip(index, sample_repo, tmp_path):
    again = Index(index.data_dir, HashEmbedder())
    assert again.version == index.version
    assert set(again.chunks) == set(index.chunks)
    assert again.graph.callers("pkg/models.py::bark") == index.graph.callers("pkg/models.py::bark")
    s = again.sync(str(sample_repo))
    assert s["embedded"] == 0  # vectors were loaded from disk too
    assert again.search_dense("bark dog noise", 1)[0][0] in again.chunks


def test_different_embedder_starts_clean(index):
    class Other(HashEmbedder):
        def __init__(self):
            super().__init__(dim=64)

    fresh = Index(index.data_dir, Other())
    assert fresh.files == {}


def test_qdrant_backed_index(sample_repo, tmp_path):
    import pytest

    pytest.importorskip("qdrant_client")
    idx = Index(str(tmp_path / "q"), HashEmbedder(), store_kind="qdrant", collection="t_idx")
    s = idx.sync(str(sample_repo))
    assert s["embedded"] == s["chunks"]
    top = idx.search_dense("make a dog and register it", 3)
    assert top and all(c in idx.chunks for c, _ in top)
    s2 = idx.sync(str(sample_repo))
    assert s2["embedded"] == 0


def test_version_changes_when_content_changes(index, sample_repo):
    v = index.version
    (sample_repo / "pkg" / "utils.py").write_text("X = 1\n")
    index.sync(str(sample_repo))
    assert index.version != v
