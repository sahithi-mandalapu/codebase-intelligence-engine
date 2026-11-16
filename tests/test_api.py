import hashlib
import hmac
import json

import pytest
from fastapi.testclient import TestClient

from cie.api import create_app


@pytest.fixture
def client(settings):
    return TestClient(create_app(settings))


@pytest.fixture
def indexed_client(client, sample_repo):
    r = client.post("/index", json={"source": str(sample_repo)})
    assert r.status_code == 200
    return client


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_ask_before_index_is_a_409(client):
    assert client.post("/ask", json={"question": "hi"}).status_code == 409


def test_index_bad_path(client):
    assert client.post("/index", json={"source": "/definitely/not/here"}).status_code == 404


def test_index_then_ask(indexed_client):
    r = indexed_client.post("/ask", json={"question": "Which functions call `bark`?", "k": 5})
    assert r.status_code == 200
    body = r.json()
    names = {c["name"] for c in body["context"]}
    assert {"greet", "Dog.speak"} <= names
    assert "called by" in body["graph_notes"]
    assert body["cached"] is False and body["timings_ms"]["total"] >= 0


def test_second_identical_question_is_cached(indexed_client):
    q = {"question": "what does make_dog do?"}
    first = indexed_client.post("/ask", json=q).json()
    second = indexed_client.post("/ask", json=q).json()
    assert first["cached"] is False and second["cached"] is True
    assert second["answer"] == first["answer"]
    assert indexed_client.get("/stats").json()["cache"]["hits"] >= 1


def test_cache_is_invalidated_when_repo_changes(indexed_client, sample_repo):
    q = {"question": "what does make_dog do?"}
    indexed_client.post("/ask", json=q)
    (sample_repo / "pkg" / "utils.py").write_text("X = 1\n")
    assert indexed_client.post("/index/update").status_code == 200
    assert indexed_client.post("/ask", json=q).json()["cached"] is False


def test_symbol_deps_trace_stats(indexed_client):
    s = indexed_client.get("/symbol", params={"name": "bark"}).json()["matches"][0]
    assert s["callers"] == ["pkg/models.py::Dog.speak", "pkg/service.py::greet"]
    d = indexed_client.get("/deps", params={"path": "pkg/service.py"}).json()
    assert d["imports"] == ["pkg/models.py", "pkg/utils.py"]
    t = indexed_client.get("/trace", params={"src": "greet", "dst": "bark"}).json()
    assert t["edges"] == ["calls"]
    st = indexed_client.get("/stats").json()["index"]
    assert st["files"] == 7
    assert indexed_client.get("/symbol", params={"name": "nope_nope"}).status_code == 404


def test_update_endpoint_reports_incremental_work(indexed_client, sample_repo):
    f = sample_repo / "pkg" / "service.py"
    f.write_text(f.read_text() + "\n\ndef extra():\n    return 1\n")
    r = indexed_client.post("/index/update").json()
    assert r["changed"] == 1 and r["embedded"] < r["chunks"]


def _sig(secret, body):
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def test_webhook_ping_and_ignored_events(client):
    assert client.post("/webhook/github", content=b"{}", headers={"X-GitHub-Event": "ping"}).json()["msg"] == "pong"
    assert client.post("/webhook/github", content=b"{}", headers={"X-GitHub-Event": "issues"}).json()["ignored"] == "issues"


def test_webhook_signature_check(client, monkeypatch):
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", "s3cret")
    body = json.dumps({"ref": "refs/heads/main"}).encode()
    bad = client.post("/webhook/github", content=body, headers={"X-GitHub-Event": "ping", "X-Hub-Signature-256": "sha256=nope"})
    assert bad.status_code == 401
    ok = client.post("/webhook/github", content=body,
                     headers={"X-GitHub-Event": "ping", "X-Hub-Signature-256": _sig("s3cret", body)})
    assert ok.status_code == 200


def test_push_webhook_triggers_incremental_update(indexed_client, sample_repo):
    f = sample_repo / "pkg" / "service.py"
    f.write_text(f.read_text() + "\n\ndef pushed():\n    return 2\n")
    r = indexed_client.post("/webhook/github", content=b"{}", headers={"X-GitHub-Event": "push"})
    assert r.json()["queued"] is True
    # TestClient runs background tasks before returning, so the update has happened
    syms = indexed_client.get("/symbol", params={"name": "pushed"})
    assert syms.status_code == 200


def test_push_webhook_before_index_is_409(client):
    assert client.post("/webhook/github", content=b"{}", headers={"X-GitHub-Event": "push"}).status_code == 409
