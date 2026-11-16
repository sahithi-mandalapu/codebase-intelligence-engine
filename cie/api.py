import hashlib
import hmac
import os

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from pydantic import BaseModel

from ._version import __version__
from .config import Settings
from .qa import Engine


class IndexRequest(BaseModel):
    source: str  # local path or git url
    name: str = "default"


class AskRequest(BaseModel):
    question: str
    name: str = "default"
    k: int = 8
    use_graph: bool = True
    use_rerank: bool = True
    use_cache: bool = True


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    app = FastAPI(title="Codebase Intelligence Engine", version=__version__)
    engines: dict[str, Engine] = {}

    def engine(name: str) -> Engine:
        if name not in engines:
            engines[name] = Engine(settings, name)
        return engines[name]

    def indexed(name: str) -> Engine:
        eng = engine(name)
        if not eng.index.files:
            raise HTTPException(409, f"repo '{name}' isn't indexed yet, POST /index first")
        return eng

    @app.get("/health")
    def health():
        return {"status": "ok", "version": __version__}

    @app.post("/index")
    def index_repo(req: IndexRequest):
        try:
            return engine(req.name).index_repo(req.source)
        except FileNotFoundError as e:
            raise HTTPException(404, str(e))

    @app.post("/index/update")
    def update(name: str = "default"):
        try:
            return engine(name).update()
        except RuntimeError as e:
            raise HTTPException(409, str(e))

    @app.post("/ask")
    def ask(req: AskRequest):
        eng = indexed(req.name)
        return eng.ask(req.question, k=req.k, use_graph=req.use_graph, use_rerank=req.use_rerank,
                       use_cache=req.use_cache).to_dict()

    @app.get("/symbol")
    def symbol(name: str, repo: str = "default"):
        eng = indexed(repo)
        g = eng.index.graph
        ids = g.find_symbols(name)
        if not ids:
            raise HTTPException(404, f"no symbol matching '{name}'")
        out = []
        for sid in ids[:10]:
            s = g.symbols[sid]
            out.append({
                "id": sid, "kind": s.kind, "path": s.path, "start": s.start, "end": s.end,
                "doc": s.doc.split("\n")[0] if s.doc else "",
                **g.neighbors(sid),
            })
        return {"matches": out}

    @app.get("/deps")
    def deps(path: str, repo: str = "default"):
        eng = indexed(repo)
        g = eng.index.graph
        found = g.find_files(path)
        if not found:
            raise HTTPException(404, f"no file matching '{path}'")
        p = found[0]
        return {"path": p, "imports": g.imports(p), "imported_by": g.importers(p)}

    @app.get("/trace")
    def trace(src: str, dst: str, repo: str = "default"):
        eng = indexed(repo)
        res = eng.index.graph.trace(src, dst)
        if res is None:
            raise HTTPException(404, "couldn't find a path between those two")
        return res

    @app.get("/stats")
    def stats(repo: str = "default"):
        eng = indexed(repo)
        return {"index": eng.index.stats(), "cache": eng.cache.stats()}

    @app.post("/webhook/github")
    async def github_webhook(request: Request, background: BackgroundTasks, name: str = "default"):
        body = await request.body()
        secret = os.getenv("GITHUB_WEBHOOK_SECRET")
        if secret:
            sig = request.headers.get("X-Hub-Signature-256", "")
            expected = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
            if not hmac.compare_digest(sig, expected):
                raise HTTPException(401, "bad signature")
        event = request.headers.get("X-GitHub-Event", "")
        if event == "ping":
            return {"ok": True, "msg": "pong"}
        if event != "push":
            return {"ok": True, "ignored": event}
        eng = engine(name)
        if not eng.index.repo_root:
            raise HTTPException(409, "nothing indexed yet")
        background.add_task(eng.pull_and_update)
        return {"ok": True, "queued": True}

    return app


app = create_app()
