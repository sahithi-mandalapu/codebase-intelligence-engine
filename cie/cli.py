from __future__ import annotations

import argparse
import json
import sys

from .config import Settings
from .qa import Engine


def _engine(args) -> Engine:
    s = Settings()
    if getattr(args, "data_dir", None):
        s.data_dir = args.data_dir
    return Engine(s, args.name)


def cmd_index(args):
    print(json.dumps(_engine(args).index_repo(args.source), indent=2))


def cmd_update(args):
    print(json.dumps(_engine(args).update(), indent=2))


def cmd_ask(args):
    eng = _engine(args)
    ans = eng.ask(args.question, k=args.k, use_graph=not args.no_graph, use_rerank=not args.no_rerank)
    print(ans.answer)
    print("\nsources:")
    for c in ans.citations:
        print(f"  {c['path']}:{c['start']}-{c['end']}  {c['name']}  [{', '.join(c['sources'])}]")
    print(f"\n({ans.timings_ms.get('total')} ms)")


def cmd_symbol(args):
    g = _engine(args).index.graph
    ids = g.find_symbols(args.query)
    if not ids:
        print("no match")
        return 1
    for sid in ids[:10]:
        s = g.symbols[sid]
        print(f"{s.kind} {s.qualname}  ({s.path}:{s.start}-{s.end})")
        for key, nodes in g.neighbors(sid).items():
            if nodes:
                print(f"  {key}: " + ", ".join(n.split('::', 1)[-1] + f" [{n.split('::')[0]}]" for n in nodes[:12]))


def cmd_deps(args):
    g = _engine(args).index.graph
    files = g.find_files(args.path)
    if not files:
        print("no such file")
        return 1
    p = files[0]
    print(p)
    print("  imports:     " + (", ".join(g.imports(p)) or "-"))
    print("  imported by: " + (", ".join(g.importers(p)) or "-"))


def cmd_trace(args):
    res = _engine(args).index.graph.trace(args.src, args.dst)
    if not res:
        print("no path found")
        return 1
    for node, edge in zip(res["path"], res["edges"] + [""]):
        print(f"{node} {('--' + edge + '-->') if edge else ''}")


def cmd_stats(args):
    print(json.dumps(_engine(args).index.stats(), indent=2))


def cmd_serve(args):
    import uvicorn

    uvicorn.run("cie.api:app", host=args.host, port=args.port, reload=False)


def main(argv=None):
    p = argparse.ArgumentParser(prog="cie", description="ask questions about a codebase")
    p.add_argument("--name", default="default", help="index name (lets you keep several repos)")
    p.add_argument("--data-dir", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("index", help="index a local folder or git url")
    a.add_argument("source")
    a.set_defaults(fn=cmd_index)

    a = sub.add_parser("update", help="re-index only what changed")
    a.set_defaults(fn=cmd_update)

    a = sub.add_parser("ask", help="ask a question")
    a.add_argument("question")
    a.add_argument("-k", type=int, default=8)
    a.add_argument("--no-graph", action="store_true")
    a.add_argument("--no-rerank", action="store_true")
    a.set_defaults(fn=cmd_ask)

    a = sub.add_parser("symbol", help="show a symbol and its callers / callees")
    a.add_argument("query")
    a.set_defaults(fn=cmd_symbol)

    a = sub.add_parser("deps", help="show what a file imports and who imports it")
    a.add_argument("path")
    a.set_defaults(fn=cmd_deps)

    a = sub.add_parser("trace", help="shortest dependency path between two symbols/files")
    a.add_argument("src")
    a.add_argument("dst")
    a.set_defaults(fn=cmd_trace)

    a = sub.add_parser("stats")
    a.set_defaults(fn=cmd_stats)

    a = sub.add_parser("serve", help="run the api")
    a.add_argument("--host", default="127.0.0.1")
    a.add_argument("--port", type=int, default=8000)
    a.set_defaults(fn=cmd_serve)

    args = p.parse_args(argv)
    sys.exit(args.fn(args) or 0)


if __name__ == "__main__":
    main()
