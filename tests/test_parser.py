import textwrap

from cie.ingest import parse_file


def _py(src, path="m.py"):
    return parse_file(path, "python", textwrap.dedent(src).lstrip("\n"))


def test_python_symbols_and_qualnames():
    info = _py('''
        class A:
            def f(self):
                return g()

        def g():
            pass
    ''')
    quals = {s.qualname: s.kind for s in info.symbols}
    assert quals == {"A": "class", "A.f": "method", "g": "function"}
    f = next(s for s in info.symbols if s.qualname == "A.f")
    assert "g" in f.calls


def test_class_chunk_is_just_the_header():
    info = _py('''
        class A:
            """doc"""
            x = 1

            def f(self):
                return 1
    ''')
    cls = next(c for c in info.chunks if c.kind == "class")
    assert "def f" not in cls.text
    assert any(c.kind == "method" and c.qualname == "A.f" for c in info.chunks)


def test_bases_and_decorators():
    info = _py('''
        import abc

        @abc.something
        class B(abc.ABC, Generic[T]):
            pass
    ''')
    b = next(s for s in info.symbols if s.name == "B")
    assert b.bases == ["abc.ABC", "Generic"]
    assert b.start == 3  # decorator line is included


def test_call_forms():
    info = _py('''
        def f(self):
            plain()
            self.method()
            mod.func()
            self.a.b()
    ''')
    calls = set(info.symbols[0].calls)
    assert {"plain", "self.method", "mod.func", ".b"} <= calls


def test_relative_imports_are_resolved_to_dotted_paths():
    info = _py("from .models import Dog\nfrom . import utils\nfrom ..top import thing\n", path="pkg/sub/service.py")
    assert "pkg.sub.models" in info.imports
    assert "pkg.sub.utils" in info.imports
    assert "pkg.top" in info.imports
    assert info.import_names["Dog"] == "pkg.sub.models:Dog"


def test_nested_defs_inside_if_are_found():
    info = _py('''
        if True:
            def inner():
                pass
    ''')
    assert [s.name for s in info.symbols] == ["inner"]


def test_long_function_is_split_into_windows():
    body = "\n".join(f"    x{i} = {i}" for i in range(200))
    info = _py(f"def big():\n{body}\n")
    parts = [c for c in info.chunks if c.qualname == "big"]
    assert len(parts) >= 3
    assert len({c.id for c in parts}) == len(parts)


def test_syntax_error_falls_back_to_windows():
    info = parse_file("bad.py", "python", "def (:\n  nope\n")
    assert info.symbols == []
    assert info.chunks and info.chunks[0].kind == "window"


def test_chunk_ids_survive_moving_code_down():
    a = _py("def f():\n    return 1\n")
    b = _py("# new comment\n\ndef f():\n    return 1\n")
    ca = next(c for c in a.chunks if c.qualname == "f")
    cb = next(c for c in b.chunks if c.qualname == "f")
    assert ca.id == cb.id and ca.start != cb.start


def test_javascript_functions_classes_methods():
    src = textwrap.dedent('''
        import { helper } from './util.js';
        const fs = require('fs');

        export function render(items) {
          return items.map(i => helper(i));
        }

        const add = (a, b) => a + b;

        class View extends Base {
          constructor(el) { super(); this.el = el; }
          draw() { return render(this.items); }
        }
    ''').lstrip("\n")
    info = parse_file("app.js", "javascript", src)
    by = {s.qualname: s for s in info.symbols}
    assert set(by) == {"render", "add", "View", "View.constructor", "View.draw"}
    assert by["View"].bases == ["Base"]
    assert "helper" in by["render"].calls
    assert "self.draw" not in by["View.draw"].calls and "render" in by["View.draw"].calls
    assert "./util.js" in info.imports and "fs" in info.imports


def test_markdown_split_by_heading():
    info = parse_file("README.md", "markdown", "# a\n\none\n\n## b\n\ntwo\n")
    assert [c.name for c in info.chunks] == ["a", "b"]


def test_unknown_language_gets_windows():
    info = parse_file("x.go", "go", "package main\n" + "\n".join(f"// line {i}" for i in range(130)))
    assert len(info.chunks) >= 2 and all(c.kind == "window" for c in info.chunks)
