def test_callers_and_callees(index):
    g = index.graph
    bark = "pkg/models.py::bark"
    assert g.callers(bark) == ["pkg/models.py::Dog.speak", "pkg/service.py::greet"]
    assert "pkg/models.py::Dog" in g.callees("pkg/service.py::make_dog")


def test_submodule_alias_call_is_resolved(index):
    # `from . import utils` then `utils.register(d)`
    assert "pkg/utils.py::register" in index.graph.callees("pkg/service.py::make_dog")


def test_inheritance(index):
    g = index.graph
    assert g.bases("pkg/models.py::Dog") == ["pkg/models.py::Animal"]
    assert g.subclasses("pkg/models.py::Animal") == ["pkg/models.py::Dog"]


def test_file_imports(index):
    g = index.graph
    assert g.imports("pkg/service.py") == ["pkg/models.py", "pkg/utils.py"]
    assert g.importers("pkg/models.py") == ["pkg/service.py"]
    # package __init__ shouldn't show up just because of `from . import utils`
    assert "pkg/__init__.py" not in g.imports("pkg/service.py")


def test_js_import_and_call_edges(index):
    g = index.graph
    assert g.imports("web/app.js") == ["web/util.js"]
    assert "web/util.js::helper" in g.callees("web/app.js::render")
    assert "web/app.js::render" in g.callees("web/app.js::View.draw")


def test_find_symbols_and_files(index):
    g = index.graph
    assert g.find_symbols("bark") == ["pkg/models.py::bark"]
    assert g.find_symbols("`Dog.speak`") == ["pkg/models.py::Dog.speak"]
    assert g.find_symbols("speak") == ["pkg/models.py::Animal.speak", "pkg/models.py::Dog.speak"]
    assert g.find_files("service.py") == ["pkg/service.py"]
    assert g.find_files("service") == ["pkg/service.py"]
    assert g.find_files("nope.py") == []


def test_self_calls_resolve_through_base_classes():
    from cie.graph import CodeGraph
    from cie.ingest import parse_file

    src = "class A:\n    def helper(self):\n        pass\n\nclass B(A):\n    def run(self):\n        self.helper()\n"
    g = CodeGraph().build({"m.py": parse_file("m.py", "python", src)})
    assert g.callees("m.py::B.run") == ["m.py::A.helper"]


def test_common_method_names_are_not_linked():
    from cie.graph import CodeGraph
    from cie.ingest import parse_file

    src = "def get():\n    pass\n\ndef f(d):\n    d.get('x')\n"
    g = CodeGraph().build({"m.py": parse_file("m.py", "python", src)})
    assert g.callees("m.py::f") == []


def test_trace(index):
    res = index.graph.trace("make_dog", "bark")
    assert res is None or res["path"][0] == "pkg/service.py::make_dog"
    res = index.graph.trace("greet", "bark")
    assert res["path"] == ["pkg/service.py::greet", "pkg/models.py::bark"]
    assert res["edges"] == ["calls"] and res["directed"]


def test_stats(index):
    s = index.graph.stats()
    assert s["files"] == 7 and s["edges"]["calls"] >= 4
