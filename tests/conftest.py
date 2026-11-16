import textwrap

import pytest

from cie.config import Settings
from cie.embeddings import HashEmbedder
from cie.indexer import Index

FILES = {
    "pkg/__init__.py": "",
    "pkg/models.py": '''
        class Animal:
            """Base class for animals."""

            def __init__(self, name):
                self.name = name

            def speak(self):
                return "..."


        class Dog(Animal):
            """A dog that barks."""

            def speak(self):
                return bark(self.name)


        def bark(name):
            """Make a dog noise for the given name."""
            return f"{name} says woof"
    ''',
    "pkg/utils.py": '''
        REGISTRY = []


        def register(animal):
            """Add an animal to the global registry."""
            REGISTRY.append(animal)
            return len(REGISTRY)
    ''',
    "pkg/service.py": '''
        from .models import Dog, bark
        from . import utils


        def make_dog(name):
            """Create a dog and register it."""
            d = Dog(name)
            utils.register(d)
            return d


        def greet(name):
            return bark(name)
    ''',
    "web/app.js": '''
        import { helper } from './util.js';

        export function render(items) {
          return items.map(i => helper(i));
        }

        class View extends Base {
          constructor(el) { super(); this.el = el; }
          draw() { return render(this.items); }
        }
    ''',
    "web/util.js": '''
        export const helper = (x) => {
          return x * 2;
        };
    ''',
    "README.md": "# demo\n\nsmall zoo project\n\n## usage\n\nrun make_dog to get a dog\n",
}


@pytest.fixture
def sample_repo(tmp_path):
    root = tmp_path / "repo"
    for rel, body in FILES.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(body).lstrip("\n"))
    return root


@pytest.fixture
def index(sample_repo, tmp_path):
    idx = Index(str(tmp_path / "data"), HashEmbedder())
    idx.sync(str(sample_repo))
    return idx


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=str(tmp_path / "cie_data"), embedder="hash", reranker="lexical", llm="none",
                    vector_store="numpy", qdrant_url="", redis_url="")
