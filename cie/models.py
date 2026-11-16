from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field


def short_hash(*parts: str) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(p.encode("utf-8", "ignore"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


@dataclass
class Chunk:
    id: str
    path: str
    start: int  # 1-indexed, inclusive
    end: int
    text: str
    kind: str  # function | method | class | module | doc | window
    name: str = ""
    qualname: str = ""
    language: str = ""

    def header(self) -> str:
        label = self.qualname or self.name or self.kind
        return f"{self.path}:{self.start}-{self.end} ({self.kind} {label})"

    def search_text(self) -> str:
        # this is what gets embedded / indexed. path + name give the model a bit of context
        return f"{self.path} {self.kind} {self.qualname or self.name}\n{self.text}"


@dataclass
class Symbol:
    id: str  # "path::qualname"
    path: str
    name: str
    qualname: str
    kind: str  # function | method | class
    start: int
    end: int
    bases: list[str] = field(default_factory=list)
    # call strings look like "foo", "mod.foo", "self.foo" or ".foo" (unknown receiver)
    calls: list[str] = field(default_factory=list)
    doc: str = ""


@dataclass
class FileInfo:
    path: str
    language: str
    sha: str
    imports: list[str] = field(default_factory=list)
    # local name -> "module:original_name" ("module:" if it's a plain module import)
    import_names: dict[str, str] = field(default_factory=dict)
    symbols: list[Symbol] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "FileInfo":
        d = dict(d)
        d["symbols"] = [Symbol(**s) for s in d.get("symbols", [])]
        d["chunks"] = [Chunk(**c) for c in d.get("chunks", [])]
        return cls(**d)


def file_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()
