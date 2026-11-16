from __future__ import annotations

import os
import subprocess
from pathlib import Path

LANGS = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".java": "java",
    ".rs": "rust",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".rb": "ruby",
    ".md": "markdown",
    ".rst": "markdown",
}

SKIP_DIRS = {
    ".git", "node_modules", "venv", ".venv", "env", "__pycache__", "dist", "build",
    ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache", "site-packages", ".idea",
    ".vscode", ".cie_data", "vendor", "third_party",
}

MAX_BYTES = 400_000  # skip giant generated files


def clone(url: str, dest: str) -> str:
    """shallow clone (or fast-forward pull if we already have it)."""
    if os.path.isdir(os.path.join(dest, ".git")):
        subprocess.run(["git", "-C", dest, "pull", "--ff-only"], capture_output=True, timeout=300)
        return dest
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", url, dest], check=True, capture_output=True, timeout=600)
    return dest


def walk(root: str):
    """yields (relative_posix_path, language, text) for every source file we care about."""
    root_path = Path(root).resolve()
    for dirpath, dirnames, filenames in os.walk(root_path):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.endswith(".egg-info"))
        for fn in sorted(filenames):
            lang = LANGS.get(os.path.splitext(fn)[1].lower())
            if lang is None:
                continue
            full = Path(dirpath) / fn
            try:
                if full.stat().st_size > MAX_BYTES:
                    continue
                text = full.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            yield full.relative_to(root_path).as_posix(), lang, text
