"""Pluggable persistence backend. Layer-0: local JSON files on disk.

Swappable later for a real database without changing
`project_memory.py`'s public API — callers only depend on `MemoryStore`.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class MemoryStore(ABC):
    @abstractmethod
    def get(self, key: str) -> Any | None: ...

    @abstractmethod
    def set(self, key: str, value: Any) -> None: ...

    @abstractmethod
    def list_keys(self, prefix: str = "") -> list[str]: ...

    @abstractmethod
    def delete(self, key: str) -> None: ...


class JsonFileStore(MemoryStore):
    def __init__(self, root: str = ".laracoder_memory"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        safe = key.replace("/", "__")
        return self.root / f"{safe}.json"

    def get(self, key: str) -> Any | None:
        path = self._path(key)
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def set(self, key: str, value: Any) -> None:
        self._path(key).write_text(json.dumps(value, indent=2, default=str))

    def list_keys(self, prefix: str = "") -> list[str]:
        return sorted(
            p.stem.replace("__", "/") for p in self.root.glob("*.json")
            if p.stem.replace("__", "/").startswith(prefix)
        )

    def delete(self, key: str) -> None:
        path = self._path(key)
        if path.exists():
            path.unlink()
