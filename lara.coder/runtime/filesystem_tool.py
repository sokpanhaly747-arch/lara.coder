"""Typed, path-jailed filesystem operations for the agent's file tool.

This is the only way codegen output ever reaches disk. All paths are
resolved through `Sandbox.resolve`, so a malicious or buggy generated
path (e.g. "../../etc/passwd") is rejected rather than written.
"""
from __future__ import annotations

from dataclasses import dataclass

from runtime.sandbox import Sandbox


@dataclass
class FileSystemTool:
    sandbox: Sandbox

    def write(self, path: str, content: str) -> str:
        target = self.sandbox.resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return str(target)

    def read(self, path: str) -> str:
        return self.sandbox.resolve(path).read_text(encoding="utf-8")

    def list(self, path: str = ".") -> list[str]:
        base = self.sandbox.resolve(path)
        return sorted(str(p.relative_to(self.sandbox.root)) for p in base.rglob("*") if p.is_file())

    def exists(self, path: str) -> bool:
        return self.sandbox.resolve(path).exists()

    def delete(self, path: str) -> None:
        target = self.sandbox.resolve(path)
        if target.exists():
            target.unlink()
