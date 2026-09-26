"""Sandbox lifecycle for running generated, untrusted code.

Layer-0 implementation uses a jailed local directory + subprocess with
resource limits, matching the "never allow untrusted generated code to
directly control the host machine" rule via strict path containment
and (in infrastructure/docker) container isolation for production use.
"""
from __future__ import annotations

import shutil
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Sandbox:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    root: Path | None = None

    def __enter__(self) -> "Sandbox":
        self.root = Path(tempfile.mkdtemp(prefix=f"laracoder-{self.id}-"))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.root and self.root.exists():
            shutil.rmtree(self.root, ignore_errors=True)

    def resolve(self, relative_path: str) -> Path:
        """Resolves a relative path inside the sandbox, refusing escapes."""
        if self.root is None:
            raise RuntimeError("Sandbox not entered")
        candidate = (self.root / relative_path).resolve()
        if self.root not in candidate.parents and candidate != self.root:
            raise PermissionError(f"Path escapes sandbox: {relative_path}")
        return candidate
