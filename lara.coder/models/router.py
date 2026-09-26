"""Routes a task to the right model adapter.

Different stages of the pipeline have different cost/quality tradeoffs:
requirement analysis and classification can use a small/cheap model,
while code generation and root-cause debugging want the strongest
available model. The router centralizes that policy so it can be
tuned without touching callers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from models.adapter import ModelAdapter


class TaskKind(str, Enum):
    CLASSIFY = "classify"          # cheap: intent/error classification
    PLAN = "plan"                  # mid: requirements, task graphs
    DESIGN = "design"               # mid: UX/UI specs
    CODEGEN = "codegen"             # strong: writing source files
    DEBUG = "debug"                 # strong: root cause + fix
    REVIEW = "review"               # mid: quality/UX review


@dataclass
class ModelRouter:
    """Maps TaskKind -> registered ModelAdapter."""

    adapters: dict[TaskKind, ModelAdapter] = field(default_factory=dict)
    default: ModelAdapter | None = None

    def register(self, kind: TaskKind, adapter: ModelAdapter) -> None:
        self.adapters[kind] = adapter

    def get(self, kind: TaskKind) -> ModelAdapter:
        adapter = self.adapters.get(kind, self.default)
        if adapter is None:
            raise RuntimeError(f"No model adapter registered for {kind} and no default set")
        return adapter
