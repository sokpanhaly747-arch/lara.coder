"""Structured agent state — the source of truth for a build run.

Conversation text is an input/output channel; this object is what the
orchestrator actually reasons over and what memory/ persists. Every
field here should be enough to resume a crashed run without replaying
the whole chat.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Phase(str, Enum):
    UNDERSTAND = "understand"
    PLAN = "plan"
    DESIGN = "design"
    CODE = "code"
    RUN = "run"
    TEST = "test"
    DEBUG = "debug"
    REVIEW = "review"
    IMPROVE = "improve"
    DONE = "done"
    FAILED = "failed"


@dataclass
class TaskResult:
    task_id: str
    phase: Phase
    success: bool
    summary: str
    artifacts: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentState:
    project_id: str
    user_idea: str
    phase: Phase = Phase.UNDERSTAND

    requirements: dict | None = None      # planning.schemas.Requirements
    product_plan: dict | None = None      # planning.schemas.ProductPlan
    design_spec: dict | None = None       # design.schemas.DesignSpec
    task_graph: dict | None = None        # planning.schemas.TaskGraph

    files_written: list[str] = field(default_factory=list)
    history: list[TaskResult] = field(default_factory=list)

    build_iterations: int = 0
    fix_attempts_total: int = 0
    improve_iterations: int = 0

    open_errors: list[dict] = field(default_factory=list)
    quality_report: dict | None = None

    def record(self, result: TaskResult) -> None:
        self.history.append(result)

    def is_terminal(self) -> bool:
        return self.phase in (Phase.DONE, Phase.FAILED)
