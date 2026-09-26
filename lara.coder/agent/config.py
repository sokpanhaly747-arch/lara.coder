"""Run configuration and safety limits for the agent loop.

Every autonomous retry path in this project (debugging, improvement
iterations) must be bounded by something in here. No module should
hardcode its own "try forever" logic.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LoopLimits:
    max_build_iterations: int = 10       # plan -> code -> run -> test cycles
    max_fix_attempts_per_error: int = 3   # debugging.auto_fix retries for one error
    max_total_fix_attempts: int = 15      # global cap across all errors in a run
    max_improve_iterations: int = 5       # post-passing "make it better" cycles
    per_step_timeout_seconds: int = 120
    total_run_timeout_seconds: int = 3600


@dataclass(frozen=True)
class AgentConfig:
    limits: LoopLimits
    tech_stack: str = "nextjs-typescript-tailwind"
    workdir: str = "/workspace"
    require_human_approval_for: tuple[str, ...] = ("delete_project", "deploy")
