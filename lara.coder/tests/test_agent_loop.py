"""Exercises the full agent loop with fake phase handlers, proving the
orchestrator policy (agent/orchestrator.py) and loop limits
(agent/config.py) work end to end without needing a live model,
runtime, or browser."""
from __future__ import annotations

import pytest

from agent.config import AgentConfig, LoopLimits
from agent.loop import AgentLoop
from agent.state import AgentState, Phase, TaskResult


def make_config(**limit_overrides) -> AgentConfig:
    return AgentConfig(limits=LoopLimits(**limit_overrides))


async def ok(state: AgentState, config: AgentConfig, **kw) -> TaskResult:
    return TaskResult(task_id="t", phase=state.phase, success=True, summary="ok")


@pytest.mark.asyncio
async def test_happy_path_reaches_done():
    async def understand(state, config):
        state.requirements = {"summary": "x"}
        return TaskResult("t", state.phase, True, "understood")

    async def plan(state, config):
        state.product_plan = {"pages": []}
        return TaskResult("t", state.phase, True, "planned")

    async def design(state, config):
        state.design_spec = {"screens": []}
        return TaskResult("t", state.phase, True, "designed")

    async def review(state, config):
        state.quality_report = {"passes_bar": True}
        return TaskResult("t", state.phase, True, "reviewed")

    handlers = {
        Phase.UNDERSTAND: understand,
        Phase.PLAN: plan,
        Phase.DESIGN: design,
        Phase.CODE: ok,
        Phase.RUN: ok,
        Phase.TEST: ok,
        Phase.REVIEW: review,
        Phase.IMPROVE: ok,
    }
    config = make_config(max_improve_iterations=1)
    loop = AgentLoop(config=config, handlers=handlers)
    state = AgentState(project_id="p1", user_idea="a shop")

    final = await loop.run(state)

    assert final.phase == Phase.DONE
    assert final.build_iterations >= 1
    assert final.improve_iterations == 1


@pytest.mark.asyncio
async def test_debug_loop_respects_fix_attempt_limit():
    async def understand(state, config):
        state.requirements = {"summary": "x"}
        return TaskResult("t", state.phase, True, "ok")

    async def plan(state, config):
        state.product_plan = {"pages": []}
        return TaskResult("t", state.phase, True, "ok")

    async def design(state, config):
        state.design_spec = {"screens": []}
        return TaskResult("t", state.phase, True, "ok")

    async def test_phase(state, config):
        state.open_errors = [{"message": "still broken"}]  # never resolved
        return TaskResult("t", state.phase, True, "found error")

    async def debug(state, config):
        return TaskResult("t", state.phase, True, "attempted fix")

    handlers = {
        Phase.UNDERSTAND: understand,
        Phase.PLAN: plan,
        Phase.DESIGN: design,
        Phase.CODE: ok,
        Phase.RUN: ok,
        Phase.TEST: test_phase,
        Phase.DEBUG: debug,
    }
    config = make_config(max_total_fix_attempts=3)
    loop = AgentLoop(config=config, handlers=handlers)
    state = AgentState(project_id="p2", user_idea="a shop")

    final = await loop.run(state)

    assert final.phase == Phase.FAILED
    assert final.fix_attempts_total == 3  # bounded, not infinite


def test_task_graph_ready_tasks_respect_dependencies():
    from planning.schemas import Task, TaskGraph

    graph = TaskGraph(tasks=[
        Task(id="a", title="A", description=""),
        Task(id="b", title="B", description="", depends_on=["a"]),
    ])
    assert [t.id for t in graph.ready_tasks()] == ["a"]

    graph.tasks[0].status = "done"
    assert [t.id for t in graph.ready_tasks()] == ["b"]


def test_error_classifier_detects_missing_module():
    from debugging.error_classifier import classify, ErrorCategory

    result = classify("Error: Cannot find module 'react-dom/client' in src/index.tsx")
    assert result.category == ErrorCategory.MISSING_DEPENDENCY
    assert result.file_hint == "src/index.tsx"


def test_sandbox_blocks_path_escape():
    from runtime.sandbox import Sandbox

    with Sandbox() as sb:
        with pytest.raises(PermissionError):
            sb.resolve("../../etc/passwd")
