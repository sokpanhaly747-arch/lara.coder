"""Outer run loop: orchestrate -> act -> observe -> persist -> repeat.

This file wires together every other module but contains almost no
policy of its own — policy lives in `orchestrator.py`, limits in
`config.py`. Concrete `act()` implementations are injected so this
loop can be unit tested with fakes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable, Protocol

from agent.config import AgentConfig
from agent.orchestrator import next_phase
from agent.state import AgentState, Phase, TaskResult


class PhaseHandler(Protocol):
    async def __call__(self, state: AgentState, config: AgentConfig) -> TaskResult: ...


@dataclass
class AgentLoop:
    config: AgentConfig
    handlers: dict[Phase, PhaseHandler]
    on_step: Callable[[AgentState, TaskResult], Awaitable[None]] | None = None

    async def run(self, state: AgentState) -> AgentState:
        while not state.is_terminal():
            phase = next_phase(state, self.config)
            state.phase = phase

            if state.is_terminal():
                break

            handler = self.handlers.get(phase)
            if handler is None:
                raise RuntimeError(f"No handler registered for phase {phase}")

            result = await handler(state, self.config)
            state.record(result)

            if phase == Phase.CODE:
                state.build_iterations += 1
            if phase == Phase.DEBUG:
                state.fix_attempts_total += 1
            if phase == Phase.IMPROVE:
                state.improve_iterations += 1

            if self.on_step:
                await self.on_step(state, result)

            if not result.success and phase not in (Phase.TEST, Phase.DEBUG, Phase.REVIEW):
                state.phase = Phase.FAILED
                break

        return state
