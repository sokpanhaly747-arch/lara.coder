"""Decides the single next action given the current AgentState.

This is a pure-ish decision function: (state) -> next phase / action.
Keeping it separate from `loop.py` (which actually executes actions)
makes the decision policy independently testable.
"""
from __future__ import annotations

from agent.config import AgentConfig
from agent.state import AgentState, Phase


def next_phase(state: AgentState, config: AgentConfig) -> Phase:
    """Pure decision: what phase should run next, given state + limits."""
    limits = config.limits

    if state.phase == Phase.UNDERSTAND:
        return Phase.PLAN if state.requirements else Phase.UNDERSTAND

    if state.phase == Phase.PLAN:
        return Phase.DESIGN if state.product_plan else Phase.PLAN

    if state.phase == Phase.DESIGN:
        return Phase.CODE if state.design_spec else Phase.DESIGN

    if state.phase == Phase.CODE:
        return Phase.RUN

    if state.phase == Phase.RUN:
        return Phase.TEST

    if state.phase == Phase.TEST:
        if state.open_errors:
            return Phase.DEBUG
        return Phase.REVIEW

    if state.phase == Phase.DEBUG:
        if state.fix_attempts_total >= limits.max_total_fix_attempts:
            return Phase.FAILED
        return Phase.RUN  # re-run after attempted fix

    if state.phase == Phase.REVIEW:
        report = state.quality_report or {}
        if report.get("passes_bar"):
            if state.improve_iterations >= limits.max_improve_iterations:
                return Phase.DONE
            return Phase.IMPROVE
        if state.build_iterations >= limits.max_build_iterations:
            return Phase.FAILED
        return Phase.PLAN  # quality bar not met, revise plan

    if state.phase == Phase.IMPROVE:
        return Phase.CODE

    return state.phase
