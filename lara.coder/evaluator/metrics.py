"""Scoring functions for each measurable dimension of a build run.

`evaluate_run` produces the `QualityReport` (as a dict) that
`agent.orchestrator.next_phase` reads via `state.quality_report` to
decide REVIEW -> DONE vs REVIEW -> PLAN.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from agent.state import AgentState


@dataclass
class QualityReport:
    build_success: bool
    test_pass_rate: float          # 0..1
    unresolved_errors: int
    visual_issues: int
    accessibility_issues: int
    passes_bar: bool
    notes: list[str] = field(default_factory=list)


# Configurable thresholds — tune per project maturity, not hardcoded logic.
MIN_TEST_PASS_RATE = 0.9
MAX_UNRESOLVED_ERRORS = 0
MAX_ACCESSIBILITY_ISSUES = 0


def evaluate_run(
    state: AgentState,
    test_pass_rate: float,
    visual_issues: int,
    accessibility_issues: int,
) -> QualityReport:
    build_success = state.phase.value not in ("failed",)
    unresolved = len(state.open_errors)

    passes_bar = (
        build_success
        and test_pass_rate >= MIN_TEST_PASS_RATE
        and unresolved <= MAX_UNRESOLVED_ERRORS
        and accessibility_issues <= MAX_ACCESSIBILITY_ISSUES
    )

    notes = []
    if not build_success:
        notes.append("Build did not complete successfully.")
    if test_pass_rate < MIN_TEST_PASS_RATE:
        notes.append(f"Test pass rate {test_pass_rate:.0%} below bar {MIN_TEST_PASS_RATE:.0%}.")
    if unresolved:
        notes.append(f"{unresolved} unresolved error(s).")
    if accessibility_issues:
        notes.append(f"{accessibility_issues} accessibility issue(s).")

    return QualityReport(
        build_success=build_success,
        test_pass_rate=test_pass_rate,
        unresolved_errors=unresolved,
        visual_issues=visual_issues,
        accessibility_issues=accessibility_issues,
        passes_bar=passes_bar,
        notes=notes,
    )
