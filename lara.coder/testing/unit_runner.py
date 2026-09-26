"""Runs the generated project's own unit/integration test suite.

Delegates to whatever test command the stack config defines (e.g.
`npm test`) via `runtime.terminal_tool`, then parses output into a
structured result the orchestrator/evaluator can consume.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from runtime.terminal_tool import TerminalTool


@dataclass
class UnitTestResult:
    passed: bool
    total: int
    failed: int
    failures: list[str] = field(default_factory=list)
    raw_output: str = ""


async def run_unit_tests(terminal: TerminalTool, command: str = "npm test -- --run") -> UnitTestResult:
    result = await terminal.run(command, timeout_seconds=180)
    output = result.stdout + result.stderr
    # Layer-0: naive pass/fail; a later layer parses the test framework's
    # structured (JSON) reporter output instead of exit-code-only.
    return UnitTestResult(
        passed=result.exit_code == 0,
        total=0,
        failed=0 if result.exit_code == 0 else 1,
        failures=[] if result.exit_code == 0 else [output[-2000:]],
        raw_output=output,
    )
