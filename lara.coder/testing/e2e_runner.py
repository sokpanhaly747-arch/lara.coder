"""Drives the browser through the product's key user flows.

Flows come from `planning.schemas.ProductPlan.user_flows`; each flow
is executed step by step against `browser.controller`, capturing
console errors and failed interactions as structured findings for
`debugging/`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from browser.controller import BrowserController


@dataclass
class FlowStepResult:
    step: str
    success: bool
    error: str | None = None


@dataclass
class E2EResult:
    flow_name: str
    passed: bool
    steps: list[FlowStepResult] = field(default_factory=list)
    console_errors: list[str] = field(default_factory=list)


async def run_flow(controller: BrowserController, flow_name: str, steps: list[dict]) -> E2EResult:
    step_results: list[FlowStepResult] = []
    for step in steps:
        action = step.get("action")
        if action == "goto":
            await controller.goto(step["path"])
            step_results.append(FlowStepResult(step=str(step), success=True))
        elif action == "click":
            res = await controller.click(step["selector"])
            step_results.append(FlowStepResult(step=str(step), success=res.success, error=res.error))
        else:
            step_results.append(FlowStepResult(step=str(step), success=False, error="unknown action"))

    console_errors = await controller.get_console_errors()
    passed = all(s.success for s in step_results) and not console_errors
    return E2EResult(flow_name=flow_name, passed=passed, steps=step_results, console_errors=console_errors)
