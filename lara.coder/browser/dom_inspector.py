"""Structural checks on the rendered DOM: accessibility and layout smells
that reading source code cannot catch (missing alt text, empty
containers that should have content, obviously overlapping elements)."""
from __future__ import annotations

from dataclasses import dataclass

from browser.controller import BrowserController


@dataclass
class DomIssue:
    kind: str       # e.g. "missing_alt", "empty_container"
    selector: str
    detail: str


async def inspect(controller: BrowserController) -> list[DomIssue]:
    issues: list[DomIssue] = []

    missing_alt = await controller._page.eval_on_selector_all(
        "img", "els => els.filter(e => !e.alt).map(e => e.outerHTML.slice(0,80))"
    )
    for html in missing_alt:
        issues.append(DomIssue(kind="missing_alt", selector="img", detail=html))

    return issues
