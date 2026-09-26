"""Screenshot-based visual quality checks.

Distinct from `browser.dom_inspector` (structural) — this looks at the
rendered pixels: does the screenshot's aspect/size look reasonable,
and (once a vision-capable review step is wired up) does it look
polished, not just "not broken".
"""
from __future__ import annotations

from dataclasses import dataclass

from browser.screenshot import ScreenshotResult


@dataclass
class VisualCheck:
    passed: bool
    notes: list[str]


def basic_sanity_check(shot: ScreenshotResult) -> VisualCheck:
    notes: list[str] = []
    if shot.height < 100:
        notes.append("Page height suspiciously small — likely render failure")
    if shot.width < 300:
        notes.append("Page width suspiciously small")
    return VisualCheck(passed=not notes, notes=notes)
