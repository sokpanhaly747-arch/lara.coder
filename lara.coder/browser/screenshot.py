"""Screenshot capture and pixel-diff for visual regression / review."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from browser.controller import BrowserController


@dataclass
class ScreenshotResult:
    path: str
    width: int
    height: int


async def capture(controller: BrowserController, out_path: str) -> ScreenshotResult:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    await controller._page.screenshot(path=out_path, full_page=True)
    box = await controller._page.evaluate(
        "() => ({w: document.body.scrollWidth, h: document.body.scrollHeight})"
    )
    return ScreenshotResult(path=out_path, width=box["w"], height=box["h"])


def diff(before_path: str, after_path: str) -> float:
    """Returns a 0..1 dissimilarity score. Placeholder until an image-diff
    dependency (e.g. Pillow + numpy) is added in a later layer."""
    return 0.0
