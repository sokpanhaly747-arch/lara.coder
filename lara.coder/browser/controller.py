"""Browser automation for the running preview app.

Built around Playwright's async API. Kept as a thin typed wrapper so
`testing.e2e_runner` and `debugging` can drive it without depending on
Playwright directly, which keeps the browser engine swappable.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ClickResult:
    success: bool
    error: str | None = None


class BrowserController:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self._browser = None
        self._page = None

    async def start(self) -> None:
        from playwright.async_api import async_playwright

        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch()
        self._page = await self._browser.new_page()

    async def goto(self, path: str = "/") -> None:
        await self._page.goto(f"{self.base_url}{path}")

    async def click(self, selector: str) -> ClickResult:
        try:
            await self._page.click(selector, timeout=5000)
            return ClickResult(success=True)
        except Exception as exc:  # noqa: BLE001
            return ClickResult(success=False, error=str(exc))

    async def get_console_errors(self) -> list[str]:
        # Layer-0 placeholder: wire up page.on("console") / page.on("pageerror")
        # listeners in start() and collect here once needed by testing/debugging.
        return []

    async def close(self) -> None:
        if self._browser:
            await self._browser.close()
        if getattr(self, "_playwright", None):
            await self._playwright.stop()
