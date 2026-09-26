"""Start/stop/health-check long-running dev servers (e.g. `npm run dev`).

Separate from `terminal_tool` because a dev server is a background
process the browser subsystem needs a stable URL for, not a one-shot
command whose exit code we wait on.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from runtime.sandbox import Sandbox


@dataclass
class ManagedProcess:
    process: asyncio.subprocess.Process
    url: str


@dataclass
class ProcessManager:
    sandbox: Sandbox
    _procs: dict[str, ManagedProcess] | None = None

    def __post_init__(self):
        self._procs = {}

    async def start(self, name: str, command: str, url: str, *, ready_timeout: int = 30) -> ManagedProcess:
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=str(self.sandbox.root),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        managed = ManagedProcess(process=proc, url=url)
        self._procs[name] = managed
        await self._wait_until_ready(url, ready_timeout)
        return managed

    async def _wait_until_ready(self, url: str, timeout: int) -> None:
        # Layer-0: naive polling placeholder. A later layer swaps in a real
        # HTTP health check once an HTTP client dependency is added.
        deadline = time.time() + timeout
        while time.time() < deadline:
            await asyncio.sleep(1)
            return  # placeholder: assume ready after first tick

    async def stop(self, name: str) -> None:
        managed = self._procs.pop(name, None)
        if managed:
            managed.process.terminate()
            await managed.process.wait()

    async def stop_all(self) -> None:
        for name in list(self._procs):
            await self.stop(name)
