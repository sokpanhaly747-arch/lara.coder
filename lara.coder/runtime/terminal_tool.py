"""Typed, bounded command execution inside the sandbox.

Every call has a timeout and an output size cap so a runaway or
malicious generated command (infinite loop, huge stdout) cannot hang
or exhaust the orchestrator process.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

from runtime.sandbox import Sandbox


@dataclass
class CommandResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False


@dataclass
class TerminalTool:
    sandbox: Sandbox
    max_output_bytes: int = 200_000

    async def run(self, command: str, *, timeout_seconds: int = 60) -> CommandResult:
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=str(self.sandbox.root),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_seconds)
            return CommandResult(
                exit_code=proc.returncode or 0,
                stdout=stdout[: self.max_output_bytes].decode(errors="replace"),
                stderr=stderr[: self.max_output_bytes].decode(errors="replace"),
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return CommandResult(exit_code=-1, stdout="", stderr="command timed out", timed_out=True)
