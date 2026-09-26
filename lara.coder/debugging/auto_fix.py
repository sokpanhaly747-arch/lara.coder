"""Applies a minimal patch for a diagnosed root cause, then hands back
to testing for re-verification. Bounded by
`LoopLimits.max_fix_attempts_per_error` at the call site (agent/loop.py) —
this module performs exactly one attempt per call."""
from __future__ import annotations

from dataclasses import dataclass

from debugging.root_cause import RootCauseAnalysis
from models.adapter import ModelAdapter, Message
from runtime.filesystem_tool import FileSystemTool


@dataclass
class FixAttempt:
    file_path: str
    applied: bool
    new_content: str | None = None
    error: str | None = None


FIX_SYSTEM = """You fix one specific bug at a time. Given the root cause,
fix description, and full current file content, return the COMPLETE
corrected file content only — no prose, no markdown fences, no diff
syntax, just the full new file."""


async def apply_fix(
    file_path: str,
    analysis: RootCauseAnalysis,
    fs: FileSystemTool,
    adapter: ModelAdapter,
) -> FixAttempt:
    try:
        current = fs.read(file_path)
    except FileNotFoundError as exc:
        return FixAttempt(file_path=file_path, applied=False, error=str(exc))

    prompt = (
        f"Root cause: {analysis.root_cause}\n"
        f"Fix: {analysis.fix_description}\n\nCurrent file:\n{current}"
    )
    response = await adapter.complete([Message(role="user", content=prompt)], system=FIX_SYSTEM, temperature=0.1)
    new_content = response.text.strip().removeprefix("```").removesuffix("```")

    fs.write(file_path, new_content)
    return FixAttempt(file_path=file_path, applied=True, new_content=new_content)
