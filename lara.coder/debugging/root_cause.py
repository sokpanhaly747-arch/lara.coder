"""Model-assisted root-cause analysis for errors the classifier can't
resolve mechanically. Given the error plus the offending file's
content, asks the model for a specific, falsifiable hypothesis rather
than a vague "there's a bug somewhere" summary."""
from __future__ import annotations

from dataclasses import dataclass

from debugging.error_classifier import ClassifiedError
from models.adapter import ModelAdapter
from models.inference import structured_complete

SYSTEM_PROMPT = """You are debugging a software error. Given the error and
the relevant file content, state the specific root cause (not just a
restatement of the error) and the minimal change needed to fix it.
Return JSON: {root_cause, fix_description, confidence (0-1)}"""


@dataclass
class RootCauseAnalysis:
    root_cause: str
    fix_description: str
    confidence: float


async def analyze(error: ClassifiedError, file_content: str, adapter: ModelAdapter) -> RootCauseAnalysis:
    prompt = f"Error ({error.category}): {error.message}\n\nFile ({error.file_hint}):\n{file_content}"
    return await structured_complete(
        adapter,
        prompt=prompt,
        parse=lambda d: RootCauseAnalysis(**d),
        system=SYSTEM_PROMPT,
    )
