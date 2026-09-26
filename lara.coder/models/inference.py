"""Structured-output helpers built on top of a ModelAdapter.

Planning/design/codegen all need "give me back typed data", not just
prose. This wraps a provider call with: a strict system instruction to
emit JSON only, retry-with-repair on parse failure, and validation
against a caller-supplied schema function.
"""
from __future__ import annotations

import json
from typing import Callable, TypeVar

from models.adapter import Message, ModelAdapter

T = TypeVar("T")

JSON_ONLY_SYSTEM = (
    "Respond with ONLY valid JSON matching the requested shape. "
    "No prose, no markdown fences, no preamble."
)


async def structured_complete(
    adapter: ModelAdapter,
    prompt: str,
    parse: Callable[[dict], T],
    *,
    system: str | None = None,
    max_retries: int = 2,
) -> T:
    """Call the model and parse its JSON response into a typed object.

    `parse` should raise on invalid shape; on failure we ask the model
    to repair its own output rather than failing the whole pipeline.
    """
    sys_prompt = f"{JSON_ONLY_SYSTEM}\n\n{system or ''}".strip()
    messages = [Message(role="user", content=prompt)]

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        response = await adapter.complete(messages, system=sys_prompt, temperature=0.1)
        text = response.text.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
        try:
            data = json.loads(text)
            return parse(data)
        except Exception as exc:  # noqa: BLE001 - deliberately broad, we repair below
            last_error = exc
            messages.append(Message(role="assistant", content=response.text))
            messages.append(
                Message(
                    role="user",
                    content=f"That was not valid: {exc}. Return corrected JSON only.",
                )
            )
    raise ValueError(f"Model failed to produce valid structured output: {last_error}")
