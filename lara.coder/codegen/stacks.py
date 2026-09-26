"""Configurable tech stack definitions.

The system prompt says the exact stack should stay configurable; this
is the seam. Adding a new stack means adding an entry here, not
touching agent/codegen logic.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StackConfig:
    id: str
    language: str
    framework: str
    styling: str
    test_framework: str
    run_command: str
    install_command: str
    file_extension: str


NEXTJS_TS_TAILWIND = StackConfig(
    id="nextjs-typescript-tailwind",
    language="typescript",
    framework="next.js",
    styling="tailwindcss",
    test_framework="vitest",
    run_command="npm run dev",
    install_command="npm install",
    file_extension=".tsx",
)

STACKS: dict[str, StackConfig] = {
    NEXTJS_TS_TAILWIND.id: NEXTJS_TS_TAILWIND,
}


def get_stack(stack_id: str) -> StackConfig:
    try:
        return STACKS[stack_id]
    except KeyError as exc:
        raise ValueError(f"Unknown stack '{stack_id}'. Available: {list(STACKS)}") from exc
