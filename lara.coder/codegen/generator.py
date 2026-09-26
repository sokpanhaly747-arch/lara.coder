"""Task -> generated source file(s).

Deliberately narrow interface: given one Task plus the surrounding
DesignSpec/ProductPlan context, produce a set of {path: content} pairs.
The orchestrator is responsible for calling `runtime.filesystem_tool`
to actually write them — this module never touches disk, which makes
it unit-testable by asserting on returned paths/content.
"""
from __future__ import annotations

from dataclasses import dataclass

from codegen.stacks import StackConfig
from models.adapter import ModelAdapter, Message
from planning.schemas import Task


@dataclass
class GeneratedFile:
    path: str
    content: str


CODEGEN_SYSTEM = """You are a senior {framework} engineer. Generate production
-quality {language} code for the given task using {styling} for styling.
Follow the existing project conventions. Output must be ONLY a JSON array
of {{"path": ..., "content": ...}} objects, no prose, no markdown fences."""


async def generate_for_task(
    task: Task,
    context: str,
    stack: StackConfig,
    adapter: ModelAdapter,
) -> list[GeneratedFile]:
    system = CODEGEN_SYSTEM.format(
        framework=stack.framework, language=stack.language, styling=stack.styling
    )
    prompt = f"Task: {task.title}\n{task.description}\n\nProject context:\n{context}"
    response = await adapter.complete(
        [Message(role="user", content=prompt)], system=system, temperature=0.1
    )

    import json

    text = response.text.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
    items = json.loads(text)
    return [GeneratedFile(path=i["path"], content=i["content"]) for i in items]
