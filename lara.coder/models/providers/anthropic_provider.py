"""Anthropic implementation of ModelAdapter.

This is the reference provider. It deliberately has no special
privileges over any other provider — agent/planning/etc. code never
imports this module directly, only `models.router`.
"""
from __future__ import annotations

from typing import AsyncIterator

from models.adapter import (
    Message,
    ModelAdapter,
    ModelResponse,
    ToolCall,
    ToolSpec,
)


class AnthropicAdapter(ModelAdapter):
    name = "anthropic"

    def __init__(self, model: str = "claude-sonnet-4-6", api_key: str | None = None):
        self.model = model
        self._api_key = api_key  # resolved from env if None; see infrastructure/config

    def _client(self):
        import anthropic  # imported lazily so the package is optional until used

        return anthropic.AsyncAnthropic(api_key=self._api_key)

    @staticmethod
    def _to_anthropic_messages(messages: list[Message]) -> list[dict]:
        return [
            {"role": m.role if m.role != "tool" else "user", "content": m.content}
            for m in messages
            if m.role != "system"
        ]

    async def complete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.2,
    ) -> ModelResponse:
        client = self._client()
        kwargs: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": self._to_anthropic_messages(messages),
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.parameters}
                for t in tools
            ]

        response = await client.messages.create(**kwargs)

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name, arguments=block.input))

        return ModelResponse(
            text="".join(text_parts),
            tool_calls=tool_calls,
            stop_reason=response.stop_reason or "end_turn",
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
            raw=response,
        )

    async def stream(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.2,
    ) -> AsyncIterator[str]:
        client = self._client()
        kwargs: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": self._to_anthropic_messages(messages),
        }
        if system:
            kwargs["system"] = system

        async with client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield text
