"""Typed API over MemoryStore for project-level knowledge.

Covers what the system prompt calls out explicitly: project
requirements, architecture decisions, codebase knowledge, and
conversation context — each addressable and queryable independently,
not one giant blob.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from agent.state import AgentState
from memory.store import MemoryStore


@dataclass
class Decision:
    timestamp: str
    summary: str
    rationale: str


class ProjectMemory:
    def __init__(self, store: MemoryStore, project_id: str):
        self.store = store
        self.project_id = project_id

    def _key(self, name: str) -> str:
        return f"{self.project_id}/{name}"

    def save_state(self, state: AgentState) -> None:
        self.store.set(self._key("agent_state"), asdict(state))

    def load_state(self) -> dict | None:
        return self.store.get(self._key("agent_state"))

    def record_decision(self, summary: str, rationale: str) -> None:
        decisions = self.store.get(self._key("decisions")) or []
        decisions.append(asdict(Decision(
            timestamp=datetime.now(timezone.utc).isoformat(),
            summary=summary,
            rationale=rationale,
        )))
        self.store.set(self._key("decisions"), decisions)

    def get_decisions(self) -> list[dict]:
        return self.store.get(self._key("decisions")) or []

    def remember_codebase_fact(self, key: str, fact: str) -> None:
        facts = self.store.get(self._key("codebase_facts")) or {}
        facts[key] = fact
        self.store.set(self._key("codebase_facts"), facts)

    def get_codebase_facts(self) -> dict:
        return self.store.get(self._key("codebase_facts")) or {}
