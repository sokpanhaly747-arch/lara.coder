"""Typed shapes produced/consumed across the planning stage."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Ambiguity:
    question: str
    default_assumption: str


@dataclass
class Requirements:
    summary: str
    goals: list[str]
    target_users: list[str]
    must_have_features: list[str]
    nice_to_have_features: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    ambiguities: list[Ambiguity] = field(default_factory=list)


@dataclass
class Entity:
    name: str
    fields: dict[str, str]  # field_name -> type
    relations: list[str] = field(default_factory=list)


@dataclass
class Page:
    name: str
    route: str
    purpose: str
    key_components: list[str] = field(default_factory=list)


@dataclass
class ProductPlan:
    pages: list[Page]
    entities: list[Entity]
    api_endpoints: list[str]
    needs_auth: bool
    user_flows: list[str] = field(default_factory=list)
    acceptance_criteria: list[str] = field(default_factory=list)


@dataclass
class Task:
    id: str
    title: str
    description: str
    depends_on: list[str] = field(default_factory=list)
    status: str = "pending"  # pending | in_progress | done | failed


@dataclass
class TaskGraph:
    tasks: list[Task]

    def ready_tasks(self) -> list[Task]:
        done_ids = {t.id for t in self.tasks if t.status == "done"}
        return [
            t for t in self.tasks
            if t.status == "pending" and all(dep in done_ids for dep in t.depends_on)
        ]
