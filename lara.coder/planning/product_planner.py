"""Requirements -> ProductPlan: pages, data entities, API surface, flows."""
from __future__ import annotations

from models.adapter import ModelAdapter
from models.inference import structured_complete
from planning.schemas import Entity, Page, ProductPlan, Requirements

SYSTEM_PROMPT = """You are a product architect. Given structured requirements,
produce a concrete product plan: pages/routes, data entities with fields
and relations, required API endpoints, whether authentication is needed,
key user flows, and acceptance criteria. Return JSON matching the
ProductPlan shape: {pages[{name, route, purpose, key_components[]}],
entities[{name, fields{}, relations[]}], api_endpoints[], needs_auth,
user_flows[], acceptance_criteria[]}"""


def _parse(data: dict) -> ProductPlan:
    return ProductPlan(
        pages=[Page(**p) for p in data.get("pages", [])],
        entities=[Entity(**e) for e in data.get("entities", [])],
        api_endpoints=data.get("api_endpoints", []),
        needs_auth=bool(data.get("needs_auth", False)),
        user_flows=data.get("user_flows", []),
        acceptance_criteria=data.get("acceptance_criteria", []),
    )


async def plan(requirements: Requirements, adapter: ModelAdapter) -> ProductPlan:
    prompt = f"Requirements:\n{requirements}"
    return await structured_complete(adapter, prompt=prompt, parse=_parse, system=SYSTEM_PROMPT)
