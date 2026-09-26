"""ProductPlan -> screen list + navigation structure (the UX layer).

Focuses on flow and information architecture before any visual design:
which screens exist, how a user moves between them, and what each
screen's job is — independent of exact components or styling.
"""
from __future__ import annotations

from models.adapter import ModelAdapter
from models.inference import structured_complete
from planning.schemas import ProductPlan

SYSTEM_PROMPT = """You are a UX designer. Given a product plan, define the
navigation structure and, for each page, its primary user goal and how
a user typically arrives at / leaves it. Return JSON:
{navigation[], screen_goals: {route: goal}}"""


async def plan_ux(product_plan: ProductPlan, adapter: ModelAdapter) -> dict:
    prompt = f"Product plan pages: {[p.route for p in product_plan.pages]}\nFlows: {product_plan.user_flows}"
    return await structured_complete(adapter, prompt=prompt, parse=lambda d: d, system=SYSTEM_PROMPT)
