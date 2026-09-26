"""Screens + UX plan -> concrete component breakdown and layouts.

Every screen is required to define loading/empty/error states up
front (see ComponentSpec.states default) so codegen never has to
retrofit them after the fact.
"""
from __future__ import annotations

from models.adapter import ModelAdapter
from models.inference import structured_complete
from design.schemas import ComponentSpec, DesignSpec, ScreenSpec
from design.design_system import tokens_for_product
from planning.schemas import ProductPlan

SYSTEM_PROMPT = """You are a UI designer. For each page in the product plan,
choose a layout pattern and list the components it needs (reusing
components across screens where sensible). Every screen must account
for loading, empty, and error states. Return JSON:
{screens[{name, route, layout, components[], responsive_notes,
accessibility_notes}], components[{name, purpose, states[], reused_on[]}]}"""


async def plan_ui(product_plan: ProductPlan, adapter: ModelAdapter) -> DesignSpec:
    prompt = f"Pages: {[(p.name, p.route, p.purpose) for p in product_plan.pages]}"
    raw = await structured_complete(adapter, prompt=prompt, parse=lambda d: d, system=SYSTEM_PROMPT)

    return DesignSpec(
        tokens=tokens_for_product(product_plan.pages[0].purpose if product_plan.pages else ""),
        screens=[ScreenSpec(**s) for s in raw.get("screens", [])],
        components=[ComponentSpec(**c) for c in raw.get("components", [])],
        navigation=raw.get("navigation", []),
    )
