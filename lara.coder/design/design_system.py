"""Shared design tokens applied consistently across generated screens.

Kept rule-based with sensible defaults rather than always calling the
model, so every generated product starts from a coherent, accessible
baseline (contrast-safe palette, consistent spacing scale) that the
model then specializes per product.
"""
from __future__ import annotations

from design.schemas import DesignTokens

DEFAULT_TOKENS = DesignTokens(
    color_palette={
        "background": "#0B0F14",
        "surface": "#131A22",
        "primary": "#4F7CFF",
        "text": "#E6EDF3",
        "text-muted": "#8B97A6",
        "border": "#22303C",
        "success": "#3FB68B",
        "danger": "#E5534B",
    },
    font_family="Inter, system-ui, sans-serif",
    spacing_scale=[4, 8, 12, 16, 24, 32, 48, 64],
    radius_scale=[4, 8, 12, 16],
)


def tokens_for_product(product_summary: str) -> DesignTokens:
    """Placeholder for product-specific theming; returns the safe default
    baseline today. A later layer can specialize palette/tone by product
    category (e.g. a clothing shop vs. a fintech dashboard)."""
    return DEFAULT_TOKENS
