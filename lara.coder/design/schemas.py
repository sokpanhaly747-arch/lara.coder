"""Typed shapes for the design stage."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DesignTokens:
    color_palette: dict[str, str]
    font_family: str
    spacing_scale: list[int]
    radius_scale: list[int]


@dataclass
class ComponentSpec:
    name: str
    purpose: str
    states: list[str] = field(default_factory=lambda: ["default", "loading", "empty", "error"])
    reused_on: list[str] = field(default_factory=list)


@dataclass
class ScreenSpec:
    name: str
    route: str
    layout: str  # e.g. "sidebar+content", "centered-form", "grid"
    components: list[str]
    responsive_notes: str = ""
    accessibility_notes: str = ""


@dataclass
class DesignSpec:
    tokens: DesignTokens
    screens: list[ScreenSpec]
    components: list[ComponentSpec]
    navigation: list[str] = field(default_factory=list)
