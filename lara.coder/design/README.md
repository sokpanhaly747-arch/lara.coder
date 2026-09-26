# design/

Converts product requirements into concrete UI/UX specifications
before any code is generated. A working interface is not automatically
a good one — this module is what keeps that distinction real.

- `schemas.py` — `DesignSpec`, `Screen`, `Component`, `DesignSystem`.
- `ux_planner.py` — user flows -> screen list + navigation.
- `ui_planner.py` — screens -> component breakdown, layout, states.
- `design_system.py` — shared tokens (color, type, spacing) + reusable components.
