# planning/

Converts a natural-language idea into structured, actionable data.

- `schemas.py` — `Requirements`, `ProductPlan`, `TaskGraph` dataclasses.
- `requirement_analyzer.py` — idea -> Requirements, flags ambiguities.
- `product_planner.py` — Requirements -> ProductPlan (pages, entities, flows).
- `task_planner.py` — ProductPlan + DesignSpec -> ordered, dependency-aware TaskGraph.
