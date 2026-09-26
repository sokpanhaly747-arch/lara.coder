# dataset/

Training examples for improving the underlying model(s): software
development, frontend/backend, UI/UX, design-to-code, debugging, tool
use, agent workflows, self-correction, code review, requirements, and
architecture decisions.

- `schema/` — JSON Schema each example type must validate against.
- `loaders/` — load/validate/split example sets for `training/`.
- `examples/` — the actual example files (empty at layer-0; populated
  as real training data is curated).

New categories should be addable by adding a schema + loader, without
restructuring this folder.
