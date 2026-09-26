# codegen/

Converts a TaskGraph + DesignSpec into real source files, written via
`runtime.filesystem_tool` (never directly to disk from here — keeps
codegen testable with a fake filesystem).

- `stacks.py` — supported tech stacks and their file conventions.
- `generator.py` — Task -> generated file(s), using the model for the
  actual code and the stack config for where files go.
- `templates/` — starter scaffolds per stack (e.g. Next.js skeleton).
