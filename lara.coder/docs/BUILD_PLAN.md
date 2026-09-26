# Layered build plan

Matches the system spec's "do not pretend everything is implemented"
principle. Each layer should be independently testable before starting
the next.

1. **Architecture, interfaces, schemas, config** — DONE (this scaffold).
   `agent/state.py`, `planning/schemas.py`, `design/schemas.py`,
   `agent/config.py`.
2. **Model integration** — DONE at interface level (`models/`), needs
   an `ANTHROPIC_API_KEY` to actually call out.
3. **Core agent loop** — DONE at interface level (`agent/loop.py`,
   `agent/orchestrator.py`); phase handlers in `tests/` are fakes today,
   need to be wired to the real planning/design/codegen/runtime modules.
4. **Tools (filesystem/terminal/process)** — DONE (`runtime/`).
5. **Runtime sandboxing** — layer-0 local-dir jail done
   (`runtime/sandbox.py`); container isolation via
   `infrastructure/docker/Dockerfile` is the production hardening step.
6. **Testing (unit/e2e)** — interfaces done (`testing/`); needs a real
   generated project to run against.
7. **Debugging** — interfaces + rule-based classifier done
   (`debugging/`); root-cause/auto-fix need a live model call to
   exercise end to end.
8. **Browser integration** — interfaces done (`browser/`), needs
   `playwright install chromium` and a running dev server to exercise.
9. **Evaluation** — scoring function done (`evaluator/metrics.py`) plus
   one example benchmark; needs the full loop wired up to actually run.
10. **User interface** — `app/backend` has a stub WebSocket endpoint;
    `app/frontend` is not yet scaffolded.
11. **Dataset + training** — schema, loader, and config done; the
    actual training loops (`training/sft/train.py`,
    `training/dpo/train.py`) are explicit `NotImplementedError` stubs,
    not faked implementations, since they need real training
    infrastructure to be meaningful.

## What to do next

The highest-leverage next step is wiring `agent/loop.py`'s phase
handlers to the real modules (today only tested with fakes in
`tests/test_agent_loop.py`) and running the `clothing-shop-basic`
benchmark end to end against a real Anthropic API key.

## Structure audit (repair pass)

A later pass brought every module up to the full required subfolder
list (94 required subfolders across the 15 top-level modules; see
`ARCHITECTURE.md`'s "Subfolder convention" section for what each one
is for). No existing folder was renamed or removed, and no existing
`.py` implementation was replaced — each new subfolder holds either a
config/data location for an existing implementation, or an honestly
labeled future-work placeholder. One naming note: the architecture
spec calls for `infrastructure/configs/` (plural); the original
scaffold already had a working `infrastructure/config/settings.yaml`
(singular). Both now exist — `config/` keeps the real settings file,
`configs/` is the new environment-overlay location — rather than
renaming the working file and risking breaking anything that reads
it.
