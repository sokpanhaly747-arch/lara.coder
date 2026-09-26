# Lara Coder

## What this is

An AI system that turns a natural-language product idea into a
working, tested software product. It is **not a chatbot that prints
code** — it behaves like a software product team: it understands the
request, plans requirements, designs the UI/UX, decides architecture,
generates code, creates/edits files, runs the app, tests it, inspects
the browser, finds and fixes errors, re-tests, reviews quality, and
iterates until the product meets a quality bar.

## Core purpose

```
USER IDEA -> UNDERSTAND -> PLAN -> DESIGN -> CODE -> RUN -> TEST -> DEBUG
          -> FIX -> REVIEW -> IMPROVE -> FINAL PRODUCT
```

The model is the intelligence. The agent is the decision-making
system. Tools are its ability to act on a dev environment. The runtime
is where generated software executes. The browser is how it sees what
it built. The evaluator is quality control. Memory keeps project
context across the whole run. See `ARCHITECTURE.md` for the full data
flow and dependency rules between modules.

## How the system works (development lifecycle)

1. **Understand** — `planning.requirement_analyzer` turns the raw idea
   into structured `Requirements`, surfacing ambiguities with stated
   default assumptions rather than guessing silently.
2. **Plan** — `planning.product_planner` produces a `ProductPlan`
   (pages, data entities, API surface, auth need, user flows); then
   `planning.task_planner` turns that into a dependency-ordered
   `TaskGraph`.
3. **Design** — `design.ux_planner` and `design.ui_planner` turn pages
   into navigation, screens, components, and design tokens
   (`design.design_system`), always accounting for loading/empty/error
   states.
4. **Code** — `codegen.generator` turns one `Task` at a time into
   files, written through `runtime.filesystem_tool` (sandboxed, path-
   jailed — see `runtime.sandbox`).
5. **Run** — `runtime.process_manager` starts the generated app's dev
   server inside the sandbox.
6. **Test** — `testing.unit_runner` and `testing.e2e_runner` (driving
   `browser.controller`) check the result; `browser.dom_inspector` and
   `testing.visual_tester` catch what reading source code can't.
7. **Debug** — `debugging.error_classifier` -> `debugging.root_cause`
   -> `debugging.auto_fix` -> back to Test, bounded by
   `agent.config.LoopLimits` so this never loops forever.
8. **Review / Improve** — `evaluator.metrics.evaluate_run` scores the
   result objectively; if it passes the bar the loop can still run a
   bounded number of IMPROVE iterations before finishing.

All of the above is coordinated by `agent.orchestrator.next_phase`
(pure decision logic) and executed by `agent.loop.AgentLoop`, over a
single typed `agent.state.AgentState` — not raw chat text — persisted
via `memory.project_memory`.

## Module map

Each top-level module has its own `README.md` with more detail.
Subfolders that don't yet have real code contain a `README.md`
explaining what belongs there and, where relevant, which sibling file
holds the current (simpler) implementation.

| Module | Responsibility |
|---|---|
| `dataset/` | Training examples for coding, UI/UX, debugging, agent traces, preference data — raw through curated |
| `training/` | SFT / QLoRA / DPO / distillation experiments and supporting scripts |
| `models/` | Model-agnostic adapter + router + inference layer (base models, adapters, embeddings) |
| `agent/` | Orchestration loop, structured agent state, and per-role glue (planner/coder/researcher/reviewer/debugger/optimizer) |
| `planning/` | Natural language -> requirements -> product plan -> task graph |
| `design/` | Requirements -> UX flows, UI specs, design system, components, accessibility |
| `codegen/` | Plans + design specs -> real source files (frontend/backend/database/api/auth) |
| `runtime/` | Sandboxed filesystem/terminal/process/package execution |
| `browser/` | Preview, navigate, interact, screenshot, inspect the running app |
| `testing/` | Unit / integration / e2e / visual / accessibility / security / performance tests |
| `debugging/` | Error parsing -> root cause -> fix generation -> regression check -> recovery |
| `evaluator/` | Objective benchmarks and quality scoring for every stage above |
| `memory/` | Structured, controllable project / conversation / codebase / decision memory |
| `app/` | User-facing chat + file explorer + editor + live preview + dashboard |
| `infrastructure/` | Sandboxing, containers, config, logging, monitoring, security, deployment, CI/CD |

## Current implementation status

**Real, tested implementation today:**
- `agent/` — `AgentState`, `AgentConfig`/`LoopLimits`, the pure
  `next_phase` orchestrator policy, and `AgentLoop.run`, exercised end
  to end in `tests/test_agent_loop.py` with fake phase handlers (no
  live model/network needed).
- `planning/schemas.py`'s `TaskGraph.ready_tasks()` dependency
  resolution.
- `debugging/error_classifier.py`'s rule-based `classify()`.
- `runtime/sandbox.py`'s path-jailing (`Sandbox.resolve` rejects path
  escapes).
- `models/adapter.py`, `models/router.py`, `models/inference.py`, and
  the `models/providers/anthropic_provider.py` implementation — real
  interfaces, callable today given an API key.
- `runtime/filesystem_tool.py`, `runtime/terminal_tool.py`,
  `runtime/process_manager.py` — real, sandboxed, with timeouts and
  output caps.

**Typed interfaces / extension points (not yet exercised end to end):**
`planning/`, `design/`, `codegen/generator.py`, `debugging/root_cause.py`
and `auto_fix.py`, `browser/`, `testing/*_runner.py`,
`evaluator/metrics.py`, `memory/` — all have real, typed, importable
code, but need a live model call and/or a real generated project to
exercise. See `docs/BUILD_PLAN.md` for the wiring order.

**Explicit future work (deliberately not faked):**
`training/sft/train.py` and `training/dpo/train.py` raise
`NotImplementedError` rather than pretending to train something. Most
subfolders added in the latest structure audit (e.g.
`testing/security/`, `debugging/regression/`, `app/editor/`) are
placeholders — see each one's `README.md`.

## How to run tests

```bash
pip install -e ".[dev]"
pytest
```

The suite needs no API key and no network access — it tests
orchestration logic, dependency resolution, error classification, and
sandbox containment with fakes.

## Contributing / security

See `CONTRIBUTING.md` for conventions (model independence, sandboxing,
bounded loops, no faked capabilities) and `SECURITY.md` for the threat
model and current vs. planned containment controls.
