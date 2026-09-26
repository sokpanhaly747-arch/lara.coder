# Architecture

## Guiding rules

1. **Model independence.** Nothing outside `models/` knows which LLM is
   in use. Everything else talks to `models.adapter.ModelAdapter`.
2. **Structured state, not chat scrollback.** The agent's working memory
   is a typed `AgentState` object (`agent/state.py`), persisted via
   `memory/`. Conversation text is an input/output channel, not the
   source of truth.
3. **Tools are explicit and typed.** Every capability the agent can
   invoke (write a file, run a shell command, open a browser tab) is a
   `Tool` with a schema, in `runtime/` or `browser/`, never raw
   `eval`/`exec` on generated code.
4. **Untrusted code never touches the host.** Generated application
   code only ever runs inside `runtime.sandbox.Sandbox`.
5. **Every loop has a limit.** The debug/fix/retest loop
   (`debugging/`) and the improve loop (`agent/loop.py`) are bounded by
   `agent/config.py::LoopLimits` — no unbounded autonomous retries.
6. **Every stage is measurable.** `evaluator/` defines pass/fail
   metrics per stage so "it works" is never just a vibe.

## Data flow

Summary:

```
User Request -> Requirements -> Product Plan -> Task Graph -> UX/UI Design
  -> Agent Orchestrator -> Code Generation -> Runtime -> Browser -> Testing
  -> Debugging -> Re-test -> Evaluation -> Memory -> Final Product
```

In detail:

```
User idea (text)
   |
   v
planning.requirement_analyzer   -> Requirements (schemas.py)
   |
   v
planning.product_planner        -> ProductPlan (pages, entities, flows)
   |
   v
design.ux_planner / ui_planner  -> DesignSpec (screens, components)
   |
   v
planning.task_planner           -> TaskGraph (ordered, dependency-aware)
   |
   v
agent.orchestrator  <----------------------------+
   |  picks next task                            |
   v                                              |
codegen.generator     -> writes files via         |
   |                     runtime.filesystem_tool   |
   v                                              |
runtime.process_manager -> installs deps, runs app |
   |                                              |
   v                                              |
browser.controller     -> loads app, screenshots   |
   |                                              |
   v                                              |
testing.*_runner       -> unit/e2e/visual results  |
   |                                              |
   v                                              |
debugging.error_classifier -> root_cause -> auto_fix
   |  (bounded retries, re-enters loop) ----------+
   v
evaluator.metrics -> QualityReport
   |
   v
Final product (or next improvement iteration)
```

`memory.project_memory` is written to at every arrow above so the loop
can resume after a crash or a new session.

## Module dependency direction

`agent` depends on `planning`, `design`, `codegen`, `runtime`,
`browser`, `testing`, `debugging`, `evaluator`, `memory`, `models`.
None of those modules depend back on `agent` — they are libraries the
orchestrator composes, which keeps them independently testable.

`app` depends on `agent` (as a library) and exposes it over HTTP/WS.
`training` and `dataset` are offline concerns that only depend on
`models` (to know the target interface) and are not imported by the
runtime agent path at all.

## Subfolder convention

Every top-level module is required to expose a specific set of
subfolders (see the repository tree in `docs/BUILD_PLAN.md`'s audit
output, or run `find <module> -maxdepth 1 -type d`). Where a subfolder
duplicates something a top-level `.py` file in that module already
does (e.g. `runtime/sandbox/` alongside `runtime/sandbox.py`), the
`.py` file remains the real, imported implementation, and the
subfolder is either (a) a place to store *data/config* for that
capability, or (b) a documented extension point for a capability that
doesn't exist yet — never a second, competing implementation. Each
such subfolder's `README.md` says which of the two it is.
