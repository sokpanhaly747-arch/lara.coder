# agent/

The decision-making core. Coordinates planning, design, codegen,
runtime, browser, testing, debugging, and memory to turn a user idea
into a working product — bounded by explicit loop limits.

- `state.py` — typed `AgentState`, the single source of truth (not chat text).
- `config.py` — `LoopLimits` and other run configuration.
- `orchestrator.py` — decides the next action given current state.
- `loop.py` — the outer run loop: orchestrate -> act -> observe -> repeat.
