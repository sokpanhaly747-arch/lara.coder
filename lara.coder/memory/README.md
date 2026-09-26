# memory/

Structured, controllable persistence for project knowledge — distinct
from `agent.state.AgentState` (the live in-run object): this is what
survives across sessions/restarts.

- `store.py` — pluggable key-value/document backend (starts as local
  JSON files, swappable for a real DB later).
- `project_memory.py` — typed API over the store: save/load AgentState,
  record architecture decisions, query past project knowledge.
