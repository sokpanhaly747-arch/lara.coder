# Contributing

## Ground rules

- **Model independence.** Never import a provider SDK (e.g. `anthropic`)
  outside `models/`. Everything else depends on `models.adapter.ModelAdapter`.
- **No untrusted code on the host.** Anything that runs generated
  application code goes through `runtime.sandbox.Sandbox`.
- **Every autonomous loop is bounded.** New retry logic must read its
  limit from `agent.config.LoopLimits`, not a local constant.
- **Don't fake a capability.** If something isn't implemented, raise
  `NotImplementedError` with a note on what real implementation should
  do, rather than returning a plausible-looking fake result. See
  `training/sft/train.py` for the pattern.
- **Every module stays independently testable.** `agent/` composes
  `planning/`, `design/`, `codegen/`, `runtime/`, `browser/`,
  `testing/`, `debugging/`, `evaluator/`, `memory/`, `models/` — none
  of those should import back from `agent/`.

## Adding a required-but-empty subfolder's real implementation

Many subfolders under each module currently hold only a `README.md`
explaining what belongs there and pointing at the sibling file that
implements the current, simpler version (e.g. `runtime/sandbox/README.md`
points at `runtime/sandbox.py`). When implementing one:

1. Read that README for context on why the folder exists and what it's
   for.
2. Add real code/config there.
3. Update the README to describe the real implementation instead of
   the placeholder text, and update `docs/BUILD_PLAN.md`'s status for
   that piece.

## Running tests

```bash
pip install -e ".[dev]"
pytest
```

The test suite (`tests/test_agent_loop.py`) exercises the orchestrator,
loop limits, task-graph dependency resolution, error classification,
and sandbox path-jailing with fakes/no live model calls — it should
stay runnable with zero API keys and zero network access.

## Adding training data

Add examples under `dataset/curated/` (or `raw/` -> `cleaned/` ->
`curated/` if going through a cleaning pipeline) matching
`dataset/schema/example_schema.json`, validated via
`dataset.loaders.dataset_loader.load_examples`. Do not commit
synthetic examples to `dataset/curated/` directly — put them in
`dataset/synthetic/` so provenance stays clear.
