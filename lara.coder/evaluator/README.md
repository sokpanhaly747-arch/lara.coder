# evaluator/

Objective, numeric measurement of every stage — the anti-"vibes"
module. `metrics.py` defines the scoring functions; `benchmarks/`
holds fixed task sets to run them against over time (e.g. across model
or prompt changes).

- `metrics.py` — scoring functions per dimension (build success, test
  pass rate, UI quality heuristics, debugging success rate, etc).
- `benchmarks/` — versioned sets of example product requests with
  expected acceptance criteria, for regression testing the whole system.
