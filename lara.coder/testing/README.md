# testing/

Runs unit, e2e, visual, accessibility, security, and performance
checks and reports structured results back into the agent loop
(never just pass/fail — always with enough detail for `debugging/`
to act on).

- `unit_runner.py` — runs the project's own unit test suite.
- `e2e_runner.py` — drives `browser.controller` through key user flows.
- `visual_tester.py` — screenshot-based visual quality/regression checks.
