# debugging/

Turns a raw error into a verified fix:
ERROR -> CLASSIFICATION -> ROOT CAUSE -> FIX -> RE-TEST -> VERIFICATION.

Every retry here is bounded by `agent.config.LoopLimits` — this module
never loops on its own.

- `error_classifier.py` — categorizes raw errors (syntax, type, runtime, logic, config).
- `root_cause.py` — model-assisted root-cause analysis given error + surrounding code.
- `auto_fix.py` — proposes and applies a minimal patch, then hands back to testing.
