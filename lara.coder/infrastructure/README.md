# infrastructure/

Sandboxing, containers, and deployment/runtime configuration —
separate from application code so environment concerns don't leak
into `runtime/`'s Python interfaces.

- `docker/` — container image(s) for isolated code execution (production
  hardening of `runtime.sandbox.Sandbox`).
- `config/` — environment/settings loading (API keys, resource limits).
