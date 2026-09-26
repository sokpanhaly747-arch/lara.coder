# Security

## Threat model

This project executes **AI-generated, untrusted code** as a core part
of its function (the RUN/TEST/DEBUG loop). The primary security goal
is containing that code, not just protecting this repository's own
source.

## Controls in place today

- **Sandboxed filesystem.** `runtime.sandbox.Sandbox` resolves every
  path and rejects anything that escapes the sandbox root
  (`runtime/sandbox.py`, enforced in `tests/test_agent_loop.py`).
- **Bounded command execution.** `runtime.terminal_tool.TerminalTool`
  enforces a timeout and an output-size cap on every shell command, so
  a runaway or malicious generated command cannot hang the process or
  exhaust memory via stdout.
- **No secrets in code or config.** API keys are read from environment
  variables only (see `.env.example`); `infrastructure/config/settings.yaml`
  and `infrastructure/configs/` intentionally hold no secrets.
- **Loop limits.** `agent.config.LoopLimits` bounds every autonomous
  retry path (build iterations, fix attempts, improve iterations) so a
  misbehaving generation can't spin forever.

## Controls planned, not yet implemented

- **Container-level isolation** for the sandbox (today: a jailed local
  temp directory + subprocess; `infrastructure/docker/Dockerfile`
  defines the target container image for production hardening, but the
  runtime does not yet actually launch generated code inside it).
- **Network isolation** for sandboxed processes (`infrastructure/security/`
  is a placeholder for this policy).
- **Dependency/security scanning** of generated code
  (`testing/security/` is a placeholder — not implemented).
- **Secrets scanning in CI** (`infrastructure/ci_cd/` is a placeholder —
  no CI pipeline exists yet).

## Reporting a vulnerability

This is a development-stage project scaffold; if you find a security
issue, open an issue describing it without including exploit details
in a public thread — request a private channel first.
