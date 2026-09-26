# Isolated execution image for generated application code.
# Production hardening of runtime.sandbox.Sandbox: no host network beyond
# what's explicitly allowed, non-root user, resource limits set by the
# container runtime (see docs on `--memory`/`--cpus` at deploy time).
FROM node:20-slim

RUN useradd --create-home --uid 10001 sandboxuser
USER sandboxuser
WORKDIR /workspace

# Generated projects install their own deps at runtime via
# runtime.terminal_tool; this image just provides a safe, minimal base.
