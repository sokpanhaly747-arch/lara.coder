# runtime/

Controlled interfaces between the agent and an isolated execution
environment. **Untrusted generated code never touches the host
machine directly** — everything routes through `Sandbox`.

- `sandbox.py` — sandbox lifecycle (create/destroy isolated workdir/container).
- `filesystem_tool.py` — typed read/write/list, path-jailed to the sandbox.
- `terminal_tool.py` — typed command execution with timeouts and output caps.
- `process_manager.py` — start/stop/health-check long-running dev servers.
