# app/

The user-facing application: chat interface, project file tree, code
editor, live preview, agent activity log, and test results dashboard.

- `backend/` — FastAPI service exposing `agent.loop.AgentLoop` over
  HTTP/WebSocket (chat messages in, streamed agent activity + files out).
- `frontend/` — Next.js UI (chat pane, file explorer, editor, preview
  iframe, activity/log panel). Scaffolded in a later layer.
