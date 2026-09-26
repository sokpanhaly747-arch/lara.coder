"""FastAPI service exposing the agent as a chat + build API.

Layer-0: a minimal skeleton wiring a WebSocket per project run to
`agent.loop.AgentLoop`, so the frontend can stream phase/task updates
as they happen instead of polling.
"""
from __future__ import annotations

from fastapi import FastAPI, WebSocket

app = FastAPI(title="Lara Coder API")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.websocket("/ws/projects/{project_id}")
async def project_socket(websocket: WebSocket, project_id: str) -> None:
    """Streams AgentState updates for a running build.

    Layer-0 placeholder: accepts the connection and echoes a stub
    event. Wiring this to a real `AgentLoop.run(...)` with
    `on_step` pushing JSON frames is the next implementation layer.
    """
    await websocket.accept()
    await websocket.send_json({"project_id": project_id, "phase": "understand", "note": "stub"})
    await websocket.close()
