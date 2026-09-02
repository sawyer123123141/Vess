"""Local browser preview for Vess' native 64x64 frame."""

from __future__ import annotations

import threading

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

from output.display import DisplayTarget
from state import State

_INDEX = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Vess</title>
  <style>
    body { background: #101114; color: #e7e7e7; font: 16px system-ui, sans-serif;
           margin: 2rem; text-align: center; }
    img { background: #000; image-rendering: pixelated; width: 512px; height: 512px; }
    section { margin-top: 1.5rem; }
    button, input { font: inherit; margin: 0 .3rem; }
    #debug { background: #181a1f; border: 1px solid #343842; margin: 0.6rem auto;
             max-width: 512px; padding: 0.8rem; text-align: left; white-space: pre-wrap; }
  </style>
</head>
<body>
  <img id="preview" alt="Live Vess face preview">
  <section>
    <label>Colour <input id="color" type="color" value="#64b4ff"></label>
    <button id="reset" type="button">Use mood colour</button>
  </section>
  <section>
    <h2>Diagnostics</h2>
    <pre id="debug">Connecting…</pre>
  </section>
  <script>
    const preview = document.querySelector('#preview');
    const color = document.querySelector('#color');
    const debug = document.querySelector('#debug');
    let previewUrl = '';

    async function poll() {
      try {
        const response = await fetch(`/frame.png?${Date.now()}`, { cache: 'no-store' });
        if (response.ok) {
          const nextUrl = URL.createObjectURL(await response.blob());
          const oldUrl = previewUrl;
          previewUrl = nextUrl;
          preview.src = nextUrl;
          if (oldUrl) URL.revokeObjectURL(oldUrl);
        }
      } finally {
        setTimeout(poll, 1000 / 30);
      }
    }

    async function pollDebug() {
      try {
        const response = await fetch('/debug', { cache: 'no-store' });
        if (response.ok) debug.textContent = formatDebug(await response.json());
      } finally {
        setTimeout(pollDebug, 500);
      }
    }

    function formatDebug(snapshot) {
      const runtime = Object.entries(snapshot.runtime)
          .map(([name, value]) => `${name}: ${value}`).join('  ');
      const values = Object.entries(snapshot.values)
          .map(([name, value]) => `${name}: ${value}`).join('\\n');
      const events = snapshot.events.slice(-8).reverse().map(event => {
        const { timestamp, event: name, ...details } = event;
        return `${new Date(timestamp * 1000).toLocaleTimeString()}  ${name} ${JSON.stringify(details)}`;
      }).join('\\n');
      return [runtime, values, events].filter(Boolean).join('\\n\\n');
    }

    color.addEventListener('change', async () => {
      const hex = color.value.slice(1);
      const channels = [0, 2, 4].map(i => parseInt(hex.slice(i, i + 2), 16));
      await fetch('/color', { method: 'PUT', headers: { 'content-type': 'application/json' },
                               body: JSON.stringify({ color: channels }) });
    });
    document.querySelector('#reset').addEventListener('click', async () => {
      await fetch('/color', { method: 'DELETE' });
    });
    poll();
    pollDebug();
  </script>
</body>
</html>
"""


class ColorRequest(BaseModel):
    color: list[int]


class WebPreview(DisplayTarget):
    """Keeps the latest frame available without slowing the render loop."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None

    def show(self, frame: np.ndarray) -> None:
        with self._lock:
            self._frame = frame.copy()

    def png(self) -> bytes | None:
        with self._lock:
            frame = None if self._frame is None else self._frame.copy()
        if frame is None:
            return None
        ok, encoded = cv2.imencode(
            ".png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError("cannot encode preview frame")
        return encoded.tobytes()


class WebServer:
    """Runs the local control UI without owning the render loop."""

    def __init__(
        self,
        state: State,
        port: int,
        event_log: object | None = None,
        command_registry: object | None = None,
    ) -> None:
        self.preview = WebPreview()
        self.command_registry = command_registry
        config = uvicorn.Config(
            create_app(
                state,
                self.preview,
                event_log,
                command_registry=command_registry,
            ),
            host="127.0.0.1",
            port=port,
            log_level="warning",
            access_log=False,
            ws="none",
        )
        self._server = uvicorn.Server(config)
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._server.run, name="web", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=2.0)


def create_app(
    state: State,
    preview: WebPreview,
    event_log: object | None = None,
    *,
    command_registry: object | None = None,
) -> FastAPI:
    """Build the local web app around the supplied state and frame target."""
    app = FastAPI()

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _INDEX

    @app.get("/frame.png")
    def frame() -> Response:
        encoded = preview.png()
        if encoded is None:
            raise HTTPException(status_code=503, detail="preview not ready")
        return Response(encoded, media_type="image/png")

    @app.get("/debug")
    def debug() -> dict[str, object]:
        return state.debug_snapshot()

    @app.get("/commands")
    def commands() -> dict[str, object]:
        if command_registry is None:
            raise HTTPException(status_code=503, detail="command registry unavailable")
        return command_registry.catalog()

    @app.post("/commands")
    def execute_command(payload: dict[str, object]) -> dict[str, object]:
        if command_registry is None:
            raise HTTPException(status_code=503, detail="command registry unavailable")
        call = command_registry.validate(payload)
        if call is None:
            raise HTTPException(status_code=422, detail="command is not allowed")
        try:
            result = command_registry.execute(call)
        except Exception as error:
            raise HTTPException(status_code=500, detail="command execution failed") from error
        if event_log is not None:
            event_log.append("command_executed", result.event_payload)
        state.record_debug(
            "command_executed",
            source="web",
            **result.event_payload,
        )
        return {
            "spoken_response": result.spoken_response,
            "command": result.event_payload,
        }

    @app.put("/color")
    def set_color(request: ColorRequest) -> dict[str, list[int]]:
        color = request.color
        if len(color) != 3 or any(channel < 0 or channel > 255 for channel in color):
            raise HTTPException(status_code=422, detail="color must be three values from 0 to 255")
        with state.locked():
            state.color = tuple(color)
        if event_log is not None:
            event_log.append("color_override_set", {"color": color})
        return {"color": color}

    @app.delete("/color")
    def reset_color() -> dict[str, None]:
        with state.locked():
            state.color = None
        if event_log is not None:
            event_log.append("color_override_cleared", {})
        return {"color": None}

    return app
