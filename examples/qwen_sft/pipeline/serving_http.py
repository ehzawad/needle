"""The tangible serving microservice boundary — a stdlib HTTP front door.

Real-world analog: a KServe / Triton inference endpoint (or an Envoy-fronted model
service) exposing the gate/respond contract over HTTP. It is deliberately the ONE
network surface in this otherwise file-backed modular monolith. For a mock/CI release the
injected gate/serve callables are model-free and this process stays CPU-only; for a real
release the SAME process loads the promoted model in-process after ``cli._cmd_serve`` pins
the A5000 and runs the device preflight (this HTTP module itself never imports torch — the
model lives behind the injected callables). Both ``/gate`` and ``/respond`` share one
thread-safe :class:`~pipeline.serving.TurnRecorder`, so turns are per-session and
monotonic across the two routes and every served turn is logged exactly once.

Routes (JSON in/out). Each POST accepts an optional ``session_id``; when absent the server
generates one and echoes it (plus the assigned ``turn``) so a client can keep a
conversation together. Anonymous requests are never merged into one global session.

  * ``POST /gate``     ``{"query": ..., "session_id"?: ...}`` -> the policy-wrapped gate
    decision (logged as a gate-only turn; never runs generation)
  * ``POST /respond``  ``{"query": ..., "session_id"?: ...}`` -> the cross-checked
    :class:`TurnResult`
  * ``GET  /healthz``  -> liveness + served artifact id + circuit state + device report
"""
from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Mapping
from dataclasses import asdict, is_dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from pipeline.serving import TurnRecorder

__all__ = ["build_http_server", "serve_forever", "parse_host_port"]

_MAX_BODY = 1 << 20  # 1 MiB request cap — this is a demo endpoint, not a public API.


def parse_host_port(value: str, *, default_host: str = "127.0.0.1") -> tuple[str, int]:
    """Parse ``HOST:PORT`` (or bare ``PORT``) into ``(host, port)``."""
    text = str(value).strip()
    if ":" in text:
        host, _, port = text.rpartition(":")
        host = host or default_host
    else:
        host, port = default_host, text
    try:
        return host, int(port)
    except ValueError as exc:
        raise ValueError(f"serving_http: invalid HOST:PORT {value!r}") from exc


def _as_json(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Mapping):
        return {str(k): _as_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_json(v) for v in value]
    return value


_BAD_LENGTH = object()  # sentinel: non-numeric Content-Length -> HTTP 400


def build_http_server(
    host: str,
    port: int,
    *,
    serve: Callable[..., Any],
    gate: Callable[..., Any],
    recorder: TurnRecorder,
    artifact_id: str | None,
    healthz: Callable[[], Mapping[str, Any]] | None = None,
) -> ThreadingHTTPServer:
    """Build (but do not start) a :class:`ThreadingHTTPServer` over ``serve``/``gate``.

    ``serve`` is the session-aware frozen 7-step server; ``gate`` is the logged,
    policy-wrapped safe gate (what ``/gate`` returns) — both accept ``(query, session_id,
    turn)``. ``recorder`` is the shared :class:`~pipeline.serving.TurnRecorder` that assigns
    the per-session, monotonic turn number BEFORE dispatch, so ``/gate`` and ``/respond``
    interleave into one sequence. ``healthz`` supplies the liveness payload (including the
    device report). ``/respond`` is logged exactly once (by ``serve``); this HTTP layer adds
    no second log.
    """

    def _health() -> Mapping[str, Any]:
        if healthz is not None:
            return healthz()
        return {"ok": True, "artifact_id": artifact_id}

    class _Handler(BaseHTTPRequestHandler):
        server_version = "ScopeServe/1.0"
        protocol_version = "HTTP/1.1"

        def log_message(self, *args: Any) -> None:  # keep the endpoint quiet by default
            return

        def _send(self, code: int, body: Mapping[str, Any]) -> None:
            data = (json.dumps(_as_json(body), ensure_ascii=False) + "\n").encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _read_request(self) -> Any:
            """Return ``(query, session_id | None)``, ``None`` (bad/empty body), or
            ``_BAD_LENGTH`` (non-numeric Content-Length)."""
            raw_len = self.headers.get("Content-Length")
            try:
                length = int(raw_len) if raw_len not in (None, "") else 0
            except (TypeError, ValueError):
                return _BAD_LENGTH
            if length <= 0 or length > _MAX_BODY:
                return None
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                return None
            if not isinstance(payload, Mapping):
                return None
            query = payload.get("query")
            if not isinstance(query, str):
                return None
            sid = payload.get("session_id")
            if not isinstance(sid, str) or not sid:
                sid = None
            return (query, sid)

        def do_GET(self) -> None:  # noqa: N802 — stdlib handler naming
            if self.path.split("?", 1)[0] == "/healthz":
                self._send(200, {"status": "ok", **dict(_health())})
            else:
                self._send(404, {"error": "not found", "path": self.path})

        def do_POST(self) -> None:  # noqa: N802 — stdlib handler naming
            route = self.path.split("?", 1)[0]
            if route not in ("/gate", "/respond"):
                self._send(404, {"error": "not found", "path": self.path})
                return
            parsed = self._read_request()
            if parsed is _BAD_LENGTH:
                self._send(400, {"error": "invalid Content-Length header"})
                return
            if parsed is None:
                self._send(400, {"error": 'expected JSON body {"query": "<text>"}'})
                return
            query, session_id = parsed
            if session_id is None:  # never merge anonymous requests into one session
                session_id = "sess-" + uuid.uuid4().hex
            # Assign the per-session monotonic turn ONCE here so /gate and /respond share
            # one sequence; pass it down so neither route double-counts.
            turn = recorder.next_turn(session_id)
            try:
                if route == "/gate":
                    decision = gate(query, session_id, turn)
                    body = {**dict(decision), "session_id": session_id, "turn": turn}
                else:
                    result = serve(query, session_id, turn)
                    body = {**_as_json(result), "session_id": session_id, "turn": turn}
                self._send(200, body)
            except Exception as exc:  # never leak a stack trace; fail closed with 500.
                self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

    return ThreadingHTTPServer((host, port), _Handler)


def serve_forever(server: ThreadingHTTPServer) -> None:
    """Serve until interrupted, then shut the socket down cleanly."""
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
