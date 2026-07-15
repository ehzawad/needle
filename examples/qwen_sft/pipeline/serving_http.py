"""The tangible serving microservice boundary — a stdlib HTTP front door.

Real-world analog: a KServe / Triton inference endpoint (or an Envoy-fronted model
service) exposing the gate/respond contract over HTTP. It is deliberately the ONE
network surface in this otherwise file-backed modular monolith, and it is control-plane /
CPU only: it wraps an already-built :data:`~pipeline.contracts.ServeFn` (from
:func:`pipeline.serving.make_server`) plus its gate. For a mock/CI release both callables
are model-free; for a real release the injected callables would route model work through
the guarded GPU worker — this HTTP process itself never imports torch.

Routes (JSON in/out):

  * ``POST /gate``     ``{"query": ...}`` -> the policy-wrapped gate decision
  * ``POST /respond``  ``{"query": ...}`` -> the cross-checked :class:`TurnResult`
  * ``GET  /healthz``  -> liveness + the served artifact id + circuit state
"""
from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, is_dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from pipeline.contracts import GateFn, ServeFn

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


def build_http_server(
    host: str,
    port: int,
    *,
    serve: ServeFn,
    gate: GateFn,
    artifact_id: str | None,
    healthz: Callable[[], Mapping[str, Any]] | None = None,
) -> ThreadingHTTPServer:
    """Build (but do not start) a :class:`ThreadingHTTPServer` over ``serve``/``gate``.

    ``serve`` is the frozen 7-step :data:`ServeFn`; ``gate`` is its policy-wrapped safe
    gate (what ``/gate`` returns). ``healthz`` supplies the liveness payload.
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

        def _read_query(self) -> str | None:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > _MAX_BODY:
                return None
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                return None
            if not isinstance(payload, Mapping):
                return None
            query = payload.get("query")
            return query if isinstance(query, str) else None

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
            query = self._read_query()
            if query is None:
                self._send(400, {"error": 'expected JSON body {"query": "<text>"}'})
                return
            try:
                if route == "/gate":
                    self._send(200, dict(gate(query)))
                else:
                    self._send(200, serve(query))
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
