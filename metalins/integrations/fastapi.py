"""FastAPI / ASGI integration — middleware that logs each HTTP turn.

Add `MetalinsMiddleware` to a FastAPI (or any ASGI) app and every HTTP
request/response pair is reported to Metalins automatically — no explicit
`agent.log(...)` in your route handlers.

    import metalins
    from metalins.integrations.fastapi import MetalinsMiddleware

    agent = metalins.Agent(api_key="ml_live_...", name="my-api").start()

    app = FastAPI()
    app.add_middleware(MetalinsMiddleware, agent=agent)

The middleware is *pure ASGI* — it imports nothing from FastAPI or
Starlette, so it adds no framework dependency to the SDK and works with
any ASGI server (FastAPI, Starlette, Quart, …). `app.add_middleware`
constructs it as `MetalinsMiddleware(app, agent=agent, **options)`.

One Metalins event is logged per completed HTTP request:

  - input  = ``"{method} {path}?{query}\\n"`` followed by the request body,
  - output = the response body.

Both are hashed locally by `agent.log(...)` — raw request and response
bodies never leave the process. Request metadata (method, path, status
code, captured byte counts) rides along in the event metadata.

Design choices that mirror the LangChain integration:

  - A logging failure is swallowed by default — a verification hiccup
    must never break the host API. Pass `raise_on_error=True` to surface
    it instead.
  - A request whose handler raises (no clean response) is *not* logged,
    the same way a failed LangChain chain logs nothing. The exception
    propagates untouched so the app's own error handling still runs.

Buffering safeguards:

  - Request and response bodies are buffered only up to `max_body_bytes`
    (1 MiB by default) so a large upload or streamed download cannot grow
    memory without bound. When a body is truncated the event metadata
    records it (`request_truncated` / `response_truncated`). Pass
    `max_body_bytes=None` to disable the cap.
  - Capture of either side can be turned off independently with
    `capture_request_body=False` / `capture_response_body=False` — the
    turn is still logged, just with that half empty.

Path filtering:

  - Pass `exclude_paths` to skip noisy endpoints (health checks, metrics)
    by exact path or path prefix.
  - Pass `should_log`, a `callable(scope) -> bool`, for full control over
    which requests are logged. When given it takes precedence over
    `exclude_paths`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Awaitable, Callable, Iterable, Protocol

if TYPE_CHECKING:  # pragma: no cover - typing only
    # ASGI primitives, named here only for readable type hints.
    Scope = dict[str, Any]
    Message = dict[str, Any]
    Receive = Callable[[], Awaitable[Message]]
    Send = Callable[[Message], Awaitable[None]]
    ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]


class _Loggable(Protocol):
    """The slice of `metalins.Agent` the middleware actually uses.

    Anything exposing this `log(...)` signature works — the real `Agent`,
    or a stand-in in tests — so the middleware never imports `Agent`.
    """

    def log(
        self,
        input: str | bytes,
        output: str | bytes,
        metadata: dict[str, Any] | None = None,
    ) -> Any: ...


# Default ceiling on how many bytes of each body we buffer for hashing.
# Bodies larger than this are hashed truncated and flagged in metadata,
# so a multi-megabyte upload or download can never grow memory without
# bound. 1 MiB comfortably covers ordinary JSON/form payloads.
_DEFAULT_MAX_BODY_BYTES = 1024 * 1024


class MetalinsMiddleware:
    """ASGI middleware that logs one Metalins event per HTTP request.

    Constructed by `app.add_middleware(MetalinsMiddleware, agent=agent)`.
    Non-HTTP scopes (lifespan, websocket) pass straight through untouched.
    """

    def __init__(
        self,
        app: "ASGIApp",
        agent: _Loggable,
        *,
        raise_on_error: bool = False,
        max_body_bytes: int | None = _DEFAULT_MAX_BODY_BYTES,
        capture_request_body: bool = True,
        capture_response_body: bool = True,
        exclude_paths: Iterable[str] | None = None,
        should_log: "Callable[[Scope], bool] | None" = None,
    ) -> None:
        if max_body_bytes is not None and max_body_bytes < 0:
            raise ValueError("max_body_bytes must be None or a non-negative int")
        self.app = app
        self._agent = agent
        self._raise_on_error = raise_on_error
        self._max_body_bytes = max_body_bytes
        self._capture_request_body = capture_request_body
        self._capture_response_body = capture_response_body
        self._exclude_paths = tuple(exclude_paths or ())
        self._should_log = should_log

    async def __call__(
        self, scope: "Scope", receive: "Receive", send: "Send"
    ) -> None:
        # Only HTTP turns are loggable. Everything else (lifespan,
        # websocket) is none of our business — forward it verbatim.
        if scope.get("type") != "http" or not self._wants(scope):
            await self.app(scope, receive, send)
            return

        # --- capture the request body as the app reads it ---------------
        # Wrapping `receive` (rather than draining the stream up front)
        # keeps the body flowing to the app chunk by chunk, so streaming
        # request handlers keep working; we just tee a bounded copy.
        req_body = bytearray()
        req_state = {"truncated": False}

        async def wrapped_receive() -> "Message":
            message = await receive()
            if (
                self._capture_request_body
                and message["type"] == "http.request"
            ):
                self._accumulate(req_body, req_state, message.get("body", b""))
            return message

        # --- capture the response status + body as the app sends it -----
        resp_body = bytearray()
        resp_state: dict[str, Any] = {"truncated": False, "status": None}

        async def wrapped_send(message: "Message") -> None:
            mtype = message["type"]
            if mtype == "http.response.start":
                resp_state["status"] = message.get("status")
            elif (
                self._capture_response_body
                and mtype == "http.response.body"
            ):
                self._accumulate(
                    resp_body, resp_state, message.get("body", b"")
                )
            await send(message)

        # If the handler raises, the turn never produced a clean
        # input/output pair — log nothing (mirroring the LangChain
        # handler's on_*_error) and let the exception propagate so the
        # app's own error handling still runs.
        await self.app(scope, wrapped_receive, wrapped_send)

        self._safe_log(scope, bytes(req_body), req_state, bytes(resp_body), resp_state)

    # ----------------------------------------------------------- internals

    def _wants(self, scope: "Scope") -> bool:
        """Whether this request should be logged."""
        if self._should_log is not None:
            return bool(self._should_log(scope))
        if self._exclude_paths:
            path = scope.get("path", "")
            if any(path == p or path.startswith(p) for p in self._exclude_paths):
                return False
        return True

    def _accumulate(
        self, buf: bytearray, state: dict[str, Any], chunk: bytes
    ) -> None:
        """Append `chunk` to `buf`, honoring the byte cap.

        Once the cap is reached the body is marked truncated and further
        bytes are dropped — the buffer never grows past the limit.
        """
        if not chunk:
            return
        if self._max_body_bytes is None:
            buf.extend(chunk)
            return
        room = self._max_body_bytes - len(buf)
        if room <= 0:
            state["truncated"] = True
            return
        if len(chunk) > room:
            buf.extend(chunk[:room])
            state["truncated"] = True
        else:
            buf.extend(chunk)

    def _safe_log(
        self,
        scope: "Scope",
        req_body: bytes,
        req_state: dict[str, Any],
        resp_body: bytes,
        resp_state: dict[str, Any],
    ) -> None:
        try:
            method = scope.get("method", "")
            path = scope.get("path", "")
            query = scope.get("query_string", b"") or b""
            if isinstance(query, str):
                query = query.encode("latin-1")
            # Prefix the body with the request line so identical bodies on
            # different routes/methods hash to distinct events. Built as
            # bytes throughout to stay agnostic to binary payloads.
            request_line = method.encode("latin-1") + b" " + path.encode("utf-8")
            if query:
                request_line += b"?" + query
            input_repr = request_line + b"\n" + req_body

            metadata = {
                "http": {
                    "method": method,
                    "path": path,
                    "status_code": resp_state.get("status"),
                    "request_bytes": len(req_body),
                    "response_bytes": len(resp_body),
                    "request_truncated": req_state.get("truncated", False),
                    "response_truncated": resp_state.get("truncated", False),
                }
            }
            self._agent.log(
                input=input_repr, output=resp_body, metadata=metadata
            )
        except Exception:
            if self._raise_on_error:
                raise
