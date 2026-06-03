"""Tests for the FastAPI / ASGI middleware integration.

Two layers:

  - Core tests drive the middleware directly at the ASGI contract level
    with a hand-written echo app — no web framework or HTTP transport
    needed, so capture/log behavior is exercised with only the SDK's own
    deps and full control over request/response chunking.
  - One integration test mounts the middleware on a real Starlette app
    via `add_middleware` to prove the public usage path works; it skips
    if Starlette is not installed.
"""
import asyncio
import json

import pytest

from metalins.integrations.fastapi import MetalinsMiddleware


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

class _FakeAgent:
    """Stand-in for Agent — records log() calls without any HTTP."""

    def __init__(self):
        self.calls = []

    def log(self, input, output, metadata=None):
        self.calls.append(
            {"input": input, "output": output, "metadata": metadata}
        )
        return {}


def _echo_app(*, status: int = 200, resp_chunks: int = 1):
    """A minimal ASGI app that echoes the request body back.

    Reads the whole request body, then sends the response in
    `resp_chunks` pieces so multi-part `http.response.body` streaming is
    covered.
    """

    async def app(scope, receive, send):
        assert scope["type"] == "http"
        body = b""
        more = True
        while more:
            message = await receive()
            body += message.get("body", b"")
            more = message.get("more_body", False)

        await send({
            "type": "http.response.start",
            "status": status,
            "headers": [(b"content-type", b"application/json")],
        })
        if resp_chunks <= 1:
            await send({"type": "http.response.body", "body": body})
        else:
            size = max(1, len(body) // resp_chunks)
            pieces = [body[i:i + size] for i in range(0, len(body), size)] or [b""]
            for i, piece in enumerate(pieces):
                await send({
                    "type": "http.response.body",
                    "body": piece,
                    "more_body": i < len(pieces) - 1,
                })

    return app


def call_asgi(
    app,
    *,
    method: str = "GET",
    path: str = "/",
    query: bytes = b"",
    body: bytes = b"",
    req_chunks: int = 1,
):
    """Drive an ASGI `app` through one HTTP request; return sent messages.

    Builds the scope, feeds the request body (optionally split across
    `req_chunks` `http.request` messages), and collects every message the
    app sends.
    """
    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "query_string": query,
        "headers": [],
    }

    if req_chunks <= 1:
        req_messages = [{"type": "http.request", "body": body, "more_body": False}]
    else:
        size = max(1, len(body) // req_chunks)
        pieces = [body[i:i + size] for i in range(0, len(body), size)] or [b""]
        req_messages = [
            {
                "type": "http.request",
                "body": piece,
                "more_body": i < len(pieces) - 1,
            }
            for i, piece in enumerate(pieces)
        ]
    req_iter = iter(req_messages)

    async def receive():
        try:
            return next(req_iter)
        except StopIteration:
            return {"type": "http.disconnect"}

    sent: list = []

    async def send(message):
        sent.append(message)

    asyncio.run(app(scope, receive, send))
    return sent


def response_of(sent):
    """Reduce sent ASGI messages to (status, full_body)."""
    status = None
    body = b""
    for m in sent:
        if m["type"] == "http.response.start":
            status = m["status"]
        elif m["type"] == "http.response.body":
            body += m.get("body", b"")
    return status, body


# --------------------------------------------------------------------------- #
# Core capture behavior                                                       #
# --------------------------------------------------------------------------- #

def test_logs_one_event_per_request_with_bodies():
    agent = _FakeAgent()
    mw = MetalinsMiddleware(_echo_app(), agent=agent)

    sent = call_asgi(mw, method="POST", path="/chat", body=b"hello world")
    status, out = response_of(sent)
    assert status == 200
    assert out == b"hello world"

    assert len(agent.calls) == 1
    call = agent.calls[0]
    # input carries the request line then the body; output is the response.
    assert call["input"] == b"POST /chat\nhello world"
    assert call["output"] == b"hello world"


def test_metadata_records_method_path_and_status():
    agent = _FakeAgent()
    mw = MetalinsMiddleware(_echo_app(status=201), agent=agent)

    call_asgi(mw, method="POST", path="/v1/items", query=b"tag=x", body=b"{}")

    meta = agent.calls[0]["metadata"]["http"]
    assert meta["method"] == "POST"
    assert meta["path"] == "/v1/items"
    assert meta["status_code"] == 201
    assert meta["request_bytes"] == 2
    assert meta["response_bytes"] == 2
    assert meta["request_truncated"] is False
    assert meta["response_truncated"] is False


def test_query_string_included_in_input():
    agent = _FakeAgent()
    mw = MetalinsMiddleware(_echo_app(), agent=agent)

    call_asgi(mw, method="GET", path="/search", query=b"q=cats&n=3")

    assert agent.calls[0]["input"] == b"GET /search?q=cats&n=3\n"


def test_request_and_response_streamed_in_chunks_fully_captured():
    agent = _FakeAgent()
    mw = MetalinsMiddleware(_echo_app(resp_chunks=4), agent=agent)

    payload = b"abcdefghij" * 3  # 30 bytes
    sent = call_asgi(mw, method="POST", path="/stream", body=payload, req_chunks=5)
    _, out = response_of(sent)
    assert out == payload

    call = agent.calls[0]
    assert call["input"] == b"POST /stream\n" + payload
    assert call["output"] == payload
    assert call["metadata"]["http"]["request_bytes"] == len(payload)
    assert call["metadata"]["http"]["response_bytes"] == len(payload)


# --------------------------------------------------------------------------- #
# Safeguards: truncation, capture toggles, filtering                          #
# --------------------------------------------------------------------------- #

def test_bodies_truncated_at_max_body_bytes():
    agent = _FakeAgent()
    mw = MetalinsMiddleware(_echo_app(), agent=agent, max_body_bytes=8)

    call_asgi(mw, method="POST", path="/big", body=b"0123456789ABCDEF")  # 16

    call = agent.calls[0]
    # The request line is not counted toward the body cap; only the first
    # 8 body bytes are kept.
    assert call["input"] == b"POST /big\n01234567"
    assert call["output"] == b"01234567"
    meta = call["metadata"]["http"]
    assert meta["request_truncated"] is True
    assert meta["response_truncated"] is True


def test_truncation_across_multiple_chunks():
    # Cap is hit mid-stream across several request chunks.
    agent = _FakeAgent()
    mw = MetalinsMiddleware(_echo_app(), agent=agent, max_body_bytes=5)

    call_asgi(mw, method="POST", path="/big", body=b"abcdefghij", req_chunks=5)

    call = agent.calls[0]
    assert call["input"] == b"POST /big\nabcde"
    assert call["metadata"]["http"]["request_truncated"] is True


def test_unlimited_when_max_body_bytes_none():
    agent = _FakeAgent()
    mw = MetalinsMiddleware(_echo_app(), agent=agent, max_body_bytes=None)

    big = b"x" * 5000
    call_asgi(mw, method="POST", path="/big", body=big)

    call = agent.calls[0]
    assert call["output"] == big
    assert call["metadata"]["http"]["response_truncated"] is False


def test_capture_toggles_off_leave_bodies_empty_but_still_log():
    agent = _FakeAgent()
    mw = MetalinsMiddleware(
        _echo_app(),
        agent=agent,
        capture_request_body=False,
        capture_response_body=False,
    )

    call_asgi(mw, method="POST", path="/chat", body=b"secret payload")

    call = agent.calls[0]
    assert call["input"] == b"POST /chat\n"  # request line only, no body
    assert call["output"] == b""
    # The turn is still logged — only the bodies are omitted.
    assert call["metadata"]["http"]["status_code"] == 200


def test_exclude_paths_skips_matching_requests():
    agent = _FakeAgent()
    mw = MetalinsMiddleware(
        _echo_app(), agent=agent, exclude_paths=["/health", "/metrics"]
    )

    call_asgi(mw, method="GET", path="/health")          # excluded (exact)
    call_asgi(mw, method="GET", path="/health/live")     # excluded (prefix)
    call_asgi(mw, method="POST", path="/chat", body=b"hi")  # logged

    assert len(agent.calls) == 1
    assert agent.calls[0]["metadata"]["http"]["path"] == "/chat"


def test_excluded_request_still_reaches_the_app():
    # Filtering must not swallow the request — the app still responds.
    agent = _FakeAgent()
    mw = MetalinsMiddleware(_echo_app(), agent=agent, exclude_paths=["/health"])

    sent = call_asgi(mw, method="GET", path="/health", body=b"ping")
    status, out = response_of(sent)
    assert status == 200
    assert out == b"ping"
    assert agent.calls == []


def test_should_log_predicate_takes_precedence():
    agent = _FakeAgent()
    mw = MetalinsMiddleware(
        _echo_app(),
        agent=agent,
        exclude_paths=["/chat"],  # ignored because should_log is set
        should_log=lambda scope: scope.get("method") == "POST",
    )

    call_asgi(mw, method="GET", path="/chat")
    call_asgi(mw, method="POST", path="/chat", body=b"hi")

    assert len(agent.calls) == 1
    assert agent.calls[0]["metadata"]["http"]["method"] == "POST"


# --------------------------------------------------------------------------- #
# Error handling                                                              #
# --------------------------------------------------------------------------- #

def test_log_failure_is_swallowed_by_default():
    class _BoomAgent:
        def log(self, **kwargs):
            raise RuntimeError("verification backend down")

    mw = MetalinsMiddleware(_echo_app(), agent=_BoomAgent())
    # A logging failure must not break the response.
    sent = call_asgi(mw, method="POST", path="/chat", body=b"hi")
    status, out = response_of(sent)
    assert status == 200
    assert out == b"hi"


def test_log_failure_raises_when_opted_in():
    class _BoomAgent:
        def log(self, **kwargs):
            raise RuntimeError("verification backend down")

    mw = MetalinsMiddleware(_echo_app(), agent=_BoomAgent(), raise_on_error=True)
    with pytest.raises(RuntimeError):
        call_asgi(mw, method="POST", path="/chat", body=b"hi")


def test_handler_exception_logs_nothing_and_propagates():
    agent = _FakeAgent()

    async def boom_app(scope, receive, send):
        await receive()  # consume request, then fail before responding
        raise RuntimeError("handler exploded")

    mw = MetalinsMiddleware(boom_app, agent=agent)
    with pytest.raises(RuntimeError, match="handler exploded"):
        call_asgi(mw, method="POST", path="/chat", body=b"hi")

    # A turn that never completed must not be logged.
    assert agent.calls == []


def test_non_http_scope_passes_through_untouched():
    agent = _FakeAgent()
    seen = {"lifespan": False}

    async def app(scope, receive, send):
        if scope["type"] == "lifespan":
            seen["lifespan"] = True

    mw = MetalinsMiddleware(app, agent=agent)

    async def drive():
        await mw({"type": "lifespan"}, _noop_receive, _noop_send)

    asyncio.run(drive())
    assert seen["lifespan"] is True
    assert agent.calls == []  # nothing logged for a non-HTTP scope


async def _noop_receive():
    return {"type": "lifespan.startup"}


async def _noop_send(message):
    return None


def test_rejects_negative_max_body_bytes():
    with pytest.raises(ValueError):
        MetalinsMiddleware(_echo_app(), agent=_FakeAgent(), max_body_bytes=-1)


# --------------------------------------------------------------------------- #
# Real-framework integration (skips without Starlette)                        #
# --------------------------------------------------------------------------- #

def test_add_middleware_on_real_starlette_app():
    pytest.importorskip("starlette")
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Route
    from starlette.testclient import TestClient

    agent = _FakeAgent()

    async def handler(request):
        body = await request.json()
        return JSONResponse({"echo": body})

    app = Starlette(routes=[Route("/echo", handler, methods=["POST"])])
    app.add_middleware(MetalinsMiddleware, agent=agent)

    client = TestClient(app)
    resp = client.post("/echo", json={"name": "diana"})
    assert resp.status_code == 200
    assert resp.json() == {"echo": {"name": "diana"}}

    # Exactly one event, with the response body captured for hashing.
    assert len(agent.calls) == 1
    call = agent.calls[0]
    meta = call["metadata"]["http"]
    assert meta["path"] == "/echo"
    assert meta["method"] == "POST"
    assert meta["status_code"] == 200
    assert json.loads(call["output"]) == {"echo": {"name": "diana"}}
    # The request body is captured in the input (Starlette sends compact JSON).
    assert b'"name"' in call["input"] and b'"diana"' in call["input"]
