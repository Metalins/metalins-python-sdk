# FastAPI Integration — MetalinsMiddleware

`MetalinsMiddleware` is a pure ASGI middleware that automatically records every
HTTP request/response pair as a Metalins verification event. Add it once at
startup and every route in your FastAPI (or any ASGI) app is covered — no
changes to individual route handlers required.

## Installation

```bash
pip install metalins
# Optional: pull FastAPI as a convenience
pip install "metalins[fastapi]"
```

## Quick Start

```python
import metalins
from metalins.integrations.fastapi import MetalinsMiddleware
from fastapi import FastAPI

# 1. Create and start the agent once at application startup.
agent = metalins.Agent(api_key="ml_live_...", name="my-api").start()

# 2. Add the middleware — that's it.
app = FastAPI()
app.add_middleware(MetalinsMiddleware, agent=agent)


@app.post("/chat")
async def chat(body: dict):
    # No metalins.log() call needed — the middleware handles it.
    return {"reply": "Hello!"}
```

## What Gets Logged

One Metalins event is created per completed HTTP request:

| Field   | Value |
|---------|-------|
| `input` | `"{METHOD} {path}\n{request_body}"` |
| `output` | raw response body bytes |

Both sides are hashed locally inside `agent.log()` — raw request and response
bodies never leave your process.

If the route handler raises an unhandled exception, the turn is considered
failed and **nothing is logged** (the exception propagates as normal).

## Configuration Reference

```python
app.add_middleware(
    MetalinsMiddleware,
    agent=agent,

    # Swallow agent.log() failures so a Metalins outage never breaks your API.
    # Set True only during development/debugging.
    raise_on_error=False,          # default

    # Cap buffered body size. Large uploads/downloads are hashed truncated
    # and flagged in metadata. Set None to disable the cap.
    max_body_bytes=1024 * 1024,    # default: 1 MiB

    # Toggle capture of each side independently.
    capture_request_body=True,     # default
    capture_response_body=True,    # default

    # Skip exact paths or path prefixes (health checks, metrics, etc.).
    exclude_paths=["/health", "/metrics", "/ready"],

    # Full control: a callable(scope) -> bool. Overrides exclude_paths.
    should_log=None,               # default: log everything
)
```

### Excluding Noisy Endpoints

```python
app.add_middleware(
    MetalinsMiddleware,
    agent=agent,
    exclude_paths=["/health", "/metrics", "/docs", "/openapi.json"],
)
```

`exclude_paths` matches both exact paths **and** path prefixes, so
`"/health"` also excludes `"/health/live"` and `"/health/ready"`.

### Custom Log Predicate

For more granular control — e.g. log only POST requests — pass `should_log`:

```python
app.add_middleware(
    MetalinsMiddleware,
    agent=agent,
    should_log=lambda scope: scope.get("method") in ("POST", "PUT", "PATCH"),
)
```

When `should_log` is set it takes precedence over `exclude_paths`.

## Metadata

Every logged event carries HTTP metadata:

```python
{
    "http": {
        "method": "POST",
        "path": "/chat",
        "status_code": 200,
        "request_bytes": 42,
        "response_bytes": 128,
        "request_truncated": False,
        "response_truncated": False,
    }
}
```

`request_truncated` / `response_truncated` are `True` when the body was
clipped at `max_body_bytes`.

## Lifecycle — Shutdown

Stop the verification background loop cleanly on application shutdown:

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app):
    agent.start()
    yield
    agent.stop()

app = FastAPI(lifespan=lifespan)
app.add_middleware(MetalinsMiddleware, agent=agent)
```

## Framework Compatibility

`MetalinsMiddleware` is **pure ASGI** — it imports nothing from FastAPI or
Starlette at runtime. It works with any ASGI framework: FastAPI, Starlette,
Quart, BlackSheep, etc.

```python
# Starlette
from starlette.applications import Starlette
app = Starlette(...)
app.add_middleware(MetalinsMiddleware, agent=agent)

# Raw ASGI
wrapped = MetalinsMiddleware(your_asgi_app, agent=agent)
```

## Full Example

```python
import metalins
from metalins.integrations.fastapi import MetalinsMiddleware
from fastapi import FastAPI
from contextlib import asynccontextmanager
from pydantic import BaseModel


agent = metalins.Agent(api_key="ml_live_...", name="customer-support-api")


@asynccontextmanager
async def lifespan(app):
    agent.start()
    yield
    agent.stop()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    MetalinsMiddleware,
    agent=agent,
    exclude_paths=["/health", "/docs", "/openapi.json"],
    max_body_bytes=512 * 1024,  # 512 KiB
)


class ChatRequest(BaseModel):
    message: str


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat")
async def chat(req: ChatRequest):
    # ... call your LLM here ...
    return {"reply": f"You said: {req.message}"}
```
