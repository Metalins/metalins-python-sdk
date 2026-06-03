# Anthropic SDK Integration — metalins.trace and @metalins.monitor

The Anthropic integration provides two primitives for recording Metalins
verification events when calling the Anthropic SDK directly (no LangChain
required):

- **`metalins.trace(agent)`** — a context manager that wraps any block
  containing Anthropic API calls.
- **`@metalins.monitor(agent)`** — a decorator that wraps a function whose
  first argument is the messages input and whose return value is the output.

Both work for sync and async code, capture input/output, and call
`agent.log(...)` exactly once per turn.

## Installation

```bash
pip install metalins anthropic
```

No extra `metalins` optional-dependency is needed — the Anthropic integration
module imports nothing from the `anthropic` package at load time. It works by
duck-typing the response object, so it is compatible with any version of the
Anthropic SDK.

## Context Manager — `metalins.trace`

### Sync Usage

```python
import anthropic
import metalins

agent = metalins.Agent(api_key="ml_live_...", name="my-claude-agent").start()
client = anthropic.Anthropic()

messages = [{"role": "user", "content": "Summarise this article: ..."}]

with metalins.trace(agent) as t:
    t.set_input(messages)                 # capture what you're sending
    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        messages=messages,
    )
    t.set_output(response)                # capture the Anthropic response
```

### Async Usage

```python
import anthropic
import metalins

agent = metalins.Agent(api_key="ml_live_...", name="my-async-agent").start()
async_client = anthropic.AsyncAnthropic()

async def handle_request(messages: list[dict]) -> str:
    async with metalins.trace(agent) as t:
        t.set_input(messages)
        response = await async_client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=512,
            messages=messages,
        )
        t.set_output(response)
    return response.content[0].text
```

### How the Context Manager Works

```python
with metalins.trace(agent) as t:
    ...
```

`t` is a `TraceContext` object with two methods:

| Method | Purpose |
|--------|---------|
| `t.set_input(value)` | Record the input for this turn. Accepts a list of message dicts, a string, bytes, or any JSON-serialisable object. |
| `t.set_output(value)` | Record the output. Accepts an Anthropic `Message`, a plain string, or any JSON-serialisable object. Text is automatically extracted from `response.content[0].text`. |

If the block raises an exception, **nothing is logged** — the exception
propagates normally and the turn is treated as failed.

## Decorator — `@metalins.monitor`

Use `monitor` when you have a function dedicated to calling Claude. The first
positional argument is captured as the input and the return value as the output.

### Sync Decorator

```python
import anthropic
import metalins

agent = metalins.Agent(api_key="ml_live_...", name="classification-agent").start()
client = anthropic.Anthropic()

@metalins.monitor(agent)
def classify_ticket(messages: list[dict]) -> str:
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=256,
        messages=messages,
    )
    return response.content[0].text

# Call normally — Metalins logs input + output automatically.
result = classify_ticket([{"role": "user", "content": "My order never arrived."}])
```

### Async Decorator

```python
import anthropic
import metalins

agent = metalins.Agent(api_key="ml_live_...", name="refund-agent").start()
async_client = anthropic.AsyncAnthropic()

@metalins.monitor(agent)
async def process_refund(messages: list[dict]) -> str:
    response = await async_client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        messages=messages,
    )
    return response.content[0].text

# In your async route / task:
reply = await process_refund([{"role": "user", "content": "Refund order #42"}])
```

### What `monitor` Captures

| Logged field | Source |
|---|---|
| `input` | The first positional argument (JSON-serialised if it's a list/dict) |
| `output` | The return value — text extracted if it's an Anthropic Message, otherwise JSON-serialised |

If the function raises, **nothing is logged**.

## Configuration Options

Both `trace` and `monitor` accept the same keyword arguments:

```python
metalins.trace(
    agent,

    # Extra key/values attached to the Metalins event.
    metadata={"service": "refund-agent", "env": "prod"},

    # Re-raise agent.log() failures. Default False — log failures are
    # silenced so a Metalins outage never breaks your application.
    raise_on_error=False,
)

@metalins.monitor(
    agent,
    metadata={"model": "claude-haiku-4-5"},
    raise_on_error=False,
)
async def run(messages): ...
```

## Choosing Between `trace` and `monitor`

| Scenario | Recommended primitive |
|---|---|
| Ad-hoc calls spread across a function body | `metalins.trace` context manager |
| A dedicated function that calls Claude and returns its result | `@metalins.monitor` decorator |
| Async FastAPI route that calls Claude directly | Either — `trace` gives more granular control |
| Multi-step reasoning where you want one event per turn | `metalins.trace` — call `set_input`/`set_output` once per conversation turn |

## Full Example — FastAPI Route with Async Anthropic

```python
import anthropic
import metalins
from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager

metalins_agent = metalins.Agent(api_key="ml_live_...", name="support-bot")
anthropic_client = anthropic.AsyncAnthropic()


@asynccontextmanager
async def lifespan(app):
    metalins_agent.start()
    yield
    metalins_agent.stop()


app = FastAPI(lifespan=lifespan)


class ChatRequest(BaseModel):
    messages: list[dict]


@app.post("/chat")
async def chat(req: ChatRequest):
    async with metalins.trace(metalins_agent) as t:
        t.set_input(req.messages)
        response = await anthropic_client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=1024,
            messages=req.messages,
        )
        t.set_output(response)
    return {"reply": response.content[0].text}
```

## Full Example — Decorator Pattern with Metadata

```python
import anthropic
import metalins

agent = metalins.Agent(api_key="ml_live_...", name="email-triage").start()
client = anthropic.Anthropic()

SYSTEM_PROMPT = "Classify the following support email into: billing, technical, general."


@metalins.monitor(agent, metadata={"pipeline": "email-triage", "model": "claude-haiku-4-5"})
def triage_email(messages: list[dict]) -> str:
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=128,
        system=SYSTEM_PROMPT,
        messages=messages,
    )
    return response.content[0].text


category = triage_email([{"role": "user", "content": "I was charged twice this month."}])
print(category)  # "billing"
```

## Error Handling

```python
# Default: swallow log errors (recommended for production)
with metalins.trace(agent) as t:
    ...  # agent.log() failure is silenced — your app keeps working

# Opt in to raising: useful during local development
with metalins.trace(agent, raise_on_error=True) as t:
    ...  # will raise if agent.log() fails
```

The same option is available on `@metalins.monitor(agent, raise_on_error=True)`.
