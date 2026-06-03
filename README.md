# metalins (Python SDK)

Thin client for [Metalins](https://metalins.com) — identity verification for AI agents.

The SDK is a wrapper over the Metalins developer API: it captures your agent's
behavior, reports it to the server, answers the server's verification checks,
and receives signed identity proofs. **All scoring and comparison runs
server-side.**

## Install

```bash
pip install metalins
```

## Quick start — the `Agent` facade

`Agent` is the one-import entry point for a long-lived agent. It registers the
agent on first run, persists its state so a restart resumes the same agent, and
runs a background loop that answers the server's verification checks on a
cadence — so verification keeps working whether or not the agent is busy.

```python
import metalins

agent = metalins.Agent(api_key="ml_live_...", name="my-customer-bot")
agent.start()                                  # background check loop on

# ... wherever the agent finishes a turn:
agent.log(input=user_message, output=agent_reply)

# Read status, or issue a signed identity proof for another party.
status = agent.get_status()
proof = agent.issue_proof(ttl_seconds=3600)

agent.stop()                                   # on shutdown
```

Or as a context manager, which starts and stops the loop for you:

```python
with metalins.Agent(api_key="ml_live_...", name="my-customer-bot") as agent:
    agent.log(input=user_message, output=agent_reply)
```

The SDK hashes payloads locally — raw prompt and response text never leave your
process. The background loop does only hashing and HTTP; no model is involved.

## State persistence

`Agent` keeps its session — the agent id, its secret, and the running hash
chain — in a `StateStore` so a restart resumes the same agent. The default is a
local JSON file at `~/.metalins/<name>.json` with owner-only (`0600`)
permissions, zero config.

To keep the secret somewhere else (a database row, a secrets manager), pass any
object with `load() -> dict | None` and `save(dict) -> None`:

```python
agent = metalins.Agent(api_key="ml_live_...", name="my-bot", store=my_store)
```

## LangChain

Attach the callback handler and every top-level chain or LLM call is logged
automatically — no explicit `agent.log(...)` in your turn code. Install the
extra: `pip install metalins[langchain]`.

```python
from metalins import Agent
from metalins.integrations.langchain import MetalinsCallbackHandler

agent = Agent(api_key="ml_live_...", name="my-bot").start()
handler = MetalinsCallbackHandler(agent)

chain.invoke(user_input, config={"callbacks": [handler]})
```

## FastAPI

Add the ASGI middleware and every HTTP request/response pair is logged
automatically — no explicit `agent.log(...)` in your route handlers. The
middleware is pure ASGI, so it also works with Starlette and any other ASGI app.

```python
import metalins
from fastapi import FastAPI
from metalins.integrations.fastapi import MetalinsMiddleware

agent = metalins.Agent(api_key="ml_live_...", name="my-api").start()

app = FastAPI()
app.add_middleware(MetalinsMiddleware, agent=agent)
```

Skip noisy endpoints with `exclude_paths=["/health"]`, or pass
`should_log=lambda scope: ...` for full control. Request and response bodies are
hashed locally and buffered only up to `max_body_bytes` (1 MiB by default).

## Lower-level: `Client` + `AgentSession`

`Agent` is built from two primitives you can also use directly. `Client` is a
thin wrapper with one method per developer-API endpoint; `AgentSession` holds
the per-agent hash chain needed to answer verification checks.

```python
ml = metalins.Client(api_key="ml_live_...")

session = ml.start_session(name="my-bot", model="claude-sonnet")
session.log_event("user asked about pricing", "the agent's reply ...")

# Persist / rehydrate the session yourself.
saved = session.to_dict()
session = ml.attach_session(metalins.AgentSession.from_dict(saved))
```

Every developer-API endpoint is also a direct method on `Client`
(`create_agent`, `log_event`, `answer_check`, `list_pending_checks`,
`list_agents`, `get_agent`, `issue_proof`, `revoke_agent`), each returning the
server's JSON response as a plain dict.

## License

Apache 2.0. See [LICENSE](LICENSE).
