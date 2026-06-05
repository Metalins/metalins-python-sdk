# metalins

**Zero Trust identity verification for AI agents.**

Your agents in production are black boxes. Metalins verifies they're still the same agents you deployed — same model, same behavior, continuously. It's the behavioral verification layer in the Zero Trust stack for AI agents.

## How it works

1. The SDK hashes your agent's inputs and outputs **locally** — raw prompts and responses never leave your infrastructure.
2. Signed hashes are sent to `api.metalins.ai`, where the behavioral engine runs.
3. The engine returns a continuous verification status: `verified`, `caution`, or `not_verified`.

Your data stays in your infra. We only see fingerprints.

## Install

```bash
pip install metalins
```

## Quick start

Three lines to start verifying your agent:

```python
import metalins

agent = metalins.Agent(api_key="ml_live_...", name="my-agent")
agent.start()

# Log each turn — hashing happens locally, automatically
agent.log(input=user_message, output=agent_reply)

# Check verification status at any time
status = agent.get_status()  # "verified" | "caution" | "not_verified"
```

Or as a context manager:

```python
with metalins.Agent(api_key="ml_live_...", name="my-agent") as agent:
    agent.log(input=user_message, output=agent_reply)
```

Get your API key at [metalins.ai](https://metalins.ai).

## Integrations

### LangChain

```python
from metalins import Agent
from metalins.integrations.langchain import MetalinsCallbackHandler

agent = Agent(api_key="ml_live_...", name="my-bot").start()
handler = MetalinsCallbackHandler(agent)

chain.invoke(user_input, config={"callbacks": [handler]})
```

Every chain and LLM call is logged automatically — no manual `agent.log()` needed.

### FastAPI / Starlette

```python
import metalins
from metalins.integrations.fastapi import MetalinsMiddleware

agent = metalins.Agent(api_key="ml_live_...", name="my-api").start()
app.add_middleware(MetalinsMiddleware, agent=agent)
```

Every request/response pair is logged automatically. Bodies are hashed locally and never buffered in full (1 MiB cap by default). Skip noisy endpoints with `exclude_paths=["/health"]`.

### Anthropic SDK

```python
import metalins

agent = metalins.Agent(api_key="ml_live_...", name="my-claude-agent").start()

with metalins.trace(agent):
    response = client.messages.create(...)
```

Or use the `@metalins.monitor` decorator on any function that calls the Anthropic SDK.

## What leaves your infrastructure

Only hashed fingerprints — never raw text:

| What we receive | What stays with you |
|-----------------|---------------------|
| SHA-256 hash of input | Raw prompt text |
| SHA-256 hash of output | Raw response text |
| Timestamp + agent ID | Your users' data |
| HMAC-signed event chain | Your model config |

The behavioral engine compares fingerprint patterns over time. It does not reconstruct your prompts or responses.

## State persistence

The SDK persists the agent session (ID, secret, hash chain) to `~/.metalins/<name>.json` with `0600` permissions by default. To store it elsewhere — a database, a secrets manager — pass any object with `load()` and `save()`:

```python
agent = metalins.Agent(api_key="ml_live_...", name="my-bot", store=my_store)
```

## License

Apache 2.0. See [LICENSE](LICENSE).
