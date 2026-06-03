"""Framework integrations for the Metalins SDK.

Each submodule wires `metalins.Agent` into a specific agent framework
so a turn is logged without an explicit `agent.log(...)` call. The
integrations are kept here, behind their own imports, so the base
`metalins` package never depends on any framework.

Currently shipped:
- `metalins.integrations.langchain` — a LangChain callback handler.
"""
