# LangChain Integration — MetalinsCallbackHandler

`MetalinsCallbackHandler` is a LangChain callback handler that records one
Metalins verification event per top-level chain or LLM invocation. Attach it
once; every call to `chain.invoke(...)` or a bare LLM's `llm.invoke(...)` is
automatically captured — no manual `agent.log()` needed.

## Installation

```bash
pip install "metalins[langchain]"
# This installs metalins + langchain-core
```

## Quick Start

```python
import metalins
from metalins.integrations.langchain import MetalinsCallbackHandler
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

# 1. Create and start the Metalins agent.
agent = metalins.Agent(api_key="ml_live_...", name="my-langchain-bot").start()

# 2. Create the callback handler.
handler = MetalinsCallbackHandler(agent)

# 3. Pass the handler when invoking the chain or LLM.
llm = ChatAnthropic(model="claude-haiku-4-5")
response = llm.invoke(
    [HumanMessage(content="What is 2+2?")],
    config={"callbacks": [handler]},
)
print(response.content)
```

## What Gets Logged

One Metalins event per **top-level** invocation:

| Field   | Value |
|---------|-------|
| `input` | JSON-serialised input (messages list, prompt string, dict, etc.) |
| `output` | JSON-serialised output (AIMessage, dict, string, etc.) |

Nested chains — e.g. a `SequentialChain` made of three sub-chains — produce
**one** event for the outer chain, not one per sub-step. This matches the
natural granularity of a "turn": one user request → one agent response.

If the chain raises an error the turn is considered failed and **nothing is
logged** (the exception propagates normally).

## Attaching the Handler

### Per-call (recommended for multi-agent setups)

```python
response = chain.invoke(user_input, config={"callbacks": [handler]})
```

### Globally on the chain object

```python
chain = my_chain.with_config({"callbacks": [handler]})
response = chain.invoke(user_input)
```

### On the LLM directly

```python
llm = ChatAnthropic(model="claude-opus-4-5", callbacks=[handler])
response = llm.invoke(messages)
```

### With LangChain agents (AgentExecutor)

```python
from langchain.agents import AgentExecutor

executor = AgentExecutor(agent=lc_agent, tools=tools)
result = executor.invoke(
    {"input": "Refund order #42"},
    config={"callbacks": [handler]},
)
```

## Configuration

```python
handler = MetalinsCallbackHandler(
    agent,
    # Re-raise agent.log() errors instead of swallowing them.
    raise_on_error=False,  # default
)
```

## Full Example — RAG Pipeline

```python
import metalins
from metalins.integrations.langchain import MetalinsCallbackHandler
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

agent = metalins.Agent(api_key="ml_live_...", name="rag-bot").start()
handler = MetalinsCallbackHandler(agent)

llm = ChatAnthropic(model="claude-haiku-4-5")
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Context: {context}"),
    ("human", "{question}"),
])
chain = prompt | llm | StrOutputParser()

answer = chain.invoke(
    {"context": "Metalins is an AI identity verification platform.", "question": "What is Metalins?"},
    config={"callbacks": [handler]},
)
print(answer)
# One event logged: input = serialised {context, question}, output = answer string
```

## Multiple Agents

If your application runs multiple independent agents (e.g. a triage agent and a
refund agent) give each its own `MetalinsAgent` and its own handler:

```python
triage_agent  = metalins.Agent(api_key="ml_live_...", name="triage-agent").start()
refund_agent  = metalins.Agent(api_key="ml_live_...", name="refund-agent").start()

triage_handler = MetalinsCallbackHandler(triage_agent)
refund_handler = MetalinsCallbackHandler(refund_agent)

# Route to the right handler at call time.
triage_chain.invoke(user_msg, config={"callbacks": [triage_handler]})
refund_chain.invoke(user_msg, config={"callbacks": [refund_handler]})
```

## Lifecycle — Shutdown

```python
import atexit

agent = metalins.Agent(api_key="ml_live_...", name="my-bot").start()
atexit.register(agent.stop)
```

Or as a context manager:

```python
with metalins.Agent(api_key="ml_live_...", name="my-bot") as agent:
    handler = MetalinsCallbackHandler(agent)
    chain.invoke(user_input, config={"callbacks": [handler]})
```
