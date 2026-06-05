"""LangChain integration — a callback handler that logs each agent turn.

Attach `MetalinsCallbackHandler` to a LangChain chain or agent and every
top-level invocation is reported to Metalins automatically — no explicit
`agent.log(...)` in your turn code.

    from metalins import Agent
    from metalins.integrations.langchain import MetalinsCallbackHandler

    agent = Agent(api_key="ml_live_...", name="my-bot").start()
    handler = MetalinsCallbackHandler(agent)

    chain.invoke(user_input, config={"callbacks": [handler]})

Requires `langchain-core` — install the optional extra:
`pip install metalins[langchain]`.
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
from uuid import UUID

try:
    from langchain_core.callbacks import BaseCallbackHandler
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "The LangChain integration needs 'langchain-core'. Install it "
        "with: pip install metalins[langchain]"
    ) from exc

if TYPE_CHECKING:
    from metalins.agent import Agent


def _stringify(obj: Any) -> str:
    """Render a LangChain payload to a stable string for hashing.

    The string is hashed locally, never sent — what matters is that the
    same payload always renders the same way.
    """
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, (dict, list)):
        try:
            return json.dumps(obj, sort_keys=True, default=str)
        except (TypeError, ValueError):
            return str(obj)
    return str(obj)


class MetalinsCallbackHandler(BaseCallbackHandler):
    """A LangChain callback handler that logs each top-level turn.

    One Metalins event per top-level chain or LLM invocation
    (`parent_run_id is None`). Nested steps are not logged separately,
    so a chain of chains produces one event per turn, not one per link;
    a bare LLM call with no wrapping chain is covered too.

    A failure to log is swallowed by default — a verification hiccup
    must never break the host agent. Pass `raise_on_error=True` to
    surface it instead.
    """

    def __init__(self, agent: "Agent", *, raise_on_error: bool = False):
        super().__init__()
        self._agent = agent
        self._raise_on_error = raise_on_error
        # run_id -> stringified input, for runs awaiting their end event.
        self._pending: dict[UUID, str] = {}

    # -- chain-level ----------------------------------------------------- #

    def on_chain_start(
        self,
        serialized: Any,
        inputs: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        if parent_run_id is None:
            self._pending[run_id] = _stringify(inputs)

    def on_chain_end(
        self,
        outputs: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        if parent_run_id is None:
            self._flush(run_id, _stringify(outputs))

    def on_chain_error(
        self, error: BaseException, *, run_id: UUID, **kwargs: Any
    ) -> None:
        # The turn failed — drop the stashed input, log nothing.
        self._pending.pop(run_id, None)

    # -- llm-level (covers a bare LLM call with no wrapping chain) ------- #

    def on_llm_start(
        self,
        serialized: Any,
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        if parent_run_id is None:
            self._pending[run_id] = _stringify(prompts)

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        if parent_run_id is None:
            self._flush(run_id, _stringify(response))

    def on_llm_error(
        self, error: BaseException, *, run_id: UUID, **kwargs: Any
    ) -> None:
        self._pending.pop(run_id, None)

    # -- internals ------------------------------------------------------- #

    def _flush(self, run_id: UUID, output: str) -> None:
        input_repr = self._pending.pop(run_id, "")
        try:
            self._agent.log(input=input_repr, output=output)
        except Exception:
            if self._raise_on_error:
                raise
