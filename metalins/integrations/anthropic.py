"""Anthropic SDK integration — context manager and decorator for tracing.

Use ``metalins.trace`` to wrap any block that calls the Anthropic SDK, and
``metalins.monitor`` to decorate a coroutine or regular function whose first
positional argument is the messages list (or any value you want as the input).

Both capture the raw input and output and call ``agent.log(...)`` exactly
once per turn, so every Anthropic interaction is recorded in the Metalins
verification hash chain without any changes to your business logic.

Context-manager usage::

    import anthropic
    import metalins

    agent = metalins.Agent(api_key="ml_live_...", name="my-bot").start()
    client = anthropic.Anthropic()

    with metalins.trace(agent) as t:
        response = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello"}],
        )
        t.set_output(response)   # optional — auto-extracted on exit

The trace extracts ``input`` from ``t.input`` (set via ``t.set_input``) and
``output`` from ``t.output`` (set via ``t.set_output``).  If neither setter is
called the context manager sniffs the last assigned ``response`` variable that
looks like an Anthropic ``Message`` by duck-typing.

Decorator usage::

    @metalins.monitor(agent, agent_id_param=None)
    async def process_refund(messages: list[dict], *, context: str = "") -> str:
        response = await async_client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=512,
            messages=messages,
        )
        return response.content[0].text

    await process_refund([{"role": "user", "content": "Refund #42"}])

The decorator logs the function's first positional argument as the *input* and
its return value as the *output*.  For async coroutines it wraps with
``asyncio.iscoroutinefunction``; for sync functions the wrapper is sync.

Error handling:

* A failure inside the traced block propagates normally — the event is **not**
  logged (a failed turn leaves no input/output pair).
* A failure in ``agent.log(...)`` is swallowed by default so a verification
  hiccup never breaks the host application.  Pass ``raise_on_error=True`` to
  the context manager / decorator to surface it.

Requires the ``anthropic`` package only when you actually use it — the
integration module itself imports nothing from ``anthropic`` at load time so it
adds no hard dependency to the SDK.
"""
from __future__ import annotations

import asyncio
import functools
import json
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover — typing only
    from metalins.agent import Agent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stringify(obj: Any) -> str:
    """Render an arbitrary value to a stable, deterministic string.

    The string is hashed locally — it never leaves the process.  What
    matters is that identical data always maps to the same string.
    """
    if obj is None:
        return ""
    if isinstance(obj, (bytes, bytearray)):
        try:
            return obj.decode("utf-8", errors="replace")
        except Exception:
            return repr(obj)
    if isinstance(obj, str):
        return obj
    # Anthropic SDK ``Message`` objects expose ``model_dump()`` (pydantic v2)
    # or ``dict()`` (pydantic v1 / legacy) — try both before falling back to
    # json.dumps so we get a stable JSON string rather than __repr__.
    if hasattr(obj, "model_dump"):
        try:
            return json.dumps(obj.model_dump(), sort_keys=True, default=str)
        except Exception:
            pass
    if hasattr(obj, "dict"):
        try:
            return json.dumps(obj.dict(), sort_keys=True, default=str)
        except Exception:
            pass
    if isinstance(obj, (dict, list)):
        try:
            return json.dumps(obj, sort_keys=True, default=str)
        except (TypeError, ValueError):
            return str(obj)
    return str(obj)


def _extract_text_from_message(response: Any) -> str:
    """Pull plain text out of an Anthropic ``Message`` if possible.

    Falls back to _stringify so the result is always a non-empty string
    when there is content to hash.
    """
    # Anthropic SDK: response.content is a list of ContentBlock objects.
    # Each text block has a .text attribute.
    content = getattr(response, "content", None)
    if isinstance(content, list) and content:
        parts = []
        for block in content:
            text = getattr(block, "text", None)
            if text is not None:
                parts.append(str(text))
        if parts:
            return "\n".join(parts)
    return _stringify(response)


# ---------------------------------------------------------------------------
# TraceContext — the object yielded by the context manager
# ---------------------------------------------------------------------------

class TraceContext:
    """Holds the input and output for one traced Anthropic turn.

    You can set them explicitly::

        with metalins.trace(agent) as t:
            t.set_input(messages)
            response = client.messages.create(...)
            t.set_output(response)

    Or rely on the defaults: if ``set_input`` is never called the raw call
    to ``messages.create`` captures nothing as input — pass the messages list
    to ``set_input`` before calling the API.  If ``set_output`` is never called
    the context manager logs an empty output string (you almost always want to
    call ``set_output``).

    These are intentionally simple attributes — no magic — so tests can
    inspect them directly.
    """

    def __init__(self) -> None:
        self._input: str = ""
        self._output: str = ""
        self._input_set: bool = False
        self._output_set: bool = False

    def set_input(self, value: Any) -> None:
        """Capture the input for this turn."""
        self._input = _stringify(value)
        self._input_set = True

    def set_output(self, value: Any) -> None:
        """Capture the output for this turn."""
        self._output = _extract_text_from_message(value)
        self._output_set = True

    @property
    def input(self) -> str:
        return self._input

    @property
    def output(self) -> str:
        return self._output


# ---------------------------------------------------------------------------
# Context manager — metalins.trace(agent)
# ---------------------------------------------------------------------------

class AnthropicTrace:
    """Context manager that logs one Metalins event for an Anthropic turn.

    Instantiate via ``metalins.trace(agent)``, not directly.
    """

    def __init__(
        self,
        agent: "Agent",
        *,
        metadata: dict[str, Any] | None = None,
        raise_on_error: bool = False,
    ) -> None:
        self._agent = agent
        self._metadata = metadata
        self._raise_on_error = raise_on_error
        self._ctx = TraceContext()
        self._failed = False

    def __enter__(self) -> TraceContext:
        return self._ctx

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        # If the block raised we do not log — a failed turn has no clean
        # input/output pair, same as the LangChain/FastAPI handlers.
        if exc_type is not None:
            return  # propagate the exception
        self._flush()

    async def __aenter__(self) -> TraceContext:
        return self._ctx

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if exc_type is not None:
            return
        self._flush()

    def _flush(self) -> None:
        try:
            self._agent.log(
                input=self._ctx.input,
                output=self._ctx.output,
                metadata=self._metadata,
            )
        except Exception:
            if self._raise_on_error:
                raise


# ---------------------------------------------------------------------------
# Decorator — metalins.monitor(agent)
# ---------------------------------------------------------------------------

class AnthropicMonitor:
    """Decorator factory that logs one Metalins event per function call.

    Use ``metalins.monitor(agent)`` as a decorator::

        @metalins.monitor(agent)
        async def run(messages): ...

    The decorated function's first positional argument is logged as the
    *input* and its return value as the *output*.  Works for both sync and
    async functions.
    """

    def __init__(
        self,
        agent: "Agent",
        *,
        metadata: dict[str, Any] | None = None,
        raise_on_error: bool = False,
    ) -> None:
        self._agent = agent
        self._metadata = metadata
        self._raise_on_error = raise_on_error

    def __call__(self, fn: Callable) -> Callable:
        if asyncio.iscoroutinefunction(fn):
            return self._wrap_async(fn)
        return self._wrap_sync(fn)

    def _wrap_sync(self, fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            input_repr = _stringify(args[0]) if args else ""
            result = fn(*args, **kwargs)  # let exceptions propagate
            self._flush(input_repr, result)
            return result

        return wrapper

    def _wrap_async(self, fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            input_repr = _stringify(args[0]) if args else ""
            result = await fn(*args, **kwargs)  # let exceptions propagate
            self._flush(input_repr, result)
            return result

        return wrapper

    def _flush(self, input_repr: str, result: Any) -> None:
        try:
            self._agent.log(
                input=input_repr,
                output=_extract_text_from_message(result),
                metadata=self._metadata,
            )
        except Exception:
            if self._raise_on_error:
                raise


# ---------------------------------------------------------------------------
# Module-level convenience helpers (re-exported from metalins namespace)
# ---------------------------------------------------------------------------

def trace(
    agent: "Agent",
    *,
    metadata: dict[str, Any] | None = None,
    raise_on_error: bool = False,
) -> AnthropicTrace:
    """Return a context manager that logs one Metalins event for an Anthropic turn.

    Works as both a sync and async context manager::

        # sync
        with metalins.trace(agent) as t:
            t.set_input(messages)
            response = client.messages.create(...)
            t.set_output(response)

        # async
        async with metalins.trace(agent) as t:
            t.set_input(messages)
            response = await async_client.messages.create(...)
            t.set_output(response)

    Args:
        agent: A :class:`metalins.Agent` instance (or any object with a
            ``log(input, output, metadata=None)`` method).
        metadata: Optional dict merged into the Metalins event metadata.
        raise_on_error: If ``True``, re-raise exceptions from ``agent.log``.
            Default is ``False`` — logging failures are silently swallowed so
            a verification hiccup never breaks the application.

    Returns:
        An :class:`AnthropicTrace` context manager that yields a
        :class:`TraceContext`.
    """
    return AnthropicTrace(agent, metadata=metadata, raise_on_error=raise_on_error)


def monitor(
    agent: "Agent",
    *,
    metadata: dict[str, Any] | None = None,
    raise_on_error: bool = False,
) -> AnthropicMonitor:
    """Return a decorator that logs one Metalins event per function call.

    Wrap any function (sync or async) whose first argument is the input to
    the LLM and whose return value is the LLM output::

        @metalins.monitor(agent)
        async def process_refund(messages: list[dict]) -> str:
            response = await client.messages.create(...)
            return response.content[0].text

    Args:
        agent: A :class:`metalins.Agent` instance (or any object with a
            ``log(input, output, metadata=None)`` method).
        metadata: Optional dict merged into the Metalins event metadata.
        raise_on_error: If ``True``, re-raise exceptions from ``agent.log``.
            Default is ``False``.

    Returns:
        A decorator (:class:`AnthropicMonitor` instance).
    """
    return AnthropicMonitor(agent, metadata=metadata, raise_on_error=raise_on_error)
