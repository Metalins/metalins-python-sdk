"""Tests for the Anthropic SDK integration.

All Anthropic SDK objects are mocked — the real ``anthropic`` package is not
needed to run these tests.

Coverage:
  - ``metalins.trace`` context manager (sync and async)
  - ``metalins.monitor`` decorator (sync and async)
  - Input / output capture via set_input / set_output
  - Automatic text extraction from Anthropic Message objects
  - _stringify handles str, bytes, dict, list, pydantic-like objects
  - Error in traced block → no log call (exception propagates)
  - Error in agent.log() → swallowed by default, raised when opted-in
  - metadata forwarded to agent.log()
"""
from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from metalins.integrations.anthropic import (
    AnthropicTrace,
    AnthropicMonitor,
    TraceContext,
    _stringify,
    _extract_text_from_message,
    trace,
    monitor,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeAgent:
    """Stand-in for Agent — records log() calls without any network I/O."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def log(
        self,
        input: str | bytes,
        output: str | bytes,
        metadata: dict | None = None,
    ) -> dict:
        self.calls.append({"input": input, "output": output, "metadata": metadata})
        return {}


def _make_message(text: str = "Hello from Claude") -> MagicMock:
    """Return a MagicMock that looks like an Anthropic Message."""
    block = MagicMock()
    block.text = text
    msg = MagicMock()
    msg.content = [block]
    # Expose model_dump so _stringify can serialise it
    msg.model_dump.return_value = {"content": [{"text": text, "type": "text"}]}
    return msg


# ---------------------------------------------------------------------------
# _stringify
# ---------------------------------------------------------------------------

def test_stringify_str_passthrough():
    assert _stringify("hello") == "hello"


def test_stringify_bytes_decoded():
    assert _stringify(b"hello") == "hello"


def test_stringify_dict_sorted_json():
    result = _stringify({"b": 2, "a": 1})
    assert result == json.dumps({"a": 1, "b": 2}, sort_keys=True)


def test_stringify_list():
    result = _stringify([1, 2, 3])
    assert result == "[1, 2, 3]"


def test_stringify_none_empty_string():
    assert _stringify(None) == ""


def test_stringify_object_with_model_dump():
    obj = MagicMock()
    obj.model_dump.return_value = {"key": "value"}
    result = _stringify(obj)
    assert '"key"' in result
    assert '"value"' in result


def test_stringify_object_with_dict_method():
    class _Obj:
        def dict(self):
            return {"x": 42}

    result = _stringify(_Obj())
    assert '"x"' in result and "42" in result


# ---------------------------------------------------------------------------
# _extract_text_from_message
# ---------------------------------------------------------------------------

def test_extract_text_from_anthropic_message():
    msg = _make_message("I can help with that!")
    result = _extract_text_from_message(msg)
    assert result == "I can help with that!"


def test_extract_text_multiple_blocks():
    b1, b2 = MagicMock(), MagicMock()
    b1.text, b2.text = "Part 1", "Part 2"
    msg = MagicMock()
    msg.content = [b1, b2]
    msg.model_dump.return_value = {}
    result = _extract_text_from_message(msg)
    assert result == "Part 1\nPart 2"


def test_extract_text_no_text_attr_falls_back_to_stringify():
    msg = MagicMock()
    msg.content = []
    msg.model_dump.return_value = {"id": "msg_123"}
    result = _extract_text_from_message(msg)
    assert '"id"' in result


def test_extract_text_plain_string():
    assert _extract_text_from_message("raw string") == "raw string"


# ---------------------------------------------------------------------------
# TraceContext
# ---------------------------------------------------------------------------

def test_trace_context_defaults_empty():
    ctx = TraceContext()
    assert ctx.input == ""
    assert ctx.output == ""


def test_trace_context_set_input():
    ctx = TraceContext()
    ctx.set_input([{"role": "user", "content": "hi"}])
    assert '"role"' in ctx.input
    assert '"user"' in ctx.input


def test_trace_context_set_output_with_message():
    ctx = TraceContext()
    msg = _make_message("Great idea!")
    ctx.set_output(msg)
    assert ctx.output == "Great idea!"


# ---------------------------------------------------------------------------
# AnthropicTrace — sync context manager
# ---------------------------------------------------------------------------

def test_trace_logs_input_and_output():
    agent = _FakeAgent()
    msg = _make_message("I'll help you with that.")

    with trace(agent) as t:
        t.set_input([{"role": "user", "content": "hello"}])
        t.set_output(msg)

    assert len(agent.calls) == 1
    call = agent.calls[0]
    assert '"user"' in call["input"]
    assert call["output"] == "I'll help you with that."


def test_trace_no_log_on_exception():
    agent = _FakeAgent()

    with pytest.raises(ValueError, match="boom"):
        with trace(agent) as t:
            t.set_input("something")
            raise ValueError("boom")

    assert agent.calls == []


def test_trace_swallows_log_error_by_default():
    class _BoomAgent:
        def log(self, **kwargs):
            raise RuntimeError("backend down")

    # Should not raise — log failure is silenced
    with trace(_BoomAgent()) as t:
        t.set_input("x")
        t.set_output("y")


def test_trace_raises_log_error_when_opted_in():
    class _BoomAgent:
        def log(self, **kwargs):
            raise RuntimeError("backend down")

    with pytest.raises(RuntimeError, match="backend down"):
        with trace(_BoomAgent(), raise_on_error=True) as t:
            t.set_input("x")
            t.set_output("y")


def test_trace_forwards_metadata():
    agent = _FakeAgent()
    meta = {"agent_id": "refund-agent", "env": "prod"}

    with trace(agent, metadata=meta) as t:
        t.set_input("hello")
        t.set_output("world")

    assert agent.calls[0]["metadata"] == meta


def test_trace_empty_input_output_still_logs():
    """Even if set_input/set_output are never called, one event is logged."""
    agent = _FakeAgent()

    with trace(agent):
        pass  # no set_input or set_output

    assert len(agent.calls) == 1
    assert agent.calls[0]["input"] == ""
    assert agent.calls[0]["output"] == ""


# ---------------------------------------------------------------------------
# AnthropicTrace — async context manager
# ---------------------------------------------------------------------------

def test_async_trace_logs_input_and_output():
    agent = _FakeAgent()
    msg = _make_message("Async reply!")

    async def run():
        async with trace(agent) as t:
            t.set_input([{"role": "user", "content": "async hello"}])
            t.set_output(msg)

    asyncio.run(run())

    assert len(agent.calls) == 1
    assert agent.calls[0]["output"] == "Async reply!"


def test_async_trace_no_log_on_exception():
    agent = _FakeAgent()

    async def run():
        with pytest.raises(RuntimeError):
            async with trace(agent) as t:
                t.set_input("something")
                raise RuntimeError("async boom")

    asyncio.run(run())
    assert agent.calls == []


# ---------------------------------------------------------------------------
# AnthropicMonitor — sync decorator
# ---------------------------------------------------------------------------

def test_monitor_sync_logs_first_arg_and_return():
    agent = _FakeAgent()

    @monitor(agent)
    def process(messages: list) -> str:
        return "Done"

    result = process([{"role": "user", "content": "process this"}])
    assert result == "Done"
    assert len(agent.calls) == 1
    call = agent.calls[0]
    assert '"user"' in call["input"]
    assert call["output"] == "Done"


def test_monitor_sync_no_log_on_exception():
    agent = _FakeAgent()

    @monitor(agent)
    def fail(messages: list) -> str:
        raise ValueError("sync fail")

    with pytest.raises(ValueError, match="sync fail"):
        fail([{"role": "user", "content": "hi"}])

    assert agent.calls == []


def test_monitor_sync_no_args_uses_empty_input():
    agent = _FakeAgent()

    @monitor(agent)
    def no_args() -> str:
        return "result"

    result = no_args()
    assert result == "result"
    assert agent.calls[0]["input"] == ""


def test_monitor_sync_with_anthropic_message_return():
    agent = _FakeAgent()
    msg = _make_message("Response text")

    @monitor(agent)
    def call_claude(messages: list):
        return msg

    call_claude([{"role": "user", "content": "hello"}])
    assert agent.calls[0]["output"] == "Response text"


# ---------------------------------------------------------------------------
# AnthropicMonitor — async decorator
# ---------------------------------------------------------------------------

def test_monitor_async_logs_first_arg_and_return():
    agent = _FakeAgent()

    @monitor(agent)
    async def process(messages: list) -> str:
        return "Async done"

    result = asyncio.run(process([{"role": "user", "content": "async request"}]))
    assert result == "Async done"
    assert len(agent.calls) == 1
    call = agent.calls[0]
    assert '"user"' in call["input"]
    assert call["output"] == "Async done"


def test_monitor_async_no_log_on_exception():
    agent = _FakeAgent()

    @monitor(agent)
    async def fail(messages: list) -> str:
        raise RuntimeError("async fail")

    with pytest.raises(RuntimeError, match="async fail"):
        asyncio.run(fail([{"role": "user", "content": "hi"}]))

    assert agent.calls == []


def test_monitor_async_with_anthropic_message_return():
    agent = _FakeAgent()
    msg = _make_message("Async Claude response")

    @monitor(agent)
    async def call_claude(messages: list):
        return msg

    asyncio.run(call_claude([{"role": "user", "content": "hello"}]))
    assert agent.calls[0]["output"] == "Async Claude response"


# ---------------------------------------------------------------------------
# Monitor — log error handling
# ---------------------------------------------------------------------------

def test_monitor_swallows_log_error_by_default():
    class _BoomAgent:
        def log(self, **kwargs):
            raise RuntimeError("backend down")

    @monitor(_BoomAgent())
    def process(messages):
        return "ok"

    result = process(["hi"])
    assert result == "ok"  # application still works


def test_monitor_raises_log_error_when_opted_in():
    class _BoomAgent:
        def log(self, **kwargs):
            raise RuntimeError("backend down")

    @monitor(_BoomAgent(), raise_on_error=True)
    def process(messages):
        return "ok"

    with pytest.raises(RuntimeError, match="backend down"):
        process(["hi"])


def test_monitor_forwards_metadata():
    agent = _FakeAgent()
    meta = {"service": "refund-agent"}

    @monitor(agent, metadata=meta)
    def process(messages):
        return "ok"

    process(["hi"])
    assert agent.calls[0]["metadata"] == meta


# ---------------------------------------------------------------------------
# Preserves function metadata through wrapping
# ---------------------------------------------------------------------------

def test_monitor_preserves_function_name_and_docstring():
    agent = _FakeAgent()

    @monitor(agent)
    def my_func(messages):
        """My docstring."""
        return "ok"

    assert my_func.__name__ == "my_func"
    assert my_func.__doc__ == "My docstring."


def test_monitor_async_preserves_function_name():
    agent = _FakeAgent()

    @monitor(agent)
    async def my_async_func(messages):
        """Async docstring."""
        return "ok"

    assert my_async_func.__name__ == "my_async_func"
    assert my_async_func.__doc__ == "Async docstring."
