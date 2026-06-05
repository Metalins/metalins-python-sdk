"""Tests for the SDK V2 surface — StateStore, CheckWorker, the Agent
facade, and the LangChain integration (HTTP mocked with respx)."""
import json
import os
import stat
from uuid import uuid4

import pytest
import respx
from httpx import Response

import metalins
from metalins.state import FileStateStore, default_state_path
from metalins.worker import CheckWorker

BASE = "http://api.test"
SECRET = "ab" * 32


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

class MemStore:
    """An in-memory StateStore for tests — implements the protocol."""

    def __init__(self, initial=None):
        self.data = initial

    def load(self):
        return self.data

    def save(self, state):
        self.data = dict(state)


def _register_route(agent_id="agt_x", secret=SECRET):
    return respx.post(f"{BASE}/v1/agents").mock(
        return_value=Response(201, json={
            "agent_id": agent_id,
            "agent_secret": secret,
            "created_at": "2026-05-21T10:00:00Z",
            "secret_warning": "store it",
        })
    )


# --------------------------------------------------------------------------- #
# FileStateStore                                                              #
# --------------------------------------------------------------------------- #

def test_file_store_roundtrip(tmp_path):
    store = FileStateStore(tmp_path / "sub" / "agent.json")
    assert store.load() is None  # nothing saved yet

    state = {"agent_id": "agt_1", "agent_secret": SECRET, "event_count": 2}
    store.save(state)
    assert store.load() == state


def test_file_store_file_is_owner_only(tmp_path):
    store = FileStateStore(tmp_path / "agent.json")
    store.save({"agent_id": "agt_1"})
    mode = stat.S_IMODE(os.stat(tmp_path / "agent.json").st_mode)
    assert mode == 0o600, f"expected 0600, got {oct(mode)}"


def test_file_store_atomic_no_temp_left(tmp_path):
    store = FileStateStore(tmp_path / "agent.json")
    store.save({"a": 1})
    store.save({"a": 2})
    # Only the target file — no stray temp files from the atomic write.
    assert sorted(p.name for p in tmp_path.iterdir()) == ["agent.json"]
    assert store.load() == {"a": 2}


def test_default_state_path_slugifies():
    p = default_state_path("My Prod Bot!")
    assert p.name == "my-prod-bot.json"
    assert p.parent.name == ".metalins"


# --------------------------------------------------------------------------- #
# Client.list_pending_checks                                                  #
# --------------------------------------------------------------------------- #

@respx.mock
def test_list_pending_checks():
    client = metalins.Client(api_key="ml_test", base_url=BASE)
    check = {
        "check_id": "chk_1",
        "target_event_count": 1,
        "nonce": "cd" * 16,
        "issued_at": "2026-05-21T10:00:00Z",
        "expires_at": "2026-05-21T10:05:00Z",
    }
    respx.get(f"{BASE}/v1/agents/agt_x/checks").mock(
        return_value=Response(200, json={"agent_id": "agt_x", "checks": [check]})
    )
    checks = client.list_pending_checks("agt_x")
    assert checks == [check]


# --------------------------------------------------------------------------- #
# Agent facade — register / resume / log                                      #
# --------------------------------------------------------------------------- #

@respx.mock
def test_agent_registers_on_first_run_and_persists():
    store = MemStore()
    reg = _register_route()
    agent = metalins.Agent(
        api_key="ml_test", name="bot", base_url=BASE, store=store
    )
    assert reg.called, "first run must register the agent"
    assert agent.agent_id == "agt_x"
    # The session — secret included — was persisted for the next run.
    assert store.data is not None
    assert store.data["agent_id"] == "agt_x"
    assert store.data["agent_secret"] == SECRET


@respx.mock
def test_agent_resumes_from_store_without_registering():
    # Store already holds a session from an earlier run.
    saved = {
        "agent_id": "agt_saved",
        "agent_secret": SECRET,
        "event_count": 0,
        "digest_history": {
            "0": metalins.derive_initial_digest(SECRET),
        },
    }
    store = MemStore(initial=saved)
    reg = _register_route()
    agent = metalins.Agent(
        api_key="ml_test", name="bot", base_url=BASE, store=store
    )
    assert not reg.called, "a resumed agent must not re-register"
    assert agent.agent_id == "agt_saved"


@respx.mock
def test_agent_attach_adopts_existing_without_registering():
    store = MemStore()
    reg = _register_route()
    agent = metalins.Agent.attach(
        api_key="ml_test",
        agent_id="agt_existing",
        agent_secret=SECRET,
        base_url=BASE,
        store=store,
    )
    assert not reg.called, "attach must not register a new agent"
    assert agent.agent_id == "agt_existing"
    # Session seeded at genesis from the given id + secret, then persisted.
    assert store.data["agent_id"] == "agt_existing"
    assert store.data["agent_secret"] == SECRET
    assert (
        store.data["digest_history"]["0"]
        == metalins.derive_initial_digest(SECRET)
    )


@respx.mock
def test_agent_attach_resumes_from_store():
    saved = {
        "agent_id": "agt_existing",
        "agent_secret": SECRET,
        "event_count": 3,
        "digest_history": {"0": metalins.derive_initial_digest(SECRET)},
    }
    store = MemStore(initial=saved)
    reg = _register_route()
    agent = metalins.Agent.attach(
        api_key="ml_test",
        agent_id="agt_existing",
        agent_secret=SECRET,
        base_url=BASE,
        store=store,
    )
    assert not reg.called
    assert agent.session.event_count == 3  # resumed, not re-seeded


def test_agent_requires_name_or_attach():
    # Constructing Agent with neither a name nor an adoption pair is an
    # error — there is nothing to register and nothing to adopt.
    with pytest.raises(ValueError):
        metalins.Agent(api_key="ml_test", base_url=BASE, store=MemStore())


@respx.mock
def test_agent_log_advances_and_persists():
    store = MemStore()
    _register_route()
    respx.post(f"{BASE}/v1/agents/agt_x/events").mock(
        return_value=Response(200, json={
            "agent_id": "agt_x", "event_count": 1, "pending_checks": [],
        })
    )
    agent = metalins.Agent(
        api_key="ml_test", name="bot", base_url=BASE, store=store
    )
    agent.log(input="user asked", output="agent answered")
    assert agent.session.event_count == 1
    # The advanced chain was persisted — a restart would resume at 1.
    assert store.data["event_count"] == 1
    assert "1" in store.data["digest_history"]


# --------------------------------------------------------------------------- #
# CheckWorker                                                                 #
# --------------------------------------------------------------------------- #

@respx.mock
def test_check_worker_poll_once_answers_pending():
    store = MemStore()
    _register_route()
    respx.post(f"{BASE}/v1/agents/agt_x/events").mock(
        return_value=Response(200, json={
            "agent_id": "agt_x", "event_count": 1, "pending_checks": [],
        })
    )
    agent = metalins.Agent(
        api_key="ml_test", name="bot", base_url=BASE, store=store
    )
    # Log once so the agent holds digest[1] — the check below targets it.
    agent.log(input="q", output="a")

    nonce = "cd" * 16
    respx.get(f"{BASE}/v1/agents/agt_x/checks").mock(
        return_value=Response(200, json={"agent_id": "agt_x", "checks": [{
            "check_id": "chk_1",
            "target_event_count": 1,
            "nonce": nonce,
            "issued_at": "2026-05-21T10:00:00Z",
            "expires_at": "2026-05-21T10:05:00Z",
        }]})
    )
    answer_route = respx.post(f"{BASE}/v1/agents/agt_x/checks/chk_1").mock(
        return_value=Response(200, json={
            "check_id": "chk_1", "accepted": True, "detail": None,
        })
    )

    answered = agent.poll_now()
    assert answered == 1
    assert answer_route.called
    sent = json.loads(answer_route.calls[0].request.content)
    expected = metalins.compute_check_answer(
        agent.session.digest_history[1], nonce, SECRET
    )
    assert sent["answer"] == expected


@respx.mock
def test_check_worker_start_stop_lifecycle():
    store = MemStore()
    _register_route()
    respx.get(f"{BASE}/v1/agents/agt_x/checks").mock(
        return_value=Response(200, json={"agent_id": "agt_x", "checks": []})
    )
    agent = metalins.Agent(
        api_key="ml_test", name="bot", base_url=BASE, store=store,
        poll_interval=0.05,
    )
    worker = agent._worker
    assert not worker.running
    agent.start()
    assert worker.running
    agent.stop()
    assert not worker.running


def test_check_worker_rejects_bad_interval():
    with pytest.raises(ValueError):
        CheckWorker(session=None, interval=0)


# --------------------------------------------------------------------------- #
# LangChain integration                                                       #
# --------------------------------------------------------------------------- #

class _FakeAgent:
    """Stand-in for Agent — records log() calls without any HTTP."""

    def __init__(self):
        self.calls = []

    def log(self, input, output, metadata=None):
        self.calls.append({"input": input, "output": output})
        return {}


def test_langchain_handler_logs_top_level_chain():
    pytest.importorskip("langchain_core")
    from metalins.integrations.langchain import MetalinsCallbackHandler

    agent = _FakeAgent()
    handler = MetalinsCallbackHandler(agent)

    run_id = uuid4()
    handler.on_chain_start({}, {"question": "hi"}, run_id=run_id, parent_run_id=None)
    handler.on_chain_end({"answer": "hello"}, run_id=run_id, parent_run_id=None)

    assert len(agent.calls) == 1
    assert "hi" in agent.calls[0]["input"]
    assert "hello" in agent.calls[0]["output"]


def test_langchain_handler_ignores_nested_runs():
    pytest.importorskip("langchain_core")
    from metalins.integrations.langchain import MetalinsCallbackHandler

    agent = _FakeAgent()
    handler = MetalinsCallbackHandler(agent)

    parent, child = uuid4(), uuid4()
    # A nested step (parent_run_id set) must not produce its own event.
    handler.on_chain_start({}, {"x": 1}, run_id=child, parent_run_id=parent)
    handler.on_chain_end({"y": 2}, run_id=child, parent_run_id=parent)
    assert agent.calls == []


def test_langchain_handler_swallows_log_errors():
    pytest.importorskip("langchain_core")
    from metalins.integrations.langchain import MetalinsCallbackHandler

    class _BoomAgent:
        def log(self, **kwargs):
            raise RuntimeError("network down")

    handler = MetalinsCallbackHandler(_BoomAgent())
    run_id = uuid4()
    handler.on_chain_start({}, {"q": "x"}, run_id=run_id, parent_run_id=None)
    # A logging failure must not propagate into the host chain.
    handler.on_chain_end({"a": "y"}, run_id=run_id, parent_run_id=None)
