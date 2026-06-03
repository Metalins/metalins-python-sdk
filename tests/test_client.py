"""Tests for the Metalins SDK Client — a thin wrapper over the
developer API (mocked HTTP)."""
import hashlib
import json

import pytest
import respx
from httpx import Response

import metalins
from metalins.errors import AgentNotFound, AuthenticationError

BASE = "http://api.test"


@pytest.fixture
def client():
    return metalins.Client(api_key="ml_test_xxx", base_url=BASE)


def test_requires_api_key():
    with pytest.raises(AuthenticationError):
        metalins.Client(api_key="")


# --------------------------------------------------------------------------- #
# One method per developer-API endpoint                                       #
# --------------------------------------------------------------------------- #

@respx.mock
def test_create_agent(client):
    respx.post(f"{BASE}/v1/agents").mock(
        return_value=Response(201, json={
            "agent_id": "agt_abc",
            "agent_secret": "ab" * 32,
            "created_at": "2026-05-21T10:00:00Z",
            "secret_warning": "store it now",
        })
    )
    data = client.create_agent(name="my-bot", model="claude-sonnet")
    assert data["agent_id"] == "agt_abc"
    assert data["agent_secret"] == "ab" * 32


@respx.mock
def test_log_event(client):
    respx.post(f"{BASE}/v1/agents/agt_abc/events").mock(
        return_value=Response(200, json={
            "agent_id": "agt_abc",
            "event_count": 1,
            "pending_checks": [],
        })
    )
    data = client.log_event("agt_abc", input_hash="aa", output_hash="bb")
    assert data["event_count"] == 1


@respx.mock
def test_answer_check(client):
    respx.post(f"{BASE}/v1/agents/agt_abc/checks/chk_1").mock(
        return_value=Response(200, json={
            "check_id": "chk_1", "accepted": True, "detail": None,
        })
    )
    data = client.answer_check("agt_abc", "chk_1", answer="deadbeef", progress=3)
    assert data["accepted"] is True


@respx.mock
def test_list_and_get_agent(client):
    respx.get(f"{BASE}/v1/agents").mock(
        return_value=Response(200, json={
            "agents": [{"agent_id": "agt_abc"}], "count": 1,
        })
    )
    respx.get(f"{BASE}/v1/agents/agt_abc").mock(
        return_value=Response(200, json={"agent_id": "agt_abc", "tier": "T1"})
    )
    assert client.list_agents()["count"] == 1
    assert client.get_agent("agt_abc")["tier"] == "T1"


@respx.mock
def test_issue_proof(client):
    respx.post(f"{BASE}/v1/agents/agt_abc/proofs").mock(
        return_value=Response(201, json={
            "proof_id": "prf_1",
            "agent_id": "agt_abc",
            "proof": "eyJ.fake",
            "issued_at": "2026-05-21T10:00:00Z",
            "expires_at": "2026-05-21T11:00:00Z",
            "scope": None,
        })
    )
    data = client.issue_proof("agt_abc", ttl_seconds=3600)
    assert data["proof_id"] == "prf_1"


@respx.mock
def test_revoke_agent(client):
    respx.delete(f"{BASE}/v1/agents/agt_abc").mock(
        return_value=Response(200, json={
            "agent_id": "agt_abc", "revoked_at": "2026-05-21T10:00:00Z",
        })
    )
    data = client.revoke_agent("agt_abc", reason="done")
    assert data["agent_id"] == "agt_abc"


@respx.mock
def test_agent_not_found_raises(client):
    respx.get(f"{BASE}/v1/agents/agt_missing").mock(
        return_value=Response(404, json={"detail": "Agent not found"})
    )
    with pytest.raises(AgentNotFound):
        client.get_agent("agt_missing")


# --------------------------------------------------------------------------- #
# Streaming-session convenience layer                                         #
# --------------------------------------------------------------------------- #

@respx.mock
def test_start_session_derives_initial_digest(client):
    secret = "cd" * 32
    respx.post(f"{BASE}/v1/agents").mock(
        return_value=Response(201, json={
            "agent_id": "agt_sess",
            "agent_secret": secret,
            "created_at": "2026-05-21T10:00:00Z",
            "secret_warning": "x",
        })
    )
    session = client.start_session(name="streamer")
    expected = hashlib.sha256(bytes.fromhex(secret) + b"init").hexdigest()
    assert session.agent_id == "agt_sess"
    assert session.event_count == 0
    assert session.digest_history[0] == expected


@respx.mock
def test_session_log_event_auto_answers_check(client):
    """Full loop: register a session, log an event, auto-answer the
    verification check the server returns."""
    secret = "ef" * 32
    respx.post(f"{BASE}/v1/agents").mock(
        return_value=Response(201, json={
            "agent_id": "agt_loop",
            "agent_secret": secret,
            "created_at": "2026-05-21T10:00:00Z",
            "secret_warning": "x",
        })
    )
    session = client.start_session(name="loop")

    respx.post(f"{BASE}/v1/agents/agt_loop/events").mock(
        return_value=Response(200, json={
            "agent_id": "agt_loop",
            "event_count": 1,
            "pending_checks": [{
                "check_id": "chk_x",
                "target_event_count": 1,
                "nonce": "cd" * 16,
                "issued_at": "2026-05-21T10:00:00Z",
                "expires_at": "2026-05-21T10:05:00Z",
            }],
        })
    )
    answer_route = respx.post(f"{BASE}/v1/agents/agt_loop/checks/chk_x").mock(
        return_value=Response(200, json={
            "check_id": "chk_x", "accepted": True, "detail": None,
        })
    )

    session.log_event("user said hi", "agent replied")

    assert session.event_count == 1
    assert answer_route.called
    # The answer submitted must be the canonical value for digest[1].
    sent = json.loads(answer_route.calls[0].request.content)
    expected = metalins.compute_check_answer(
        session.digest_history[1], "cd" * 16, secret
    )
    assert sent["answer"] == expected
    assert sent["progress"] == 1
