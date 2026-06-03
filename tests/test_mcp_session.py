"""Unit tests for the client-side verification protocol primitives."""
from __future__ import annotations

import hashlib

from metalins.mcp_session import (
    AgentSession,
    compute_check_answer,
    derive_initial_digest,
    derive_next_digest,
)


# --------------------------------------------------------------------------- #
# Hash primitives must match the server byte-for-byte.                        #
# --------------------------------------------------------------------------- #

def test_compute_check_answer_matches_canonical_formula():
    digest = "ab" * 32
    nonce = "cd" * 16
    secret = "ef" * 32
    expected = hashlib.sha256(
        bytes.fromhex(digest) + bytes.fromhex(nonce) + bytes.fromhex(secret)
    ).hexdigest()
    assert compute_check_answer(digest, nonce, secret) == expected


def test_compute_check_answer_changes_with_any_input():
    base = compute_check_answer("ab" * 32, "cd" * 16, "ef" * 32)
    assert base != compute_check_answer("ac" * 32, "cd" * 16, "ef" * 32)
    assert base != compute_check_answer("ab" * 32, "00" * 16, "ef" * 32)
    assert base != compute_check_answer("ab" * 32, "cd" * 16, "ee" * 32)


def test_derive_initial_digest_matches_server_formula():
    """Server seeds digest[0] = sha256(secret_bytes || b'init')."""
    secret = "ab" * 32
    expected = hashlib.sha256(bytes.fromhex(secret) + b"init").hexdigest()
    assert derive_initial_digest(secret) == expected


def test_derive_next_digest_matches_server_formula():
    """Server feeds hex strings as UTF-8 — preserved client-side."""
    prev = hashlib.sha256(b"init").hexdigest()
    in_hash = hashlib.sha256(b"hello").hexdigest()
    out_hash = hashlib.sha256(b"world").hexdigest()

    h = hashlib.sha256()
    h.update(bytes.fromhex(prev))
    h.update(in_hash.encode("utf-8"))
    h.update(out_hash.encode("utf-8"))
    expected = h.hexdigest()

    assert derive_next_digest(prev, in_hash, out_hash) == expected


# --------------------------------------------------------------------------- #
# AgentSession state management                                               #
# --------------------------------------------------------------------------- #

def test_session_to_dict_from_dict_roundtrip():
    s = AgentSession(
        agent_id="agt_test",
        agent_secret="aa" * 32,
        event_count=3,
        digest_history={0: "00" * 32, 1: "11" * 32, 2: "22" * 32, 3: "33" * 32},
    )
    snapshot = s.to_dict()
    assert snapshot["agent_id"] == "agt_test"
    # JSON-safe keys are strings.
    assert all(isinstance(k, str) for k in snapshot["digest_history"])

    rehydrated = AgentSession.from_dict(snapshot)
    assert rehydrated.agent_id == s.agent_id
    assert rehydrated.agent_secret == s.agent_secret
    assert rehydrated.event_count == s.event_count
    # Keys should be ints again.
    assert all(isinstance(k, int) for k in rehydrated.digest_history)
    assert rehydrated.digest_history == s.digest_history


def test_session_raises_without_client():
    s = AgentSession(
        agent_id="x", agent_secret="aa" * 32, event_count=0,
        digest_history={0: "00" * 32},
    )
    try:
        s.log_event("a", "b")
    except RuntimeError as e:
        assert "no client" in str(e).lower()
    else:
        raise AssertionError("expected RuntimeError")


# --------------------------------------------------------------------------- #
# answer_check — declines malformed checks, answers well-formed ones          #
# --------------------------------------------------------------------------- #

class _RecordingClient:
    """Minimal stand-in for Client — records answer_check calls."""

    def __init__(self):
        self.calls: list[dict] = []

    def answer_check(self, agent_id, check_id, *, answer=None,
                     decline_reason=None, progress=None):
        self.calls.append({
            "agent_id": agent_id,
            "check_id": check_id,
            "answer": answer,
            "decline_reason": decline_reason,
            "progress": progress,
        })
        return {"check_id": check_id, "accepted": answer is not None}


def test_answer_check_declines_malformed_check():
    """A check carrying an unrecognized field is declined, not answered."""
    client = _RecordingClient()
    s = AgentSession(
        agent_id="a", agent_secret="aa" * 32, event_count=1,
        digest_history={0: "00" * 32, 1: "11" * 32}, _client=client,
    )
    s.answer_check({
        "check_id": "c1",
        "target_event_count": 1,
        "nonce": "cd" * 16,
        "requires_secret_reveal": True,
    })
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["answer"] is None
    assert call["decline_reason"] is not None
    assert "requires_secret_reveal" in call["decline_reason"]


def test_answer_check_answers_well_formed_check():
    client = _RecordingClient()
    history = {0: derive_initial_digest("aa" * 32)}
    history[1] = derive_next_digest(history[0], "be" * 32, "ef" * 32)
    s = AgentSession(
        agent_id="a", agent_secret="aa" * 32, event_count=1,
        digest_history=history, _client=client,
    )
    s.answer_check({
        "check_id": "c2", "target_event_count": 1, "nonce": "cd" * 16,
    })
    call = client.calls[0]
    expected = compute_check_answer(history[1], "cd" * 16, "aa" * 32)
    assert call["answer"] == expected
    assert call["progress"] == 1
    assert call["decline_reason"] is None


# --------------------------------------------------------------------------- #
# Honest-vs-clone simulation: compute answers against synthetic chain state   #
# --------------------------------------------------------------------------- #

def test_honest_answer_matches_when_clone_answer_does_not():
    """Build a digest chain; honest computes correctly, clone uses zeros."""
    secret = "ab" * 32
    history = {0: derive_initial_digest(secret)}
    for i in range(1, 10):
        in_hash = hashlib.sha256(f"in{i}".encode()).hexdigest()
        out_hash = hashlib.sha256(f"out{i}".encode()).hexdigest()
        history[i] = derive_next_digest(history[i - 1], in_hash, out_hash)

    target_t = 5
    nonce = "cd" * 16
    expected = compute_check_answer(history[target_t], nonce, secret)

    honest = compute_check_answer(history[target_t], nonce, secret)
    clone = compute_check_answer("00" * 32, nonce, secret)

    assert honest == expected
    assert clone != expected
