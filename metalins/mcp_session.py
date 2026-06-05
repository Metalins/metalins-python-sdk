"""AgentSession — client-side state for the streaming verification protocol.

A session holds:
  - agent_id (logical identifier shared with the server)
  - agent_secret (32-byte secret returned at register, never re-derivable)
  - digest_history[event_count] -> sha256 hex (mirrors the server's chain)
  - event_count (latest)

The session mirrors the server's running hash chain so the client can
answer the server's verification checks. For a check targeting a past
event the client recomputes
  answer = sha256(local_digest[target_t] || nonce || agent_secret)
which the server matches against its own value. A client that does not
hold the chain cannot produce the right answer.

Persistence is the caller's responsibility — `to_dict()` / `from_dict()`
round-trip the state through any store.
"""
from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from metalins.client import Client


# The fields a well-formed verification check is allowed to carry. A
# check carrying any field outside this set is treated as malformed and
# declined (see `AgentSession.answer_check`) rather than answered —
# answering an unrecognized field could expose secret material. If the
# server adds a new legitimate field it must be added here in lockstep.
_KNOWN_CHECK_FIELDS = frozenset(
    {"check_id", "target_event_count", "nonce", "issued_at", "expires_at"}
)

# A well-formed check nonce is 16 random bytes — 32 hex characters. A
# check whose nonce is shorter than this (or not hexadecimal) is
# malformed: `answer_check` declines it rather than computing a proof
# over attacker-shaped input. Schema validation alone is not enough —
# a malformed check can carry only recognized field *names* while
# still putting an off-spec *value* in one of them.
_MIN_NONCE_HEX_LEN = 32


def _to_bytes(data: str | bytes) -> bytes:
    if isinstance(data, str):
        return data.encode("utf-8")
    return data


def _sha256_hex(data: str | bytes) -> str:
    return hashlib.sha256(_to_bytes(data)).hexdigest()


def compute_check_answer(
    digest_at_t_hex: str,
    nonce_hex: str,
    agent_secret_hex: str,
) -> str:
    """Compute a verification-check answer — must match the server exactly.

      answer = sha256(digest_bytes || nonce_bytes || secret_bytes)

    All inputs are hex strings; output is a hex string.
    """
    h = hashlib.sha256()
    h.update(bytes.fromhex(digest_at_t_hex))
    h.update(bytes.fromhex(nonce_hex))
    h.update(bytes.fromhex(agent_secret_hex))
    return h.hexdigest()


def derive_initial_digest(agent_secret_hex: str) -> str:
    """Reproduce the genesis digest the server seeds the chain with at
    registration:

      digest[0] = sha256(secret_bytes || b"init")

    The client derives it locally so a freshly registered AgentSession
    starts from the same chain head as the server.
    """
    h = hashlib.sha256()
    h.update(bytes.fromhex(agent_secret_hex))
    h.update(b"init")
    return h.hexdigest()


def derive_next_digest(prev_digest_hex: str, input_hash_hex: str, output_hash_hex: str) -> str:
    """Reproduce the server's hash chain update.

      next_digest = sha256(prev_digest_bytes || input_hash_hex_bytes || output_hash_hex_bytes)

    Note: the server feeds the hex *strings* as UTF-8 into the hash, not the
    decoded bytes. This is preserved here to keep the chains in sync.
    """
    h = hashlib.sha256()
    h.update(bytes.fromhex(prev_digest_hex))
    h.update(input_hash_hex.encode("utf-8"))
    h.update(output_hash_hex.encode("utf-8"))
    return h.hexdigest()


@dataclass
class AgentSession:
    """A local agent session for the streaming verification protocol.

    Constructed by `Client.start_session(...)`. Persist `to_dict()` so the
    agent survives a restart with its digest history intact — losing the
    history means the client can no longer answer checks that target
    events from before the loss.
    """
    agent_id: str
    agent_secret: str  # hex (32 bytes)
    event_count: int = 0
    digest_history: dict[int, str] = field(default_factory=dict)
    _client: "Client | None" = field(default=None, repr=False)
    # Guards the event_count / digest_history mutation so a background
    # poller answering a check never observes a half-applied chain
    # update. Never serialized (to_dict skips it; from_dict makes a fresh
    # one).
    _lock: Any = field(
        default_factory=threading.RLock, repr=False, compare=False
    )

    # ----------------------------------------------------------------- API

    def log_event(
        self,
        input_data: str | bytes,
        output_data: str | bytes,
        metadata: dict[str, Any] | None = None,
        auto_answer_checks: bool = True,
    ) -> dict:
        """Hash payloads, log to the server, update the local digest, and
        optionally auto-answer any pending verification checks the server
        returned.

        Returns the server's raw JSON response.
        """
        if self._client is None:
            raise RuntimeError("AgentSession has no client bound; use Client.attach_session().")

        input_hash = _sha256_hex(input_data)
        output_hash = _sha256_hex(output_data)

        response = self._client.log_event(
            self.agent_id,
            input_hash=input_hash,
            output_hash=output_hash,
            metadata=metadata or {},
        )

        # Update the local hash chain to mirror the server. Held under
        # the lock so a background poller answering a check never reads a
        # half-applied (event_count bumped, digest not yet stored) chain.
        with self._lock:
            prev_digest = self.digest_history.get(self.event_count)
            if prev_digest is None:
                # No prior digest — caller likely deserialized a partial
                # session. We cannot reconstruct without the previous
                # state; fail loudly rather than silently feed zeros and
                # break the chain.
                raise RuntimeError(
                    f"Missing digest for event_count={self.event_count}. "
                    "Session state corrupted; re-register or restore from "
                    "a snapshot."
                )
            new_digest = derive_next_digest(
                prev_digest, input_hash, output_hash
            )
            self.event_count += 1
            self.digest_history[self.event_count] = new_digest

        if auto_answer_checks:
            for check in response.get("pending_checks") or []:
                try:
                    self.answer_check(check)
                except Exception:
                    # A check can fail (expired, etc.); never let that
                    # fail the log_event call itself.
                    pass

        return response

    def answer_check(self, check: dict, *, refuse_malformed: bool = True) -> dict:
        """Answer a single verification check.

        `check` is the dict shape the server surfaces under
        `pending_checks` (in the log_event response): check_id,
        target_event_count, nonce, issued_at, expires_at.

        Three deterministic behaviors — hashing only, no model involved:

        - If `refuse_malformed` is set (the default) the check is
          validated before anything is computed. A malformed check —
          one carrying a field outside `_KNOWN_CHECK_FIELDS`, OR a
          recognized field holding an off-spec value (a truncated /
          non-hex nonce, a target_event_count outside the agent's
          range) — is declined: the client submits a `decline_reason`
          and no answer. Answering a malformed check could coax a proof
          out of the agent over attacker-shaped input.
        - For a well-formed check: answer = sha256(digest_at_t ‖ nonce ‖
          agent_secret).
        - The client also reports `progress` — its event_count at answer
          time — so the server sees how far the agent had progressed
          when it replied.
        """
        if self._client is None:
            raise RuntimeError("AgentSession has no client bound.")

        check_id = check["check_id"]

        # Decline malformed checks — both unrecognized field *names* and
        # recognized fields carrying off-spec *values*.
        if refuse_malformed:
            unknown = sorted(set(check) - _KNOWN_CHECK_FIELDS)
            if unknown:
                return self._client.answer_check(
                    self.agent_id,
                    check_id,
                    decline_reason=(
                        "declined: check carries unrecognized field(s) "
                        f"{unknown} — outside the known check schema"
                    ),
                )
            bad_value = self._malformed_value_reason(check)
            if bad_value:
                return self._client.answer_check(
                    self.agent_id,
                    check_id,
                    decline_reason=f"declined: {bad_value}",
                )

        target_t = int(check["target_event_count"])
        nonce = check["nonce"]

        # Snapshot the digest + our progress atomically vs. log_event.
        with self._lock:
            digest_at_t = self.digest_history.get(target_t)
            progress = self.event_count

        if digest_at_t is None:
            # We do not hold this digest — submit a deliberately wrong
            # answer so the check fails honestly rather than silently.
            answer = compute_check_answer("00" * 32, nonce, self.agent_secret)
        else:
            answer = compute_check_answer(digest_at_t, nonce, self.agent_secret)

        return self._client.answer_check(
            self.agent_id,
            check_id,
            answer=answer,
            progress=progress,
        )

    def _malformed_value_reason(self, check: dict) -> str | None:
        """Return why a check's *values* are malformed, or None if they
        are plausible.

        Field *names* are validated separately against
        `_KNOWN_CHECK_FIELDS`. This catches the other half: a recognized
        field carrying an off-spec value — a truncated or non-hex nonce,
        or a target_event_count outside the range the agent could
        plausibly have produced.
        """
        nonce = check.get("nonce")
        if not isinstance(nonce, str) or len(nonce) < _MIN_NONCE_HEX_LEN:
            return (
                "nonce is too short — expected at least "
                f"{_MIN_NONCE_HEX_LEN} hex characters"
            )
        try:
            int(nonce, 16)
        except ValueError:
            return "nonce is not valid hexadecimal"

        try:
            target_t = int(check.get("target_event_count"))
        except (TypeError, ValueError):
            return "target_event_count is not an integer"
        with self._lock:
            progress = self.event_count
        if target_t < 1 or target_t > progress:
            return (
                f"target_event_count {target_t} is outside this agent's "
                f"range [1, {progress}]"
            )
        return None

    def get_status(self) -> dict:
        """Read this agent's verification status from the server."""
        if self._client is None:
            raise RuntimeError("AgentSession has no client bound.")
        return self._client.get_agent(self.agent_id)

    def issue_proof(self, *, ttl_seconds: int = 3600, scope: str | None = None) -> dict:
        """Issue a signed identity proof for this agent."""
        if self._client is None:
            raise RuntimeError("AgentSession has no client bound.")
        return self._client.issue_proof(
            self.agent_id, ttl_seconds=ttl_seconds, scope=scope
        )

    # ------------------------------------------------------------- Persist

    def to_dict(self) -> dict:
        """Snapshot the session for storage. JSON-safe.

        Keys in digest_history are converted to strings (JSON limitation);
        `from_dict` converts them back to ints.
        """
        return {
            "agent_id": self.agent_id,
            "agent_secret": self.agent_secret,
            "event_count": self.event_count,
            "digest_history": {str(k): v for k, v in self.digest_history.items()},
        }

    @classmethod
    def from_dict(cls, data: dict, client: "Client | None" = None) -> "AgentSession":
        """Rehydrate a session from a snapshot. Pass `client` to bind it."""
        return cls(
            agent_id=data["agent_id"],
            agent_secret=data["agent_secret"],
            event_count=int(data.get("event_count", 0)),
            digest_history={int(k): v for k, v in data.get("digest_history", {}).items()},
            _client=client,
        )

    def bind_client(self, client: "Client") -> "AgentSession":
        self._client = client
        return self
