"""High-level agent facade — register once, then log and forget.

`Agent` is the one-import entry point for a long-lived agent. It owns
three pieces and hides their wiring:

  - an `AgentSession` — the verification state (id, secret, hash chain),
  - a `StateStore`   — so a process restart resumes the same agent,
  - a `CheckWorker`  — the background loop that answers verification
    checks even while the agent is idle.

A backend service adds one block at startup and one call per turn:

    import metalins

    agent = metalins.Agent(api_key="ml_live_...", name="my-bot")
    agent.start()
    ...
    agent.log(input=user_msg, output=agent_reply)
    ...
    agent.stop()

Or as a context manager, which starts and stops the loop for you:

    with metalins.Agent(api_key="ml_live_...", name="my-bot") as agent:
        agent.log(input=user_msg, output=agent_reply)
"""
from __future__ import annotations

from typing import Any

from metalins.client import DEFAULT_BASE_URL, Client
from metalins.mcp_session import AgentSession, derive_initial_digest
from metalins.state import FileStateStore, StateStore, default_state_path
from metalins.worker import CheckWorker


class Agent:
    """Register-or-resume an agent and run its verification loop.

    On first construction with a given name the agent is registered and
    its state (including the one-time secret) is persisted through the
    `StateStore`. Later constructions load that state and resume the
    same agent — no re-register. Pass an explicit `store` to keep the
    secret somewhere other than the default on-disk file.

    To connect to an agent that already exists — one created in the
    dashboard, or registered elsewhere — use `Agent.attach(...)` instead
    of constructing directly; it adopts that agent rather than
    registering a new one.
    """

    def __init__(
        self,
        api_key: str,
        name: str | None = None,
        *,
        base_url: str = DEFAULT_BASE_URL,
        store: StateStore | None = None,
        model: str | None = None,
        framework: str | None = None,
        metadata: dict[str, Any] | None = None,
        poll_interval: float = 30.0,
        _adopt: tuple[str, str] | None = None,
    ):
        # `_adopt` is (agent_id, agent_secret) — set by `Agent.attach`.
        # Without it, a `name` is required so a new agent can be
        # registered.
        if _adopt is None and not name:
            raise ValueError(
                "Agent(...) needs a name to register a new agent — or use "
                "Agent.attach(api_key, agent_id, agent_secret) to connect "
                "an agent that already exists."
            )
        self.name = name
        self._adopt = _adopt
        self._client = Client(api_key, base_url=base_url)
        store_key = name or (_adopt[0] if _adopt else "agent")
        self._store: StateStore = store or FileStateStore(
            default_state_path(store_key)
        )
        self._session = self._register_or_attach(
            model=model, framework=framework, metadata=metadata
        )
        self._worker = CheckWorker(self._session, interval=poll_interval)

    @classmethod
    def attach(
        cls,
        api_key: str,
        agent_id: str,
        agent_secret: str,
        *,
        name: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        store: StateStore | None = None,
        poll_interval: float = 30.0,
    ) -> "Agent":
        """Connect to an agent that already exists instead of registering
        a new one.

        Use this when the agent was created in the dashboard (or
        registered elsewhere): pass its `agent_id` and the `agent_secret`
        shown once when it was created. On first run the session is
        seeded from that pair at a fresh genesis and persisted; later
        runs resume from the store.

        Adopting assumes the agent has not yet logged events through
        another client — the SDK starts its hash chain from genesis.
        """
        return cls(
            api_key,
            name=name,
            base_url=base_url,
            store=store,
            poll_interval=poll_interval,
            _adopt=(agent_id, agent_secret),
        )

    def _register_or_attach(
        self,
        *,
        model: str | None,
        framework: str | None,
        metadata: dict[str, Any] | None,
    ) -> AgentSession:
        saved = self._store.load()
        if saved is not None:
            # Resume the agent persisted on an earlier run.
            return self._client.attach_session(AgentSession.from_dict(saved))
        if self._adopt is not None:
            # Adopt an existing agent — seed the session from its id and
            # secret at genesis, then persist so later runs resume.
            agent_id, secret = self._adopt
            session = AgentSession(
                agent_id=agent_id,
                agent_secret=secret,
                event_count=0,
                digest_history={0: derive_initial_digest(secret)},
                _client=self._client,
            )
            self._store.save(session.to_dict())
            return session
        # First run — register a fresh agent and persist its state
        # (including the one-time secret) so later runs can resume.
        session = self._client.start_session(
            name=self.name,
            model=model,
            framework=framework,
            metadata=metadata,
        )
        self._store.save(session.to_dict())
        return session

    # --------------------------------------------------------------- API

    @property
    def agent_id(self) -> str:
        return self._session.agent_id

    @property
    def session(self) -> AgentSession:
        """The underlying `AgentSession` — for callers that need the
        lower-level primitives directly."""
        return self._session

    def start(self) -> "Agent":
        """Begin answering verification checks in the background.

        Returns `self` so it chains: `agent = Agent(...).start()`.
        """
        self._worker.start()
        return self

    def log(
        self,
        input: str | bytes,
        output: str | bytes,
        metadata: dict[str, Any] | None = None,
    ) -> dict:
        """Record one interaction.

        Hashes the payloads locally — raw input and output never leave
        the process — reports the hashes, answers any checks riding on
        the response, and persists the advanced state so a restart
        resumes cleanly. Returns the server's raw JSON response.
        """
        response = self._session.log_event(input, output, metadata=metadata)
        self._store.save(self._session.to_dict())
        return response

    def poll_now(self) -> int:
        """Answer any pending checks immediately instead of waiting for
        the next background tick. Returns the number answered."""
        return self._worker.poll_once()

    def get_status(self) -> dict:
        """Read this agent's verification status from the server."""
        return self._session.get_status()

    def issue_proof(
        self, *, ttl_seconds: int = 3600, scope: str | None = None
    ) -> dict:
        """Issue a signed identity proof for this agent."""
        return self._session.issue_proof(
            ttl_seconds=ttl_seconds, scope=scope
        )

    def stop(self) -> None:
        """Stop the background loop, persist final state, and close the
        HTTP client. Safe to call more than once."""
        self._worker.stop()
        self._store.save(self._session.to_dict())
        self._client.close()

    def __enter__(self) -> "Agent":
        return self.start()

    def __exit__(self, *_exc: object) -> None:
        self.stop()
