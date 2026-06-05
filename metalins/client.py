"""Metalins Client — a thin wrapper over the Metalins developer API.

One method per public endpoint. No verification logic lives here: the
server scores and compares; this client only carries requests and
responses over HTTPS, authenticated with an API key.

For the streaming-verification flow (log events, answer the server's
verification checks) use `AgentSession`, which keeps the per-agent
hash chain the client needs to answer checks. `start_session()`
registers an agent and hands back a ready-to-use session.
"""
from __future__ import annotations

from typing import Any

import httpx

from metalins.errors import (
    AgentNotFound,
    AuthenticationError,
    MetalinsError,
    ServerError,
)
from metalins.mcp_session import AgentSession, derive_initial_digest

DEFAULT_BASE_URL = "https://api.metalins.com"


class Client:
    """Authenticated client for the Metalins developer API.

    Each method maps to exactly one endpoint and returns the server's
    JSON response as a plain dict — no reshaping — so callers stay free
    to use the API however they need.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 30.0,
    ):
        if not api_key:
            raise AuthenticationError("api_key is required")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self._http = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )

    def __enter__(self) -> "Client":
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def close(self) -> None:
        self._http.close()

    # ------------------------------------------------------- developer API
    # One method per endpoint. Returns the raw JSON response dict.

    def create_agent(
        self,
        *,
        name: str,
        model: str | None = None,
        framework: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict:
        """Register a new agent — `POST /v1/agents`.

        The response carries `agent_secret` exactly once; store it. The
        secret is required for the agent to answer verification checks
        and is never returned again.
        """
        body: dict[str, Any] = {"name": name}
        if model is not None:
            body["model"] = model
        if framework is not None:
            body["framework"] = framework
        if metadata is not None:
            body["metadata"] = metadata
        return self._request("POST", "/v1/agents", json=body, expected_status=201)

    def log_event(
        self,
        agent_id: str,
        *,
        input_hash: str,
        output_hash: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict:
        """Record one interaction — `POST /v1/agents/{id}/events`.

        Send sha256 hex digests, never raw text. The response carries the
        running `event_count` and any `pending_checks` the agent should
        answer.
        """
        body: dict[str, Any] = {
            "input_hash": input_hash,
            "output_hash": output_hash,
        }
        if metadata is not None:
            body["metadata"] = metadata
        return self._request(
            "POST", f"/v1/agents/{agent_id}/events", json=body
        )

    def answer_check(
        self,
        agent_id: str,
        check_id: str,
        *,
        answer: str | None = None,
        decline_reason: str | None = None,
        progress: int | None = None,
    ) -> dict:
        """Answer a verification check — `POST /v1/agents/{id}/checks/{check_id}`.

        Send `answer` for a well-formed check, or `decline_reason` when
        the agent recognizes a malformed check and refuses it.
        """
        body: dict[str, Any] = {}
        if answer is not None:
            body["answer"] = answer
        if decline_reason is not None:
            body["decline_reason"] = decline_reason
        if progress is not None:
            body["progress"] = progress
        return self._request(
            "POST", f"/v1/agents/{agent_id}/checks/{check_id}", json=body
        )

    def list_pending_checks(self, agent_id: str) -> list[dict]:
        """List the verification checks awaiting this agent —
        `GET /v1/agents/{id}/checks`.

        `log_event` returns `pending_checks` riding on its response, but
        an agent that goes quiet would never see a check issued in the
        meantime. This polls for pending checks without logging an event,
        so a background loop can answer them before they expire. Returns
        the list of checks (same shape as `log_event`'s `pending_checks`).
        """
        resp = self._request("GET", f"/v1/agents/{agent_id}/checks")
        return resp.get("checks", [])

    def list_agents(self, *, limit: int = 50, offset: int = 0) -> dict:
        """List the agents owned by this account — `GET /v1/agents`."""
        return self._request(
            "GET", "/v1/agents", params={"limit": limit, "offset": offset}
        )

    def get_agent(self, agent_id: str) -> dict:
        """Read one agent's verification status — `GET /v1/agents/{id}`."""
        return self._request("GET", f"/v1/agents/{agent_id}")

    def issue_proof(
        self,
        agent_id: str,
        *,
        ttl_seconds: int = 3600,
        scope: str | None = None,
    ) -> dict:
        """Issue a signed identity proof — `POST /v1/agents/{id}/proofs`.

        `ttl_seconds` must be one of 300 / 3600 / 86400. The proof is a
        signed token the agent can hand to a relying party.
        """
        body: dict[str, Any] = {"ttl_seconds": ttl_seconds}
        if scope is not None:
            body["scope"] = scope
        return self._request(
            "POST", f"/v1/agents/{agent_id}/proofs", json=body,
            expected_status=201,
        )

    def revoke_agent(self, agent_id: str, *, reason: str | None = None) -> dict:
        """Revoke an agent — `DELETE /v1/agents/{id}`. Permanent."""
        params = {"reason": reason} if reason is not None else None
        return self._request(
            "DELETE", f"/v1/agents/{agent_id}", params=params
        )

    # ------------------------------------------------- streaming-session API

    def start_session(
        self,
        *,
        name: str,
        model: str | None = None,
        framework: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentSession:
        """Register an agent and return a bound `AgentSession`.

        The session holds the `agent_secret` and the hash chain it needs
        to answer verification checks. Persist `session.to_dict()` so the
        agent survives a restart with that history intact.
        """
        data = self.create_agent(
            name=name, model=model, framework=framework, metadata=metadata
        )
        secret = data["agent_secret"]
        return AgentSession(
            agent_id=data["agent_id"],
            agent_secret=secret,
            event_count=0,
            digest_history={0: derive_initial_digest(secret)},
            _client=self,
        )

    def attach_session(self, session: AgentSession) -> AgentSession:
        """Bind an existing (e.g. deserialized) session to this client."""
        session.bind_client(self)
        return session

    # ----------------------------------------------------------- internals

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        expected_status: int = 200,
    ) -> dict[str, Any]:
        try:
            r = self._http.request(method, path, json=json, params=params)
        except httpx.HTTPError as e:
            raise MetalinsError(f"Network error: {e}") from e

        if r.status_code == 401:
            raise AuthenticationError(_extract_detail(r))
        if r.status_code == 404:
            raise AgentNotFound(_extract_detail(r))
        if 400 <= r.status_code < 500:
            raise MetalinsError(
                f"Client error {r.status_code}: {_extract_detail(r)}"
            )
        if r.status_code >= 500:
            raise ServerError(
                f"Server error {r.status_code}: {_extract_detail(r)}"
            )
        if r.status_code != expected_status:
            raise MetalinsError(
                f"Unexpected status {r.status_code}: {_extract_detail(r)}"
            )

        if not r.content:
            return {}
        return r.json()


def _extract_detail(r: httpx.Response) -> str:
    try:
        return r.json().get("detail", r.text)
    except Exception:
        return r.text
