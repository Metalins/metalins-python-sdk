"""Metalins SDK — client for AI agent identity verification.

Apache 2.0. Copyright 2026 Jose Hernandez (Metalins).

A thin wrapper over the Metalins developer API. It handles the client
side of the verification protocol:
- captures your agent's behavior and reports it to the Metalins server,
- answers the server's verification checks automatically,
- receives signed identity claims.

All scoring and comparison logic runs server-side.

Two entry points:
- `Agent` — the high-level facade: register-or-resume, a background
  loop that answers verification checks, `start()/log()/stop()`.
- `Client` + `AgentSession` — the lower-level primitives `Agent` is
  built from, for callers that want to drive the protocol themselves.
"""
from metalins.client import Client
from metalins.agent import Agent
from metalins.worker import CheckWorker
from metalins.state import FileStateStore, StateStore, default_state_path
from metalins.mcp_session import (
    AgentSession,
    compute_check_answer,
    derive_initial_digest,
    derive_next_digest,
)
from metalins.errors import (
    MetalinsError,
    AuthenticationError,
    AgentNotFound,
    ServerError,
)

__version__ = "0.4.0"

__all__ = [
    "Agent",
    "Client",
    "CheckWorker",
    "StateStore",
    "FileStateStore",
    "default_state_path",
    "AgentSession",
    "compute_check_answer",
    "derive_initial_digest",
    "derive_next_digest",
    "MetalinsError",
    "AuthenticationError",
    "AgentNotFound",
    "ServerError",
]
