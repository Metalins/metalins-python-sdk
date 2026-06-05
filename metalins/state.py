"""Durable storage for an agent's session state.

`Agent` keeps its session — the agent id, its secret, and the running
hash chain — in a `StateStore` so a process restart resumes the same
agent instead of registering a new one. Losing the stored state means
the agent can no longer answer checks that target events from before
the loss, so persistence is mandatory for any long-lived agent.

`StateStore` is a two-method protocol; supply your own (a database row,
a secrets manager) by implementing `load()` / `save()`. The default
`FileStateStore` writes a single JSON file with owner-only permissions
and is what the one-line `Agent(...)` bootstrap uses with no config.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class StateStore(Protocol):
    """Where an `Agent` persists its session between runs.

    `save` receives the JSON-safe dict from `AgentSession.to_dict()`.
    `load` returns the dict last saved, or `None` on the first run (no
    state yet) — that is the signal for `Agent` to register a new agent.
    """

    def load(self) -> dict | None: ...

    def save(self, state: dict) -> None: ...


class FileStateStore:
    """A `StateStore` backed by one JSON file with `0600` permissions.

    The file holds the agent secret, so it is created owner-read/write
    only, in a `0700` directory. Writes are atomic — a temp file in the
    same directory is written and then renamed over the target — so a
    crash mid-write never leaves a truncated state file.
    """

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path).expanduser()

    def load(self) -> dict | None:
        if not self.path.exists():
            return None
        text = self.path.read_text(encoding="utf-8")
        if not text.strip():
            return None
        return json.loads(text)

    def save(self, state: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        # Write to a temp file in the same directory (so the rename is
        # atomic — same filesystem), 0600 from creation, then replace.
        fd, tmp = tempfile.mkstemp(
            dir=self.path.parent, prefix=".", suffix=".tmp"
        )
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(state, fh, separators=(",", ":"), sort_keys=True)
            os.replace(tmp, self.path)
        except BaseException:
            # Leave no half-written temp file behind on any failure.
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        # Defensive: ensure the final file is owner-only even if it
        # pre-existed with looser permissions.
        os.chmod(self.path, 0o600)


def _slugify(name: str) -> str:
    """Turn an agent name into a filesystem-safe slug for the default
    store path."""
    slug = re.sub(r"[^a-z0-9-]+", "-", name.lower()).strip("-")
    return slug or "agent"


def default_state_path(name: str) -> Path:
    """The default per-agent state file: `~/.metalins/<slug>.json`.

    Used when `Agent(...)` is constructed without an explicit store, so
    the one-line bootstrap needs no path configuration.
    """
    return Path.home() / ".metalins" / f"{_slugify(name)}.json"
