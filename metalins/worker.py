"""Background loop that answers verification checks on a cadence.

The server issues verification checks for an agent; the agent answers
each with a short deterministic computation. `log_event` already
returns any checks riding on its response — but an agent that is not
actively logging would let those checks expire unanswered.

`CheckWorker` runs a daemon thread that polls for pending checks every
`interval` seconds and answers each one, so verification keeps working
whether or not the agent is busy. Every step is hashing and HTTP — no
model is ever involved.
"""
from __future__ import annotations

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from metalins.mcp_session import AgentSession


class CheckWorker:
    """A daemon thread that polls for, and answers, pending checks.

    Bound to one `AgentSession`. `start()` spawns the thread; `stop()`
    joins it. `poll_once()` runs a single poll+answer pass and is also
    safe to call directly — useful for a manual flush or in tests.
    """

    def __init__(self, session: "AgentSession", *, interval: float = 30.0):
        if interval <= 0:
            raise ValueError("interval must be positive")
        self._session = session
        self._interval = float(interval)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def interval(self) -> float:
        return self._interval

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Spawn the background poll loop. A no-op if already running."""
        if self.running:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="metalins-check-worker", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the loop to exit and wait up to `timeout` for it."""
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)
        self._thread = None

    def poll_once(self) -> int:
        """Poll for pending checks and answer them.

        Returns the number answered. A single failing check (expired, a
        network blip) is skipped so it cannot block the others.
        """
        client = self._session._client
        if client is None:
            raise RuntimeError("AgentSession has no client bound.")
        checks = client.list_pending_checks(self._session.agent_id)
        answered = 0
        for check in checks:
            try:
                self._session.answer_check(check)
                answered += 1
            except Exception:
                # One bad check must not stop the rest of the batch.
                pass
        return answered

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.poll_once()
            except Exception:
                # A transient failure (network, server 5xx) must never
                # kill the loop — wait out the interval and retry.
                pass
            # Interruptible sleep: stop() sets the event and returns
            # immediately instead of waiting out the full interval.
            self._stop.wait(self._interval)
