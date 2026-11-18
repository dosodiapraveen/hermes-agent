"""Every wait on the connection-lifecycle lock is BOUNDED (#board-827).

``read_header_bytes_preopen`` is the byte-probe the terminal pre-exec guard calls on every
command, and it holds ``_live_lock`` across a raw ``open()``/``read()``. One caller whose read
wedges (a cloud placeholder, a TCC-gated path, a stuck mount) therefore parked every later guard
call for the life of the process. These tests hold the lock in another thread and assert that the
guard-facing entry points return a fail-closed verdict in bounded time instead of inheriting the
wedge.

Without a bound each of these blocks for the full hold below, which is exactly the production
symptom: a refused terminal call whose refusal time tracks the caller's own timeout.
"""

from __future__ import annotations

import threading
import time

import pytest

import hermes_cli.sqlite_safe_read as mod
from hermes_cli.sqlite_safe_read import (
    LiveConnectionError,
    has_live_connection,
    offline_file_access,
    read_header_bytes_preopen,
)

# The lock bound under test is shrunk so the holder comfortably outlives it: the point is that
# the WAIT EXPIRES, not that the default constant has a particular value.
BOUND_S = 0.3
HOLD_S = 6.0


@pytest.fixture
def wedged_lock(monkeypatch):
    """Hold ``_live_lock`` on another thread far longer than the bound, as a wedged reader does."""
    monkeypatch.setattr(mod, "_LIVE_LOCK_TIMEOUT_S", BOUND_S)
    started = threading.Event()

    def _hold():
        with mod._live_lock:
            started.set()
            time.sleep(HOLD_S)

    t = threading.Thread(target=_hold, daemon=True)
    t.start()
    assert started.wait(5.0), "holder thread never acquired the lock"
    yield t
    t.join(timeout=HOLD_S + 5.0)


def test_read_header_bytes_preopen_is_bounded_when_the_lock_is_wedged(wedged_lock):
    """THE production call: the guard's byte-probe must not inherit another thread's wedge."""
    start = time.monotonic()
    result = read_header_bytes_preopen("/tmp/does-not-matter-827.db")
    elapsed = time.monotonic() - start

    assert result is None, f"expected the fail-closed answer, got {result!r}"
    assert elapsed < 2.0, (
        f"read_header_bytes_preopen waited {elapsed:.2f}s on a wedged lock "
        f"(bound is {mod._LIVE_LOCK_TIMEOUT_S}s); the terminal guard would park for that long"
    )


def test_has_live_connection_is_bounded_and_fails_closed(wedged_lock):
    """Expiry must answer True (assume live), never False: True makes callers refuse raw access."""
    start = time.monotonic()
    verdict = has_live_connection("/tmp/does-not-matter-827.db")
    elapsed = time.monotonic() - start

    assert verdict is True, "a wedged registry must not be reported as 'nothing live' (fails open)"
    assert elapsed < 2.0, f"has_live_connection waited {elapsed:.2f}s on a wedged lock"


def test_offline_file_access_is_bounded_and_refuses(wedged_lock):
    """The raw-I/O context manager must raise the fail-closed error, not block."""
    start = time.monotonic()
    with pytest.raises(LiveConnectionError):
        with offline_file_access("/tmp/does-not-matter-827.db", what="read"):
            pytest.fail("the body must not run while the lock is wedged")
    elapsed = time.monotonic() - start

    assert elapsed < 2.0, f"offline_file_access waited {elapsed:.2f}s on a wedged lock"


def test_the_bound_is_shorter_than_a_terminal_guard_ceiling():
    """Sanity: the lock bound must be small enough that a guard call still fits under its 90s cap."""
    assert 0 < mod._LIVE_LOCK_TIMEOUT_S <= 10, mod._LIVE_LOCK_TIMEOUT_S
