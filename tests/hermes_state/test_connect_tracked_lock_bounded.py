"""connect_tracked HOLDS the connection-lifecycle lock across real file I/O.

Board #741/#827. The previous pass bounded the three READERS of ``_live_lock``
(``read_header_bytes_preopen``, ``has_live_connection``, ``offline_file_access``) and capped the
terminal guard deadline at 90s. Measured against the live gateway, none of it engaged: 1,436
``deadline-terminal.pre-exec-guard`` workers were parked on ``_live_lock`` while zero
"connection-lifecycle lock has been held" warnings existed in any log -- the bounded waiters never
fired because the waiters were never the problem.

The holder is ``connect_tracked``: it took the lock with a bare ``with _live_lock:`` that spans
``opener()`` -> ``open()`` -> ``os_open()``, plus the first SQLite reads of a multi-GB ``state.db``
and its WAL recovery. One slow open therefore parked every guard call in the process with no bound
of its own.

These tests pin the holder's bound. Each one FAILS on the unbounded form: the acquire blocks for
the whole budget of the sleep instead of refusing fast.
"""

from __future__ import annotations

import sqlite3
import string
import threading
import time
from pathlib import Path

import pytest

from hermes_cli import sqlite_safe_read as ssr


def _slow_opener(seconds: float, calls: dict) -> object:
    """An opener that holds whatever lock it is under for *seconds*, like a slow file open."""

    def opener(path: str, **kwargs):
        calls["count"] = calls.get("count", 0) + 1
        time.sleep(seconds)
        return sqlite3.connect(path, **kwargs)

    return opener


@pytest.fixture()
def lock_budget(monkeypatch):
    """Shrink the lock budget so the tests are fast; restore via monkeypatch."""

    def _set(seconds: float) -> None:
        monkeypatch.setattr(ssr, "_LIVE_LOCK_TIMEOUT_S", seconds, raising=True)

    return _set


def test_connect_tracked_refuses_fast_when_another_open_holds_the_lock(tmp_path, lock_budget):
    """The control case: a holder that never returns must not park a second caller forever."""
    db = tmp_path / "state.db"
    sqlite3.connect(db).close()
    lock_budget(0.3)

    holding = threading.Event()
    release = threading.Event()

    def hold():
        with ssr._live_lock:
            holding.set()
            release.wait(10)

    holder = threading.Thread(target=hold, daemon=True)
    holder.start()
    assert holding.wait(5), "the holder thread never took the lock"

    started = time.monotonic()
    with pytest.raises(ssr.LiveConnectionError):
        ssr.connect_tracked(db)
    elapsed = time.monotonic() - started
    release.set()
    holder.join(5)

    assert elapsed < 2.0, (
        f"connect_tracked waited {elapsed:.2f}s against a {ssr._LIVE_LOCK_TIMEOUT_S}s budget and a "
        "holder that never released: the acquire is still unbounded, so one slow open parks every "
        "terminal-guard call in the process (board #741)"
    )


def test_the_lock_is_released_when_the_opener_raises(tmp_path, lock_budget):
    """A manual acquire can leak what a `with` block could not; the finally must release it."""
    db = tmp_path / "state.db"
    sqlite3.connect(db).close()
    lock_budget(0.3)

    def boom(path: str, **kwargs):
        raise OSError("simulated open failure")

    with pytest.raises(OSError):
        ssr.connect_tracked(db, connect_fn=boom)

    assert ssr._live_lock.acquire(timeout=0.3), (
        "connect_tracked left _live_lock held after the opener raised: every later guard call "
        "would queue behind a lock nobody owns (board #741)"
    )
    ssr._live_lock.release()


def test_no_lock_leak_across_many_opens(tmp_path):
    """Repeated successful opens must each return the lock; a leak surfaces as an exhausted RLock."""
    db = tmp_path / "state.db"
    sqlite3.connect(db).close()

    for _ in range(50):
        conn = ssr.connect_tracked(db)
        conn.close()

    assert ssr._live_lock.acquire(timeout=0.5), "the lifecycle lock leaked across 50 opens"
    ssr._live_lock.release()


def test_a_healthy_open_is_not_shrunk_by_the_budget(tmp_path, lock_budget):
    """The bound must not break the normal path: a fast open still connects and is tracked."""
    db = tmp_path / "state.db"
    sqlite3.connect(db).close()
    lock_budget(5.0)

    conn = ssr.connect_tracked(db)
    try:
        assert conn.execute("select 1").fetchone() == (1,)
        assert ssr.has_live_connection(db) is True, (
            "connect_tracked returned without registering the connection, so a byte-probe could "
            "cancel its locks"
        )
    finally:
        conn.close()

    assert ssr.has_live_connection(db) is False
