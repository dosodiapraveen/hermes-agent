"""The pre-execution guard wait is CAPPED, independent of the caller's command timeout (#board-741).

``test_terminal_pre_guard_deadline.py`` pins that a wedged guard returns instead of hanging
forever, but it drives the deadline with a 0.05s command timeout, so it passes with or without
a ceiling. It therefore never caught the case that actually hurt: a long command timeout
(180-400s) handed the guard the whole budget, and a wedged probe burned all of it -- 80% of the
refusals logged across ~/.hermes/logs came out ABOVE 90s. These tests pin the ceiling itself.
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest

import tools.terminal_tool as terminal_module


def _plan(timeout: float) -> SimpleNamespace:
    return SimpleNamespace(
        config={},
        env_type="local",
        effective_task_id="pre-guard-cap-test",
        cwd="/tmp",
        effective_timeout=timeout,
        promoted_from_foreground_timeout=None,
    )


@pytest.fixture
def pipeline(monkeypatch):
    """Stub planning/env/approval/execution; returns the list of executions that happened."""
    calls: list[str] = []
    monkeypatch.setattr(terminal_module, "_acquire_env", lambda *_a, **_k: object())
    monkeypatch.setattr(
        terminal_module, "_run_approval_guards", lambda *_a, **_k: terminal_module._ApprovalVerdict(),
    )
    monkeypatch.setattr(
        terminal_module, "_run_foreground",
        lambda *_a, **_k: calls.append("foreground") or "foreground-ran",
    )
    return calls


def _run_with_timeout(monkeypatch, timeout: float) -> float:
    """Run terminal_tool against a permanently wedged guard; return the wall clock it burned."""
    monkeypatch.setattr(terminal_module, "_plan_execution", lambda *_a, **_k: _plan(timeout))

    def _wedged(*_a, **_k):
        time.sleep(3600)

    monkeypatch.setattr(terminal_module, "_pre_exec_block", _wedged)
    start = time.monotonic()
    result = json.loads(terminal_module.terminal_tool("echo ok"))
    elapsed = time.monotonic() - start
    assert result["status"] == "error", result
    assert "did not finish" in result["error"], result
    return elapsed


def test_ceiling_constant_is_bounded():
    """The ceiling exists, is positive, and is far below the 180-420s refusals that motivated it."""
    assert hasattr(terminal_module, "_PRE_EXEC_GUARD_MAX_TIMEOUT_S"), (
        "no guard ceiling: a wedged probe is bounded only by the caller's command timeout"
    )
    ceiling = terminal_module._PRE_EXEC_GUARD_MAX_TIMEOUT_S
    assert 0 < ceiling <= 120, f"ceiling {ceiling}s is not a bound worth having"


def test_refusal_time_does_not_track_a_long_command_timeout(monkeypatch, pipeline):
    """A 300s command whose guard wedges is refused at the CEILING, not at 300s.

    This is the exact regression that shipped: refusal deadlines in the logs spanned
    30..420s, clustering at each caller's timeout, because ``max(timeout, MIN)`` had no
    upper bound.
    """
    monkeypatch.setattr(terminal_module, "_PRE_EXEC_GUARD_MAX_TIMEOUT_S", 0.25)
    monkeypatch.setattr(terminal_module, "_PRE_EXEC_GUARD_MIN_TIMEOUT_S", 0.0)

    elapsed = _run_with_timeout(monkeypatch, timeout=300.0)

    assert elapsed < 5.0, f"a 300s command held the guard for {elapsed:.2f}s -- the ceiling is inert"
    assert pipeline == [], "a guard with no verdict must not fail open into execution"


def test_ceiling_does_not_shrink_a_healthy_guard(monkeypatch, pipeline):
    """The cap bounds only a WEDGED guard: a fast verdict still passes straight through."""
    monkeypatch.setattr(terminal_module, "_PRE_EXEC_GUARD_MAX_TIMEOUT_S", 0.25)
    monkeypatch.setattr(terminal_module, "_PRE_EXEC_GUARD_MIN_TIMEOUT_S", 0.0)
    monkeypatch.setattr(terminal_module, "_plan_execution", lambda *_a, **_k: _plan(300.0))
    monkeypatch.setattr(terminal_module, "_pre_exec_block", lambda *_a, **_k: None)

    assert terminal_module.terminal_tool("echo ok") == "foreground-ran"
    assert pipeline == ["foreground"]


def test_ceiling_still_fails_closed_above_the_minimum(monkeypatch, pipeline):
    """A short command timeout keeps the MIN floor; the ceiling never lowers it below that."""
    monkeypatch.setattr(terminal_module, "_PRE_EXEC_GUARD_MAX_TIMEOUT_S", 0.25)
    monkeypatch.setattr(terminal_module, "_PRE_EXEC_GUARD_MIN_TIMEOUT_S", 0.05)

    elapsed = _run_with_timeout(monkeypatch, timeout=0.01)

    assert 0.04 <= elapsed < 5.0, f"expected the MIN floor to govern, burned {elapsed:.2f}s"
    assert pipeline == []
