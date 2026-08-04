"""PR-2: pillar_c_localize_culprit unit helpers (no live probing required)."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / "scripts" / "fail-slow"))

from pillar_c_localize_culprit import (  # noqa: E402
    RAW_HEAD_MAX,
    build_sql,
    resolve_mode,
    resolve_secondary_mode,
    worker_pids_by_rank,
    _pick_pid_for_rank,
    _pid_worker_score,
)


def test_resolve_mode_auto_p3_sw_a():
    assert resolve_mode("P3-SW-A", "auto") == "step_ms"


def test_resolve_mode_auto_p3_sw_b():
    assert resolve_mode("P3-SW-B", "auto") == "step_ms"


def test_resolve_mode_auto_p3_sw_c():
    assert resolve_mode("P3-SW-C", "auto") == "host_rss"


def test_resolve_mode_auto_p3_ext_a():
    assert resolve_mode("P3-EXT-A", "auto") == "host_rss"


def test_resolve_mode_explicit():
    assert resolve_mode("P3-SW-A", "host_rss") == "host_rss"


def test_build_sql_comm_max_window():
    sql = build_sql("comm_max", trigger_step=100, window=20)
    assert "global_step >= 80" in sql
    assert "global_step <= 100" in sql
    assert "max(duration_ms)" in sql


def test_build_sql_host_rss():
    sql = build_sql("host_rss", trigger_step=50, window=10)
    assert "rss_kb" in sql


def test_build_sql_step_ms_window():
    sql = build_sql("step_ms", trigger_step=140, window=20)
    assert "local_step >= 120" in sql
    assert "local_step <= 140" in sql
    assert "max(step_duration_sec)" in sql
    assert "step_ms" not in sql


def test_resolve_secondary_p3_sw_a():
    assert resolve_secondary_mode("P3-SW-A", "step_ms") == "host_rss"


def test_resolve_secondary_host_rss_none():
    assert resolve_secondary_mode("P3-SW-C", "host_rss") is None


def test_raw_head_max_is_2000():
    assert RAW_HEAD_MAX == 2000


def test_pid_worker_score_prefers_victim_and_shm(monkeypatch):
    monkeypatch.setattr(
        "pillar_c_localize_culprit._read_local_rank",
        lambda pid: {100: 7, 200: 7, 300: 0}[pid],
    )
    monkeypatch.setattr(
        "pillar_c_localize_culprit._has_torch_trace_shm",
        lambda pid: pid == 100,
    )
    monkeypatch.setattr("pillar_c_localize_culprit._is_live_pid", lambda pid: True)
    assert _pid_worker_score(100, victim_rank=7) < _pid_worker_score(200, victim_rank=7)
    assert _pick_pid_for_rank([200, 100], victim_rank=7) == 100


def test_worker_pids_by_rank_one_per_rank(monkeypatch):
    def fake_candidates():
        return [100, 200, 300, 400]

    def fake_lr(pid):
        return {100: 7, 200: 7, 300: 3, 400: 3}[pid]

    monkeypatch.setattr(
        "pillar_c_localize_culprit._candidate_worker_pids", fake_candidates
    )
    monkeypatch.setattr("pillar_c_localize_culprit._read_local_rank", fake_lr)
    monkeypatch.setattr("pillar_c_localize_culprit._is_live_pid", lambda pid: True)
    monkeypatch.setattr(
        "pillar_c_localize_culprit._has_torch_trace_shm",
        lambda pid: pid in (100, 400),
    )
    by_rank = worker_pids_by_rank(victim_rank=7)
    assert by_rank[7] == 100
    assert by_rank[3] == 400
    assert len(by_rank) == 2
