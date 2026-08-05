"""Torch autostart profiling spec resolution."""

from __future__ import annotations

import pytest

import probing
from probing.ext import torch as torch_ext
from probing.profiling.torch_probe import TorchProbeConfig


@pytest.fixture(autouse=True)
def _default_all_rank_scope(monkeypatch):
    monkeypatch.delenv("PROBING_TORCH_PROFILING_RANKS", raising=False)


def test_torch_profiling_spec_prefers_config(monkeypatch):
    monkeypatch.delenv("PROBING_TORCH_PROFILING", raising=False)
    probing.config.set("probing.torch.profiling", "1.0,backward=on")
    try:
        assert torch_ext._torch_profiling_spec() == "1.0,backward=on"
        cfg = TorchProbeConfig.parse(torch_ext._torch_profiling_spec())
        assert cfg.enabled
        assert cfg.backward
    finally:
        probing.config.remove("probing.torch.profiling")


def test_torch_profiling_spec_falls_back_to_env(monkeypatch):
    probing.config.remove("probing.torch.profiling")
    monkeypatch.setenv("PROBING_TORCH_PROFILING", "ordered:0.1,backward=on,shadow=off")
    try:
        spec = torch_ext._torch_profiling_spec()
        assert spec == "ordered:0.1,backward=on,shadow=off"
        assert probing.config.get_str("probing.torch.profiling") == spec
        cfg = TorchProbeConfig.parse(spec)
        assert cfg.backward
        assert cfg.shadow_baseline == 0
    finally:
        probing.config.remove("probing.torch.profiling")
        monkeypatch.delenv("PROBING_TORCH_PROFILING", raising=False)


@pytest.mark.parametrize(
    ("selector", "rank", "local_rank", "node_rank", "expected"),
    [
        ("all", 17, 1, 2, True),
        ("none", 0, 0, 0, False),
        ("node0", 7, 7, 0, True),
        ("node0", 8, 0, 1, False),
        ("local0", 24, 0, 3, True),
        ("local0", 25, 1, 3, False),
        ("rank0", 0, 0, 0, True),
        ("rank0", 1, 1, 0, False),
        ("0,8-15", 12, 4, 1, True),
        ("0,8-15", 7, 7, 0, False),
        ("typo", 0, 0, 0, False),
    ],
)
def test_torch_profiling_rank_selector(
    monkeypatch, selector, rank, local_rank, node_rank, expected
):
    monkeypatch.setenv("RANK", str(rank))
    monkeypatch.setenv("LOCAL_RANK", str(local_rank))
    monkeypatch.setenv("GROUP_RANK", str(node_rank))
    assert torch_ext.torch_profiling_rank_selected(selector) is expected


def test_node0_selector_falls_back_to_rank_and_local_world_size(monkeypatch):
    monkeypatch.delenv("GROUP_RANK", raising=False)
    monkeypatch.delenv("NODE_RANK", raising=False)
    monkeypatch.setenv("RANK", "9")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    assert torch_ext.torch_profiling_rank_selected("node0") is False


def test_unscoped_single_process_is_selected(monkeypatch):
    for name in (
        "RANK",
        "PMI_RANK",
        "OMPI_COMM_WORLD_RANK",
        "LOCAL_RANK",
        "GROUP_RANK",
        "NODE_RANK",
    ):
        monkeypatch.delenv(name, raising=False)
    assert torch_ext.torch_profiling_rank_selected("node0") is True
