from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[3] / "scripts" / "capability_cost_sampler.py"
SPEC = importlib.util.spec_from_file_location("capability_cost_sampler", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
sampler = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sampler
SPEC.loader.exec_module(sampler)


def test_cold_watcher_keeps_deleted_segment_bytes(tmp_path):
    first = tmp_path / "writer-000001.memc"
    first.write_bytes(b"a" * 10)
    watcher = sampler.ColdSegmentWatcher(tmp_path)
    assert watcher.sample() == (10, 10)

    first.write_bytes(b"a" * 16)
    assert watcher.sample() == (16, 16)
    first.unlink()
    assert watcher.sample() == (16, 0)

    second = tmp_path / "writer-000002.memc"
    second.write_bytes(b"b" * 7)
    assert watcher.sample() == (23, 7)


def test_tree_bytes_and_proc_parsers(tmp_path):
    (tmp_path / "hot").mkdir()
    (tmp_path / "hot" / "a.memt").write_bytes(b"x" * 4)
    (tmp_path / "hot" / "b").write_bytes(b"x" * 9)
    assert sampler.tree_bytes(tmp_path / "hot") == 13
    assert sampler.tree_bytes(tmp_path / "hot", ".memt") == 4
    assert sampler.parse_proc_io(["rchar: 2\n", "write_bytes: 123\n"]) == 123
    assert sampler.parse_proc_status_rss(["Name: train\n", "VmRSS: 456 kB\n"]) == 456
