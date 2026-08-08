from __future__ import annotations

import gc
from unittest.mock import patch
import weakref

import torch

from probing.profiling.torch.module_utils import get_toplevel_module


def test_toplevel_scan_skips_dead_weakref_proxy() -> None:
    dead_module = torch.nn.Linear(2, 2)
    dead_proxy = weakref.proxy(dead_module)
    del dead_module
    gc.collect()

    live_module = torch.nn.Linear(2, 2)
    with patch.object(gc, "get_objects", return_value=[dead_proxy, live_module]):
        assert get_toplevel_module() == [live_module]
