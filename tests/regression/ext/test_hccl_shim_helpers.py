"""Regression tests for HCCL shim deployment helpers."""

import os
from unittest.mock import patch

from probing.hccl import ld_library_path_prefix


def test_ld_library_path_prefix_points_to_proxy_directory():
    with patch("probing.hccl.shim_dir", return_value="/tmp/probing-hccl"):
        assert ld_library_path_prefix() == f"/tmp/probing-hccl{os.pathsep}"
