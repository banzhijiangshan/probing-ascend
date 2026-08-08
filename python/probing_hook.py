"""Wheel / develop entry executed from ``probing.pth`` or ``probing_hook.pth``.

This module deliberately performs the activation check without importing the
``probing`` package.  Importing ``probing.site_hook`` first would execute
``probing.__init__`` and start the native engine even when ``PROBING`` is not
enabled, contaminating uninstrumented baseline processes.
"""

from __future__ import annotations

import os
import re
import sys


def _script_name() -> str:
    try:
        return os.path.basename(sys.argv[0])
    except (IndexError, AttributeError):
        return "<unknown>"


def _is_lightweight_entrypoint() -> bool:
    helper_suffixes = (
        "probing/nccl/__main__.py",
        "probing\\nccl\\__main__.py",
        "probing/skills/__main__.py",
        "probing\\skills\\__main__.py",
        "probing/dev_pth.py",
        "probing\\dev_pth.py",
    )
    if sys.argv and any(sys.argv[0].endswith(s) for s in helper_suffixes):
        return True
    try:
        index = sys.argv.index("-m")
        module = sys.argv[index + 1]
    except (ValueError, IndexError):
        return False
    return module in ("probing.nccl", "probing.skills", "probing.dev_pth")


def _should_import_probing() -> bool:
    raw = os.environ.get("PROBING", "0")
    probe_value = raw
    if raw.startswith("init:"):
        parts = raw.split("+", 1)
        probe_value = parts[1] if len(parts) > 1 else "0"

    token = probe_value.strip().lower()
    script = _script_name()
    if token in ("0", "", "false", "no", "off"):
        return False
    # torchrun must pass PROBING through to the rank processes.
    if re.search("torchrun", script) is not None:
        return False
    if script == "probing" or _is_lightweight_entrypoint():
        return False
    if token in ("1", "followed", "2", "nested"):
        return True
    if token.startswith("regex:"):
        pattern = probe_value.split(":", 1)[1]
        candidates = [candidate for candidate in sys.argv if candidate]
        candidates.append(script)
        try:
            return any(re.search(pattern, candidate) for candidate in candidates)
        except re.error:
            # Let site_hook report the malformed expression in a targeted run.
            return True
    return probe_value == script


if _should_import_probing():
    from probing.site_hook import run_site_hook

    run_site_hook()
