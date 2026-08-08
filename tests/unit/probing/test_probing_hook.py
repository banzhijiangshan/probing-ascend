from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


PYTHON_ROOT = Path(__file__).resolve().parents[3] / "python"


def _imported_modules(probing_value: str | None) -> list[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PYTHON_ROOT)
    if probing_value is None:
        env.pop("PROBING", None)
    else:
        env["PROBING"] = probing_value
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            (
                "import json, sys; import probing_hook; "
                "print(json.dumps(sorted(m for m in sys.modules "
                "if m == 'probing' or m.startswith('probing.'))))"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return json.loads(result.stdout)


def test_disabled_hook_does_not_import_package() -> None:
    assert _imported_modules(None) == []
    assert _imported_modules("0") == []
    assert _imported_modules("off") == []


def test_nonmatching_script_does_not_import_package() -> None:
    assert _imported_modules("another_training_script.py") == []


def test_torchrun_retargets_followed_mode_to_training_script() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PYTHON_ROOT)
    env["PROBING"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            (
                "import os, sys; "
                "sys.argv = ['/opt/bin/torchrun', '--nproc-per-node=8', "
                "'train_bench_probe_npu.py']; "
                "import probing_hook; print(os.environ['PROBING']); "
                "print(any(m == 'probing' or m.startswith('probing.') "
                "for m in sys.modules))"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.stdout.splitlines() == ["train_bench_probe_npu.py", "False"]
