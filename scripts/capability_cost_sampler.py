#!/usr/bin/env python3
"""Sample Probing's three resource accounts for capability validation.

The output schema is the one required by CAPABILITY-SPEC B2:

``ts,cold_bytes_cumulative,cold_bytes_current,hot_bytes,proc_io_write_bytes,rss_kb``

``cold_bytes_cumulative`` is a watcher counter. It retains bytes from MEMC
segments after TTL deletion and counts growth of an open segment exactly once;
it is therefore deliberately different from a point-in-time ``du``.
"""

from __future__ import annotations

import argparse
import csv
import os
import signal
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, TextIO


def tree_bytes(root: Path, suffix: str | None = None) -> int:
    """Return bytes in regular files below ``root``; vanished files are ignored."""
    total = 0
    if not root.exists():
        return total
    for path in root.rglob("*"):
        try:
            if path.is_file() and (suffix is None or path.name.endswith(suffix)):
                total += path.stat().st_size
        except FileNotFoundError:
            # Compactor/TTL can rename or remove a segment during the walk.
            continue
    return total


@dataclass
class ColdSegmentWatcher:
    """Track all bytes ever materialized by observed MEMC segment inodes."""

    root: Path
    _max_size: dict[tuple[int, int], int] = field(default_factory=dict)
    cumulative: int = 0

    def sample(self) -> tuple[int, int]:
        current = 0
        if not self.root.exists():
            return self.cumulative, current
        for path in self.root.rglob("*.memc"):
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue
            if not path.is_file():
                continue
            size = stat.st_size
            current += size
            key = (stat.st_dev, stat.st_ino)
            previous = self._max_size.get(key, 0)
            if size > previous:
                self.cumulative += size - previous
                self._max_size[key] = size
        return self.cumulative, current


def parse_proc_io(lines: Iterable[str]) -> int:
    for line in lines:
        key, _, value = line.partition(":")
        if key.strip() == "write_bytes":
            return int(value.strip())
    raise ValueError("write_bytes is absent from proc io")


def parse_proc_status_rss(lines: Iterable[str]) -> int:
    for line in lines:
        key, _, value = line.partition(":")
        if key.strip() == "VmRSS":
            return int(value.strip().split()[0])
    raise ValueError("VmRSS is absent from proc status")


def proc_metrics(pid: int, proc_root: Path = Path("/proc")) -> tuple[int, int]:
    base = proc_root / str(pid)
    with (base / "io").open(encoding="utf-8") as handle:
        write_bytes = parse_proc_io(handle)
    with (base / "status").open(encoding="utf-8") as handle:
        rss_kb = parse_proc_status_rss(handle)
    return write_bytes, rss_kb


def default_hot_dir(pid: int) -> Path:
    root = Path(os.environ.get("PROBING_DATA_DIR", "/dev/shm/probing"))
    candidate = root / str(pid)
    return candidate if candidate.exists() else root


def default_cold_dir() -> Path:
    configured = os.environ.get("PROBING_COLD_DIR")
    return Path(configured) if configured else Path(tempfile.gettempdir()) / "probing-cold"


def write_samples(
    *,
    pid: int,
    hot_dir: Path,
    cold_dir: Path,
    output: TextIO,
    interval: float,
    duration: float | None,
) -> int:
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(
        (
            "ts",
            "cold_bytes_cumulative",
            "cold_bytes_current",
            "hot_bytes",
            "proc_io_write_bytes",
            "rss_kb",
        )
    )
    output.flush()
    watcher = ColdSegmentWatcher(cold_dir)
    started = time.monotonic()
    stop = False

    def request_stop(_signum, _frame):
        nonlocal stop
        stop = True

    old_handlers = {
        signum: signal.signal(signum, request_stop)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        while not stop:
            tick = time.monotonic()
            try:
                write_bytes, rss_kb = proc_metrics(pid)
            except (FileNotFoundError, ProcessLookupError):
                return 0
            cumulative, current = watcher.sample()
            writer.writerow(
                (
                    f"{time.time():.6f}",
                    cumulative,
                    current,
                    tree_bytes(hot_dir),
                    write_bytes,
                    rss_kb,
                )
            )
            output.flush()
            if duration is not None and time.monotonic() - started >= duration:
                return 0
            remaining = interval - (time.monotonic() - tick)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True, help="training process PID")
    parser.add_argument("--hot-dir", type=Path, help="MEMT directory (defaults from PID/env)")
    parser.add_argument("--cold-dir", type=Path, help="MEMC directory")
    parser.add_argument("--output", type=Path, help="CSV path (default: stdout)")
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="sampling interval seconds; must be <=1 for B2 acceptance",
    )
    parser.add_argument("--duration", type=float, help="optional run duration seconds")
    args = parser.parse_args(argv)
    if args.pid <= 0:
        parser.error("--pid must be positive")
    if not 0 < args.interval <= 1.0:
        parser.error("--interval must be in (0, 1.0]")
    if args.duration is not None and args.duration <= 0:
        parser.error("--duration must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    hot_dir = args.hot_dir or default_hot_dir(args.pid)
    cold_dir = args.cold_dir or default_cold_dir()
    if args.output is None:
        return write_samples(
            pid=args.pid,
            hot_dir=hot_dir,
            cold_dir=cold_dir,
            output=sys.stdout,
            interval=args.interval,
            duration=args.duration,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as output:
        return write_samples(
            pid=args.pid,
            hot_dir=hot_dir,
            cold_dir=cold_dir,
            output=output,
            interval=args.interval,
            duration=args.duration,
        )


if __name__ == "__main__":
    raise SystemExit(main())
