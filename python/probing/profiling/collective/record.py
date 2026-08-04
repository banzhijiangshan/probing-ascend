"""Persisted collective communication rows (query as ``python.comm_collective``).

**Timing semantics**: ``duration_ms`` is Python wall-clock around the
``torch.distributed`` API call (launch/API layer) — not NCCL execution time.
Precise NCCL-native timing lives in ``nccl.coll_perf`` / ``nccl.proxy_ops``
(NCCL profiler plugin); this table is the coarse fallback and carries the
training-step context (``global_step`` etc.) that the plugin tables lack.

``lite`` mode (default): one ``comm_collective`` row + closed ``trace_event`` pair
(timing + context, no span stack / ``inspect.stack``).

``full`` mode: live ``comm.*`` spans on the stack (for nesting during the call).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional

import probing

from probing.core import table
from probing.parallel import current_role
from probing.tracing import record_span, span, step
from probing.tracing.coordinates import row_fields
from probing.util.env import FALSE_VALUES, TRUE_VALUES


def _comm_label(op: str) -> str:
    return op if op.startswith("comm.") else f"comm.{op}"


class CommRecordMode(str, Enum):
    LITE = "lite"
    FULL = "full"


@table("comm_collective", lazy=True)
@dataclass
class CommCollective:
    micro_step: int = 0
    local_step: int = 0
    global_step: int = 0
    micro_batches: int = 1
    rank: int = -1
    world_size: int = -1
    # Extensible parallel role (e.g. "dp=2,pp=1,tp=0"); see probing.parallel.role_key.
    role: str = ""
    op: str = ""
    group_rank: int = 0
    group_size: int = 0
    participate_ranks: str = ""
    tensor_shape: str = ""
    tensor_dtype: str = ""
    bytes: int = 0
    duration_ms: float = 0.0
    async_op: int = 0


def _env_flag(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name, "")
    if raw is None:
        return default
    v = str(raw).strip().lower()
    if not v:
        return default
    if v in FALSE_VALUES or v in ("0", "off", "no"):
        return False
    if v in TRUE_VALUES or v in ("1", "on", "yes"):
        return True
    return default


def _comm_collective_lazy_enabled() -> bool:
    """PR-2 B6: skip ``python.comm_collective`` writes on non-culprit ranks.

    Default ``on`` — resident-phase ranks (torch profiling ``rate=0``) never
    allocate the 20 MiB ring. Set ``PROBING_TORCH_COMM_COLLECTIVE_LAZY=0`` to
    restore B5d behaviour (all ranks always write).
    """
    return _env_flag("PROBING_TORCH_COMM_COLLECTIVE_LAZY", default=True)


def _current_torch_rate() -> Optional[float]:
    """Best-effort read of ``probing.torch.profiling`` rate (or None)."""
    try:
        raw = probing.config.get_str("probing.torch.profiling")
    except Exception:
        return None
    if not raw:
        return None
    from probing.profiling.torch_probe import TorchProbeConfig  # local import to avoid cycles

    try:
        cfg = TorchProbeConfig.parse(raw)
    except Exception:
        return None
    if not cfg.enabled:
        return 0.0
    return float(cfg.rate)


def _skip_comm_collective_on_this_rank() -> bool:
    """True when this rank should not write to ``python.comm_collective``.

    Only skip when the lazy gate is armed *and* torch profiling rate is 0
    (i.e., resident / non-culprit rank). Culprit ranks that have been SET-
    upgraded to ``rate>0`` still write.
    """
    if not _comm_collective_lazy_enabled():
        return False
    rate = _current_torch_rate()
    if rate is None:
        return False
    return rate <= 0.0


def _role_row_fields() -> dict:
    return {"role": current_role()}


def _step_row_fields() -> dict:
    return row_fields(step.snapshot())


def _context_fields(
    *,
    op: str,
    group_rank: int,
    group_size: int,
    participate_ranks: Iterable[int],
    tensor_shape: str = "",
    tensor_dtype: str = "",
    nbytes: int = 0,
    async_op: bool = False,
) -> dict:
    ranks_json = json.dumps(list(participate_ranks)) if participate_ranks else ""
    return {
        **_step_row_fields(),
        **_role_row_fields(),
        "op": op,
        "group_rank": group_rank,
        "group_size": group_size,
        "participate_ranks": ranks_json,
        "tensor_shape": tensor_shape,
        "tensor_dtype": tensor_dtype,
        "bytes": nbytes,
        "async_op": int(async_op),
    }


def record_comm_lite(
    *,
    op: str,
    duration_ms: float,
    group_rank: int,
    group_size: int,
    participate_ranks: Optional[Iterable[int]] = None,
    tensor_shape: str = "",
    tensor_dtype: str = "",
    nbytes: int = 0,
    async_op: bool = False,
    write_trace_event: bool = True,
) -> None:
    """Append timing + context; optionally mirror to ``python.trace_event``."""
    if _skip_comm_collective_on_this_rank():
        # PR-2 B6: non-culprit rank at rate=0 — don't allocate the 20 MiB
        # ``python.comm_collective`` ring.  The Rust ``note_last_comm`` cursor
        # still runs so cross-rank comm-latency probes keep working.
        try:
            from probing._core import note_last_comm

            note_last_comm(
                op,
                group_size,
                nbytes,
                int((_step_row_fields() or {}).get("global_step", -1)),
            )
        except Exception:
            pass
        return
    fields = _context_fields(
        op=op,
        group_rank=group_rank,
        group_size=group_size,
        participate_ranks=participate_ranks or (),
        tensor_shape=tensor_shape,
        tensor_dtype=tensor_dtype,
        nbytes=nbytes,
        async_op=async_op,
    )
    CommCollective(duration_ms=duration_ms, **fields).save()
    try:
        from probing._core import note_last_comm

        note_last_comm(
            op,
            group_size,
            nbytes,
            int(fields.get("global_step", -1)),
        )
    except Exception:
        pass
    if write_trace_event:
        from probing.tracing.backends import persistence_enabled

        if persistence_enabled():
            record_span(
                op,
                duration_ns=int(duration_ms * 1e6),
                attrs={**fields, "duration_ms": duration_ms, "comm": _comm_label(op)},
                source="collective_tracer",
            )


def begin_comm_span(
    op: str,
    *,
    group_rank: int,
    group_size: int,
    participate_ranks: Iterable[int],
    tensor_shape: str,
    tensor_dtype: str,
    nbytes: int,
    async_op: bool = False,
):
    """Enter a ``comm.*`` span (``full`` mode only)."""
    meta = _context_fields(
        op=op,
        group_rank=group_rank,
        group_size=group_size,
        participate_ranks=participate_ranks,
        tensor_shape=tensor_shape,
        tensor_dtype=tensor_dtype,
        nbytes=nbytes,
        async_op=async_op,
    )
    span_attrs = {k: v for k, v in meta.items() if k != "source"}
    cm = span(op, source="collective_tracer", comm=_comm_label(op), **span_attrs)
    cm.__enter__()
    return cm, meta


def finish_comm_span(
    cm,
    meta: dict,
    *,
    op: str,
    duration_ms: float,
    group_rank: int,
    group_size: int,
) -> None:
    """Close span and append ``python.comm_collective`` row (``full`` mode)."""
    if cm is not None:
        cm.__exit__(None, None, None)

    row = {**meta, "op": op, "group_rank": group_rank, "group_size": group_size}
    CommCollective(duration_ms=duration_ms, **row).save()
