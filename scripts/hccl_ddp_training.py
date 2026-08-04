#!/usr/bin/env python3
"""Four-node NPU DDP training smoke test with Probing/HCCL timings."""

import dataclasses
import json
import os
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch_npu  # noqa: F401 - registers the NPU backend


WARMUP_STEPS = 2
MEASURE_STEPS = 8
BATCH_SIZE = 64
HIDDEN_SIZE = 1024


def percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction)))
    return ordered[index]


def stats(values):
    if not values:
        return {"count": 0, "mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    return {
        "count": len(values),
        "mean_ms": statistics.fmean(values),
        "p50_ms": statistics.median(values),
        "p95_ms": percentile(values, 0.95),
        "max_ms": max(values),
    }


def ddp_allreduce_hook(state, bucket):
    """Synchronous gradient all-reduce with per-bucket end-to-end timing."""
    torch.npu.synchronize()
    started = time.perf_counter()
    tensor = bucket.buffer()
    dist.all_reduce(tensor)
    tensor.div_(state["world_size"])
    torch.npu.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1e3
    if state["step"] >= WARMUP_STEPS:
        state["comm_ms"].append(elapsed_ms)
        state["comm_bytes"].append(tensor.numel() * tensor.element_size())
    future = torch.futures.Future()
    future.set_result(tensor)
    return future


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    output_dir = Path(os.environ["TRAIN_OUTPUT_DIR"])
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl", init_method="env://")

    tracer = None
    comm_table_rows = []
    if rank == 0:
        from probing.profiling.collective import trace_all_collectives
        from probing.profiling.collective.record import CommCollective, CommRecordMode

        tracer = trace_all_collectives(
            trace_file=str(output_dir / "python-collective.log"),
            verbose=False,
            cuda_sync=True,
            mode=CommRecordMode.LITE,
            resolve_group_ranks=False,
            trace_event=False,
        )

    torch.manual_seed(20260803)
    model = nn.Sequential(
        nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
        nn.GELU(),
        nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
        nn.GELU(),
        nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
        nn.GELU(),
        nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
    ).npu()
    ddp = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        broadcast_buffers=False,
        bucket_cap_mb=4,
    )
    hook_state = {"world_size": world_size, "step": -1, "comm_ms": [], "comm_bytes": []}
    ddp.register_comm_hook(hook_state, ddp_allreduce_hook)
    optimizer = torch.optim.AdamW(ddp.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    generator = torch.Generator().manual_seed(1701 + rank)
    host_inputs = [torch.randn(BATCH_SIZE, HIDDEN_SIZE, generator=generator) for _ in range(2)]
    host_targets = [torch.randn(BATCH_SIZE, HIDDEN_SIZE, generator=generator) for _ in range(2)]
    step_ms = []
    losses = []

    for step_index in range(WARMUP_STEPS + MEASURE_STEPS):
        hook_state["step"] = step_index
        inputs = host_inputs[step_index % 2].npu(non_blocking=False)
        targets = host_targets[step_index % 2].npu(non_blocking=False)
        torch.npu.synchronize()
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        predictions = ddp(inputs)
        loss = loss_fn(predictions, targets)
        loss.backward()
        optimizer.step()
        torch.npu.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1e3
        if step_index >= WARMUP_STEPS:
            step_ms.append(elapsed_ms)
            losses.append(float(loss.detach().cpu()))
        print(
            f"TRAIN rank={rank} step={step_index} loss={float(loss.detach().cpu()):.6f} "
            f"step_ms={elapsed_ms:.3f}",
            flush=True,
        )

    parameter_checksum = torch.stack(
        [parameter.detach().float().sum() for parameter in ddp.module.parameters()]
    ).sum()
    gathered_checksums = [torch.zeros_like(parameter_checksum) for _ in range(world_size)]
    dist.all_gather(gathered_checksums, parameter_checksum)
    torch.npu.synchronize()
    parameter_checksums = [float(value.cpu()) for value in gathered_checksums]
    checksum_max_delta = max(parameter_checksums) - min(parameter_checksums)

    dist.barrier()
    if rank == 0:
        fields = [field.name for field in dataclasses.fields(CommCollective)]
        comm_table_rows = [
            dict(zip(fields, data)) for _timestamp, data in CommCollective.take(10000)
        ]

    result = {
        "rank": rank,
        "world_size": world_size,
        "warmup_steps": WARMUP_STEPS,
        "measured_steps": MEASURE_STEPS,
        "final_loss": losses[-1],
        "step": stats(step_ms),
        "ddp_bucket_communication": stats(hook_state["comm_ms"]),
        "communicated_bytes": sum(hook_state["comm_bytes"]),
        "probing_python_collective_rows": len(comm_table_rows),
        "probing_python_collective": stats(
            [float(row["duration_ms"]) for row in comm_table_rows if row["op"] == "all_reduce"]
        ),
        "probing_operations": sorted({row["op"] for row in comm_table_rows}),
        "parameter_checksums": parameter_checksums,
        "parameter_checksum_max_delta": checksum_max_delta,
    }
    result_path = output_dir / f"training-result-rank{rank}.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print("RESULT " + json.dumps(result, sort_keys=True), flush=True)

    if tracer is not None:
        tracer.remove_hooks()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
