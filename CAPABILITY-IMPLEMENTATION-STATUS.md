# Capability handoff implementation status

中文开发改动说明见 [`CAPABILITY-DEVELOPMENT-CHANGES_zh.md`](CAPABILITY-DEVELOPMENT-CHANGES_zh.md)。

Baseline: `probing-capability-handoff` dated 2026-08-03. This file separates
repository-complete work from hardware acceptance; an implementation is not
reported as experimentally accepted until its stated NPU/card-count criterion
has been measured.

| ID | Repository status | Remaining acceptance gate |
|---|---|---|
| A1 | Existing per-step wall timing retained; step rows now use the completed-step snapshot so `torch_step_timing` and `torch_trace` do not drift by one step. | 8-card, 200-step continuity/no-NaN check. |
| A2 | Implemented boundary clocks, explicit `pre/post backward`, and canonical `torch_phase_accounting` SQL in `scripts/capability_views.sql`. | Confirm actual stage set/timebase and median residual <5% on NPU. |
| A3 | The Probing-side five-field accounting surface now exists. | Requires the training producer, real `rank_*.jsonl`, and the paper's exact `step_accounting_analysis.py`; run all five equivalence criteria on >=500 8-card steps. |
| A4 | Implemented canonical `step_windows(rank, local_step, ts_start, ts_end)` view using hook-time Unix boundaries. | Run at `PROBING_GPU_SAMPLE_MS<=200`; coverage >=90% and reproduce the util/power reversal. |
| B1 | Existing table-level five-tier switches documented. | Launch/query all five tiers. |
| B2 | Added `scripts/capability_cost_sampler.py`; it separately emits cumulative cold bytes (including TTL-deleted segments), current cold bytes, hot bytes, `/proc` write bytes, and RSS at <=1 s cadence. | Run all five tiers; compare cold-byte and process-I/O deltas and report peaks. Use a run-specific `PROBING_COLD_DIR` so unrelated processes are excluded. |
| C1/C3 | Shadow timing and federated query engine already exist. | DiD/shadow overhead runs and 8/32/64-card query latency/bytes curve. |
| D3 | Sealed-segment recovery exists; no claim of zero loss was added. | Five trials each of SIGTERM, SIGKILL, and HCCL watchdog abort; measure loss bound and independent read. |
| E3 | Not fabricated: no 27/27 executable result is claimed. | Requires the supplied 27-case definition plus representative real tables for every predicate; record SQL/YAML LOC, table count, and column count. |
| F1 | Ascend remains `site_hook`, not transparent ptrace. | Optional NVIDIA ptrace validation only. |
| G1 | Main-thread live sync already exists; the required `on,rate=0` startup placeholder is now explicit in English/Chinese docs. | Ascend `rate=0 -> SET rate=1` latency <=3 steps; disabled-first-step failure is an acknowledged boundary. |
| G2 | No control-plane claim added for `eval`. | Ascend 10-trial matrix for single expression / semicolon / import. |
| G3/G4/G5/G7 | Added the six required semantic warnings to the SQL catalog and reference docs. | Query `probe.probing.column_docs` in a built wheel and archive the result. |
| G8 | Added startup-scoped `PROBING_TORCH_PROFILING_RANKS`; `node0` implements the existing rank-0-node workaround and invalid selectors fail closed. | Preserve pre-fix 8/16/32/64 import data, then verify scoped 32/64-card failure rate <=10%. |

## B2 sampler

Use a dedicated hot/cold directory for each arm:

```bash
export PROBING_DATA_DIR=/dev/shm/probing-b2-l4
export PROBING_COLD_DIR=/afs-a3-weight-share/yinjinrun.p-huawei/b2-l4/cold
python scripts/capability_cost_sampler.py \
  --pid "$TRAIN_PID" \
  --hot-dir "$PROBING_DATA_DIR" \
  --cold-dir "$PROBING_COLD_DIR" \
  --output b2-l4-cost.csv \
  --interval 1
```

The watcher must start before the run. Starting it after TTL deletion cannot
recover segments it never observed.

## Go/no-go

The code-side blockers identified by the handoff now have concrete interfaces
and repeatable queries. The kilocard go/no-go remains **NO-GO until the hardware
gates above are archived**; local unit tests cannot substitute for the required
Ascend scale experiments.
