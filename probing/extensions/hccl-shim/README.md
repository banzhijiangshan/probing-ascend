# probing-hccl-shim

HCCL loads MSProf via `dlopen("libprofapi.so")`. This crate builds a **`libprofapi.so` shim** that:

1. Proxies the CANN 8.5 `Msprof*` API surface needed by HCCL, runtime, and GE
2. Records events into probing mmap tables (`hccl.*`)
3. Forwards to the real CANN library (`libprofapi.so.real` or `PROBING_HCCL_PROFAPI_REAL`)

## Build

```bash
make hccl-shim-lib
# → python/probing/shim/hccl/libprofapi.so
```

## Deploy on Ascend training

```bash
# 1. Copy real MSProf API next to shim (once per CANN version)
python -m probing.hccl --install-real "$ASCEND_HOME/lib64/libprofapi.so"

# 2. Make HCCL's own dlopen("libprofapi.so") resolve to the proxy. LD_PRELOAD
# alone is insufficient because CANN 8.5 obtains the detailed-report functions
# with dlsym on its own libprofapi handle.
export LD_LIBRARY_PATH="$(python -m probing.hccl --shim-dir):${LD_LIBRARY_PATH:-}"

# 3. probing memtable + optional debug
export PROBING=2
export PROBING_HCCL_SHIM_LOG=1   # optional
export PROFILING_MODE=true

# 4. Enable CANN/HCCL profiling as usual, then train
torchrun ... train.py
```

Query (same process or after inject):

```sql
SELECT count(*) FROM hccl.host_ops;
SELECT count(*) FROM hccl.collectives;
SELECT * FROM hccl.tasks LIMIT 20;
SELECT * FROM hccl.mc2_streams LIMIT 10;
SELECT * FROM global.hccl.tasks LIMIT 20;  -- multi-rank
```

## Tables

| SQL table | Source | Key columns |
|-----------|--------|-------------|
| `hccl.host_ops` | `MsprofReportApi` | `event_class`, `item_name`, `duration_ns`, timing |
| `hccl.collectives` | `MsprofReportCompactInfo` (HcclOpInfo) + host HCCL op API | `row_source` (`api`/`compact`), `count`, `group_hash`, `alg_hash` |
| `hccl.tasks` | `MsprofReportAdditionalInfo` → `MsprofHcclInfo` | `task_name`, `plane_index`, `rank_in_plane`, `data_size` |
| `hccl.mc2_streams` | MC2 comm AdditionalInfo | `comm_stream_ids`, `rank_id`, `aicpu_kfc_stream_id` |
| `hccl.context_ids` | ContextId AdditionalInfo | `ctx_id_min`, `ctx_id_max` |

Join collective timing with metadata:

```sql
SELECT a.op_name, a.duration_ns, c.count, c.group_hash, c.alg_hash
FROM hccl.collectives a
JOIN hccl.collectives c
  ON a.thread_id = c.thread_id
 AND a.row_source = 'api'
 AND c.row_source = 'compact'
 AND abs(a.ts - c.ts) < 1000000;
```

## Name resolution

`MsprofRegTypeInfo` and `MsprofGetHashId` populate a hash→name cache. Known HCCL task/op strings are pre-seeded on first call so `item_name` / `event_class` decode without waiting for runtime registration.

## Real library resolution

1. `PROBING_HCCL_PROFAPI_REAL`
2. `<shim-dir>/libprofapi.so.real`
3. `$ASCEND_HOME`, `$ASCEND_HOME_PATH`, or `$ASCEND_INSTALL_PATH`, including
   `lib64` and architecture-qualified `aarch64-linux/lib64` / `x86_64-linux/lib64`

The shim never `dlopen("libprofapi.so")` by bare name (would reload itself).

## Notes

- Struct layouts are validated against the CANN 8.5 public MSProf headers; pin
  CANN versions and run the ABI fixture tests before deploying a different major release.
- AdditionalInfo routing uses CANN level/type IDs first, with registered names and data lengths as compatibility fallbacks.
- Profiling must be enabled (`GetIfProfile()` / MSProf subscribe) or tables stay empty.
- Non-Linux: crate builds for CI; plugin symbols are Linux-only.
