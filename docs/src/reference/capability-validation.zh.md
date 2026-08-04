# 论文能力验证查询

本页给出 A2/A3/A4 的唯一 SQL 口径。先在目标进程的查询会话执行
[`scripts/capability_views.sql`](../../../scripts/capability_views.sql)，再运行下面的验收查询。
这些 view 是会话内对象；重启进程后需重新创建。

## 前置配置

```bash
export PROBING=2
export PROBING_TORCH_PROFILING=on,rate=1.0,backward=on
export PROBING_GPU_SAMPLE_MS=200
```

`python.torch_trace.monotonic_time_sec` 在 hook 发生时记录单调时钟，作为相位跨度和
gap 的规范时基；`wall_time_sec` 记录同一边界的 Unix 秒，只用于与周期表的 Unix
微秒 `ts` 关联。异步设备事件延迟落盘不会改变二者。`duration` 仍表示设备事件
（或 CPU fallback）测得的模块执行时长，不可与边界时钟混作同一量。首个 discovery
step 和每个 rank 的首个 `LAG` window 允许缺失。

## A2/A3：相位账本

```sql
SELECT rank, local_step,
       data_gap_ms, forward_ms, backward_ms, compute_ms,
       wait_gap_ms, optimizer_ms, step_ms, residual_ms, residual_ratio
FROM torch_phase_accounting
ORDER BY local_step, rank;
```

判据不在 SQL 内硬编码：五个组成量必须为正且非 NULL；`residual_ratio` 中位数应小于
5%，且 residual 不应系统性为负。训练侧 jsonl 对账时，字段映射为
`data_ms→data_gap_ms`、`compute_ms→compute_ms`、`wait_ms→wait_gap_ms`、
`comm_ms→optimizer_ms`、`step_ms→step_ms`。`comm_ms` 这个映射只适用于当前 NPU
训练脚本的 optimizer 窗口，不能与 CUDA 版跨平台比较。

## A4：周期表归步

本地查询没有 `_rank` 注入列，应使用进程自身的 rank 常量；联邦查询则使用 `_rank`：

```sql
SELECT w.local_step, w.rank,
       AVG(g.gpu_util_pct) AS util,
       AVG(g.power_w) AS power,
       COUNT(*) AS n_samples
FROM step_windows w
JOIN gpu.utilization g
  ON g.ts BETWEEN w.ts_start AND w.ts_end
GROUP BY w.local_step, w.rank
ORDER BY w.local_step, w.rank;
```

覆盖率查询：

```sql
WITH associated AS (
  SELECT w.rank, w.local_step, COUNT(g.ts) AS n
  FROM step_windows w
  LEFT JOIN gpu.utilization g
    ON g.ts BETWEEN w.ts_start AND w.ts_end
  GROUP BY w.rank, w.local_step
)
SELECT AVG(CASE WHEN n > 0 THEN 1.0 ELSE 0.0 END) AS coverage
FROM associated;
```

默认 1000 ms 周期无法保证数百毫秒 step 的 90% 覆盖率；验收必须记录将
`PROBING_GPU_SAMPLE_MS` 调至不大于 200 ms 所产生的额外代价。
