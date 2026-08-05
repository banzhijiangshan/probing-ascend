# 能力交接开发改动说明

本文档记录根据 `probing-capability-handoff`（2026-08-03）完成的代码侧补充。
本次开发覆盖 A2/A3/A4、B2、G8 以及相关数据语义、测试和运维文档。
文中“已完成”仅表示代码和可重复验证接口已经落库，不代表 8/16/32/64 卡
Ascend 实验已经通过。

## 1. PyTorch 阶段边界与 step 对齐

文件：`python/probing/profiling/torch_probe.py`

`python.torch_trace` 新增两个 hook 边界时钟：

- `wall_time_sec`：hook 发生时的 Unix 秒，用于和 `gpu.utilization` 等周期表的
  Unix 微秒时间戳关联。
- `monotonic_time_sec`：hook 发生时的单调时钟秒，用于计算阶段跨度和阶段间 gap，
  避免系统时钟调整造成负耗时。

异步 GPU/NPU event 可能在若干 optimizer step 后才落盘。这两个字段在 hook
发生时写入，因此不会被延迟落盘时间污染。

本次同时完成以下修复：

- 保留 `pre backward` 边界行，并与 `post backward` 共同构成完整 backward 区间。
- 为 forward、backward 和 optimizer 的前后边界统一写入上述时钟。
- 在 optimizer post hook 入口保存已完成 step 的坐标快照。
- 写入 `python.torch_step_timing` 时使用该快照，避免 `train.step()` 已推进后把耗时
  错记到下一个 step。

因此 `python.torch_trace` 和 `python.torch_step_timing` 现在可以按
`rank, local_step` 稳定关联。

## 2. A2/A3：统一阶段账本

文件：`scripts/capability_views.sql`

新增 `torch_phase_boundaries` 和 `torch_phase_accounting` 两个会话级 SQL view。
标准输出包括：

- `data_gap_ms`
- `forward_ms`
- `backward_ms`
- `compute_ms`
- `wait_gap_ms`
- `optimizer_ms`
- `step_ms`
- `residual_ms`
- `residual_ratio`

其中阶段差分统一使用 `monotonic_time_sec`。`residual_ratio` 用来检查组成阶段与
整步墙钟是否闭合，目标验收口径为中位数小于 5%，但该阈值必须在真实 NPU
数据上验证，SQL 本身不伪造或硬编码通过结论。

使用前需要开启完整阶段采样：

```bash
export PROBING=2
export PROBING_TORCH_PROFILING=on,rate=1.0,backward=on
```

随后在查询会话中执行：

```sql
-- 载入 scripts/capability_views.sql 后
SELECT *
FROM torch_phase_accounting
ORDER BY local_step, rank;
```

## 3. A4：周期指标归属到训练 step

文件：`scripts/capability_views.sql`

新增标准视图：

```text
step_windows(rank, local_step, ts_start, ts_end)
```

它使用相邻 optimizer 结束边界构造每个 step 的 Unix 微秒时间窗，可将 NPU
利用率、功率和 HBM 带宽等周期数据关联到具体 step：

```sql
SELECT w.rank, w.local_step,
       AVG(g.gpu_util_pct) AS util,
       AVG(g.power_w) AS power,
       COUNT(*) AS n_samples
FROM step_windows w
JOIN gpu.utilization g
  ON g.ts BETWEEN w.ts_start AND w.ts_end
GROUP BY w.rank, w.local_step;
```

为达到交接要求中的覆盖率目标，验收时需要设置：

```bash
export PROBING_GPU_SAMPLE_MS=200
```

默认 1000 ms 周期无法保证数百毫秒 step 的 90% 采样覆盖率。

## 4. B2：运行成本采样器

文件：`scripts/capability_cost_sampler.py`

新增独立采样脚本，以不超过 1 秒的间隔输出：

```text
ts,cold_bytes_cumulative,cold_bytes_current,hot_bytes,proc_io_write_bytes,rss_kb
```

各字段含义如下：

| 字段 | 含义 |
|---|---|
| `cold_bytes_cumulative` | 运行期间观察到的冷存储累计写入量，文件被 TTL 删除后也不回退 |
| `cold_bytes_current` | 当前仍存在的 `.memc` 文件总量 |
| `hot_bytes` | 当前热存储目录占用 |
| `proc_io_write_bytes` | `/proc/<pid>/io` 中训练进程累计写字节数 |
| `rss_kb` | 训练进程常驻内存 |

示例：

```bash
python scripts/capability_cost_sampler.py \
  --pid "$TRAIN_PID" \
  --hot-dir "$PROBING_DATA_DIR" \
  --cold-dir "$PROBING_COLD_DIR" \
  --output b2-cost.csv \
  --interval 1
```

采样器必须在实验开始前启动，并为每个实验臂使用独立冷存储目录；否则无法恢复
启动前已经被 TTL 删除的数据，也可能把其他进程的数据计入结果。

## 5. G8：限制大规模 profiling 初始化范围

文件：`python/probing/ext/torch.py`

新增启动期变量 `PROBING_TORCH_PROFILING_RANKS`，用于控制哪些 rank 安装昂贵的
PyTorch module hook：

| 配置 | 行为 |
|---|---|
| `all` | 所有 rank，默认值 |
| `none` | 所有 rank 均不初始化 |
| `node0` | 仅 global rank 0 所在节点上的全部 local rank |
| `local0` | 每个节点仅 local rank 0 |
| `rank0` / `global0` | 仅 global rank 0 |
| `0,8-15` | 显式 global rank 列表或闭区间 |

非法 selector 会 fail-closed，即不初始化任何 rank，避免配置拼写错误导致数百个
rank 意外同时安装 hook。Ascend 大规模作业建议：

```bash
export PROBING_TORCH_PROFILING=on,rate=1.0,backward=on
export PROBING_TORCH_PROFILING_RANKS=node0
```

该变量是启动期安全边界，不能在运行中扩大范围。如果计划通过 SQL `SET` 从常驻档
切换为详采档，进程启动时必须使用 `on,rate=0` 安装占位 tracer；首个 optimizer
step 完全禁用 profiling 时，后续不能热安装 module hook，只能重启进程。

## 6. 数据语义约束

文件：`probing/core/resources/tables.yaml`、中英文 SQL 表参考文档。

新增并通过 catalog 测试固定以下告警：

- `gpu.utilization.gpu_util_pct` 是设备 duty cycle，包含 HCCL 自旋等待，不能单独
  当作计算负载，需结合功率和 HBM 带宽判断。
- DCMI 路径未采集 `aivector_util_pct`，不可用时为 `-1`。
- `python.comm_collective.duration_ms` 是 Python API 墙钟时间，不是 NCCL/HCCL
  设备执行时间。
- Ascend 线程 scope 的 `cpu_total_pct` 可能溢出，只能作为辅助信息。
- Ascend 环境的 `wchan` 可能恒为空，不能作为唯一阻塞判据。
- HCCS `error_count` 是累计计数器，只能取 `max`、`last` 或做首尾差，禁止跨行
  直接 `SUM`。

## 7. 测试与验证结果

本次补充了以下自动化覆盖：

- rank selector 的 `all`、`none`、`node0`、`local0`、`rank0`、列表、范围和非法值。
- backward 前后边界及边界时钟。
- `torch_step_timing` 使用已完成 step 快照。
- 冷存储文件增长、TTL 删除和新 segment 出现时的累计字节统计。
- 六项数据语义警告可以通过 semantic catalog 查询。

已完成的本地检查：

- `cargo test -p probing-core semantic_catalog --no-default-features`：12 项通过。
- 目标 semantic warning 回归测试通过。
- `cargo check -p probing-python` 通过。
- Python 语法/轻量 smoke 检查通过。
- `git diff --check` 通过。

本地未完成项：完整 Python/Torch pytest 缺少目标运行时依赖；
`cargo test -p probing-python` 在 macOS 上受本地 Python C 符号链接环境限制，
但 `cargo check` 已通过。

## 8. 仍需真实硬件验收

以下结论不能用本地单元测试替代：

- 8 卡、200 step 的连续性、空值和 step 对齐验证。
- 500 step 以上五字段账本等价性及 `residual_ratio` 中位数小于 5%。
- `PROBING_GPU_SAMPLE_MS<=200` 时周期指标归步覆盖率达到 90%。
- 五个数据采集档位的磁盘、进程 I/O、RSS 和性能成本对比。
- 8/16/32/64 卡下 rank-scoped profiling 的初始化失败率和稳定性。
- 8/32/64 卡联邦查询延迟与传输字节曲线。

因此当前状态是：**代码侧交接目标已补齐；千卡 go/no-go 仍为 NO-GO，直到上述
Ascend 规模实验结果被执行并归档。**
