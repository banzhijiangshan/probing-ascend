# 华为昇腾（Ascend NPU）版 Probing — 给 Agent / 新人的使用说明

> **先读本文件。** 上游通用产品说明见 [`README.md`](README.md) / [`README.cn.md`](README.cn.md)；NPU 适配细节见 [`docs/NPU_ADAPT.md`](docs/NPU_ADAPT.md)。
> 本树是在 upstream Probing 上加了 **Ascend NPU 采集** 的工作副本（DCMI / `npu-smi` / HCCS），用于 MindSpeed / Megatron 训练侧遥测与 fail-slow 诊断。

---

## 0. 30 秒定位

| 问题 | 答案 |
|------|------|
| 这是什么 | 进程内探针：挂到训练进程后，用 SQL 查 `gpu.*` / `cpu.*` / Torch step / collective 等表 |
| 和 NVIDIA 版差在哪 | GPU 后端走 **NPU**（`PROBING_GPU_BACKEND=npu`），功率/温度/利用率来自 DCMI 或 `npu-smi`；机间带宽代理表是 `gpu.hccs` |
| 别人 clone 下来能不能用 | **库代码本身不依赖你们机器上的绝对路径**；用环境变量配置即可。训练 Job 脚本（`probing-cases`）另说 |
| 有没有绑死 `/data/yysong` | **源码没有写死共享盘路径**；文档里若出现本机路径仅为示例 |
| 遥测写到哪 | 默认进程内 mmap（Linux 常在 `/dev/shm/probing/<pid>/`），可用 `PROBING_DATA_DIR` / `PROBING_EXPORT_DIR` 改 |

---

## 1. 仓库身份（GitHub / 路径）

- **本目录当前状态**：工作区副本；元数据里仍指向上游仓库名（如 DeepLink-org / reiase 的 `probing`）。
  **以本树内容为准**；是否已 push 到独立 GitHub 远程，以 `git remote -v` 为准。
- **上游产品**：通用 Probing（SQL 引擎、inject、skills、MCP）。本 fork 增量主要在 `probing/extensions/gpu` 的 NPU backend 与 `gpu.hccs`。
- **同机相关但不在本仓库**：训练 case / 启动脚本在旁路目录 `probing-cases`（若存在）；不要和本库混为一谈。

---

## 2. 硬路径结论（别人能不能方便用）

| 类别 | 结论 |
|------|------|
| Rust / Python **库代码** | 路径靠 **环境变量**（`PROBING_DATA_DIR`、`PROBING_DCMI_LIB`、`PROBING_CTRL_ROOT` 等），**无** `/data/yysong`、`/afs-a3-...` 写死依赖 |
| 默认落盘 | Linux：`PROBING_DATA_DIR` 未设时常见 `/dev/shm/probing`；控制套接字默认 `/tmp/probing/` |
| 文档示例 | `docs/NPU_ADAPT.md` 等可能出现本机路径，**复制时请改成你的 clone 路径** |
| 训练入口脚本 | 若使用外部的 `probing-cases`，那些脚本可能写死共享盘——那是 **cases 层**，不是本库 |

**别人下载后最小可用路径：** clone → 本机构建 wheel 或 `make develop` → 在有 NPU 的机器上 `PROBING=1` 起训练或 `probing -t <pid> inject`。

---

## 3. 构建（开发机 / aarch64 常见）

依赖：Rust（`cargo`）、Python 3、`maturin`。

```bash
cd <本仓库根目录>
export CARGO_HOME="${CARGO_HOME:-$HOME/.cargo}"
export PATH="$HOME/.cargo/bin:$PATH"

# 可编辑开发
make develop

# 打 wheel（给训练 Pod pip 安装）
make wheel
# 产物一般在 target/wheels 或 dist/；拷到你自己的 wheels 目录即可
```

Linux 默认 maturin features：`extension-module,gpu,gpu-cuda,kmsg`（见 `Makefile`）。
无 NPU 的机器上 `make test` 部分用例会 skip/fail，属预期。

---

## 4. 在训练里打开（推荐）

在 **Ascend 训练进程启动前** 设置（MindSpeed / torchrun 父进程要能传到 worker）：

```bash
export PROBING=1                    # 或 2：当前进程 + 子进程都挂探针
export PROBING_GPU_BACKEND=npu      # 或 auto（有 npu-smi 时会发现 NPU）
export PROBING_NPU_SOURCE=auto      # dcmi | smi | auto
export PROBING_GPU=on
export PROBING_GPU_SAMPLE_MS=1000
export PROBING_CPU=on
export PROBING_CPU_SAMPLE_MS=1000
# HCCS（NVLink 带宽代理）默认在 NPU backend 下 auto-on
# export PROBING_HCCS=on
# export PROBING_HCCS_BW_EVERY=30   # 昂贵 hccs-bw，默认 0=不开

# Torch / Megatron（按需；MetaX 与 NPU 行为不同，NPU 上可开）
# export PROBING_TORCH_PROFILING=on
# export PROBING_MEGATRON=on

# 可选：指定落盘与导出
# export PROBING_DATA_DIR=/dev/shm/probing-myjob
# export PROBING_EXPORT_DIR=/path/to/exports
```

然后照常启动训练。进程内会起 HTTP + SQL 引擎。

**已跑起来再挂：**

```bash
probing -t <训练 PID> inject
probing -t <pid> query "SELECT device_id, gpu_util_pct, power_w, temp_c FROM gpu.utilization LIMIT 8"
```

---

## 5. NPU 上常见能查的表 / 字段

表名仍用 `gpu.*` 前缀（历史兼容）；`backend` 字段为 `npu`。

| 表 | 用途 | 典型列 |
|----|------|--------|
| `gpu.devices` | 设备静态信息 | device_id、型号、显存容量 |
| `gpu.utilization` | 秒级采样 | `gpu_util_pct`、`power_w`、`temp_c`、`mem_used_pct`、`aivector_util_pct`、`hbm_bw_util_pct`… |
| `gpu.hccs` | 机间 HCCS 带宽代理 | `tx_bps` / `rx_bps`（可选更贵的 bw） |
| `cpu.utilization` | 进程 CPU | 利用率 / RSS 等 |
| `python.torch_step_timing` | step 墙钟（开 Torch/Megatron 钩子后） | `step_duration_sec` |
| `python.comm_collective` | 集合通信墙钟 | `duration_ms`、collective 名 |

**已知弱项 / 常抓不到：**

- Minder 式 **PFC** 计数：多数 Ascend Pod 无对应 sysfs → `pfc_available=0`
- **Graphics engine / Tensor pipe** 细粒度：无原生 API（部分用 aivector 代理）
- NVIDIA 专用 **NCCL profiler** 技能：不能直接套；HCCL 需另路径
- 环形缓冲默认只保留近段时间；长跑要靠导出 / cold / 外部 sidecar

字段筛选实验结论（若同组织有）：见旁路 `question_0721_probe/FIELD_SCREEN_REPORT.md`（不在本仓库内时忽略）。

---

## 6. Agent 怎么用

```bash
./skills/install.sh          # 装到 .cursor / .claude / .agents
probing skill list
probing -t <pid> skill run health_overview
```

MCP：探针起来后连 `http://<host>:<port>/mcp`。默认只读；写操作需 `PROBING_MCP_ALLOW_WRITE=1`。
改代码分层约定：[`AGENTS.md`](AGENTS.md)。

---

## 7. 最小验收清单（给 AI 执行）

1. `python -c "import probing, probing._core; print('ok')"`
2. 有 NPU 的节点上：`npu-smi info | head` 可用
3. `PROBING=1 PROBING_GPU_BACKEND=npu` 起一小段训练或 inject 到已有进程
4. `query`：`gpu.utilization` 出现非空 `power_w` / `gpu_util_pct`（DCMI 或 smi 至少一条通路通）
5. 需要带宽时再查 `gpu.hccs`

失败时优先查：wheel 是否装到训练用的同一个 Python；`PROBING` 是否传到 worker；`PROBING_GPU_BACKEND` 是否误设成 `cuda`。

---

## 8. 相关文件地图

```
README.NPU.md          ← 本文件（NPU fork 入口）
README.md / README.cn.md ← 上游通用产品说明
docs/NPU_ADAPT.md      ← Ascend 适配与 Minder 字段映射
docs/src/reference/env-vars.md ← 全量环境变量
AGENTS.md              ← 改代码 / skills 约定
python/probing/        ← Python 钩子与表
probing/extensions/gpu ← NPU / CUDA GPU 后端（Rust）
```

---

*维护：NPU 使用与路径约定优先改本文件；通用产品能力改 upstream 风格的 README；适配实现细节改 `docs/NPU_ADAPT.md`。*
