# Ascend NPU 适配说明（`feat/ascend-npu` 分支）

本分支在 upstream `master` 基础上，为华为昇腾 NPU 环境增加**轻量适配**，不改动 CUDA 路径。

## 改动摘要

| 区域 | 改动 |
|------|------|
| Rust `probing-gpu` | 新增 `Npu` backend：**优先 DCMI（`libdcmi.so`）** 采 util/HBM/温度/功率，失败则回退 `npu-smi` |
| Rust `gpu.hccs` | 独立采集器：`npu-smi -t hccs`（可选 `hccs-bw`），NVLink 带宽代理 |
| Rust `gpu.utilization` | 增补 `aivector_util_pct`、`hbm_bw_util_pct`（tensor / HBM BW 代理） |
| Python `torch_probe` | `_get_backend()` 优先检测 `torch.npu`（需 `torch_npu`） |
| Python `timing/backend.py` | 已有 NPU 支持（本分支未改逻辑，仅对齐优先级） |
| 环境变量 | 见下表 |

## Minder 7 字段 ↔ Ascend / Probing 映射

| Minder 字段 | Probing 表.列 | 状态 |
|-------------|---------------|------|
| PFC tx packet rate | `rdma.mlx_hca.pfc_tx_rate`（`pfc_available`） | 尽力：扫 IB `*pfc*`/`*pause*` 与 netdev pause；多数 Ascend Pod **无计数器** → `pfc_available=0` |
| CPU usage | `cpu.utilization` | 已有 |
| GPU duty cycle | `gpu.utilization.gpu_util_pct`（Aicore / NPU util） | 已有；TP 组内噪声偏大 |
| GPU power draw | `gpu.utilization.power_w` | 已有（DCMI / smi） |
| Graphics engine activity | — | **无 Ascend API**；保持 `-1`（`renderer_util_pct`） |
| Tensor activity | `gpu.utilization.aivector_util_pct` | 代理：`npu-smi` Aivector；DCMI 路径暂无 |
| NVLink bandwidth | `gpu.hccs.tx_bps` / `rx_bps`（可选 `bw_*_gbs`） | 新增；NPU 上默认 auto 开启 |

补充（非 Minder 但有用）：`hbm_bw_util_pct`、`mem_controller_util_pct`（HBM 占用）、`python.comm_collective` 时长。

## 环境变量

| 变量 | 默认 | 说明 |
|------|------|------|
| `PROBING_GPU_BACKEND` | `npu`（cases）/ `auto` | 选 NPU 后端 |
| `PROBING_NPU_SOURCE` | `auto` | `dcmi` \| `smi` \| `auto` |
| `PROBING_DCMI_LIB` | — | 指定 `libdcmi.so` |
| `PROBING_GPU_SAMPLE_MS` | cases 内 3s | util 采样间隔 |
| `PROBING_HCCS` | auto | `on`/`off`；默认在探测到 NPU backend 时开启 |
| `PROBING_HCCS_SAMPLE_MS` | `1000` | HCCS 计数采样间隔；`0` 关闭 |
| `PROBING_HCCS_BW_EVERY` | `0` | 每 N 次计数采样再跑一次昂贵的 `hccs-bw`；`0`=从不 |

## 在 MindSpeed / Megatron 训练 Job 中使用

```bash
export PROBING=1
export PROBING_GPU_BACKEND=npu    # 或 auto（有 npu-smi 时自动发现）
export PROBING_GPU_SAMPLE_MS=1000
# HCCS 默认 auto-on；需要瞬时 GB/s 时：
# export PROBING_HCCS_BW_EVERY=30

DRY_RUN=0 bash /path/to/probing-cases/scripts/launch_case.sh case01-dp8
```

遥测写入 `gpu.devices` / `gpu.utilization` / `gpu.hccs`（schema 名保留 GPU 前缀，字段 `backend` 值为 `npu`）。

## 限制（后续可迭代）

- **CUDA 专用**：`timing` 模块的 `cuda_event_wait_value32_ffi` 流同步门仍仅支持 CUDA
- **NCCL 技能**：`nccl_*` 诊断面向 NVIDIA；Ascend 上 HCCL 需另开 collector
- **Graphics engine**：无对应指标
- **PFC**：依赖主机 RoCE/IB sysfs；本集群多数节点缺失
- **Web UI**：Dashboard 标签仍显示 “GPU”，数据来自 `gpu.*` 表

## 验证

```bash
cd <本仓库根目录>   # 例如 clone 后的 probing-huawei /
cargo test -p probing-gpu
python -m pytest tests/unit -q -k "npu or gpu or backend"  # 可选
```

在计算节点（有 `npu-smi`）上：

```bash
export PROBING_GPU_BACKEND=npu
probing -t <pid> inject   # 或 PROBING=1 启动训练
# SQL: SELECT device_id, aivector_util_pct, hbm_bw_util_pct, power_w FROM gpu.utilization LIMIT 8;
# SQL: SELECT device_id, tx_bps, rx_bps FROM gpu.hccs LIMIT 8;
```
