//! Per-table mmap ring capacity + retention defaults.
//!
//! Two orthogonal knobs share this file (both consumed by Python
//! `ExternalTable`, CPU/GPU collectors, and env / SET overrides):
//!
//! 1. **Size** (Pillar C PR-1): ring capacity in MiB. Env
//!    ``PROBING_EXTTBL_<TABLE>_MB``.
//! 2. **Retention** (Pillar C PR-3): keep the last N *steps* or T *seconds*
//!    of data before the ring is allowed to recycle a chunk. Envs
//!    ``PROBING_EXTTBL_<TABLE>_RETAIN_STEPS`` /
//!    ``..._RETAIN_SECS``.
//!
//! Retention is an *advisory* target enforced at the write layer (see
//! `exttbls.rs`): a chunk is only recycled if all its rows are older than
//! `now - retain_*`, otherwise the chunk is sealed and the writer warns
//! `retention truncated: cap=X rows/secs, requested=<N>`. The MEMT ring
//! itself does not read these values.

const DEFAULT_NUM_CHUNKS: u32 = 8;
const MIN_CHUNK_BYTES: u32 = 4 * 1024;
const MAX_CHUNK_BYTES: u32 = 16 * 1024 * 1024;
const GENERIC_DEFAULT_MB: usize = 20;

/// Default hot-ring budget in MiB for well-known tables.
pub fn per_table_default_mb(name: &str) -> usize {
    match name {
        "cpu.utilization" | "cpu.tasks" | "gpu.utilization" => 8,
        "gpu.hccs" => 4,
        _ => GENERIC_DEFAULT_MB,
    }
}

/// Env key: ``PROBING_EXTTBL_CPU_UTILIZATION_MB`` for ``cpu.utilization``.
fn env_key_suffix(name: &str, suffix: &str) -> String {
    format!(
        "PROBING_EXTTBL_{}_{}",
        name.replace('.', "_").to_uppercase(),
        suffix
    )
}

fn env_override_mb(name: &str) -> Option<usize> {
    let key = env_key_suffix(name, "MB");
    std::env::var(&key)
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|&v| v > 0)
}

/// Total ring capacity in bytes for *name* (env override wins).
pub fn table_ring_capacity_bytes(name: &str) -> usize {
    let mb = env_override_mb(name).unwrap_or_else(|| per_table_default_mb(name));
    mb.saturating_mul(1024 * 1024)
}

/// ``(chunk_size_bytes, num_chunks)`` for [`ExposedTable::create`].
pub fn table_mmap_chunk_layout(name: &str) -> (u32, u32) {
    let total = table_ring_capacity_bytes(name) as u64;
    let num = DEFAULT_NUM_CHUNKS;
    let chunk = ((total / num as u64) as u32).clamp(MIN_CHUNK_BYTES, MAX_CHUNK_BYTES);
    (chunk, num)
}

// ── Retention (PR-3) ─────────────────────────────────────────────────

/// Default retention window (steps) for tables whose primary index is
/// a monotonic training step (torch_trace, comm_collective).
///
/// Handbook §3.2: 500 steps covers a typical retrospective window
/// W ∈ [100, 200] with 2× margin.
pub fn per_table_default_retain_steps(name: &str) -> Option<u32> {
    match name {
        "python.torch_trace" | "python.comm_collective" => Some(500),
        _ => None,
    }
}

/// Default retention window (seconds) for tables whose primary index is
/// wall-clock time (cpu.utilization, gpu.utilization).
///
/// Handbook §3.2: 1 hour covers host slow-leak / HBM slow-decay windows.
pub fn per_table_default_retain_secs(name: &str) -> Option<u32> {
    match name {
        "cpu.utilization" | "gpu.utilization" => Some(3600),
        _ => None,
    }
}

fn parse_env_u32(key: &str) -> Option<u32> {
    std::env::var(key)
        .ok()
        .and_then(|s| s.trim().parse::<u32>().ok())
        .filter(|&v| v > 0)
}

/// Retention window in **steps** for *name*, honoring env override.
///
/// Precedence: ``PROBING_EXTTBL_<TABLE>_RETAIN_STEPS`` > per-table default.
pub fn table_retain_steps(name: &str) -> Option<u32> {
    let key = env_key_suffix(name, "RETAIN_STEPS");
    parse_env_u32(&key).or_else(|| per_table_default_retain_steps(name))
}

/// Retention window in **seconds** for *name*, honoring env override.
///
/// Precedence: ``PROBING_EXTTBL_<TABLE>_RETAIN_SECS`` > per-table default.
pub fn table_retain_secs(name: &str) -> Option<u32> {
    let key = env_key_suffix(name, "RETAIN_SECS");
    parse_env_u32(&key).or_else(|| per_table_default_retain_secs(name))
}

/// Bundle of retention hints applied to a table when it's created.
///
/// Both fields are independent hints; both may be `Some` if the user
/// wants "retain the max of steps OR secs". Consumers (exttbls.rs
/// write path) decide policy.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TableRetention {
    pub retain_steps: Option<u32>,
    pub retain_secs: Option<u32>,
}

impl TableRetention {
    pub fn is_empty(&self) -> bool {
        self.retain_steps.is_none() && self.retain_secs.is_none()
    }
}

/// Full retention bundle for *name* (env override wins over defaults).
pub fn table_retention(name: &str) -> TableRetention {
    TableRetention {
        retain_steps: table_retain_steps(name),
        retain_secs: table_retain_secs(name),
    }
}

/// Config-key form (`probing.exttbl.<table>.retain_steps` / `..retain_secs` /
/// `..size_mb`). Reserved for future `SET` integration; the current write
/// path reads env only, but this helper documents the accepted namespace.
pub fn config_key(table: &str, suffix: &str) -> String {
    format!("probing.exttbl.{table}.{suffix}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn defaults_match_pillar_c_pr1() {
        assert_eq!(per_table_default_mb("cpu.utilization"), 8);
        assert_eq!(per_table_default_mb("cpu.tasks"), 8);
        assert_eq!(per_table_default_mb("gpu.utilization"), 8);
        assert_eq!(per_table_default_mb("gpu.hccs"), 4);
        assert_eq!(per_table_default_mb("python.torch_trace"), 20);
    }

    #[test]
    fn env_override_mb() {
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::set_var("PROBING_EXTTBL_GPU_UTILIZATION_MB", "16");
        assert_eq!(table_ring_capacity_bytes("gpu.utilization"), 16 * 1024 * 1024);
        std::env::remove_var("PROBING_EXTTBL_GPU_UTILIZATION_MB");
    }

    #[test]
    fn layout_is_eight_chunks() {
        let (chunk, num) = table_mmap_chunk_layout("cpu.utilization");
        assert_eq!(num, 8);
        assert!(chunk as u64 * num as u64 >= 8 * 1024 * 1024);
    }

    // ── PR-3 retention ─────────────────────────────────────────────

    #[test]
    fn retain_steps_default_by_table() {
        assert_eq!(per_table_default_retain_steps("python.torch_trace"), Some(500));
        assert_eq!(per_table_default_retain_steps("python.comm_collective"), Some(500));
        assert_eq!(per_table_default_retain_steps("cpu.utilization"), None);
        assert_eq!(per_table_default_retain_steps("gpu.hccs"), None);
    }

    #[test]
    fn retain_secs_default_by_table() {
        assert_eq!(per_table_default_retain_secs("cpu.utilization"), Some(3600));
        assert_eq!(per_table_default_retain_secs("gpu.utilization"), Some(3600));
        assert_eq!(per_table_default_retain_secs("python.torch_trace"), None);
    }

    #[test]
    fn retain_steps_env_override() {
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::set_var("PROBING_EXTTBL_PYTHON_TORCH_TRACE_RETAIN_STEPS", "123");
        assert_eq!(table_retain_steps("python.torch_trace"), Some(123));
        std::env::remove_var("PROBING_EXTTBL_PYTHON_TORCH_TRACE_RETAIN_STEPS");
        // fallback to default
        assert_eq!(table_retain_steps("python.torch_trace"), Some(500));
    }

    #[test]
    fn retain_secs_env_override() {
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::set_var("PROBING_EXTTBL_CPU_UTILIZATION_RETAIN_SECS", "1800");
        assert_eq!(table_retain_secs("cpu.utilization"), Some(1800));
        std::env::remove_var("PROBING_EXTTBL_CPU_UTILIZATION_RETAIN_SECS");
        assert_eq!(table_retain_secs("cpu.utilization"), Some(3600));
    }

    #[test]
    fn retention_bundle_populated_when_defaults_apply() {
        let r = table_retention("python.torch_trace");
        assert_eq!(r.retain_steps, Some(500));
        assert_eq!(r.retain_secs, None);
        assert!(!r.is_empty());

        let r2 = table_retention("gpu.utilization");
        assert_eq!(r2.retain_secs, Some(3600));

        let r3 = table_retention("nccl.proxy_ops");
        assert!(r3.is_empty());
    }

    #[test]
    fn config_key_namespace() {
        assert_eq!(
            config_key("python.torch_trace", "retain_steps"),
            "probing.exttbl.python.torch_trace.retain_steps"
        );
    }

    #[test]
    fn env_override_zero_rejected() {
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::set_var("PROBING_EXTTBL_PYTHON_TORCH_TRACE_RETAIN_STEPS", "0");
        // parse_env_u32 filters v > 0 → fall back to default
        assert_eq!(table_retain_steps("python.torch_trace"), Some(500));
        std::env::remove_var("PROBING_EXTTBL_PYTHON_TORCH_TRACE_RETAIN_STEPS");
    }
}
