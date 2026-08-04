//! Background collector for Ascend HCCS link counters → `gpu.hccs`.
//!
//! Kept separate from `gpu.utilization` so expensive `npu-smi -t hccs` (and
//! optional `hccs-bw`) never block the 1 Hz util path.

use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex, MutexGuard,
};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use once_cell::sync::Lazy;
use probing_core::sync::lock_mutex;
use probing_memtable::discover::ExposedTable;
use probing_memtable::{DType, Schema, Value};
use thiserror::Error;

#[cfg(all(not(target_os = "macos"), not(windows)))]
use super::backend::npu::hccs::{
    discover_hccs_targets, sample_hccs_bw, sample_hccs_chip, HccsChipSample,
};

use probing_memtable::ring_config;

const DEFAULT_SAMPLE_INTERVAL_MS: u64 = 1000;

struct PrevCounters {
    at: Instant,
    tx_bytes: u64,
    rx_bytes: u64,
}

/// Autostart interval for HCCS sampling.
///
/// - `PROBING_HCCS=off` → disabled
/// - `PROBING_HCCS_SAMPLE_MS=0` → disabled
/// - `PROBING_HCCS=on` or positive `PROBING_HCCS_SAMPLE_MS` → enabled
/// - default **auto**: enabled when NPU backend is present
pub fn hccs_autostart_interval_ms() -> Option<u64> {
    if matches!(
        std::env::var("PROBING_HCCS").ok().as_deref(),
        Some(v) if matches!(v.trim(), "0" | "off" | "false" | "no")
    ) {
        return None;
    }

    let forced_on = matches!(
        std::env::var("PROBING_HCCS").ok().as_deref(),
        Some(v) if matches!(v.trim(), "1" | "on" | "true" | "yes")
    );

    let ms = std::env::var("PROBING_HCCS_SAMPLE_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok());

    if ms == Some(0) {
        return None;
    }
    if let Some(interval) = ms {
        return Some(interval);
    }
    if forced_on {
        return Some(DEFAULT_SAMPLE_INTERVAL_MS);
    }

    // auto
    #[cfg(all(not(target_os = "macos"), not(windows)))]
    {
        if super::backend::selected_backends()
            .iter()
            .any(|b| b.kind() == super::backend::GpuBackendKind::Npu)
        {
            return Some(DEFAULT_SAMPLE_INTERVAL_MS);
        }
    }
    let _ = forced_on;
    None
}

fn hccs_bw_every_n() -> u64 {
    std::env::var("PROBING_HCCS_BW_EVERY")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(0)
}

pub fn start_hccs_sampling(interval_ms: u64) -> Result<(), HccsCollectorError> {
    match HccsCollector::instance().start(Duration::from_millis(interval_ms)) {
        Ok(()) | Err(HccsCollectorError::AlreadyRunning) => Ok(()),
        Err(e) => Err(e),
    }
}

pub fn start_hccs_sampling_from_env() {
    let Some(interval_ms) = hccs_autostart_interval_ms() else {
        log::debug!("HCCS sampling not started (disabled or no NPU backend)");
        return;
    };
    match start_hccs_sampling(interval_ms) {
        Ok(()) => log::info!(
            "HCCS sampling started (interval={interval_ms}ms, bw_every={})",
            hccs_bw_every_n()
        ),
        Err(HccsCollectorError::AlreadyRunning) => log::debug!("HCCS sampling already running"),
        Err(e) => log::warn!("HCCS sampling start failed: {e}"),
    }
}

fn hccs_schema() -> Schema {
    Schema::new()
        .col("ts", DType::I64)
        .col("device_id", DType::I32)
        .col("npu_id", DType::I32)
        .col("chip_id", DType::I32)
        .col("tx_bytes", DType::I64)
        .col("rx_bytes", DType::I64)
        .col("tx_bps", DType::F32)
        .col("rx_bps", DType::F32)
        .col("tx_packets", DType::I64)
        .col("rx_packets", DType::I64)
        .col("error_count", DType::I64)
        .col("retry_count", DType::I64)
        .col("health_ok", DType::I32)
        .col("bw_tx_gbs", DType::F32)
        .col("bw_rx_gbs", DType::F32)
        .col("wall_ns", DType::I64)
}

#[derive(Error, Debug)]
pub enum HccsCollectorError {
    #[error("HCCS collector already running")]
    AlreadyRunning,
    #[error("Failed to open HCCS memtable")]
    OpenFailed(#[from] probing_memtable::MemtableError),
}

fn ts_micros() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as i64
}

fn bytes_rate(prev_bytes: u64, prev_at: Instant, cur_bytes: u64, now: Instant) -> f32 {
    let dt = now.duration_since(prev_at).as_secs_f64();
    if dt <= 1e-6 {
        return -1.0;
    }
    let delta = if cur_bytes >= prev_bytes {
        cur_bytes - prev_bytes
    } else {
        // counter wrap / reset
        cur_bytes
    };
    (delta as f64 / dt) as f32
}

fn push_row(
    table: &mut ExposedTable,
    ts: i64,
    wall_ns: u64,
    sample: &HccsChipSample,
    tx_bps: f32,
    rx_bps: f32,
    bw_tx: f32,
    bw_rx: f32,
) {
    if !table.push_row(&[
        Value::I64(ts),
        Value::I32(sample.device_id),
        Value::I32(sample.npu_id),
        Value::I32(sample.chip_id),
        Value::I64(sample.tx_bytes as i64),
        Value::I64(sample.rx_bytes as i64),
        Value::F32(tx_bps),
        Value::F32(rx_bps),
        Value::I64(sample.tx_packets as i64),
        Value::I64(sample.rx_packets as i64),
        Value::I64(sample.error_count as i64),
        Value::I64(sample.retry_count as i64),
        Value::I32(if sample.health_ok { 1 } else { 0 }),
        Value::F32(bw_tx),
        Value::F32(bw_rx),
        Value::I64(wall_ns as i64),
    ]) {
        log::warn!("hccs collector: push_row failed for gpu.hccs");
    }
}

fn lock_table(m: &Mutex<ExposedTable>) -> MutexGuard<'_, ExposedTable> {
    lock_mutex(m, "hccs memtable")
}

fn lock_collector<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    lock_mutex(m, "hccs collector")
}

struct HccsCollector {
    running: Arc<AtomicBool>,
    handle: Mutex<Option<JoinHandle<()>>>,
    table: Mutex<Option<Arc<Mutex<ExposedTable>>>>,
}

impl HccsCollector {
    fn instance() -> &'static Self {
        static INSTANCE: Lazy<HccsCollector> = Lazy::new(|| HccsCollector {
            running: Arc::new(AtomicBool::new(false)),
            handle: Mutex::new(None),
            table: Mutex::new(None),
        });
        &INSTANCE
    }

    fn shared_table(&self) -> probing_memtable::MemtableResult<Arc<Mutex<ExposedTable>>> {
        let mut guard = lock_collector(&self.table);
        if guard.is_none() {
            let (chunk_size, num_chunks) =
                ring_config::table_mmap_chunk_layout("gpu.hccs");
            *guard = Some(Arc::new(Mutex::new(ExposedTable::create(
                "gpu.hccs",
                &hccs_schema(),
                chunk_size,
                num_chunks,
            )?)));
        }
        guard.as_ref().cloned().ok_or_else(|| {
            probing_memtable::MemtableError::InvalidBuffer("hccs table unavailable")
        })
    }

    fn start(&self, interval: Duration) -> Result<(), HccsCollectorError> {
        if self.running.swap(true, Ordering::SeqCst) {
            return Err(HccsCollectorError::AlreadyRunning);
        }

        let running = self.running.clone();
        let table = self.shared_table()?;
        let bw_every = hccs_bw_every_n();

        let handle = thread::spawn(move || {
            #[cfg(all(not(target_os = "macos"), not(windows)))]
            {
                let targets = discover_hccs_targets();
                if targets.is_empty() {
                    log::warn!("HCCS collector: no NPU chips discovered; exiting");
                    running.store(false, Ordering::SeqCst);
                    return;
                }
                let mut prev: HashMap<i32, PrevCounters> = HashMap::new();
                let mut tick: u64 = 0;
                while running.load(Ordering::SeqCst) {
                    let wall_start = Instant::now();
                    let ts = ts_micros();
                    let now = Instant::now();
                    let do_bw = bw_every > 0 && tick % bw_every == 0;
                    tick = tick.wrapping_add(1);

                    let mut exposed = lock_table(&table);
                    for &(npu_id, chip_id, device_id) in &targets {
                        let Some(sample) = sample_hccs_chip(npu_id, chip_id, device_id) else {
                            continue;
                        };
                        let (tx_bps, rx_bps) = if let Some(p) = prev.get(&device_id) {
                            (
                                bytes_rate(p.tx_bytes, p.at, sample.tx_bytes, now),
                                bytes_rate(p.rx_bytes, p.at, sample.rx_bytes, now),
                            )
                        } else {
                            (-1.0, -1.0)
                        };
                        let (bw_tx, bw_rx) = if do_bw {
                            sample_hccs_bw(npu_id, chip_id)
                                .map(|b| (b.tx_gbs, b.rx_gbs))
                                .unwrap_or((-1.0, -1.0))
                        } else {
                            (-1.0, -1.0)
                        };
                        let wall_ns = wall_start.elapsed().as_nanos() as u64;
                        push_row(
                            &mut exposed,
                            ts,
                            wall_ns,
                            &sample,
                            tx_bps,
                            rx_bps,
                            bw_tx,
                            bw_rx,
                        );
                        prev.insert(
                            device_id,
                            PrevCounters {
                                at: now,
                                tx_bytes: sample.tx_bytes,
                                rx_bytes: sample.rx_bytes,
                            },
                        );
                    }
                    drop(exposed);
                    thread::sleep(interval);
                }
            }
            #[cfg(any(target_os = "macos", windows))]
            {
                let _ = (table, interval, bw_every);
                log::debug!("HCCS collector unsupported on this platform");
            }
            running.store(false, Ordering::SeqCst);
        });

        *lock_collector(&self.handle) = Some(handle);
        Ok(())
    }
}
