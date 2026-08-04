//! Python-facing `ExternalTable`, backed by **mmap memtables**.
//!
//! Each table is an [`ExposedTable`] (MEMT ring buffer) under
//! `<PROBING_DATA_DIR>/<pid>/`:
//!
//! - ``foo`` → ``python.foo`` → SQL ``python.foo``
//! - ``nccl.proxy_ops`` → ``nccl.proxy_ops`` → SQL ``nccl.proxy_ops``
//! - the training process only ever pays the cost of an mmap row write —
//!   query-side materialisation happens in whoever runs the SQL.
//!
//! The first appended row fixes the column dtypes (the Python API only
//! declares column names). A leading `timestamp` column (microseconds since
//! epoch, `I64`) is always present, matching the previous TimeSeries layout.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard};

use crate::features::native_bridge::with_detached_native;
use once_cell::sync::Lazy;
use probing_core::runtime::{BlockOnFallback, RuntimeError};
use probing_core::sync::lock_mutex;
use probing_memtable::discover::ExposedTable;
use probing_memtable::docs;
use probing_memtable::{infer_extern_column_dtype, ring_config, DType, Schema as MtSchema, Value};
use probing_proto::prelude::Ele;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyType};
use pyo3::{pyclass, pymethods, Bound, PyResult, Python};
use thiserror::Error;

use crate::features::convert::{ele_to_python, python_to_ele};

type PyTableRow = (Py<PyAny>, Vec<Py<PyAny>>);

#[derive(Debug, Error)]
enum ExternTableError {
    #[error("column count mismatch")]
    ColumnMismatch,
    #[error("table not initialized")]
    NotInitialized,
    #[error("push_row failed: schema mismatch or row too large")]
    PushFailed,
    #[error(transparent)]
    Memtable(#[from] probing_memtable::MemtableError),
}

/// SQL schema (and filename prefix) for Python extern tables.
pub const EXTERN_TABLE_SCHEMA: &str = "python";

/// Mmap filename for an extern table.
///
/// - ``foo`` → ``python.foo`` (legacy Python plugin tables)
/// - ``nccl.proxy_ops`` → ``nccl.proxy_ops`` (schema-qualified, matches memtable discovery)
fn mmap_basename(name: &str) -> String {
    if name.contains('.') {
        name.to_string()
    } else {
        format!("{EXTERN_TABLE_SCHEMA}.{name}")
    }
}

/// Legacy ``python.*`` tables prepend a ``timestamp`` column; schema-qualified
/// tables (``nccl.proxy_ops``) match native writer layouts exactly.
fn uses_timestamp_column(name: &str) -> bool {
    !name.contains('.')
}

fn build_schema_with_docs(
    name: &str,
    columns: &[String],
    dtypes: &[DType],
    table_doc: Option<&str>,
    column_docs: &HashMap<String, String>,
) -> MtSchema {
    let mut schema = MtSchema::new();
    if let Some(doc) = table_doc {
        schema = schema.table_doc(doc);
    }
    if uses_timestamp_column(name) {
        schema = schema.col("timestamp", DType::I64);
    }
    for (col, dt) in columns.iter().zip(dtypes.iter()) {
        schema = if let Some(doc) = column_docs.get(col) {
            schema.col_doc(col, *dt, doc.as_str())
        } else {
            schema.col(col, *dt)
        };
    }
    schema
}

fn register_python_table_docs(
    name: &str,
    table_doc: Option<&str>,
    column_docs: &HashMap<String, String>,
) {
    let (table_schema, table_name) = if let Some((schema, table)) = name.split_once('.') {
        (schema.to_string(), table.to_string())
    } else {
        (EXTERN_TABLE_SCHEMA.to_string(), name.to_string())
    };
    let pairs: Vec<(String, String)> = column_docs
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    docs::register_column_docs(&table_schema, &table_name, table_doc, &pairs);
}

/// Ring layout: fixed chunk count; chunk byte size derives from capacity.
const NUM_CHUNKS: u32 = 8;
const MIN_CHUNK_BYTES: usize = 4 * 1024;
const MAX_CHUNK_BYTES: usize = 8 * 1024 * 1024;

fn value_to_object(py: Python, v: &Ele) -> Py<PyAny> {
    ele_to_python(py, v).unwrap_or_else(|_| py.None())
}

fn now_micros() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or(0)
}

#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct PyExternalTableConfig {
    #[pyo3(get)]
    chunk_size: usize,
    #[pyo3(get)]
    discard_threshold: usize,
    #[pyo3(get)]
    discard_strategy: String,
    /// PR-3: retain-last-N-steps hint (`None` → capacity-only).
    #[pyo3(get)]
    retain_steps: Option<u32>,
    /// PR-3: retain-last-N-seconds hint (`None` → capacity-only).
    #[pyo3(get)]
    retain_secs: Option<u32>,
}

impl Default for PyExternalTableConfig {
    fn default() -> Self {
        Self::config_for_table("")
    }
}

impl PyExternalTableConfig {
    /// Per-table ring defaults + ``PROBING_EXTTBL_<TABLE>_MB`` /
    /// ``..._RETAIN_STEPS`` / ``..._RETAIN_SECS`` overrides.
    ///
    /// Size (PR-1):
    /// * ``cpu.utilization`` / ``cpu.tasks`` / ``gpu.utilization`` → 8 MiB
    /// * ``gpu.hccs`` → 4 MiB
    /// * other → 20 MiB
    ///
    /// Retention (PR-3, independent from size):
    /// * ``python.torch_trace`` / ``python.comm_collective`` →
    ///   `retain_steps = Some(500)`
    /// * ``cpu.utilization`` / ``gpu.utilization`` →
    ///   `retain_secs = Some(3600)`
    /// * other → both `None`
    pub fn config_for_table(name: &str) -> Self {
        let qualified = if name.contains('.') {
            name.to_string()
        } else if name.is_empty() {
            String::new()
        } else {
            format!("{EXTERN_TABLE_SCHEMA}.{name}")
        };
        let lookup = if qualified.is_empty() {
            "python.torch_trace"
        } else {
            qualified.as_str()
        };
        let discard_threshold = ring_config::table_ring_capacity_bytes(lookup);
        let retention = ring_config::table_retention(lookup);
        PyExternalTableConfig {
            chunk_size: 10000,
            discard_threshold,
            discard_strategy: "BaseMemorySize".to_string(),
            retain_steps: retention.retain_steps,
            retain_secs: retention.retain_secs,
        }
    }
}

#[pymethods]
impl PyExternalTableConfig {
    #[new]
    #[pyo3(signature = (chunk_size, discard_threshold, discard_strategy, retain_steps=None, retain_secs=None))]
    fn new(
        chunk_size: usize,
        discard_threshold: usize,
        discard_strategy: String,
        retain_steps: Option<u32>,
        retain_secs: Option<u32>,
    ) -> Self {
        PyExternalTableConfig {
            chunk_size,
            discard_threshold,
            discard_strategy,
            retain_steps,
            retain_secs,
        }
    }

    #[classmethod]
    fn for_table(_cls: &Bound<'_, PyType>, name: &str) -> Self {
        Self::config_for_table(name)
    }

    #[allow(clippy::wrong_self_convention)] // Python-facing method name, kept for API compat
    fn into_py(&self, py: Python<'_>) -> Py<PyAny> {
        let dict = PyDict::new(py);
        if let Err(e) = dict.set_item("chunk_size", self.chunk_size) {
            log::error!("PyExternalTableConfig::into_py chunk_size: {e}");
        }
        if let Err(e) = dict.set_item("discard_threshold", self.discard_threshold) {
            log::error!("PyExternalTableConfig::into_py discard_threshold: {e}");
        }
        if let Err(e) = dict.set_item("discard_strategy", &self.discard_strategy) {
            log::error!("PyExternalTableConfig::into_py discard_strategy: {e}");
        }
        if let Err(e) = dict.set_item("retain_steps", self.retain_steps) {
            log::error!("PyExternalTableConfig::into_py retain_steps: {e}");
        }
        if let Err(e) = dict.set_item("retain_secs", self.retain_secs) {
            log::error!("PyExternalTableConfig::into_py retain_secs: {e}");
        }
        dict.into()
    }
}

/// Total ring capacity in bytes derived from the (legacy) discard config.
///
/// - `BaseMemorySize`: `discard_threshold` *is* a byte budget.
/// - `BaseElementCount`: estimate 64 bytes/row.
/// - anything else: 16 MiB default.
fn ring_capacity_bytes(discard_threshold: usize, strategy: &str) -> usize {
    let raw = match strategy {
        "BaseMemorySize" => discard_threshold,
        "BaseElementCount" => discard_threshold.saturating_mul(64),
        _ => 16 * 1024 * 1024,
    };
    raw.clamp(MIN_CHUNK_BYTES * NUM_CHUNKS as usize, 1 << 30)
}

fn ring_chunk_bytes(capacity: usize) -> u32 {
    (capacity / NUM_CHUNKS as usize).clamp(MIN_CHUNK_BYTES, MAX_CHUNK_BYTES) as u32
}

/// Column dtype inferred from the first appended value.
fn ele_dtype(e: &Ele) -> DType {
    match e {
        Ele::I32(_) => DType::I32,
        Ele::I64(_) => DType::I64,
        Ele::F32(_) => DType::F32,
        Ele::F64(_) => DType::F64,
        Ele::BOOL(_) => DType::U8,
        Ele::DataTime(_) => DType::U64,
        Ele::Text(_) | Ele::Url(_) | Ele::Nil => DType::Str,
    }
}

/// Owned cell value: coerced from an [`Ele`] to match the column dtype, so a
/// `Vec<Value>` row can borrow from it.
enum OwnedVal {
    U8(u8),
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
    U64(u64),
    S(String),
}

fn ele_to_owned(e: &Ele, dt: DType) -> OwnedVal {
    let as_f64 = |e: &Ele| match e {
        Ele::I32(v) => *v as f64,
        Ele::I64(v) => *v as f64,
        Ele::F32(v) => *v as f64,
        Ele::F64(v) => *v,
        Ele::BOOL(v) => *v as u8 as f64,
        Ele::DataTime(v) => *v as f64,
        _ => 0.0,
    };
    match dt {
        DType::U8 => OwnedVal::U8(match e {
            Ele::BOOL(v) => *v as u8,
            other => as_f64(other) as u8,
        }),
        DType::I32 => OwnedVal::I32(as_f64(e) as i32),
        DType::I64 => OwnedVal::I64(as_f64(e) as i64),
        DType::F32 => OwnedVal::F32(as_f64(e) as f32),
        DType::F64 => OwnedVal::F64(as_f64(e)),
        DType::U64 => OwnedVal::U64(as_f64(e) as u64),
        DType::U32 => OwnedVal::U64(as_f64(e) as u64),
        DType::Str | DType::Bytes => OwnedVal::S(match e {
            Ele::Text(s) | Ele::Url(s) => s.clone(),
            Ele::Nil => String::new(),
            other => other.to_string(),
        }),
    }
}

fn owned_to_value(o: &OwnedVal) -> Value<'_> {
    match o {
        OwnedVal::U8(v) => Value::U8(*v),
        OwnedVal::I32(v) => Value::I32(*v),
        OwnedVal::I64(v) => Value::I64(*v),
        OwnedVal::F32(v) => Value::F32(*v),
        OwnedVal::F64(v) => Value::F64(*v),
        OwnedVal::U64(v) => Value::U64(*v),
        OwnedVal::S(s) => Value::Str(s),
    }
}

/// State behind one extern table. The mmap ring is created lazily on the
/// first append because the Python API declares names but not types.
pub struct ExternBacking {
    name: String,
    columns: Vec<String>,
    capacity_bytes: usize,
    dtypes: Vec<DType>,
    table: Option<ExposedTable>,
    table_doc: Option<String>,
    column_docs: HashMap<String, String>,
    // ── PR-3 retention state ─────────────────────────────────────────
    /// Retain-last-N-steps window (chunks whose min_step is younger than
    /// `latest_step - retain_steps` must not be recycled).
    retain_steps: Option<u32>,
    /// Retain-last-N-seconds window (chunks whose min_ts is younger than
    /// `latest_ts - retain_secs*1e6` must not be recycled).
    retain_secs: Option<u32>,
    /// Column index of a monotonic training-step column, when the schema
    /// has one named exactly `step`. Detected on first `ensure_table`.
    step_col_idx: Option<usize>,
    /// Per-chunk oldest step / µs-timestamp observed by the writer, sized
    /// once the table is registered. `i64::MAX` = chunk empty.
    per_chunk_min_step: Vec<i64>,
    per_chunk_min_ts: Vec<i64>,
    /// Counters exposed for PR-3 verify: how many times a chunk recycle
    /// dropped rows younger than the retention window.
    retention_violations_step: u64,
    retention_violations_secs: u64,
    /// One-shot cache of `write_chunk` before each push, so the writer
    /// can detect ring advances after the fact.
    prev_write_chunk: Option<usize>,
}

impl ExternBacking {
    fn new(
        name: &str,
        columns: Vec<String>,
        capacity_bytes: usize,
        table_doc: Option<String>,
        column_docs: HashMap<String, String>,
    ) -> Self {
        if !column_docs.is_empty() || table_doc.is_some() {
            register_python_table_docs(name, table_doc.as_deref(), &column_docs);
        }
        // PR-3 retention lookup uses the qualified name that mmap uses
        // (e.g. `python.torch_trace`), so a bare "torch_trace" also hits
        // the per-table defaults.
        let qualified = mmap_basename(name);
        let retention = ring_config::table_retention(&qualified);
        Self {
            name: name.to_string(),
            columns,
            capacity_bytes,
            dtypes: vec![],
            table: None,
            table_doc,
            column_docs,
            retain_steps: retention.retain_steps,
            retain_secs: retention.retain_secs,
            step_col_idx: None,
            per_chunk_min_step: vec![i64::MAX; NUM_CHUNKS as usize],
            per_chunk_min_ts: vec![i64::MAX; NUM_CHUNKS as usize],
            retention_violations_step: 0,
            retention_violations_secs: 0,
            prev_write_chunk: None,
        }
    }

    /// Locate an integer column named exactly `step` (case-sensitive), used
    /// as the retention key for step-indexed tables. Legacy `python.*`
    /// schemas prepend `timestamp` so the caller offsets accordingly.
    fn detect_step_col(name: &str, columns: &[String], dtypes: &[DType]) -> Option<usize> {
        let ts_offset = if uses_timestamp_column(name) { 1 } else { 0 };
        columns.iter().position(|c| c == "step").and_then(|i| {
            let idx = i + ts_offset;
            let dt = dtypes.get(i).copied()?;
            matches!(dt, DType::I32 | DType::I64 | DType::U32 | DType::U64).then_some(idx)
        })
    }

    fn ensure_registered(&mut self) -> Result<(), ExternTableError> {
        if self.table.is_some() {
            return Ok(());
        }
        self.dtypes = self
            .columns
            .iter()
            .map(|col| infer_extern_column_dtype(col))
            .collect();
        let schema = build_schema_with_docs(
            &self.name,
            &self.columns,
            &self.dtypes,
            self.table_doc.as_deref(),
            &self.column_docs,
        );
        let chunk_bytes = ring_chunk_bytes(self.capacity_bytes);
        let filename = mmap_basename(&self.name);
        let table = ExposedTable::create(&filename, &schema, chunk_bytes, NUM_CHUNKS)?;
        self.step_col_idx = Self::detect_step_col(&self.name, &self.columns, &self.dtypes);
        self.table = Some(table);
        Ok(())
    }

    fn row_count(&self) -> usize {
        self.table.as_ref().map_or(0, |t| {
            let view = t.view();
            (0..view.num_chunks()).map(|c| view.num_rows(c)).sum()
        })
    }

    fn ensure_table(&mut self, first_row: &[Ele]) -> Result<(), ExternTableError> {
        if self.table.is_some() && self.row_count() > 0 {
            return Ok(());
        }
        self.table = None;
        self.dtypes.clear();
        self.per_chunk_min_step = vec![i64::MAX; NUM_CHUNKS as usize];
        self.per_chunk_min_ts = vec![i64::MAX; NUM_CHUNKS as usize];
        self.prev_write_chunk = None;

        let dtypes: Vec<DType> = first_row.iter().map(ele_dtype).collect();
        let schema = build_schema_with_docs(
            &self.name,
            &self.columns,
            &dtypes,
            self.table_doc.as_deref(),
            &self.column_docs,
        );
        let chunk_bytes = ring_chunk_bytes(self.capacity_bytes);
        let filename = mmap_basename(&self.name);
        let table = ExposedTable::create(&filename, &schema, chunk_bytes, NUM_CHUNKS)?;
        self.step_col_idx = Self::detect_step_col(&self.name, &self.columns, &dtypes);
        self.dtypes = dtypes;
        self.table = Some(table);
        Ok(())
    }

    fn append(&mut self, timestamp: i64, values: &[Ele]) -> Result<(), ExternTableError> {
        if values.len() != self.columns.len() {
            return Err(ExternTableError::ColumnMismatch);
        }
        self.ensure_table(values)?;

        let owned: Vec<OwnedVal> = values
            .iter()
            .zip(self.dtypes.iter())
            .map(|(e, dt)| ele_to_owned(e, *dt))
            .collect();
        let mut row: Vec<Value> = Vec::with_capacity(owned.len() + 1);
        if uses_timestamp_column(&self.name) {
            row.push(Value::I64(timestamp));
        }
        row.extend(owned.iter().map(owned_to_value));

        // Extract this row's step (if the table has one) before we lose
        // ownership; used by both the pre-push tracker and the
        // post-advance retention check.
        let this_step = self.extract_step(&owned);

        // ExposedTable::push_row validates schema and auto-advances chunks.
        let Some(table) = self.table.as_mut() else {
            return Err(ExternTableError::NotInitialized);
        };
        let write_chunk_before = table.view().write_chunk();
        if !table.push_row(&row) {
            return Err(ExternTableError::PushFailed);
        }
        let write_chunk_after = table.view().write_chunk();

        // Post-write bookkeeping: update per-chunk min_step for the chunk
        // this row landed in, and (if the writer advanced) check whether
        // the just-vacated chunk still held rows within the retention
        // window — that's a truncation event we want to expose.
        //
        // MEMT auto-advances only when the current chunk is full, so
        // `write_chunk_after != write_chunk_before` means the row that
        // just landed spilled to the next slot. If that spill overwrote
        // a chunk still within retention, warn once per event.
        if let Some(step) = this_step {
            let slot = write_chunk_after;
            if self.per_chunk_min_step[slot] == i64::MAX || step < self.per_chunk_min_step[slot] {
                self.per_chunk_min_step[slot] = step;
            }
        }
        // timestamp always tracked (legacy python.* tables use i64 µs);
        // qualified tables may not — we still store whatever the writer
        // handed us (`timestamp` arg, monotonic-ish per-call).
        {
            let slot = write_chunk_after;
            if self.per_chunk_min_ts[slot] == i64::MAX || timestamp < self.per_chunk_min_ts[slot] {
                self.per_chunk_min_ts[slot] = timestamp;
            }
        }

        if write_chunk_after != write_chunk_before {
            self.check_retention_on_advance(write_chunk_after, this_step, timestamp);
            // The chunk we just moved *into* is being recycled; reset its
            // tracked min_* to the row we just wrote (which is now the
            // oldest row in that slot post-recycle).
            self.per_chunk_min_step[write_chunk_after] = this_step.unwrap_or(i64::MAX);
            self.per_chunk_min_ts[write_chunk_after] = timestamp;
        }

        self.prev_write_chunk = Some(write_chunk_after);
        Ok(())
    }

    /// Fetch the step from an owned row (post-dtype coercion). Only
    /// numeric dtypes count; strings and floats-as-step are ignored.
    fn extract_step(&self, owned: &[OwnedVal]) -> Option<i64> {
        let idx = self.step_col_idx?;
        // step_col_idx is expressed in mmap-column space (accounting for
        // the leading `timestamp` for legacy python.* tables). owned[] is
        // in user-column space, so subtract the timestamp offset back.
        let user_idx = if uses_timestamp_column(&self.name) {
            idx.checked_sub(1)?
        } else {
            idx
        };
        owned.get(user_idx).and_then(|v| match v {
            OwnedVal::I32(x) => Some(*x as i64),
            OwnedVal::I64(x) => Some(*x),
            OwnedVal::U64(x) => Some(*x as i64),
            OwnedVal::U8(x) => Some(*x as i64),
            _ => None,
        })
    }

    /// Called after a ring advance (`write_chunk` incremented). The chunk
    /// at `slot_now` is the destination we're about to overwrite; if its
    /// pre-recycle content was still within the retention window, we log
    /// a `retention truncated` warning and bump the counter.
    fn check_retention_on_advance(
        &mut self,
        slot_now: usize,
        current_step: Option<i64>,
        current_ts: i64,
    ) {
        if let (Some(retain_steps), Some(cur_step)) = (self.retain_steps, current_step) {
            let vacated_min = self.per_chunk_min_step[slot_now];
            let cutoff = cur_step.saturating_sub(retain_steps as i64);
            // vacated_min == i64::MAX → empty chunk, no violation.
            if vacated_min != i64::MAX && vacated_min >= cutoff {
                self.retention_violations_step = self.retention_violations_step.saturating_add(1);
                log::warn!(
                    "retention truncated: table={} recycled chunk={} min_step={} cutoff={} (retain_steps={})",
                    self.name, slot_now, vacated_min, cutoff, retain_steps
                );
            }
        }
        if let Some(retain_secs) = self.retain_secs {
            let vacated_min = self.per_chunk_min_ts[slot_now];
            let cutoff = current_ts.saturating_sub((retain_secs as i64).saturating_mul(1_000_000));
            if vacated_min != i64::MAX && vacated_min >= cutoff {
                self.retention_violations_secs = self.retention_violations_secs.saturating_add(1);
                log::warn!(
                    "retention truncated: table={} recycled chunk={} min_ts={} cutoff={} (retain_secs={})",
                    self.name, slot_now, vacated_min, cutoff, retain_secs
                );
            }
        }
    }

    /// Expose current retention counters and effective config to Python
    /// (used by the smoke tests and by future SQL introspection).
    pub fn retention_snapshot(&self) -> (Option<u32>, Option<u32>, u64, u64) {
        (
            self.retain_steps,
            self.retain_secs,
            self.retention_violations_step,
            self.retention_violations_secs,
        )
    }

    /// Override the retention window at runtime. Called by
    /// `probing.core.config` when a `SET probing.exttbl.<t>.retain_*`
    /// arrives; returns the previous value so the caller can log a diff.
    pub fn set_retention(
        &mut self,
        steps: Option<u32>,
        secs: Option<u32>,
    ) -> (Option<u32>, Option<u32>) {
        let prev = (self.retain_steps, self.retain_secs);
        if steps.is_some() {
            self.retain_steps = steps;
        }
        if secs.is_some() {
            self.retain_secs = secs;
        }
        prev
    }

    fn read_row_values(&self, cursor: &mut probing_memtable::RowCursor<'_>) -> Vec<Ele> {
        self.dtypes
            .iter()
            .map(|dt| match dt {
                DType::U8 => Ele::BOOL(cursor.next_u8() != 0),
                DType::I32 => Ele::I32(cursor.next_i32()),
                DType::I64 => Ele::I64(cursor.next_i64()),
                DType::F32 => Ele::F32(cursor.next_f32()),
                DType::F64 => Ele::F64(cursor.next_f64()),
                DType::U64 => Ele::DataTime(cursor.next_u64()),
                DType::U32 => Ele::I64(cursor.next_u32() as i64),
                DType::Str => Ele::Text(cursor.next_str().to_string()),
                DType::Bytes => Ele::Text(String::from_utf8_lossy(cursor.next_bytes()).to_string()),
            })
            .collect()
    }

    /// Rows in chronological order; when `limit` is set, only the most
    /// recent `limit` rows are returned (still oldest → newest).
    fn take(&self, limit: Option<usize>) -> Vec<(Ele, Vec<Ele>)> {
        let Some(table) = &self.table else {
            return vec![];
        };
        let view = table.view();
        let mut out: Vec<(Ele, Vec<Ele>)> = Vec::new();
        for chunk in view.chunks_logical() {
            for row in view.rows(chunk) {
                let mut cursor = row.cursor();
                let (ts, vals) = if uses_timestamp_column(&self.name) {
                    let ts = Ele::I64(cursor.next_i64());
                    let vals = self.read_row_values(&mut cursor);
                    (ts, vals)
                } else {
                    let vals = self.read_row_values(&mut cursor);
                    let ts = vals.first().cloned().unwrap_or(Ele::Nil);
                    (ts, vals)
                };
                out.push((ts, vals));
            }
        }
        if let Some(limit) = limit {
            if out.len() > limit {
                out.drain(..out.len() - limit);
            }
        }
        out
    }
}

impl std::fmt::Debug for ExternBacking {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExternBacking")
            .field("name", &self.name)
            .field("columns", &self.columns)
            .field("created", &self.table.is_some())
            .finish()
    }
}

pub static EXTERN_TABLES: Lazy<Mutex<HashMap<String, Arc<Mutex<ExternBacking>>>>> =
    Lazy::new(|| Mutex::new(Default::default()));

fn lock_extern_tables() -> MutexGuard<'static, HashMap<String, Arc<Mutex<ExternBacking>>>> {
    lock_mutex(&EXTERN_TABLES, "EXTERN_TABLES")
}

fn lock_backing(backing: &Mutex<ExternBacking>) -> MutexGuard<'_, ExternBacking> {
    lock_mutex(backing, "ExternBacking")
}

#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct ExternalTable(Option<Arc<Mutex<ExternBacking>>>, usize);

const BRIDGE_DEGRADED_MSG: &str =
    "probing native bridge unavailable; external table operations are disabled";

fn require_backing(
    backing: &Option<Arc<Mutex<ExternBacking>>>,
) -> PyResult<&Arc<Mutex<ExternBacking>>> {
    backing
        .as_ref()
        .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err(BRIDGE_DEGRADED_MSG))
}

impl BlockOnFallback for ExternalTable {
    fn on_block_on_failure(err: RuntimeError) -> Self {
        log::error!("ExternalTable bridge degraded: {err}; table unusable");
        ExternalTable(None, 0)
    }
}

impl ExternalTable {
    fn extract_eles(values: Vec<Py<PyAny>>) -> PyResult<Vec<Ele>> {
        Python::attach(|py| {
            values
                .into_iter()
                .map(|v| python_to_ele(v.bind(py)))
                .collect()
        })
    }

    fn create_backing(
        name: &str,
        columns: Vec<String>,
        discard_threshold: Option<usize>,
        discard_strategy: &str,
        table_doc: Option<String>,
        column_docs: HashMap<String, String>,
        retain_steps: Option<u32>,
        retain_secs: Option<u32>,
    ) -> Arc<Mutex<ExternBacking>> {
        let threshold = discard_threshold
            .unwrap_or_else(|| PyExternalTableConfig::config_for_table(name).discard_threshold);
        let capacity = ring_capacity_bytes(threshold, discard_strategy);
        let backing = Arc::new(Mutex::new(ExternBacking::new(
            name,
            columns,
            capacity,
            table_doc,
            column_docs,
        )));
        // Explicit overrides win over env/default retention.
        if retain_steps.is_some() || retain_secs.is_some() {
            lock_backing(backing.as_ref()).set_retention(retain_steps, retain_secs);
        }
        lock_backing(backing.as_ref())
            .ensure_registered()
            .unwrap_or_else(|e| {
                log::error!("failed to register extern table for SQL catalog: {e}");
            });
        backing
    }
}

#[pymethods]
impl ExternalTable {
    #[new]
    #[pyo3(signature = (
        name, columns, chunk_size = 10000, discard_threshold = None,
        discard_strategy = "BaseMemorySize".to_string(),
        table_doc = None, column_docs = None,
        retain_steps = None, retain_secs = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        name: &str,
        columns: Vec<String>,
        chunk_size: usize,
        discard_threshold: Option<usize>,
        discard_strategy: String,
        table_doc: Option<String>,
        column_docs: Option<HashMap<String, String>>,
        retain_steps: Option<u32>,
        retain_secs: Option<u32>,
    ) -> Self {
        let _ = chunk_size; // ring chunking is byte-based; kept for API compat
        let name = name.to_string();
        with_detached_native(move || {
            let ncolumn = columns.len();
            let backing = Self::create_backing(
                &name,
                columns,
                discard_threshold,
                &discard_strategy,
                table_doc,
                column_docs.unwrap_or_default(),
                retain_steps,
                retain_secs,
            );
            lock_extern_tables().insert(name, backing.clone());
            ExternalTable(Some(backing), ncolumn)
        })
    }

    #[classmethod]
    fn get(_cls: &Bound<'_, PyType>, name: &str) -> PyResult<ExternalTable> {
        let name = name.to_string();
        with_detached_native(move || {
            let binding = lock_extern_tables();
            if let Some(backing) = binding.get(&name) {
                let ncolumn = lock_backing(backing.as_ref()).columns.len();
                Ok(ExternalTable(Some(backing.clone()), ncolumn))
            } else {
                Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "table {name} not found"
                )))
            }
        })
    }

    #[classmethod]
    #[pyo3(signature = (
        name, columns, chunk_size = 10000, discard_threshold = None,
        discard_strategy = "BaseMemorySize".to_string(),
        table_doc = None, column_docs = None,
        retain_steps = None, retain_secs = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn get_or_create(
        _cls: &Bound<'_, PyType>,
        name: &str,
        columns: Vec<String>,
        chunk_size: usize,
        discard_threshold: Option<usize>,
        discard_strategy: String,
        table_doc: Option<String>,
        column_docs: Option<HashMap<String, String>>,
        retain_steps: Option<u32>,
        retain_secs: Option<u32>,
    ) -> PyResult<ExternalTable> {
        let _ = chunk_size;
        let name = name.to_string();
        with_detached_native(move || {
            let mut binding = lock_extern_tables();
            if let Some(backing) = binding.get(&name) {
                // Existing table: allow SET-style retention override.
                if retain_steps.is_some() || retain_secs.is_some() {
                    lock_backing(backing.as_ref()).set_retention(retain_steps, retain_secs);
                }
                let ncolumn = lock_backing(backing.as_ref()).columns.len();
                Ok(ExternalTable(Some(backing.clone()), ncolumn))
            } else {
                let ncolumn = columns.len();
                let backing = Self::create_backing(
                    &name,
                    columns,
                    discard_threshold,
                    &discard_strategy,
                    table_doc,
                    column_docs.unwrap_or_default(),
                    retain_steps,
                    retain_secs,
                );
                binding.insert(name, backing.clone());
                Ok(ExternalTable(Some(backing), ncolumn))
            }
        })
    }

    #[classmethod]
    fn drop(_cls: &Bound<'_, PyType>, name: &str) -> PyResult<()> {
        let name = name.to_string();
        with_detached_native(move || {
            let _ = lock_extern_tables().remove(&name);
            Ok(())
        })
    }

    fn names(&self) -> PyResult<Vec<String>> {
        let backing = require_backing(&self.0)?.clone();
        with_detached_native(move || Ok(lock_backing(backing.as_ref()).columns.clone()))
    }

    fn append(&mut self, values: Vec<Py<PyAny>>) -> PyResult<()> {
        if values.len() != self.1 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "column count mismatch",
            ));
        }
        let eles = Self::extract_eles(values)?;
        let backing = require_backing(&self.0)?.clone();
        with_detached_native(move || {
            lock_backing(backing.as_ref())
                .append(now_micros(), &eles)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
        })
    }

    fn append_ts(&mut self, t: i64, values: Vec<Py<PyAny>>) -> PyResult<()> {
        if values.len() != self.1 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "column count mismatch",
            ));
        }
        let eles = Self::extract_eles(values)?;
        let backing = require_backing(&self.0)?.clone();
        with_detached_native(move || {
            lock_backing(backing.as_ref())
                .append(t, &eles)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
        })
    }

    fn append_many(&mut self, rows: Vec<Vec<Py<PyAny>>>) -> PyResult<()> {
        for row in rows {
            self.append(row)?;
        }
        Ok(())
    }

    #[pyo3(signature = (limit=None))]
    fn take(&self, limit: Option<usize>) -> PyResult<Vec<PyTableRow>> {
        let backing = require_backing(&self.0)?.clone();
        with_detached_native(move || {
            let rows = lock_backing(backing.as_ref()).take(limit);
            let result = rows
                .iter()
                .map(|(t, vals)| {
                    Python::attach(|py| {
                        let t = value_to_object(py, t);
                        let vals = vals
                            .iter()
                            .map(|v| value_to_object(py, v))
                            .collect::<Vec<_>>();
                        (t, vals)
                    })
                })
                .collect();
            Ok(result)
        })
    }

    /// Return a dict `{retain_steps, retain_secs, violations_step,
    /// violations_secs}` — used by PR-3 smoke tests to verify the
    /// retention window is being observed.
    fn retention(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let backing = require_backing(&self.0)?.clone();
        // No Python↔native re-entry needed here; a plain locked read
        // avoids the `with_detached_native` fallback-type bound.
        let (steps, secs, vs, vt) = lock_backing(backing.as_ref()).retention_snapshot();
        let dict = PyDict::new(py);
        dict.set_item("retain_steps", steps)?;
        dict.set_item("retain_secs", secs)?;
        dict.set_item("violations_step", vs)?;
        dict.set_item("violations_secs", vt)?;
        Ok(dict.into())
    }

    /// Runtime override of the retention window. `None` on either arg
    /// leaves that dimension unchanged. Returns the previous
    /// `(retain_steps, retain_secs)` for logging.
    #[pyo3(signature = (retain_steps=None, retain_secs=None))]
    fn set_retention(
        &mut self,
        retain_steps: Option<u32>,
        retain_secs: Option<u32>,
    ) -> PyResult<(Option<u32>, Option<u32>)> {
        let backing = require_backing(&self.0)?.clone();
        let prev = lock_backing(backing.as_ref()).set_retention(retain_steps, retain_secs);
        Ok(prev)
    }
}

/// Register table/column documentation for SQL `DESCRIBE` (without creating a table).
#[pyfunction]
#[pyo3(signature = (qualified_name, table_doc=None, column_docs=None))]
pub fn register_table_docs(
    qualified_name: &str,
    table_doc: Option<&str>,
    column_docs: Option<HashMap<String, String>>,
) -> PyResult<()> {
    register_python_table_docs(qualified_name, table_doc, &column_docs.unwrap_or_default());
    Ok(())
}

#[cfg(test)]
mod register_docs_tests {
    use super::*;
    use probing_memtable::docs;

    #[test]
    fn register_table_docs_exposes_python_schema() {
        let table = format!("py_doc_test_{}", std::process::id());
        let qualified = format!("python.{table}");
        let mut column_docs = HashMap::new();
        column_docs.insert("latency_ms".to_string(), "latency in ms".to_string());
        register_table_docs(&qualified, Some("Python doc test table"), Some(column_docs)).unwrap();
        let rows = docs::snapshot();
        let row = rows
            .iter()
            .find(|r| r.table_schema == "python" && r.table_name == table)
            .expect("python table docs");
        assert_eq!(row.description.as_deref(), Some("Python doc test table"));
        assert_eq!(
            row.columns.get("latency_ms"),
            Some(&"latency in ms".to_string())
        );
    }
}

#[cfg(test)]
mod for_table_tests {
    use super::*;

    #[test]
    fn for_table_sets_tiered_defaults() {
        let cpu = PyExternalTableConfig::config_for_table("cpu.utilization");
        assert_eq!(cpu.discard_threshold, 8 * 1024 * 1024);
        let hccs = PyExternalTableConfig::config_for_table("gpu.hccs");
        assert_eq!(hccs.discard_threshold, 4 * 1024 * 1024);
    }

    /// PR-3: `for_table` also populates the retain window matching the
    /// defaults in `ring_config`. torch_trace → 500 steps, cpu.util →
    /// 3600 s. Ensures the config surface exposes what the write path
    /// actually enforces.
    #[test]
    fn retain_steps_default_by_table() {
        let tt = PyExternalTableConfig::config_for_table("python.torch_trace");
        assert_eq!(tt.retain_steps, Some(500));
        assert_eq!(tt.retain_secs, None);

        let comm = PyExternalTableConfig::config_for_table("python.comm_collective");
        assert_eq!(comm.retain_steps, Some(500));

        let cpu = PyExternalTableConfig::config_for_table("cpu.utilization");
        assert_eq!(cpu.retain_steps, None);
        assert_eq!(cpu.retain_secs, Some(3600));

        let gpu = PyExternalTableConfig::config_for_table("gpu.utilization");
        assert_eq!(gpu.retain_secs, Some(3600));

        // Unqualified names route through `python.<name>`, so bare
        // `torch_trace` should still land the step default.
        let short = PyExternalTableConfig::config_for_table("torch_trace");
        assert_eq!(short.retain_steps, Some(500));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extensions::python::PythonProbeDataSource;
    use probing_core::core::{Engine, UnifiedMemtableProbeDataSource};
    use pyo3::ffi::c_str;

    /// Route all mmap files of this test process into one tempdir.
    static TEST_DATA_DIR: Lazy<tempfile::TempDir> = Lazy::new(|| {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("PROBING_DATA_DIR", dir.path());
        dir
    });

    fn setup() {
        let _ = &*TEST_DATA_DIR;
        pyo3::Python::initialize();
        Python::attach(|py| {
            use pyo3::types::PyModule;
            use pyo3::PyTypeInfo;

            let sys = PyModule::import(py, "sys").unwrap();
            let modules = sys.getattr("modules").unwrap();

            let probing = if modules.contains("probing").unwrap_or(false) {
                PyModule::import(py, "probing").unwrap()
            } else {
                let m = PyModule::new(py, "probing").unwrap();
                modules.set_item("probing", &m).unwrap();
                m
            };

            if !probing.hasattr("ExternalTable").unwrap_or(false) {
                probing
                    .setattr("ExternalTable", ExternalTable::type_object(py))
                    .unwrap();
            }
        });
    }

    /// Create a table with a unique name and three rows; idempotent per name.
    fn setup_table(name: &str) {
        setup();
        Python::attach(|py| {
            py.run(
                &std::ffi::CString::new(format!(
                    r#"
import probing
if not hasattr(probing, "_made_{name}"):
    t = probing.ExternalTable.get_or_create("{name}", ["a", "b"])
    t.append([1, 2])
    t.append([3, 4])
    t.append([5, 6])
    probing._made_{name} = True
"#
                ))
                .unwrap(),
                None,
                None,
            )
            .unwrap();
        });
    }

    async fn engine_with_python() -> Engine {
        Engine::builder()
            .with_default_namespace("probe")
            .with_data_source(PythonProbeDataSource::create("python"))
            .with_data_source(Arc::new(UnifiedMemtableProbeDataSource))
            .build()
            .await
            .unwrap()
    }

    #[test]
    fn test_create_new_table() {
        setup();
        let table = ExternalTable::new(
            "table1",
            vec!["a".to_string(), "b".to_string()],
            10000,
            Some(20000000),
            "BaseMemorySize".to_string(),
            None,
            None,
            None,
            None,
        );
        assert_eq!(table.names().unwrap(), vec!["a", "b"]);
    }

    #[test]
    fn ensure_registered_infers_numeric_dtypes() {
        setup();
        let name = format!("comm_like_{}", std::process::id());
        let _table = ExternalTable::new(
            &name,
            vec![
                "rank".to_string(),
                "duration_ms".to_string(),
                "op".to_string(),
            ],
            10000,
            Some(20_000_000),
            "BaseMemorySize".to_string(),
            None,
            None,
            None,
            None,
        );
        let binding = lock_extern_tables();
        let backing = binding.get(&name).expect("backing");
        let guard = lock_backing(backing.as_ref());
        assert_eq!(
            guard.dtypes,
            vec![DType::I64, DType::F64, DType::Str],
            "placeholder mmap must not default all columns to Str"
        );
    }

    #[test]
    fn test_create_table_in_python() {
        setup();
        Python::attach(|py| {
            py.run(
                c_str!(
                    r#"
import probing
table = probing.ExternalTable.get_or_create("table2", ["a", "b"])
"#
                ),
                None,
                None,
            )
            .unwrap();
            let binding = EXTERN_TABLES.lock().unwrap();
            assert!(binding.contains_key("table2"));
        });
    }

    #[test]
    fn test_drop_table_in_python() {
        setup();
        Python::attach(|py| {
            py.run(
                c_str!(
                    r#"
import probing
probing.ExternalTable.get_or_create("table_to_drop", ["a", "b"])
probing.ExternalTable.drop("table_to_drop")
                    "#
                ),
                None,
                None,
            )
            .unwrap();
            let binding = EXTERN_TABLES.lock().unwrap();
            assert!(!binding.contains_key("table_to_drop"));
        });
    }

    #[test]
    fn test_append_take_roundtrip_and_mmap_file() {
        setup();
        let mut table = ExternalTable::new(
            "roundtrip",
            vec!["x".to_string(), "msg".to_string()],
            10000,
            Some(1_000_000),
            "BaseMemorySize".to_string(),
            None,
            None,
            None,
            None,
        );
        Python::attach(|py| {
            let vals: Vec<Py<PyAny>> = vec![
                1i64.into_pyobject(py).unwrap().into_any().unbind(),
                "hello".into_pyobject(py).unwrap().into_any().unbind(),
            ];
            table.append(vals).unwrap();
            let vals: Vec<Py<PyAny>> = vec![
                2i64.into_pyobject(py).unwrap().into_any().unbind(),
                "world".into_pyobject(py).unwrap().into_any().unbind(),
            ];
            table.append(vals).unwrap();
        });

        // mmap file exists on disk under <data_dir>/<pid>/python.roundtrip
        let path = probing_memtable::discover::default_dir()
            .join(std::process::id().to_string())
            .join("python.roundtrip");
        assert!(path.is_file(), "mmap file missing: {path:?}");

        // Qualified schema.table → mmap basename used as-is
        let mut nccl = ExternalTable::new(
            "nccl.proxy_ops",
            vec!["rank".to_string()],
            10000,
            Some(1_000_000),
            "BaseMemorySize".to_string(),
            None,
            None,
            None,
            None,
        );
        Python::attach(|py| {
            let vals: Vec<Py<PyAny>> = vec![1i64.into_pyobject(py).unwrap().into_any().unbind()];
            nccl.append(vals).unwrap();
        });
        let nccl_path = probing_memtable::discover::default_dir()
            .join(std::process::id().to_string())
            .join("nccl.proxy_ops");
        assert!(nccl_path.is_file(), "mmap file missing: {nccl_path:?}");

        // take() returns rows oldest → newest, with coerced values
        let rows = table.take(None).unwrap();
        assert_eq!(rows.len(), 2);
        Python::attach(|py| {
            let (_, vals) = &rows[0];
            assert_eq!(vals[0].extract::<i64>(py).unwrap(), 1);
            assert_eq!(vals[1].extract::<String>(py).unwrap(), "hello");
            let (_, vals) = &rows[1];
            assert_eq!(vals[0].extract::<i64>(py).unwrap(), 2);
            assert_eq!(vals[1].extract::<String>(py).unwrap(), "world");
        });

        // take(limit) keeps the most recent rows
        let rows = table.take(Some(1)).unwrap();
        assert_eq!(rows.len(), 1);
        Python::attach(|py| {
            assert_eq!(rows[0].1[1].extract::<String>(py).unwrap(), "world");
        });
    }

    #[test]
    fn test_see_py_table_data_in_engine() {
        setup_table("table4");
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(4)
            .enable_all()
            .build()
            .unwrap();
        let engine = rt.block_on(engine_with_python());
        let tables = rt.block_on(async {
            engine
                .async_query("select * from python.table4 ")
                .await
                .unwrap()
        });
        let df = tables.expect("Table 'table4' should be queryable");
        assert_eq!(df.len(), 3, "Should have 3 rows");
        // timestamp + a + b
        assert_eq!(df.names.len(), 3, "Should have 3 columns: {:?}", df.names);
        assert_eq!(df.names[0], "timestamp");
    }

    #[test]
    fn test_calculate_in_sql_with_filter() {
        setup_table("table5");
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(4)
            .enable_all()
            .build()
            .unwrap();
        let engine = rt.block_on(engine_with_python());
        let tables = rt.block_on(async {
            engine
                .async_query("select a + b as c from python.table5 where a > 1")
                .await
                .unwrap()
        });
        let df = tables.expect("Query should return results");
        assert_eq!(df.len(), 2, "Should have 2 rows where a > 1");
    }

    #[test]
    fn test_aggregate_in_sql() {
        setup_table("table6");
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(4)
            .enable_all()
            .build()
            .unwrap();
        let engine = rt.block_on(engine_with_python());
        let tables = rt.block_on(async {
            engine
                .async_query("select sum(a), sum(b) from python.table6")
                .await
                .unwrap()
        });
        let df = tables.expect("Aggregation query should return results");
        assert!(!df.cols.is_empty(), "Should have aggregation results");
    }

    #[test]
    fn test_static_python_tables_not_shadowed() {
        // Extern mmap tables under schema `python` must not hide the static
        // namespace (backtrace, expression tables) — the merged catalog
        // resolves mmap first, then falls through to the inner provider.
        setup_table("table7");
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();
        let engine = rt.block_on(engine_with_python());
        // `python.\`time.time()\`` is served by the static namespace's
        // expression path; it must still resolve with extern tables present.
        let result = rt.block_on(async {
            engine
                .async_query("select * from python.`time.time()`")
                .await
        });
        assert!(
            result.is_ok(),
            "static python namespace shadowed: {result:?}"
        );
    }
}
