use std::collections::BTreeMap;
use std::env;
use std::path::Path;

use probing_memtable::MemTable;

fn percentile(sorted: &[f64], fraction: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let index = ((sorted.len() - 1) as f64 * fraction).round() as usize;
    sorted[index.min(sorted.len() - 1)]
}

fn print_stats(label: &str, values_ms: &mut [f64]) {
    values_ms.sort_by(f64::total_cmp);
    let mean = if values_ms.is_empty() {
        0.0
    } else {
        values_ms.iter().sum::<f64>() / values_ms.len() as f64
    };
    println!(
        "{label} count={} mean_ms={mean:.6} p50_ms={:.6} p95_ms={:.6} max_ms={:.6}",
        values_ms.len(),
        percentile(values_ms, 0.50),
        percentile(values_ms, 0.95),
        values_ms.last().copied().unwrap_or(0.0),
    );
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = env::args()
        .nth(1)
        .ok_or("usage: hccl_memtable_summary <table-file>")?;
    let table = MemTable::open_file(&path)?;
    let basename = Path::new(&path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("");
    let mut total_rows = 0usize;

    match basename {
        "hccl.host_ops" => {
            let mut hccl_duration_ms = Vec::new();
            let mut operations = BTreeMap::<String, usize>::new();
            let mut source_groups = BTreeMap::<String, usize>::new();
            for chunk in table.chunks_logical() {
                for row in table.rows(chunk) {
                    total_rows += 1;
                    if row.col_str(9) == "host_hccl_op" {
                        hccl_duration_ms.push(row.col_i64(3).max(0) as f64 / 1e6);
                        *operations.entry(row.col_str(8).to_owned()).or_default() += 1;
                        *source_groups
                            .entry(format!(
                                "level={} type_id={} op={}",
                                row.col_i32(5),
                                row.col_i32(6),
                                row.col_str(8)
                            ))
                            .or_default() += 1;
                    }
                }
            }
            println!(
                "table={basename} total_rows={total_rows} operations={operations:?} source_groups={source_groups:?}"
            );
            print_stats("host_hccl_op", &mut hccl_duration_ms);
        }
        "hccl.collectives" => {
            let mut api_duration_ms = Vec::new();
            let mut api_operations = BTreeMap::<String, usize>::new();
            let mut per_operation_ms = BTreeMap::<String, Vec<f64>>::new();
            let mut compact_rows = 0usize;
            for chunk in table.chunks_logical() {
                for row in table.rows(chunk) {
                    total_rows += 1;
                    if row.col_str(2) == "api" {
                        let operation = row.col_str(7).to_owned();
                        let duration_ms = row.col_i64(5).max(0) as f64 / 1e6;
                        api_duration_ms.push(duration_ms);
                        *api_operations.entry(operation.clone()).or_default() += 1;
                        per_operation_ms
                            .entry(operation)
                            .or_default()
                            .push(duration_ms);
                    } else if row.col_str(2) == "compact" {
                        compact_rows += 1;
                    }
                }
            }
            println!(
                "table={basename} total_rows={total_rows} compact_rows={compact_rows} operations={api_operations:?}"
            );
            print_stats("collective_api", &mut api_duration_ms);
            for (operation, durations) in &mut per_operation_ms {
                print_stats(&format!("collective_api operation={operation}"), durations);
            }
        }
        "hccl.tasks" => {
            let mut estimated_ms = Vec::new();
            let mut tasks = BTreeMap::<String, usize>::new();
            for chunk in table.chunks_logical() {
                for row in table.rows(chunk) {
                    total_rows += 1;
                    *tasks.entry(row.col_str(6).to_owned()).or_default() += 1;
                    let duration_us = row.col_f64(27);
                    if duration_us > 0.0 {
                        estimated_ms.push(duration_us / 1e3);
                    }
                }
            }
            println!("table={basename} total_rows={total_rows} tasks={tasks:?}");
            print_stats("task_estimate", &mut estimated_ms);
        }
        _ => {
            for chunk in table.chunks_logical() {
                total_rows += table.num_rows(chunk);
            }
            println!("table={basename} total_rows={total_rows}");
        }
    }
    Ok(())
}
