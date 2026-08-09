use datafusion::arrow::datatypes::SchemaRef;
use datafusion::logical_expr::Expr;
use datafusion_sql::unparser::expr_to_sql;

use crate::core::plugin_advanced::can_push_filter_exact_for_schema;

/// Build a remote `probe.*` SQL statement with conservative filter/limit pushdown.
pub fn build_remote_table_sql(
    schema_name: &str,
    table_name: &str,
    table_schema: &SchemaRef,
    projection: Option<&Vec<usize>>,
    filters: &[Expr],
    limit: Option<usize>,
) -> String {
    let cols = match projection {
        Some(idxs) if !idxs.is_empty() => idxs
            .iter()
            .map(|i| {
                let name = table_schema.field(*i).name();
                format!("\"{name}\"")
            })
            .collect::<Vec<_>>()
            .join(", "),
        _ => "*".to_string(),
    };

    let mut sql = format!("SELECT {cols} FROM probe.{schema_name}.{table_name}");

    let where_parts: Vec<String> = filters
        .iter()
        .filter(|f| can_push_filter_exact_for_schema(table_schema, f))
        // Expr::Display is a diagnostic representation (for example it emits
        // Utf8("all_reduce")), not executable SQL.  Round-trip through
        // DataFusion's SQL unparser so literals retain valid SQL syntax on
        // remote peers.  If a filter cannot be unparsed, omit the pushdown and
        // let the coordinator apply it after merging.
        .filter_map(|f| expr_to_sql(f).ok().map(|sql| sql.to_string()))
        .collect();
    if !where_parts.is_empty() {
        sql.push_str(" WHERE ");
        sql.push_str(&where_parts.join(" AND "));
    }

    if let Some(n) = limit {
        sql.push_str(&format!(" LIMIT {n}"));
    }

    sql
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion::logical_expr::{col, lit};

    use super::*;

    #[test]
    fn builds_where_and_limit() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("rank", DataType::Int32, true),
            Field::new("step", DataType::Int64, true),
        ]));
        let filters = vec![col("rank").eq(lit(1i32))];
        let sql = build_remote_table_sql("python", "metrics", &schema, None, &filters, Some(10));
        assert!(sql.contains("FROM probe.python.metrics"));
        assert!(sql.contains("WHERE"));
        assert!(sql.contains("LIMIT 10"));
    }

    #[test]
    fn remote_sql_uses_probe_catalog_not_global() {
        let schema = Arc::new(Schema::new(vec![Field::new("rank", DataType::Int32, true)]));
        let sql = build_remote_table_sql("demo", "metrics", &schema, None, &[], None);
        assert!(sql.contains("probe.demo.metrics"));
        assert!(!sql.contains("global."));
        assert!(!sql.contains("_probe_node"));
    }

    #[test]
    fn string_literal_filter_is_valid_sql() {
        let schema = Arc::new(Schema::new(vec![Field::new("op", DataType::Utf8, true)]));
        let filters = vec![col("op").eq(lit("all_reduce"))];
        let sql =
            build_remote_table_sql("python", "comm_collective", &schema, None, &filters, None);
        assert!(sql.contains("'all_reduce'"), "{sql}");
        assert!(!sql.contains("Utf8("), "{sql}");
    }
}
