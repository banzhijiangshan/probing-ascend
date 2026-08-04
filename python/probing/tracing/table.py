"""Trace event table schema (SQL / federation)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Iterable, List, Optional

import probing

# Materialized span rows derived from ``python.trace_event`` (start/end join).
# Use span ``time`` (ns since epoch), not the memtable ingestion ``timestamp``.
SPANS_SQL = """
SELECT
    s.trace_id,
    s.span_id,
    COALESCE(s.parent_id, -1) AS parent_span_id,
    s.name,
    s.phase,
    CAST(CAST(s.time AS BIGINT) / 1000 AS BIGINT) AS start_us,
    CAST(CAST(e.time AS BIGINT) / 1000 AS BIGINT) AS end_us,
    CAST((CAST(e.time AS BIGINT) - CAST(s.time AS BIGINT)) / 1000 AS BIGINT) AS duration_us,
    s.thread_id,
    s.location,
    s.attributes
FROM python.trace_event s
JOIN python.trace_event e
  ON s.span_id = e.span_id AND e.record_type = 'span_end'
WHERE s.record_type = 'span_start'
"""

_TABLE_NAME = "trace_event"
_FIELDS = [
    "record_type",
    "trace_id",
    "span_id",
    "name",
    "time",
    "thread_id",
    "parent_id",
    "phase",
    "location",
    "attributes",
    "event_attributes",
]
_QUALIFIED = "python.trace_event"
_TABLE_DOC = "Row model for trace records."


@dataclass
class TraceEvent:
    """Row model for trace records.

    Each saved instance is one of: span_start, span_end, event.
    """

    record_type: str
    trace_id: int
    span_id: int
    name: str
    time: int
    thread_id: int = 0
    parent_id: Optional[int] = -1
    phase: Optional[str] = ""
    location: Optional[str] = ""
    attributes: Optional[str] = ""
    event_attributes: Optional[str] = ""

    _table: ClassVar[Any] = None

    @classmethod
    def _ensure_table(cls):
        """Create ``python.trace_event`` on first write (not at import)."""
        if cls._table is None:
            cls._table = probing.ExternalTable(
                _TABLE_NAME,
                _FIELDS,
                table_doc=_TABLE_DOC,
            )
            probing.register_table_docs(_QUALIFIED, _TABLE_DOC, None)
        return cls._table

    @classmethod
    def init_table(cls):
        return cls._ensure_table()

    @classmethod
    def drop(cls):
        cls._table = None
        try:
            probing.ExternalTable.drop(_TABLE_NAME)
        except Exception:
            pass

    @classmethod
    def _row(cls, obj: "TraceEvent") -> tuple:
        return tuple(getattr(obj, f) for f in _FIELDS)

    @classmethod
    def append(cls, obj: "TraceEvent") -> None:
        cls._ensure_table().append(cls._row(obj))

    @classmethod
    def append_many(cls, instances: Iterable["TraceEvent"]) -> None:
        cls._ensure_table().append_many([cls._row(i) for i in instances])

    @classmethod
    def take(cls, n: int) -> List:
        return cls._ensure_table().take(n)

    def save(self) -> None:
        type(self).append(self)
