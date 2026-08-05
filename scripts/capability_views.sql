-- Canonical capability views for CAPABILITY-SPEC A2/A3/A4.
--
-- Preconditions:
--   PROBING_TORCH_PROFILING=on,rate=1.0,backward=on
--   PROBING_GPU_SAMPLE_MS<=200   (for >=90% periodic-sample coverage)
--
-- monotonic_time_sec and wall_time_sec are captured at the hook boundary, not
-- at append time. That is essential because GPU-event rows can be persisted
-- several optimizer steps later. Phase differences use the monotonic clock;
-- periodic-table joins use Unix wall time. The first discovery step and the
-- first LAG window are expected to be absent; do not turn those NULLs into zeros.

CREATE VIEW torch_phase_boundaries AS
SELECT
  rank,
  local_step,
  MIN(CASE WHEN stage = 'pre forward' THEN monotonic_time_sec END) AS fwd_start_sec,
  MAX(CASE WHEN stage = 'post forward' THEN monotonic_time_sec END) AS fwd_end_sec,
  MIN(CASE WHEN stage = 'pre backward' THEN monotonic_time_sec END) AS bwd_start_sec,
  MAX(CASE WHEN stage = 'post backward' THEN monotonic_time_sec END) AS bwd_end_sec,
  MIN(CASE WHEN stage = 'pre step' THEN monotonic_time_sec END) AS opt_start_sec,
  MAX(CASE WHEN stage = 'post step' THEN monotonic_time_sec END) AS opt_end_sec,
  MAX(CASE WHEN stage = 'post step' THEN wall_time_sec END) AS opt_end_wall_sec
FROM python.torch_trace
GROUP BY rank, local_step;

CREATE VIEW torch_phase_accounting AS
WITH sequenced AS (
  SELECT
    b.*,
    LAG(opt_end_sec) OVER (
      PARTITION BY rank ORDER BY local_step
    ) AS previous_opt_end_sec
  FROM torch_phase_boundaries b
), phases AS (
  SELECT
    rank,
    local_step,
    (fwd_start_sec - previous_opt_end_sec) * 1000.0 AS data_gap_ms,
    (fwd_end_sec - fwd_start_sec) * 1000.0 AS forward_ms,
    (bwd_end_sec - bwd_start_sec) * 1000.0 AS backward_ms,
    (bwd_end_sec - fwd_start_sec) * 1000.0 AS compute_ms,
    (opt_start_sec - bwd_end_sec) * 1000.0 AS wait_gap_ms,
    (opt_end_sec - opt_start_sec) * 1000.0 AS optimizer_ms,
    previous_opt_end_sec,
    opt_end_sec
  FROM sequenced
)
SELECT
  p.*,
  t.step_duration_sec * 1000.0 AS step_ms,
  t.step_duration_sec * 1000.0
    - (p.data_gap_ms + p.compute_ms + p.wait_gap_ms + p.optimizer_ms)
      AS residual_ms,
  CASE
    WHEN t.step_duration_sec > 0 THEN
      (t.step_duration_sec * 1000.0
        - (p.data_gap_ms + p.compute_ms + p.wait_gap_ms + p.optimizer_ms))
      / (t.step_duration_sec * 1000.0)
    ELSE NULL
  END AS residual_ratio
FROM phases p
JOIN python.torch_step_timing t
  ON p.rank = t.rank AND p.local_step = t.local_step;

CREATE VIEW step_windows AS
WITH boundaries AS (
  SELECT
    rank,
    local_step,
    opt_end_wall_sec AS ts_end_sec,
    LAG(opt_end_wall_sec) OVER (
      PARTITION BY rank ORDER BY local_step
    ) AS ts_start_sec
  FROM torch_phase_boundaries
)
SELECT
  rank,
  local_step,
  CAST(ts_start_sec * 1000000 AS BIGINT) AS ts_start,
  CAST(ts_end_sec * 1000000 AS BIGINT) AS ts_end
FROM boundaries
WHERE ts_start_sec IS NOT NULL AND ts_end_sec > ts_start_sec;
