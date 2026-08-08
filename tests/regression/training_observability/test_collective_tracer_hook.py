"""CollectiveTracer hook behavior with mocked torch.distributed (no GPU)."""

from unittest.mock import MagicMock, patch

import pytest

from probing.profiling.collective.coll import CollectiveTracer
from probing.profiling.collective.record import CommCollective, CommRecordMode

from .conftest import table_rows


@pytest.fixture
def mock_dist():
    dist = MagicMock()
    dist.is_initialized.return_value = True
    dist.get_world_size.return_value = 4
    dist.get_rank.return_value = 1
    return dist


@pytest.fixture
def mock_tensor():
    import torch

    t = torch.zeros(128)
    return t


@pytest.mark.training_observability
class TestCollectiveTracerHook:
    def test_skips_single_process_job(self, mock_tensor):
        tracer = CollectiveTracer(mode=CommRecordMode.LITE)
        calls = {"n": 0}

        def fake_all_reduce(tensor, *args, **kwargs):
            calls["n"] += 1
            return None

        wrapper = tracer._trace_wrapper("all_reduce", fake_all_reduce)

        with patch.dict("os.environ", {"WORLD_SIZE": "1"}, clear=False):
            with patch("probing.profiling.collective.coll.dist") as dist:
                dist.is_initialized.return_value = False
                wrapper(mock_tensor)

        assert calls["n"] == 1
        assert len(table_rows(CommCollective, 5)) == 0

    def test_lite_hook_records_after_collective(self, mock_dist, mock_tensor):
        tracer = CollectiveTracer(
            mode=CommRecordMode.LITE,
            trace_event=False,
            cuda_sync=False,
        )

        def fake_all_reduce(tensor, *args, **kwargs):
            return None

        wrapper = tracer._trace_wrapper("all_reduce", fake_all_reduce)

        with patch("probing.profiling.collective.coll.dist", mock_dist):
            wrapper(mock_tensor)

        rows = table_rows(CommCollective, 5)
        assert len(rows) == 1
        assert rows[0]["op"] == "all_reduce"
        assert rows[0]["group_size"] == 4
        assert rows[0]["bytes"] > 0
        assert rows[0]["timing_source"] == "host_api"

    def test_sync_timing_is_labeled_as_device_completion(
        self, mock_dist, mock_tensor
    ):
        tracer = CollectiveTracer(
            mode=CommRecordMode.LITE,
            trace_event=False,
            cuda_sync=True,
        )
        tracer._maybe_sync = MagicMock()
        wrapper = tracer._trace_wrapper("all_reduce", lambda tensor: None)

        with patch("probing.profiling.collective.coll.dist", mock_dist):
            wrapper(mock_tensor)

        rows = table_rows(CommCollective, 5)
        assert len(rows) == 1
        assert rows[0]["timing_source"] == "device_sync_wall"
        assert tracer._maybe_sync.call_count == 2

    def test_full_mode_opens_live_span_during_call(self, mock_dist, mock_tensor):
        tracer = CollectiveTracer(
            mode=CommRecordMode.FULL,
            cuda_sync=False,
            resolve_group_ranks=False,
        )
        seen = {}

        def fake_all_reduce(tensor, *args, **kwargs):
            from probing.tracing import current_span

            span = current_span()
            seen["had_span"] = span is not None
            seen["name"] = getattr(span, "name", None) if span else None
            return None

        wrapper = tracer._trace_wrapper("all_reduce", fake_all_reduce)

        with patch("probing.profiling.collective.coll.dist", mock_dist):
            wrapper(mock_tensor)

        assert seen.get("had_span") is True
        assert seen.get("name") == "all_reduce"
        rows = table_rows(CommCollective, 5)
        assert len(rows) == 1

    def test_sync_prefers_npu_when_available(self):
        with patch("probing.profiling.collective.coll.torch") as mock_torch:
            mock_torch.npu.is_available.return_value = True
            mock_torch.cuda.is_available.return_value = True
            tracer = CollectiveTracer(cuda_sync=True)
            tracer._maybe_sync()

        mock_torch.npu.synchronize.assert_called_once_with()
        mock_torch.cuda.synchronize.assert_not_called()

    def test_async_work_finalizes_only_once(self, mock_dist, mock_tensor):
        tracer = CollectiveTracer(
            mode=CommRecordMode.LITE,
            trace_event=False,
            cuda_sync=False,
        )
        work = MagicMock()
        work.wait.return_value = True
        work.exception.return_value = None
        wrapper = tracer._trace_wrapper("all_reduce", MagicMock(return_value=work))

        with patch("probing.profiling.collective.coll.dist", mock_dist):
            timed_work = wrapper(mock_tensor, async_op=True)
            assert timed_work.wait() is True
            assert timed_work.wait() is True
            assert timed_work.exception() is None

        assert work.wait.call_count == 2
        assert work.exception.call_count == 1
        rows = table_rows(CommCollective, 5)
        assert len(rows) == 1
