import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.decode import (
    DecodeRequest,
    DecodeTransferQueue,
    SchedulerDisaggregationDecodeMixin,
)
from sglang.srt.disaggregation.decode_migration import SchedulerDecodeMigrationMixin
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import BindDecodeMigrationReqInput
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.result_disposition import (
    ResultDispositionHandler,
)


class _FakeBatch:
    def __init__(self, reqs):
        self.reqs = reqs
        self.batch_is_full = True

    def prepare_for_prebuilt(self):
        pass

    def process_prebuilt(self, server_args, future_map):
        pass

    def filter_batch(self):
        pass

    def is_empty(self):
        return False


class DecodeMigrationDestinationAdmissionTests(unittest.TestCase):
    def _scheduler(self, req):
        running_batch = MagicMock()
        running_batch.batch_size.return_value = 0
        running_batch.is_empty.return_value = True
        tree_cache = SimpleNamespace(root_node=object(), inc_lock_ref=MagicMock())
        return SimpleNamespace(
            server_args=SimpleNamespace(
                enable_decode_migration=True,
                disaggregation_decode_enable_radix_cache=False,
            ),
            disaggregation_mode=DisaggregationMode.NULL,
            waiting_queue=[req],
            req_to_token_pool=SimpleNamespace(size=4),
            token_to_kv_pool_allocator=object(),
            tree_cache=tree_cache,
            model_config=object(),
            enable_overlap=False,
            spec_algorithm=object(),
            max_running_requests=4,
            running_batch=running_batch,
            chunked_req=None,
            future_map={},
            grammar_manager=SimpleNamespace(
                has_waiting_grammars=lambda: False,
            ),
            enable_priority_scheduling=False,
            enable_hisparse=False,
            batch_result_processor=SimpleNamespace(
                process_batch_result_prebuilt=MagicMock()
            ),
        )

    @patch(
        "sglang.srt.disaggregation.decode.set_time_batch",
        autospec=True,
    )
    @patch(
        "sglang.srt.disaggregation.decode.ScheduleBatch.init_new",
        autospec=True,
    )
    def test_migration_uses_shared_prebuilt_admission_path(
        self, init_new, _set_time_batch
    ):
        req = SimpleNamespace(
            has_prebuilt_kv=True,
            last_node=object(),
            kv_committed_len=12,
            prefix_indices=torch.arange(4),
            init_next_round_input=MagicMock(),
            set_extend_input_len=MagicMock(),
            fill_len=0,
        )
        ordinary_req = SimpleNamespace(has_prebuilt_kv=False)
        scheduler = self._scheduler(req)
        scheduler.waiting_queue.append(ordinary_req)
        batch = _FakeBatch([req])
        init_new.return_value = batch

        new_batch = SchedulerDisaggregationDecodeMixin.get_new_prebuilt_batch(
            scheduler, prebuilt_kv_only=True
        )
        SchedulerDisaggregationDecodeMixin.admit_prebuilt_batch(scheduler, new_batch)

        req.init_next_round_input.assert_called_once_with(None)
        self.assertEqual(req.fill_len, 12)
        req.set_extend_input_len.assert_called_once_with(8)
        scheduler.batch_result_processor.process_batch_result_prebuilt.assert_called_once_with(
            batch
        )
        self.assertIs(scheduler.running_batch, batch)
        self.assertEqual(scheduler.waiting_queue, [ordinary_req])

    @patch(
        "sglang.srt.disaggregation.decode.set_time_batch",
        autospec=True,
    )
    @patch(
        "sglang.srt.disaggregation.decode.ScheduleBatch.init_new",
        autospec=True,
    )
    def test_preserves_full_transfer_when_no_radix_prefix_was_matched(
        self, init_new, _set_time_batch
    ):
        req = SimpleNamespace(
            has_prebuilt_kv=True,
            last_node=None,
            kv_committed_len=12,
            prefix_indices=torch.empty(0, dtype=torch.int64),
            init_next_round_input=MagicMock(),
            set_extend_input_len=MagicMock(),
            fill_len=0,
        )
        scheduler = self._scheduler(req)
        init_new.return_value = _FakeBatch([req])

        new_batch = SchedulerDisaggregationDecodeMixin.get_new_prebuilt_batch(
            scheduler, prebuilt_kv_only=True
        )
        SchedulerDisaggregationDecodeMixin.admit_prebuilt_batch(scheduler, new_batch)

        req.init_next_round_input.assert_called_once_with(None)


class ResultDispositionHandlerTests(unittest.TestCase):
    def test_decode_metrics_view_filters_without_copying_schedule_batch(self):
        first = object()
        second = object()
        batch = SimpleNamespace(
            reqs=[first, second],
            seq_lens_cpu=torch.tensor([7, 11]),
            dp_cooperation_info=object(),
            forward_iter=3,
            copy=MagicMock(side_effect=AssertionError("must not copy ScheduleBatch")),
        )

        view = ResultDispositionHandler.decode_metrics_view(batch, [first])

        self.assertEqual(view.reqs, [second])
        self.assertEqual(view.seq_lens_cpu.tolist(), [11])
        self.assertEqual(view.batch_size(), 1)


class DecodeMigrationReceiverInitializationTests(unittest.TestCase):
    @patch(
        "sglang.srt.managers.scheduler.create_decode_transfer_queues",
        autospec=True,
    )
    @patch(
        "sglang.srt.managers.scheduler.kv_cache_builder.get_draft_kv_pool",
        autospec=True,
        return_value=(None, None),
    )
    def test_hybrid_receiver_uses_standard_decode_queues_with_radix_matching(
        self, _get_draft_kv_pool, create_queues
    ):
        prealloc_queue = object()
        transfer_queue = object()
        create_queues.return_value = (prealloc_queue, transfer_queue)
        scheduler = SimpleNamespace(
            draft_worker=object(),
            spec_algorithm=object(),
            model_config=object(),
            server_args=SimpleNamespace(
                disaggregation_mode="null",
                disaggregation_transfer_backend="nixl",
                enable_decode_migration=True,
                language_only=False,
                encoder_transfer_backend="zmq_to_scheduler",
            ),
            req_to_metadata_buffer_idx_allocator=object(),
            disagg_metadata_buffers=object(),
        )

        Scheduler.init_disaggregation(scheduler)

        create_queues.assert_called_once_with(scheduler, None, enable_radix_cache=True)
        self.assertIs(scheduler.disagg_decode_prealloc_queue, prealloc_queue)
        self.assertIs(scheduler.disagg_decode_transfer_queue, transfer_queue)


class DecodeMigrationEarlyReservationTests(unittest.TestCase):
    def test_bind_replaces_placeholders_and_releases_unused_tail(self):
        req = SimpleNamespace(
            rid="destination",
            bootstrap_room=17,
            decode_migration_id="migration",
            decode_migration_bound=False,
            req_pool_idx=0,
            kv_allocated_len=8,
            kv_committed_len=8,
            origin_input_ids=array("q", [1] * 8),
            origin_input_ids_unpadded=array("q", [1] * 8),
            output_ids=array("q"),
            prefix_indices=torch.empty(0, dtype=torch.int64),
            sampling_params=SimpleNamespace(max_new_tokens=1, min_new_tokens=0),
            set_extend_input_len=MagicMock(),
        )
        receiver = MagicMock()
        decode_req = DecodeRequest(req=req, kv_receiver=receiver)
        allocator = SimpleNamespace(page_size=1, free=MagicMock())
        scheduler = SimpleNamespace(
            ps=SimpleNamespace(dp_rank=0),
            disagg_decode_prealloc_queue=SimpleNamespace(queue=[]),
            disagg_decode_transfer_queue=SimpleNamespace(queue=[decode_req]),
            token_to_kv_pool_allocator=allocator,
            req_to_token_pool=SimpleNamespace(
                req_to_token=torch.arange(8, dtype=torch.int64).reshape(1, 8)
            ),
        )

        result = SchedulerDecodeMigrationMixin.bind_decode_migration_destination(
            scheduler,
            BindDecodeMigrationReqInput(
                rid="destination",
                migration_id="migration",
                bootstrap_room=17,
                committed_input_ids=[10, 11, 20, 21],
                pending_input_ids=[22],
                committed_len=4,
                logical_len=5,
                max_new_tokens=9,
                routed_dp_rank=0,
            ),
        )

        self.assertTrue(result.success)
        self.assertEqual(req.origin_input_ids.tolist(), [10, 11, 20, 21])
        self.assertEqual(req.output_ids.tolist(), [])
        self.assertEqual(req.decode_migration_pending_input_id, 22)
        self.assertEqual(req.kv_allocated_len, 4)
        self.assertTrue(req.decode_migration_bound)
        allocator.free.assert_called_once()
        receiver.resume_waiting_timeout.assert_called_once()

    def test_bind_updates_placeholder_still_waiting_for_preallocation(self):
        req = SimpleNamespace(
            rid="destination",
            bootstrap_room=17,
            decode_migration_id="migration",
            decode_migration_bound=False,
            req_pool_idx=None,
            kv_allocated_len=0,
            kv_committed_len=0,
            origin_input_ids=array("q", [0] * 8),
            origin_input_ids_unpadded=array("q", [0] * 8),
            output_ids=array("q"),
            prefix_indices=torch.empty(0, dtype=torch.int64),
            sampling_params=SimpleNamespace(max_new_tokens=1, min_new_tokens=0),
            set_extend_input_len=MagicMock(),
        )
        receiver = MagicMock()
        decode_req = DecodeRequest(req=req, kv_receiver=receiver)
        allocator = SimpleNamespace(page_size=1, free=MagicMock())
        scheduler = SimpleNamespace(
            ps=SimpleNamespace(dp_rank=0),
            disagg_decode_prealloc_queue=SimpleNamespace(queue=[decode_req]),
            disagg_decode_transfer_queue=SimpleNamespace(queue=[]),
            token_to_kv_pool_allocator=allocator,
            req_to_token_pool=SimpleNamespace(req_to_token=None),
        )

        result = SchedulerDecodeMigrationMixin.bind_decode_migration_destination(
            scheduler,
            BindDecodeMigrationReqInput(
                rid="destination",
                migration_id="migration",
                bootstrap_room=17,
                committed_input_ids=[10, 11, 20, 21],
                pending_input_ids=[22],
                committed_len=4,
                logical_len=5,
                max_new_tokens=9,
                routed_dp_rank=0,
            ),
        )

        self.assertTrue(result.success)
        self.assertEqual(req.origin_input_ids.tolist(), [10, 11, 20, 21])
        self.assertEqual(req.decode_migration_pending_input_id, 22)
        self.assertEqual(req.kv_allocated_len, 0)
        self.assertTrue(req.decode_migration_bound)
        allocator.free.assert_not_called()
        receiver.resume_waiting_timeout.assert_not_called()

    def test_completed_transfer_waits_for_exact_state_bind(self):
        req = SimpleNamespace(
            rid="destination",
            decode_migration_bound=False,
            decode_migration_id="migration",
        )
        decode_req = DecodeRequest(req=req, kv_receiver=MagicMock())
        queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
        queue.queue = [decode_req]
        queue.scheduler = SimpleNamespace(enable_decode_hicache=False)
        queue.enable_staging = False
        queue._poll_with_metadata_gate = MagicMock(return_value=[KVPoll.Success])
        queue._commit_transfer_to_req = MagicMock()

        transferred = queue.pop_transferred()

        self.assertEqual(transferred, [])
        self.assertEqual(queue.queue, [decode_req])
        queue._commit_transfer_to_req.assert_not_called()


class DecodeMigrationDestinationTimeoutTests(unittest.TestCase):
    @patch("sglang.srt.managers.scheduler.release_kv_cache")
    @patch("sglang.srt.managers.scheduler.time.perf_counter", return_value=20)
    @patch(
        "sglang.srt.managers.scheduler.envs.SGLANG_REQ_WAITING_TIMEOUT.get",
        return_value=10,
    )
    def test_waiting_timeout_releases_prebuilt_destination_kv(
        self, _timeout, _clock, release_kv_cache
    ):
        req = MagicMock(
            rid="destination",
            has_prebuilt_kv=True,
            time_stats=SimpleNamespace(wait_queue_entry_time=1),
        )
        tree_cache = object()
        scheduler = SimpleNamespace(
            waiting_queue=[req],
            enable_hicache_storage=False,
            disaggregation_mode=DisaggregationMode.NULL,
            tree_cache=tree_cache,
            ipc_channels=SimpleNamespace(
                send_to_tokenizer=SimpleNamespace(send_output=MagicMock())
            ),
        )

        Scheduler._abort_on_waiting_timeout(scheduler)

        self.assertEqual(scheduler.waiting_queue, [])
        release_kv_cache.assert_called_once_with(req, tree_cache)


if __name__ == "__main__":
    unittest.main()
