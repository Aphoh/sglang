import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.base.conn import NIXL_LOW_LATENCY_RECEIVER
from sglang.srt.disaggregation.decode import (
    DecodeRequest,
    DecodeTransferQueue,
    SchedulerDisaggregationDecodeMixin,
    _ensure_prebuilt_root_lock,
)
from sglang.srt.disaggregation.prebuilt_kv import (
    PrebuiltKVFrontier,
    PrebuiltKVState,
    bind_prebuilt_kv_from_transfer,
)
from sglang.srt.disaggregation.utils import DisaggregationMode, MetadataBuffers
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.metrics_reporter import (
    decode_metrics_batch_view,
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


class DecodeMetricsBatchViewTests(unittest.TestCase):
    def test_excludes_discarded_request_from_batch_indexed_metrics(self):
        first = object()
        second = object()
        batch = SimpleNamespace(
            reqs=[first, second],
            seq_lens_cpu=torch.tensor([7, 11]),
            dp_cooperation_info=object(),
            forward_iter=3,
        )

        view = decode_metrics_batch_view(batch, [first])

        self.assertEqual(view.reqs, [second])
        self.assertEqual(view.seq_lens_cpu.tolist(), [11])
        self.assertEqual(view.batch_size(), 1)


class DecodeMigrationDestinationAdmissionTests(unittest.TestCase):
    def test_prebuilt_request_is_anchored_before_transfer_can_fail(self):
        req = SimpleNamespace(prebuilt_kv=object(), last_node=None)
        root = object()
        tree_cache = SimpleNamespace(
            root_node=root,
            inc_lock_ref=MagicMock(),
            is_chunk_cache=lambda: False,
        )

        _ensure_prebuilt_root_lock(req, tree_cache)

        self.assertIs(req.last_node, root)
        tree_cache.inc_lock_ref.assert_called_once_with(root)

    def _scheduler(self, req):
        running_batch = MagicMock()
        running_batch.batch_size.return_value = 0
        running_batch.is_empty.return_value = True
        tree_cache = SimpleNamespace(
            root_node=object(),
            inc_lock_ref=MagicMock(),
            is_chunk_cache=lambda: False,
        )
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
            prebuilt_kv=object(),
            last_node=object(),
            kv_committed_len=12,
            prefix_indices=torch.arange(4),
            init_next_round_input=MagicMock(),
            set_extend_input_len=MagicMock(),
            fill_len=0,
        )
        ordinary_req = SimpleNamespace(prebuilt_kv=None)
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
            prebuilt_kv=object(),
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

    @patch(
        "sglang.srt.disaggregation.decode.set_time_batch",
        autospec=True,
    )
    @patch(
        "sglang.srt.disaggregation.decode.ScheduleBatch.init_new",
        autospec=True,
    )
    def test_chunk_cache_keeps_prebuilt_request_unanchored(
        self, init_new, _set_time_batch
    ):
        req = SimpleNamespace(
            prebuilt_kv=object(),
            last_node=None,
            kv_committed_len=12,
            prefix_indices=torch.empty(0, dtype=torch.int64),
            init_next_round_input=MagicMock(),
            set_extend_input_len=MagicMock(),
            fill_len=0,
        )
        scheduler = self._scheduler(req)
        scheduler.tree_cache = SimpleNamespace(
            inc_lock_ref=MagicMock(),
            is_chunk_cache=lambda: True,
        )
        init_new.return_value = _FakeBatch([req])

        new_batch = SchedulerDisaggregationDecodeMixin.get_new_prebuilt_batch(
            scheduler, prebuilt_kv_only=True
        )
        SchedulerDisaggregationDecodeMixin.admit_prebuilt_batch(scheduler, new_batch)

        self.assertIsNone(req.last_node)
        scheduler.tree_cache.inc_lock_ref.assert_not_called()


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

        create_queues.assert_called_once_with(
            scheduler,
            None,
            enable_radix_cache=True,
            nixl_transport_config=NIXL_LOW_LATENCY_RECEIVER,
        )
        self.assertIs(scheduler.disagg_decode_prealloc_queue, prealloc_queue)
        self.assertIs(scheduler.disagg_decode_transfer_queue, transfer_queue)


class DecodeMigrationEarlyReservationTests(unittest.TestCase):
    def test_frontier_metadata_round_trip(self):
        buffers = MetadataBuffers(
            2,
            hidden_size=16,
            hidden_states_dtype=torch.float32,
            decode_migration_max_tokens=32,
        )
        buffers.set_decode_migration_frontier(
            1,
            committed_input_ids=[10, 11, 20, 21],
            pending_input_id=22,
            prompt_len=2,
            logical_len=5,
            output_tokens_seen=3,
            max_new_tokens=9,
            min_new_tokens=1,
        )

        self.assertEqual(
            buffers.get_decode_migration_frontier(1),
            PrebuiltKVFrontier(
                committed_input_ids=[10, 11, 20, 21],
                pending_input_id=22,
                prompt_len=2,
                committed_len=4,
                logical_len=5,
                output_tokens_seen=3,
                max_new_tokens=9,
                min_new_tokens=1,
            ),
        )

    def test_frontier_binding_replaces_placeholder_state(self):
        req = SimpleNamespace(
            origin_input_ids=array("q", [0] * 8),
            origin_input_ids_unpadded=array("q", [0] * 8),
            output_ids=array("q"),
            prebuilt_kv=PrebuiltKVState("migration"),
            req_pool_idx=None,
            kv_allocated_len=0,
            sampling_params=SimpleNamespace(max_new_tokens=1, min_new_tokens=0),
        )
        decode_req = DecodeRequest(
            req=req, kv_receiver=MagicMock(), metadata_buffer_index=0
        )
        scheduler = SimpleNamespace(
            disagg_metadata_buffers=SimpleNamespace(
                get_decode_migration_frontier=lambda _index: PrebuiltKVFrontier(
                    committed_input_ids=[10, 11, 20, 21],
                    pending_input_id=22,
                    prompt_len=2,
                    committed_len=4,
                    logical_len=5,
                    output_tokens_seen=3,
                    max_new_tokens=9,
                    min_new_tokens=1,
                )
            ),
            disagg_decode_prealloc_queue=SimpleNamespace(queue=[decode_req]),
        )

        error = bind_prebuilt_kv_from_transfer(scheduler, decode_req)

        self.assertIsNone(error)
        self.assertEqual(req.origin_input_ids.tolist(), [10, 11, 20, 21])
        self.assertEqual(req.prebuilt_kv.pending_input_id, 22)
        self.assertTrue(req.prebuilt_kv.ready)
        self.assertEqual(req.sampling_params.max_new_tokens, 9)

    @patch("sglang.srt.disaggregation.decode.bind_prebuilt_kv_from_transfer")
    def test_completed_transfer_binds_from_transferred_frontier(self, bind_prebuilt):
        req = SimpleNamespace(
            rid="destination",
            prebuilt_kv=PrebuiltKVState("migration"),
            bootstrap_room=17,
            return_logprob=False,
            finished_reason=None,
        )
        decode_req = DecodeRequest(
            req=req, kv_receiver=MagicMock(), metadata_buffer_index=0
        )
        queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
        queue.queue = [decode_req]
        queue.scheduler = SimpleNamespace(
            enable_decode_hicache=False,
            enable_hisparse=False,
            metrics_reporter=SimpleNamespace(enable_metrics=False),
        )
        queue.enable_staging = False
        queue.metadata_buffers = SimpleNamespace(
            bootstrap_room=torch.zeros((1, 1), dtype=torch.int64)
        )
        queue.req_to_metadata_buffer_idx_allocator = SimpleNamespace(free=MagicMock())
        queue._poll_with_metadata_gate = MagicMock(return_value=[KVPoll.Success])
        queue._commit_transfer_to_req = MagicMock()
        bind_prebuilt.side_effect = lambda _scheduler, _decode_req: setattr(
            req.prebuilt_kv, "pending_input_id", 22
        )

        transferred = queue.pop_transferred()

        self.assertEqual(transferred, [req])
        self.assertEqual(queue.queue, [])
        bind_prebuilt.assert_called_once_with(queue.scheduler, decode_req)
        queue._commit_transfer_to_req.assert_called_once_with(decode_req)


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
            prebuilt_kv=object(),
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
