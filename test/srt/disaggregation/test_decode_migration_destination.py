import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.disaggregation.decode_migration import SchedulerDecodeMigrationMixin
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.scheduler import Scheduler


class _FakeBatch:
    def __init__(self, reqs):
        self.reqs = reqs

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
        return SimpleNamespace(
            server_args=SimpleNamespace(enable_decode_migration=True),
            disaggregation_mode=DisaggregationMode.NULL,
            waiting_queue=[req],
            req_to_token_pool=SimpleNamespace(size=4),
            token_to_kv_pool_allocator=object(),
            tree_cache=object(),
            model_config=object(),
            enable_overlap=False,
            spec_algorithm=object(),
            max_running_requests=4,
            running_batch=running_batch,
            future_map={},
            batch_result_processor=SimpleNamespace(
                process_batch_result_prebuilt=MagicMock()
            ),
        )

    @patch(
        "sglang.srt.disaggregation.decode_migration.set_time_batch",
        autospec=True,
    )
    @patch(
        "sglang.srt.disaggregation.decode_migration.ScheduleBatch.init_new",
        autospec=True,
    )
    def test_preserves_preallocated_radix_prefix(self, init_new, _set_time_batch):
        req = SimpleNamespace(
            is_decode_migration_destination=True,
            last_node=object(),
            kv_committed_len=12,
            prefix_indices=torch.arange(4),
            init_next_round_input=MagicMock(),
            set_extend_input_len=MagicMock(),
            fill_len=0,
        )
        scheduler = self._scheduler(req)
        batch = _FakeBatch([req])
        init_new.return_value = batch

        SchedulerDecodeMigrationMixin.admit_ready_decode_migrations(scheduler)

        req.init_next_round_input.assert_called_once_with(None)
        self.assertEqual(req.fill_len, 12)
        req.set_extend_input_len.assert_called_once_with(8)
        scheduler.batch_result_processor.process_batch_result_prebuilt.assert_called_once_with(
            batch
        )
        self.assertIs(scheduler.running_batch, batch)

    @patch(
        "sglang.srt.disaggregation.decode_migration.set_time_batch",
        autospec=True,
    )
    @patch(
        "sglang.srt.disaggregation.decode_migration.ScheduleBatch.init_new",
        autospec=True,
    )
    def test_preserves_full_transfer_when_no_radix_prefix_was_matched(
        self, init_new, _set_time_batch
    ):
        req = SimpleNamespace(
            is_decode_migration_destination=True,
            last_node=None,
            kv_committed_len=12,
            prefix_indices=torch.empty(0, dtype=torch.int64),
            init_next_round_input=MagicMock(),
            set_extend_input_len=MagicMock(),
            fill_len=0,
        )
        scheduler = self._scheduler(req)
        init_new.return_value = _FakeBatch([req])

        SchedulerDecodeMigrationMixin.admit_ready_decode_migrations(scheduler)

        req.init_next_round_input.assert_called_once_with(None)


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


if __name__ == "__main__":
    unittest.main()
