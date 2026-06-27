import asyncio
import unittest
from collections import deque
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import torch

from sglang.srt.disaggregation.decode_migration import (
    DecodeMigrationRegistry,
    DecodeMigrationState,
    SchedulerDecodeMigrationMixin,
)
from sglang.srt.disaggregation.utils import DisaggregationMode, KVClassType
from sglang.srt.managers.io_struct import (
    FinalizeDecodeMigrationReqInput,
    FinalizeDecodeMigrationReqOutput,
    PrepareDecodeMigrationReqInput,
    PrepareDecodeMigrationReqOutput,
)
from sglang.srt.managers.scheduler_components.result_disposition import (
    ResultDisposition,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager


class _Batch:
    def __init__(self, reqs):
        self.reqs = list(reqs)
        self.decoding_reqs = None
        self.batch_is_full = True

    def filter_batch(self, keep_indices=None, **_kwargs):
        if keep_indices is None:
            keep_indices = list(range(len(self.reqs)))
        self.reqs = [self.reqs[index] for index in keep_indices]


class _Req:
    def __init__(self, rid, output_ids=None):
        self.rid = rid
        self.origin_input_ids = [10, 11]
        self.output_ids = list(output_ids or [20, 21])
        self.kv_committed_len = len(self.origin_input_ids) + len(self.output_ids) - 1
        self.req_pool_idx = 1
        self.to_finish = None
        self._finished = False

    def finished(self):
        return self._finished


class _Allocator:
    def __init__(self):
        self.next_index = 0
        self.freed = []

    def available_size(self):
        return 8

    def alloc(self):
        index = self.next_index
        self.next_index += 1
        return index

    def free(self, index):
        self.freed.append(index)


class _Sender:
    def __init__(self, **_kwargs):
        self.clear_count = 0

    def clear(self):
        self.clear_count += 1


class _Scheduler(SchedulerDecodeMigrationMixin):
    def __init__(self, reqs, *, overlap=False, dp_rank=0):
        self.server_args = SimpleNamespace(enable_decode_migration=True)
        self.disaggregation_mode = DisaggregationMode.NULL
        self.enable_overlap = overlap
        self.result_queue = deque()
        self.decode_migrations = DecodeMigrationRegistry()
        self.running_batch = _Batch(reqs)
        self.last_batch = self.running_batch
        self.cur_batch = self.running_batch
        self.req_to_metadata_buffer_idx_allocator = _Allocator()
        self.disagg_metadata_buffers = SimpleNamespace(
            output_ids=torch.zeros((8, 1), dtype=torch.int64),
            cached_tokens=torch.zeros((8, 1), dtype=torch.int64),
            bootstrap_room=torch.zeros((8, 1), dtype=torch.int64),
        )
        self.transfer_backend = object()
        self.spec_algorithm = SimpleNamespace(is_none=lambda: True)
        self.ps = SimpleNamespace(tp_rank=0, pp_rank=0, dp_rank=dp_rank)
        self.tree_cache = object()
        self.process_batch_result = MagicMock()

    def _get_decode_migration_kv_manager(self):
        return object()

    def detach_request_from_scheduling(self, req):
        removed = False
        seen_batches = set()
        for batch in (self.running_batch, self.last_batch, self.cur_batch):
            if id(batch) in seen_batches:
                continue
            seen_batches.add(id(batch))
            if req not in batch.reqs:
                continue
            batch.filter_batch(
                keep_indices=[
                    i for i, candidate in enumerate(batch.reqs) if candidate is not req
                ]
            )
            batch.batch_is_full = False
            removed |= batch is self.running_batch
        return removed


def _prepare(
    rid,
    migration_id,
    room,
    output_tokens_seen=0,
    target_sequence_length=None,
):
    return PrepareDecodeMigrationReqInput(
        rid=rid,
        migration_id=migration_id,
        bootstrap_host="127.0.0.1",
        bootstrap_port=5000,
        bootstrap_room=room,
        output_tokens_seen=output_tokens_seen,
        target_sequence_length=target_sequence_length,
    )


class DecodeMigrationSourceTests(unittest.TestCase):
    def _sender_patch(self):
        return patch(
            "sglang.srt.disaggregation.decode_migration.get_kv_class",
            side_effect=lambda _backend, kind: (
                _Sender if kind == KVClassType.SENDER else object
            ),
        )

    def test_overlap_quiesce_uses_the_already_consumed_frontier(self):
        target = _Req("target", output_ids=[20, 21])
        other = _Req("other")
        scheduler = _Scheduler([target, other], overlap=True)
        with self._sender_patch():
            output = scheduler.prepare_decode_migration(
                _prepare("target", "migration-1", 17, output_tokens_seen=1)
            )

        self.assertTrue(output.success)
        self.assertEqual(output.pending_input_ids, [21])
        self.assertEqual(scheduler.running_batch.reqs, [other])
        self.assertEqual(len(scheduler.result_queue), 0)
        scheduler.process_batch_result.assert_not_called()

    def test_overlap_prepare_defers_to_the_next_completed_frontier(self):
        req = _Req("request", output_ids=[20])
        scheduler = _Scheduler([req], overlap=True)
        # The in-flight forward has already advanced KV, but its sampled token
        # has not yet been processed into output_ids.
        req.kv_committed_len += 1
        scheduler.result_queue.append((_Batch([req]), object()))

        output = scheduler.prepare_decode_migration(
            _prepare("request", "migration", 17)
        )

        self.assertTrue(output.success)
        self.assertEqual(output.status, "armed")
        arm = scheduler.decode_migrations.get("migration")
        self.assertEqual(arm.request.target_sequence_length, 4)

        req.output_ids.append(21)
        with self._sender_patch():
            self.assertTrue(scheduler.maybe_park_decode_migration_at_boundary(req))

    def test_unrelated_overlap_result_does_not_defer_prepare(self):
        req = _Req("request", output_ids=[20])
        other = _Req("other")
        scheduler = _Scheduler([req], overlap=True)
        scheduler.result_queue.append((_Batch([other]), object()))

        with self._sender_patch():
            output = scheduler.prepare_decode_migration(
                _prepare("request", "migration", 17)
            )

        self.assertTrue(output.success)
        self.assertEqual(output.status, "prepared")

    def test_prepare_rejects_speculative_decoding(self):
        req = _Req("request")
        scheduler = _Scheduler([req])
        scheduler.spec_algorithm = SimpleNamespace(is_none=lambda: False)

        output = scheduler.prepare_decode_migration(
            _prepare("request", "migration", 17)
        )

        self.assertFalse(output.success)
        self.assertIn("speculative", output.error)

    @patch("sglang.srt.disaggregation.decode_migration.release_kv_cache")
    def test_cancel_keeps_overlap_tombstone_until_stale_result_is_consumed(
        self, release_kv_cache
    ):
        req = _Req("request", output_ids=[20])
        scheduler = _Scheduler([req], overlap=True)
        scheduler.prepare_decode_migration(
            _prepare("request", "migration", 17, target_sequence_length=4)
        )
        req.output_ids.append(21)
        req.kv_committed_len += 1
        scheduler.result_queue.append((_Batch([req]), object()))
        with self._sender_patch():
            self.assertTrue(scheduler.maybe_park_decode_migration_at_boundary(req))

        output = scheduler.finalize_decode_migration(
            FinalizeDecodeMigrationReqInput(
                rid="request", migration_id="migration", action="cancel"
            )
        )

        self.assertTrue(output.success)
        record = scheduler.decode_migrations.get("migration")
        self.assertEqual(record.state, DecodeMigrationState.AWAITING_STALE_RESULT)
        release_kv_cache.assert_not_called()
        self.assertEqual(
            scheduler.get_decode_migration_result_disposition(req),
            ResultDisposition.DISCARD,
        )
        self.assertIsNone(scheduler.decode_migrations.get("migration"))
        release_kv_cache.assert_called_once_with(
            req, scheduler.tree_cache, is_insert=False
        )

    @patch("sglang.srt.disaggregation.decode_migration.release_kv_cache")
    def test_unrelated_overlap_result_does_not_create_a_tombstone(
        self, release_kv_cache
    ):
        req = _Req("request", output_ids=[20])
        other = _Req("other")
        scheduler = _Scheduler([req], overlap=True)
        scheduler.prepare_decode_migration(
            _prepare("request", "migration", 17, target_sequence_length=4)
        )
        req.output_ids.append(21)
        req.kv_committed_len += 1
        scheduler.result_queue.append((_Batch([other]), object()))
        with self._sender_patch():
            self.assertTrue(scheduler.maybe_park_decode_migration_at_boundary(req))

        output = scheduler.finalize_decode_migration(
            FinalizeDecodeMigrationReqInput(
                rid="request", migration_id="migration", action="cancel"
            )
        )

        self.assertTrue(output.success)
        self.assertIsNone(scheduler.decode_migrations.get("migration"))
        release_kv_cache.assert_called_once_with(
            req, scheduler.tree_cache, is_insert=False
        )

    @patch("sglang.srt.disaggregation.decode_migration.release_kv_cache")
    def test_failed_transfer_releases_source_and_stops_counting_as_active(
        self, release_kv_cache
    ):
        req = _Req("request")
        scheduler = _Scheduler([req])
        with self._sender_patch():
            scheduler.prepare_decode_migration(_prepare("request", "migration", 17))

        scheduler._fail_decode_migration(
            scheduler.decode_migrations.get("migration"), "test failure"
        )

        self.assertFalse(scheduler.decode_migrations.has_active_transfers())
        self.assertIsNone(scheduler.decode_migrations.get("migration"))
        release_kv_cache.assert_called_once_with(
            req, scheduler.tree_cache, is_insert=False
        )

    def test_concurrent_requests_are_parked_independently(self):
        first = _Req("first")
        second = _Req("second")
        scheduler = _Scheduler([first, second])
        with self._sender_patch():
            one = scheduler.prepare_decode_migration(_prepare("first", "one", 17))
            two = scheduler.prepare_decode_migration(_prepare("second", "two", 18))

        self.assertTrue(one.success)
        self.assertTrue(two.success)
        self.assertEqual(scheduler.running_batch.reqs, [])
        self.assertEqual(
            {record.migration_id for record in scheduler.decode_migrations.transfers()},
            {"one", "two"},
        )

    def test_sequence_arm_parks_exactly_at_boundary(self):
        req = _Req("request", output_ids=[20])
        scheduler = _Scheduler([req], overlap=True)
        output = scheduler.prepare_decode_migration(
            _prepare(
                "request",
                "migration",
                17,
                output_tokens_seen=2,
                target_sequence_length=4,
            )
        )

        self.assertTrue(output.success)
        self.assertEqual(output.status, "armed")
        self.assertEqual(scheduler.running_batch.reqs, [req])

        req.output_ids.append(21)
        req.kv_committed_len += 1
        with self._sender_patch():
            parked = scheduler.maybe_park_decode_migration_at_boundary(req)

        self.assertTrue(parked)
        self.assertEqual(len(req.origin_input_ids) + len(req.output_ids), 4)
        self.assertEqual(scheduler.running_batch.reqs, [])
        self.assertIsNotNone(scheduler.decode_migrations.get_for_rid(req.rid))
        self.assertEqual(scheduler.decode_migrations.get("migration").committed_len, 3)

    def test_sequence_arm_exports_exact_frontier_after_overlap_overshoot(self):
        req = _Req("request", output_ids=[20])
        scheduler = _Scheduler([req], overlap=True)
        scheduler.prepare_decode_migration(
            _prepare(
                "request",
                "migration",
                17,
                output_tokens_seen=2,
                target_sequence_length=4,
            )
        )

        req.output_ids.extend([21, 22, 23])
        req.kv_committed_len += 3
        with self._sender_patch():
            parked = scheduler.maybe_park_decode_migration_at_boundary(req)

        self.assertTrue(parked)
        record = scheduler.decode_migrations.get("migration")
        self.assertEqual(record.committed_len, 3)
        self.assertEqual(record.logical_len, 4)
        self.assertEqual(record.pending_input_ids, [21])
        output = scheduler.prepare_decode_migration(
            _prepare("request", "migration", 17, output_tokens_seen=4)
        )
        self.assertEqual(output.committed_input_ids, [10, 11, 20])
        self.assertEqual(output.pending_input_ids, [21])
        self.assertEqual(output.unforwarded_committed_output_ids, [])
        self.assertEqual(output.committed_len, 3)
        self.assertEqual(output.logical_len, 4)
        self.assertEqual(output.output_tokens_seen, 2)

    def test_control_responses_report_actual_dp_rank(self):
        req = _Req("request", output_ids=[20])
        scheduler = _Scheduler([req], dp_rank=3)
        prepared = scheduler.prepare_decode_migration(
            _prepare(
                "request",
                "migration",
                17,
                target_sequence_length=8,
            )
        )

        self.assertTrue(prepared.success)
        self.assertEqual(prepared.source_dp_rank, 3)

        finalized = scheduler.finalize_decode_migration(
            FinalizeDecodeMigrationReqInput(
                rid="request", migration_id="migration", action="cancel"
            )
        )
        self.assertTrue(finalized.success)
        self.assertEqual(finalized.source_dp_rank, 3)

    def test_cancel_disarms_before_boundary(self):
        req = _Req("request", output_ids=[20])
        scheduler = _Scheduler([req])
        scheduler.prepare_decode_migration(
            _prepare("request", "migration", 17, target_sequence_length=8)
        )

        output = scheduler.finalize_decode_migration(
            FinalizeDecodeMigrationReqInput(
                rid="request", migration_id="migration", action="cancel"
            )
        )

        self.assertTrue(output.success)
        self.assertEqual(scheduler.decode_migrations.arms(), ())
        self.assertEqual(scheduler.running_batch.reqs, [req])

    def test_quiesce_poll_does_not_force_an_armed_request_to_park(self):
        req = _Req("request", output_ids=[20])
        scheduler = _Scheduler([req], overlap=True)
        scheduler.prepare_decode_migration(
            _prepare("request", "migration", 17, target_sequence_length=8)
        )

        output = scheduler.prepare_decode_migration(
            _prepare("request", "migration", 17)
        )

        self.assertTrue(output.success)
        self.assertEqual(output.status, "armed")
        self.assertEqual(scheduler.running_batch.reqs, [req])
        self.assertIsNotNone(scheduler.decode_migrations.get("migration"))

    def test_finished_request_cancels_armed_migration(self):
        req = _Req("request", output_ids=[20])
        scheduler = _Scheduler([req], overlap=True)
        scheduler.prepare_decode_migration(
            _prepare("request", "migration", 17, target_sequence_length=4)
        )
        req.output_ids.append(21)
        req.kv_committed_len += 1
        req._finished = True

        self.assertFalse(scheduler.maybe_park_decode_migration_at_boundary(req))
        self.assertEqual(scheduler.running_batch.reqs, [req])
        self.assertIsNone(scheduler.decode_migrations.get("migration"))

    def test_commit_requires_completed_transfer(self):
        req = _Req("request")
        scheduler = _Scheduler([req])
        with self._sender_patch():
            scheduler.prepare_decode_migration(_prepare("request", "migration", 17))
        output = scheduler.finalize_decode_migration(
            FinalizeDecodeMigrationReqInput(
                rid="request", migration_id="migration", action="commit"
            )
        )
        self.assertFalse(output.success)
        self.assertEqual(output.transfer_status, "bootstrapping")

    @patch("sglang.srt.disaggregation.decode_migration.release_kv_cache")
    def test_commit_releases_parked_source_and_invalidates_full_batch_cache(
        self, release_kv_cache
    ):
        req = _Req("request")
        scheduler = _Scheduler([req])
        with self._sender_patch():
            scheduler.prepare_decode_migration(_prepare("request", "migration", 17))
        # The request is parked but retains its request-pool slot until commit.
        # A prefill pass during transfer can therefore cache the empty batch as full.
        scheduler.running_batch.batch_is_full = True
        scheduler.decode_migrations.get("migration").state = (
            DecodeMigrationState.TRANSFERRED
        )

        output = scheduler.finalize_decode_migration(
            FinalizeDecodeMigrationReqInput(
                rid="request", migration_id="migration", action="commit"
            )
        )

        self.assertTrue(output.success)
        release_kv_cache.assert_called_once_with(
            req, scheduler.tree_cache, is_insert=False
        )
        self.assertFalse(scheduler.running_batch.batch_is_full)
        self.assertEqual(scheduler.decode_migrations.transfers(), ())

    @patch("sglang.srt.disaggregation.decode_migration.release_kv_cache")
    def test_abort_cleans_matching_parked_source(self, release_kv_cache):
        req = _Req("request")
        scheduler = _Scheduler([req])
        scheduler.disagg_decode_prealloc_queue = SimpleNamespace(queue=[])
        scheduler.disagg_decode_transfer_queue = SimpleNamespace(queue=[])
        with self._sender_patch():
            scheduler.prepare_decode_migration(_prepare("request", "migration", 17))

        scheduler.abort_decode_migration_receive(
            SimpleNamespace(rid="request", abort_all=False)
        )

        release_kv_cache.assert_called_once_with(
            req, scheduler.tree_cache, is_insert=False
        )
        self.assertEqual(scheduler.decode_migrations.transfers(), ())


class DecodeMigrationWaiterRoutingTests(unittest.IsolatedAsyncioTestCase):
    def manager(self):
        manager = TokenizerManager.__new__(TokenizerManager)
        manager.server_args = SimpleNamespace(dp_size=4)
        manager.decode_migration_futures = {}
        manager.auto_create_handle_loop = MagicMock()
        manager.send_to_scheduler = SimpleNamespace(send_pyobj=AsyncMock())
        return manager

    async def test_waiters_ignore_responses_from_other_dp_ranks(self):
        manager = self.manager()
        prepare = _prepare("request", "migration", 17)
        prepare.routed_dp_rank = 3
        prepare_task = asyncio.create_task(manager.prepare_decode_migration(prepare))
        await asyncio.sleep(0)

        manager._handle_decode_migration_output(
            PrepareDecodeMigrationReqOutput(
                rid="request",
                migration_id="migration",
                success=True,
                status="prepared",
                source_dp_rank=0,
            )
        )
        self.assertFalse(prepare_task.done())

        expected_prepare = PrepareDecodeMigrationReqOutput(
            rid="request",
            migration_id="migration",
            success=True,
            status="prepared",
            source_dp_rank=3,
        )
        manager._handle_decode_migration_output(expected_prepare)
        self.assertIs(await prepare_task, expected_prepare)

        finalize = FinalizeDecodeMigrationReqInput(
            rid="request",
            migration_id="migration",
            action="commit",
            routed_dp_rank=3,
        )
        finalize_task = asyncio.create_task(manager.finalize_decode_migration(finalize))
        await asyncio.sleep(0)

        manager._handle_decode_migration_output(
            FinalizeDecodeMigrationReqOutput(
                rid="request",
                migration_id="migration",
                action="commit",
                success=True,
                source_dp_rank=0,
            )
        )
        self.assertFalse(finalize_task.done())

        expected_finalize = FinalizeDecodeMigrationReqOutput(
            rid="request",
            migration_id="migration",
            action="commit",
            success=True,
            source_dp_rank=3,
        )
        manager._handle_decode_migration_output(expected_finalize)
        self.assertIs(await finalize_task, expected_finalize)
        self.assertEqual(manager.decode_migration_futures, {})


if __name__ == "__main__":
    unittest.main()
