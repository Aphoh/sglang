import asyncio
import unittest
from collections import deque
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import torch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.decode_migration import (
    DecodeMigrationRegistry,
    DecodeMigrationState,
    SchedulerDecodeMigrationMixin,
)
from sglang.srt.disaggregation.utils import DisaggregationMode, KVClassType
from sglang.srt.managers.io_struct import (
    BindDecodeMigrationReqInput,
    BindDecodeMigrationReqOutput,
    FinalizeDecodeMigrationReqInput,
    FinalizeDecodeMigrationReqOutput,
    PrepareDecodeMigrationReqInput,
    PrepareDecodeMigrationReqOutput,
    QuiesceDecodeMigrationReqInput,
    QuiesceDecodeMigrationReqOutput,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.result_disposition import (
    ResultDisposition,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager


class _Batch:
    def __init__(self, reqs):
        self.reqs = list(reqs)
        self.decoding_reqs = None
        self.batch_is_full = True
        self.filter_calls = 0

    def filter_batch(self, keep_indices=None, **_kwargs):
        self.filter_calls += 1
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
        self.send_token_offset = 0
        self.return_logprob = False
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
        self.attn_cp_cpu_group = None
        self.attn_tp_cpu_group = None
        self.tree_cache = object()
        self.process_batch_result = MagicMock()
        self.output_streamer = MagicMock()
        self.created_senders = []

    def _get_decode_migration_kv_manager(self):
        return object()

    def _create_decode_migration_sender(self, _recv_req):
        sender = _Sender()
        self.created_senders.append(sender)
        return sender

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
):
    return PrepareDecodeMigrationReqInput(
        rid=rid,
        migration_id=migration_id,
        bootstrap_host="127.0.0.1",
        bootstrap_port=5000,
        bootstrap_room=room,
    )


def _quiesce(rid, migration_id, output_tokens_seen=0):
    return QuiesceDecodeMigrationReqInput(
        rid=rid,
        migration_id=migration_id,
        output_tokens_seen=output_tokens_seen,
    )


class DecodeMigrationSourceTests(unittest.TestCase):
    def _sender_patch(self):
        return patch(
            "sglang.srt.disaggregation.decode_migration.get_kv_class",
            side_effect=lambda _backend, kind: (
                _Sender if kind == KVClassType.SENDER else object
            ),
        )

    def test_scheduler_batches_deferred_request_detaches(self):
        reqs = [_Req(str(index)) for index in range(8)]
        running_batch = _Batch(reqs)
        running_batch.decoding_reqs = list(reqs)
        scheduler = SimpleNamespace(
            running_batch=running_batch,
            last_batch=running_batch,
            cur_batch=running_batch,
            _deferred_detach_requests={},
        )

        for req in reqs[:6]:
            self.assertTrue(Scheduler.detach_request_from_scheduling(scheduler, req))
        self.assertEqual(running_batch.reqs, reqs)

        removed = Scheduler._detach_requests_from_scheduling(
            scheduler, list(scheduler._deferred_detach_requests.values())
        )

        self.assertEqual(removed, {id(req) for req in reqs[:6]})
        self.assertEqual(running_batch.reqs, reqs[6:])
        self.assertEqual(running_batch.decoding_reqs, reqs[6:])
        self.assertEqual(running_batch.filter_calls, 1)
        self.assertFalse(running_batch.batch_is_full)

    def test_quiesce_uses_the_frontend_acknowledged_frontier(self):
        target = _Req("target", output_ids=[20, 21])
        other = _Req("other")
        scheduler = _Scheduler([target, other], overlap=True)
        with self._sender_patch():
            prepared = scheduler.prepare_decode_migration(
                _prepare("target", "migration-1", 17)
            )
            output = scheduler.quiesce_decode_migration(
                _quiesce("target", "migration-1", output_tokens_seen=1)
            )

        self.assertTrue(prepared.success)
        self.assertTrue(output.success)
        # Token 21 was decoded ahead of the frontend's stream watermark. The
        # destination resumes from the last emitted token (20) and regenerates
        # 21 instead of silently suppressing an unseen token.
        self.assertEqual(output.pending_input_ids, [20])
        self.assertEqual(output.logical_len, 3)
        self.assertEqual(output.output_tokens_seen, 1)
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

        scheduler.prepare_decode_migration(_prepare("request", "migration", 17))
        output = scheduler.quiesce_decode_migration(_quiesce("request", "migration"))

        self.assertTrue(output.success)
        self.assertEqual(output.status, "quiescing")
        prepared = scheduler.decode_migrations.get("migration")
        self.assertTrue(prepared.quiesce_requested)

        req.output_ids.append(21)
        with self._sender_patch():
            self.assertTrue(scheduler.maybe_quiesce_decode_migration(req))

    def test_unrelated_overlap_result_does_not_defer_prepare(self):
        req = _Req("request", output_ids=[20])
        other = _Req("other")
        scheduler = _Scheduler([req], overlap=True)
        scheduler.result_queue.append((_Batch([other]), object()))

        with self._sender_patch():
            scheduler.prepare_decode_migration(_prepare("request", "migration", 17))
            output = scheduler.quiesce_decode_migration(
                _quiesce("request", "migration")
            )

        self.assertTrue(output.success)
        self.assertEqual(output.status, "quiesced")

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
        scheduler.prepare_decode_migration(_prepare("request", "migration", 17))
        req.output_ids.append(21)
        req.kv_committed_len += 1
        scheduler.result_queue.append((_Batch([req]), object()))
        scheduler.quiesce_decode_migration(_quiesce("request", "migration", 2))
        with self._sender_patch():
            self.assertTrue(scheduler.maybe_quiesce_decode_migration(req))

        output = scheduler.finalize_decode_migration(
            FinalizeDecodeMigrationReqInput(
                rid="request", migration_id="migration", action="cancel"
            )
        )

        self.assertTrue(output.success)
        record = scheduler.decode_migrations.get("migration")
        self.assertEqual(record.state, DecodeMigrationState.AWAITING_STALE_RESULT)
        release_kv_cache.assert_not_called()
        retry = scheduler.finalize_decode_migration(
            FinalizeDecodeMigrationReqInput(
                rid="request", migration_id="migration", action="cancel"
            )
        )
        self.assertTrue(retry.success)
        self.assertEqual(retry.transfer_status, "bootstrapping")
        self.assertEqual(
            scheduler.get_decode_migration_result_disposition(req),
            ResultDisposition.DISCARD,
        )
        self.assertIsNone(scheduler.decode_migrations.get("migration"))
        release_kv_cache.assert_called_once_with(
            req, scheduler.tree_cache, is_insert=False
        )

    @patch("sglang.srt.disaggregation.decode_migration.release_kv_cache")
    def test_commit_retry_preserves_transferred_status_while_tombstoned(
        self, release_kv_cache
    ):
        req = _Req("request", output_ids=[20])
        scheduler = _Scheduler([req], overlap=True)
        scheduler.prepare_decode_migration(_prepare("request", "migration", 17))
        req.output_ids.append(21)
        req.kv_committed_len += 1
        scheduler.result_queue.append((_Batch([req]), object()))
        scheduler.quiesce_decode_migration(_quiesce("request", "migration", 2))
        with self._sender_patch():
            self.assertTrue(scheduler.maybe_quiesce_decode_migration(req))
        scheduler.decode_migrations.get("migration").state = (
            DecodeMigrationState.TRANSFERRED
        )

        first = scheduler.finalize_decode_migration(
            FinalizeDecodeMigrationReqInput(
                rid="request", migration_id="migration", action="commit"
            )
        )
        retry = scheduler.finalize_decode_migration(
            FinalizeDecodeMigrationReqInput(
                rid="request", migration_id="migration", action="commit"
            )
        )

        self.assertTrue(first.success)
        self.assertTrue(retry.success)
        self.assertEqual(retry.transfer_status, "transferred")
        self.assertEqual(
            scheduler.get_decode_migration_result_disposition(req),
            ResultDisposition.DISCARD,
        )
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
        scheduler.prepare_decode_migration(_prepare("request", "migration", 17))
        req.output_ids.append(21)
        req.kv_committed_len += 1
        scheduler.result_queue.append((_Batch([other]), object()))
        with self._sender_patch():
            output = scheduler.quiesce_decode_migration(
                _quiesce("request", "migration", 2)
            )
        self.assertEqual(output.status, "quiesced")

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
            scheduler.quiesce_decode_migration(_quiesce("request", "migration"))
            scheduler.quiesce_decode_migration(_quiesce("request", "migration"))

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
            scheduler.quiesce_decode_migration(_quiesce("first", "one"))
            scheduler.quiesce_decode_migration(_quiesce("second", "two"))

        self.assertTrue(one.success)
        self.assertTrue(two.success)
        self.assertEqual(scheduler.running_batch.reqs, [])
        self.assertEqual(
            {record.migration_id for record in scheduler.decode_migrations.transfers()},
            {"one", "two"},
        )

    def test_prepare_keeps_request_running_until_explicit_quiesce(self):
        req = _Req("request", output_ids=[20])
        scheduler = _Scheduler([req], overlap=True)
        output = scheduler.prepare_decode_migration(
            _prepare("request", "migration", 17)
        )

        self.assertTrue(output.success)
        self.assertEqual(output.status, "ready")
        self.assertEqual(scheduler.running_batch.reqs, [req])

        req.output_ids.append(21)
        req.kv_committed_len += 1
        with self._sender_patch():
            quiesced = scheduler.quiesce_decode_migration(
                _quiesce("request", "migration", output_tokens_seen=2)
            )

        self.assertTrue(quiesced.success)
        self.assertEqual(quiesced.status, "quiesced")
        self.assertEqual(len(req.origin_input_ids) + len(req.output_ids), 4)
        self.assertEqual(scheduler.running_batch.reqs, [])
        self.assertIsNotNone(scheduler.decode_migrations.get_for_rid(req.rid))
        self.assertEqual(scheduler.decode_migrations.get("migration").committed_len, 3)

    def test_prepare_can_precede_request_admission(self):
        scheduler = _Scheduler([])

        prepared = scheduler.prepare_decode_migration(
            _prepare("request", "migration", 17)
        )

        self.assertTrue(prepared.success)
        self.assertEqual(prepared.status, "ready")
        self.assertEqual(len(scheduler.created_senders), 1)

        req = _Req("request", output_ids=[20])
        scheduler.running_batch.reqs.append(req)
        with self._sender_patch():
            quiesced = scheduler.quiesce_decode_migration(
                _quiesce("request", "migration", output_tokens_seen=1)
            )

        self.assertTrue(quiesced.success)
        self.assertEqual(quiesced.status, "quiesced")
        self.assertEqual(scheduler.running_batch.reqs, [])

    def test_quiesce_does_not_override_normal_stream_interval(self):
        req = _Req("request", output_ids=[20, 21, 22, 23, 24, 25])
        req.send_token_offset = 5
        scheduler = _Scheduler([req], overlap=True)
        scheduler.prepare_decode_migration(_prepare("request", "migration", 17))

        with self._sender_patch():
            output = scheduler.quiesce_decode_migration(
                _quiesce("request", "migration", output_tokens_seen=6)
            )

        self.assertTrue(output.success)
        scheduler.output_streamer.stream_output.assert_not_called()

    def test_quiesce_exports_acknowledged_frontier_after_overlap_overshoot(self):
        req = _Req("request", output_ids=[20])
        scheduler = _Scheduler([req], overlap=True)
        scheduler.prepare_decode_migration(_prepare("request", "migration", 17))

        req.output_ids.extend([21, 22, 23])
        req.kv_committed_len += 3
        with self._sender_patch():
            output = scheduler.quiesce_decode_migration(
                _quiesce("request", "migration", output_tokens_seen=2)
            )

        self.assertTrue(output.success)
        record = scheduler.decode_migrations.get("migration")
        self.assertEqual(record.committed_len, 3)
        self.assertEqual(record.logical_len, 4)
        self.assertEqual(record.pending_input_ids, [21])
        output = scheduler.quiesce_decode_migration(
            _quiesce("request", "migration", output_tokens_seen=4)
        )
        self.assertEqual(output.committed_input_ids, [10, 11, 20])
        self.assertEqual(output.pending_input_ids, [21])
        self.assertEqual(output.unforwarded_committed_output_ids, [])
        self.assertEqual(output.committed_len, 3)
        self.assertEqual(output.logical_len, 4)
        self.assertEqual(output.output_tokens_seen, 2)
        scheduler.output_streamer.stream_output.assert_not_called()

    @patch("sglang.srt.disaggregation.decode_migration.release_kv_cache")
    def test_failed_overlap_transfer_discards_stale_result_without_tombstone(
        self, release_kv_cache
    ):
        req = _Req("request", output_ids=[20])
        scheduler = _Scheduler([req], overlap=True)
        scheduler.prepare_decode_migration(_prepare("request", "migration", 17))
        req.output_ids.append(21)
        req.kv_committed_len += 1
        scheduler.result_queue.append((_Batch([req]), object()))
        scheduler.quiesce_decode_migration(
            _quiesce("request", "migration", output_tokens_seen=2)
        )
        with self._sender_patch():
            self.assertTrue(scheduler.maybe_quiesce_decode_migration(req))

        scheduler._fail_decode_migration(
            scheduler.decode_migrations.get("migration"), "test failure"
        )
        self.assertEqual(
            scheduler.get_decode_migration_result_disposition(req),
            ResultDisposition.DISCARD,
        )
        self.assertIsNone(scheduler.decode_migrations.get("migration"))
        release_kv_cache.assert_called_once_with(
            req, scheduler.tree_cache, is_insert=False
        )

    def test_control_responses_report_actual_dp_rank(self):
        req = _Req("request", output_ids=[20])
        scheduler = _Scheduler([req], dp_rank=3)
        prepared = scheduler.prepare_decode_migration(
            _prepare("request", "migration", 17)
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

    def test_cancel_discards_prepared_source(self):
        req = _Req("request", output_ids=[20])
        scheduler = _Scheduler([req])
        scheduler.prepare_decode_migration(_prepare("request", "migration", 17))

        output = scheduler.finalize_decode_migration(
            FinalizeDecodeMigrationReqInput(
                rid="request", migration_id="migration", action="cancel"
            )
        )

        self.assertTrue(output.success)
        self.assertEqual(scheduler.decode_migrations.prepared_sources(), ())
        self.assertEqual(scheduler.running_batch.reqs, [req])

    def test_repeated_prepare_does_not_quiesce_request(self):
        req = _Req("request", output_ids=[20])
        scheduler = _Scheduler([req], overlap=True)
        scheduler.prepare_decode_migration(_prepare("request", "migration", 17))

        output = scheduler.prepare_decode_migration(
            _prepare("request", "migration", 17)
        )

        self.assertTrue(output.success)
        self.assertEqual(output.status, "ready")
        self.assertEqual(scheduler.running_batch.reqs, [req])
        self.assertIsNotNone(scheduler.decode_migrations.get("migration"))

    def test_finished_request_discards_prepared_migration(self):
        req = _Req("request", output_ids=[20])
        scheduler = _Scheduler([req], overlap=True)
        scheduler.prepare_decode_migration(_prepare("request", "migration", 17))
        req.output_ids.append(21)
        req.kv_committed_len += 1
        req._finished = True

        self.assertFalse(scheduler.maybe_quiesce_decode_migration(req))
        self.assertEqual(scheduler.running_batch.reqs, [req])
        self.assertIsNone(scheduler.decode_migrations.get("migration"))

    def test_commit_is_accepted_before_transfer_completion(self):
        req = _Req("request")
        scheduler = _Scheduler([req])
        with self._sender_patch():
            scheduler.prepare_decode_migration(_prepare("request", "migration", 17))
            scheduler.quiesce_decode_migration(_quiesce("request", "migration"))
        output = scheduler.finalize_decode_migration(
            FinalizeDecodeMigrationReqInput(
                rid="request", migration_id="migration", action="commit"
            )
        )
        self.assertTrue(output.success)
        self.assertTrue(output.commit_pending)
        self.assertEqual(output.transfer_status, "bootstrapping")
        self.assertTrue(scheduler.decode_migrations.get("migration").commit_requested)

    @patch("sglang.srt.disaggregation.decode_migration.release_kv_cache")
    @patch(
        "sglang.srt.disaggregation.decode_migration.poll_and_all_reduce_attn_cp_tp_group"
    )
    def test_pending_commit_releases_after_transfer_and_is_idempotent(
        self, poll_transfers, release_kv_cache
    ):
        req = _Req("request")
        scheduler = _Scheduler([req])
        with self._sender_patch():
            scheduler.prepare_decode_migration(_prepare("request", "migration", 17))
            scheduler.quiesce_decode_migration(_quiesce("request", "migration"))
        accepted = scheduler.finalize_decode_migration(
            FinalizeDecodeMigrationReqInput(
                rid="request", migration_id="migration", action="commit"
            )
        )
        self.assertTrue(accepted.success)
        self.assertTrue(accepted.commit_pending)

        poll_transfers.return_value = [KVPoll.Success]
        scheduler.process_decode_migration_transfers()

        release_kv_cache.assert_called_once_with(
            req, scheduler.tree_cache, is_insert=False
        )
        self.assertIsNone(scheduler.decode_migrations.get("migration"))
        retry = scheduler.finalize_decode_migration(
            FinalizeDecodeMigrationReqInput(
                rid="request", migration_id="migration", action="commit"
            )
        )
        self.assertTrue(retry.success)
        self.assertFalse(retry.commit_pending)
        self.assertEqual(retry.transfer_status, "transferred")

    @patch("sglang.srt.disaggregation.decode_migration.release_kv_cache")
    def test_commit_releases_parked_source_and_invalidates_full_batch_cache(
        self, release_kv_cache
    ):
        req = _Req("request")
        scheduler = _Scheduler([req])
        with self._sender_patch():
            scheduler.prepare_decode_migration(_prepare("request", "migration", 17))
            scheduler.quiesce_decode_migration(_quiesce("request", "migration"))
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
            scheduler.quiesce_decode_migration(_quiesce("request", "migration"))

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
                status="ready",
                source_dp_rank=0,
            )
        )
        self.assertFalse(prepare_task.done())

        expected_prepare = PrepareDecodeMigrationReqOutput(
            rid="request",
            migration_id="migration",
            success=True,
            status="ready",
            source_dp_rank=3,
        )
        manager._handle_decode_migration_output(expected_prepare)
        self.assertIs(await prepare_task, expected_prepare)

        quiesce = _quiesce("request", "migration", output_tokens_seen=2)
        quiesce.routed_dp_rank = 3
        quiesce_task = asyncio.create_task(manager.quiesce_decode_migration(quiesce))
        await asyncio.sleep(0)

        manager._handle_decode_migration_output(
            QuiesceDecodeMigrationReqOutput(
                rid="request",
                migration_id="migration",
                success=True,
                status="quiesced",
                source_dp_rank=0,
            )
        )
        self.assertFalse(quiesce_task.done())

        expected_quiesce = QuiesceDecodeMigrationReqOutput(
            rid="request",
            migration_id="migration",
            success=True,
            status="quiesced",
            source_dp_rank=3,
        )
        manager._handle_decode_migration_output(expected_quiesce)
        self.assertIs(await quiesce_task, expected_quiesce)

        bind = BindDecodeMigrationReqInput(
            rid="request",
            migration_id="migration",
            bootstrap_room=17,
            committed_input_ids=[1, 2, 3],
            pending_input_ids=[4],
            committed_len=3,
            logical_len=4,
            routed_dp_rank=3,
        )
        bind_task = asyncio.create_task(manager.bind_decode_migration_destination(bind))
        await asyncio.sleep(0)

        manager._handle_decode_migration_output(
            BindDecodeMigrationReqOutput(
                rid="request",
                migration_id="migration",
                success=True,
                status="ready",
                source_dp_rank=0,
            )
        )
        self.assertFalse(bind_task.done())

        expected_bind = BindDecodeMigrationReqOutput(
            rid="request",
            migration_id="migration",
            success=True,
            status="ready",
            source_dp_rank=3,
        )
        manager._handle_decode_migration_output(expected_bind)
        self.assertIs(await bind_task, expected_bind)

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

    async def test_result_dispatcher_routes_bind_output(self):
        manager = self.manager()
        manager.init_communicators = MagicMock()
        manager.init_request_dispatcher()
        future = asyncio.get_running_loop().create_future()
        manager.decode_migration_futures[("migration", 3)] = future
        expected = BindDecodeMigrationReqOutput(
            rid="request",
            migration_id="migration",
            success=True,
            status="ready",
            source_dp_rank=3,
        )

        manager._result_dispatcher(expected)

        self.assertIs(await future, expected)


if __name__ == "__main__":
    unittest.main()
