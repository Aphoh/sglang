import asyncio
import unittest
from collections import deque
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import torch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.decode_migration import SchedulerDecodeMigrationMixin
from sglang.srt.disaggregation.decode_migration_state import (
    AwaitingStaleDecodeResult,
    DecodeMigrationRegistry,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import (
    CancelDecodeMigrationReqInput,
    CancelDecodeMigrationReqOutput,
    PrepareDecodeMigrationReqInput,
    PrepareDecodeMigrationReqOutput,
    QuiesceDecodeMigrationReqInput,
    QuiesceDecodeMigrationReqOutput,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.tokenizer_manager import TokenizerManager


class _Batch:
    def __init__(self, reqs):
        self.reqs = list(reqs)
        self.decoding_reqs = None
        self.batch_is_full = True
        self.filter_calls = 0

    def filter_batch(self, keep_indices=None, **_kwargs):
        self.filter_calls += 1
        keep_indices = keep_indices or []
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
        self.sampling_params = SimpleNamespace(max_new_tokens=128, min_new_tokens=0)
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
    def __init__(self):
        self.clear_count = 0
        self.aux_transfer_lens = None

    def clear(self):
        self.clear_count += 1

    def set_aux_transfer_lens(self, aux_transfer_lens):
        self.aux_transfer_lens = aux_transfer_lens


class _MetadataBuffers:
    def __init__(self):
        self.output_ids = torch.zeros((8, 64), dtype=torch.int64)
        self.cached_tokens = torch.zeros((8, 1), dtype=torch.int64)
        self.bootstrap_room = torch.zeros((8, 1), dtype=torch.int64)

    def set_decode_migration_frontier(
        self, index, *, committed_input_ids, pending_input_id, **_state
    ):
        self.output_ids[index].zero_()
        self.output_ids[index][0] = pending_input_id
        self.output_ids[index][1 : 1 + len(committed_input_ids)] = torch.tensor(
            committed_input_ids
        )

    def decode_migration_frontier_nbytes(self, committed_len):
        return (8 + committed_len) * self.output_ids.element_size()


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
        self.disagg_metadata_buffers = _MetadataBuffers()
        self.transfer_backend = object()
        self.spec_algorithm = SimpleNamespace(is_none=lambda: True)
        self.ps = SimpleNamespace(tp_rank=0, pp_rank=0, dp_rank=dp_rank)
        self.attn_cp_cpu_group = None
        self.attn_tp_cpu_group = None
        self.tree_cache = object()
        self.process_batch_result = MagicMock()
        self.output_streamer = MagicMock()
        self.ipc_channels = SimpleNamespace(
            send_to_tokenizer=SimpleNamespace(send_output=MagicMock())
        )
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


def _prepare(rid="request", migration_id="migration", room=17):
    return PrepareDecodeMigrationReqInput(
        rid=rid,
        migration_id=migration_id,
        bootstrap_host="127.0.0.1",
        bootstrap_port=5000,
        bootstrap_room=room,
    )


def _quiesce(rid="request", migration_id="migration", output_tokens_seen=0):
    return QuiesceDecodeMigrationReqInput(
        rid=rid,
        migration_id=migration_id,
        output_tokens_seen=output_tokens_seen,
    )


def _cancel(rid="request", migration_id="migration"):
    return CancelDecodeMigrationReqInput(rid=rid, migration_id=migration_id)


class DecodeMigrationSourceTests(unittest.TestCase):
    def test_scheduler_detaches_many_requests_in_one_batch_filter(self):
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
        removed = Scheduler._detach_requests_from_scheduling(
            scheduler, list(scheduler._deferred_detach_requests.values())
        )

        self.assertEqual(removed, {id(req) for req in reqs[:6]})
        self.assertEqual(running_batch.reqs, reqs[6:])
        self.assertEqual(running_batch.decoding_reqs, reqs[6:])
        self.assertEqual(running_batch.filter_calls, 1)
        self.assertFalse(running_batch.batch_is_full)

    def test_prepare_keeps_request_running_until_quiesce(self):
        req = _Req("request", output_ids=[20])
        scheduler = _Scheduler([req], overlap=True)

        prepared = scheduler.prepare_decode_migration(_prepare())
        self.assertTrue(prepared.success)
        self.assertEqual(scheduler.running_batch.reqs, [req])

        req.output_ids.append(21)
        req.kv_committed_len += 1
        quiesced = scheduler.quiesce_decode_migration(_quiesce(output_tokens_seen=2))

        self.assertTrue(quiesced.success)
        self.assertEqual(quiesced.status, "quiesced")
        self.assertEqual(scheduler.running_batch.reqs, [])
        self.assertFalse(scheduler.running_batch.batch_is_full)

    def test_quiesce_uses_frontend_acknowledged_frontier(self):
        req = _Req("request", output_ids=[20, 21])
        scheduler = _Scheduler([req], overlap=True)
        scheduler.prepare_decode_migration(_prepare())

        output = scheduler.quiesce_decode_migration(_quiesce(output_tokens_seen=1))

        self.assertTrue(output.success)
        self.assertEqual(output.committed_len, 2)
        self.assertEqual(output.logical_len, 3)
        self.assertEqual(output.output_tokens_seen, 1)
        self.assertEqual(scheduler.disagg_metadata_buffers.output_ids[0][0], 20)

    def test_overlap_quiesce_waits_for_completed_frontier(self):
        req = _Req("request", output_ids=[20])
        scheduler = _Scheduler([req], overlap=True)
        req.kv_committed_len += 1
        scheduler.result_queue.append((_Batch([req]), object()))
        scheduler.prepare_decode_migration(_prepare())

        request = _quiesce()
        request.http_worker_ipc = "tokenizer-3"
        output = scheduler.quiesce_decode_migration(request)
        self.assertEqual(output.status, "quiescing")

        req.output_ids.append(21)
        self.assertTrue(scheduler.maybe_quiesce_decode_migration(req))
        self.assertEqual(scheduler.running_batch.reqs, [])
        completion = (
            scheduler.ipc_channels.send_to_tokenizer.send_output.call_args.args[0]
        )
        self.assertEqual(completion.status, "quiesced")
        routed_request = (
            scheduler.ipc_channels.send_to_tokenizer.send_output.call_args.args[1]
        )
        self.assertIs(routed_request, request)
        self.assertEqual(routed_request.http_worker_ipc, "tokenizer-3")

    def test_overlap_finish_notifies_pending_quiesce(self):
        req = _Req("request", output_ids=[20])
        scheduler = _Scheduler([req], overlap=True)
        scheduler.result_queue.append((_Batch([req]), object()))
        scheduler.prepare_decode_migration(_prepare())
        self.assertEqual(
            scheduler.quiesce_decode_migration(_quiesce()).status, "quiescing"
        )

        req._finished = True
        self.assertFalse(scheduler.maybe_quiesce_decode_migration(req))

        completion = (
            scheduler.ipc_channels.send_to_tokenizer.send_output.call_args.args[0]
        )
        self.assertEqual(completion.status, "finished")
        self.assertIsNone(scheduler.decode_migrations.get("migration"))

    @patch("sglang.srt.disaggregation.decode_migration.release_kv_cache")
    @patch(
        "sglang.srt.disaggregation.decode_migration.poll_and_all_reduce_attn_cp_tp_group"
    )
    def test_transfer_success_releases_source(self, poll_transfers, release_kv_cache):
        req = _Req("request")
        scheduler = _Scheduler([req])
        scheduler.prepare_decode_migration(_prepare())
        scheduler.quiesce_decode_migration(_quiesce())
        poll_transfers.return_value = [KVPoll.Success]

        scheduler.process_decode_migration_transfers()

        self.assertIsNone(scheduler.decode_migrations.get("migration"))
        release_kv_cache.assert_called_once_with(
            req, scheduler.tree_cache, is_insert=False
        )

    @patch("sglang.srt.disaggregation.decode_migration.release_kv_cache")
    @patch(
        "sglang.srt.disaggregation.decode_migration.poll_and_all_reduce_attn_cp_tp_group"
    )
    def test_overlap_discards_one_stale_result_before_release(
        self, poll_transfers, release_kv_cache
    ):
        req = _Req("request", output_ids=[20])
        scheduler = _Scheduler([req], overlap=True)
        scheduler.prepare_decode_migration(_prepare())
        req.output_ids.append(21)
        req.kv_committed_len += 1
        scheduler.result_queue.append((_Batch([req]), object()))
        scheduler.quiesce_decode_migration(_quiesce(output_tokens_seen=2))
        self.assertTrue(scheduler.maybe_quiesce_decode_migration(req))
        self.assertEqual(scheduler.decode_migrations.kv_owner_reqs(), (req,))
        poll_transfers.return_value = [KVPoll.Success]

        scheduler.process_decode_migration_transfers()

        record = scheduler.decode_migrations.get("migration")
        self.assertIsInstance(record, AwaitingStaleDecodeResult)
        self.assertEqual(scheduler.decode_migrations.kv_owner_reqs(), (req,))
        release_kv_cache.assert_not_called()
        self.assertFalse(scheduler.should_process_decode_result(req))
        self.assertIsNone(scheduler.decode_migrations.get("migration"))
        self.assertEqual(scheduler.decode_migrations.kv_owner_reqs(), ())
        release_kv_cache.assert_called_once_with(
            req, scheduler.tree_cache, is_insert=False
        )
        self.assertTrue(scheduler.should_process_decode_result(req))

    @patch("sglang.srt.disaggregation.decode_migration.release_kv_cache")
    def test_failed_transfer_releases_source(self, release_kv_cache):
        req = _Req("request")
        scheduler = _Scheduler([req])
        scheduler.prepare_decode_migration(_prepare())
        scheduler.quiesce_decode_migration(_quiesce())

        scheduler._fail_decode_migration(
            scheduler.decode_migrations.get("migration"), "test failure"
        )

        self.assertFalse(scheduler.decode_migrations.has_active_transfers())
        release_kv_cache.assert_called_once_with(
            req, scheduler.tree_cache, is_insert=False
        )

    def test_cancel_only_discards_prepared_source(self):
        req = _Req("request")
        scheduler = _Scheduler([req], dp_rank=3)
        prepared = scheduler.prepare_decode_migration(_prepare())

        cancelled = scheduler.cancel_decode_migration(_cancel())

        self.assertEqual(prepared.source_dp_rank, 3)
        self.assertTrue(cancelled.success)
        self.assertEqual(cancelled.status, "cancelled")
        self.assertEqual(cancelled.source_dp_rank, 3)
        self.assertEqual(scheduler.running_batch.reqs, [req])
        self.assertEqual(scheduler.created_senders[0].clear_count, 1)

    def test_cancel_completes_pending_quiesce(self):
        req = _Req("request", output_ids=[20])
        scheduler = _Scheduler([req], overlap=True)
        scheduler.result_queue.append((_Batch([req]), object()))
        scheduler.prepare_decode_migration(_prepare())
        self.assertEqual(
            scheduler.quiesce_decode_migration(_quiesce()).status, "quiescing"
        )

        cancelled = scheduler.cancel_decode_migration(_cancel())

        self.assertEqual(cancelled.status, "cancelled")
        completion = (
            scheduler.ipc_channels.send_to_tokenizer.send_output.call_args.args[0]
        )
        self.assertEqual(completion.status, "finished")
        self.assertEqual(
            scheduler.ipc_channels.send_to_tokenizer.send_output.call_count, 1
        )

    def test_abort_completes_pending_quiesce(self):
        req = _Req("request", output_ids=[20])
        scheduler = _Scheduler([req], overlap=True)
        scheduler.disagg_decode_prealloc_queue = SimpleNamespace(queue=[])
        scheduler.disagg_decode_transfer_queue = SimpleNamespace(queue=[])
        scheduler.result_queue.append((_Batch([req]), object()))
        scheduler.prepare_decode_migration(_prepare())
        scheduler.quiesce_decode_migration(_quiesce())

        scheduler.abort_decode_migration_receive(
            SimpleNamespace(rid="request", abort_all=False)
        )

        completion = (
            scheduler.ipc_channels.send_to_tokenizer.send_output.call_args.args[0]
        )
        self.assertEqual(completion.status, "finished")

    def test_cancel_rejects_quiesced_source(self):
        req = _Req("request")
        scheduler = _Scheduler([req])
        scheduler.prepare_decode_migration(_prepare())
        scheduler.quiesce_decode_migration(_quiesce())

        output = scheduler.cancel_decode_migration(_cancel())

        self.assertFalse(output.success)
        self.assertEqual(output.status, "quiesced")
        self.assertIsNotNone(scheduler.decode_migrations.get("migration"))

    def test_concurrent_requests_are_parked_independently(self):
        first = _Req("first")
        second = _Req("second")
        scheduler = _Scheduler([first, second])
        scheduler.prepare_decode_migration(_prepare("first", "one", 17))
        scheduler.prepare_decode_migration(_prepare("second", "two", 18))
        scheduler.quiesce_decode_migration(_quiesce("first", "one"))
        scheduler.quiesce_decode_migration(_quiesce("second", "two"))

        self.assertEqual(scheduler.running_batch.reqs, [])
        self.assertEqual(
            {record.migration_id for record in scheduler.decode_migrations.transfers()},
            {"one", "two"},
        )

    def test_finished_request_discards_preparation(self):
        req = _Req("request", output_ids=[20])
        scheduler = _Scheduler([req], overlap=True)
        scheduler.prepare_decode_migration(_prepare())
        req._finished = True

        self.assertFalse(scheduler.maybe_quiesce_decode_migration(req))
        self.assertIsNone(scheduler.decode_migrations.get("migration"))
        self.assertEqual(scheduler.running_batch.reqs, [req])

    @patch("sglang.srt.disaggregation.decode_migration.release_kv_cache")
    def test_abort_cleans_matching_parked_source(self, release_kv_cache):
        req = _Req("request")
        scheduler = _Scheduler([req])
        scheduler.disagg_decode_prealloc_queue = SimpleNamespace(queue=[])
        scheduler.disagg_decode_transfer_queue = SimpleNamespace(queue=[])
        scheduler.prepare_decode_migration(_prepare())
        scheduler.quiesce_decode_migration(_quiesce())

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

    async def _assert_rank_routing(self, request, send, wrong, expected):
        task = asyncio.create_task(send(request))
        await asyncio.sleep(0)
        self.manager_under_test._handle_decode_migration_output(wrong)
        self.assertFalse(task.done())
        self.manager_under_test._handle_decode_migration_output(expected)
        self.assertIs(await task, expected)

    async def test_controls_ignore_responses_from_other_dp_ranks(self):
        manager = self.manager_under_test = self.manager()
        prepare = _prepare()
        prepare.routed_dp_rank = 3
        await self._assert_rank_routing(
            prepare,
            manager.prepare_decode_migration,
            PrepareDecodeMigrationReqOutput(
                rid="request",
                migration_id="migration",
                success=True,
                status="ready",
                source_dp_rank=0,
            ),
            PrepareDecodeMigrationReqOutput(
                rid="request",
                migration_id="migration",
                success=True,
                status="ready",
                source_dp_rank=3,
            ),
        )

        quiesce = _quiesce(output_tokens_seen=2)
        quiesce.routed_dp_rank = 3
        await self._assert_rank_routing(
            quiesce,
            manager.quiesce_decode_migration,
            QuiesceDecodeMigrationReqOutput(
                rid="request",
                migration_id="migration",
                success=True,
                status="quiesced",
                source_dp_rank=0,
            ),
            QuiesceDecodeMigrationReqOutput(
                rid="request",
                migration_id="migration",
                success=True,
                status="quiesced",
                source_dp_rank=3,
            ),
        )

        cancel = _cancel()
        cancel.routed_dp_rank = 3
        await self._assert_rank_routing(
            cancel,
            manager.cancel_decode_migration,
            CancelDecodeMigrationReqOutput(
                rid="request",
                migration_id="migration",
                success=True,
                status="cancelled",
                source_dp_rank=0,
            ),
            CancelDecodeMigrationReqOutput(
                rid="request",
                migration_id="migration",
                success=True,
                status="cancelled",
                source_dp_rank=3,
            ),
        )
        self.assertEqual(manager.decode_migration_futures, {})

    async def test_quiescing_waits_for_event_driven_completion(self):
        manager = self.manager_under_test = self.manager()
        request = _quiesce(output_tokens_seen=2)
        request.routed_dp_rank = 3
        task = asyncio.create_task(manager.quiesce_decode_migration(request))
        await asyncio.sleep(0)

        manager._handle_decode_migration_output(
            QuiesceDecodeMigrationReqOutput(
                rid="request",
                migration_id="migration",
                success=True,
                status="quiescing",
                source_dp_rank=3,
            )
        )
        self.assertFalse(task.done())

        completed = QuiesceDecodeMigrationReqOutput(
            rid="request",
            migration_id="migration",
            success=True,
            status="quiesced",
            source_dp_rank=3,
        )
        manager._handle_decode_migration_output(completed)
        self.assertIs(await task, completed)

    async def test_cancel_can_overlap_quiesce_for_same_migration(self):
        manager = self.manager_under_test = self.manager()
        quiesce = _quiesce()
        quiesce.routed_dp_rank = 3
        cancel = _cancel()
        cancel.routed_dp_rank = 3
        quiesce_task = asyncio.create_task(manager.quiesce_decode_migration(quiesce))
        cancel_task = asyncio.create_task(manager.cancel_decode_migration(cancel))
        await asyncio.sleep(0)

        cancelled = CancelDecodeMigrationReqOutput(
            rid="request",
            migration_id="migration",
            success=True,
            status="cancelled",
            source_dp_rank=3,
        )
        quiesced = QuiesceDecodeMigrationReqOutput(
            rid="request",
            migration_id="migration",
            success=False,
            status="finished",
            source_dp_rank=3,
        )
        manager._handle_decode_migration_output(cancelled)
        manager._handle_decode_migration_output(quiesced)

        self.assertIs(await cancel_task, cancelled)
        self.assertIs(await quiesce_task, quiesced)


if __name__ == "__main__":
    unittest.main()
