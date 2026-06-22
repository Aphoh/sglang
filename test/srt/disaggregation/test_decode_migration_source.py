import unittest
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.disaggregation.decode_migration import SchedulerDecodeMigrationMixin
from sglang.srt.disaggregation.utils import DisaggregationMode, KVClassType
from sglang.srt.managers.io_struct import (
    FinalizeDecodeMigrationReqInput,
    PrepareDecodeMigrationReqInput,
)


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
    def __init__(self, reqs, *, overlap=False):
        self.server_args = SimpleNamespace(enable_decode_migration=True)
        self.disaggregation_mode = DisaggregationMode.NULL
        self.enable_overlap = overlap
        self.result_queue = deque()
        self.decode_migration_transfers = {}
        self.decode_migration_by_rid = {}
        self.decode_migration_arms = {}
        self.decode_migration_arm_by_rid = {}
        self._decode_migration_overlap_result_processed = False
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
        self.ps = SimpleNamespace(tp_rank=0, pp_rank=0, dp_rank=0)
        self.tree_cache = object()
        self.process_batch_result = MagicMock()

    def _get_decode_migration_kv_manager(self):
        return object()


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

    def test_overlap_resolves_once_and_parks_only_target(self):
        target = _Req("target", output_ids=[20])
        other = _Req("other")
        scheduler = _Scheduler([target, other], overlap=True)
        pending_batch = _Batch([target, other])
        scheduler.result_queue.append((pending_batch, object()))

        def process_result(_batch, _result):
            target.output_ids.append(21)
            target.kv_committed_len += 1

        scheduler.process_batch_result.side_effect = process_result
        with self._sender_patch():
            output = scheduler.prepare_decode_migration(
                _prepare("target", "migration-1", 17, output_tokens_seen=1)
            )

        self.assertTrue(output.success)
        self.assertEqual(output.pending_input_ids, [21])
        self.assertEqual(scheduler.running_batch.reqs, [other])
        self.assertEqual(len(scheduler.result_queue), 0)
        self.assertTrue(scheduler._decode_migration_overlap_result_processed)
        scheduler.process_batch_result.assert_called_once()

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
        self.assertEqual(set(scheduler.decode_migration_transfers), {"one", "two"})

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
        self.assertTrue(req.is_decode_migration_source_parked)
        self.assertEqual(
            scheduler.decode_migration_transfers["migration"].committed_len, 3
        )

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
        record = scheduler.decode_migration_transfers["migration"]
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
        self.assertEqual(scheduler.decode_migration_arms, {})
        self.assertEqual(scheduler.decode_migration_arm_by_rid, {})
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
        self.assertIn("migration", scheduler.decode_migration_arms)

    def test_finish_race_does_not_park_request(self):
        req = _Req("request", output_ids=[20])
        scheduler = _Scheduler([req], overlap=True)
        scheduler.result_queue.append((_Batch([req]), object()))

        def process_result(_batch, _result):
            req.output_ids.append(21)
            req.kv_committed_len += 1
            req._finished = True

        scheduler.process_batch_result.side_effect = process_result
        output = scheduler.prepare_decode_migration(
            _prepare("request", "migration", 17)
        )

        self.assertFalse(output.success)
        self.assertEqual(output.status, "finished")
        self.assertEqual(scheduler.running_batch.reqs, [req])
        self.assertEqual(scheduler.decode_migration_transfers, {})

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
    def test_commit_releases_parked_source_once(self, release_kv_cache):
        req = _Req("request")
        scheduler = _Scheduler([req])
        with self._sender_patch():
            scheduler.prepare_decode_migration(_prepare("request", "migration", 17))
        scheduler.decode_migration_transfers["migration"].status = "transferred"

        output = scheduler.finalize_decode_migration(
            FinalizeDecodeMigrationReqInput(
                rid="request", migration_id="migration", action="commit"
            )
        )

        self.assertTrue(output.success)
        release_kv_cache.assert_called_once_with(
            req, scheduler.tree_cache, is_insert=False
        )
        self.assertEqual(scheduler.decode_migration_transfers, {})
        self.assertEqual(scheduler.decode_migration_by_rid, {})

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
        self.assertEqual(scheduler.decode_migration_transfers, {})
        self.assertEqual(scheduler.decode_migration_by_rid, {})


if __name__ == "__main__":
    unittest.main()
