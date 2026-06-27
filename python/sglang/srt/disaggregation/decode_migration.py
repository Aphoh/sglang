"""Request-local support for live decode-to-decode migration.

The destination reuses SGLang's disaggregated-decode receiver. The source
resolves any outstanding overlap result, removes only the target request from
scheduling, and retains its KV until one-way finalization.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.decode_migration_state import (
    build_decode_migration_frontier,
)
from sglang.srt.disaggregation.prefill import create_prefill_kv_manager
from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    KVClassType,
    get_kv_class,
    poll_and_all_reduce_attn_cp_tp_group,
)
from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import (
    FinalizeDecodeMigrationReqInput,
    FinalizeDecodeMigrationReqOutput,
    PrepareDecodeMigrationReqInput,
    PrepareDecodeMigrationReqOutput,
)
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.mem_cache.common import kv_to_page_indices, release_kv_cache
from sglang.srt.observability.req_time_stats import set_time_batch

if TYPE_CHECKING:
    from sglang.srt.disaggregation.base import BaseKVManager, BaseKVSender
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)


@dataclass
class DecodeMigrationTransfer:
    migration_id: str
    req: Req
    sender: "BaseKVSender"
    metadata_buffer_index: int
    bootstrap_room: int
    committed_len: int
    logical_len: int
    output_tokens_seen: int
    pending_input_ids: list[int]
    created_at: float
    transfer_start: int = 0
    transfer_end: int = 0
    send_started: bool = False
    status: str = "bootstrapping"
    error: Optional[str] = None
    source_released: bool = False


class SchedulerDecodeMigrationMixin:
    """Scheduler control plane for source-side decode migration."""

    def init_decode_migration(self: "Scheduler") -> None:
        self.decode_migration_transfers: Dict[str, DecodeMigrationTransfer] = {}
        self.decode_migration_by_rid: Dict[str, str] = {}
        self.decode_migration_arms: Dict[str, PrepareDecodeMigrationReqInput] = {}
        self.decode_migration_arm_by_rid: Dict[str, str] = {}
        self._decode_migration_kv_manager: Optional[BaseKVManager] = None
        self._decode_migration_overlap_result_processed = False
        if not self.server_args.enable_decode_migration:
            return
        # Fail startup early on transfer-backend incompatibility.
        self._get_decode_migration_kv_manager()

    def _get_decode_migration_kv_manager(self: "Scheduler") -> "BaseKVManager":
        if self._decode_migration_kv_manager is None:
            if self.disaggregation_mode == DisaggregationMode.PREFILL:
                self._decode_migration_kv_manager = (
                    self.disagg_prefill_bootstrap_queue.kv_manager
                )
            else:
                self._decode_migration_kv_manager = create_prefill_kv_manager(
                    token_to_kv_pool=(self.token_to_kv_pool_allocator.get_kvcache()),
                    draft_token_to_kv_pool=None,
                    metadata_buffers=self.disagg_metadata_buffers,
                    transfer_backend=self.transfer_backend,
                    scheduler=self,
                    tp_rank=self.ps.tp_rank,
                    pp_rank=self.ps.pp_rank,
                )
        return self._decode_migration_kv_manager

    def process_decode_migration_receives(self: "Scheduler") -> None:
        """Advance destination handshakes on non-PD-decode workers."""
        if (
            not self.server_args.enable_decode_migration
            or self.disaggregation_mode == DisaggregationMode.DECODE
        ):
            return
        self.process_decode_queue()

    def admit_ready_decode_migrations(self: "Scheduler") -> None:
        """Merge transferred destination requests without recomputing their KV."""
        if (
            not self.server_args.enable_decode_migration
            or self.disaggregation_mode == DisaggregationMode.DECODE
        ):
            return

        ready = [
            req
            for req in self.waiting_queue
            if getattr(req, "is_decode_migration_destination", False)
        ]
        if not ready:
            return

        available = min(self.req_to_token_pool.size, self.max_running_requests)
        available -= self.running_batch.batch_size()
        if available <= 0:
            return
        selected = ready[:available]
        selected_ids = {id(req) for req in selected}
        self.waiting_queue = [
            req for req in self.waiting_queue if id(req) not in selected_ids
        ]

        for req in selected:
            # Decode preallocation already established the complete destination
            # KV layout, with or without decode-side radix matching. Preserve
            # that exact ownership state. Re-matching here can observe a prefix
            # inserted while transfer was in flight and mark transferred indices
            # as cache-protected even though this request does not own that lock.
            req.init_next_round_input(None)
            if req.kv_committed_len is not None:
                req.fill_len = req.kv_committed_len
                req.set_extend_input_len(req.fill_len - len(req.prefix_indices))
        set_time_batch(selected, "set_forward_entry_time")

        batch = ScheduleBatch.init_new(
            selected,
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            self.spec_algorithm,
        )
        batch.prepare_for_prebuilt()
        batch.process_prebuilt(self.server_args, self.future_map)
        self.batch_result_processor.process_batch_result_prebuilt(batch)
        batch.filter_batch()
        if batch.is_empty():
            return
        if self.running_batch.is_empty():
            self.running_batch = batch
        else:
            self.running_batch.merge_batch(batch)
        self.running_batch.batch_is_full = False

    def abort_decode_migration_receive(self: "Scheduler", recv_req) -> None:
        """Abort matching destination handshakes and parked source requests."""
        if not self.server_args.enable_decode_migration:
            return
        if self.disaggregation_mode != DisaggregationMode.DECODE:
            for decode_req in self.disagg_decode_prealloc_queue.queue:
                if recv_req.abort_all or decode_req.req.rid.startswith(recv_req.rid):
                    decode_req.kv_receiver.abort()
            for decode_req in self.disagg_decode_transfer_queue.queue:
                if recv_req.abort_all or decode_req.req.rid.startswith(recv_req.rid):
                    decode_req.kv_receiver.abort()

        for migration_id, record in list(self.decode_migration_transfers.items()):
            if recv_req.abort_all or record.req.rid.startswith(recv_req.rid):
                self._release_decode_migration_source(record)
                self._release_decode_migration_transport(record)
                self.decode_migration_transfers.pop(migration_id, None)
                self.decode_migration_by_rid.pop(record.req.rid, None)

        for migration_id, arm in list(self.decode_migration_arms.items()):
            if recv_req.abort_all or arm.rid.startswith(recv_req.rid):
                self._clear_decode_migration_arm(migration_id, arm.rid)

    def _find_decode_migration_req(self: "Scheduler", rid: str) -> Optional[Req]:
        for req in self.running_batch.reqs:
            if req.rid == rid:
                return req
        logger.debug("Decode migration request lookup missed rid=%s", rid)
        return None

    def _prepare_failure(
        self: "Scheduler",
        recv_req: PrepareDecodeMigrationReqInput,
        status: str,
        error: Optional[str] = None,
        **state,
    ) -> PrepareDecodeMigrationReqOutput:
        return PrepareDecodeMigrationReqOutput(
            rid=recv_req.rid,
            migration_id=recv_req.migration_id,
            success=False,
            status=status,
            source_dp_rank=self.ps.dp_rank or 0,
            error=error,
            **state,
        )

    def _resolve_decode_migration_overlap_result(self: "Scheduler") -> None:
        """Resolve the outstanding overlap result once before request parking."""
        if not self.enable_overlap or not self.result_queue:
            return
        if len(self.result_queue) != 1:
            raise RuntimeError(
                f"Expected at most one outstanding overlap result, got "
                f"{len(self.result_queue)}"
            )
        batch, result = self.result_queue.popleft()
        self.process_batch_result(batch, result)
        self._decode_migration_overlap_result_processed = True

    @staticmethod
    def _filter_req_from_batch(batch: Optional[ScheduleBatch], req: Req) -> bool:
        if batch is None or all(candidate is not req for candidate in batch.reqs):
            return False
        keep_indices = [
            i for i, candidate in enumerate(batch.reqs) if candidate is not req
        ]
        batch.filter_batch(keep_indices=keep_indices)
        if batch.decoding_reqs is not None:
            batch.decoding_reqs = [
                candidate for candidate in batch.decoding_reqs if candidate is not req
            ]
        batch.batch_is_full = False
        return True

    def _park_decode_migration_req(self: "Scheduler", req: Req) -> None:
        req.is_decode_migration_source_parked = True
        removed = self._filter_req_from_batch(self.running_batch, req)
        if self.last_batch is not self.running_batch:
            self._filter_req_from_batch(self.last_batch, req)
        if (
            self.cur_batch is not None
            and self.cur_batch is not self.running_batch
            and self.cur_batch is not self.last_batch
        ):
            self._filter_req_from_batch(self.cur_batch, req)
        if not removed:
            raise RuntimeError("Request left the running decode batch before migration")

    def _clear_decode_migration_arm(
        self: "Scheduler", migration_id: str, rid: str
    ) -> None:
        self.decode_migration_arms.pop(migration_id, None)
        if self.decode_migration_arm_by_rid.get(rid) == migration_id:
            self.decode_migration_arm_by_rid.pop(rid, None)

    def maybe_park_decode_migration_at_boundary(self: "Scheduler", req: Req) -> bool:
        migration_id = self.decode_migration_arm_by_rid.get(req.rid)
        if migration_id is None:
            return False
        arm = self.decode_migration_arms.get(migration_id)
        if arm is None:
            self.decode_migration_arm_by_rid.pop(req.rid, None)
            return False
        if req.finished() or req.to_finish is not None:
            self._clear_decode_migration_arm(migration_id, req.rid)
            return False
        target = arm.target_sequence_length
        logical_len = len(req.origin_input_ids) + len(req.output_ids)
        if target is None or logical_len < target:
            return False

        self._clear_decode_migration_arm(migration_id, req.rid)
        output = self._prepare_decode_migration_now(arm, req=req, resolve_overlap=False)
        if not output.success:
            logger.error(
                "Failed to park armed decode migration rid=%s migration_id=%s "
                "status=%s error=%s",
                req.rid,
                migration_id,
                output.status,
                output.error,
            )
            return False
        return True

    def _release_decode_migration_source(
        self: "Scheduler", record: DecodeMigrationTransfer
    ) -> None:
        if record.source_released:
            return
        if record.req.req_pool_idx is not None:
            release_kv_cache(record.req, self.tree_cache, is_insert=False)
        record.source_released = True

    def prepare_decode_migration(
        self: "Scheduler", recv_req: PrepareDecodeMigrationReqInput
    ) -> PrepareDecodeMigrationReqOutput:
        if not self.server_args.enable_decode_migration:
            return self._prepare_failure(
                recv_req,
                "error",
                "Decode migration requires --enable-decode-migration",
            )

        existing = self.decode_migration_by_rid.get(recv_req.rid)
        if existing is not None:
            record = self.decode_migration_transfers.get(existing)
            if existing == recv_req.migration_id and record is not None:
                return self._prepared_output(recv_req, record)
            return self._prepare_failure(
                recv_req, "busy", f"Request already has active migration {existing}"
            )

        armed = self.decode_migration_arm_by_rid.get(recv_req.rid)
        if armed is not None and armed != recv_req.migration_id:
            return self._prepare_failure(
                recv_req, "busy", f"Request already has armed migration {armed}"
            )

        req = self._find_decode_migration_req(recv_req.rid)
        target = recv_req.target_sequence_length
        logical_len = (
            len(req.origin_input_ids) + len(req.output_ids) if req is not None else 0
        )
        if target is not None and target > logical_len:
            if target <= 0:
                return self._prepare_failure(
                    recv_req, "error", "target_sequence_length must be positive"
                )
            self.decode_migration_arms[recv_req.migration_id] = recv_req
            self.decode_migration_arm_by_rid[recv_req.rid] = recv_req.migration_id
            logger.info(
                "Armed decode migration rid=%s migration_id=%s target=%d current=%d",
                recv_req.rid,
                recv_req.migration_id,
                target,
                logical_len,
            )
            return PrepareDecodeMigrationReqOutput(
                rid=recv_req.rid,
                migration_id=recv_req.migration_id,
                success=True,
                status="armed",
                bootstrap_host=recv_req.bootstrap_host,
                bootstrap_port=recv_req.bootstrap_port,
                bootstrap_room=recv_req.bootstrap_room,
                logical_len=logical_len,
                output_tokens_seen=recv_req.output_tokens_seen,
                source_dp_rank=self.ps.dp_rank or 0,
            )

        if armed == recv_req.migration_id:
            arm = self.decode_migration_arms[armed]
            return PrepareDecodeMigrationReqOutput(
                rid=recv_req.rid,
                migration_id=recv_req.migration_id,
                success=True,
                status="armed",
                bootstrap_host=arm.bootstrap_host,
                bootstrap_port=arm.bootstrap_port,
                bootstrap_room=arm.bootstrap_room,
                logical_len=logical_len,
                output_tokens_seen=arm.output_tokens_seen,
                source_dp_rank=self.ps.dp_rank or 0,
            )

        return self._prepare_decode_migration_now(recv_req, req=req)

    def _prepare_decode_migration_now(
        self: "Scheduler",
        recv_req: PrepareDecodeMigrationReqInput,
        *,
        req: Optional[Req] = None,
        resolve_overlap: bool = True,
    ) -> PrepareDecodeMigrationReqOutput:

        if resolve_overlap:
            try:
                self._resolve_decode_migration_overlap_result()
            except Exception as exc:
                logger.exception(
                    "Failed to resolve overlap result for rid=%s migration_id=%s",
                    recv_req.rid,
                    recv_req.migration_id,
                )
                return self._prepare_failure(
                    recv_req, "error", f"Failed to resolve source frontier: {exc}"
                )

        if req is None:
            req = self._find_decode_migration_req(recv_req.rid)
        if req is None:
            return self._prepare_failure(
                recv_req, "not_found", "Request is not in the running decode batch"
            )
        if req.finished() or req.to_finish is not None:
            return self._prepare_failure(
                recv_req,
                "finished",
                prompt_len=len(req.origin_input_ids),
                logical_len=len(req.origin_input_ids) + len(req.output_ids),
                output_tokens_seen=recv_req.output_tokens_seen,
            )

        prompt_ids = list(req.origin_input_ids)
        output_ids = list(req.output_ids)
        committed_len = req.kv_committed_len
        actual_logical_len = len(prompt_ids) + len(output_ids)
        target = recv_req.target_sequence_length
        if target is not None:
            if target <= len(prompt_ids):
                return self._prepare_failure(
                    recv_req,
                    "error",
                    "target_sequence_length must extend beyond the prompt",
                    prompt_len=len(prompt_ids),
                    committed_len=committed_len,
                    logical_len=actual_logical_len,
                    output_tokens_seen=recv_req.output_tokens_seen,
                )
            if target > actual_logical_len or target - 1 > committed_len:
                return self._prepare_failure(
                    recv_req,
                    "error",
                    "Source has not reached the requested migration frontier",
                    prompt_len=len(prompt_ids),
                    committed_len=committed_len,
                    logical_len=actual_logical_len,
                    output_tokens_seen=recv_req.output_tokens_seen,
                )
            output_ids = output_ids[: target - len(prompt_ids)]
            committed_len = target - 1
        try:
            frontier = build_decode_migration_frontier(
                prompt_ids,
                output_ids,
                committed_len,
                recv_req.output_tokens_seen,
            )
        except ValueError as exc:
            return self._prepare_failure(
                recv_req,
                "error",
                str(exc),
                prompt_len=len(prompt_ids),
                committed_len=committed_len,
                logical_len=len(prompt_ids) + len(output_ids),
                output_tokens_seen=recv_req.output_tokens_seen,
            )

        if self.req_to_metadata_buffer_idx_allocator.available_size() <= 0:
            return self._prepare_failure(
                recv_req, "error", "No metadata buffer is available for migration"
            )
        room = recv_req.bootstrap_room
        if room <= 0:
            return self._prepare_failure(
                recv_req,
                "error",
                "bootstrap_room must be an opaque positive integer",
            )

        metadata_index = -1
        sender = None
        try:
            manager = self._get_decode_migration_kv_manager()
            sender_class = get_kv_class(self.transfer_backend, KVClassType.SENDER)
            sender = sender_class(
                mgr=manager,
                bootstrap_addr=f"{recv_req.bootstrap_host}:{recv_req.bootstrap_port}",
                bootstrap_room=room,
                dest_tp_ranks=[self.ps.tp_rank],
                pp_rank=self.ps.pp_rank,
            )
            allocated_index = self.req_to_metadata_buffer_idx_allocator.alloc()
            assert allocated_index is not None
            metadata_index = allocated_index
            self.disagg_metadata_buffers.output_ids[metadata_index][0] = (
                frontier.pending_input_ids[0]
            )
            self.disagg_metadata_buffers.cached_tokens[metadata_index].zero_()
            self.disagg_metadata_buffers.bootstrap_room[metadata_index][0] = room
            self._park_decode_migration_req(req)
        except Exception as exc:
            if sender is not None:
                sender.clear()
            if metadata_index >= 0:
                self.req_to_metadata_buffer_idx_allocator.free(metadata_index)
            if all(candidate is not req for candidate in self.running_batch.reqs):
                release_kv_cache(req, self.tree_cache, is_insert=False)
            logger.exception(
                "Failed to prepare decode migration rid=%s migration_id=%s",
                recv_req.rid,
                recv_req.migration_id,
            )
            return self._prepare_failure(
                recv_req, "error", f"Failed to create migration transfer: {exc}"
            )

        record = DecodeMigrationTransfer(
            migration_id=recv_req.migration_id,
            req=req,
            sender=sender,
            metadata_buffer_index=metadata_index,
            bootstrap_room=room,
            committed_len=committed_len,
            logical_len=frontier.logical_len,
            output_tokens_seen=frontier.output_tokens_seen,
            pending_input_ids=frontier.pending_input_ids,
            created_at=time.monotonic(),
            transfer_end=committed_len,
        )
        self.decode_migration_transfers[recv_req.migration_id] = record
        self.decode_migration_by_rid[recv_req.rid] = recv_req.migration_id

        logger.info(
            "Prepared decode migration rid=%s migration_id=%s committed=%d "
            "logical=%d actual_logical=%d seen=%d room=%d",
            recv_req.rid,
            recv_req.migration_id,
            committed_len,
            frontier.logical_len,
            actual_logical_len,
            frontier.output_tokens_seen,
            room,
        )
        return PrepareDecodeMigrationReqOutput(
            rid=recv_req.rid,
            migration_id=recv_req.migration_id,
            success=True,
            status="prepared",
            bootstrap_host=recv_req.bootstrap_host,
            bootstrap_port=recv_req.bootstrap_port,
            bootstrap_room=room,
            committed_input_ids=frontier.committed_input_ids,
            pending_input_ids=frontier.pending_input_ids,
            unforwarded_committed_output_ids=(
                frontier.unforwarded_committed_output_ids
            ),
            prompt_len=frontier.prompt_len,
            committed_len=committed_len,
            logical_len=frontier.logical_len,
            output_tokens_seen=frontier.output_tokens_seen,
            source_dp_rank=self.ps.dp_rank or 0,
        )

    def _prepared_output(
        self: "Scheduler",
        recv_req: PrepareDecodeMigrationReqInput,
        record: DecodeMigrationTransfer,
    ) -> PrepareDecodeMigrationReqOutput:
        req = record.req
        logical_ids = (list(req.origin_input_ids) + list(req.output_ids))[
            : record.logical_len
        ]
        committed_input_ids = logical_ids[: record.committed_len]
        prompt_len = len(req.origin_input_ids)
        committed_output_count = max(0, record.committed_len - prompt_len)
        output_ids = logical_ids[prompt_len:]
        seen = record.output_tokens_seen
        return PrepareDecodeMigrationReqOutput(
            rid=recv_req.rid,
            migration_id=recv_req.migration_id,
            success=True,
            status="prepared",
            bootstrap_host=recv_req.bootstrap_host,
            bootstrap_port=recv_req.bootstrap_port,
            bootstrap_room=record.bootstrap_room,
            committed_input_ids=committed_input_ids,
            pending_input_ids=record.pending_input_ids,
            unforwarded_committed_output_ids=output_ids[
                min(seen, committed_output_count) : committed_output_count
            ],
            prompt_len=prompt_len,
            committed_len=record.committed_len,
            logical_len=record.logical_len,
            output_tokens_seen=seen,
            source_dp_rank=self.ps.dp_rank or 0,
        )

    def process_decode_migration_transfers(self: "Scheduler") -> None:
        if not self.decode_migration_transfers:
            return

        records = [
            record
            for record in self.decode_migration_transfers.values()
            if record.status not in ("transferred", "failed")
        ]
        if not records:
            return
        polls = poll_and_all_reduce_attn_cp_tp_group(
            [record.sender for record in records],
            self.attn_cp_cpu_group,
            self.attn_tp_cpu_group,
        )
        timeout_s = envs.SGLANG_DISAGGREGATION_WAITING_TIMEOUT.get()

        for record, poll in zip(records, polls):
            if time.monotonic() - record.created_at > timeout_s:
                self._fail_decode_migration(record, "Migration transfer timed out")
                continue
            if poll == KVPoll.Bootstrapping:
                record.status = "bootstrapping"
                continue
            if poll == KVPoll.WaitingForInput and not record.send_started:
                decode_prefix_len = record.sender.pop_decode_prefix_len()
                if decode_prefix_len < 0 or decode_prefix_len > record.committed_len:
                    self._fail_decode_migration(
                        record,
                        f"Invalid destination prefix length {decode_prefix_len}",
                    )
                    continue

                record.transfer_start = decode_prefix_len
                token_to_kv_pool = self.token_to_kv_pool_allocator.get_kvcache()
                kv_indices = self.req_to_token_pool.req_to_token[
                    record.req.req_pool_idx,
                    record.transfer_start : record.transfer_end,
                ]
                page_indices = kv_to_page_indices(
                    kv_indices.cpu().numpy(), token_to_kv_pool.page_size
                )
                record.sender.init(len(page_indices), record.metadata_buffer_index)
                record.sender.send(page_indices, [])
                record.send_started = True
                record.status = "transferring"
                logger.info(
                    "Started decode migration transfer rid=%s migration_id=%s "
                    "range=[%d,%d) pages=%d",
                    record.req.rid,
                    record.migration_id,
                    record.transfer_start,
                    record.transfer_end,
                    len(page_indices),
                )
                continue
            if poll in (KVPoll.WaitingForInput, KVPoll.Transferring):
                record.status = "transferring"
                continue
            if poll == KVPoll.Success:
                record.status = "transferred"
                self._release_decode_migration_transport(record)
                logger.info(
                    "Decode migration transfer completed rid=%s migration_id=%s",
                    record.req.rid,
                    record.migration_id,
                )
                continue
            if poll == KVPoll.Failed:
                error = "Decode migration transfer failed"
                try:
                    record.sender.failure_exception()
                except Exception as exc:
                    error = f"{error}: {exc}"
                self._fail_decode_migration(record, error)

    def _release_decode_migration_transport(
        self: "Scheduler", record: DecodeMigrationTransfer
    ) -> None:
        if record.sender is not None:
            record.sender.clear()
        if record.metadata_buffer_index >= 0:
            self.disagg_metadata_buffers.bootstrap_room[
                record.metadata_buffer_index
            ].zero_()
            self.req_to_metadata_buffer_idx_allocator.free(record.metadata_buffer_index)
            record.metadata_buffer_index = -1

    def _fail_decode_migration(
        self: "Scheduler", record: DecodeMigrationTransfer, error: str
    ) -> None:
        logger.error(
            "Decode migration failed rid=%s migration_id=%s error=%s",
            record.req.rid,
            record.migration_id,
            error,
        )
        record.status = "failed"
        record.error = error
        self._release_decode_migration_transport(record)
        self._release_decode_migration_source(record)

    def finalize_decode_migration(
        self: "Scheduler", recv_req: FinalizeDecodeMigrationReqInput
    ) -> FinalizeDecodeMigrationReqOutput:
        record = self.decode_migration_transfers.get(recv_req.migration_id)
        armed = self.decode_migration_arms.get(recv_req.migration_id)
        if record is None and armed is not None and armed.rid == recv_req.rid:
            if recv_req.action == "cancel":
                self._clear_decode_migration_arm(recv_req.migration_id, recv_req.rid)
                return FinalizeDecodeMigrationReqOutput(
                    rid=recv_req.rid,
                    migration_id=recv_req.migration_id,
                    action=recv_req.action,
                    success=True,
                    transfer_status="unknown",
                    source_dp_rank=self.ps.dp_rank or 0,
                )
            return FinalizeDecodeMigrationReqOutput(
                rid=recv_req.rid,
                migration_id=recv_req.migration_id,
                action=recv_req.action,
                success=False,
                transfer_status="unknown",
                source_dp_rank=self.ps.dp_rank or 0,
                error="Migration source is armed but not yet parked",
            )
        if record is None or record.req.rid != recv_req.rid:
            return FinalizeDecodeMigrationReqOutput(
                rid=recv_req.rid,
                migration_id=recv_req.migration_id,
                action=recv_req.action,
                success=False,
                transfer_status="unknown",
                source_dp_rank=self.ps.dp_rank or 0,
                error="Migration generation is not active",
            )

        status = record.status
        if recv_req.action == "commit" and status != "transferred":
            return FinalizeDecodeMigrationReqOutput(
                rid=recv_req.rid,
                migration_id=recv_req.migration_id,
                action=recv_req.action,
                success=False,
                transfer_status=status,
                source_dp_rank=self.ps.dp_rank or 0,
                error="Destination cannot commit before source transfer completes",
            )
        if recv_req.action == "resume":
            return FinalizeDecodeMigrationReqOutput(
                rid=recv_req.rid,
                migration_id=recv_req.migration_id,
                action=recv_req.action,
                success=False,
                transfer_status=status,
                source_dp_rank=self.ps.dp_rank or 0,
                error="Source resumption is unsupported after request quiescence",
            )

        self._release_decode_migration_transport(record)
        self._release_decode_migration_source(record)
        self.decode_migration_transfers.pop(recv_req.migration_id, None)
        self.decode_migration_by_rid.pop(recv_req.rid, None)

        logger.info(
            "Finalized decode migration rid=%s migration_id=%s action=%s status=%s",
            recv_req.rid,
            recv_req.migration_id,
            recv_req.action,
            status,
        )
        return FinalizeDecodeMigrationReqOutput(
            rid=recv_req.rid,
            migration_id=recv_req.migration_id,
            action=recv_req.action,
            success=True,
            transfer_status=status,
            source_dp_rank=self.ps.dp_rank or 0,
        )
