"""Transactional source-side support for live decode-to-decode migration.

The destination reuses SGLang's normal disaggregated-decode receiver. The source
is quiesced, exposes an exact committed KV range through the existing P/D sender,
and retains ownership until an explicit commit arrives.

The first prototype uses the scheduler's in-place engine pause as its quiescence
barrier. This is intentionally coarse, but keeps request and KV state untouched
for rollback. The transfer record is range-based so a later per-request pause and
incremental delta sender can reuse the same lifecycle.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional

import torch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.decode import DecodePreallocQueue, DecodeTransferQueue
from sglang.srt.disaggregation.decode_migration_state import (
    build_decode_migration_frontier,
)
from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    KVClassType,
    MetadataBuffers,
    ReqToMetadataIdxAllocator,
    TransferBackend,
    get_kv_class,
    is_mla_backend,
    poll_and_all_reduce_attn_cp_tp_group,
    setup_state_kv_args,
)
from sglang.srt.environ import envs
from sglang.srt.mem_cache import kv_cache_builder
from sglang.srt.managers.io_struct import (
    FinalizeDecodeMigrationReqInput,
    FinalizeDecodeMigrationReqOutput,
    PrepareDecodeMigrationReqInput,
    PrepareDecodeMigrationReqOutput,
)
from sglang.srt.managers.schedule_batch import FINISH_ABORT, Req, ScheduleBatch
from sglang.srt.observability.req_time_stats import set_time_batch
from sglang.srt.mem_cache.common import kv_to_page_indices

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
    pending_input_ids: list[int]
    created_at: float
    transfer_start: int = 0
    transfer_end: int = 0
    send_started: bool = False
    status: str = "bootstrapping"
    error: Optional[str] = None


class SchedulerDecodeMigrationMixin:
    """Scheduler control plane for source-side decode migration."""

    def init_decode_migration(self: "Scheduler") -> None:
        self.decode_migration_transfers: Dict[str, DecodeMigrationTransfer] = {}
        self.decode_migration_by_rid: Dict[str, str] = {}
        self._decode_migration_kv_manager: Optional[BaseKVManager] = None
        if not self.server_args.enable_decode_migration:
            return
        if not hasattr(self, "req_to_metadata_buffer_idx_allocator"):
            buffer_size = self.max_running_requests * 2
            self.req_to_metadata_buffer_idx_allocator = ReqToMetadataIdxAllocator(
                buffer_size
            )
            self.disagg_metadata_buffers = MetadataBuffers(
                buffer_size,
                hidden_size=16,
                hidden_states_dtype=torch.float32,
                custom_mem_pool=self.token_to_kv_pool_allocator.get_kvcache().maybe_get_custom_mem_pool(),
            )
        # A migration-capable worker may be either endpoint regardless of its
        # ordinary serving mode. Reuse the normal decode receiver queues instead
        # of requiring the whole worker to run in PD decode mode.
        self._init_decode_migration_receiver()

        # Fail startup early on transfer-backend incompatibility and avoid
        # constructing/registering the NIXL manager while a request is paused.
        self._get_decode_migration_kv_manager()

    def _init_decode_migration_receiver(self: "Scheduler") -> None:
        if self.disagg_decode_prealloc_queue is not None:
            return

        draft_token_to_kv_pool, _ = kv_cache_builder.get_draft_kv_pool(
            draft_worker=self.draft_worker,
            spec_algorithm=self.spec_algorithm,
            server_args=self.server_args,
        )
        self.disagg_decode_transfer_queue = DecodeTransferQueue(
            gloo_group=self.attn_tp_cpu_group,
            req_to_metadata_buffer_idx_allocator=(
                self.req_to_metadata_buffer_idx_allocator
            ),
            tp_rank=self.ps.tp_rank,
            metadata_buffers=self.disagg_metadata_buffers,
            scheduler=self,
            tree_cache=self.tree_cache,
        )
        self.disagg_decode_prealloc_queue = DecodePreallocQueue(
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            draft_token_to_kv_pool=draft_token_to_kv_pool,
            req_to_metadata_buffer_idx_allocator=(
                self.req_to_metadata_buffer_idx_allocator
            ),
            metadata_buffers=self.disagg_metadata_buffers,
            scheduler=self,
            transfer_queue=self.disagg_decode_transfer_queue,
            tree_cache=self.tree_cache,
            gloo_group=self.attn_tp_cpu_group,
            tp_rank=self.ps.tp_rank,
            tp_size=self.ps.tp_size,
            dp_size=self.server_args.dp_size,
            gpu_id=self.ps.gpu_id,
            bootstrap_port=self.server_args.disaggregation_bootstrap_port,
            max_total_num_tokens=self.max_total_num_tokens,
            pp_rank=self.ps.pp_rank,
            num_reserved_decode_tokens=self.server_args.num_reserved_decode_tokens,
            transfer_backend=self.transfer_backend,
            # Aggregated workers have a normal prefix cache. Match it during
            # receive preallocation so imported KV never duplicates a prefix
            # that admission later marks as protected.
            enable_radix_cache=True,
        )

    def _get_decode_migration_kv_manager(self: "Scheduler") -> "BaseKVManager":
        if self._decode_migration_kv_manager is not None:
            return self._decode_migration_kv_manager

        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            manager = self.disagg_prefill_bootstrap_queue.kv_manager
            self._decode_migration_kv_manager = manager
            return manager

        if not self.server_args.enable_decode_migration:
            raise RuntimeError("Decode migration is not enabled")

        token_to_kv_pool = self.token_to_kv_pool_allocator.get_kvcache()
        transfer_backend = TransferBackend(
            self.server_args.disaggregation_transfer_backend
        )
        kv_args_class = get_kv_class(transfer_backend, KVClassType.KVARGS)
        kv_args = kv_args_class()
        kv_args.engine_rank = self.ps.tp_rank
        kv_args.pp_rank = self.ps.pp_rank
        kv_args.system_dp_rank = self.ps.dp_rank
        kv_args.prefill_start_layer = token_to_kv_pool.start_layer
        kv_args.prefill_end_layer = getattr(token_to_kv_pool, "end_layer", None)
        kv_args.mla_compression_ratios = None

        (
            kv_args.kv_data_ptrs,
            kv_args.kv_data_lens,
            kv_args.kv_item_lens,
        ) = token_to_kv_pool.get_contiguous_buf_infos()
        if not is_mla_backend(token_to_kv_pool):
            kv_args.kv_head_num = token_to_kv_pool.head_num
            kv_args.total_kv_head_num = self.model_config.get_total_num_kv_heads()
        kv_args.page_size = token_to_kv_pool.page_size
        (
            kv_args.aux_data_ptrs,
            kv_args.aux_data_lens,
            kv_args.aux_item_lens,
        ) = self.disagg_metadata_buffers.get_buf_infos()
        kv_args.ib_device = self.server_args.disaggregation_ib_device
        kv_args.gpu_id = self.ps.gpu_id
        setup_state_kv_args(
            kv_args,
            token_to_kv_pool,
            total_kv_layers=self.model_config.num_hidden_layers,
            req_to_token_pool=self.req_to_token_pool,
        )

        manager_class = get_kv_class(transfer_backend, KVClassType.MANAGER)
        manager = manager_class(
            kv_args,
            DisaggregationMode.PREFILL,
            self.server_args,
            is_mla_backend(token_to_kv_pool),
        )
        if (
            envs.SGLANG_DISAGG_STAGING_BUFFER.get()
            and hasattr(manager, "set_kv_buffer_tensors")
            and not is_mla_backend(token_to_kv_pool)
        ):
            kv_pool = token_to_kv_pool
            if hasattr(kv_pool, "full_kv_pool"):
                kv_pool = kv_pool.full_kv_pool
            if hasattr(kv_pool, "k_buffer") and hasattr(kv_pool, "v_buffer"):
                manager.set_kv_buffer_tensors(
                    kv_pool.k_buffer, kv_pool.v_buffer, kv_pool.page_size
                )

        self._decode_migration_kv_manager = manager
        return manager

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
        """Abort destination handshakes when the scheduler is not in decode mode."""
        if (
            not self.server_args.enable_decode_migration
            or self.disaggregation_mode == DisaggregationMode.DECODE
        ):
            return
        for decode_req in self.disagg_decode_prealloc_queue.queue:
            if recv_req.abort_all or decode_req.req.rid.startswith(recv_req.rid):
                decode_req.kv_receiver.abort()
        for decode_req in self.disagg_decode_transfer_queue.queue:
            if recv_req.abort_all or decode_req.req.rid.startswith(recv_req.rid):
                decode_req.kv_receiver.abort()

    def _find_decode_migration_req(self: "Scheduler", rid: str) -> Optional[Req]:
        for req in self.running_batch.reqs:
            if req.rid == rid:
                return req
        logger.debug("Decode migration request lookup missed rid=%s", rid)
        return None

    def prepare_decode_migration(
        self: "Scheduler", recv_req: PrepareDecodeMigrationReqInput
    ) -> PrepareDecodeMigrationReqOutput:
        if self.enable_overlap:
            return PrepareDecodeMigrationReqOutput(
                rid=recv_req.rid,
                migration_id=recv_req.migration_id,
                success=False,
                status="error",
                error=(
                    "Decode migration prototype requires "
                    "--disable-overlap-schedule for an exact quiescence frontier"
                ),
            )
        if not self.server_args.enable_decode_migration:
            return PrepareDecodeMigrationReqOutput(
                rid=recv_req.rid,
                migration_id=recv_req.migration_id,
                success=False,
                status="error",
                error="Decode migration requires --enable-decode-migration",
            )
        if self._engine_paused and not self.decode_migration_transfers:
            return PrepareDecodeMigrationReqOutput(
                rid=recv_req.rid,
                migration_id=recv_req.migration_id,
                success=False,
                status="busy",
                error="The engine is paused by another control operation",
            )

        existing = self.decode_migration_by_rid.get(recv_req.rid)
        if existing is not None:
            record = self.decode_migration_transfers.get(existing)
            if existing == recv_req.migration_id and record is not None:
                return self._prepared_output(recv_req, record)
            return PrepareDecodeMigrationReqOutput(
                rid=recv_req.rid,
                migration_id=recv_req.migration_id,
                success=False,
                status="busy",
                error=f"Request already has active migration {existing}",
            )
        if self.decode_migration_transfers:
            return PrepareDecodeMigrationReqOutput(
                rid=recv_req.rid,
                migration_id=recv_req.migration_id,
                success=False,
                status="busy",
                error="The coarse-pause prototype supports one migration at a time",
            )

        # The control message is processed between scheduler iterations. Pausing
        # here leaves the live request and all KV allocations untouched.
        self._engine_paused = True
        req = self._find_decode_migration_req(recv_req.rid)
        if req is None:
            self._engine_paused = False
            return PrepareDecodeMigrationReqOutput(
                rid=recv_req.rid,
                migration_id=recv_req.migration_id,
                success=False,
                status="not_found",
                error="Request is not in the running decode batch",
            )
        if req.finished() or req.to_finish is not None:
            self._engine_paused = False
            return PrepareDecodeMigrationReqOutput(
                rid=recv_req.rid,
                migration_id=recv_req.migration_id,
                success=False,
                status="finished",
                prompt_len=len(req.origin_input_ids),
                logical_len=len(req.origin_input_ids) + len(req.output_ids),
                output_tokens_seen=recv_req.output_tokens_seen,
            )

        prompt_ids = list(req.origin_input_ids)
        output_ids = list(req.output_ids)
        committed_len = req.kv_committed_len
        try:
            frontier = build_decode_migration_frontier(
                prompt_ids,
                output_ids,
                committed_len,
                recv_req.output_tokens_seen,
            )
        except ValueError as exc:
            self._engine_paused = False
            return PrepareDecodeMigrationReqOutput(
                rid=recv_req.rid,
                migration_id=recv_req.migration_id,
                success=False,
                status="error",
                prompt_len=len(prompt_ids),
                committed_len=committed_len,
                logical_len=len(prompt_ids) + len(output_ids),
                output_tokens_seen=recv_req.output_tokens_seen,
                error=str(exc),
            )

        if self.req_to_metadata_buffer_idx_allocator.available_size() <= 0:
            self._engine_paused = False
            return PrepareDecodeMigrationReqOutput(
                rid=recv_req.rid,
                migration_id=recv_req.migration_id,
                success=False,
                status="error",
                error="No metadata buffer is available for migration",
            )

        room = recv_req.bootstrap_room
        if room <= 0:
            self._engine_paused = False
            return PrepareDecodeMigrationReqOutput(
                rid=recv_req.rid,
                migration_id=recv_req.migration_id,
                success=False,
                status="error",
                error="bootstrap_room must be an opaque positive integer",
            )
        metadata_index = -1
        sender = None
        try:
            manager = self._get_decode_migration_kv_manager()
            sender_class = get_kv_class(self.transfer_backend, KVClassType.SENDER)
            sender = sender_class(
                mgr=manager,
                bootstrap_addr=(
                    f"{recv_req.bootstrap_host}:{recv_req.bootstrap_port}"
                ),
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
        except Exception as exc:
            if sender is not None:
                sender.clear()
            if metadata_index >= 0:
                self.req_to_metadata_buffer_idx_allocator.free(metadata_index)
            self._engine_paused = False
            logger.exception(
                "Failed to prepare decode migration rid=%s migration_id=%s",
                recv_req.rid,
                recv_req.migration_id,
            )
            return PrepareDecodeMigrationReqOutput(
                rid=recv_req.rid,
                migration_id=recv_req.migration_id,
                success=False,
                status="error",
                error=f"Failed to create migration transfer: {exc}",
            )

        record = DecodeMigrationTransfer(
            migration_id=recv_req.migration_id,
            req=req,
            sender=sender,
            metadata_buffer_index=metadata_index,
            bootstrap_room=room,
            committed_len=committed_len,
            pending_input_ids=frontier.pending_input_ids,
            created_at=time.monotonic(),
            transfer_end=committed_len,
        )
        self.decode_migration_transfers[recv_req.migration_id] = record
        self.decode_migration_by_rid[recv_req.rid] = recv_req.migration_id

        logger.info(
            "Prepared decode migration rid=%s migration_id=%s committed=%d "
            "logical=%d seen=%d room=%d",
            recv_req.rid,
            recv_req.migration_id,
            committed_len,
            frontier.logical_len,
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
        logical_ids = list(req.origin_input_ids) + list(req.output_ids)
        committed_input_ids = logical_ids[: record.committed_len]
        prompt_len = len(req.origin_input_ids)
        committed_output_count = max(0, record.committed_len - prompt_len)
        seen = min(max(0, recv_req.output_tokens_seen), len(req.output_ids))
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
            unforwarded_committed_output_ids=list(req.output_ids)[
                min(seen, committed_output_count) : committed_output_count
            ],
            prompt_len=prompt_len,
            committed_len=record.committed_len,
            logical_len=len(logical_ids),
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
            self.req_to_metadata_buffer_idx_allocator.free(
                record.metadata_buffer_index
            )
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
        # Before commit, the source remains authoritative and can resume safely.
        self._engine_paused = False

    def finalize_decode_migration(
        self: "Scheduler", recv_req: FinalizeDecodeMigrationReqInput
    ) -> FinalizeDecodeMigrationReqOutput:
        record = self.decode_migration_transfers.get(recv_req.migration_id)
        if record is None or record.req.rid != recv_req.rid:
            return FinalizeDecodeMigrationReqOutput(
                rid=recv_req.rid,
                migration_id=recv_req.migration_id,
                action=recv_req.action,
                success=False,
                transfer_status="unknown",
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
                error="Destination cannot commit before source transfer completes",
            )

        self._release_decode_migration_transport(record)
        self.decode_migration_transfers.pop(recv_req.migration_id, None)
        self.decode_migration_by_rid.pop(recv_req.rid, None)

        if recv_req.action == "resume":
            self._engine_paused = False
        else:
            # Reuse the normal scheduler finish path so all request/KV/cache
            # accounting remains centralized. The source stream is no longer
            # forwarded after commit, so its cancellation chunk is internal.
            record.req.to_finish = FINISH_ABORT()
            self._engine_paused = False

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
        )
