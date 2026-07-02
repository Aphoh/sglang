"""Request-local support for live decode-to-decode migration.

The destination reuses SGLang's disaggregated-decode receiver. The source
removes only the target request from scheduling, transfers its KV, and lets
normal overlap result processing discard the one stale source result that may
already be in flight.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Optional

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.base.conn import NIXL_LOW_LATENCY_SENDER
from sglang.srt.disaggregation.decode_migration_state import (
    AwaitingStaleDecodeResult,
    DecodeMigrationRegistry,
    DecodeMigrationState,
    DecodeMigrationTransfer,
    PreparedDecodeMigrationSource,
    build_decode_migration_frontier,
)
from sglang.srt.disaggregation.prefill import create_prefill_kv_manager
from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    KVClassType,
    build_state_transfer_indices,
    get_kv_class,
    poll_and_all_reduce_attn_cp_tp_group,
)
from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import (
    CancelDecodeMigrationReqInput,
    CancelDecodeMigrationReqOutput,
    PrepareDecodeMigrationReqInput,
    PrepareDecodeMigrationReqOutput,
    QuiesceDecodeMigrationReqInput,
    QuiesceDecodeMigrationReqOutput,
)
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.common import kv_to_page_indices, release_kv_cache

if TYPE_CHECKING:
    from sglang.srt.disaggregation.base import BaseKVManager, BaseKVSender
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)


class SchedulerDecodeMigrationMixin:
    """Scheduler control plane for source-side decode migration."""

    def init_decode_migration(self: "Scheduler") -> None:
        self.decode_migrations = DecodeMigrationRegistry()
        self._decode_migration_kv_manager: Optional[BaseKVManager] = None
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
                    nixl_transport_config=NIXL_LOW_LATENCY_SENDER,
                )
        return self._decode_migration_kv_manager

    def process_prebuilt_kv_receives(self: "Scheduler") -> None:
        """Advance prebuilt-KV handshakes outside dedicated decode mode."""
        if self.disaggregation_mode == DisaggregationMode.DECODE or not hasattr(
            self, "disagg_decode_prealloc_queue"
        ):
            return
        self.process_decode_queue()

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

        for record in self.decode_migrations.transfers():
            if recv_req.abort_all or record.req.rid.startswith(recv_req.rid):
                self._close_decode_migration(record)

        for prepared in self.decode_migrations.prepared_sources():
            if recv_req.abort_all or prepared.rid.startswith(recv_req.rid):
                self._discard_prepared_decode_migration(prepared)

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
    ) -> PrepareDecodeMigrationReqOutput:
        return PrepareDecodeMigrationReqOutput(
            rid=recv_req.rid,
            migration_id=recv_req.migration_id,
            success=False,
            status=status,
            source_dp_rank=self.ps.dp_rank or 0,
            error=error,
        )

    def _quiesce_failure(
        self: "Scheduler",
        recv_req: QuiesceDecodeMigrationReqInput,
        status: str,
        error: Optional[str] = None,
        **state,
    ) -> QuiesceDecodeMigrationReqOutput:
        return QuiesceDecodeMigrationReqOutput(
            rid=recv_req.rid,
            migration_id=recv_req.migration_id,
            success=False,
            status=status,
            source_dp_rank=self.ps.dp_rank or 0,
            error=error,
            **state,
        )

    def _park_decode_migration_req(self: "Scheduler", req: Req) -> None:
        if not self.detach_request_from_scheduling(req):
            raise RuntimeError("Request left the running decode batch before migration")

    def _clear_prepared_decode_migration(
        self: "Scheduler", migration_id: str, rid: str
    ) -> None:
        entry = self.decode_migrations.get(migration_id)
        if isinstance(entry, PreparedDecodeMigrationSource) and entry.rid == rid:
            self._discard_prepared_decode_migration(entry)

    def _discard_prepared_decode_migration(
        self: "Scheduler", prepared: PreparedDecodeMigrationSource
    ) -> None:
        pending = prepared.pending_quiesce
        if pending is not None:
            self._complete_pending_quiesce(
                prepared,
                self._quiesce_failure(
                    pending,
                    "finished",
                    "Migration source stopped before quiescence",
                    output_tokens_seen=pending.output_tokens_seen,
                ),
            )
        prepared.sender.clear()
        self.decode_migrations.discard(prepared.migration_id)

    def _complete_pending_quiesce(
        self: "Scheduler",
        prepared: PreparedDecodeMigrationSource,
        output: QuiesceDecodeMigrationReqOutput,
    ) -> None:
        pending = prepared.pending_quiesce
        if pending is None:
            return
        prepared.pending_quiesce = None
        self.ipc_channels.send_to_tokenizer.send_output(output, pending)

    def maybe_quiesce_decode_migration(self: "Scheduler", req: Req) -> bool:
        prepared = self.decode_migrations.get_for_rid(req.rid)
        if not isinstance(prepared, PreparedDecodeMigrationSource):
            return False
        control_req = prepared.pending_quiesce
        if req.finished() or req.to_finish is not None:
            if control_req is not None:
                output = self._quiesce_failure(
                    control_req,
                    "finished",
                    prompt_len=len(req.origin_input_ids),
                    logical_len=len(req.origin_input_ids) + len(req.output_ids),
                    output_tokens_seen=control_req.output_tokens_seen,
                )
                self._complete_pending_quiesce(prepared, output)
            self._clear_prepared_decode_migration(prepared.migration_id, req.rid)
            return False
        if control_req is None:
            return False

        output = self._quiesce_decode_migration_now(
            prepared,
            control_req,
            req=req,
        )
        self._complete_pending_quiesce(prepared, output)
        if not output.success:
            logger.error(
                "Failed to quiesce prepared decode migration rid=%s migration_id=%s "
                "status=%s error=%s",
                req.rid,
                prepared.migration_id,
                output.status,
                output.error,
            )
            return False
        return True

    def should_process_decode_result(self: "Scheduler", req: Req) -> bool:
        owner = self.decode_migrations.get_result_owner_for_req(req)
        if owner is None:
            return True
        if isinstance(owner, AwaitingStaleDecodeResult):
            self._release_decode_migration_source(owner.req)
            self.decode_migrations.discard(owner.migration_id)
        return False

    def _prepared_source_output(
        self: "Scheduler",
        recv_req: PrepareDecodeMigrationReqInput,
        prepared: PreparedDecodeMigrationSource,
    ) -> PrepareDecodeMigrationReqOutput:
        return PrepareDecodeMigrationReqOutput(
            rid=recv_req.rid,
            migration_id=prepared.migration_id,
            success=True,
            status="ready",
            source_dp_rank=self.ps.dp_rank or 0,
        )

    def _prepare_decode_migration_source(
        self: "Scheduler",
        recv_req: PrepareDecodeMigrationReqInput,
    ) -> PrepareDecodeMigrationReqOutput:
        try:
            sender = self._create_decode_migration_sender(recv_req)
        except Exception as exc:
            logger.exception(
                "Failed to prepare decode migration rid=%s migration_id=%s",
                recv_req.rid,
                recv_req.migration_id,
            )
            return self._prepare_failure(
                recv_req, "error", f"Failed to create migration sender: {exc}"
            )
        prepared = self.decode_migrations.prepare(recv_req, sender)
        logger.info(
            "Prepared decode migration source rid=%s migration_id=%s room=%d",
            prepared.rid,
            prepared.migration_id,
            prepared.request.bootstrap_room,
        )
        return self._prepared_source_output(recv_req, prepared)

    def _create_decode_migration_sender(
        self: "Scheduler", recv_req: PrepareDecodeMigrationReqInput
    ) -> "BaseKVSender":
        if recv_req.bootstrap_room <= 0:
            raise ValueError("bootstrap_room must be an opaque positive integer")
        manager = self._get_decode_migration_kv_manager()
        sender_class = get_kv_class(self.transfer_backend, KVClassType.SENDER)
        return sender_class(
            mgr=manager,
            bootstrap_addr=f"{recv_req.bootstrap_host}:{recv_req.bootstrap_port}",
            bootstrap_room=recv_req.bootstrap_room,
            dest_tp_ranks=[self.ps.tp_rank],
            pp_rank=self.ps.pp_rank,
        )

    def _release_decode_migration_source(self: "Scheduler", req: Req) -> None:
        release_kv_cache(req, self.tree_cache, is_insert=False)
        # A parked request no longer belongs to running_batch, so the
        # scheduler can cache it as full while this transfer still owns
        # the request-pool slot. Releasing that slot must reopen admission.
        self.running_batch.batch_is_full = False

    def _close_decode_migration(
        self: "Scheduler",
        record: DecodeMigrationTransfer,
    ) -> None:
        self._release_decode_migration_transport(record)
        if self._has_queued_decode_migration_result(record.req):
            self.decode_migrations.await_stale_result(record)
            return
        self._release_decode_migration_source(record.req)
        self.decode_migrations.discard(record.migration_id)

    def _has_queued_decode_migration_result(self: "Scheduler", req: Req) -> bool:
        return self.enable_overlap and any(
            any(candidate is req for candidate in batch.reqs)
            for batch, _ in self.result_queue
        )

    def prepare_decode_migration(
        self: "Scheduler", recv_req: PrepareDecodeMigrationReqInput
    ) -> PrepareDecodeMigrationReqOutput:
        if not self.server_args.enable_decode_migration:
            return self._prepare_failure(
                recv_req,
                "error",
                "Decode migration requires --enable-decode-migration",
            )
        if not self.spec_algorithm.is_none():
            return self._prepare_failure(
                recv_req,
                "error",
                "Decode migration does not support speculative decoding",
            )

        existing = self.decode_migrations.get_for_rid(recv_req.rid)
        if isinstance(existing, AwaitingStaleDecodeResult):
            return self._prepare_failure(
                recv_req,
                "finished",
                "Migration source is no longer active",
            )
        if isinstance(existing, DecodeMigrationTransfer):
            if existing.migration_id == recv_req.migration_id:
                return PrepareDecodeMigrationReqOutput(
                    rid=recv_req.rid,
                    migration_id=recv_req.migration_id,
                    success=True,
                    status="ready",
                    source_dp_rank=self.ps.dp_rank or 0,
                )
            return self._prepare_failure(
                recv_req,
                "busy",
                f"Request already has active migration {existing.migration_id}",
            )

        if (
            isinstance(existing, PreparedDecodeMigrationSource)
            and existing.migration_id != recv_req.migration_id
        ):
            return self._prepare_failure(
                recv_req,
                "busy",
                f"Request already has prepared migration {existing.migration_id}",
            )

        same_id = self.decode_migrations.get(recv_req.migration_id)
        if same_id is not None and same_id.rid != recv_req.rid:
            return self._prepare_failure(
                recv_req,
                "busy",
                f"Migration id already belongs to request {same_id.rid}",
            )

        if isinstance(existing, PreparedDecodeMigrationSource):
            return self._prepared_source_output(recv_req, existing)

        req = self._find_decode_migration_req(recv_req.rid)
        if req is not None and (req.finished() or req.to_finish is not None):
            return self._prepare_failure(recv_req, "finished")
        return self._prepare_decode_migration_source(recv_req)

    def quiesce_decode_migration(
        self: "Scheduler",
        recv_req: QuiesceDecodeMigrationReqInput,
    ) -> QuiesceDecodeMigrationReqOutput:
        if not self.server_args.enable_decode_migration:
            return self._quiesce_failure(
                recv_req,
                "error",
                "Decode migration requires --enable-decode-migration",
            )

        existing = self.decode_migrations.get_for_rid(recv_req.rid)
        if isinstance(existing, AwaitingStaleDecodeResult):
            return self._quiesce_failure(
                recv_req, "finished", "Migration source is no longer active"
            )
        if isinstance(existing, DecodeMigrationTransfer):
            if existing.migration_id != recv_req.migration_id:
                return self._quiesce_failure(
                    recv_req,
                    "busy",
                    f"Request already has active migration {existing.migration_id}",
                )
            existing.output_tokens_seen = min(
                max(existing.output_tokens_seen, recv_req.output_tokens_seen),
                max(0, existing.logical_len - len(existing.req.origin_input_ids)),
            )
            return self._quiesced_output(recv_req, existing)

        if not isinstance(existing, PreparedDecodeMigrationSource):
            return self._quiesce_failure(
                recv_req, "not_found", "Migration source has not been prepared"
            )
        if existing.migration_id != recv_req.migration_id:
            return self._quiesce_failure(
                recv_req,
                "busy",
                f"Request already has prepared migration {existing.migration_id}",
            )

        req = self._find_decode_migration_req(recv_req.rid)
        if req is None or req.finished() or req.to_finish is not None:
            self._discard_prepared_decode_migration(existing)
            return self._quiesce_failure(recv_req, "finished")

        # Under overlap scheduling, a launched result may have committed KV but
        # not yet updated output_ids. Finish normal result processing before
        # detaching the request so the exported frontier remains consistent.
        if self._has_queued_decode_migration_result(req):
            existing.pending_quiesce = recv_req
            return QuiesceDecodeMigrationReqOutput(
                rid=recv_req.rid,
                migration_id=recv_req.migration_id,
                success=True,
                status="quiescing",
                output_tokens_seen=recv_req.output_tokens_seen,
                source_dp_rank=self.ps.dp_rank or 0,
            )

        return self._quiesce_decode_migration_now(existing, recv_req, req=req)

    def cancel_decode_migration(
        self: "Scheduler", recv_req: CancelDecodeMigrationReqInput
    ) -> CancelDecodeMigrationReqOutput:
        entry = self.decode_migrations.get(recv_req.migration_id)
        rank = self.ps.dp_rank or 0
        if entry is None:
            return CancelDecodeMigrationReqOutput(
                rid=recv_req.rid,
                migration_id=recv_req.migration_id,
                success=True,
                status="not_found",
                source_dp_rank=rank,
            )
        if entry.rid != recv_req.rid:
            return CancelDecodeMigrationReqOutput(
                rid=recv_req.rid,
                migration_id=recv_req.migration_id,
                success=False,
                status="error",
                source_dp_rank=rank,
                error="Migration id belongs to another request",
            )
        if isinstance(entry, (DecodeMigrationTransfer, AwaitingStaleDecodeResult)):
            return CancelDecodeMigrationReqOutput(
                rid=recv_req.rid,
                migration_id=recv_req.migration_id,
                success=False,
                status="quiesced",
                source_dp_rank=rank,
                error="Source quiescence is irreversible",
            )
        self._discard_prepared_decode_migration(entry)
        return CancelDecodeMigrationReqOutput(
            rid=recv_req.rid,
            migration_id=recv_req.migration_id,
            success=True,
            status="cancelled",
            source_dp_rank=rank,
        )

    def _quiesce_decode_migration_now(
        self: "Scheduler",
        prepared: PreparedDecodeMigrationSource,
        recv_req: QuiesceDecodeMigrationReqInput,
        *,
        req: Optional[Req] = None,
    ) -> QuiesceDecodeMigrationReqOutput:
        if req is None:
            req = self._find_decode_migration_req(recv_req.rid)
        if req is None:
            return self._quiesce_failure(
                recv_req, "not_found", "Request is not in the running decode batch"
            )
        if req.finished() or req.to_finish is not None:
            return self._quiesce_failure(
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
        if recv_req.output_tokens_seen > 0:
            target_logical_len = len(prompt_ids) + recv_req.output_tokens_seen
            if (
                target_logical_len > actual_logical_len
                or target_logical_len - 1 > committed_len
            ):
                return self._quiesce_failure(
                    recv_req,
                    "error",
                    "Source has not reached the acknowledged output frontier",
                    prompt_len=len(prompt_ids),
                    committed_len=committed_len,
                    logical_len=actual_logical_len,
                    output_tokens_seen=recv_req.output_tokens_seen,
                )
            output_ids = output_ids[: recv_req.output_tokens_seen]
            committed_len = target_logical_len - 1
        try:
            frontier = build_decode_migration_frontier(
                prompt_ids,
                output_ids,
                committed_len,
                recv_req.output_tokens_seen,
            )
        except ValueError as exc:
            return self._quiesce_failure(
                recv_req,
                "error",
                str(exc),
                prompt_len=len(prompt_ids),
                committed_len=committed_len,
                logical_len=len(prompt_ids) + len(output_ids),
                output_tokens_seen=recv_req.output_tokens_seen,
            )

        if self.req_to_metadata_buffer_idx_allocator.available_size() <= 0:
            return self._quiesce_failure(
                recv_req, "error", "No metadata buffer is available for migration"
            )
        room = prepared.request.bootstrap_room
        if room <= 0:
            return self._quiesce_failure(
                recv_req,
                "error",
                "bootstrap_room must be an opaque positive integer",
            )

        metadata_index = -1
        sender = prepared.sender
        try:
            allocated_index = self.req_to_metadata_buffer_idx_allocator.alloc()
            assert allocated_index is not None
            metadata_index = allocated_index
            committed_output_tokens = max(
                0, frontier.committed_len - frontier.prompt_len
            )
            max_new_tokens = req.sampling_params.max_new_tokens
            min_new_tokens = req.sampling_params.min_new_tokens
            self.disagg_metadata_buffers.set_decode_migration_frontier(
                metadata_index,
                committed_input_ids=frontier.committed_input_ids,
                pending_input_id=frontier.pending_input_id,
                prompt_len=frontier.prompt_len,
                logical_len=frontier.logical_len,
                output_tokens_seen=frontier.output_tokens_seen,
                max_new_tokens=(
                    max(1, max_new_tokens - committed_output_tokens)
                    if max_new_tokens is not None
                    else None
                ),
                min_new_tokens=(
                    max(0, min_new_tokens - committed_output_tokens)
                    if min_new_tokens is not None
                    else None
                ),
            )
            sender.set_aux_transfer_lens(
                {
                    0: self.disagg_metadata_buffers.decode_migration_frontier_nbytes(
                        frontier.committed_len
                    )
                }
            )
            self.disagg_metadata_buffers.cached_tokens[metadata_index].zero_()
            self.disagg_metadata_buffers.bootstrap_room[metadata_index][0] = room
            self._park_decode_migration_req(req)
        except Exception as exc:
            sender.clear()
            self.decode_migrations.discard(prepared.migration_id)
            if metadata_index >= 0:
                self.req_to_metadata_buffer_idx_allocator.free(metadata_index)
            if all(candidate is not req for candidate in self.running_batch.reqs):
                release_kv_cache(req, self.tree_cache, is_insert=False)
            logger.exception(
                "Failed to quiesce decode migration rid=%s migration_id=%s",
                recv_req.rid,
                recv_req.migration_id,
            )
            return self._quiesce_failure(
                recv_req, "error", f"Failed to create migration transfer: {exc}"
            )

        record = DecodeMigrationTransfer(
            migration_id=recv_req.migration_id,
            req=req,
            sender=sender,
            metadata_buffer_index=metadata_index,
            committed_len=committed_len,
            logical_len=frontier.logical_len,
            output_tokens_seen=frontier.output_tokens_seen,
            created_at=time.monotonic(),
        )
        self.decode_migrations.activate(record)

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
        return QuiesceDecodeMigrationReqOutput(
            rid=recv_req.rid,
            migration_id=recv_req.migration_id,
            success=True,
            status="quiesced",
            unforwarded_committed_output_ids=(
                frontier.unforwarded_committed_output_ids
            ),
            prompt_len=frontier.prompt_len,
            committed_len=committed_len,
            logical_len=frontier.logical_len,
            output_tokens_seen=frontier.output_tokens_seen,
            source_dp_rank=self.ps.dp_rank or 0,
        )

    def _quiesced_output(
        self: "Scheduler",
        recv_req: QuiesceDecodeMigrationReqInput,
        record: DecodeMigrationTransfer,
    ) -> QuiesceDecodeMigrationReqOutput:
        req = record.req
        prompt_len = len(req.origin_input_ids)
        committed_output_count = max(0, record.committed_len - prompt_len)
        output_ids = list(req.output_ids)[: record.logical_len - prompt_len]
        seen = record.output_tokens_seen
        return QuiesceDecodeMigrationReqOutput(
            rid=recv_req.rid,
            migration_id=recv_req.migration_id,
            success=True,
            status="quiesced",
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
        if not self.decode_migrations.has_active_transfers():
            return

        records = list(self.decode_migrations.transfers())
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
                record.state = DecodeMigrationState.BOOTSTRAPPING
                continue
            if (
                poll == KVPoll.WaitingForInput
                and record.state == DecodeMigrationState.BOOTSTRAPPING
            ):
                decode_prefix_len = record.sender.pop_decode_prefix_len()
                if decode_prefix_len < 0 or decode_prefix_len > record.committed_len:
                    self._fail_decode_migration(
                        record,
                        f"Invalid destination prefix length {decode_prefix_len}",
                    )
                    continue

                token_to_kv_pool = self.token_to_kv_pool_allocator.get_kvcache()
                kv_indices = self.req_to_token_pool.req_to_token[
                    record.req.req_pool_idx,
                    decode_prefix_len : record.committed_len,
                ]
                page_indices = kv_to_page_indices(
                    kv_indices.cpu().numpy(), token_to_kv_pool.page_size
                )
                state_indices = build_state_transfer_indices(
                    state_types=(
                        self._get_decode_migration_kv_manager().kv_args.state_types
                    ),
                    req_to_token_pool=self.req_to_token_pool,
                    token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                    req_pool_idx=record.req.req_pool_idx,
                    seq_len=record.committed_len,
                    sliding_window_size=self.sliding_window_size,
                    dsa_page_size=token_to_kv_pool.page_size,
                )
                record.sender.init(len(page_indices), record.metadata_buffer_index)
                record.sender.send(page_indices, state_indices)
                record.state = DecodeMigrationState.TRANSFERRING
                logger.info(
                    "Started decode migration transfer rid=%s migration_id=%s "
                    "range=[%d,%d) pages=%d",
                    record.req.rid,
                    record.migration_id,
                    decode_prefix_len,
                    record.committed_len,
                    len(page_indices),
                )
                continue
            if poll in (KVPoll.WaitingForInput, KVPoll.Transferring):
                record.state = DecodeMigrationState.TRANSFERRING
                continue
            if poll == KVPoll.Success:
                logger.info(
                    "Decode migration transfer completed rid=%s migration_id=%s",
                    record.req.rid,
                    record.migration_id,
                )
                self._close_decode_migration(record)
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
        record.sender.clear()
        self.disagg_metadata_buffers.bootstrap_room[
            record.metadata_buffer_index
        ].zero_()
        self.req_to_metadata_buffer_idx_allocator.free(record.metadata_buffer_index)

    def _fail_decode_migration(
        self: "Scheduler", record: DecodeMigrationTransfer, error: str
    ) -> None:
        logger.error(
            "Decode migration failed rid=%s migration_id=%s error=%s",
            record.req.rid,
            record.migration_id,
            error,
        )
        self._close_decode_migration(record)
