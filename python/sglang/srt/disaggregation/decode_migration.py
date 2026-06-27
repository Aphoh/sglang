"""Request-local support for live decode-to-decode migration.

The destination reuses SGLang's disaggregated-decode receiver. The source
removes only the target request from scheduling, retains its KV until one-way
finalization, and lets normal overlap result processing discard the one stale
source result that may already be in flight.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import TYPE_CHECKING, Literal, Optional

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
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler_components.result_disposition import (
    ResultDisposition,
)
from sglang.srt.mem_cache.common import kv_to_page_indices, release_kv_cache

if TYPE_CHECKING:
    from sglang.srt.disaggregation.base import BaseKVManager, BaseKVSender
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)


class DecodeMigrationState(str, Enum):
    BOOTSTRAPPING = "bootstrapping"
    TRANSFERRING = "transferring"
    TRANSFERRED = "transferred"
    AWAITING_STALE_RESULT = "awaiting_stale_result"


@dataclass
class ArmedDecodeMigration:
    request: PrepareDecodeMigrationReqInput

    @property
    def migration_id(self) -> str:
        return self.request.migration_id

    @property
    def rid(self) -> str:
        return self.request.rid


@dataclass(frozen=True)
class DecodeMigrationFinalization:
    action: Literal["commit", "cancel"]
    transfer_status: Literal["bootstrapping", "transferring", "transferred"]


@dataclass
class DecodeMigrationTransfer:
    migration_id: str
    req: Req
    sender: "BaseKVSender | None"
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
    state: DecodeMigrationState = DecodeMigrationState.BOOTSTRAPPING
    finalization: DecodeMigrationFinalization | None = None

    @property
    def rid(self) -> str:
        return self.req.rid


@dataclass
class DecodeMigrationRegistry:
    """Own source migration lifecycle state indexed by migration and request."""

    _by_migration_id: dict[str, ArmedDecodeMigration | DecodeMigrationTransfer] = field(
        default_factory=dict
    )
    _migration_id_by_rid: dict[str, str] = field(default_factory=dict)

    def get(
        self, migration_id: str
    ) -> ArmedDecodeMigration | DecodeMigrationTransfer | None:
        return self._by_migration_id.get(migration_id)

    def get_for_rid(
        self, rid: str
    ) -> ArmedDecodeMigration | DecodeMigrationTransfer | None:
        migration_id = self._migration_id_by_rid.get(rid)
        return self.get(migration_id) if migration_id is not None else None

    def arm(self, request: PrepareDecodeMigrationReqInput) -> ArmedDecodeMigration:
        existing = self.get(request.migration_id)
        if existing is not None:
            assert existing.rid == request.rid
            assert isinstance(existing, ArmedDecodeMigration)
            return existing
        assert self.get_for_rid(request.rid) is None
        arm = ArmedDecodeMigration(request)
        self._by_migration_id[arm.migration_id] = arm
        self._migration_id_by_rid[arm.rid] = arm.migration_id
        return arm

    def activate(self, record: DecodeMigrationTransfer) -> None:
        existing = self.get(record.migration_id)
        if existing is not None:
            assert existing.rid == record.req.rid
        by_rid = self.get_for_rid(record.req.rid)
        if by_rid is not None:
            assert by_rid.migration_id == record.migration_id
        self._by_migration_id[record.migration_id] = record
        self._migration_id_by_rid[record.req.rid] = record.migration_id

    def discard(
        self, migration_id: str
    ) -> ArmedDecodeMigration | DecodeMigrationTransfer | None:
        entry = self._by_migration_id.pop(migration_id, None)
        if (
            entry is not None
            and self._migration_id_by_rid.get(entry.rid) == migration_id
        ):
            self._migration_id_by_rid.pop(entry.rid, None)
        return entry

    def arms(self) -> tuple[ArmedDecodeMigration, ...]:
        return tuple(
            entry
            for entry in self._by_migration_id.values()
            if isinstance(entry, ArmedDecodeMigration)
        )

    def transfers(self) -> tuple[DecodeMigrationTransfer, ...]:
        return tuple(
            entry
            for entry in self._by_migration_id.values()
            if isinstance(entry, DecodeMigrationTransfer)
        )

    def has_active_transfers(self) -> bool:
        return any(
            isinstance(entry, DecodeMigrationTransfer)
            and entry.state != DecodeMigrationState.AWAITING_STALE_RESULT
            for entry in self._by_migration_id.values()
        )

    def get_transfer_for_req(self, req: Req) -> DecodeMigrationTransfer | None:
        entry = self.get_for_rid(req.rid)
        return (
            entry
            if isinstance(entry, DecodeMigrationTransfer) and entry.req is req
            else None
        )


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

        for arm in self.decode_migrations.arms():
            if recv_req.abort_all or arm.rid.startswith(recv_req.rid):
                self.decode_migrations.discard(arm.migration_id)

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

    def _park_decode_migration_req(self: "Scheduler", req: Req) -> None:
        if not self.detach_request_from_scheduling(req):
            raise RuntimeError("Request left the running decode batch before migration")

    def _clear_decode_migration_arm(
        self: "Scheduler", migration_id: str, rid: str
    ) -> None:
        entry = self.decode_migrations.get(migration_id)
        if isinstance(entry, ArmedDecodeMigration) and entry.rid == rid:
            self.decode_migrations.discard(migration_id)

    def maybe_park_decode_migration_at_boundary(self: "Scheduler", req: Req) -> bool:
        arm = self.decode_migrations.get_for_rid(req.rid)
        if not isinstance(arm, ArmedDecodeMigration):
            return False
        if req.finished() or req.to_finish is not None:
            self._clear_decode_migration_arm(arm.migration_id, req.rid)
            return False
        target = arm.request.target_sequence_length
        logical_len = len(req.origin_input_ids) + len(req.output_ids)
        if target is None or logical_len < target:
            return False

        self.decode_migrations.discard(arm.migration_id)
        output = self._prepare_decode_migration_now(arm.request, req=req)
        if not output.success:
            logger.error(
                "Failed to park armed decode migration rid=%s migration_id=%s "
                "status=%s error=%s",
                req.rid,
                arm.migration_id,
                output.status,
                output.error,
            )
            return False
        return True

    def get_decode_migration_result_disposition(
        self: "Scheduler", req: Req
    ) -> ResultDisposition:
        record = self.decode_migrations.get_transfer_for_req(req)
        if record is None:
            return ResultDisposition.PROCESS
        if record.state == DecodeMigrationState.AWAITING_STALE_RESULT:
            self._release_decode_migration_source(record)
            self.decode_migrations.discard(record.migration_id)
        return ResultDisposition.DISCARD

    def _armed_output(
        self: "Scheduler",
        recv_req: PrepareDecodeMigrationReqInput,
        arm: ArmedDecodeMigration,
        logical_len: int,
    ) -> PrepareDecodeMigrationReqOutput:
        return PrepareDecodeMigrationReqOutput(
            rid=recv_req.rid,
            migration_id=arm.migration_id,
            success=True,
            status="armed",
            bootstrap_host=arm.request.bootstrap_host,
            bootstrap_port=arm.request.bootstrap_port,
            bootstrap_room=arm.request.bootstrap_room,
            logical_len=logical_len,
            output_tokens_seen=arm.request.output_tokens_seen,
            source_dp_rank=self.ps.dp_rank or 0,
        )

    def _arm_decode_migration(
        self: "Scheduler",
        recv_req: PrepareDecodeMigrationReqInput,
        logical_len: int,
    ) -> PrepareDecodeMigrationReqOutput:
        arm = self.decode_migrations.arm(recv_req)
        logger.info(
            "Armed decode migration rid=%s migration_id=%s target=%d current=%d",
            arm.rid,
            arm.migration_id,
            arm.request.target_sequence_length,
            logical_len,
        )
        return self._armed_output(recv_req, arm, logical_len)

    def _release_decode_migration_source(
        self: "Scheduler", record: DecodeMigrationTransfer
    ) -> None:
        release_kv_cache(record.req, self.tree_cache, is_insert=False)
        # A parked request no longer belongs to running_batch, so the
        # scheduler can cache it as full while this transfer still owns
        # the request-pool slot. Releasing that slot must reopen admission.
        self.running_batch.batch_is_full = False

    def _close_decode_migration(
        self: "Scheduler",
        record: DecodeMigrationTransfer,
        finalization: DecodeMigrationFinalization | None = None,
    ) -> None:
        if record.state == DecodeMigrationState.AWAITING_STALE_RESULT:
            return
        self._release_decode_migration_transport(record)
        if self._has_queued_decode_migration_result(record.req):
            record.state = DecodeMigrationState.AWAITING_STALE_RESULT
            record.finalization = finalization
            return
        self._release_decode_migration_source(record)
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
        if isinstance(existing, DecodeMigrationTransfer):
            if existing.state == DecodeMigrationState.AWAITING_STALE_RESULT:
                return self._prepare_failure(
                    recv_req,
                    "finished",
                    "Migration source is no longer active",
                )
            if existing.migration_id == recv_req.migration_id:
                return self._prepared_output(recv_req, existing)
            return self._prepare_failure(
                recv_req,
                "busy",
                f"Request already has active migration {existing.migration_id}",
            )

        if (
            isinstance(existing, ArmedDecodeMigration)
            and existing.migration_id != recv_req.migration_id
        ):
            return self._prepare_failure(
                recv_req,
                "busy",
                f"Request already has armed migration {existing.migration_id}",
            )

        same_id = self.decode_migrations.get(recv_req.migration_id)
        if same_id is not None and same_id.rid != recv_req.rid:
            return self._prepare_failure(
                recv_req,
                "busy",
                f"Migration id already belongs to request {same_id.rid}",
            )

        req = self._find_decode_migration_req(recv_req.rid)
        target = recv_req.target_sequence_length
        logical_len = (
            len(req.origin_input_ids) + len(req.output_ids) if req is not None else 0
        )
        if target is not None and target <= 0:
            return self._prepare_failure(
                recv_req, "error", "target_sequence_length must be positive"
            )

        if isinstance(existing, ArmedDecodeMigration):
            return self._armed_output(recv_req, existing, logical_len)

        # The overlap loop accepts controls before it consumes the previous
        # decode result. That result already advances KV but has not yet added
        # its sampled token to output_ids, so defer direct preparation to the
        # next normal result frontier.
        if req is not None and self._has_queued_decode_migration_result(req):
            recv_req = replace(
                recv_req,
                target_sequence_length=max(logical_len + 1, target or 0),
            )
            target = recv_req.target_sequence_length

        if target is not None and target > logical_len:
            return self._arm_decode_migration(recv_req, logical_len)

        return self._prepare_decode_migration_now(recv_req, req=req)

    def _prepare_decode_migration_now(
        self: "Scheduler",
        recv_req: PrepareDecodeMigrationReqInput,
        *,
        req: Optional[Req] = None,
    ) -> PrepareDecodeMigrationReqOutput:
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
        if not self.decode_migrations.has_active_transfers():
            return

        records = [
            record
            for record in self.decode_migrations.transfers()
            if record.state
            in (
                DecodeMigrationState.BOOTSTRAPPING,
                DecodeMigrationState.TRANSFERRING,
            )
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
                record.state = DecodeMigrationState.BOOTSTRAPPING
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
                record.state = DecodeMigrationState.TRANSFERRING
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
                record.state = DecodeMigrationState.TRANSFERRING
                continue
            if poll == KVPoll.Success:
                record.state = DecodeMigrationState.TRANSFERRED
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
            record.sender = None
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
        self._close_decode_migration(record)

    def finalize_decode_migration(
        self: "Scheduler", recv_req: FinalizeDecodeMigrationReqInput
    ) -> FinalizeDecodeMigrationReqOutput:
        entry = self.decode_migrations.get(recv_req.migration_id)
        if isinstance(entry, ArmedDecodeMigration) and entry.rid == recv_req.rid:
            if recv_req.action == "cancel":
                self.decode_migrations.discard(recv_req.migration_id)
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
        if not isinstance(entry, DecodeMigrationTransfer) or entry.rid != recv_req.rid:
            return FinalizeDecodeMigrationReqOutput(
                rid=recv_req.rid,
                migration_id=recv_req.migration_id,
                action=recv_req.action,
                success=False,
                transfer_status="unknown",
                source_dp_rank=self.ps.dp_rank or 0,
                error="Migration generation is not active",
            )

        record = entry
        if record.state == DecodeMigrationState.AWAITING_STALE_RESULT:
            finalization = record.finalization
            if finalization is None:
                return FinalizeDecodeMigrationReqOutput(
                    rid=recv_req.rid,
                    migration_id=recv_req.migration_id,
                    action=recv_req.action,
                    success=False,
                    transfer_status="unknown",
                    source_dp_rank=self.ps.dp_rank or 0,
                    error="Migration source cleanup is still in progress",
                )
            if recv_req.action != finalization.action:
                return FinalizeDecodeMigrationReqOutput(
                    rid=recv_req.rid,
                    migration_id=recv_req.migration_id,
                    action=recv_req.action,
                    success=False,
                    transfer_status=finalization.transfer_status,
                    source_dp_rank=self.ps.dp_rank or 0,
                    error=f"Migration was already finalized with {finalization.action}",
                )
            return FinalizeDecodeMigrationReqOutput(
                rid=recv_req.rid,
                migration_id=recv_req.migration_id,
                action=recv_req.action,
                success=True,
                transfer_status=finalization.transfer_status,
                source_dp_rank=self.ps.dp_rank or 0,
            )

        status = record.state
        if recv_req.action == "commit" and status != DecodeMigrationState.TRANSFERRED:
            return FinalizeDecodeMigrationReqOutput(
                rid=recv_req.rid,
                migration_id=recv_req.migration_id,
                action=recv_req.action,
                success=False,
                transfer_status=status.value,
                source_dp_rank=self.ps.dp_rank or 0,
                error="Destination cannot commit before source transfer completes",
            )
        self._close_decode_migration(
            record,
            DecodeMigrationFinalization(
                action=recv_req.action,
                transfer_status=status.value,
            ),
        )

        logger.info(
            "Finalized decode migration rid=%s migration_id=%s action=%s status=%s",
            recv_req.rid,
            recv_req.migration_id,
            recv_req.action,
            status.value,
        )
        return FinalizeDecodeMigrationReqOutput(
            rid=recv_req.rid,
            migration_id=recv_req.migration_id,
            action=recv_req.action,
            success=True,
            transfer_status=status.value,
            source_dp_rank=self.ps.dp_rank or 0,
        )
