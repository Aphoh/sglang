"""
Migration support for decode workers.

This module provides functionality to migrate an in-flight request's KV cache
from one decode worker to another, reusing the prefill->decode KV transfer protocol.

Life cycle of a migration request:
1. Receive MigrateReq with request ID and bootstrap info
2. Find the request in running_batch
3. Remove request from running_batch (stop generation)
4. Setup KV sender using the same mechanism as prefill
5. Transfer KV cache to destination decode worker
6. Deallocate KV cache when transfer completes
"""

from __future__ import annotations

import logging
import os
import random
import time
from http import HTTPStatus
from typing import TYPE_CHECKING, List, Optional, Type

import torch

from sglang.srt.disaggregation.base import BaseKVManager, BaseKVSender, KVPoll
from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    KVClassType,
    TransferBackend,
    get_kv_class,
    is_mla_backend,
    kv_to_page_indices,
    kv_to_page_num,
    poll_and_all_reduce,
    prepare_abort,
)
from sglang.srt.managers.io_struct import MigrateReq, MigrateReqOutput
from sglang.srt.managers.schedule_batch import (
    FINISH_LENGTH,
    Req,
    RequestStage,
)
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.utils import get_int_env_var
from sglang.srt.mem_cache.memory_pool import (
    HybridLinearKVPool,
    NSATokenToKVPool,
    SWAKVPool,
)

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)
MIGRATION_WAIT_TIMEOUT_S = get_int_env_var(
    "SGLANG_DISAGGREGATION_WAITING_TIMEOUT", 300
)


class SchedulerMigrationMixin:
    """
    Mixin for Scheduler to handle KV cache migration from decode workers.
    
    This reuses the same KV transfer infrastructure as prefill->decode transfer,
    allowing a decode worker to send its KV cache to another decode worker.
    """

    def init_migration(self: "Scheduler") -> None:
        """Initialize migration-related data structures."""
        self.disagg_migration_inflight_queue: List[Req] = []
        self._migration_kv_manager: Optional[BaseKVManager] = None

    def _handle_migration_failure(
        self: "Scheduler", req: Req, error_message: str
    ) -> None:
        logger.error(error_message)
        if getattr(req, "disagg_kv_sender", None) is not None:
            try:
                req.disagg_kv_sender.failure_exception()
            except Exception as e:
                error_message = f"{error_message} with exception {e}"
                logger.error(error_message)
            if hasattr(req.disagg_kv_sender, "clear"):
                req.disagg_kv_sender.clear()
        release_kv_cache(req, self.tree_cache)
        if (
            hasattr(req, "metadata_buffer_index")
            and req.metadata_buffer_index is not None
            and req.metadata_buffer_index >= 0
        ):
            self.req_to_metadata_buffer_idx_allocator.free(req.metadata_buffer_index)
            req.metadata_buffer_index = -1
        prepare_abort(
            req, error_message, status_code=HTTPStatus.INTERNAL_SERVER_ERROR
        )

    def _get_migration_kv_manager(self: "Scheduler") -> BaseKVManager:
        """Get or create the KV manager for migration transfers.
        
        Lazily initializes the KV manager on first use.
        Also registers this decode worker's parallel info with its own bootstrap server
        so that the destination decode worker can fetch it during KV receive.
        """
        if self._migration_kv_manager is not None:
            return self._migration_kv_manager

        transfer_backend = TransferBackend(
            self.server_args.disaggregation_transfer_backend
        )
        token_to_kv_pool = self.token_to_kv_pool_allocator.get_kvcache()

        kv_args_class = get_kv_class(transfer_backend, KVClassType.KVARGS)
        kv_args = kv_args_class()
        kv_args.engine_rank = self.tp_rank
        kv_args.pp_rank = 0  # Migration is within a single PP stage
        kv_args.system_dp_rank = self.dp_rank
        kv_args.decode_tp_size = self.tp_size
        kv_args.prefill_pp_size = 1
        kv_args.prefill_start_layer = token_to_kv_pool.start_layer

        kv_data_ptrs, kv_data_lens, kv_item_lens = (
            token_to_kv_pool.get_contiguous_buf_infos()
        )

        kv_args.kv_data_ptrs = kv_data_ptrs
        kv_args.kv_data_lens = kv_data_lens
        kv_args.kv_item_lens = kv_item_lens

        if not is_mla_backend(token_to_kv_pool):
            kv_args.kv_head_num = token_to_kv_pool.head_num
        kv_args.page_size = token_to_kv_pool.page_size

        # Handle auxiliary metadata buffers if available
        if (
            hasattr(self, "disagg_metadata_buffers")
            and self.disagg_metadata_buffers is not None
        ):
            kv_args.aux_data_ptrs, kv_args.aux_data_lens, kv_args.aux_item_lens = (
                self.disagg_metadata_buffers.get_buf_infos()
            )
        else:
            kv_args.aux_data_ptrs = []
            kv_args.aux_data_lens = []
            kv_args.aux_item_lens = []

        kv_args.ib_device = self.server_args.disaggregation_ib_device
        kv_args.gpu_id = self.gpu_id

        # Handle state buffers for hybrid models
        if hasattr(token_to_kv_pool, "get_state_buf_infos"):
            state_data_ptrs, state_data_lens, state_item_lens = (
                token_to_kv_pool.get_state_buf_infos()
            )
            kv_args.state_data_ptrs = state_data_ptrs
            kv_args.state_data_lens = state_data_lens
            kv_args.state_item_lens = state_item_lens

            if isinstance(token_to_kv_pool, SWAKVPool):
                kv_args.state_type = "swa"
            elif isinstance(token_to_kv_pool, HybridLinearKVPool):
                kv_args.state_type = "mamba"
            elif isinstance(token_to_kv_pool, NSATokenToKVPool):
                kv_args.state_type = "nsa"
            else:
                kv_args.state_type = "none"
        else:
            kv_args.state_data_ptrs = []
            kv_args.state_data_lens = []
            kv_args.state_item_lens = []
            kv_args.state_type = "none"

        kv_manager_class: Type[BaseKVManager] = get_kv_class(
            transfer_backend, KVClassType.MANAGER
        )
        # Use PREFILL mode since we're sending KV cache (like prefill does)
        self._migration_kv_manager = kv_manager_class(
            kv_args,
            DisaggregationMode.PREFILL,
            self.server_args,
            is_mla_backend(token_to_kv_pool),
        )
        return self._migration_kv_manager

    def process_migrate_request(self: "Scheduler", recv_req: MigrateReq) -> None:
        """Process a migration request.
        
        Finds the request in running_batch, removes it from active processing,
        sets up KV sender, and adds it to the migration inflight queue.
        Also sends a MigrateReqOutput back to tokenizer with the pending output_ids.
        
        Args:
            recv_req: The migration request with rid and bootstrap info.
        """
        rid = recv_req.rid
        bootstrap_host = recv_req.bootstrap_host
        bootstrap_port = recv_req.bootstrap_port
        tokens_seen = recv_req.tokens_seen

        # Generate bootstrap_room that encodes this worker's DP rank.
        # The receiver will use: bootstrap_room % dp_size == src_dp_rank
        # This allows the receiver to target the correct source DP rank without
        # needing a separate data_parallel_rank field (which would also affect routing).
        dp_rank = getattr(self, "dp_rank", 0) or 0
        dp_size = getattr(self.server_args, "dp_size", 1) or 1
        base = random.randint(0, 2**62)
        bootstrap_room = base - (base % dp_size) + dp_rank

        logger.debug(
            f"Processing migration request: {rid=}, tokens_seen={tokens_seen}, "
            f"bootstrap={bootstrap_host}:{bootstrap_port}, room={bootstrap_room}, "
            f"src_dp_rank={getattr(self, 'dp_rank', None)}, src_tp_rank={getattr(self, 'tp_rank', None)}"
        )

        # Find the request in running_batch
        req = self._find_and_remove_request(rid)
        if req is None:
            logger.debug(
                f"Migration: request {rid} not found on this worker (expected in DP setups) "
                f"src_dp_rank={getattr(self, 'dp_rank', None)}, src_tp_rank={getattr(self, 'tp_rank', None)}, "
                f"room={bootstrap_room}"
            )
            # Send error response back to tokenizer
            output = MigrateReqOutput(
                rid=rid,
                src_dp_rank=getattr(self, "dp_rank", None),
                src_tp_rank=getattr(self, "tp_rank", None),
                bootstrap_room=bootstrap_room,
                pending_output_ids=[],
                total_tokens=0,
                success=False,
                not_found=True,
                error=f"Request {rid} not found in running_batch",
            )
            self.send_to_tokenizer.send_output(output, recv_req)
            return

        if getattr(req, "kv_committed_freed", False):
            logger.warning(
                f"Migration: request {rid} already freed (likely finished). "
                f"src_dp_rank={getattr(self, 'dp_rank', None)}, src_tp_rank={getattr(self, 'tp_rank', None)}, "
                f"room={bootstrap_room}"
            )
            output = MigrateReqOutput(
                rid=rid,
                src_dp_rank=getattr(self, "dp_rank", None),
                src_tp_rank=getattr(self, "tp_rank", None),
                bootstrap_room=bootstrap_room,
                pending_output_ids=[],
                total_tokens=0,
                success=False,
                not_found=True,
                error=f"Request {rid} not found in running_batch",
            )
            self.send_to_tokenizer.send_output(output, recv_req)
            return

        # Calculate pending output tokens
        # tokens_seen is total tokens frontend has seen (origin_input_ids + output tokens yielded)
        # We need to send output_ids that the frontend hasn't seen yet
        origin_input_len = len(req.origin_input_ids)
        logical_tokens = origin_input_len + len(req.output_ids)
        committed_len = getattr(req, "kv_committed_len", None)
        allocated_len = getattr(req, "kv_allocated_len", None)
        committed_source = "kv_committed_len"
        if committed_len is None or committed_len <= 0:
            committed_len = allocated_len
            committed_source = "kv_allocated_len"
        if committed_len is None:
            committed_len = logical_tokens
            committed_source = "logical_tokens"

        if logical_tokens not in (committed_len, committed_len + 1):
            error_message = (
                f"Migration KV length mismatch for request {rid}: "
                f"logical_tokens={logical_tokens}, kv_committed_len={getattr(req, 'kv_committed_len', None)}, "
                f"kv_allocated_len={allocated_len} (source={committed_source})"
            )
            self._handle_migration_failure(req, error_message)
            output = MigrateReqOutput(
                rid=rid,
                src_dp_rank=getattr(self, "dp_rank", None),
                src_tp_rank=getattr(self, "tp_rank", None),
                bootstrap_room=bootstrap_room,
                pending_output_ids=[],
                total_tokens=logical_tokens,
                success=False,
                error=error_message,
            )
            self.send_to_tokenizer.send_output(output, recv_req)
            self.stream_output([req], req.return_logprob)
            return

        effective_tokens = min(logical_tokens, committed_len)
        token_to_kv_pool = self.token_to_kv_pool_allocator.get_kvcache()
        page_size = token_to_kv_pool.page_size
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :effective_tokens
        ]
        page_indices = kv_to_page_indices(kv_indices.cpu().numpy(), page_size)
        expected_pages = kv_to_page_num(effective_tokens, page_size)
        if len(page_indices) != expected_pages:
            error_message = (
                f"Migration page_indices mismatch for request {rid}: "
                f"effective_tokens={effective_tokens}, page_size={page_size}, "
                f"page_indices_len={len(page_indices)}, expected_pages={expected_pages}"
            )
            self._handle_migration_failure(req, error_message)
            output = MigrateReqOutput(
                rid=rid,
                src_dp_rank=getattr(self, "dp_rank", None),
                src_tp_rank=getattr(self, "tp_rank", None),
                bootstrap_room=bootstrap_room,
                pending_output_ids=[],
                total_tokens=logical_tokens,
                success=False,
                error=error_message,
            )
            self.send_to_tokenizer.send_output(output, recv_req)
            self.stream_output([req], req.return_logprob)
            return

        req.migration_logical_tokens = logical_tokens
        req.migration_committed_len = committed_len
        req.migration_effective_tokens = effective_tokens
        req.migration_page_indices = page_indices
        
        # How many output tokens has frontend seen?
        output_tokens_seen = max(0, tokens_seen - origin_input_len)
        # Pending output tokens are everything after that
        pending_output_ids = list(req.output_ids[output_tokens_seen:])
        if logical_tokens == committed_len + 1 and pending_output_ids:
            if pending_output_ids[-1] == req.output_ids[-1]:
                pending_output_ids.pop()
            else:
                logger.warning(
                    "Migration pending_output_ids did not include expected aux token "
                    "rid=%s output_tokens_seen=%s pending_output_ids_len=%s",
                    req.rid,
                    output_tokens_seen,
                    len(pending_output_ids),
                )

        logger.debug(
            f"Migration request found: rid={rid}, "
            f"origin_input_ids_len={origin_input_len}, "
            f"output_ids_len={len(req.output_ids)}, "
            f"total_tokens={logical_tokens}, "
            f"kv_committed_len={getattr(req, 'kv_committed_len', None)}, "
            f"kv_allocated_len={allocated_len}, "
            f"effective_tokens={effective_tokens}, "
            f"page_indices_len={len(page_indices)}, "
            f"tokens_seen_by_frontend={tokens_seen}, "
            f"output_tokens_seen={output_tokens_seen}, "
            f"pending_output_ids_len={len(pending_output_ids)}"
        )

        # Send response back to tokenizer with pending outputs
        output = MigrateReqOutput(
            rid=rid,
            src_dp_rank=getattr(self, "dp_rank", None),
            src_tp_rank=getattr(self, "tp_rank", None),
            bootstrap_room=bootstrap_room,
            pending_output_ids=pending_output_ids,
            total_tokens=logical_tokens,
            success=True,
        )
        self.send_to_tokenizer.send_output(output, recv_req)

        # Setup KV sender for migration
        self._setup_migration_sender(req, bootstrap_host, bootstrap_port, bootstrap_room)

        # Add to migration inflight queue
        # Mark as migrating so normal completion path won't release KV cache
        req.is_migrating = True
        self.disagg_migration_inflight_queue.append(req)
        # Debug/timing metadata for migration instrumentation
        req.migration_bootstrap_room = bootstrap_room
        req.migration_ts_enqueued = time.time()
        req.migration_ts_send_called = None
        req.migration_ts_success = None
        logger.debug(
            f"Request {rid} added to migration queue: "
            f"room={bootstrap_room}, "
            f"src_dp_rank={getattr(self, 'dp_rank', None)}, src_tp_rank={getattr(self, 'tp_rank', None)}, "
            f"queue_size={len(self.disagg_migration_inflight_queue)}, "
            f"kv_allocated_len={req.kv_allocated_len}, "
            f"kv_committed_len={req.kv_committed_len}, "
            f"req_pool_idx={req.req_pool_idx}"
        )

    def _find_and_remove_request(self: "Scheduler", rid: str) -> Optional[Req]:
        """Find a request by ID in running_batch and remove it.
        
        This properly updates all batch tensors (req_pool_indices, seq_lens, etc.)
        to maintain consistency with the reqs list.
        
        Args:
            rid: The request ID to find.
            
        Returns:
            The request if found, None otherwise.
        """
        if self.running_batch.is_empty():
            return None

        # Find the request index in running_batch
        req_index = None
        req = None
        for i, r in enumerate(self.running_batch.reqs):
            if r.rid == rid:
                req_index = i
                req = r
                break
        
        if req_index is None:
            return None

        # Build keep_indices (all indices except the one to remove)
        keep_indices = [i for i in range(len(self.running_batch.reqs)) if i != req_index]
        
        if len(keep_indices) == 0:
            # Removing the last request
            self.running_batch.reqs = []
            self.running_batch.req_pool_indices = torch.empty(0, dtype=torch.int32, device=self.device)
            self.running_batch.seq_lens = torch.empty(0, dtype=torch.int32, device=self.device)
            self.running_batch.seq_lens_cpu = torch.empty(0, dtype=torch.int32)
            self.running_batch.orig_seq_lens = torch.empty(0, dtype=torch.int32, device=self.device)
            self.running_batch.output_ids = torch.empty(0, dtype=torch.int64, device=self.device)
            self.running_batch.seq_lens_sum = 0
        else:
            # Use filter_batch with explicit keep_indices to properly update all tensors
            self.running_batch.filter_batch(keep_indices=keep_indices)
        
        logger.debug(f"Removed request {rid} from running_batch, remaining={len(self.running_batch.reqs)}")
        return req

    def _setup_migration_sender(
        self: "Scheduler",
        req: Req,
        bootstrap_host: str,
        bootstrap_port: int,
        bootstrap_room: int,
    ) -> None:
        """Setup KV sender for the migration request.
        
        This mirrors the setup in PrefillBootstrapQueue.add() but for migration.
        
        Args:
            req: The request to migrate.
            bootstrap_host: Destination bootstrap host.
            bootstrap_port: Destination bootstrap port.
            bootstrap_room: Destination bootstrap room ID.
        """
        transfer_backend = TransferBackend(
            self.server_args.disaggregation_transfer_backend
        )
        # Use the migration kv_manager which is in PREFILL mode (sender mode).
        # The decode prealloc queue's kv_manager is in DECODE mode (receiver mode) which
        # doesn't have the sender's parallel info registered with its bootstrap server.
        # The migration kv_manager sets up a proper sender bootstrap server that decode_long's
        # receiver can query via /route to get the connection info.
        kv_manager = self._get_migration_kv_manager()

        kv_sender_class = get_kv_class(transfer_backend, KVClassType.SENDER)
        dest_tp_ranks = [self.tp_rank]

        req.disagg_kv_sender = kv_sender_class(
            mgr=kv_manager,
            bootstrap_addr=f"{bootstrap_host}:{bootstrap_port}",
            bootstrap_room=bootstrap_room,
            dest_tp_ranks=dest_tp_ranks,
            pp_rank=0,  # Migration is within a single PP stage
        )

        req.migration_kv_sent = False

    def _send_migration_kv(self: "Scheduler", req: Req) -> None:
        """Send the KV cache for a migration request.
        
        Args:
            req: The request whose KV cache to send.
        """
        # Update fill_ids to reflect actual computed tokens before sending
        # This ensures the request metadata is consistent for the receiver
        req.fill_ids = list(req.origin_input_ids) + list(req.output_ids)

        # Send the precomputed page indices for this migration.
        token_to_kv_pool = self.token_to_kv_pool_allocator.get_kvcache()
        page_size = token_to_kv_pool.page_size
        page_indices = req.migration_page_indices

        logger.debug(
            f"_send_migration_kv: rid={req.rid}, "
            f"effective_tokens={getattr(req, 'migration_effective_tokens', None)}, "
            f"origin_input_ids_len={len(req.origin_input_ids)}, "
            f"output_ids_len={len(req.output_ids)}, "
            f"page_size={page_size}, "
            f"page_indices_len={len(page_indices)}, "
            f"sender_num_kv_indices={req.disagg_kv_sender.num_kv_indices}"
        )

        # Send the KV cache using page indices
        req.disagg_kv_sender.send(
            kv_indices=page_indices,
        )
        req.migration_ts_send_called = time.time()
        logger.debug(
            f"_send_migration_kv: send called: rid={req.rid}, "
            f"room={getattr(req, 'migration_bootstrap_room', None)}, "
            f"src_dp_rank={getattr(self, 'dp_rank', None)}, src_tp_rank={getattr(self, 'tp_rank', None)}, "
            f"effective_tokens={getattr(req, 'migration_effective_tokens', None)}, "
            f"page_indices_len={len(page_indices)}"
        )

    @torch.no_grad()
    def process_migration_inflight_queue(self: "Scheduler") -> List[Req]:
        """Poll migration transfers and handle completed ones.
        
        Similar to process_disagg_prefill_inflight_queue but for migrations.
        
        Returns:
            List of requests that completed migration.
        """
        if len(self.disagg_migration_inflight_queue) == 0:
            return []

        done_reqs = []
        undone_reqs: List[Req] = []

        polls = poll_and_all_reduce(
            [req.disagg_kv_sender for req in self.disagg_migration_inflight_queue],
            self.attn_tp_cpu_group,
        )

        for req, poll in zip(self.disagg_migration_inflight_queue, polls):
            enq_ts = getattr(req, "migration_ts_enqueued", None)
            if enq_ts and (time.time() - enq_ts) > MIGRATION_WAIT_TIMEOUT_S:
                error_message = (
                    f"Migration timed out for request {req.rid} "
                    f"after {MIGRATION_WAIT_TIMEOUT_S}s "
                    f"room={getattr(req, 'migration_bootstrap_room', None)}"
                )
                self._handle_migration_failure(req, error_message)
                done_reqs.append(req)
                continue

            # Still waiting for receiver to connect
            if poll == KVPoll.Bootstrapping:
                undone_reqs.append(req)
                continue
            
            # Receiver connected - initialize and send if not already done
            if poll == KVPoll.WaitingForInput and not getattr(req, 'migration_kv_sent', False):
                page_indices = getattr(req, "migration_page_indices", None)
                if page_indices is None:
                    error_message = (
                        f"Migration missing page_indices for request {req.rid} "
                        f"room={getattr(req, 'migration_bootstrap_room', None)}"
                    )
                    self._handle_migration_failure(req, error_message)
                    done_reqs.append(req)
                    continue

                # Compute num_pages from page indices (same as prefill does)
                page_size = self.token_to_kv_pool_allocator.get_kvcache().page_size
                num_pages = len(page_indices)
                effective_tokens = getattr(req, "migration_effective_tokens", None)
                expected_pages = (
                    kv_to_page_num(effective_tokens, page_size)
                    if effective_tokens is not None
                    else None
                )
                if expected_pages is not None and num_pages != expected_pages:
                    error_message = (
                        f"Migration page_indices mismatch for request {req.rid}: "
                        f"page_indices_len={num_pages}, expected_pages={expected_pages}, "
                        f"page_size={page_size}"
                    )
                    self._handle_migration_failure(req, error_message)
                    done_reqs.append(req)
                    continue

                t_enqueued_s = (time.time() - enq_ts) if enq_ts else None
                logger.debug(
                    f"process_migration_inflight_queue: rid={req.rid}, "
                    f"room={getattr(req, 'migration_bootstrap_room', None)}, "
                    f"src_dp_rank={getattr(self, 'dp_rank', None)}, src_tp_rank={getattr(self, 'tp_rank', None)}, "
                    f"t_enqueued_s={t_enqueued_s}, "
                    f"effective_tokens={effective_tokens}, "
                    f"page_size={page_size}, "
                    f"num_pages={num_pages}, "
                    f"origin_input_ids_len={len(req.origin_input_ids)}, "
                    f"output_ids_len={len(req.output_ids)}"
                )
                # Allocate metadata buffer and populate with the last output token.
                # The receiver expects to receive the next token via aux data (like prefill does).
                # For migration, this should be the last token decode1 generated.
                aux_index = None
                if (
                    hasattr(self, "req_to_metadata_buffer_idx_allocator")
                    and self.req_to_metadata_buffer_idx_allocator is not None
                    and hasattr(self, "disagg_metadata_buffers")
                    and self.disagg_metadata_buffers is not None
                ):
                    aux_index = self.req_to_metadata_buffer_idx_allocator.alloc()
                    req.metadata_buffer_index = aux_index
                    # Populate the aux buffer with the last output token
                    # For migration, the receiver will append this token to its output_ids
                    if len(req.output_ids) > 0:
                        last_output_token = req.output_ids[-1]
                        self.disagg_metadata_buffers.output_ids[aux_index][0] = last_output_token
                        # Also set cached_tokens to 0 (similar to prefill)
                        self.disagg_metadata_buffers.cached_tokens[aux_index][0] = 0
                
                # Initialize sender with num_pages (not num_tokens)
                req.disagg_kv_sender.init(num_pages, aux_index=aux_index)
                
                # Send KV cache
                self._send_migration_kv(req)
                req.migration_kv_sent = True
                undone_reqs.append(req)
                continue
            
            if poll in [KVPoll.WaitingForInput, KVPoll.Transferring]:
                undone_reqs.append(req)
            elif poll == KVPoll.Success:
                # Transfer completed successfully
                req.migration_ts_success = time.time()
                enq = getattr(req, "migration_ts_enqueued", None)
                send_called = getattr(req, "migration_ts_send_called", None)
                total_s = (req.migration_ts_success - enq) if enq else None
                send_to_success_s = (
                    (req.migration_ts_success - send_called)
                    if (send_called is not None)
                    else None
                )
                logger.debug(
                    f"Migration KV transfer success: rid={req.rid}, "
                    f"room={getattr(req, 'migration_bootstrap_room', None)}, "
                    f"src_dp_rank={getattr(self, 'dp_rank', None)}, src_tp_rank={getattr(self, 'tp_rank', None)}, "
                    f"t_total_s={total_s}, t_send_to_success_s={send_to_success_s}, "
                    f"kv_allocated_len={req.kv_allocated_len}, "
                    f"kv_committed_len={req.kv_committed_len}, "
                    f"origin_input_ids_len={len(req.origin_input_ids)}, "
                    f"output_ids_len={len(req.output_ids)}, "
                    f"req_pool_idx={req.req_pool_idx}"
                )
                try:
                    release_kv_cache(req, self.tree_cache)
                except Exception as e:
                    logger.info(f"rid={req.rid} error releasing KV cache: {e}")
                req.finished_reason = FINISH_LENGTH(length=0)
                if hasattr(req.disagg_kv_sender, "clear"):
                    req.disagg_kv_sender.clear()
                # Free the metadata buffer index if allocated
                if (
                    hasattr(req, "metadata_buffer_index")
                    and req.metadata_buffer_index is not None
                    and req.metadata_buffer_index >= 0
                ):
                    self.req_to_metadata_buffer_idx_allocator.free(req.metadata_buffer_index)
                    req.metadata_buffer_index = -1
                done_reqs.append(req)
                logger.debug(f"Migration cleanup completed for request {req.rid}")
            elif poll == KVPoll.Failed:
                error_message = (
                    f"Migration transfer failed for request {req.rid} "
                    f"on rank={self.tp_rank}"
                )
                logger.error(
                    f"Migration KV transfer failed: rid={req.rid}, "
                    f"room={getattr(req, 'migration_bootstrap_room', None)}, "
                    f"src_dp_rank={getattr(self, 'dp_rank', None)}, src_tp_rank={getattr(self, 'tp_rank', None)}"
                )
                self._handle_migration_failure(req, error_message)
                done_reqs.append(req)
            else:
                raise ValueError(f"Unexpected polling state {poll=}")

        for req in done_reqs:
            req.time_stats.completion_time = time.perf_counter()

        # Stream output for completed migrations
        self.stream_output(
            done_reqs,
            any(req.return_logprob for req in done_reqs),
            None,
        )

        self.disagg_migration_inflight_queue = undone_reqs

        return done_reqs

