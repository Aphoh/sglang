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
from sglang.srt.mem_cache.memory_pool import (
    HybridLinearKVPool,
    NSATokenToKVPool,
    SWAKVPool,
)

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)


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
        bootstrap_room = recv_req.bootstrap_room
        tokens_seen = recv_req.tokens_seen

        logger.info(
            f"Processing migration request: {rid=}, tokens_seen={tokens_seen}, "
            f"bootstrap={bootstrap_host}:{bootstrap_port}, room={bootstrap_room}"
        )

        # Find the request in running_batch
        req = self._find_and_remove_request(rid)
        if req is None:
            logger.warning(f"Migration failed: request {rid} not found in running_batch")
            # Send error response back to tokenizer
            output = MigrateReqOutput(
                rid=rid,
                pending_output_ids=[],
                total_tokens=0,
                success=False,
                error=f"Request {rid} not found in running_batch",
            )
            self.send_to_tokenizer.send_pyobj(output)
            return

        # Calculate pending output tokens
        # tokens_seen is total tokens frontend has seen (origin_input_ids + output tokens yielded)
        # We need to send output_ids that the frontend hasn't seen yet
        origin_input_len = len(req.origin_input_ids)
        total_tokens = origin_input_len + len(req.output_ids)
        
        # How many output tokens has frontend seen?
        output_tokens_seen = max(0, tokens_seen - origin_input_len)
        # Pending output tokens are everything after that
        pending_output_ids = list(req.output_ids[output_tokens_seen:])

        logger.info(
            f"Migration request found: rid={rid}, "
            f"origin_input_ids_len={origin_input_len}, "
            f"output_ids_len={len(req.output_ids)}, "
            f"total_tokens={total_tokens}, "
            f"tokens_seen_by_frontend={tokens_seen}, "
            f"output_tokens_seen={output_tokens_seen}, "
            f"pending_output_ids_len={len(pending_output_ids)}"
        )

        # Send response back to tokenizer with pending outputs
        output = MigrateReqOutput(
            rid=rid,
            pending_output_ids=pending_output_ids,
            total_tokens=total_tokens,
            success=True,
        )
        self.send_to_tokenizer.send_pyobj(output)

        # Setup KV sender for migration
        self._setup_migration_sender(req, bootstrap_host, bootstrap_port, bootstrap_room)

        # Add to migration inflight queue
        self.disagg_migration_inflight_queue.append(req)
        logger.info(f"Request {rid} added to migration queue")

    def _find_and_remove_request(self: "Scheduler", rid: str) -> Optional[Req]:
        """Find a request by ID in running_batch and remove it.
        
        Args:
            rid: The request ID to find.
            
        Returns:
            The request if found, None otherwise.
        """
        if self.running_batch.is_empty():
            return None

        # Find the request in running_batch
        for i, req in enumerate(self.running_batch.reqs):
            if req.rid == rid:
                # Remove from running_batch
                self.running_batch.reqs.pop(i)
                logger.debug(f"Removed request {rid} from running_batch")
                return req

        return None

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
        # Use the existing decode prealloc queue's kv_manager instead of creating a new one
        # This ensures the sender uses the same manager that's already registered with the
        # bootstrap server and whose bootstrap thread will receive the receiver's connection
        kv_manager = self.disagg_decode_prealloc_queue.kv_manager

        kv_sender_class = get_kv_class(transfer_backend, KVClassType.SENDER)
        dest_tp_ranks = [self.tp_rank]

        req.disagg_kv_sender = kv_sender_class(
            mgr=kv_manager,
            bootstrap_addr=f"{bootstrap_host}:{bootstrap_port}",
            bootstrap_room=bootstrap_room,
            dest_tp_ranks=dest_tp_ranks,
            pp_rank=0,  # Migration is within a single PP stage
        )

        # Store num_tokens for later when we call init() and send()
        # We need to wait for the receiver to connect first (status = WaitingForInput)
        # IMPORTANT: fill_ids is stale from initial prefill transfer, so we must compute
        # the actual token count from origin_input_ids + output_ids
        actual_computed_tokens = len(req.origin_input_ids) + len(req.output_ids)
        req.migration_num_tokens = actual_computed_tokens
        req.migration_kv_sent = False

    def _send_migration_kv(self: "Scheduler", req: Req) -> None:
        """Send the KV cache for a migration request.
        
        Args:
            req: The request whose KV cache to send.
        """
        # Use migration_num_tokens which was computed from origin_input_ids + output_ids
        # fill_ids is stale from initial prefill transfer and doesn't reflect generated tokens
        num_tokens_to_send = req.migration_num_tokens
        
        # Verify we're not trying to send more than what's allocated
        # The last generated token doesn't have KV yet, so we may need to adjust
        if hasattr(req, 'kv_allocated_len') and num_tokens_to_send > req.kv_allocated_len:
            # Clamp to what's actually allocated
            logger.info(
                f"_send_migration_kv: clamping num_tokens_to_send from {num_tokens_to_send} to {req.kv_allocated_len}"
            )
            num_tokens_to_send = req.kv_allocated_len
        
        # Update fill_ids to reflect actual computed tokens before sending
        # This ensures the request metadata is consistent for the receiver
        req.fill_ids = list(req.origin_input_ids) + list(req.output_ids)
        
        # Get the KV indices for this request
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :num_tokens_to_send
        ]
        
        # Convert to page indices (same as prefill does)
        token_to_kv_pool = self.token_to_kv_pool_allocator.get_kvcache()
        page_size = token_to_kv_pool.page_size
        page_indices = kv_to_page_indices(kv_indices.cpu().numpy(), page_size)
        
        logger.info(
            f"_send_migration_kv: rid={req.rid}, "
            f"num_tokens_to_send={num_tokens_to_send}, "
            f"origin_input_ids_len={len(req.origin_input_ids)}, "
            f"output_ids_len={len(req.output_ids)}, "
            f"kv_indices_len={len(kv_indices)}, "
            f"page_size={page_size}, "
            f"page_indices_len={len(page_indices)}, "
            f"sender_num_kv_indices={req.disagg_kv_sender.num_kv_indices}"
        )
        
        # Sample first few KV values for verification
        kv_sample = None
        try:
            # Get first layer's K cache, first few indices
            k_cache = token_to_kv_pool.k_buffer[0]  # First layer
            sample_indices = kv_indices[:min(4, len(kv_indices))].tolist()
            kv_sample = k_cache[sample_indices[0], :4].tolist() if sample_indices else None
        except Exception as e:
            kv_sample = f"error: {e}"

        # Send the KV cache using page indices
        req.disagg_kv_sender.send(
            kv_indices=page_indices,
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
            # Still waiting for receiver to connect
            if poll == KVPoll.Bootstrapping:
                undone_reqs.append(req)
                continue
            
            # Receiver connected - initialize and send if not already done
            if poll == KVPoll.WaitingForInput and not getattr(req, 'migration_kv_sent', False):
                # Compute num_pages from num_tokens (same as prefill does)
                page_size = self.token_to_kv_pool_allocator.get_kvcache().page_size
                num_pages = kv_to_page_num(req.migration_num_tokens, page_size)
                logger.info(
                    f"process_migration_inflight_queue: rid={req.rid}, "
                    f"migration_num_tokens={req.migration_num_tokens}, "
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
                release_kv_cache(req, self.tree_cache)
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
                logger.info(f"Migration completed for request {req.rid}")
            elif poll == KVPoll.Failed:
                error_message = (
                    f"Migration transfer failed for request {req.rid} "
                    f"on rank={self.tp_rank}"
                )
                try:
                    req.disagg_kv_sender.failure_exception()
                except Exception as e:
                    error_message += f" with exception {e}"
                logger.error(error_message)
                release_kv_cache(req, self.tree_cache)
                # Free the metadata buffer index if allocated
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

