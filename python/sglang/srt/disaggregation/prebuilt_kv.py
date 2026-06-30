"""Admission state for requests whose KV was materialized by another worker."""

from __future__ import annotations

from array import array
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from sglang.srt.utils.common import ceil_align

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler


@dataclass(slots=True)
class PrebuiltKVState:
    transfer_id: str
    pending_input_id: Optional[int] = None

    @property
    def ready(self) -> bool:
        return self.pending_input_id is not None


@dataclass(frozen=True, slots=True)
class PrebuiltKVFrontier:
    committed_input_ids: list[int]
    pending_input_id: int
    prompt_len: int
    committed_len: int
    logical_len: int
    output_tokens_seen: int
    max_new_tokens: Optional[int]
    min_new_tokens: Optional[int]


def bind_prebuilt_kv_from_transfer(scheduler: Scheduler, decode_req) -> Optional[str]:
    try:
        frontier = scheduler.disagg_metadata_buffers.get_decode_migration_frontier(
            decode_req.metadata_buffer_index
        )
    except ValueError as exc:
        return str(exc)
    if frontier is None:
        return "Prebuilt KV transfer omitted frontier metadata"
    if frontier.prompt_len > frontier.committed_len:
        return "Prebuilt KV prompt length exceeds committed length"
    return _bind_prebuilt_kv_state(scheduler, decode_req, frontier)


def _bind_prebuilt_kv_state(
    scheduler: Scheduler,
    decode_req,
    frontier: PrebuiltKVFrontier,
) -> Optional[str]:
    if len(frontier.committed_input_ids) != frontier.committed_len:
        return "committed_input_ids length does not match"
    if frontier.logical_len != frontier.committed_len + 1:
        return "Prebuilt KV state requires exactly one pending input token"

    req = decode_req.req
    assert req.prebuilt_kv is not None
    in_prealloc_queue = any(
        candidate is decode_req
        for candidate in scheduler.disagg_decode_prealloc_queue.queue
    )
    reserved_len = (
        len(req.origin_input_ids) + max(len(req.output_ids) - 1, 0)
        if req.req_pool_idx is None and in_prealloc_queue
        else req.kv_allocated_len
    )
    if reserved_len < frontier.committed_len:
        return "Prebuilt KV reservation is smaller than transferred state"
    if req.req_pool_idx is not None:
        free_start = ceil_align(
            frontier.committed_len,
            scheduler.token_to_kv_pool_allocator.page_size,
        )
        if free_start < reserved_len:
            unused = scheduler.req_to_token_pool.req_to_token[req.req_pool_idx][
                free_start:reserved_len
            ]
            scheduler.token_to_kv_pool_allocator.free(unused)

    req.origin_input_ids = array("q", frontier.committed_input_ids)
    req.origin_input_ids_unpadded = req.origin_input_ids
    req.output_ids = array("q")
    req.prebuilt_kv.pending_input_id = frontier.pending_input_id
    req.full_untruncated_fill_ids = req.origin_input_ids
    if req.req_pool_idx is not None:
        req.kv_committed_len = frontier.committed_len
        req.kv_allocated_len = frontier.committed_len
        req.fill_len = frontier.committed_len
        req.set_extend_input_len(frontier.committed_len - len(req.prefix_indices))
    if frontier.max_new_tokens is not None:
        req.sampling_params.max_new_tokens = frontier.max_new_tokens
    if frontier.min_new_tokens is not None:
        req.sampling_params.min_new_tokens = frontier.min_new_tokens
    if req.req_pool_idx is not None and hasattr(
        decode_req.kv_receiver, "resume_waiting_timeout"
    ):
        decode_req.kv_receiver.resume_waiting_timeout()
    return None
