"""Fused FP4 index scoring and selection of the DeepSeek V4.1 ratio-1/2 index
layers on SM100. One CuTe DSL pass scores a block of four query rows against the
index-K pages of one split and keeps the split's best keys (and, for a candidate
source, its best eight-position blocks); a merge pass per row reduces the splits
to the selected compressed positions, pool slots, or block ids."""

from __future__ import annotations

import functools
import importlib
import math
import os
from typing import Optional

import torch

from sglang.kernels.jit.cute_aot_cache import get_jit_cache

HEADS = 32
HEAD_BYTES = 64  # a 128-dim head as packed e2m1 pairs
INDEX_PAGE_SLOTS = 128  # compressed positions per index-K page
INDEX_PAGE_BYTES = INDEX_PAGE_SLOTS * (HEAD_BYTES + 4)  # payload rows, then scale words
SLOT_PAGE = 64  # positions per entry of the merge's slot table
ROWS_PER_CTA = 4
PAGES_PER_GROUP = 8
TOPK = 512  # positions a row selects
CANDIDATE_BLOCKS = 2048  # blocks a candidate source publishes per row
BLOCK_PARTIAL_WIDTH = 4096  # block keys a split keeps per row

_MODULES = {
    "score": "fp4_indexer_fused_score",
    "candidates": "fp4_indexer_fused_candidates",
    "merge_pages": "fp4_indexer_fused_merge_pages",
    "merge_blocks": "fp4_indexer_fused_merge_blocks",
}
_HERE = os.path.dirname(os.path.abspath(__file__))


@functools.cache
def _compiled_cache():
    return get_jit_cache(
        "fp4_indexer_fused",
        source_paths=(__file__,)
        + tuple(os.path.join(_HERE, f"{name}.py") for name in _MODULES.values()),
        enable_tvm_ffi=True,
    )


@functools.cache
def fused_indexer_available() -> bool:
    """The kernels are sm_100a binaries: CC 10.0 with a CUDA torch build."""
    if not torch.cuda.is_available() or torch.version.cuda is None:
        return False
    return torch.cuda.get_device_capability() == (10, 0)


def _program(name: str):
    """The compiled launcher of one entry; shapes are dynamic, so one build per
    entry serves every call of the process (and later processes through the
    persistent CuTe cache)."""
    cache = _compiled_cache()
    key = (name,)
    if key not in cache:
        module = importlib.import_module(f"{__package__}.{_MODULES[name]}")
        cache[key] = module.compile_program()
    return cache[key]


def _flat(t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    assert t.is_contiguous(), "the fused indexer reads contiguous buffers"
    assert t.data_ptr() % 16 == 0, "the fused indexer needs 16-byte aligned buffers"
    return t.view(dtype).reshape(-1)


def fused_index_splits(rows: int, page_count: int, *, sm_count: int) -> int:
    """Page splits of a row block: enough CTAs to fill the device, never less
    than one group of pages per split."""
    ctas = max(1, math.ceil(rows / ROWS_PER_CTA))
    groups = max(1, math.ceil(page_count / PAGES_PER_GROUP))
    return max(1, min(sm_count // ctas, groups))


def fused_index_score(
    *,
    q_fp4: torch.Tensor,
    q_sf: torch.Tensor,
    weights: torch.Tensor,
    lengths: torch.Tensor,
    page_table: torch.Tensor,
    k_cache: torch.Tensor,
    partial: torch.Tensor,
    block_partial: Optional[torch.Tensor],
    splits: int,
) -> None:
    """Score every row against its first ``lengths[r]`` compressed positions and
    keep, per split, the ``TOPK`` best keys in ``partial`` ``[rows, splits, TOPK]``
    int64; with ``block_partial`` ``[rows, splits, BLOCK_PARTIAL_WIDTH]`` int64
    also the best eight-position blocks. ``q_fp4`` ``[rows, HEADS, 64]`` int8 and
    ``q_sf`` ``[rows, HEADS]`` int32 are the packed fp4 queries, ``weights``
    ``[rows, HEADS]`` fp32, ``page_table`` ``[rows, width]`` int32 the index-K
    pages of each row, ``k_cache`` ``[pages, INDEX_PAGE_BYTES]`` uint8 the pool."""
    rows = q_fp4.shape[0]
    assert q_fp4.shape == (rows, HEADS, HEAD_BYTES), q_fp4.shape
    assert q_sf.shape == (rows, HEADS) and q_sf.dtype == torch.int32
    assert weights.shape == (rows, HEADS) and weights.dtype == torch.float32
    assert lengths.shape == (rows,) and lengths.dtype == torch.int32
    assert page_table.dim() == 2 and page_table.shape[0] == rows
    assert k_cache.dim() == 2 and k_cache.shape[1] == INDEX_PAGE_BYTES, k_cache.shape
    assert partial.shape == (rows, splits, TOPK) and partial.dtype == torch.int64
    if block_partial is None:
        block_partial = partial.new_empty(2)
        program = _program("score")
    else:
        assert block_partial.shape == (rows, splits, BLOCK_PARTIAL_WIDTH)
        program = _program("candidates")
    # TMA extents are element counts, innermost first; strides in 16-byte units.
    program(
        _flat(q_fp4, torch.uint8),
        HEAD_BYTES,
        rows * HEADS,
        HEAD_BYTES // 16,
        _flat(q_sf, torch.int32),
        HEADS,
        rows,
        HEADS * 4 // 16,
        _flat(k_cache, torch.uint8),
        HEAD_BYTES,
        INDEX_PAGE_BYTES // HEAD_BYTES,
        k_cache.shape[0],
        HEAD_BYTES // 16,
        INDEX_PAGE_BYTES // 16,
        _flat(weights, torch.float32),
        _flat(lengths, torch.int32),
        _flat(page_table, torch.int32),
        _flat(partial, torch.int64),
        _flat(block_partial, torch.int64),
        rows,
        page_table.shape[1],
        splits,
        math.ceil(rows / ROWS_PER_CTA) * splits,
        1,
        1,
    )


def fused_index_merge_pages(
    *,
    partial: torch.Tensor,
    pages: torch.Tensor,
    table_width: int,
    out: torch.Tensor,
    out_offset: int = 0,
) -> None:
    """Reduce the splits of ``partial`` to each row's ``TOPK`` positions, mapped
    through ``pages`` (``SLOT_PAGE`` positions per entry, ``table_width`` per
    row; a width of 0 shares one table across rows) into ``out[r, out_offset:]``
    ``[rows, >= TOPK]`` int32, unordered, -1 padded."""
    rows, splits, _ = partial.shape
    assert out.dim() == 2 and out.dtype == torch.int32 and out.shape[0] >= rows
    assert out.shape[1] >= out_offset + TOPK
    _program("merge_pages")(
        _flat(partial, torch.int64),
        _flat(pages, torch.int32),
        _flat(out, torch.int32),
        splits,
        table_width,
        out.shape[1],
        out_offset,
        rows,
        1,
        1,
    )


def fused_index_merge_blocks(*, block_partial: torch.Tensor, out: torch.Tensor) -> None:
    """Reduce the splits of ``block_partial`` to each row's ``CANDIDATE_BLOCKS``
    block ids into ``out`` ``[rows, CANDIDATE_BLOCKS]`` int32, unordered, -1
    padded."""
    rows, splits, _ = block_partial.shape
    assert out.shape == (rows, CANDIDATE_BLOCKS) and out.dtype == torch.int32
    _program("merge_blocks")(
        _flat(block_partial, torch.int64),
        _flat(out, torch.int32),  # unused by this entry
        _flat(out, torch.int32),
        splits,
        0,
        CANDIDATE_BLOCKS,
        0,
        rows,
        1,
        1,
    )
