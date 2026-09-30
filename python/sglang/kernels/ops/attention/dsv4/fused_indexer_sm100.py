# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from ai-dynamo/rhino crates/models/rhino-model-deepseek-v4-kernels/kernels/dsv4p1_fused_indexer/fused_indexer.py
"""DeepSeek V4.1 fused indexer.

Score: Each tcgen05 block-scaled MMA scores one 128-token page. Each split
selects the top 512 candidates in shared memory and writes them to an int64
workspace.
Merge: The kernel reduces split results, maps selected tokens through the page
table, and writes them in ascending order. It fills unused outputs with -1.
Numerics: ReLU output rounds to BF16, BF16-weight products round to BF16, sums
use F32, scores round to BF16, and ties select the lower token.
"""

from __future__ import annotations

import functools

import cuda.bindings.driver as cuda
import cutlass
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
import torch
from cutlass import cute, pipeline, utils
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.nvgpu.common import OperandMajorMode
from cutlass.cutlass_dsl import T, dsl_user_op

from sglang.kernels.jit.cute_aot_cache import get_jit_cache

HEADS = 32
HEAD_DIM = 128
GROUP = 32
GROUPS = HEAD_DIM // GROUP
PAGE_TOKENS = 128
PAGE_BYTES = 8704
PAGE_WORDS = PAGE_BYTES // 4
PAYLOAD_WORDS = PAGE_TOKENS * HEAD_DIM // 8
TOKEN_WORDS = HEAD_DIM // 8
QUERY_WORDS = HEADS * TOKEN_WORDS
BLOCK_Q = 4
DECODE_BLOCK_Q = 1
DECODE_MAX_ROWS = 128  # rows at or below this take one row per CTA
MMA_M = PAGE_TOKENS
MMA_N = BLOCK_Q * HEADS
MMA_K = HEAD_DIM
STAGES = 4
SCORE_THREADS = 128
MERGE_THREADS = 1024
TOPK = 512
CAPACITY = 1024
PRUNE_AT = CAPACITY - PAGE_TOKENS
HIST_SLOTS = 272
MERGE_KEYS_PER_THREAD = 16
MERGE_SLOTS = MERGE_KEYS_PER_THREAD + 1
MERGE_CHUNK = MERGE_KEYS_PER_THREAD * MERGE_THREADS
PRUNE_SLOTS = CAPACITY // SCORE_THREADS
SCORE_WARPS = SCORE_THREADS // 32
SF_WORDS = PAGE_TOKENS
TMEM_COLUMNS = {DECODE_BLOCK_Q: 128, BLOCK_Q: 256}
T2R_REPETITION = {
    DECODE_BLOCK_Q: tcgen05.Repetition.x32,
    BLOCK_Q: tcgen05.Repetition.x128,
}
# Dynamic shared memory per CTA, with slack for the 1024-byte operand alignment.
A_STAGE_BYTES = MMA_M * MMA_K // 2


def score_smem_bytes(block_q: int) -> int:
    """Dynamic shared memory of `score_partial_kernel` for `block_q` rows per CTA."""
    return (
        STAGES * A_STAGE_BYTES
        + block_q * HEADS * MMA_K // 2
        + (2 * STAGES + 1) * SF_WORDS * 4
        + block_q * CAPACITY * 8
        + (block_q * HEADS // 2 + HIST_SLOTS + SCORE_THREADS + 4 * block_q) * 4
        + 4096
    )


MERGE_BATCH = 16  # keys per thread per streamed batch
MERGE_CAP = 2048  # candidates that the compact buffer holds
MERGE_CAP_SLOTS = MERGE_CAP // MERGE_THREADS
MERGE_TARGET = 1280  # candidates that the sampled pivot is expected to keep
MERGE_SMEM_BYTES = HIST_SLOTS * 4 + MERGE_CAP * 8 + TOPK * 4


@cute.jit
def bf16_round(value: cutlass.Float32) -> cutlass.Float32:
    """Round a finite F32 to the nearest even BF16 and return it as F32."""
    return value.to(cutlass.BFloat16).to(cutlass.Float32)


@dsl_user_op
def pack_bf16_pair(lo, hi, *, loc=None, ip=None):
    """Round two F32 values to BF16 (nearest even) and pack them, `lo` in the low half."""
    value = llvm.inline_asm(
        T.i32(),
        [lo.ir_value(loc=loc, ip=ip), hi.ir_value(loc=loc, ip=ip)],
        "cvt.rn.bf16x2.f32 $0, $2, $1;",
        "=r,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return cutlass.Uint32(value)


@dsl_user_op
def relu_bf16_pair(lo, hi, *, loc=None, ip=None):
    """`pack_bf16_pair` of `max(lo, 0)` and `max(hi, 0)`."""
    value = llvm.inline_asm(
        T.i32(),
        [lo.ir_value(loc=loc, ip=ip), hi.ir_value(loc=loc, ip=ip)],
        "cvt.rn.relu.bf16x2.f32 $0, $2, $1;",
        "=r,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return cutlass.Uint32(value)


@dsl_user_op
def mul_bf16_pair(a, b, *, loc=None, ip=None):
    """Elementwise BF16 product of two packed pairs, rounded to nearest even."""
    value = llvm.inline_asm(
        T.i32(),
        [a.ir_value(loc=loc, ip=ip), b.ir_value(loc=loc, ip=ip)],
        "mul.rn.bf16x2 $0, $1, $2;",
        "=r,r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return cutlass.Uint32(value)


@cute.jit
def add_bf16_pair(total: cutlass.Float32, pair: cutlass.Uint32) -> cutlass.Float32:
    """Add the low half and then the high half of a packed BF16 pair to an F32 total."""
    total = total + cutlass.Uint16(pair).bitcast(cutlass.BFloat16).to(cutlass.Float32)
    return total + cutlass.Uint16(pair >> 16).bitcast(cutlass.BFloat16).to(
        cutlass.Float32
    )


@cute.jit
def ordered_key(bits: cutlass.Uint32, column: cutlass.Int32) -> cutlass.Uint64:
    """Unique 48-bit key that orders by score descending and then by column ascending.

    `bits` is a BF16-rounded F32 score, so its low 16 bits carry no information
    and the key holds only the high 16 bits of the order-preserving transform.
    """
    normalized = cutlass.Uint32(bits)
    if (bits & cutlass.Uint32(0x7FFFFFFF)) == 0:
        normalized = cutlass.Uint32(0)
    ordered = normalized ^ cutlass.Uint32(0x80000000)
    if (normalized & cutlass.Uint32(0x80000000)) != 0:
        ordered = normalized ^ cutlass.Uint32(0xFFFFFFFF)
    return (cutlass.Uint64(ordered >> 16) << 32) | cutlass.Uint64(
        cutlass.Uint32(0xFFFFFFFF) - cutlass.Uint32(column)
    )


@cute.jit
def key_column(key: cutlass.Uint64) -> cutlass.Int32:
    return cutlass.Int32(
        cutlass.Uint32(0xFFFFFFFF) - cutlass.Uint32(key & cutlass.Uint64(0xFFFFFFFF))
    )


@cute.jit
def warp_append(
    keep: cutlass.Boolean,
    count: cute.Pointer,
    lane: cutlass.Int32,
    lane_lt: cutlass.Uint32,
) -> cutlass.Int32:
    """Reserve one slot per lane with `keep` set, with one atomic per warp. Every lane must call it."""
    mask = cutlass.Uint32(cute.arch.vote_ballot_sync(keep))
    base = cutlass.Int32(0)
    if lane == 0:
        base = cute.arch.atomic_add(
            count, cutlass.Int32(cute.arch.popc(mask)), scope="cta", sem="relaxed"
        )
    base = cute.arch.shuffle_sync(base, 0)
    return base + cutlass.Int32(cute.arch.popc(mask & lane_lt))


@cute.jit
def insert_key(
    key: cutlass.Uint64,
    row_keys: cute.Tensor,
    count: cute.Pointer,
    threshold: cutlass.Uint64,
    lane: cutlass.Int32,
    lane_lt: cutlass.Uint32,
):
    """Append the warp's candidates `>= threshold` to a row's buffer. Every lane must call it."""
    keep = (key >= threshold) & (key != cutlass.Uint64(0))
    slot = warp_append(keep, count, lane, lane_lt)
    if keep:
        row_keys[slot] = key


@cute.jit
def match_any(value: cutlass.Int32) -> cutlass.Uint32:
    """Mask of the lanes whose `value` equals this lane's value. Every lane must call it."""
    mask = llvm.inline_asm(
        T.i32(),
        [value.ir_value()],
        "match.any.sync.b32 $0, $1, 0xffffffff;",
        "=r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return cutlass.Uint32(cutlass.Int32(mask))


@cute.jit
def count_digits(
    key: cutlass.Uint64,
    valid: cutlass.Boolean,
    shift: cutlass.Uint64,
    hist: cute.Tensor,
    lane_lt: cutlass.Uint32,
):
    """Add one key per lane to the digit histogram, one atomic per distinct digit in the warp."""
    digit = cutlass.Int32((key >> shift) & cutlass.Uint64(255))
    if not valid:
        digit = cutlass.Int32(-1)
    peers = match_any(digit)
    if valid & ((peers & lane_lt) == cutlass.Uint32(0)):
        cute.arch.atomic_add(
            hist.iterator + digit,
            cutlass.Int32(cute.arch.popc(peers)),
            scope="cta",
            sem="relaxed",
        )


@cute.jit
def select_digit(hist: cute.Tensor, remaining: cutlass.Int32, lane: cutlass.Int32):
    """One warp: find the digit that holds the `remaining`-th largest key, then clear the bins.

    Lane `l` owns the digits `8l..8l+7`. Writes the digit to `hist[256]`, the
    keys still needed inside it to `hist[257]`, its count to `hist[258]`, and
    the total count to `hist[259]`.
    """
    bins = [hist[lane * 8 + j] for j in range(8)]
    total = cutlass.Int32(0)
    for j in cutlass.range_constexpr(8):
        total += bins[j]
        hist[lane * 8 + j] = cutlass.Int32(0)
    above = total
    for offset in cutlass.range_constexpr(5):
        other = cute.arch.shuffle_sync_down(above, 1 << offset)
        if lane + (1 << offset) < 32:
            above += other
    if lane == 0:
        hist[259] = above
    acc = above - total
    for j in cutlass.range_constexpr(7, -1, -1):
        if (acc < remaining) & (acc + bins[j] >= remaining):
            hist[256] = lane * 8 + j
            hist[257] = remaining - acc
            hist[258] = bins[j]
        acc += bins[j]


@cute.jit
def select_threshold(
    mine: cute.Tensor,
    hist: cute.Tensor,
    warp: cutlass.Int32,
    lane: cutlass.Int32,
    lane_lt: cutlass.Uint32,
    remaining: cutlass.Int32,
    column_limit: cutlass.Int32,
    slots: cutlass.Constexpr,
) -> cutlass.Uint64:
    """Threshold `t` so that `key >= t` selects the `remaining` largest keys of the CTA, or every key.

    `mine` holds `slots` 48-bit `ordered_key` values per thread, 0 for an empty
    slot, and every column is below `column_limit`. Byte-wise radix select with
    an early stop when a whole digit is selected. `hist[0:256]` must be zero on
    entry and is zero on exit. Every thread must call it.
    """
    threshold = cutlass.Uint64(0)
    active = cutlass.Boolean(True)
    for step in range(6):
        shift = cutlass.Uint64(40 - step * 8)
        shared = cutlass.Boolean(False)
        if step >= 2:
            shared = (cutlass.Uint64(column_limit - 1) >> shift) == cutlass.Uint64(0)
        if active & shared:
            threshold = threshold | (cutlass.Uint64(255) << shift)
        elif active:
            prefix = threshold >> (shift + cutlass.Uint64(8))
            for j in cutlass.range_constexpr(slots):
                key = mine[j]
                valid = (key != cutlass.Uint64(0)) & (
                    (key >> (shift + cutlass.Uint64(8))) == prefix
                )
                if cutlass.Uint32(cute.arch.vote_ballot_sync(valid)) != cutlass.Uint32(
                    0
                ):
                    count_digits(key, valid, shift, hist, lane_lt)
            cute.arch.sync_threads()
            if warp == 0:
                select_digit(hist, remaining, lane)
            cute.arch.sync_threads()
            if hist[259] >= remaining:
                threshold = threshold | (cutlass.Uint64(hist[256]) << shift)
                if hist[257] == hist[258]:
                    active = cutlass.Boolean(False)
                remaining = hist[257]
            else:
                threshold = cutlass.Uint64(1)
                active = cutlass.Boolean(False)
    return threshold


@cute.jit
def prune(
    keys: cute.Tensor,
    count: cutlass.Int32,
    hist: cute.Tensor,
    scratch: cute.Tensor,
    thread: cutlass.Int32,
    warp: cutlass.Int32,
    lane: cutlass.Int32,
    lane_lt: cutlass.Uint32,
    column_limit: cutlass.Int32,
) -> cutlass.Uint64:
    """Keep the TOPK largest keys of `keys[0:count]` in place. Returns the new threshold.

    Every thread must call it. Ends in a barrier.
    """
    mine = cute.make_rmem_tensor(cute.make_layout((PRUNE_SLOTS,)), cutlass.Uint64)
    for j in cutlass.range_constexpr(PRUNE_SLOTS):
        mine[j] = cutlass.Uint64(0)
        if j * SCORE_THREADS + thread < count:
            mine[j] = keys[j * SCORE_THREADS + thread]
    threshold = select_threshold(
        mine, hist, warp, lane, lane_lt, cutlass.Int32(TOPK), column_limit, PRUNE_SLOTS
    )
    masks = [
        cutlass.Uint32(
            cute.arch.vote_ballot_sync(
                (mine[j] >= threshold) & (mine[j] != cutlass.Uint64(0))
            )
        )
        for j in range(PRUNE_SLOTS)
    ]
    if lane == 0:
        for j in cutlass.range_constexpr(PRUNE_SLOTS):
            scratch[j * SCORE_WARPS + warp] = cutlass.Int32(cute.arch.popc(masks[j]))
    cute.arch.sync_threads()
    position = cutlass.Int32(0)
    for j in cutlass.range_constexpr(PRUNE_SLOTS):
        base = position
        for w in cutlass.range_constexpr(SCORE_WARPS):
            if w < warp:
                base += scratch[j * SCORE_WARPS + w]
            position += scratch[j * SCORE_WARPS + w]
        if ((masks[j] >> cutlass.Uint32(lane)) & cutlass.Uint32(1)) != cutlass.Uint32(
            0
        ):
            keys[base + cutlass.Int32(cute.arch.popc(masks[j] & lane_lt))] = mine[j]
    cute.arch.sync_threads()
    return threshold


@cute.jit
def sf_atom_word(index: cutlass.Int32) -> cutlass.Int32:
    """Word offset of row `index` (K groups 0..3) in the 128-row UE8M0 scale-factor atom.

    The tcgen05 scale-factor atom stores row `r` at byte `16 * (r % 32) + 4 * (r // 32)`
    plus the K group; this matches `blockscaled_utils.make_smem_layout_sfa` for K = 128.
    """
    return (index % 32) * 4 + index // 32


@cute.jit
def item_page(
    item: cutlass.Int32,
    begin: cutlass.Int32,
    row0: cutlass.Int32,
    row_pages: cute.Tensor,
    page_table: cute.Tensor,
    block_q: cutlass.Constexpr,
):
    """Work item -> (page slot, row in block, cache page or -1 when the row has no such page)."""
    p = begin + item // block_q
    r = item % block_q
    page = cutlass.Int32(-1)
    for q in cutlass.range_constexpr(block_q):
        if (r == q) & (p < row_pages[q]):
            page = page_table[row0 + q, p]
    return p, r, page


@cute.jit
def is_live(
    item: cutlass.Int32,
    begin: cutlass.Int32,
    row0: cutlass.Int32,
    row_pages: cute.Tensor,
    page_table: cute.Tensor,
    block_q: cutlass.Constexpr,
) -> cutlass.Boolean:
    """True when the item's row is the first row in the block that uses its cache page at this slot."""
    p, r, page = item_page(item, begin, row0, row_pages, page_table, block_q)
    live = page >= 0
    for q in cutlass.range_constexpr(block_q - 1):
        if (q < r) & (p < row_pages[q]):  # noqa: SIM102
            if page_table[row0 + q, p] == page:
                live = cutlass.Boolean(False)
    return live


@cute.jit
def next_live(
    item: cutlass.Int32,
    total: cutlass.Int32,
    begin: cutlass.Int32,
    row0: cutlass.Int32,
    row_pages: cute.Tensor,
    page_table: cute.Tensor,
    block_q: cutlass.Constexpr,
) -> cutlass.Int32:
    """The first live item after `item`, or `total`."""
    item = item + 1
    searching = item < total
    while searching:
        if is_live(item, begin, row0, row_pages, page_table, block_q):
            searching = cutlass.Boolean(False)
        else:
            item += 1
            searching = item < total
    return item


@cute.jit
def issue_load(
    page: cutlass.Int32,
    stage: cutlass.Int32,
    full_bar: cute.Pointer,
    tma_atom_a: cute.CopyAtom,
    tAgA: cute.Tensor,
    tAsA: cute.Tensor,
    tma_atom_sf: cute.CopyAtom,
    tSFgSF: cute.Tensor,
    tSFsSF: cute.Tensor,
):
    """Warp 0: TMA one cache page's payload and scale words into `stage`."""
    with cute.arch.elect_one():
        cute.arch.mbarrier_arrive_and_expect_tx(
            full_bar + stage, A_STAGE_BYTES + SF_WORDS * 4
        )
    cute.copy(
        tma_atom_a,
        tAgA[(None, page)],
        tAsA[(None, stage)],
        tma_bar_ptr=full_bar + stage,
    )
    cute.copy(
        tma_atom_sf,
        tSFgSF[(None, page)],
        tSFsSF[(None, stage)],
        tma_bar_ptr=full_bar + stage,
    )


@cute.kernel
def score_partial_kernel(
    tiled_mma: cute.TiledMma,
    tma_atom_a: cute.CopyAtom,
    mA: cute.Tensor,
    tma_atom_b: cute.CopyAtom,
    mB: cute.Tensor,
    tma_atom_sf: cute.CopyAtom,
    mSF: cute.Tensor,
    a_smem_layout: cute.ComposedLayout,
    b_smem_layout: cute.ComposedLayout,
    sfa_smem_layout: cute.Layout,
    sfb_smem_layout: cute.Layout,
    query_scales: cute.Tensor,
    weights: cute.Tensor,
    lengths: cute.Tensor,
    page_table: cute.Tensor,
    partial: cute.Tensor,
    block_q: cutlass.Constexpr,
):
    mma_n = block_q * HEADS
    split, row_block, _ = cute.arch.block_idx()
    thread, _, _ = cute.arch.thread_idx()
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    rows = lengths.shape[0]
    splits = partial.shape[1]
    row0 = row_block * block_q

    allocator = utils.SmemAllocator()
    sA = allocator.allocate_tensor(
        cutlass.Float4E2M1FN,
        a_smem_layout.outer,
        byte_alignment=1024,
        swizzle=a_smem_layout.inner,
    )
    sB = allocator.allocate_tensor(
        cutlass.Float4E2M1FN,
        b_smem_layout.outer,
        byte_alignment=1024,
        swizzle=b_smem_layout.inner,
    )
    sSFA = allocator.allocate_tensor(
        cutlass.Float8E8M0FNU, sfa_smem_layout, byte_alignment=128
    )
    sSFB = allocator.allocate_tensor(
        cutlass.Float8E8M0FNU, sfb_smem_layout, byte_alignment=128
    )
    sf_raw = allocator.allocate_tensor(
        cutlass.Int32, cute.make_layout((SF_WORDS, STAGES)), byte_alignment=128
    )
    keys = allocator.allocate_tensor(
        cutlass.Uint64, cute.make_layout((block_q * CAPACITY,)), byte_alignment=16
    )
    w_pairs = allocator.allocate_tensor(
        cutlass.Uint32, cute.make_layout((mma_n // 2,)), byte_alignment=16
    )
    hist = allocator.allocate_tensor(
        cutlass.Int32, cute.make_layout((HIST_SLOTS,)), byte_alignment=16
    )
    scratch = allocator.allocate_tensor(
        cutlass.Int32, cute.make_layout((SCORE_THREADS,)), byte_alignment=16
    )
    counts = allocator.allocate_tensor(
        cutlass.Int32, cute.make_layout((block_q,)), byte_alignment=16
    )
    thresholds = allocator.allocate_tensor(
        cutlass.Uint64, cute.make_layout((block_q,)), byte_alignment=16
    )
    full_bar = allocator.allocate_array(cutlass.Int64, STAGES, byte_alignment=8)
    q_bar = allocator.allocate_array(cutlass.Int64, 1, byte_alignment=8)
    mma_bar = allocator.allocate_array(cutlass.Int64, 1, byte_alignment=8)
    tmem_holding = allocator.allocate_array(cutlass.Int32, 1, byte_alignment=16)
    sfa_words = cute.make_tensor(
        cute.recast_ptr(sSFA.iterator, dtype=cutlass.Int32),
        cute.make_layout((SF_WORDS, STAGES), stride=(1, SF_WORDS)),
    )
    sfb_words = cute.make_tensor(
        cute.recast_ptr(sSFB.iterator, dtype=cutlass.Int32),
        cute.make_layout((SF_WORDS,)),
    )
    workspace = cute.make_tensor(
        cute.recast_ptr(partial.iterator, dtype=cutlass.Uint64), partial.layout
    )

    if warp == 0:
        with cute.arch.elect_one():
            for stage in cutlass.range_constexpr(STAGES):
                cute.arch.mbarrier_init(full_bar + stage, 1)
            cute.arch.mbarrier_init(q_bar, 1)
            cute.arch.mbarrier_init(mma_bar, 1)
    cute.arch.mbarrier_init_fence()
    cute.arch.sync_threads()

    # Row facts, uniform across the CTA.
    row_len = cute.make_rmem_tensor(cute.make_layout((block_q,)), cutlass.Int32)
    row_pages = cute.make_rmem_tensor(cute.make_layout((block_q,)), cutlass.Int32)
    max_pages = cutlass.Int32(0)
    for r in cutlass.range_constexpr(block_q):
        length = cutlass.Int32(0)
        if row0 + r < rows:
            length = lengths[row0 + r]
        row_len[r] = length
        row_pages[r] = cute.ceil_div(length, PAGE_TOKENS)
        max_pages = cutlass.max(max_pages, row_pages[r])
    begin = (max_pages * split) // splits
    end = (max_pages * (split + 1)) // splits

    # Head weights (BF16 pairs) and the query scale-factor atom.
    row = row0 + thread // HEADS
    head = thread % HEADS
    scale_word = cutlass.Int32(0)
    if row < rows:
        scale_word = query_scales[row, head]
    if thread < mma_n:
        sfb_words[sf_atom_word(thread)] = scale_word
    if thread < mma_n // 2:
        pair_row = row0 + (2 * thread) // HEADS
        pair_head = (2 * thread) % HEADS
        weight_lo = cutlass.Float32(0)
        weight_hi = cutlass.Float32(0)
        if pair_row < rows:
            weight_lo = weights[pair_row, pair_head]
            weight_hi = weights[pair_row, pair_head + 1]
        w_pairs[thread] = pack_bf16_pair(weight_lo, weight_hi)
    if thread < block_q:
        counts[thread] = cutlass.Int32(0)
        thresholds[thread] = cutlass.Uint64(0)
    for j in cutlass.range_constexpr(cute.ceil_div(HIST_SLOTS, SCORE_THREADS)):
        if j * SCORE_THREADS + thread < HIST_SLOTS:
            hist[j * SCORE_THREADS + thread] = cutlass.Int32(0)
    column_limit = page_table.shape[1] * PAGE_TOKENS
    cute.arch.fence_view_async_shared()

    # MMA partitions.
    mma_tiler = (MMA_M, mma_n, MMA_K)
    thr_mma = tiled_mma.get_slice(0)
    gA = cute.local_tile(
        mA, cute.slice_(mma_tiler, (None, 0, None)), (None, None, None)
    )
    gB = cute.local_tile(
        mB, cute.slice_(mma_tiler, (0, None, None)), (None, None, None)
    )
    tCgA = thr_mma.partition_A(gA)
    tCgB = thr_mma.partition_B(gB)
    tCrA = tiled_mma.make_fragment_A(sA)
    tCrB = tiled_mma.make_fragment_B(sB)
    tAsA, tAgA = cpasync.tma_partition(
        tma_atom_a,
        0,
        cute.make_layout(1),
        cute.group_modes(sA, 0, 3),
        cute.group_modes(tCgA, 0, 3),
    )
    tBsB, tBgB = cpasync.tma_partition(
        tma_atom_b,
        0,
        cute.make_layout(1),
        cute.group_modes(sB, 0, 3),
        cute.group_modes(tCgB, 0, 3),
    )
    gSF = cute.local_tile(mSF, (SF_WORDS,), (None, None))
    tSFsSF, tSFgSF = cpasync.tma_partition(
        tma_atom_sf, 0, cute.make_layout(1), sf_raw, gSF
    )
    tAgA = tAgA[(None, 0, 0, None)]
    tSFgSF = tSFgSF[(None, 0, None)]
    acc_shape = tiled_mma.partition_shape_C((MMA_M, mma_n))
    tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)
    tmem = utils.TmemAllocator(
        tmem_holding,
        barrier_for_retrieve=pipeline.NamedBarrier(
            barrier_id=1, num_threads=SCORE_THREADS
        ),
    )
    tmem_pool = tmem.reserve(TMEM_COLUMNS[block_q])
    tmem.relinquish_alloc_permit()
    tCtAcc = tmem_pool.allocate_tensor(tCtAcc_fake.layout, cutlass.Float32)
    tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
        tiled_mma,
        mma_tiler,
        GROUP,
        cute.slice_(sfa_smem_layout, (None, None, None, 0)),
    )
    tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
        tiled_mma,
        mma_tiler,
        GROUP,
        cute.slice_(sfb_smem_layout, (None, None, None, 0)),
    )
    tCtSFA = tmem_pool.allocate_tensor(tCtSFA_layout, cutlass.Float8E8M0FNU)
    tCtSFB = tmem_pool.allocate_tensor(tCtSFB_layout, cutlass.Float8E8M0FNU)
    copy_atom_s2t = cute.make_copy_atom(
        tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.ONE), cutlass.Float8E8M0FNU
    )
    tCsSFA_compact = cute.filter_zeros(sSFA)
    tCtSFA_compact = cute.filter_zeros(tCtSFA)
    tiled_copy_s2t_sfa = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSFA_compact)
    thr_s2t_sfa = tiled_copy_s2t_sfa.get_slice(0)
    tCsSFA = tcgen05.get_s2t_smem_desc_tensor(
        tiled_copy_s2t_sfa, thr_s2t_sfa.partition_S(tCsSFA_compact)
    )
    tCtSFA_p = thr_s2t_sfa.partition_D(tCtSFA_compact)
    tCsSFB_compact = cute.filter_zeros(sSFB)
    tCtSFB_compact = cute.filter_zeros(tCtSFB)
    tiled_copy_s2t_sfb = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSFB_compact)
    thr_s2t_sfb = tiled_copy_s2t_sfb.get_slice(0)
    tCsSFB = tcgen05.get_s2t_smem_desc_tensor(
        tiled_copy_s2t_sfb, thr_s2t_sfb.partition_S(tCsSFB_compact)
    )
    tCtSFB_p = thr_s2t_sfb.partition_D(tCtSFB_compact)

    # Epilogue: one TMEM lane per thread, the 128 accumulator columns of one token.
    tAcc = tCtAcc[((None, None), 0, 0)]
    copy_atom_t2r = cute.make_copy_atom(
        tcgen05.Ld32x32bOp(T2R_REPETITION[block_q]), cutlass.Float32
    )
    tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc)
    thr_t2r = tiled_copy_t2r.get_slice(thread)
    tTR_tAcc = thr_t2r.partition_S(tAcc)
    cAcc = cute.make_identity_tensor((MMA_M, mma_n))
    tTR_cAcc = thr_t2r.partition_D(cAcc)
    tTR_rAcc = cute.make_rmem_tensor(tTR_cAcc.shape, cutlass.Float32)
    lane = tTR_cAcc[0][0]
    lane_id = cute.arch.lane_idx()
    lane_lt = cute.arch.lanemask_lt()

    # Query operand and its scales.
    if warp == 0:
        cpasync.prefetch_descriptor(tma_atom_a)
        cpasync.prefetch_descriptor(tma_atom_sf)
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(q_bar, mma_n * MMA_K // 2)
        cute.copy(
            tma_atom_b,
            tBgB[(None, row_block, 0, 0)],
            tBsB[(None, 0)],
            tma_bar_ptr=q_bar,
        )
    cute.arch.sync_threads()
    if warp == 0:
        cute.copy(tiled_copy_s2t_sfb, tCsSFB[(None, None, None, None, 0)], tCtSFB_p)
    cute.arch.mbarrier_wait(q_bar, 0)

    # Work items: (page slot p, row r) pairs in this split, one per distinct cache page.
    # `ring[stage]` is the item whose page is in flight or resident in `stage`;
    # loads run STAGES - 1 items ahead of the MMA.
    total = (end - begin) * block_q
    ring = cute.make_rmem_tensor(cute.make_layout((STAGES,)), cutlass.Int32)
    head = cutlass.Int32(0)
    if total > 0:  # noqa: SIM102
        if not is_live(head, begin, row0, row_pages, page_table, block_q):
            head = next_live(head, total, begin, row0, row_pages, page_table, block_q)
    for stage in cutlass.range_constexpr(STAGES - 1):
        ring[stage] = head
        if head < total:
            if warp == 0:
                p, r, page = item_page(
                    head, begin, row0, row_pages, page_table, block_q
                )
                issue_load(
                    page,
                    stage,
                    full_bar,
                    tma_atom_a,
                    tAgA,
                    tAsA,
                    tma_atom_sf,
                    tSFgSF,
                    tSFsSF,
                )
            head = next_live(head, total, begin, row0, row_pages, page_table, block_q)
    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
    iteration = cutlass.Int32(0)
    item = ring[0]
    while item < total:
        stage = iteration % STAGES
        refill = (iteration + STAGES - 1) % STAGES
        ring[refill] = head
        if head < total:
            if warp == 0:
                p, r, page = item_page(
                    head, begin, row0, row_pages, page_table, block_q
                )
                issue_load(
                    page,
                    refill,
                    full_bar,
                    tma_atom_a,
                    tAgA,
                    tAsA,
                    tma_atom_sf,
                    tSFgSF,
                    tSFsSF,
                )
            head = next_live(head, total, begin, row0, row_pages, page_table, block_q)
        # Resolve which rows use this page and which tokens may score before the
        # page lands, so the page-table loads overlap the TMA and the MMA.
        p, r, page = item_page(item, begin, row0, row_pages, page_table, block_q)
        token = p * PAGE_TOKENS + lane
        uses = []
        allowed = []
        for q in cutlass.range_constexpr(block_q):
            uses_page = cutlass.Boolean(False)
            if q == r:
                uses_page = cutlass.Boolean(True)
            elif (q > r) & (p < row_pages[q]):
                uses_page = page_table[row0 + q, p] == page
            eligible = uses_page & (token < row_len[q])
            uses.append(uses_page)
            allowed.append(eligible)
        cute.arch.mbarrier_wait(full_bar + stage, (iteration // STAGES) & 1)
        # Relay the page's plain scale words into the tcgen05 scale-factor atom.
        sfa_words[sf_atom_word(thread), stage] = sf_raw[thread, stage]
        cute.arch.fence_view_async_shared()
        cute.arch.sync_threads()
        if warp == 0:
            cute.copy(
                tiled_copy_s2t_sfa, tCsSFA[(None, None, None, None, stage)], tCtSFA_p
            )
            cute.gemm(
                tiled_mma,
                tCtAcc,
                [tCrA[(None, None, None, stage)], tCtSFA],
                [tCrB[(None, None, None, 0)], tCtSFB],
                tCtAcc,
            )
            with cute.arch.elect_one():
                tcgen05.commit(mma_bar)
        cute.arch.mbarrier_wait(mma_bar, iteration & 1)
        cute.copy(tiled_copy_t2r, tTR_tAcc, tTR_rAcc)
        cute.arch.fence_view_async_tmem_load()
        for q in cutlass.range_constexpr(block_q):
            if uses[q]:
                key = cutlass.Uint64(0)
                if allowed[q]:
                    total_score = cutlass.Float32(0)
                    for h in cutlass.range_constexpr(0, HEADS, 2):
                        dots = relu_bf16_pair(
                            tTR_rAcc[q * HEADS + h], tTR_rAcc[q * HEADS + h + 1]
                        )
                        total_score = add_bf16_pair(
                            total_score,
                            mul_bf16_pair(dots, w_pairs[(q * HEADS + h) // 2]),
                        )
                    score = bf16_round(total_score)
                    key = ordered_key(score.bitcast(cutlass.Uint32), token)
                row_keys = cute.make_tensor(
                    keys.iterator + q * CAPACITY, cute.make_layout((CAPACITY,))
                )
                insert_key(
                    key, row_keys, counts.iterator + q, thresholds[q], lane_id, lane_lt
                )
        cute.arch.sync_threads()
        for q in cutlass.range_constexpr(block_q):
            if counts[q] > PRUNE_AT:
                row_keys = cute.make_tensor(
                    keys.iterator + q * CAPACITY, cute.make_layout((CAPACITY,))
                )
                threshold = prune(
                    row_keys,
                    counts[q],
                    hist,
                    scratch,
                    thread,
                    warp,
                    lane_id,
                    lane_lt,
                    column_limit,
                )
                if thread == 0:
                    counts[q] = cutlass.Int32(TOPK)
                    thresholds[q] = threshold
                cute.arch.sync_threads()
        iteration += 1
        item = ring[iteration % STAGES]

    for q in cutlass.range_constexpr(block_q):
        if row0 + q < rows:
            row_keys = cute.make_tensor(
                keys.iterator + q * CAPACITY, cute.make_layout((CAPACITY,))
            )
            count = counts[q]
            if count > TOPK:
                prune(
                    row_keys,
                    count,
                    hist,
                    scratch,
                    thread,
                    warp,
                    lane_id,
                    lane_lt,
                    column_limit,
                )
                count = cutlass.Int32(TOPK)
            for index in cutlass.range_constexpr(TOPK // SCORE_THREADS):
                slot = index * SCORE_THREADS + thread
                value = cutlass.Int64(0)
                if slot < count:
                    value = cutlass.Int64(row_keys[slot])
                workspace[row0 + q, split, slot] = value

    cute.arch.sync_threads()
    tmem.free(tmem_pool.base_ptr)


@cute.jit
def launch_score_partial(
    block_q: cutlass.Constexpr,
    query: cute.Tensor,
    query_scales: cute.Tensor,
    weights: cute.Tensor,
    cache: cute.Tensor,
    lengths: cute.Tensor,
    page_table: cute.Tensor,
    partial: cute.Tensor,
    stream: cuda.CUstream,
):
    """Launch `score_partial_kernel` with `block_q` rows per CTA. Grid: (splits, ceil(rows / block_q))."""
    mma_n = block_q * HEADS
    rows = lengths.shape[0]
    pages = cache.shape[0]
    fp4_cache = cute.recast_tensor(cache, cutlass.Float4E2M1FN)
    fp4_query = cute.recast_tensor(query, cutlass.Float4E2M1FN)
    # A: (tokens, dims, pages); B: (rows x heads, dims, 1); SF: (words, pages).
    mA = cute.make_tensor(
        fp4_cache.iterator,
        cute.make_layout((MMA_M, MMA_K, pages), stride=(MMA_K, 1, PAGE_BYTES * 2)),
    )
    mB = cute.make_tensor(
        fp4_query.iterator,
        cute.make_layout((rows * HEADS, MMA_K, 1), stride=(MMA_K, 1, MMA_K)),
    )
    mSF = cute.make_tensor(
        cute.recast_ptr(cache.iterator, dtype=cutlass.Int32) + PAYLOAD_WORDS,
        cute.make_layout((SF_WORDS, pages), stride=(1, PAGE_WORDS)),
    )
    tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
        cutlass.Float4E2M1FN,
        cutlass.Float4E2M1FN,
        OperandMajorMode.K,
        OperandMajorMode.K,
        cutlass.Float8E8M0FNU,
        GROUP,
        tcgen05.CtaGroup.ONE,
        (MMA_M, mma_n),
    )
    mma_tiler = (MMA_M, mma_n, MMA_K)
    a_smem_layout = sm100_utils.make_smem_layout_a(
        tiled_mma, mma_tiler, cutlass.Float4E2M1FN, STAGES
    )
    b_smem_layout = sm100_utils.make_smem_layout_b(
        tiled_mma, mma_tiler, cutlass.Float4E2M1FN, 1
    )
    sfa_smem_layout = blockscaled_utils.make_smem_layout_sfa(
        tiled_mma, mma_tiler, GROUP, STAGES
    )
    sfb_smem_layout = blockscaled_utils.make_smem_layout_sfb(
        tiled_mma, mma_tiler, GROUP, 1
    )
    tma_op = cpasync.CopyBulkTensorTileG2SOp()
    tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
        tma_op,
        mA,
        cute.slice_(a_smem_layout, (None, None, None, 0)),
        mma_tiler,
        tiled_mma,
        (1, 1, 1, 1),
    )
    tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
        tma_op,
        mB,
        cute.slice_(b_smem_layout, (None, None, None, 0)),
        mma_tiler,
        tiled_mma,
        (1, 1, 1, 1),
    )
    tma_atom_sf, tma_tensor_sf = cpasync.make_tiled_tma_atom(
        tma_op,
        mSF,
        cute.make_layout((SF_WORDS,)),
        (SF_WORDS,),
    )
    score_partial_kernel(
        tiled_mma,
        tma_atom_a,
        tma_tensor_a,
        tma_atom_b,
        tma_tensor_b,
        tma_atom_sf,
        tma_tensor_sf,
        a_smem_layout,
        b_smem_layout,
        sfa_smem_layout,
        sfb_smem_layout,
        query_scales,
        weights,
        lengths,
        page_table,
        partial,
        block_q,
    ).launch(
        grid=(partial.shape[1], cute.ceil_div(rows, block_q), 1),
        block=(SCORE_THREADS, 1, 1),
        smem=score_smem_bytes(block_q),
        stream=stream,
    )


@cute.jit
def score_partial(
    query: cute.Tensor,
    query_scales: cute.Tensor,
    weights: cute.Tensor,
    cache: cute.Tensor,
    lengths: cute.Tensor,
    page_table: cute.Tensor,
    partial: cute.Tensor,
    stream: cuda.CUstream,
):
    """Write each (row, split)'s best 512 keys to `partial`, BLOCK_Q rows per CTA.

    Rows that share a cache page read it once, which suits a prefill batch.
    """
    launch_score_partial(
        BLOCK_Q,
        query,
        query_scales,
        weights,
        cache,
        lengths,
        page_table,
        partial,
        stream,
    )


@cute.jit
def score_partial_decode(
    query: cute.Tensor,
    query_scales: cute.Tensor,
    weights: cute.Tensor,
    cache: cute.Tensor,
    lengths: cute.Tensor,
    page_table: cute.Tensor,
    partial: cute.Tensor,
    stream: cuda.CUstream,
):
    """`score_partial` with one row per CTA, so that a batch of at most DECODE_MAX_ROWS rows fills the machine."""
    launch_score_partial(
        DECODE_BLOCK_Q,
        query,
        query_scales,
        weights,
        cache,
        lengths,
        page_table,
        partial,
        stream,
    )


@cute.jit
def merge_exact(
    keys: cute.Tensor,
    total: cutlass.Int32,
    hist: cute.Tensor,
    cand: cute.Tensor,
    thread: cutlass.Int32,
    warp: cutlass.Int32,
    lane: cutlass.Int32,
    lane_lt: cutlass.Uint32,
    column_limit: cutlass.Int32,
) -> cutlass.Int32:
    """Select the exact TOPK keys from `keys[0:total]` in register-resident chunks.

    `cand` carries winners between chunks. `hist[270]` counts them.
    The function returns the number kept. Every thread must call it.
    """
    kept = cutlass.Int32(0)
    mine = cute.make_rmem_tensor(cute.make_layout((MERGE_SLOTS,)), cutlass.Uint64)
    for chunk in range(cute.ceil_div(total, MERGE_CHUNK)):
        base = chunk * MERGE_CHUNK
        for j in cutlass.range_constexpr(MERGE_KEYS_PER_THREAD):
            index = base + j * MERGE_THREADS + thread
            mine[j] = cutlass.Uint64(0)
            if index < total:
                mine[j] = cutlass.Uint64(keys[index])
        mine[MERGE_KEYS_PER_THREAD] = cutlass.Uint64(0)
        if thread < kept:
            mine[MERGE_KEYS_PER_THREAD] = cand[thread]
        cute.arch.sync_threads()
        if thread == 0:
            hist[270] = cutlass.Int32(0)
        threshold = select_threshold(
            mine,
            hist,
            warp,
            lane,
            lane_lt,
            cutlass.Int32(TOPK),
            column_limit,
            MERGE_SLOTS,
        )
        for j in cutlass.range_constexpr(MERGE_SLOTS):
            key = mine[j]
            keep = (key >= threshold) & (key != cutlass.Uint64(0))
            slot = warp_append(keep, hist.iterator + 270, lane, lane_lt)
            if keep & (slot < TOPK):
                cand[slot] = key
        cute.arch.sync_threads()
        kept = cutlass.min(hist[270], cutlass.Int32(TOPK))
    return kept


@cute.kernel
def merge_pages_kernel(
    partial: cute.Tensor,
    page_table: cute.Tensor,
    page_indices: cute.Tensor,
    raw_indices: cute.Tensor | None,
):
    row, _, _ = cute.arch.block_idx()
    thread, _, _ = cute.arch.thread_idx()
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane = cute.arch.lane_idx()
    lane_lt = cute.arch.lanemask_lt()
    allocator = utils.SmemAllocator()
    hist = allocator.allocate_tensor(
        cutlass.Int32, cute.make_layout((HIST_SLOTS,)), byte_alignment=16
    )
    cand = allocator.allocate_tensor(
        cutlass.Uint64, cute.make_layout((MERGE_CAP,)), byte_alignment=16
    )
    cols = allocator.allocate_tensor(
        cutlass.Int32, cute.make_layout((TOPK,)), byte_alignment=16
    )
    total = partial.shape[1] * partial.shape[2]
    keys = cute.make_tensor(
        cute.recast_ptr(partial.iterator, dtype=cutlass.Uint64) + row * total,
        cute.make_layout((total,)),
    )
    column_limit = page_table.shape[1] * PAGE_TOKENS
    if thread < HIST_SLOTS:
        hist[thread] = cutlass.Int32(0)
    if thread < TOPK:
        page_indices[row, thread] = cutlass.Int32(-1)
        if cutlass.const_expr(raw_indices is not None):
            raw_indices[row, thread] = cutlass.Int32(-1)
    cute.arch.sync_threads()

    # A pivot from one sampled key per thread keeps about MERGE_TARGET keys.
    # Stream every key once and compact the keys above the pivot into `cand`;
    # the exact select then runs on that small buffer. When the pivot keeps too
    # few or too many keys, the chunked exact select runs instead.
    pivot = cutlass.Uint64(1)
    if total > MERGE_CAP:
        stride = total // MERGE_THREADS
        sample = cute.make_rmem_tensor(cute.make_layout((1,)), cutlass.Uint64)
        sample[0] = cutlass.Uint64(keys[thread * stride + thread % stride])
        wanted = cute.ceil_div(MERGE_TARGET * MERGE_THREADS, total)
        pivot = select_threshold(
            sample, hist, warp, lane, lane_lt, wanted, column_limit, 1
        )
    if thread == 0:
        hist[270] = cutlass.Int32(0)
    cute.arch.sync_threads()
    batch = cute.make_rmem_tensor(cute.make_layout((MERGE_BATCH,)), cutlass.Uint64)
    for step in range(cute.ceil_div(total, MERGE_BATCH * MERGE_THREADS)):
        base = step * MERGE_BATCH * MERGE_THREADS
        for j in cutlass.range_constexpr(MERGE_BATCH):
            index = base + j * MERGE_THREADS + thread
            batch[j] = cutlass.Uint64(0)
            if index < total:
                batch[j] = cutlass.Uint64(keys[index])
        for j in cutlass.range_constexpr(MERGE_BATCH):
            key = batch[j]
            keep = (key >= pivot) & (key != cutlass.Uint64(0))
            slot = warp_append(keep, hist.iterator + 270, lane, lane_lt)
            if keep & (slot < MERGE_CAP):
                cand[slot] = key
    cute.arch.sync_threads()
    count = hist[270]
    kept = cutlass.Int32(0)
    if total <= TOPK:
        # One split: every nonzero key is a winner, so `cand` already holds
        # the selection and no threshold is needed.
        kept = count
    elif (count <= MERGE_CAP) & ((count >= TOPK) | (pivot == cutlass.Uint64(1))):
        mine = cute.make_rmem_tensor(
            cute.make_layout((MERGE_CAP_SLOTS,)), cutlass.Uint64
        )
        for j in cutlass.range_constexpr(MERGE_CAP_SLOTS):
            mine[j] = cutlass.Uint64(0)
            if j * MERGE_THREADS + thread < count:
                mine[j] = cand[j * MERGE_THREADS + thread]
        cute.arch.sync_threads()
        if thread == 0:
            hist[270] = cutlass.Int32(0)
        threshold = select_threshold(
            mine,
            hist,
            warp,
            lane,
            lane_lt,
            cutlass.Int32(TOPK),
            column_limit,
            MERGE_CAP_SLOTS,
        )
        for j in cutlass.range_constexpr(MERGE_CAP_SLOTS):
            key = mine[j]
            keep = (key >= threshold) & (key != cutlass.Uint64(0))
            slot = warp_append(keep, hist.iterator + 270, lane, lane_lt)
            if keep & (slot < TOPK):
                cand[slot] = key
        cute.arch.sync_threads()
        kept = cutlass.min(hist[270], cutlass.Int32(TOPK))
    else:
        kept = merge_exact(
            keys, total, hist, cand, thread, warp, lane, lane_lt, column_limit
        )

    # Order the winning columns: each warp sorts 32 columns in registers, then
    # the rank of a column is the number of smaller columns over all 16 runs.
    # Empty slots hold INT_MAX and rank last.
    column = cutlass.Int32(0x7FFFFFFF)
    if thread < kept:
        column = key_column(cand[thread])
    for size in cutlass.range_constexpr(1, 6):
        for step in cutlass.range_constexpr(size - 1, -1, -1):
            other_col = cute.arch.shuffle_sync_bfly(column, 1 << step)
            swap = column > other_col
            if (((lane >> step) ^ (lane >> size)) & 1) != 0:
                swap = column < other_col
            if swap:
                column = other_col
    if thread < TOPK:
        cols[thread] = column
    cute.arch.sync_threads()
    element = thread % TOPK
    target = cols[element]
    below = cutlass.Int32(0)
    for run in cutlass.range_constexpr(TOPK // 64):
        run_base = (thread // TOPK) * (TOPK // 2) + run * 32
        count = cutlass.Int32(0)
        if cols[run_base + 31] < target:
            count = cutlass.Int32(32)
        else:
            for step in cutlass.range_constexpr(4, -1, -1):
                if cols[run_base + count + (1 << step) - 1] < target:
                    count += 1 << step
        below += count
    ranks = cute.make_tensor(
        cute.recast_ptr(cand.iterator, dtype=cutlass.Int32),
        cute.make_layout((2 * MERGE_THREADS,)),
    )
    ranks[thread] = below
    cute.arch.sync_threads()
    if thread < kept:
        rank = ranks[thread] + ranks[thread + TOPK]
        page_indices[row, rank] = (
            page_table[row, target // PAGE_TOKENS] * PAGE_TOKENS + target % PAGE_TOKENS
        )
        if cutlass.const_expr(raw_indices is not None):
            raw_indices[row, rank] = target


@cute.jit
def merge_pages(
    partial: cute.Tensor,
    page_table: cute.Tensor,
    page_indices: cute.Tensor,
    raw_indices: cute.Tensor | None,
    stream: cuda.CUstream,
):
    merge_pages_kernel(partial, page_table, page_indices, raw_indices).launch(
        grid=(partial.shape[0], 1, 1),
        block=(MERGE_THREADS, 1, 1),
        smem=MERGE_SMEM_BYTES,
        stream=stream,
    )


_FUSED_INDEXER_CACHE = get_jit_cache("dsv41_fused_indexer", source_paths=(__file__,))


def _compile_indexer(entrypoint: str, has_raw: bool):
    score_key = (entrypoint, has_raw)
    merge_key = ("merge_pages", has_raw)
    if score_key not in _FUSED_INDEXER_CACHE or merge_key not in _FUSED_INDEXER_CACHE:
        rows = cute.sym_int()
        splits = cute.sym_int()
        pages = cute.sym_int()
        table_width = cute.sym_int()
        output_width = cute.sym_int()

        query = cute.runtime.make_fake_compact_tensor(
            cutlass.Uint8, (rows, HEADS, TOKEN_WORDS * 4), stride_order=(2, 1, 0)
        )
        query_scales = cute.runtime.make_fake_compact_tensor(
            cutlass.Int32, (rows, HEADS), stride_order=(1, 0)
        )
        weights = cute.runtime.make_fake_compact_tensor(
            cutlass.Float32, (rows, HEADS), stride_order=(1, 0)
        )
        cache = cute.runtime.make_fake_compact_tensor(
            cutlass.Uint8, (pages, PAGE_BYTES), stride_order=(1, 0)
        )
        lengths = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (rows,))
        page_table = cute.runtime.make_fake_compact_tensor(
            cutlass.Int32, (rows, table_width), stride_order=(1, 0)
        )
        partial = cute.runtime.make_fake_compact_tensor(
            cutlass.Int64, (rows, splits, TOPK), stride_order=(2, 1, 0)
        )
        page_indices = cute.runtime.make_fake_tensor(
            cutlass.Int32,
            (rows, output_width),
            stride=(cute.sym_int64(), 1),
        )
        raw_indices = (
            cute.runtime.make_fake_tensor(
                cutlass.Int32,
                (rows, output_width),
                stride=(cute.sym_int64(), 1),
            )
            if has_raw
            else None
        )
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

        score_kernel = (
            score_partial_decode
            if entrypoint == "score_partial_decode"
            else score_partial
        )
        compiled_score = cute.compile(
            score_kernel,
            query,
            query_scales,
            weights,
            cache,
            lengths,
            page_table,
            partial,
            stream,
            options="--enable-tvm-ffi",
        )
        compiled_merge = cute.compile(
            merge_pages,
            partial,
            page_table,
            page_indices,
            raw_indices,
            stream,
            options="--enable-tvm-ffi",
        )
        _FUSED_INDEXER_CACHE[score_key] = compiled_score
        _FUSED_INDEXER_CACHE[merge_key] = compiled_merge
    return _FUSED_INDEXER_CACHE[score_key], _FUSED_INDEXER_CACHE[merge_key]


def fused_indexer_splits(rows: int, page_count: int, sm_count: int) -> int:
    block_q = 1 if rows <= 128 else 4
    ctas = max((rows + block_q - 1) // block_q, 1)
    return min(max(sm_count // ctas, 1), max(page_count, 1))


@functools.cache
def _device_sm_count(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def fused_indexer_topk(
    q_fp4: torch.Tensor,
    q_sf: torch.Tensor,
    weights: torch.Tensor,
    k_cache: torch.Tensor,
    lengths: torch.Tensor,
    page_table: torch.Tensor,
    page_indices: torch.Tensor,
    raw_indices: torch.Tensor | None = None,
) -> None:
    """Run the fused FP4 indexer and write the selected page and raw positions."""
    assert q_fp4.ndim == 3 and q_fp4.shape[1:] == (HEADS, TOKEN_WORDS * 4)
    assert q_fp4.dtype in (torch.uint8, torch.int8)
    rows = q_fp4.shape[0]
    assert q_sf.shape == (rows, HEADS) and q_sf.dtype == torch.int32
    assert weights.shape == (rows, HEADS) and weights.dtype == torch.float32
    assert k_cache.ndim == 2 and k_cache.shape[1] == PAGE_BYTES
    assert k_cache.dtype == torch.uint8
    assert lengths.shape == (rows,) and lengths.dtype == torch.int32
    assert page_table.ndim == 2 and page_table.shape[0] == rows
    assert page_table.dtype == torch.int32
    assert page_indices.ndim == 2 and page_indices.shape[0] == rows
    assert page_indices.shape[1] >= TOPK and page_indices.dtype == torch.int32
    assert page_indices.stride(1) == 1
    assert raw_indices is None or raw_indices.shape == page_indices.shape
    assert raw_indices is None or raw_indices.dtype == torch.int32
    assert raw_indices is None or raw_indices.stride(1) == 1
    assert q_fp4.is_cuda
    assert all(
        tensor.device == q_fp4.device
        for tensor in (q_sf, weights, k_cache, lengths, page_table, page_indices)
    )
    assert raw_indices is None or raw_indices.device == q_fp4.device
    if rows == 0:
        return

    device_index = (
        q_fp4.device.index
        if q_fp4.device.index is not None
        else torch.cuda.current_device()
    )
    splits = fused_indexer_splits(
        rows, page_table.shape[1], _device_sm_count(device_index)
    )
    query = q_fp4.contiguous().view(torch.uint8)
    query_scales = q_sf.contiguous()
    weights = weights.contiguous()
    cache = k_cache.contiguous()
    lengths = lengths.contiguous()
    page_table = page_table.contiguous()
    partial = torch.empty((rows, splits, TOPK), dtype=torch.int64, device=q_fp4.device)
    entrypoint = "score_partial_decode" if rows <= DECODE_MAX_ROWS else "score_partial"
    compiled_score, compiled_merge = _compile_indexer(
        entrypoint, raw_indices is not None
    )
    compiled_score(
        query,
        query_scales,
        weights,
        cache,
        lengths,
        page_table,
        partial,
    )
    output_page_indices = page_indices[:rows]
    output_raw_indices = raw_indices[:rows] if raw_indices is not None else None
    compiled_merge(
        partial,
        page_table,
        output_page_indices,
        output_raw_indices,
    )
