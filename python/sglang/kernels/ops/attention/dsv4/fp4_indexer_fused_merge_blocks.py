from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import arith as cutlass_arith
from cutlass.cute.runtime import make_fake_stream, make_fake_tensor
from cutlass.experimental import primitives as prims

NUM_MAIN_STAGES = 1
SMEM_HIST_OFF = 0
SMEM_HIST_STAGE_BYTES = 1024
SMEM_HIST_STRIDE = 1024
SMEM_CTL_OFF = 1024
SMEM_CTL_STAGE_BYTES = 16
SMEM_CTL_STRIDE = 16
SMEM_TOTAL = 1152
THREADS = 512
TARGET = 2048
STRIDE = 4096
MAP_PAGES = 0
TARGET_ARCH = "sm_100a"
SMEM_BYTES = 1152


@cute.kernel
def kernel_fp4_indexer_fused_merge_blocks(
    partial: cute.Pointer,
    pages: cute.Pointer,
    output: cute.Pointer,
    splits: cutlass.Int32,
    table_width: cutlass.Int32,
    out_stride: cutlass.Int32,
    out_offset: cutlass.Int32,
):
    tid = cutlass.Int32(cute.arch.thread_idx()[0])
    warp = cutlass.Int32(cute.arch.warp_idx())
    lane = cutlass.Int32(cute.arch.lane_idx())
    bid = cutlass.Int32(cute.arch.block_idx()[0])
    num_bids = cutlass.Int32(cute.arch.grid_dim()[0])
    blockIdx = cute.arch.block_idx()
    gridDim = cute.arch.grid_dim()
    smem_raw = cute.arch.get_dyn_smem(cutlass.Uint8, alignment=1024)
    smem = smem_raw.toint()
    _flat_layout = cute.make_layout((2147483647,), stride=(1,))
    _partial = cute.make_tensor(partial, _flat_layout)
    _pages = cute.make_tensor(pages, _flat_layout)
    _output = cute.make_tensor(output, _flat_layout)
    hist = cute.recast_ptr(smem_raw, dtype=cutlass.Int32)
    _hist = cute.make_tensor(hist, _flat_layout)
    hist_addr = smem + 0
    ctl = cute.recast_ptr(smem_raw + 1024, dtype=cutlass.Int32)
    _ctl = cute.make_tensor(ctl, _flat_layout)
    ctl_addr = smem + 1024
    v = cute.make_rmem_tensor((1,), cutlass.Int32)
    prefix = cute.make_rmem_tensor((1,), cutlass.Uint64)
    mask = cute.make_rmem_tensor((1,), cutlass.Uint64)
    need = cute.make_rmem_tensor((1,), cutlass.Int32)
    done = cute.make_rmem_tensor((1,), cutlass.Int32)
    above = cute.make_rmem_tensor((1,), cutlass.Int32)
    sel = cute.make_rmem_tensor((1,), cutlass.Int32)
    keep = cute.make_rmem_tensor((1,), cutlass.Int32)
    first = cute.make_rmem_tensor((1,), cutlass.Int32)
    v_1 = cute.make_rmem_tensor((1,), cutlass.Int32)
    row = cutlass.Int32(bid)
    tid_0 = cutlass.Int32(tid)
    lane_lt = cutlass.Uint32((2147483647 >> (31 - lane)))
    base = cutlass.Int32(((row * splits) * STRIDE))
    out = cutlass.Int32(((row * out_stride) + out_offset))
    if splits == 1:
        for c in cutlass.range(cutlass.Int32(0), cutlass.Int32(4), cutlass.Int32(1)):
            idx = cutlass.Int32(((c * 512) + tid_0))
            k = cutlass.Uint64(cutlass.Uint64(_partial[(base + idx)]))
            v[0] = cutlass.Int32(-1)
            if k != 0:
                pos = cutlass.Int32(
                    cutlass.Int32(
                        (cutlass.Uint32(4294967295) - cutlass.Uint32((k & 4294967295)))
                    )
                )
                if MAP_PAGES:
                    v[0] = cutlass.Int32(
                        (
                            (
                                _pages[
                                    (
                                        (row * table_width)
                                        + cutlass.Int32(
                                            cutlass_arith.divsi(
                                                cutlass.Int32(pos).ir_value(),
                                                cutlass.Int32(64).ir_value(),
                                            )
                                        )
                                    )
                                ]
                                * 64
                            )
                            + (pos % 64)
                        )
                    )
                else:
                    v[0] = cutlass.Int32(pos)
            prims.store_ext(cutlass.Int32(v[0]).ir_value(), (output + (out + idx)))
    else:
        if tid_0 == 0:
            prims.store_ext(cutlass.Int32(0).ir_value(), (ctl + (3)))
        prefix[0] = cutlass.Uint64(0)
        mask[0] = cutlass.Uint64(0)
        need[0] = cutlass.Int32(TARGET)
        done[0] = cutlass.Int32(0)
        for p in cutlass.range(cutlass.Int32(0), cutlass.Int32(6), cutlass.Int32(1)):
            if done[0] == 0:
                if tid_0 < 256:
                    prims.store_ext(cutlass.Int32(0).ir_value(), (hist + (tid_0)))
                cute.arch.sync_threads()
                for s in cutlass.range(
                    cutlass.Int32(0), cutlass.Int32(splits), cutlass.Int32(1)
                ):
                    for c_1 in cutlass.range(
                        cutlass.Int32(0), cutlass.Int32(4), cutlass.Int32(1)
                    ):
                        k_1 = cutlass.Uint64(
                            cutlass.Uint64(
                                _partial[
                                    (((base + (s * STRIDE)) + (c_1 * 512)) + tid_0)
                                ]
                            )
                        )
                        if (k_1 != 0) & ((k_1 & mask[0]) == prefix[0]):
                            cute.arch.red(
                                hist
                                + cutlass.Int32(
                                    (
                                        (k_1 >> cutlass.Uint64((48 - (8 * (p + 1)))))
                                        & 255
                                    )
                                ),
                                cutlass.Int32(1),
                                op="add",
                                dtype=cutlass.Int32,
                                sem="relaxed",
                                scope="cta",
                            )
                cute.arch.sync_threads()
                if tid_0 == 0:
                    above[0] = cutlass.Int32(0)
                    sel[0] = cutlass.Int32(-1)
                    for b in cutlass.range(
                        cutlass.Int32(0), cutlass.Int32(256), cutlass.Int32(1)
                    ):
                        digit = cutlass.Int32((255 - b))
                        count = cutlass.Int32(_hist[digit])
                        if (sel[0] < 0) & (need[0] <= (above[0] + count)):
                            sel[0] = cutlass.Int32(digit)
                            prims.store_ext(
                                cutlass.Int32((need[0] - above[0])).ir_value(),
                                (ctl + (1)),
                            )
                            prims.store_ext(
                                cutlass.Int32(count).ir_value(), (ctl + (2))
                            )
                        _if_condition_0 = cutlass.Boolean((sel[0] < 0))
                        above[0] = cutlass.Int32(
                            cutlass.select_(
                                _if_condition_0,
                                cutlass.Int32((above[0] + count)),
                                above[0],
                            )
                        )
                    prims.store_ext(cutlass.Int32(sel[0]).ir_value(), (ctl + (0)))
                cute.arch.sync_threads()
                sel_digit = cutlass.Int32(_ctl[0])
                sel_need = cutlass.Int32(_ctl[1])
                sel_count = cutlass.Int32(_ctl[2])
                if sel_digit < 0:
                    done[0] = cutlass.Int32(1)
                else:
                    prefix[0] = cutlass.Uint64(
                        (
                            prefix[0]
                            | (
                                cutlass.Uint64(sel_digit)
                                << cutlass.Uint64((48 - (8 * (p + 1))))
                            )
                        )
                    )
                    mask[0] = cutlass.Uint64(
                        (
                            mask[0]
                            | (
                                cutlass.Uint64(255)
                                << cutlass.Uint64((48 - (8 * (p + 1))))
                            )
                        )
                    )
                    need[0] = cutlass.Int32(sel_need)
                    _if_condition_1 = cutlass.Boolean((sel_count == sel_need))
                    done[0] = cutlass.Int32(
                        cutlass.select_(_if_condition_1, cutlass.Int32(1), done[0])
                    )
        cute.arch.sync_threads()
        for s_1 in cutlass.range(
            cutlass.Int32(0), cutlass.Int32(splits), cutlass.Int32(1)
        ):
            for c_2 in cutlass.range(
                cutlass.Int32(0), cutlass.Int32(4), cutlass.Int32(1)
            ):
                k_2 = cutlass.Uint64(
                    cutlass.Uint64(
                        _partial[(((base + (s_1 * STRIDE)) + (c_2 * 512)) + tid_0)]
                    )
                )
                keep[0] = cutlass.Int32(0)
                _if_condition_2 = cutlass.Boolean(
                    ((k_2 != 0) & (prefix[0] <= (k_2 & mask[0])))
                )
                keep[0] = cutlass.Int32(
                    cutlass.select_(_if_condition_2, cutlass.Int32(1), keep[0])
                )
                _vote_0 = cute.arch.vote_ballot_sync(
                    cutlass.Boolean((keep[0] == 1)), cutlass.Uint32(0xFFFFFFFF)
                ).bitcast(cutlass.Uint32)
                ballot = cutlass.Uint32(_vote_0)
                _popc_0 = cutlass.Int32(cute.arch.popc(cutlass.Uint32(ballot)))
                nkeep = cutlass.Uint32(cutlass.Uint32(_popc_0))
                if nkeep != 0:
                    _popc_1 = cutlass.Int32(
                        cute.arch.popc(cutlass.Uint32((ballot & lane_lt)))
                    )
                    rank = cutlass.Uint32(cutlass.Uint32(_popc_1))
                    first[0] = cutlass.Int32(0)
                    if lane == 0:
                        _atomic_old_0 = cute.arch.atomic_add(
                            ctl + 3,
                            cutlass.Int32(cutlass.Int32(nkeep)),
                            sem="relaxed",
                            scope="cta",
                        )
                        first[0] = cutlass.Int32(_atomic_old_0)
                    _shfl_0 = cute.arch.shuffle_sync(
                        first[0], 0, mask=0xFFFFFFFF, mask_and_clamp=31
                    )
                    first[0] = cutlass.Int32(_shfl_0)
                    if keep[0] == 1:
                        pos_1 = cutlass.Int32(
                            cutlass.Int32(
                                (
                                    cutlass.Uint32(4294967295)
                                    - cutlass.Uint32((k_2 & 4294967295))
                                )
                            )
                        )
                        v_1[0] = cutlass.Int32(pos_1)
                        if MAP_PAGES:
                            v_1[0] = cutlass.Int32(
                                (
                                    (
                                        _pages[
                                            (
                                                (row * table_width)
                                                + cutlass.Int32(
                                                    cutlass_arith.divsi(
                                                        cutlass.Int32(pos_1).ir_value(),
                                                        cutlass.Int32(64).ir_value(),
                                                    )
                                                )
                                            )
                                        ]
                                        * 64
                                    )
                                    + (pos_1 % 64)
                                )
                            )
                        prims.store_ext(
                            cutlass.Int32(v_1[0]).ir_value(),
                            (output + ((out + first[0]) + cutlass.Int32(rank))),
                        )
        cute.arch.sync_threads()
        total = cutlass.Int32(_ctl[3])
        for c_3 in cutlass.range(cutlass.Int32(0), cutlass.Int32(4), cutlass.Int32(1)):
            idx_1 = cutlass.Int32(((c_3 * 512) + tid_0))
            if idx_1 >= total:
                prims.store_ext(cutlass.Int32(-1).ir_value(), (output + (out + idx_1)))


@cute.jit
def launch_fp4_indexer_fused_merge_blocks(
    partial: cute.Tensor,
    pages: cute.Tensor,
    output: cute.Tensor,
    splits: cutlass.Int32,
    table_width: cutlass.Int32,
    out_stride: cutlass.Int32,
    out_offset: cutlass.Int32,
    grid_x: cutlass.Int32,
    grid_y: cutlass.Int32,
    grid_z: cutlass.Int32,
    stream: cuda.CUstream,
):
    kernel_fp4_indexer_fused_merge_blocks(
        partial.iterator,
        pages.iterator,
        output.iterator,
        splits,
        table_width,
        out_stride,
        out_offset,
    ).launch(
        grid=(grid_x, grid_y, grid_z),
        block=(512, 1, 1),
        smem=1152,
        min_blocks_per_mp=1,
        stream=stream,
    )


def compile_program():
    return cute.compile(
        launch_fp4_indexer_fused_merge_blocks,
        make_fake_tensor(
            cutlass.Int64, (cute.sym_int64(symbol="partial"),), (1,), assumed_align=16
        ),
        make_fake_tensor(
            cutlass.Int32, (cute.sym_int64(symbol="pages"),), (1,), assumed_align=16
        ),
        make_fake_tensor(
            cutlass.Int32, (cute.sym_int64(symbol="output"),), (1,), assumed_align=16
        ),
        cutlass.Int32(0),
        cutlass.Int32(0),
        cutlass.Int32(0),
        cutlass.Int32(0),
        cutlass.Int32(1),
        cutlass.Int32(1),
        cutlass.Int32(1),
        make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi --ptxas-options=--opt-level=2 --gpu-arch=sm_100a",
    )
