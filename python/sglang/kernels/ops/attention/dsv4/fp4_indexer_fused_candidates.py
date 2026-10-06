from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import arith as cutlass_arith
from cutlass.cute.runtime import make_fake_stream, make_fake_tensor
from cutlass.experimental import primitives as prims
from cutlass.experimental.cuda.tensor_map import (
    TensorMap,
    TensorMapFloatOOBFill,
    TensorMapL2Promotion,
    TensorMapSwizzle,
    create_tensor_map_tiled,
)

TMEM_NCOLS = 392
TMEM_ACC_TMEM_OFFSET = 0
TMEM_KVSF_TMEM_OFFSET = 384
TMEM_QSF_TMEM_OFFSET = 388
NUM_KV_PIPE_STAGES = 8
NUM_ACC_PIPE_STAGES = 3
NUM_SINGLE_PIPE_STAGES = 1
SMEM_SMEM_Q_OFF = 1024
SMEM_SMEM_Q_STAGE_BYTES = 8192
SMEM_SMEM_Q_STRIDE = 8704
SMEM_SMEM_QSF_OFF = 9216
SMEM_SMEM_QSF_STAGE_BYTES = 512
SMEM_SMEM_QSF_STRIDE = 8704
SMEM_SMEM_KV_OFF = 9728
SMEM_SMEM_KV_STAGE_BYTES = 8192
SMEM_SMEM_KV_STRIDE = 8704
SMEM_SMEM_KVSF_OFF = 17920
SMEM_SMEM_KVSF_STAGE_BYTES = 512
SMEM_SMEM_KVSF_STRIDE = 8704
SMEM_SMEM_KEYS_OFF = 79360
SMEM_SMEM_KEYS_STAGE_BYTES = 65536
SMEM_SMEM_KEYS_STRIDE = 65536
SMEM_SMEM_THR_OFF = 144896
SMEM_SMEM_THR_STAGE_BYTES = 64
SMEM_SMEM_THR_STRIDE = 64
SMEM_SMEM_CTL_OFF = 144960
SMEM_SMEM_CTL_STAGE_BYTES = 1264
SMEM_SMEM_CTL_STRIDE = 1264
SMEM_SMEM_BMAX_OFF = 146224
SMEM_SMEM_BMAX_STAGE_BYTES = 2048
SMEM_SMEM_BMAX_STRIDE = 2048
SMEM_TOTAL = 148352
THREADS = 640
CANDIDATES = 1
TARGET_ARCH = "sm_100a"
SMEM_BYTES = 148352


@cute.kernel
def kernel_fp4_indexer_fused_candidates(
    query: cutlass.GridConstant[TensorMap],
    query_scales: cutlass.GridConstant[TensorMap],
    cache: cutlass.GridConstant[TensorMap],
    weights: cute.Pointer,
    lengths: cute.Pointer,
    page_table: cute.Pointer,
    partial: cute.Pointer,
    block_partial: cute.Pointer,
    rows: cutlass.Int32,
    table_width: cutlass.Int32,
    splits: cutlass.Int32,
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
    _weights = cute.make_tensor(weights, _flat_layout)
    _lengths = cute.make_tensor(lengths, _flat_layout)
    _page_table = cute.make_tensor(page_table, _flat_layout)
    _partial = cute.make_tensor(partial, _flat_layout)
    _block_partial = cute.make_tensor(block_partial, _flat_layout)
    smem_q = cute.recast_ptr(
        smem_raw + 1024, swizzle_=cute.make_swizzle(4, 2, 3), dtype=cutlass.Uint8
    )
    _smem_q = cute.make_tensor(smem_q, _flat_layout)
    smem_q_addr = smem + 1024
    smem_qsf = cute.recast_ptr(smem_raw + 9216, dtype=cutlass.Uint8)
    _smem_qsf = cute.make_tensor(smem_qsf, _flat_layout)
    smem_qsf_addr = smem + 9216
    smem_kv = cute.recast_ptr(
        smem_raw + 9728, swizzle_=cute.make_swizzle(4, 2, 3), dtype=cutlass.Uint8
    )
    _smem_kv = cute.make_tensor(smem_kv, _flat_layout)
    smem_kv_addr = smem + 9728
    smem_kvsf = cute.recast_ptr(smem_raw + 17920, dtype=cutlass.Uint8)
    _smem_kvsf = cute.make_tensor(smem_kvsf, _flat_layout)
    smem_kvsf_addr = smem + 17920
    smem_keys = cute.recast_ptr(smem_raw + 79360, dtype=cutlass.Uint64)
    _smem_keys = cute.make_tensor(smem_keys, _flat_layout)
    smem_keys_addr = smem + 79360
    smem_thr = cute.recast_ptr(smem_raw + 144896, dtype=cutlass.Uint64)
    _smem_thr = cute.make_tensor(smem_thr, _flat_layout)
    smem_thr_addr = smem + 144896
    smem_ctl = cute.recast_ptr(smem_raw + 144960, dtype=cutlass.Int32)
    _smem_ctl = cute.make_tensor(smem_ctl, _flat_layout)
    smem_ctl_addr = smem + 144960
    smem_bmax = cute.recast_ptr(smem_raw + 146224, dtype=cutlass.Int32)
    _smem_bmax = cute.make_tensor(smem_bmax, _flat_layout)
    smem_bmax_addr = smem + 146224
    q_full_addr = cute.recast_ptr(smem_raw, dtype=cutlass.Uint64)
    kv_full_addr = cute.recast_ptr(smem_raw + 8, dtype=cutlass.Uint64)
    kv_empty_addr = cute.recast_ptr(smem_raw + 72, dtype=cutlass.Uint64)
    acc_full_addr = cute.recast_ptr(smem_raw + 136, dtype=cutlass.Uint64)
    acc_empty_addr = cute.recast_ptr(smem_raw + 160, dtype=cutlass.Uint64)
    if warp == 0:
        with cute.arch.elect_one():
            cute.arch.mbarrier_init(q_full_addr + 0, 1)
            cute.arch.mbarrier_init(kv_full_addr + 0, 2)
            cute.arch.mbarrier_init(kv_full_addr + 1, 2)
            cute.arch.mbarrier_init(kv_full_addr + 2, 2)
            cute.arch.mbarrier_init(kv_full_addr + 3, 2)
            cute.arch.mbarrier_init(kv_full_addr + 4, 2)
            cute.arch.mbarrier_init(kv_full_addr + 5, 2)
            cute.arch.mbarrier_init(kv_full_addr + 6, 2)
            cute.arch.mbarrier_init(kv_full_addr + 7, 2)
            cute.arch.mbarrier_init(kv_empty_addr + 0, 1)
            cute.arch.mbarrier_init(kv_empty_addr + 1, 1)
            cute.arch.mbarrier_init(kv_empty_addr + 2, 1)
            cute.arch.mbarrier_init(kv_empty_addr + 3, 1)
            cute.arch.mbarrier_init(kv_empty_addr + 4, 1)
            cute.arch.mbarrier_init(kv_empty_addr + 5, 1)
            cute.arch.mbarrier_init(kv_empty_addr + 6, 1)
            cute.arch.mbarrier_init(kv_empty_addr + 7, 1)
            cute.arch.mbarrier_init(acc_full_addr + 0, 2)
            cute.arch.mbarrier_init(acc_full_addr + 1, 2)
            cute.arch.mbarrier_init(acc_full_addr + 2, 2)
            cute.arch.mbarrier_init(acc_empty_addr + 0, 256)
            cute.arch.mbarrier_init(acc_empty_addr + 1, 256)
            cute.arch.mbarrier_init(acc_empty_addr + 2, 256)
    cute.arch.mbarrier_init_fence()
    cute.arch.sync_threads()
    _tmem_hold = cute.recast_ptr(smem_raw + 184, dtype=cutlass.Uint32)
    if warp == 0:
        cute.arch.alloc_tmem(512, _tmem_hold, is_two_cta=False, arch="sm_100")
    cute.arch.sync_threads()
    prims.tcgen05_fence("after_thread_sync")
    _tmem_base_ptr = cute.arch.retrieve_tmem_ptr(cutlass.Float32, 4, _tmem_hold)
    taddr = _tmem_base_ptr.toint()
    tmem_acc_tmem = cutlass.Int32(taddr)
    tmem_kvsf_tmem = cutlass.Int32(taddr + 384)
    tmem_qsf_tmem = cutlass.Int32(taddr + 388)
    # Ordered hardware-warpgroup register redistribution.
    if warp >= 0 and warp <= 3:
        cute.arch.setmaxregister_decrease(32)
    cute.arch.sync_threads()
    if warp == 0:
        len0 = cute.make_rmem_tensor((1,), cutlass.Int32)
        len1 = cute.make_rmem_tensor((1,), cutlass.Int32)
        len2 = cute.make_rmem_tensor((1,), cutlass.Int32)
        len3 = cute.make_rmem_tensor((1,), cutlass.Int32)
        kv_stage = cute.make_rmem_tensor((1,), cutlass.Uint32)
        pg0 = cute.make_rmem_tensor((1,), cutlass.Int32)
        pg1 = cute.make_rmem_tensor((1,), cutlass.Int32)
        pg2 = cute.make_rmem_tensor((1,), cutlass.Int32)
        pg3 = cute.make_rmem_tensor((1,), cutlass.Int32)
        _phase_kv_empty = cute.make_rmem_tensor((1,), cutlass.Uint32)
        np0 = cute.make_rmem_tensor((1,), cutlass.Int32)
        np1 = cute.make_rmem_tensor((1,), cutlass.Int32)
        np2 = cute.make_rmem_tensor((1,), cutlass.Int32)
        np3 = cute.make_rmem_tensor((1,), cutlass.Int32)
        lmask = cute.make_rmem_tensor((1,), cutlass.Int32)
        pg = cute.make_rmem_tensor((1,), cutlass.Int32)
        rmask = cute.make_rmem_tensor((1,), cutlass.Int32)
        row_block = cutlass.Int32(
            cutlass.Int32(
                cutlass_arith.divsi(
                    cutlass.Int32(bid).ir_value(), cutlass.Int32(splits).ir_value()
                )
            )
        )
        split = cutlass.Int32((bid % splits))
        row0 = cutlass.Int32((row_block * 4))
        len0[0] = cutlass.Int32(0)
        len1[0] = cutlass.Int32(0)
        len2[0] = cutlass.Int32(0)
        len3[0] = cutlass.Int32(0)
        if row0 < rows:
            len0[0] = cutlass.Int32(_lengths[row0])
        if (row0 + 1) < rows:
            len1[0] = cutlass.Int32(_lengths[(row0 + 1)])
        if (row0 + 2) < rows:
            len2[0] = cutlass.Int32(_lengths[(row0 + 2)])
        if (row0 + 3) < rows:
            len3[0] = cutlass.Int32(_lengths[(row0 + 3)])
        rp0 = cutlass.Int32(
            cutlass.Int32(
                cutlass_arith.divsi(
                    cutlass.Int32(((len0[0] + 128) - 1)).ir_value(),
                    cutlass.Int32(128).ir_value(),
                )
            )
        )
        rp1 = cutlass.Int32(
            cutlass.Int32(
                cutlass_arith.divsi(
                    cutlass.Int32(((len1[0] + 128) - 1)).ir_value(),
                    cutlass.Int32(128).ir_value(),
                )
            )
        )
        rp2 = cutlass.Int32(
            cutlass.Int32(
                cutlass_arith.divsi(
                    cutlass.Int32(((len2[0] + 128) - 1)).ir_value(),
                    cutlass.Int32(128).ir_value(),
                )
            )
        )
        rp3 = cutlass.Int32(
            cutlass.Int32(
                cutlass_arith.divsi(
                    cutlass.Int32(((len3[0] + 128) - 1)).ir_value(),
                    cutlass.Int32(128).ir_value(),
                )
            )
        )
        _max_0 = cutlass.max(rp0, rp1)
        _max_1 = cutlass.max(rp2, rp3)
        _max_2 = cutlass.max(_max_0, _max_1)
        max_pages = cutlass.Int32(_max_2)
        max_groups = cutlass.Int32(
            cutlass.Int32(
                cutlass_arith.divsi(
                    cutlass.Int32(((max_pages + 8) - 1)).ir_value(),
                    cutlass.Int32(8).ir_value(),
                )
            )
        )
        begin = cutlass.Int32(
            cutlass.Int32(
                cutlass_arith.divsi(
                    cutlass.Int32((max_groups * split)).ir_value(),
                    cutlass.Int32(splits).ir_value(),
                )
            )
        )
        end = cutlass.Int32(
            cutlass.Int32(
                cutlass_arith.divsi(
                    cutlass.Int32((max_groups * (split + 1))).ir_value(),
                    cutlass.Int32(splits).ir_value(),
                )
            )
        )
        if prims.elect_sync():
            cute.arch.mbarrier_arrive_and_expect_tx(q_full_addr, 8704)
            prims.cp_async_bulk_tensor_shared_cta_global(
                cute.make_ptr(
                    cutlass.Uint8,
                    cutlass.Uint32(smem_q_addr),
                    mem_space=cute.AddressSpace.smem,
                    assumed_align=16,
                ),
                query.get_ptr(),
                [cutlass.Int32(0), cutlass.Int32((row0 * 32))],
                q_full_addr,
                mode=prims.TMALoadMode.TILE,
            )
            prims.cp_async_bulk_tensor_shared_cta_global(
                cute.make_ptr(
                    cutlass.Uint8,
                    cutlass.Uint32(smem_qsf_addr),
                    mem_space=cute.AddressSpace.smem,
                    assumed_align=16,
                ),
                query_scales.get_ptr(),
                [cutlass.Int32(0), cutlass.Int32(row0)],
                q_full_addr,
                mode=prims.TMALoadMode.TILE,
            )
        tab0 = cutlass.Int32((row0 * table_width))
        tab1 = cutlass.Int32((tab0 + table_width))
        tab2 = cutlass.Int32((tab1 + table_width))
        tab3 = cutlass.Int32((tab2 + table_width))
        kv_stage[0] = cutlass.Uint32(0)
        pbegin = cutlass.Int32((begin * 8))
        _min_0 = cutlass.min((end * 8), max_pages)
        pstop = cutlass.Int32(_min_0)
        pg0[0] = cutlass.Int32(-1)
        pg1[0] = cutlass.Int32(-1)
        pg2[0] = cutlass.Int32(-1)
        pg3[0] = cutlass.Int32(-1)
        if pbegin < rp0:
            pg0[0] = cutlass.Int32(_page_table[(tab0 + pbegin)])
        if pbegin < rp1:
            pg1[0] = cutlass.Int32(_page_table[(tab1 + pbegin)])
        if pbegin < rp2:
            pg2[0] = cutlass.Int32(_page_table[(tab2 + pbegin)])
        if pbegin < rp3:
            pg3[0] = cutlass.Int32(_page_table[(tab3 + pbegin)])
        _phase_kv_empty[0] = cutlass.Uint32(1)
        for p in cutlass.range(
            cutlass.Int32(pbegin), cutlass.Int32(pstop), cutlass.Int32(1)
        ):
            nxt = cutlass.Int32((p + 1))
            np0[0] = cutlass.Int32(-1)
            np1[0] = cutlass.Int32(-1)
            np2[0] = cutlass.Int32(-1)
            np3[0] = cutlass.Int32(-1)
            if nxt < rp0:
                np0[0] = cutlass.Int32(_page_table[(tab0 + nxt)])
            if nxt < rp1:
                np1[0] = cutlass.Int32(_page_table[(tab1 + nxt)])
            if nxt < rp2:
                np2[0] = cutlass.Int32(_page_table[(tab2 + nxt)])
            if nxt < rp3:
                np3[0] = cutlass.Int32(_page_table[(tab3 + nxt)])
            lmask[0] = cutlass.Int32(0)
            _if_condition_0 = cutlass.Boolean((pg0[0] >= 0))
            lmask[0] = cutlass.Int32(
                cutlass.select_(_if_condition_0, cutlass.Int32(1), lmask[0])
            )
            _if_condition_1 = cutlass.Boolean(((pg1[0] >= 0) & (pg1[0] != pg0[0])))
            lmask[0] = cutlass.Int32(
                cutlass.select_(
                    _if_condition_1, cutlass.Int32((lmask[0] | 2)), lmask[0]
                )
            )
            _if_condition_2 = cutlass.Boolean(
                (((pg2[0] >= 0) & (pg2[0] != pg0[0])) & (pg2[0] != pg1[0]))
            )
            lmask[0] = cutlass.Int32(
                cutlass.select_(
                    _if_condition_2, cutlass.Int32((lmask[0] | 4)), lmask[0]
                )
            )
            _if_condition_3 = cutlass.Boolean(
                (
                    (((pg3[0] >= 0) & (pg3[0] != pg0[0])) & (pg3[0] != pg1[0]))
                    & (pg3[0] != pg2[0])
                )
            )
            lmask[0] = cutlass.Int32(
                cutlass.select_(
                    _if_condition_3, cutlass.Int32((lmask[0] | 8)), lmask[0]
                )
            )
            _popc_0 = cutlass.Int32(
                cute.arch.popc(
                    cutlass.Uint32(cutlass.Int32(lmask[0]).bitcast(cutlass.Uint32))
                )
            )
            nlive = cutlass.Int32(_popc_0)
            for it in cutlass.range(
                cutlass.Int32(0), cutlass.Int32(nlive), cutlass.Int32(1)
            ):
                _ffs_reversed_bits_4 = cute.arch.brev(
                    cutlass.Int32(lmask[0]).bitcast(cutlass.Uint32)
                )
                _ffs_leading_one_5 = cutlass.Int32(
                    cute.arch.bfind(_ffs_reversed_bits_4)
                )
                _ffs_0 = cutlass.Int32(
                    cutlass.select_(
                        cutlass.Int32(lmask[0]).bitcast(cutlass.Uint32)
                        == cutlass.Uint32(0),
                        cutlass.Int32(0),
                        cutlass.Int32(32) - _ffs_leading_one_5,
                    )
                )
                r = cutlass.Int32((_ffs_0 - 1))
                lmask[0] = cutlass.Int32((lmask[0] & (lmask[0] - 1)))
                pg[0] = cutlass.Int32(pg0[0])
                _if_condition_6 = cutlass.Boolean((r == 1))
                pg[0] = cutlass.Int32(
                    cutlass.select_(_if_condition_6, cutlass.Int32(pg1[0]), pg[0])
                )
                _if_condition_7 = cutlass.Boolean((r == 2))
                pg[0] = cutlass.Int32(
                    cutlass.select_(_if_condition_7, cutlass.Int32(pg2[0]), pg[0])
                )
                _if_condition_8 = cutlass.Boolean((r == 3))
                pg[0] = cutlass.Int32(
                    cutlass.select_(_if_condition_8, cutlass.Int32(pg3[0]), pg[0])
                )
                rmask[0] = cutlass.Int32(0)
                _if_condition_9 = cutlass.Boolean((pg0[0] == pg[0]))
                rmask[0] = cutlass.Int32(
                    cutlass.select_(_if_condition_9, cutlass.Int32(1), rmask[0])
                )
                _if_condition_10 = cutlass.Boolean((pg1[0] == pg[0]))
                rmask[0] = cutlass.Int32(
                    cutlass.select_(
                        _if_condition_10, cutlass.Int32((rmask[0] | 2)), rmask[0]
                    )
                )
                _if_condition_11 = cutlass.Boolean((pg2[0] == pg[0]))
                rmask[0] = cutlass.Int32(
                    cutlass.select_(
                        _if_condition_11, cutlass.Int32((rmask[0] | 4)), rmask[0]
                    )
                )
                _if_condition_12 = cutlass.Boolean((pg3[0] == pg[0]))
                rmask[0] = cutlass.Int32(
                    cutlass.select_(
                        _if_condition_12, cutlass.Int32((rmask[0] | 8)), rmask[0]
                    )
                )
                cute.arch.mbarrier_wait(kv_empty_addr + kv_stage[0], _phase_kv_empty[0])
                if prims.elect_sync():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        kv_full_addr + kv_stage[0], 8704
                    )
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        cute.make_ptr(
                            cutlass.Uint8,
                            cutlass.Uint32((smem_kv_addr + (kv_stage[0] * 8704))),
                            mem_space=cute.AddressSpace.smem,
                            assumed_align=16,
                        ),
                        cache.get_ptr(),
                        [cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(pg[0])],
                        kv_full_addr + kv_stage[0],
                        mode=prims.TMALoadMode.TILE,
                    )
                    prims.store_ext(
                        cutlass.Int32(((p << 4) | rmask[0])).ir_value(),
                        (smem_ctl + (308 + cutlass.Int32(kv_stage[0]))),
                    )
                    cute.arch.mbarrier_arrive(kv_full_addr + kv_stage[0])
                _advanced_stage_13 = cutlass.Uint32(kv_stage[0] + 1)
                _stage_wrapped_14 = cutlass.Boolean(_advanced_stage_13 == 8)
                kv_stage[0] = cutlass.Uint32(
                    cutlass.select_(_stage_wrapped_14, 0, _advanced_stage_13)
                )
                _phase_kv_empty[0] = cutlass.Uint32(
                    cutlass.select_(
                        _stage_wrapped_14,
                        _phase_kv_empty[0] ^ cutlass.Uint32(1),
                        _phase_kv_empty[0],
                    )
                )
            pg0[0] = cutlass.Int32(np0[0])
            pg1[0] = cutlass.Int32(np1[0])
            pg2[0] = cutlass.Int32(np2[0])
            pg3[0] = cutlass.Int32(np3[0])
        cute.arch.mbarrier_wait(kv_empty_addr + kv_stage[0], _phase_kv_empty[0])
        if prims.elect_sync():
            prims.store_ext(
                cutlass.Int32(-1).ir_value(),
                (smem_ctl + (308 + cutlass.Int32(kv_stage[0]))),
            )
            cute.arch.mbarrier_arrive(
                kv_full_addr + kv_stage[0], arrive_count=cutlass.Int32(2)
            )
    elif warp == 1:
        _phase_q_full_0 = cute.make_rmem_tensor((1,), cutlass.Uint32)
        kv_stage_m = cute.make_rmem_tensor((1,), cutlass.Uint32)
        acc_stage_m = cute.make_rmem_tensor((1,), cutlass.Uint32)
        marks_m = cute.make_rmem_tensor((1,), cutlass.Int32)
        rounds_m = cute.make_rmem_tensor((1,), cutlass.Int32)
        _phase_kv_full = cute.make_rmem_tensor((1,), cutlass.Uint32)
        _phase_acc_empty = cute.make_rmem_tensor((1,), cutlass.Uint32)
        flag_m = cute.make_rmem_tensor((1,), cutlass.Int32)
        pub_m = cute.make_rmem_tensor((1,), cutlass.Int32)
        _phase_q_full_0[0] = cutlass.Uint32(0)
        cute.arch.mbarrier_wait(q_full_addr, _phase_q_full_0[0])
        _phase_q_full_0[0] ^= cutlass.Uint32(1)
        cute.arch.fence_proxy("async.shared", space="cta")
        _sf_v = cute.make_rmem_tensor((4,), cutlass.Uint32)
        _smem_load_vec_15 = prims.load_ext(
            cute.make_ptr(
                cutlass.Int32,
                cutlass.Uint32(
                    (
                        smem_qsf_addr
                        + cutlass.Uint32((((((lane >> 3) ^ 0) * 32) + lane) * 4))
                    )
                ),
                mem_space=cute.AddressSpace.smem,
                assumed_align=4,
            ),
            dtype=cutlass.Int32,
        )
        _smem_load_vec_15 = _smem_load_vec_15.bitcast(cutlass.Uint32)
        _sf_v[0] = _smem_load_vec_15
        _smem_load_vec_16 = prims.load_ext(
            cute.make_ptr(
                cutlass.Int32,
                cutlass.Uint32(
                    (
                        smem_qsf_addr
                        + cutlass.Uint32((((((lane >> 3) ^ 1) * 32) + lane) * 4))
                    )
                ),
                mem_space=cute.AddressSpace.smem,
                assumed_align=4,
            ),
            dtype=cutlass.Int32,
        )
        _smem_load_vec_16 = _smem_load_vec_16.bitcast(cutlass.Uint32)
        _sf_v[1] = _smem_load_vec_16
        _smem_load_vec_17 = prims.load_ext(
            cute.make_ptr(
                cutlass.Int32,
                cutlass.Uint32(
                    (
                        smem_qsf_addr
                        + cutlass.Uint32((((((lane >> 3) ^ 2) * 32) + lane) * 4))
                    )
                ),
                mem_space=cute.AddressSpace.smem,
                assumed_align=4,
            ),
            dtype=cutlass.Int32,
        )
        _smem_load_vec_17 = _smem_load_vec_17.bitcast(cutlass.Uint32)
        _sf_v[2] = _smem_load_vec_17
        _smem_load_vec_18 = prims.load_ext(
            cute.make_ptr(
                cutlass.Int32,
                cutlass.Uint32(
                    (
                        smem_qsf_addr
                        + cutlass.Uint32((((((lane >> 3) ^ 3) * 32) + lane) * 4))
                    )
                ),
                mem_space=cute.AddressSpace.smem,
                assumed_align=4,
            ),
            dtype=cutlass.Int32,
        )
        _smem_load_vec_18 = _smem_load_vec_18.bitcast(cutlass.Uint32)
        _sf_v[3] = _smem_load_vec_18
        cute.arch.sync_warp()
        prims.store_ext(
            cutlass.Uint32(_sf_v[0]).ir_value(),
            cute.make_ptr(
                cutlass.Uint32,
                cutlass.Uint32(
                    (
                        smem_qsf_addr
                        + cutlass.Uint32((((lane * 4) + ((lane >> 3) ^ 0)) * 4))
                    )
                ),
                mem_space=cute.AddressSpace.smem,
                assumed_align=4,
            ),
        )
        prims.store_ext(
            cutlass.Uint32(_sf_v[1]).ir_value(),
            cute.make_ptr(
                cutlass.Uint32,
                cutlass.Uint32(
                    (
                        smem_qsf_addr
                        + cutlass.Uint32((((lane * 4) + ((lane >> 3) ^ 1)) * 4))
                    )
                ),
                mem_space=cute.AddressSpace.smem,
                assumed_align=4,
            ),
        )
        prims.store_ext(
            cutlass.Uint32(_sf_v[2]).ir_value(),
            cute.make_ptr(
                cutlass.Uint32,
                cutlass.Uint32(
                    (
                        smem_qsf_addr
                        + cutlass.Uint32((((lane * 4) + ((lane >> 3) ^ 2)) * 4))
                    )
                ),
                mem_space=cute.AddressSpace.smem,
                assumed_align=4,
            ),
        )
        prims.store_ext(
            cutlass.Uint32(_sf_v[3]).ir_value(),
            cute.make_ptr(
                cutlass.Uint32,
                cutlass.Uint32(
                    (
                        smem_qsf_addr
                        + cutlass.Uint32((((lane * 4) + ((lane >> 3) ^ 3)) * 4))
                    )
                ),
                mem_space=cute.AddressSpace.smem,
                assumed_align=4,
            ),
        )
        cute.arch.fence_proxy("async.shared", space="cta")
        _scale_copy_19_src_0 = cutlass.Uint64(cutlass.Uint32(smem_qsf_addr))
        _scale_copy_19_desc_0 = (
            (_scale_copy_19_src_0 >> 4) & cutlass.Uint64(0x3FFF)
        ) | cutlass.Uint64(0x400800000000)
        _scale_copy_19_dst_0 = prims.make_tmem_ptr(
            cutlass.Int32(tmem_qsf_tmem), cutlass.Int32
        )
        with cute.arch.elect_one():
            prims.tcgen05_cp(
                prims.Tcgen05CpShape.SHAPE_32X128B,
                _scale_copy_19_dst_0,
                _scale_copy_19_desc_0,
                group=prims.CTAGroup.CTA_1,
                multicast=prims.Tcgen05CpMulticast.WARPX4,
            )
        kv_stage_m[0] = cutlass.Uint32(0)
        acc_stage_m[0] = cutlass.Uint32(0)
        marks_m[0] = cutlass.Int32(0)
        rounds_m[0] = cutlass.Int32(0)
        if prims.elect_sync():
            prims.store_ext(cutlass.Int32(0).ir_value(), (smem_ctl + (300)))
            prims.store_ext(cutlass.Int32(0).ir_value(), (smem_ctl + (301)))
        _phase_kv_full[0] = cutlass.Uint32(0)
        _phase_acc_empty[0] = cutlass.Uint32(1)
        _loop_active_20 = cute.make_rmem_tensor((1,), cutlass.Boolean)
        _break_loop_counter_21 = cute.make_rmem_tensor((1,), cutlass.Int32)
        _break_loop_counter_21[0] = cutlass.Int32(cutlass.Int32(0))
        _loop_active_20[0] = cutlass.Boolean(
            _break_loop_counter_21[0] < cutlass.Int32(1073741824)
        )
        while _loop_active_20[0]:
            item_m = cutlass.Int32(_break_loop_counter_21[0])
            cute.arch.mbarrier_wait(kv_full_addr + kv_stage_m[0], _phase_kv_full[0])
            cute.arch.fence_proxy("async.shared", space="cta")
            desc_m = cutlass.Int32(_smem_ctl[(308 + cutlass.Int32(kv_stage_m[0]))])
            if desc_m == -1:
                _loop_active_20[0] = False
            if _loop_active_20[0]:
                sf_v = cute.make_rmem_tensor((4,), cutlass.Uint32)
                for i in cutlass.range(
                    cutlass.Int32(0), cutlass.Int32(4), cutlass.Int32(1)
                ):
                    _smem_load_vec_22 = prims.load_ext(
                        cute.make_ptr(
                            cutlass.Int32,
                            cutlass.Uint32(
                                (
                                    (smem_kvsf_addr + (kv_stage_m[0] * 8704))
                                    + cutlass.Uint32(
                                        (
                                            (((((lane >> 3) ^ i) * 32) + lane) * 4)
                                            ^ (
                                                (
                                                    (
                                                        (
                                                            (
                                                                (((lane >> 3) ^ i) * 32)
                                                                + lane
                                                            )
                                                            * 4
                                                        )
                                                        >> 7
                                                    )
                                                    & 3
                                                )
                                                << 4
                                            )
                                        )
                                    )
                                )
                            ),
                            mem_space=cute.AddressSpace.smem,
                            assumed_align=4,
                        ),
                        dtype=cutlass.Int32,
                    )
                    _smem_load_vec_22 = _smem_load_vec_22.bitcast(cutlass.Uint32)
                    sf_v[i] = _smem_load_vec_22
                cute.arch.sync_warp()
                for i_1 in cutlass.range(
                    cutlass.Int32(0), cutlass.Int32(4), cutlass.Int32(1)
                ):
                    prims.store_ext(
                        cutlass.Uint32(sf_v[i_1]).ir_value(),
                        cute.make_ptr(
                            cutlass.Uint32,
                            cutlass.Uint32(
                                (
                                    (smem_kvsf_addr + (kv_stage_m[0] * 8704))
                                    + cutlass.Uint32(
                                        (((lane * 4) + ((lane >> 3) ^ i_1)) * 4)
                                    )
                                )
                            ),
                            mem_space=cute.AddressSpace.smem,
                            assumed_align=4,
                        ),
                    )
                cute.arch.fence_proxy("async.shared", space="cta")
                cute.arch.mbarrier_wait(
                    acc_empty_addr + acc_stage_m[0], _phase_acc_empty[0]
                )
                prims.tcgen05_fence("after_thread_sync")
                flag_m[0] = cutlass.Int32(0)
                if marks_m[0] == 0:
                    if _smem_ctl[301] == rounds_m[0]:
                        flag_m[0] = cutlass.Int32(_smem_ctl[300])
                _if_condition_23 = cutlass.Boolean((flag_m[0] != 0))
                marks_m[0] = cutlass.Int32(
                    cutlass.select_(_if_condition_23, cutlass.Int32(2), marks_m[0])
                )
                rounds_m[0] = cutlass.Int32(
                    cutlass.select_(
                        _if_condition_23, cutlass.Int32((rounds_m[0] + 1)), rounds_m[0]
                    )
                )
                pub_m[0] = cutlass.Int32(desc_m)
                _if_condition_24 = cutlass.Boolean((marks_m[0] > 0))
                pub_m[0] = cutlass.Int32(
                    cutlass.select_(
                        _if_condition_24, cutlass.Int32((desc_m | 1073741824)), pub_m[0]
                    )
                )
                marks_m[0] = cutlass.Int32(
                    cutlass.select_(
                        _if_condition_24, cutlass.Int32((marks_m[0] - 1)), marks_m[0]
                    )
                )
                if prims.elect_sync():
                    if flag_m[0] != 0:
                        prims.store_ext(cutlass.Int32(0).ir_value(), (smem_ctl + (300)))
                    prims.store_ext(
                        cutlass.Int32(pub_m[0]).ir_value(),
                        (smem_ctl + (304 + cutlass.Int32(acc_stage_m[0]))),
                    )
                    cute.arch.mbarrier_arrive(acc_full_addr + acc_stage_m[0])
                _scale_copy_25_src_0 = cutlass.Uint64(
                    cutlass.Uint32((smem_kvsf_addr + (kv_stage_m[0] * 8704)))
                )
                _scale_copy_25_desc_0 = (
                    (_scale_copy_25_src_0 >> 4) & cutlass.Uint64(0x3FFF)
                ) | cutlass.Uint64(0x400800000000)
                _scale_copy_25_dst_0 = prims.make_tmem_ptr(
                    cutlass.Int32(tmem_kvsf_tmem), cutlass.Int32
                )
                with cute.arch.elect_one():
                    prims.tcgen05_cp(
                        prims.Tcgen05CpShape.SHAPE_32X128B,
                        _scale_copy_25_dst_0,
                        _scale_copy_25_desc_0,
                        group=prims.CTAGroup.CTA_1,
                        multicast=prims.Tcgen05CpMulticast.WARPX4,
                    )
                _mma_a_lo_0 = cute.arch.make_warp_uniform(
                    cutlass.Int32(
                        (((smem_kv_addr >> 4) & 0x3FFF) + kv_stage_m[0] * 544)
                    )
                )
                _mma_b_lo_0 = cute.arch.make_warp_uniform(
                    cutlass.Int32((((smem_q_addr >> 4) & 0x3FFF) + 0 * 544))
                )
                _mma_mxf4_26_d = prims.make_tmem_ptr(
                    cutlass.Int32((tmem_acc_tmem + (acc_stage_m[0] * 128))),
                    cutlass.Int32,
                )
                _mma_mxf4_26_sfa = prims.make_tmem_ptr(
                    cutlass.Int32(tmem_kvsf_tmem), cutlass.Int32
                )
                _mma_mxf4_26_sfb = prims.make_tmem_ptr(
                    cutlass.Int32(tmem_qsf_tmem), cutlass.Int32
                )
                _mma_mxf4_26_a = cutlass.Uint64(cutlass.Uint32(_mma_a_lo_0)) | (
                    cutlass.Uint64(0x80004020) << 32
                )
                _mma_mxf4_26_b = cutlass.Uint64(cutlass.Uint32(_mma_b_lo_0)) | (
                    cutlass.Uint64(0x80004020) << 32
                )
                if prims.elect_sync():
                    prims.tcgen05_mma_block_scale(
                        prims.Tcgen05MMAKind.MXF4,
                        prims.CTAGroup.CTA_1,
                        _mma_mxf4_26_d,
                        _mma_mxf4_26_a + 0,
                        _mma_mxf4_26_b + 0,
                        cutlass.Uint32(0x08A00480),
                        cutlass.Boolean((1) == 0),
                        _mma_mxf4_26_sfa,
                        _mma_mxf4_26_sfb,
                        scale_vec_size=prims.Tcgen05MMAScaleVecSize.BLOCK32,
                    )
                    prims.tcgen05_mma_block_scale(
                        prims.Tcgen05MMAKind.MXF4,
                        prims.CTAGroup.CTA_1,
                        _mma_mxf4_26_d,
                        _mma_mxf4_26_a + 2,
                        _mma_mxf4_26_b + 2,
                        cutlass.Uint32(0x48A004A0),
                        True,
                        _mma_mxf4_26_sfa,
                        _mma_mxf4_26_sfb,
                        scale_vec_size=prims.Tcgen05MMAScaleVecSize.BLOCK32,
                    )
                with cute.arch.elect_one():
                    cute.nvgpu.tcgen05.commit(
                        kv_empty_addr + kv_stage_m[0],
                        None,
                        cute.nvgpu.tcgen05.CtaGroup.ONE,
                    )
                with cute.arch.elect_one():
                    cute.nvgpu.tcgen05.commit(
                        acc_full_addr + acc_stage_m[0],
                        None,
                        cute.nvgpu.tcgen05.CtaGroup.ONE,
                    )
                _advanced_stage_27 = cutlass.Uint32(kv_stage_m[0] + 1)
                _stage_wrapped_28 = cutlass.Boolean(_advanced_stage_27 == 8)
                kv_stage_m[0] = cutlass.Uint32(
                    cutlass.select_(_stage_wrapped_28, 0, _advanced_stage_27)
                )
                _phase_kv_full[0] = cutlass.Uint32(
                    cutlass.select_(
                        _stage_wrapped_28,
                        _phase_kv_full[0] ^ cutlass.Uint32(1),
                        _phase_kv_full[0],
                    )
                )
                _advanced_stage_29 = cutlass.Uint32(acc_stage_m[0] + 1)
                _stage_wrapped_30 = cutlass.Boolean(_advanced_stage_29 == 3)
                acc_stage_m[0] = cutlass.Uint32(
                    cutlass.select_(_stage_wrapped_30, 0, _advanced_stage_29)
                )
                _phase_acc_empty[0] = cutlass.Uint32(
                    cutlass.select_(
                        _stage_wrapped_30,
                        _phase_acc_empty[0] ^ cutlass.Uint32(1),
                        _phase_acc_empty[0],
                    )
                )
            _break_loop_counter_21[0] += cutlass.Int32(1)
            if _break_loop_counter_21[0] >= cutlass.Int32(1073741824):
                _loop_active_20[0] = False
        end0 = cutlass.Uint32(acc_stage_m[0])
        cute.arch.mbarrier_wait(acc_empty_addr + acc_stage_m[0], _phase_acc_empty[0])
        _advanced_stage_31 = cutlass.Uint32(acc_stage_m[0] + 1)
        _stage_wrapped_32 = cutlass.Boolean(_advanced_stage_31 == 3)
        acc_stage_m[0] = cutlass.Uint32(
            cutlass.select_(_stage_wrapped_32, 0, _advanced_stage_31)
        )
        _phase_acc_empty[0] = cutlass.Uint32(
            cutlass.select_(
                _stage_wrapped_32,
                _phase_acc_empty[0] ^ cutlass.Uint32(1),
                _phase_acc_empty[0],
            )
        )
        end1 = cutlass.Uint32(acc_stage_m[0])
        cute.arch.mbarrier_wait(acc_empty_addr + acc_stage_m[0], _phase_acc_empty[0])
        _advanced_stage_33 = cutlass.Uint32(acc_stage_m[0] + 1)
        _stage_wrapped_34 = cutlass.Boolean(_advanced_stage_33 == 3)
        acc_stage_m[0] = cutlass.Uint32(
            cutlass.select_(_stage_wrapped_34, 0, _advanced_stage_33)
        )
        _phase_acc_empty[0] = cutlass.Uint32(
            cutlass.select_(
                _stage_wrapped_34,
                _phase_acc_empty[0] ^ cutlass.Uint32(1),
                _phase_acc_empty[0],
            )
        )
        cute.arch.mbarrier_wait(acc_empty_addr + acc_stage_m[0], _phase_acc_empty[0])
        if prims.elect_sync():
            prims.store_ext(
                cutlass.Int32(-1).ir_value(), (smem_ctl + (304 + cutlass.Int32(end0)))
            )
            prims.store_ext(
                cutlass.Int32(-1).ir_value(), (smem_ctl + (304 + cutlass.Int32(end1)))
            )
            cute.arch.mbarrier_arrive(
                acc_full_addr + end0, arrive_count=cutlass.Int32(2)
            )
            cute.arch.mbarrier_arrive(
                acc_full_addr + end1, arrive_count=cutlass.Int32(2)
            )
    elif warp >= 4 and warp <= 19:
        cute.arch.setmaxregister_increase(112)
        max_pages_1 = cute.make_rmem_tensor((1,), cutlass.Int32)
        next_g = cute.make_rmem_tensor((1,), cutlass.Int32)
        seen = cute.make_rmem_tensor((1,), cutlass.Int32)
        w_i = cute.make_rmem_tensor((1,), cutlass.Float32)
        acc_stage = cute.make_rmem_tensor((1,), cutlass.Uint32)
        skip = cute.make_rmem_tensor((1,), cutlass.Int32)
        flush = cute.make_rmem_tensor((1,), cutlass.Int32)
        _phase_acc_full = cute.make_rmem_tensor((1,), cutlass.Uint32)
        desc = cute.make_rmem_tensor((1,), cutlass.Int32)
        enter = cute.make_rmem_tensor((1,), cutlass.Int32)
        grp = cute.make_rmem_tensor((1,), cutlass.Int32)
        jt = cute.make_rmem_tensor((1,), cutlass.Int32)
        napp = cute.make_rmem_tensor((1,), cutlass.Int32)
        abits = cute.make_rmem_tensor((1,), cutlass.Uint32)
        bkey = cute.make_rmem_tensor((1,), cutlass.Uint64)
        aflip = cute.make_rmem_tensor((1,), cutlass.Uint32)
        bkeep = cute.make_rmem_tensor((1,), cutlass.Int32)
        bb = cute.make_rmem_tensor((1,), cutlass.Int32)
        s0 = cute.make_rmem_tensor((2,), cutlass.Float32)
        s1 = cute.make_rmem_tensor((2,), cutlass.Float32)
        hot = cute.make_rmem_tensor((1,), cutlass.Int32)
        key = cute.make_rmem_tensor((1,), cutlass.Uint64)
        kbits = cute.make_rmem_tensor((1,), cutlass.Uint32)
        keep = cute.make_rmem_tensor((1,), cutlass.Int32)
        base = cute.make_rmem_tensor((1,), cutlass.Int32)
        fin = cute.make_rmem_tensor((1,), cutlass.Int32)
        prefix = cute.make_rmem_tensor((1,), cutlass.Uint64)
        need = cute.make_rmem_tensor((1,), cutlass.Int32)
        done = cute.make_rmem_tensor((1,), cutlass.Int32)
        hi_mask = cute.make_rmem_tensor((1,), cutlass.Uint64)
        c = cute.make_rmem_tensor((1,), cutlass.Int32)
        incl = cute.make_rmem_tensor((1,), cutlass.Int32)
        ckeep = cute.make_rmem_tensor((1,), cutlass.Int32)
        cbase = cute.make_rmem_tensor((1,), cutlass.Int32)
        bprefix = cute.make_rmem_tensor((1,), cutlass.Uint64)
        bneed = cute.make_rmem_tensor((1,), cutlass.Int32)
        bdone = cute.make_rmem_tensor((1,), cutlass.Int32)
        bc = cute.make_rmem_tensor((1,), cutlass.Int32)
        bincl = cute.make_rmem_tensor((1,), cutlass.Int32)
        bckeep = cute.make_rmem_tensor((1,), cutlass.Int32)
        bcbase = cute.make_rmem_tensor((1,), cutlass.Int32)
        v = cute.make_rmem_tensor((1,), cutlass.Uint64)
        row_block_1 = cutlass.Int32(
            cutlass.Int32(
                cutlass_arith.divsi(
                    cutlass.Int32(bid).ir_value(), cutlass.Int32(splits).ir_value()
                )
            )
        )
        split_1 = cutlass.Int32((bid % splits))
        row0_1 = cutlass.Int32((row_block_1 * 4))
        bstride = cutlass.Int32((splits * 4096))
        bbase = cutlass.Int32((((row0_1 * splits) + split_1) * 4096))
        mw = cutlass.Int32((warp - 4))
        wg = cutlass.Int32(
            cutlass.Int32(
                cutlass_arith.divsi(
                    cutlass.Int32(mw).ir_value(), cutlass.Int32(8).ir_value()
                )
            )
        )
        wq = cutlass.Int32((warp % 4))
        rh = cutlass.Int32(
            (
                cutlass.Int32(
                    cutlass_arith.divsi(
                        cutlass.Int32(mw).ir_value(), cutlass.Int32(4).ir_value()
                    )
                )
                % 2
            )
        )
        rb = cutlass.Int32((rh * 2))
        tid_1 = cutlass.Int32(((mw * 32) + lane))
        row_col = cutlass.Uint32(cutlass.Uint32((rb * 32)))
        lens2 = cute.make_rmem_tensor((2,), cutlass.Int32)
        for r_1 in cutlass.range(cutlass.Int32(0), cutlass.Int32(2), cutlass.Int32(1)):
            lens2[r_1] = cutlass.Int32(0)
            if ((row0_1 + rb) + r_1) < rows:
                lens2[r_1] = cutlass.Int32(_lengths[((row0_1 + rb) + r_1)])
        lane_lt = cutlass.Uint32((2147483647 >> (31 - lane)))
        row_tmem = cutlass.Uint32(((wq * 32) << 16))
        row_tok = cutlass.Int32(((wq * 32) + lane))
        arow = cutlass.Int32(((wg * 2) + rh))
        lens4 = cute.make_rmem_tensor((4,), cutlass.Int32)
        max_pages_1[0] = cutlass.Int32(0)
        for r_2 in cutlass.range(cutlass.Int32(0), cutlass.Int32(4), cutlass.Int32(1)):
            lens4[r_2] = cutlass.Int32(0)
            if (row0_1 + r_2) < rows:
                lens4[r_2] = cutlass.Int32(_lengths[(row0_1 + r_2)])
            _max_3 = cutlass.max(
                max_pages_1[0],
                cutlass.Int32(
                    cutlass_arith.divsi(
                        cutlass.Int32(((lens4[r_2] + 128) - 1)).ir_value(),
                        cutlass.Int32(128).ir_value(),
                    )
                ),
            )
            max_pages_1[0] = cutlass.Int32(_max_3)
        max_groups_1 = cutlass.Int32(
            cutlass.Int32(
                cutlass_arith.divsi(
                    cutlass.Int32(((max_pages_1[0] + 8) - 1)).ir_value(),
                    cutlass.Int32(8).ir_value(),
                )
            )
        )
        end_g = cutlass.Int32(
            cutlass.Int32(
                cutlass_arith.divsi(
                    cutlass.Int32((max_groups_1 * (split_1 + 1))).ir_value(),
                    cutlass.Int32(splits).ir_value(),
                )
            )
        )
        next_g[0] = cutlass.Int32(
            cutlass.Int32(
                cutlass_arith.divsi(
                    cutlass.Int32((max_groups_1 * split_1)).ir_value(),
                    cutlass.Int32(splits).ir_value(),
                )
            )
        )
        seen[0] = cutlass.Int32(0)
        wf = cute.make_rmem_tensor((64,), cutlass.Float32)
        for r_3 in cutlass.range(cutlass.Int32(0), cutlass.Int32(2), cutlass.Int32(1)):
            for i_2 in cutlass.range(
                cutlass.Int32(0), cutlass.Int32(32), cutlass.Int32(1)
            ):
                w_i[0] = cutlass.Float32(0.0)
                if ((row0_1 + rb) + r_3) < rows:
                    w_i[0] = cutlass.Float32(
                        (_weights[((((row0_1 + rb) + r_3) * 32) + i_2)] * 0.5)
                    )
                wf[((r_3 * 32) + i_2)] = cutlass.Float32(w_i[0])
        if tid_1 < 8:
            prims.store_ext(cutlass.Uint64(0).ir_value(), (smem_thr + (tid_1)))
        if tid_1 < 24:
            prims.store_ext(cutlass.Int32(0).ir_value(), (smem_ctl + (276 + tid_1)))
        if tid_1 < 256:
            prims.store_ext(cutlass.Int32(0).ir_value(), (smem_ctl + (tid_1)))
        if CANDIDATES:
            for s in cutlass.range(
                cutlass.Int32(0), cutlass.Int32(1), cutlass.Int32(1)
            ):
                prims.store_ext(
                    cutlass.Int32(-1).ir_value(), (smem_bmax + ((s * 512) + tid_1))
                )
        cute.arch.barrier(barrier_id=1, number_of_threads=512)
        thr = cute.make_rmem_tensor((2,), cutlass.Uint64)
        thr_f = cute.make_rmem_tensor((2,), cutlass.Float32)
        for r_4 in cutlass.range(cutlass.Int32(0), cutlass.Int32(2), cutlass.Int32(1)):
            thr[r_4] = cutlass.Uint64(0)
            thr_f[r_4] = cutlass.Float32(float("-inf"))
        acc = cute.make_rmem_tensor((16,), cutlass.Float32)
        acc_stage[0] = cutlass.Uint32(0)
        skip[0] = cutlass.Int32(wg)
        flush[0] = cutlass.Int32(0)
        _phase_acc_full[0] = cutlass.Uint32(0)
        _loop_active_35 = cute.make_rmem_tensor((1,), cutlass.Boolean)
        _break_loop_counter_36 = cute.make_rmem_tensor((1,), cutlass.Int32)
        _break_loop_counter_36[0] = cutlass.Int32(cutlass.Int32(0))
        _loop_active_35[0] = cutlass.Boolean(
            _break_loop_counter_36[0] < cutlass.Int32(1073741824)
        )
        while _loop_active_35[0]:
            item = cutlass.Int32(_break_loop_counter_36[0])
            desc[0] = cutlass.Int32(-2)
            if skip[0] == 0:
                cute.arch.mbarrier_wait(
                    acc_full_addr + acc_stage[0], _phase_acc_full[0]
                )
                prims.tcgen05_fence("after_thread_sync")
                desc[0] = cutlass.Int32(_smem_ctl[(304 + cutlass.Int32(acc_stage[0]))])
            enter[0] = cutlass.Int32(0)
            _if_condition_37 = cutlass.Boolean((desc[0] == -1))
            flush[0] = cutlass.Int32(
                cutlass.select_(_if_condition_37, cutlass.Int32(1), flush[0])
            )
            enter[0] = cutlass.Int32(
                cutlass.select_(_if_condition_37, cutlass.Int32(1), enter[0])
            )
            _if_condition_38 = cutlass.Boolean((desc[0] >= 1073741824))
            enter[0] = cutlass.Int32(
                cutlass.select_(_if_condition_38, cutlass.Int32(1), enter[0])
            )
            desc[0] = cutlass.Int32(
                cutlass.select_(
                    _if_condition_38, cutlass.Int32((desc[0] - 1073741824)), desc[0]
                )
            )
            grp[0] = cutlass.Int32(-1)
            jt[0] = cutlass.Int32(-1)
            _if_condition_39 = cutlass.Boolean((desc[0] >= 0))
            grp[0] = cutlass.Int32(
                cutlass.select_(_if_condition_39, cutlass.Int32((desc[0] >> 7)), grp[0])
            )
            jt[0] = cutlass.Int32(
                cutlass.select_(
                    _if_condition_39, cutlass.Int32(((desc[0] >> 4) & 7)), jt[0]
                )
            )
            napp[0] = cutlass.Int32(0)
            if CANDIDATES:
                _if_condition_40 = cutlass.Boolean((desc[0] == -1))
                napp[0] = cutlass.Int32(
                    cutlass.select_(
                        _if_condition_40, cutlass.Int32((end_g - next_g[0])), napp[0]
                    )
                )
                if (desc[0] >= 0) & (grp[0] > next_g[0]):
                    seen[0] = cutlass.Int32((seen[0] + 1))
                    _if_condition_41 = cutlass.Boolean((seen[0] == 2))
                    napp[0] = cutlass.Int32(
                        cutlass.select_(_if_condition_41, cutlass.Int32(1), napp[0])
                    )
                for ai in cutlass.range(
                    cutlass.Int32(0), cutlass.Int32(napp[0]), cutlass.Int32(1)
                ):
                    ag = cutlass.Int32((next_g[0] + ai))
                    bthr_a = cutlass.Uint64(_smem_thr[(4 + arow)])
                    aslot = cutlass.Int32(
                        ((((ag & 1) * 256) + ((arow >> 1) * 128)) + row_tok)
                    )
                    ashift = cutlass.Int32(((arow & 1) * 16))
                    abits[0] = cutlass.Uint32(
                        (
                            (
                                cutlass.Uint32(_smem_bmax[aslot])
                                >> cutlass.Uint32(ashift)
                            )
                            & 65535
                        )
                    )
                    _atomic_old_0 = cute.arch.atomic_add(
                        smem_bmax + aslot,
                        cutlass.Int32(
                            cutlass.Int32(
                                ((65535 - abits[0]) << cutlass.Uint32(ashift))
                            )
                        ),
                        sem="relaxed",
                        scope="cta",
                    )
                    bkey[0] = cutlass.Uint64(0)
                    if abits[0] < 65408:
                        _if_condition_42 = cutlass.Boolean((abits[0] == 32768))
                        abits[0] = cutlass.Uint32(
                            cutlass.select_(
                                _if_condition_42, cutlass.Uint32(0), abits[0]
                            )
                        )
                        aflip[0] = cutlass.Uint32(32768)
                        _if_condition_43 = cutlass.Boolean((abits[0] >= 32768))
                        aflip[0] = cutlass.Uint32(
                            cutlass.select_(
                                _if_condition_43, cutlass.Uint32(65535), aflip[0]
                            )
                        )
                        ablk = cutlass.Int32(((ag * 128) + row_tok))
                        bkey[0] = cutlass.Uint64(
                            (
                                (cutlass.Uint64((abits[0] ^ aflip[0])) << 32)
                                | cutlass.Uint64(cutlass.Uint32((4294967295 - ablk)))
                            )
                        )
                    bkeep[0] = cutlass.Int32(0)
                    _if_condition_44 = cutlass.Boolean((bkey[0] > bthr_a))
                    bkeep[0] = cutlass.Int32(
                        cutlass.select_(_if_condition_44, cutlass.Int32(1), bkeep[0])
                    )
                    _vote_0 = cute.arch.vote_ballot_sync(
                        cutlass.Boolean((bkeep[0] == 1)), cutlass.Uint32(0xFFFFFFFF)
                    ).bitcast(cutlass.Uint32)
                    bballot = cutlass.Uint32(_vote_0)
                    _popc_1 = cutlass.Int32(cute.arch.popc(cutlass.Uint32(bballot)))
                    bnkeep = cutlass.Uint32(cutlass.Uint32(_popc_1))
                    if bnkeep != 0:
                        _popc_2 = cutlass.Int32(
                            cute.arch.popc(cutlass.Uint32((bballot & lane_lt)))
                        )
                        brank = cutlass.Uint32(cutlass.Uint32(_popc_2))
                        bb[0] = cutlass.Int32(0)
                        if lane == 0:
                            _atomic_old_1 = cute.arch.atomic_add(
                                smem_ctl + (280 + arow),
                                cutlass.Int32(cutlass.Int32(bnkeep)),
                                sem="relaxed",
                                scope="cta",
                            )
                            bb[0] = cutlass.Int32(_atomic_old_1)
                        _shfl_0 = cute.arch.shuffle_sync(
                            bb[0], 0, mask=0xFFFFFFFF, mask_and_clamp=31
                        )
                        bb[0] = cutlass.Int32(_shfl_0)
                        if bkeep[0] == 1:
                            prims.store_ext(
                                cutlass.Int64(cutlass.Int64(bkey[0])).ir_value(),
                                (
                                    block_partial
                                    + (
                                        ((bbase + (arow * bstride)) + bb[0])
                                        + cutlass.Int32(brank)
                                    )
                                ),
                            )
                        if (bb[0] + cutlass.Int32(bnkeep)) > 3968:
                            if lane == 0:
                                prims.store_ext(
                                    cutlass.Int32(1).ir_value(), (smem_ctl + (300))
                                )
                _if_condition_45 = cutlass.Boolean((napp[0] != 0))
                next_g[0] = cutlass.Int32(
                    cutlass.select_(
                        _if_condition_45,
                        cutlass.Int32((next_g[0] + napp[0])),
                        next_g[0],
                    )
                )
                seen[0] = cutlass.Int32(
                    cutlass.select_(_if_condition_45, cutlass.Int32(0), seen[0])
                )
            if desc[0] >= 0:
                rmask_1 = cutlass.Int32((desc[0] & 15))
                tok = cutlass.Int32((((desc[0] >> 4) * 128) + row_tok))
                tok_key = cutlass.Uint64(
                    cutlass.Uint64(cutlass.Uint32((4294967295 - tok)))
                )
                scores = cute.make_rmem_tensor((2,), cutlass.Float32)
                for r2 in cutlass.range(
                    cutlass.Int32(0), cutlass.Int32(2), cutlass.Int32(1)
                ):
                    _f2_0 = (cutlass.Float32(0.0), cutlass.Float32(0.0))
                    _f32x2_init_46 = (
                        cutlass.Float32(_f2_0[0]),
                        cutlass.Float32(_f2_0[1]),
                    )
                    s0[0] = cutlass.Float32(_f32x2_init_46[0])
                    s0[1] = cutlass.Float32(_f32x2_init_46[1])
                    _f2_1 = (cutlass.Float32(0.0), cutlass.Float32(0.0))
                    _f32x2_init_47 = (
                        cutlass.Float32(_f2_1[0]),
                        cutlass.Float32(_f2_1[1]),
                    )
                    s1[0] = cutlass.Float32(_f32x2_init_47[0])
                    s1[1] = cutlass.Float32(_f32x2_init_47[1])
                    for hh in cutlass.range(
                        cutlass.Int32(0), cutlass.Int32(2), cutlass.Int32(1)
                    ):
                        _tmem_load_48_atom = cute.make_copy_atom(
                            cute.nvgpu.tcgen05.Ld32x32bOp(
                                cute.nvgpu.tcgen05.Repetition.x16
                            ),
                            cutlass.Float32,
                        )
                        _tmem_load_48_src = cute.make_tensor(
                            cute.make_ptr(
                                cutlass.Float32,
                                cutlass.Uint32(
                                    (
                                        (
                                            (
                                                (
                                                    (taddr + row_tmem)
                                                    + (acc_stage[0] * 128)
                                                )
                                                + row_col
                                            )
                                            + cutlass.Uint32((r2 * 32))
                                        )
                                        + cutlass.Uint32((hh * 16))
                                    )
                                ),
                                mem_space=cutlass.AddressSpace.tmem,
                                assumed_align=1,
                            ),
                            cute.make_layout((16,), stride=(1,)),
                        )
                        _tmem_load_48_dst = cute.make_tensor(
                            acc.iterator + 0, cute.make_layout((16,), stride=(1,))
                        )
                        cute.copy(
                            _tmem_load_48_atom, _tmem_load_48_src, _tmem_load_48_dst
                        )
                        cute.arch.fence_view_async_tmem_load()
                        for h4 in cutlass.range(
                            cutlass.Int32(0), cutlass.Int32(4), cutlass.Int32(1)
                        ):
                            _f2_2 = (
                                cutlass.Float32(acc[(4 * h4)]),
                                cutlass.Float32(acc[((4 * h4) + 1)]),
                            )
                            _fabs_0 = cute.math.abs(acc[(4 * h4)])
                            _fabs_1 = cute.math.abs(acc[((4 * h4) + 1)])
                            _f2_3 = (cutlass.Float32(_fabs_0), cutlass.Float32(_fabs_1))
                            a0 = cute.arch.add_packed_f32x2(
                                (cutlass.Float32(_f2_2[0]), cutlass.Float32(_f2_2[1])),
                                (cutlass.Float32(_f2_3[0]), cutlass.Float32(_f2_3[1])),
                                rnd="rn",
                                ftz=True,
                            )
                            _f2_4 = (
                                cutlass.Float32(
                                    wf[((r2 * 32) + ((hh * 16) + (4 * h4)))]
                                ),
                                cutlass.Float32(
                                    wf[(((r2 * 32) + ((hh * 16) + (4 * h4))) + 1)]
                                ),
                            )
                            _f32x2_assign_49 = cute.arch.fma_packed_f32x2(
                                (cutlass.Float32(a0[0]), cutlass.Float32(a0[1])),
                                (cutlass.Float32(_f2_4[0]), cutlass.Float32(_f2_4[1])),
                                (cutlass.Float32(s0[0]), cutlass.Float32(s0[1])),
                                rnd="rn",
                                ftz=True,
                            )
                            s0[0] = cutlass.Float32(_f32x2_assign_49[0])
                            s0[1] = cutlass.Float32(_f32x2_assign_49[1])
                            _f2_5 = (
                                cutlass.Float32(acc[((4 * h4) + 2)]),
                                cutlass.Float32(acc[((4 * h4) + 3)]),
                            )
                            _fabs_2 = cute.math.abs(acc[((4 * h4) + 2)])
                            _fabs_3 = cute.math.abs(acc[((4 * h4) + 3)])
                            _f2_6 = (cutlass.Float32(_fabs_2), cutlass.Float32(_fabs_3))
                            a1 = cute.arch.add_packed_f32x2(
                                (cutlass.Float32(_f2_5[0]), cutlass.Float32(_f2_5[1])),
                                (cutlass.Float32(_f2_6[0]), cutlass.Float32(_f2_6[1])),
                                rnd="rn",
                                ftz=True,
                            )
                            _f2_7 = (
                                cutlass.Float32(
                                    wf[(((r2 * 32) + ((hh * 16) + (4 * h4))) + 2)]
                                ),
                                cutlass.Float32(
                                    wf[(((r2 * 32) + ((hh * 16) + (4 * h4))) + 3)]
                                ),
                            )
                            _f32x2_assign_50 = cute.arch.fma_packed_f32x2(
                                (cutlass.Float32(a1[0]), cutlass.Float32(a1[1])),
                                (cutlass.Float32(_f2_7[0]), cutlass.Float32(_f2_7[1])),
                                (cutlass.Float32(s1[0]), cutlass.Float32(s1[1])),
                                rnd="rn",
                                ftz=True,
                            )
                            s1[0] = cutlass.Float32(_f32x2_assign_50[0])
                            s1[1] = cutlass.Float32(_f32x2_assign_50[1])
                    ssum = cute.arch.add_packed_f32x2(
                        (cutlass.Float32(s0[0]), cutlass.Float32(s0[1])),
                        (cutlass.Float32(s1[0]), cutlass.Float32(s1[1])),
                        rnd="rn",
                        ftz=True,
                    )
                    total = cutlass.Float32((ssum[0] + ssum[1]))
                    _cvt_bf16_0 = cutlass.BFloat16(total)
                    _cvt_f32_0 = cutlass.Float32(_cvt_bf16_0)
                    scores[r2] = cutlass.Float32(_cvt_f32_0)
                hot[0] = cutlass.Int32(0)
                for r2_1 in cutlass.range(
                    cutlass.Int32(0), cutlass.Int32(2), cutlass.Int32(1)
                ):
                    _if_condition_51 = cutlass.Boolean((scores[r2_1] >= thr_f[r2_1]))
                    hot[0] = cutlass.Int32(
                        cutlass.select_(_if_condition_51, cutlass.Int32(1), hot[0])
                    )
                _vote_1 = cutlass.Int32(
                    cute.arch.vote_any_sync(
                        cutlass.Boolean((hot[0] == 1)), cutlass.Uint32(0xFFFFFFFF)
                    )
                )
                anyhot = cutlass.Int32(_vote_1)
                if anyhot != 0:
                    for r2_2 in cutlass.range(
                        cutlass.Int32(0), cutlass.Int32(2), cutlass.Int32(1)
                    ):
                        rr = cutlass.Int32((rb + r2_2))
                        key[0] = cutlass.Uint64(0)
                        if (((rmask_1 >> rr) & 1) != 0) & (tok < lens2[r2_2]):
                            _atomic_max_float_encode_52_bits = cutlass.Float32(
                                scores[r2_2]
                            ).bitcast(cutlass.Uint32)
                            _atomic_max_float_encode_52_mask = (
                                cutlass.Uint32(0)
                                - cutlass.Uint32(_atomic_max_float_encode_52_bits >> 31)
                            ) | cutlass.Uint32(0x80000000)
                            _amf_enc_0 = cutlass.Uint32(
                                _atomic_max_float_encode_52_bits
                                ^ _atomic_max_float_encode_52_mask
                            )
                            kbits[0] = cutlass.Uint32(_amf_enc_0)
                            _if_condition_53 = cutlass.Boolean((scores[r2_2] == 0.0))
                            kbits[0] = cutlass.Uint32(
                                cutlass.select_(
                                    _if_condition_53,
                                    cutlass.Uint32(2147483648),
                                    kbits[0],
                                )
                            )
                            key[0] = cutlass.Uint64(
                                ((cutlass.Uint64((kbits[0] >> 16)) << 32) | tok_key)
                            )
                        keep[0] = cutlass.Int32(0)
                        _if_condition_54 = cutlass.Boolean((key[0] > thr[r2_2]))
                        keep[0] = cutlass.Int32(
                            cutlass.select_(_if_condition_54, cutlass.Int32(1), keep[0])
                        )
                        _vote_2 = cute.arch.vote_ballot_sync(
                            cutlass.Boolean((keep[0] == 1)), cutlass.Uint32(0xFFFFFFFF)
                        ).bitcast(cutlass.Uint32)
                        ballot = cutlass.Uint32(_vote_2)
                        _popc_3 = cutlass.Int32(cute.arch.popc(cutlass.Uint32(ballot)))
                        nkeep = cutlass.Uint32(cutlass.Uint32(_popc_3))
                        if nkeep != 0:
                            _popc_4 = cutlass.Int32(
                                cute.arch.popc(cutlass.Uint32((ballot & lane_lt)))
                            )
                            rank = cutlass.Uint32(cutlass.Uint32(_popc_4))
                            base[0] = cutlass.Int32(0)
                            if lane == 0:
                                _atomic_old_2 = cute.arch.atomic_add(
                                    smem_ctl + (276 + rr),
                                    cutlass.Int32(cutlass.Int32(nkeep)),
                                    sem="relaxed",
                                    scope="cta",
                                )
                                base[0] = cutlass.Int32(_atomic_old_2)
                            _shfl_1 = cute.arch.shuffle_sync(
                                base[0], 0, mask=0xFFFFFFFF, mask_and_clamp=31
                            )
                            base[0] = cutlass.Int32(_shfl_1)
                            if keep[0] == 1:
                                prims.store_ext(
                                    cutlass.Uint64(key[0]).ir_value(),
                                    (
                                        smem_keys
                                        + (
                                            ((rr * 2048) + base[0])
                                            + cutlass.Int32(rank)
                                        )
                                    ),
                                )
                            if (base[0] + cutlass.Int32(nkeep)) > 1280:
                                if lane == 0:
                                    prims.store_ext(
                                        cutlass.Int32(1).ir_value(), (smem_ctl + (300))
                                    )
                if CANDIDATES:
                    bval = cute.make_rmem_tensor((2,), cutlass.Float32)
                    for r2_3 in cutlass.range(
                        cutlass.Int32(0), cutlass.Int32(2), cutlass.Int32(1)
                    ):
                        bval[r2_3] = cutlass.Float32(float("-inf"))
                        if (((rmask_1 >> (rb + r2_3)) & 1) != 0) & (tok < lens2[r2_3]):
                            bval[r2_3] = cutlass.Float32(scores[r2_3])
                            if tok == (lens2[r2_3] - 1):
                                bval[r2_3] = cutlass.Float32(float("inf"))
                    _bf16x2_0 = prims.cvt_packfloat_f32(
                        cutlass.Float32(
                            ((cutlass.Float32(bval[0]), cutlass.Float32(bval[1])))[1]
                        ),
                        cutlass.Float32(
                            ((cutlass.Float32(bval[0]), cutlass.Float32(bval[1])))[0]
                        ),
                        0,
                        prims.CVTPackFloat.BF16X2,
                        rnd=prims.FPRoundingMode.RN,
                    )
                    _shfl_xor_0 = cute.arch.shuffle_sync_bfly(
                        cutlass.Uint32(_bf16x2_0), 1, mask=0xFFFFFFFF, mask_and_clamp=31
                    )
                    _bf16x2_max_0 = prims.inline_ptx(
                        "max.bf16x2 {$w0}, {$r0}, {$r1};",
                        write_only_types=[cutlass.Uint32],
                        read_only_args=[
                            cutlass.Uint32(_bf16x2_0),
                            cutlass.Uint32(_shfl_xor_0),
                        ],
                    )
                    _shfl_xor_1 = cute.arch.shuffle_sync_bfly(
                        cutlass.Uint32(_bf16x2_max_0),
                        2,
                        mask=0xFFFFFFFF,
                        mask_and_clamp=31,
                    )
                    _bf16x2_max_1 = prims.inline_ptx(
                        "max.bf16x2 {$w0}, {$r0}, {$r1};",
                        write_only_types=[cutlass.Uint32],
                        read_only_args=[
                            cutlass.Uint32(_bf16x2_max_0),
                            cutlass.Uint32(_shfl_xor_1),
                        ],
                    )
                    _shfl_xor_2 = cute.arch.shuffle_sync_bfly(
                        cutlass.Uint32(_bf16x2_max_1),
                        4,
                        mask=0xFFFFFFFF,
                        mask_and_clamp=31,
                    )
                    _bf16x2_max_2 = prims.inline_ptx(
                        "max.bf16x2 {$w0}, {$r0}, {$r1};",
                        write_only_types=[cutlass.Uint32],
                        read_only_args=[
                            cutlass.Uint32(_bf16x2_max_1),
                            cutlass.Uint32(_shfl_xor_2),
                        ],
                    )
                    pm = cutlass.Int32(((rmask_1 >> rb) & 3))
                    if (pm != 0) & ((row_tok & 7) == 0):
                        dslot = cutlass.Int32(
                            (
                                ((((grp[0] & 1) * 256) + (rh * 128)) + (jt[0] * 16))
                                + (row_tok >> 3)
                            )
                        )
                        dword = cutlass.Uint32(cutlass.Uint32(_bf16x2_max_2))
                        if pm == 3:
                            prims.store_ext(
                                cutlass.Int32(cutlass.Int32(dword)).ir_value(),
                                (smem_bmax + (dslot)),
                            )
                        if pm == 1:
                            _atomic_old_3 = cute.arch.atomic_add(
                                smem_bmax + dslot,
                                cutlass.Int32(
                                    cutlass.Int32(((dword | 4294901760) + 1))
                                ),
                                sem="relaxed",
                                scope="cta",
                            )
                        if pm == 2:
                            _atomic_old_4 = cute.arch.atomic_add(
                                smem_bmax + dslot,
                                cutlass.Int32(cutlass.Int32(((dword | 65535) + 1))),
                                sem="relaxed",
                                scope="cta",
                            )
                prims.tcgen05_fence("after_thread_sync")
                cute.arch.mbarrier_arrive(acc_empty_addr + acc_stage[0])
            if skip[0] == 0:
                _advanced_stage_55 = cutlass.Uint32(acc_stage[0] + 1)
                _stage_wrapped_56 = cutlass.Boolean(_advanced_stage_55 == 3)
                acc_stage[0] = cutlass.Uint32(
                    cutlass.select_(_stage_wrapped_56, 0, _advanced_stage_55)
                )
                _phase_acc_full[0] = cutlass.Uint32(
                    cutlass.select_(
                        _stage_wrapped_56,
                        _phase_acc_full[0] ^ cutlass.Uint32(1),
                        _phase_acc_full[0],
                    )
                )
                _advanced_stage_57 = cutlass.Uint32(acc_stage[0] + 1)
                _stage_wrapped_58 = cutlass.Boolean(_advanced_stage_57 == 3)
                acc_stage[0] = cutlass.Uint32(
                    cutlass.select_(_stage_wrapped_58, 0, _advanced_stage_57)
                )
                _phase_acc_full[0] = cutlass.Uint32(
                    cutlass.select_(
                        _stage_wrapped_58,
                        _phase_acc_full[0] ^ cutlass.Uint32(1),
                        _phase_acc_full[0],
                    )
                )
            if skip[0] == 1:
                _advanced_stage_59 = cutlass.Uint32(acc_stage[0] + 1)
                _stage_wrapped_60 = cutlass.Boolean(_advanced_stage_59 == 3)
                acc_stage[0] = cutlass.Uint32(
                    cutlass.select_(_stage_wrapped_60, 0, _advanced_stage_59)
                )
                _phase_acc_full[0] = cutlass.Uint32(
                    cutlass.select_(
                        _stage_wrapped_60,
                        _phase_acc_full[0] ^ cutlass.Uint32(1),
                        _phase_acc_full[0],
                    )
                )
                skip[0] = cutlass.Int32(0)
            if enter[0] == 1:
                _loop_active_61 = cute.make_rmem_tensor((1,), cutlass.Boolean)
                _break_loop_counter_62 = cute.make_rmem_tensor((1,), cutlass.Int32)
                _break_loop_counter_62[0] = cutlass.Int32(cutlass.Int32(0))
                _loop_active_61[0] = cutlass.Boolean(
                    _break_loop_counter_62[0] < cutlass.Int32(1048576)
                )
                while _loop_active_61[0]:
                    rnd_i = cutlass.Int32(_break_loop_counter_62[0])
                    if lane == 0:
                        prims.store_ext(
                            cutlass.Int32(flush[0]).ir_value(), (smem_ctl + (284 + mw))
                        )
                    cute.arch.barrier(barrier_id=1, number_of_threads=512)
                    if tid_1 == 0:
                        prims.store_ext(cutlass.Int32(0).ir_value(), (smem_ctl + (300)))
                        prims.tcgen05_fence("after_thread_sync")
                        prims.store_ext(
                            cutlass.Int32((_smem_ctl[301] + 1)).ir_value(),
                            (smem_ctl + (301)),
                        )
                    fin[0] = cutlass.Int32(0)
                    for w in cutlass.range(
                        cutlass.Int32(0), cutlass.Int32(16), cutlass.Int32(1)
                    ):
                        fin[0] = cutlass.Int32((fin[0] + _smem_ctl[(284 + w)]))
                    for r2_4 in cutlass.range(
                        cutlass.Int32(0), cutlass.Int32(4), cutlass.Int32(1)
                    ):
                        seg = cutlass.Int32((r2_4 * 2048))
                        n = cutlass.Int32(_smem_ctl[(276 + r2_4)])
                        if n > 512:
                            prefix[0] = cutlass.Uint64(0)
                            need[0] = cutlass.Int32(512)
                            done[0] = cutlass.Int32(0)
                            for rnd in cutlass.range(
                                cutlass.Int32(2), cutlass.Int32(8), cutlass.Int32(1)
                            ):
                                if done[0] == 0:
                                    shift = cutlass.Int32((56 - (rnd * 8)))
                                    hi_mask[0] = cutlass.Uint64(0)
                                    _if_condition_63 = cutlass.Boolean((rnd > 0))
                                    hi_mask[0] = cutlass.Uint64(
                                        cutlass.select_(
                                            _if_condition_63,
                                            cutlass.Uint64(
                                                (
                                                    ~(
                                                        (
                                                            cutlass.Uint64(1)
                                                            << cutlass.Uint64(
                                                                (shift + 8)
                                                            )
                                                        )
                                                        - 1
                                                    )
                                                )
                                            ),
                                            hi_mask[0],
                                        )
                                    )
                                    for s_1 in cutlass.range(
                                        cutlass.Int32(0),
                                        cutlass.Int32(4),
                                        cutlass.Int32(1),
                                    ):
                                        idx = cutlass.Int32(((s_1 * 512) + tid_1))
                                        if idx < n:
                                            k = cutlass.Uint64(_smem_keys[(seg + idx)])
                                            if (k & hi_mask[0]) == prefix[0]:
                                                digit = cutlass.Int32(
                                                    cutlass.Int32(
                                                        (
                                                            (k >> cutlass.Uint64(shift))
                                                            & 255
                                                        )
                                                    )
                                                )
                                                cute.arch.red(
                                                    smem_ctl + digit,
                                                    cutlass.Int32(1),
                                                    op="add",
                                                    dtype=cutlass.Int32,
                                                    sem="relaxed",
                                                    scope="cta",
                                                )
                                    cute.arch.barrier(
                                        barrier_id=2, number_of_threads=512
                                    )
                                    bin = cutlass.Int32((255 - tid_1))
                                    c[0] = cutlass.Int32(0)
                                    if tid_1 < 256:
                                        c[0] = cutlass.Int32(_smem_ctl[bin])
                                        prims.store_ext(
                                            cutlass.Int32(0).ir_value(),
                                            (smem_ctl + (bin)),
                                        )
                                    incl[0] = cutlass.Int32(c[0])
                                    _shfl_2 = cute.arch.shuffle_sync(
                                        incl[0],
                                        (lane - 1),
                                        mask=0xFFFFFFFF,
                                        mask_and_clamp=31,
                                    )
                                    up = cutlass.Int32(_shfl_2)
                                    _if_condition_64 = cutlass.Boolean((lane >= 1))
                                    incl[0] = cutlass.Int32(
                                        cutlass.select_(
                                            _if_condition_64,
                                            cutlass.Int32((incl[0] + up)),
                                            incl[0],
                                        )
                                    )
                                    _shfl_3 = cute.arch.shuffle_sync(
                                        incl[0],
                                        (lane - 2),
                                        mask=0xFFFFFFFF,
                                        mask_and_clamp=31,
                                    )
                                    up_0 = cutlass.Int32(_shfl_3)
                                    _if_condition_65 = cutlass.Boolean((lane >= 2))
                                    incl[0] = cutlass.Int32(
                                        cutlass.select_(
                                            _if_condition_65,
                                            cutlass.Int32((incl[0] + up_0)),
                                            incl[0],
                                        )
                                    )
                                    _shfl_4 = cute.arch.shuffle_sync(
                                        incl[0],
                                        (lane - 4),
                                        mask=0xFFFFFFFF,
                                        mask_and_clamp=31,
                                    )
                                    up_1 = cutlass.Int32(_shfl_4)
                                    _if_condition_66 = cutlass.Boolean((lane >= 4))
                                    incl[0] = cutlass.Int32(
                                        cutlass.select_(
                                            _if_condition_66,
                                            cutlass.Int32((incl[0] + up_1)),
                                            incl[0],
                                        )
                                    )
                                    _shfl_5 = cute.arch.shuffle_sync(
                                        incl[0],
                                        (lane - 8),
                                        mask=0xFFFFFFFF,
                                        mask_and_clamp=31,
                                    )
                                    up_2 = cutlass.Int32(_shfl_5)
                                    _if_condition_67 = cutlass.Boolean((lane >= 8))
                                    incl[0] = cutlass.Int32(
                                        cutlass.select_(
                                            _if_condition_67,
                                            cutlass.Int32((incl[0] + up_2)),
                                            incl[0],
                                        )
                                    )
                                    _shfl_6 = cute.arch.shuffle_sync(
                                        incl[0],
                                        (lane - 16),
                                        mask=0xFFFFFFFF,
                                        mask_and_clamp=31,
                                    )
                                    up_3 = cutlass.Int32(_shfl_6)
                                    _if_condition_68 = cutlass.Boolean((lane >= 16))
                                    incl[0] = cutlass.Int32(
                                        cutlass.select_(
                                            _if_condition_68,
                                            cutlass.Int32((incl[0] + up_3)),
                                            incl[0],
                                        )
                                    )
                                    if (lane == 31) & (mw < 8):
                                        prims.store_ext(
                                            cutlass.Int32(incl[0]).ir_value(),
                                            (smem_ctl + (256 + mw)),
                                        )
                                    cute.arch.barrier(
                                        barrier_id=1, number_of_threads=512
                                    )
                                    for w_1 in cutlass.range(
                                        cutlass.Int32(0),
                                        cutlass.Int32(7),
                                        cutlass.Int32(1),
                                    ):
                                        if mw > w_1:
                                            incl[0] = cutlass.Int32(
                                                (incl[0] + _smem_ctl[(256 + w_1)])
                                            )
                                    above = cutlass.Int32((incl[0] - c[0]))
                                    if ((tid_1 < 256) & (above < need[0])) & (
                                        need[0] <= (above + c[0])
                                    ):
                                        prims.store_ext(
                                            cutlass.Int32(bin).ir_value(),
                                            (smem_ctl + (272)),
                                        )
                                        prims.store_ext(
                                            cutlass.Int32((need[0] - above)).ir_value(),
                                            (smem_ctl + (273)),
                                        )
                                        prims.store_ext(
                                            cutlass.Int32(c[0]).ir_value(),
                                            (smem_ctl + (274)),
                                        )
                                    cute.arch.barrier(
                                        barrier_id=2, number_of_threads=512
                                    )
                                    sel_digit = cutlass.Int32(_smem_ctl[272])
                                    sel_need = cutlass.Int32(_smem_ctl[273])
                                    sel_count = cutlass.Int32(_smem_ctl[274])
                                    prefix[0] = cutlass.Uint64(
                                        (
                                            prefix[0]
                                            | (
                                                cutlass.Uint64(sel_digit)
                                                << cutlass.Uint64(shift)
                                            )
                                        )
                                    )
                                    need[0] = cutlass.Int32(sel_need)
                                    _if_condition_69 = cutlass.Boolean(
                                        (sel_count == sel_need)
                                    )
                                    done[0] = cutlass.Int32(
                                        cutlass.select_(
                                            _if_condition_69, cutlass.Int32(1), done[0]
                                        )
                                    )
                            if tid_1 == 0:
                                prims.store_ext(
                                    cutlass.Int32(0).ir_value(), (smem_ctl + (275))
                                )
                            ck = cute.make_rmem_tensor((4,), cutlass.Uint64)
                            for s_2 in cutlass.range(
                                cutlass.Int32(0), cutlass.Int32(4), cutlass.Int32(1)
                            ):
                                idx_1 = cutlass.Int32(((s_2 * 512) + tid_1))
                                ck[s_2] = cutlass.Uint64(0)
                                if idx_1 < n:
                                    ck[s_2] = cutlass.Uint64(_smem_keys[(seg + idx_1)])
                            cute.arch.barrier(barrier_id=2, number_of_threads=512)
                            for s_3 in cutlass.range(
                                cutlass.Int32(0), cutlass.Int32(4), cutlass.Int32(1)
                            ):
                                k_1 = cutlass.Uint64(ck[s_3])
                                ckeep[0] = cutlass.Int32(0)
                                _if_condition_70 = cutlass.Boolean(
                                    ((n > ((s_3 * 512) + tid_1)) & (k_1 >= prefix[0]))
                                )
                                ckeep[0] = cutlass.Int32(
                                    cutlass.select_(
                                        _if_condition_70, cutlass.Int32(1), ckeep[0]
                                    )
                                )
                                _vote_3 = cute.arch.vote_ballot_sync(
                                    cutlass.Boolean((ckeep[0] == 1)),
                                    cutlass.Uint32(0xFFFFFFFF),
                                ).bitcast(cutlass.Uint32)
                                cballot = cutlass.Uint32(_vote_3)
                                _popc_5 = cutlass.Int32(
                                    cute.arch.popc(cutlass.Uint32(cballot))
                                )
                                cn = cutlass.Uint32(cutlass.Uint32(_popc_5))
                                if cn != 0:
                                    _popc_6 = cutlass.Int32(
                                        cute.arch.popc(
                                            cutlass.Uint32((cballot & lane_lt))
                                        )
                                    )
                                    crank = cutlass.Uint32(cutlass.Uint32(_popc_6))
                                    cbase[0] = cutlass.Int32(0)
                                    if lane == 0:
                                        _atomic_old_5 = cute.arch.atomic_add(
                                            smem_ctl + 275,
                                            cutlass.Int32(cutlass.Int32(cn)),
                                            sem="relaxed",
                                            scope="cta",
                                        )
                                        cbase[0] = cutlass.Int32(_atomic_old_5)
                                    _shfl_7 = cute.arch.shuffle_sync(
                                        cbase[0], 0, mask=0xFFFFFFFF, mask_and_clamp=31
                                    )
                                    cbase[0] = cutlass.Int32(_shfl_7)
                                    if ckeep[0] == 1:
                                        prims.store_ext(
                                            cutlass.Uint64(k_1).ir_value(),
                                            (
                                                smem_keys
                                                + (
                                                    (seg + cbase[0])
                                                    + cutlass.Int32(crank)
                                                )
                                            ),
                                        )
                            cute.arch.barrier(barrier_id=1, number_of_threads=512)
                            if tid_1 == 0:
                                prims.store_ext(
                                    cutlass.Int32(_smem_ctl[275]).ir_value(),
                                    (smem_ctl + (276 + r2_4)),
                                )
                                prims.store_ext(
                                    cutlass.Uint64(prefix[0]).ir_value(),
                                    (smem_thr + (r2_4)),
                                )
                        if CANDIDATES:
                            bseg = cutlass.Int32((bbase + (r2_4 * bstride)))
                            bn = cutlass.Int32(_smem_ctl[(280 + r2_4)])
                            if bn > 2048:
                                bprefix[0] = cutlass.Uint64(0)
                                bneed[0] = cutlass.Int32(2048)
                                bdone[0] = cutlass.Int32(0)
                                for rnd_1 in cutlass.range(
                                    cutlass.Int32(2), cutlass.Int32(8), cutlass.Int32(1)
                                ):
                                    if bdone[0] == 0:
                                        bshift = cutlass.Int32((56 - (rnd_1 * 8)))
                                        bhi_mask = cutlass.Uint64(
                                            (
                                                ~(
                                                    (
                                                        cutlass.Uint64(1)
                                                        << cutlass.Uint64((bshift + 8))
                                                    )
                                                    - 1
                                                )
                                            )
                                        )
                                        for s_4 in cutlass.range(
                                            cutlass.Int32(0),
                                            cutlass.Int32(8),
                                            cutlass.Int32(1),
                                        ):
                                            bidx = cutlass.Int32(((s_4 * 512) + tid_1))
                                            if bidx < bn:
                                                bk = cutlass.Uint64(
                                                    cutlass.Uint64(
                                                        _block_partial[(bseg + bidx)]
                                                    )
                                                )
                                                if (bk & bhi_mask) == bprefix[0]:
                                                    bdigit = cutlass.Int32(
                                                        cutlass.Int32(
                                                            (
                                                                (
                                                                    bk
                                                                    >> cutlass.Uint64(
                                                                        bshift
                                                                    )
                                                                )
                                                                & 255
                                                            )
                                                        )
                                                    )
                                                    cute.arch.red(
                                                        smem_ctl + bdigit,
                                                        cutlass.Int32(1),
                                                        op="add",
                                                        dtype=cutlass.Int32,
                                                        sem="relaxed",
                                                        scope="cta",
                                                    )
                                        cute.arch.barrier(
                                            barrier_id=2, number_of_threads=512
                                        )
                                        bbin = cutlass.Int32((255 - tid_1))
                                        bc[0] = cutlass.Int32(0)
                                        if tid_1 < 256:
                                            bc[0] = cutlass.Int32(_smem_ctl[bbin])
                                            prims.store_ext(
                                                cutlass.Int32(0).ir_value(),
                                                (smem_ctl + (bbin)),
                                            )
                                        bincl[0] = cutlass.Int32(bc[0])
                                        _shfl_8 = cute.arch.shuffle_sync(
                                            bincl[0],
                                            (lane - 1),
                                            mask=0xFFFFFFFF,
                                            mask_and_clamp=31,
                                        )
                                        bup = cutlass.Int32(_shfl_8)
                                        _if_condition_71 = cutlass.Boolean((lane >= 1))
                                        bincl[0] = cutlass.Int32(
                                            cutlass.select_(
                                                _if_condition_71,
                                                cutlass.Int32((bincl[0] + bup)),
                                                bincl[0],
                                            )
                                        )
                                        _shfl_9 = cute.arch.shuffle_sync(
                                            bincl[0],
                                            (lane - 2),
                                            mask=0xFFFFFFFF,
                                            mask_and_clamp=31,
                                        )
                                        bup_0 = cutlass.Int32(_shfl_9)
                                        _if_condition_72 = cutlass.Boolean((lane >= 2))
                                        bincl[0] = cutlass.Int32(
                                            cutlass.select_(
                                                _if_condition_72,
                                                cutlass.Int32((bincl[0] + bup_0)),
                                                bincl[0],
                                            )
                                        )
                                        _shfl_10 = cute.arch.shuffle_sync(
                                            bincl[0],
                                            (lane - 4),
                                            mask=0xFFFFFFFF,
                                            mask_and_clamp=31,
                                        )
                                        bup_1 = cutlass.Int32(_shfl_10)
                                        _if_condition_73 = cutlass.Boolean((lane >= 4))
                                        bincl[0] = cutlass.Int32(
                                            cutlass.select_(
                                                _if_condition_73,
                                                cutlass.Int32((bincl[0] + bup_1)),
                                                bincl[0],
                                            )
                                        )
                                        _shfl_11 = cute.arch.shuffle_sync(
                                            bincl[0],
                                            (lane - 8),
                                            mask=0xFFFFFFFF,
                                            mask_and_clamp=31,
                                        )
                                        bup_2 = cutlass.Int32(_shfl_11)
                                        _if_condition_74 = cutlass.Boolean((lane >= 8))
                                        bincl[0] = cutlass.Int32(
                                            cutlass.select_(
                                                _if_condition_74,
                                                cutlass.Int32((bincl[0] + bup_2)),
                                                bincl[0],
                                            )
                                        )
                                        _shfl_12 = cute.arch.shuffle_sync(
                                            bincl[0],
                                            (lane - 16),
                                            mask=0xFFFFFFFF,
                                            mask_and_clamp=31,
                                        )
                                        bup_3 = cutlass.Int32(_shfl_12)
                                        _if_condition_75 = cutlass.Boolean((lane >= 16))
                                        bincl[0] = cutlass.Int32(
                                            cutlass.select_(
                                                _if_condition_75,
                                                cutlass.Int32((bincl[0] + bup_3)),
                                                bincl[0],
                                            )
                                        )
                                        if (lane == 31) & (mw < 8):
                                            prims.store_ext(
                                                cutlass.Int32(bincl[0]).ir_value(),
                                                (smem_ctl + (256 + mw)),
                                            )
                                        cute.arch.barrier(
                                            barrier_id=1, number_of_threads=512
                                        )
                                        for w_2 in cutlass.range(
                                            cutlass.Int32(0),
                                            cutlass.Int32(7),
                                            cutlass.Int32(1),
                                        ):
                                            if mw > w_2:
                                                bincl[0] = cutlass.Int32(
                                                    (bincl[0] + _smem_ctl[(256 + w_2)])
                                                )
                                        babove = cutlass.Int32((bincl[0] - bc[0]))
                                        if ((tid_1 < 256) & (babove < bneed[0])) & (
                                            bneed[0] <= (babove + bc[0])
                                        ):
                                            prims.store_ext(
                                                cutlass.Int32(bbin).ir_value(),
                                                (smem_ctl + (272)),
                                            )
                                            prims.store_ext(
                                                cutlass.Int32(
                                                    (bneed[0] - babove)
                                                ).ir_value(),
                                                (smem_ctl + (273)),
                                            )
                                            prims.store_ext(
                                                cutlass.Int32(bc[0]).ir_value(),
                                                (smem_ctl + (274)),
                                            )
                                        cute.arch.barrier(
                                            barrier_id=2, number_of_threads=512
                                        )
                                        bsel_digit = cutlass.Int32(_smem_ctl[272])
                                        bsel_need = cutlass.Int32(_smem_ctl[273])
                                        bsel_count = cutlass.Int32(_smem_ctl[274])
                                        bprefix[0] = cutlass.Uint64(
                                            (
                                                bprefix[0]
                                                | (
                                                    cutlass.Uint64(bsel_digit)
                                                    << cutlass.Uint64(bshift)
                                                )
                                            )
                                        )
                                        bneed[0] = cutlass.Int32(bsel_need)
                                        _if_condition_76 = cutlass.Boolean(
                                            (bsel_count == bsel_need)
                                        )
                                        bdone[0] = cutlass.Int32(
                                            cutlass.select_(
                                                _if_condition_76,
                                                cutlass.Int32(1),
                                                bdone[0],
                                            )
                                        )
                                if tid_1 == 0:
                                    prims.store_ext(
                                        cutlass.Int32(0).ir_value(), (smem_ctl + (275))
                                    )
                                bck = cute.make_rmem_tensor((8,), cutlass.Uint64)
                                for s_5 in cutlass.range(
                                    cutlass.Int32(0), cutlass.Int32(8), cutlass.Int32(1)
                                ):
                                    bidx_1 = cutlass.Int32(((s_5 * 512) + tid_1))
                                    bck[s_5] = cutlass.Uint64(0)
                                    if bidx_1 < bn:
                                        bck[s_5] = cutlass.Uint64(
                                            cutlass.Uint64(
                                                _block_partial[(bseg + bidx_1)]
                                            )
                                        )
                                cute.arch.barrier(barrier_id=2, number_of_threads=512)
                                for s_6 in cutlass.range(
                                    cutlass.Int32(0), cutlass.Int32(8), cutlass.Int32(1)
                                ):
                                    bk_1 = cutlass.Uint64(bck[s_6])
                                    bckeep[0] = cutlass.Int32(0)
                                    _if_condition_77 = cutlass.Boolean(
                                        (
                                            (bn > ((s_6 * 512) + tid_1))
                                            & (bk_1 >= bprefix[0])
                                        )
                                    )
                                    bckeep[0] = cutlass.Int32(
                                        cutlass.select_(
                                            _if_condition_77,
                                            cutlass.Int32(1),
                                            bckeep[0],
                                        )
                                    )
                                    _vote_4 = cute.arch.vote_ballot_sync(
                                        cutlass.Boolean((bckeep[0] == 1)),
                                        cutlass.Uint32(0xFFFFFFFF),
                                    ).bitcast(cutlass.Uint32)
                                    bcballot = cutlass.Uint32(_vote_4)
                                    _popc_7 = cutlass.Int32(
                                        cute.arch.popc(cutlass.Uint32(bcballot))
                                    )
                                    bcn = cutlass.Uint32(cutlass.Uint32(_popc_7))
                                    if bcn != 0:
                                        _popc_8 = cutlass.Int32(
                                            cute.arch.popc(
                                                cutlass.Uint32((bcballot & lane_lt))
                                            )
                                        )
                                        bcrank = cutlass.Uint32(cutlass.Uint32(_popc_8))
                                        bcbase[0] = cutlass.Int32(0)
                                        if lane == 0:
                                            _atomic_old_6 = cute.arch.atomic_add(
                                                smem_ctl + 275,
                                                cutlass.Int32(cutlass.Int32(bcn)),
                                                sem="relaxed",
                                                scope="cta",
                                            )
                                            bcbase[0] = cutlass.Int32(_atomic_old_6)
                                        _shfl_13 = cute.arch.shuffle_sync(
                                            bcbase[0],
                                            0,
                                            mask=0xFFFFFFFF,
                                            mask_and_clamp=31,
                                        )
                                        bcbase[0] = cutlass.Int32(_shfl_13)
                                        if bckeep[0] == 1:
                                            prims.store_ext(
                                                cutlass.Int64(
                                                    cutlass.Int64(bk_1)
                                                ).ir_value(),
                                                (
                                                    block_partial
                                                    + (
                                                        (bseg + bcbase[0])
                                                        + cutlass.Int32(bcrank)
                                                    )
                                                ),
                                            )
                                cute.arch.barrier(barrier_id=1, number_of_threads=512)
                                if tid_1 == 0:
                                    prims.store_ext(
                                        cutlass.Int32(_smem_ctl[275]).ir_value(),
                                        (smem_ctl + (280 + r2_4)),
                                    )
                                    prims.store_ext(
                                        cutlass.Uint64(bprefix[0]).ir_value(),
                                        (smem_thr + (4 + r2_4)),
                                    )
                    if fin[0] == 16:
                        cute.arch.barrier(barrier_id=2, number_of_threads=512)
                        for r2_5 in cutlass.range(
                            cutlass.Int32(0), cutlass.Int32(4), cutlass.Int32(1)
                        ):
                            row_r = cutlass.Int32((row0_1 + r2_5))
                            if row_r < rows:
                                n_out = cutlass.Int32(_smem_ctl[(276 + r2_5)])
                                out = cutlass.Int32(
                                    (((row_r * splits) + split_1) * 512)
                                )
                                for s_7 in cutlass.range(
                                    cutlass.Int32(0), cutlass.Int32(1), cutlass.Int32(1)
                                ):
                                    oidx = cutlass.Int32(((s_7 * 512) + tid_1))
                                    v[0] = cutlass.Uint64(0)
                                    if oidx < n_out:
                                        v[0] = cutlass.Uint64(
                                            _smem_keys[((r2_5 * 2048) + oidx)]
                                        )
                                    prims.store_ext(
                                        cutlass.Int64(cutlass.Int64(v[0])).ir_value(),
                                        (partial + (out + oidx)),
                                    )
                                if CANDIDATES:
                                    bn_out = cutlass.Int32(_smem_ctl[(280 + r2_5)])
                                    bout = cutlass.Int32((bbase + (r2_5 * bstride)))
                                    for s_8 in cutlass.range(
                                        cutlass.Int32(0),
                                        cutlass.Int32(4),
                                        cutlass.Int32(1),
                                    ):
                                        boidx = cutlass.Int32(((s_8 * 512) + tid_1))
                                        if boidx >= bn_out:
                                            prims.store_ext(
                                                cutlass.Int64(0).ir_value(),
                                                (block_partial + (bout + boidx)),
                                            )
                    cute.arch.barrier(barrier_id=2, number_of_threads=512)
                    if (fin[0] == 16) | (flush[0] == 0):
                        _loop_active_61[0] = False
                    _break_loop_counter_62[0] += cutlass.Int32(1)
                    if _break_loop_counter_62[0] >= cutlass.Int32(1048576):
                        _loop_active_61[0] = False
                for r2_6 in cutlass.range(
                    cutlass.Int32(0), cutlass.Int32(2), cutlass.Int32(1)
                ):
                    thr[r2_6] = cutlass.Uint64(_smem_thr[(rb + r2_6)])
                    if thr[r2_6] != 0:
                        _atomic_max_float_decode_78_bits = cutlass.Uint32(
                            (cutlass.Uint32((thr[r2_6] >> 32)) << 16)
                        )
                        _atomic_max_float_decode_78_mask = cutlass.Uint32(
                            (_atomic_max_float_decode_78_bits >> 31) - cutlass.Uint32(1)
                        ) | cutlass.Uint32(0x80000000)
                        _amf_dec_0 = (
                            _atomic_max_float_decode_78_bits
                            ^ _atomic_max_float_decode_78_mask
                        ).bitcast(cutlass.Float32)
                        thr_f[r2_6] = cutlass.Float32(_amf_dec_0)
            if flush[0] == 1:
                _loop_active_35[0] = False
            _break_loop_counter_36[0] += cutlass.Int32(1)
            if _break_loop_counter_36[0] >= cutlass.Int32(1073741824):
                _loop_active_35[0] = False
    cute.arch.sync_threads()
    if warp == 0:
        cute.arch.dealloc_tmem(_tmem_base_ptr, 512, is_two_cta=False, arch="sm_100")
        cute.arch.relinquish_tmem_alloc_permit(is_two_cta=False)


@cute.jit
def launch_fp4_indexer_fused_candidates(
    query: cute.Tensor,
    query_dim_0: cutlass.Int64,
    query_dim_1: cutlass.Int64,
    query_stride16_0: cutlass.Int64,
    query_scales: cute.Tensor,
    query_scales_dim_0: cutlass.Int64,
    query_scales_dim_1: cutlass.Int64,
    query_scales_stride16_0: cutlass.Int64,
    cache: cute.Tensor,
    cache_dim_0: cutlass.Int64,
    cache_dim_1: cutlass.Int64,
    cache_dim_2: cutlass.Int64,
    cache_stride16_0: cutlass.Int64,
    cache_stride16_1: cutlass.Int64,
    weights: cute.Tensor,
    lengths: cute.Tensor,
    page_table: cute.Tensor,
    partial: cute.Tensor,
    block_partial: cute.Tensor,
    rows: cutlass.Int32,
    table_width: cutlass.Int32,
    splits: cutlass.Int32,
    grid_x: cutlass.Int32,
    grid_y: cutlass.Int32,
    grid_z: cutlass.Int32,
    stream: cuda.CUstream,
):
    _tma_query = create_tensor_map_tiled(
        query.iterator.toint(),
        cutlass.Uint8,
        [query_dim_0, query_dim_1],
        [query_stride16_0],
        [64, 128],
        swizzle=TensorMapSwizzle.s64b,
        l2_promotion=TensorMapL2Promotion.none,
        oob_fill=TensorMapFloatOOBFill.none,
    )
    _tma_query_scales = create_tensor_map_tiled(
        query_scales.iterator.toint(),
        cutlass.Int32,
        [query_scales_dim_0, query_scales_dim_1],
        [query_scales_stride16_0],
        [32, 4],
        swizzle=TensorMapSwizzle.none,
        l2_promotion=TensorMapL2Promotion.none,
        oob_fill=TensorMapFloatOOBFill.none,
    )
    _tma_cache = create_tensor_map_tiled(
        cache.iterator.toint(),
        cutlass.Uint8,
        [cache_dim_0, cache_dim_1, cache_dim_2],
        [cache_stride16_0, cache_stride16_1],
        [64, 136, 1],
        swizzle=TensorMapSwizzle.s64b,
        l2_promotion=TensorMapL2Promotion.none,
        oob_fill=TensorMapFloatOOBFill.none,
    )
    kernel_fp4_indexer_fused_candidates(
        _tma_query,
        _tma_query_scales,
        _tma_cache,
        weights.iterator,
        lengths.iterator,
        page_table.iterator,
        partial.iterator,
        block_partial.iterator,
        rows,
        table_width,
        splits,
    ).launch(
        grid=(grid_x, grid_y, grid_z),
        block=(640, 1, 1),
        smem=148352,
        min_blocks_per_mp=1,
        stream=stream,
    )


def compile_program():
    return cute.compile(
        launch_fp4_indexer_fused_candidates,
        make_fake_tensor(
            cutlass.Uint8, (cute.sym_int64(symbol="query"),), (1,), assumed_align=16
        ),
        cutlass.Int64(0),
        cutlass.Int64(0),
        cutlass.Int64(0),
        make_fake_tensor(
            cutlass.Int32,
            (cute.sym_int64(symbol="query_scales"),),
            (1,),
            assumed_align=16,
        ),
        cutlass.Int64(0),
        cutlass.Int64(0),
        cutlass.Int64(0),
        make_fake_tensor(
            cutlass.Uint8, (cute.sym_int64(symbol="cache"),), (1,), assumed_align=16
        ),
        cutlass.Int64(0),
        cutlass.Int64(0),
        cutlass.Int64(0),
        cutlass.Int64(0),
        cutlass.Int64(0),
        make_fake_tensor(
            cutlass.Float32, (cute.sym_int64(symbol="weights"),), (1,), assumed_align=16
        ),
        make_fake_tensor(
            cutlass.Int32, (cute.sym_int64(symbol="lengths"),), (1,), assumed_align=16
        ),
        make_fake_tensor(
            cutlass.Int32,
            (cute.sym_int64(symbol="page_table"),),
            (1,),
            assumed_align=16,
        ),
        make_fake_tensor(
            cutlass.Int64, (cute.sym_int64(symbol="partial"),), (1,), assumed_align=16
        ),
        make_fake_tensor(
            cutlass.Int64,
            (cute.sym_int64(symbol="block_partial"),),
            (1,),
            assumed_align=16,
        ),
        cutlass.Int32(0),
        cutlass.Int32(0),
        cutlass.Int32(0),
        cutlass.Int32(1),
        cutlass.Int32(1),
        cutlass.Int32(1),
        make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi --ptxas-options=--opt-level=2 --gpu-arch=sm_100a",
    )
