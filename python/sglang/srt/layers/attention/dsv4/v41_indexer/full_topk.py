"""The full top-k of the ratio-1/2 index layers outside the candidate scheme:
DeepGEMM or fused CuTe DSL on SM100, torch elsewhere."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from sglang.kernels.ops.attention.dsv4.index_logits import (
    deep_gemm_fp4_paged_mqa_logits,
)
from sglang.kernels.ops.attention.dsv4.topk import topk_transform_paged_torch
from sglang.srt.layers.attention.dsv4.indexer import topk_transform_paged_from_metadata

from .scoring import (
    decode_scores,
    dense_prefill_topk,
    get_deep_gemm_decode_data,
    get_deep_gemm_prefill_data,
    get_index_k_cache,
    prefill_requests,
    quantize_index_q,
    write_decode,
    write_prefill,
)
from .types import (
    CapturedPrefillInputs,
    DecodeInputs,
    PrefillInputs,
    Selection,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool


class FullTopKIndexer:
    def __init__(
        self,
        *,
        token_to_kv_pool: DeepSeekV4TokenToKVPool,
        req_to_token: torch.Tensor,
        use_deep_gemm_prefill: bool,
        use_deep_gemm_decode: bool,
        use_fused_indexer: bool,
    ):
        self.token_to_kv_pool = token_to_kv_pool
        self.req_to_token = req_to_token
        self.use_deep_gemm_prefill = use_deep_gemm_prefill
        self.use_deep_gemm_decode = use_deep_gemm_decode
        self.use_fused_indexer = use_fused_indexer

    def _fused_topk(
        self,
        *,
        indexer,
        q_fp4: torch.Tensor,
        q_sf: torch.Tensor,
        weights: torch.Tensor,
        k_cache: torch.Tensor,
        metadata,
        out: Selection,
        rows: slice | None = None,
    ) -> bool:
        """Return false for unsupported shapes."""
        if (
            not self.use_fused_indexer
            or metadata.compressed_page_size != 128
            or indexer.index_topk != 512
            or q_fp4.ndim != 4
            or q_fp4.shape[1] != 1
            or q_fp4.shape[2] != 32
            or q_fp4.shape[3] != 64
            or out.page_indices.shape[1] < 512
            or k_cache.ndim != 4
            or k_cache.shape[1:] != (128, 1, 68)
        ):
            return False

        row_slice = slice(0, q_fp4.shape[0]) if rows is None else rows
        q_fp4 = q_fp4[row_slice].reshape(-1, 32, 64)
        q_sf = q_sf[row_slice]
        if q_sf.ndim == 3 and q_sf.shape[1] == 1:
            q_sf = q_sf.squeeze(1)
        if q_sf.ndim != 2 or q_sf.shape[1] != 32:
            return False
        weights = weights[row_slice]
        lengths = metadata.compressed_seq_lens
        if lengths.ndim == 2 and lengths.shape[1] == 1:
            lengths = lengths.squeeze(1)
        if lengths.ndim != 1:
            return False
        lengths = lengths[row_slice]
        page_table = metadata.page_table[row_slice]
        page_indices = out.page_indices[row_slice]
        raw_indices = (
            out.raw_indices[row_slice] if out.raw_indices is not None else None
        )
        if page_indices.shape[0] != q_fp4.shape[0]:
            return False

        from sglang.kernels.ops.attention.dsv4.fused_indexer_sm100 import (
            fused_indexer_topk,
        )

        fused_indexer_topk(
            q_fp4,
            q_sf,
            weights,
            k_cache.view(k_cache.shape[0], 8704),
            lengths,
            page_table,
            page_indices,
            raw_indices,
        )
        return True

    def topk_prefill(self, inputs: PrefillInputs, out: Selection) -> None:
        if self.use_deep_gemm_prefill:
            self._deep_gemm_prefill(inputs, out)
        else:
            self._torch_prefill(inputs, out)

    def topk_prefill_captured(self, inputs: CapturedPrefillInputs, out: Selection):
        self._deep_gemm_prefill_captured(inputs, out)

    def topk_decode(self, inputs: DecodeInputs, out: Selection) -> None:
        if self.use_deep_gemm_decode:
            self._deep_gemm_decode(inputs, out)
        else:
            self._torch_decode(inputs, out)

    def _deep_gemm_prefill(self, inputs: PrefillInputs, out: Selection) -> None:
        out.reset()
        data = get_deep_gemm_prefill_data(inputs, self.req_to_token)
        if data is None:
            return
        kv = self.token_to_kv_pool.get_low_ratio_index_k_fp4(
            inputs.layer_id, data.k_slots
        )
        selected = dense_prefill_topk(data, kv, topk=inputs.indexer.index_topk)
        data.write_selection(selected=selected, out=out)

    def _torch_prefill(self, inputs: PrefillInputs, out: Selection) -> None:
        requests = prefill_requests(
            inputs=inputs,
            out=out,
            token_to_kv_pool=self.token_to_kv_pool,
            req_to_token=self.req_to_token,
        )
        for request, chunks in requests:
            for chunk in chunks:
                idx = chunk.scores.topk(request.k, dim=-1, sorted=False).indices
                write_prefill(out, request, chunk, idx)

    def _deep_gemm_decode(self, inputs: DecodeInputs, out: Selection) -> None:
        data = get_deep_gemm_decode_data(inputs, self.token_to_kv_pool)
        metadata = inputs.paged_metadata
        if self._fused_topk(
            indexer=inputs.indexer,
            q_fp4=data.q_fp4,
            q_sf=data.q_sf,
            weights=data.weights,
            k_cache=data.k_cache,
            metadata=metadata,
            out=out,
        ):
            return
        if isinstance(metadata.deep_gemm_metadata, list):
            topk_plans = metadata.topk_metadata_chunks
            assert not metadata.use_topk_v2 or topk_plans is not None
            for chunk_idx, (rows, plan) in enumerate(metadata.row_chunks()):
                logits = deep_gemm_fp4_paged_mqa_logits(
                    (data.q_fp4[rows], data.q_sf[rows]),
                    data.k_cache,
                    data.weights[rows],
                    metadata.compressed_seq_lens[rows],
                    metadata.page_table[rows],
                    plan,
                    metadata.max_compressed_seq_len,
                )
                # TODO(dark): add bf16 topk
                topk_transform_paged_from_metadata(
                    logits,
                    metadata,
                    out.page_indices,
                    out.raw_indices,
                    rows=rows,
                    topk_metadata=(
                        topk_plans[chunk_idx] if topk_plans is not None else None
                    ),
                )
            return
        logits = deep_gemm_fp4_paged_mqa_logits(
            (data.q_fp4, data.q_sf),
            data.k_cache,
            data.weights,
            metadata.compressed_seq_lens,
            metadata.page_table,
            metadata.deep_gemm_metadata,
            metadata.max_compressed_seq_len,
        )
        # TODO(dark): add bf16 topk
        topk_transform_paged_from_metadata(
            logits, metadata, out.page_indices, out.raw_indices
        )

    def _torch_decode(self, inputs: DecodeInputs, out: Selection) -> None:
        d = decode_scores(
            inputs=inputs,
            out=out,
            token_to_kv_pool=self.token_to_kv_pool,
            req_to_token=self.req_to_token,
        )
        if d is None:
            return
        k = min(inputs.indexer.index_topk, d.lmax)
        idx = d.scores.topk(k, dim=-1, sorted=False).indices
        write_decode(out, d, idx)

    def _deep_gemm_prefill_captured(
        self, inputs: CapturedPrefillInputs, out: Selection
    ) -> None:
        indexer = inputs.indexer
        metadata = inputs.paged_metadata
        assert indexer.n_local_heads == indexer.n_heads
        q_fp4, q_sf = quantize_index_q(inputs.q)
        num_tokens, num_heads = q_fp4.shape[0], q_fp4.shape[1]
        q_fp4 = q_fp4.view(num_tokens, 1, num_heads, 64)
        q_sf = q_sf.view(num_tokens, 1, num_heads)
        weights = inputs.weights.float()
        page_size = metadata.compressed_page_size
        k_cache = get_index_k_cache(
            token_to_kv_pool=self.token_to_kv_pool,
            layer_id=inputs.layer_id,
            page_size=page_size,
        )

        if self._fused_topk(
            indexer=indexer,
            q_fp4=q_fp4,
            q_sf=q_sf,
            weights=weights,
            k_cache=k_cache,
            metadata=metadata,
            out=out,
        ):
            return

        width = metadata.max_compressed_seq_len
        lens = metadata.compressed_seq_lens
        page_table = metadata.page_table
        topk = min(indexer.index_topk, width)
        for rows, plan in metadata.row_chunks():
            logits = deep_gemm_fp4_paged_mqa_logits(
                (q_fp4[rows], q_sf[rows]),
                k_cache,
                weights[rows],
                lens[rows],
                page_table[rows],
                plan,
                width,
            )
            topk_transform_paged_torch(
                logits,
                lens[rows],
                page_table[rows],
                out.page_indices[rows, :topk],
                page_size,
                out.raw_indices[rows, :topk] if out.raw_indices is not None else None,
            )
