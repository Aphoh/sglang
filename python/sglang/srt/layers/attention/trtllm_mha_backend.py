from __future__ import annotations

"""
Support attention backend for TRTLLM MHA kernels from flashinfer.
The kernel supports sm100 only, with sliding window and attention sink features.
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.flashinfer_backend import (
    FlashInferAttnBackend,
    FlashInferMultiStepDraftBackend,
)
from sglang.srt.layers.attention.triton_ops.trtllm_fp8_kv_kernel import (
    fused_fp8_set_kv_buffer,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.utils import get_bool_env_var, is_flashinfer_available

logger = logging.getLogger(__name__)

if is_flashinfer_available():
    import flashinfer

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpecInput

# Constants
# Default workspace size in MB for TRTLLM MHA
# Can be configured via SGLANG_FLASHINFER_WORKSPACE_SIZE environment variable
DEFAULT_WORKSPACE_SIZE_MB = 512

# Reuse this workspace buffer across all TRTLLM MHA wrappers
global_zero_init_workspace_buffer = None
_eagle_cg_dbg_count = 0
_EAGLE_CG_DBG_MAX_LOGS = 2000


@dataclass
class TRTLLMMHAMetadata:
    # Sequence lengths for the forward batch
    cache_seqlens_int32: torch.Tensor = None
    # Maximum sequence length for query
    max_seq_len_q: int = 1
    # Maximum sequence length for key
    max_seq_len_k: int = 0
    # Cumulative sequence lengths for `query
    cu_seqlens_q: torch.Tensor = None
    # Cumulative sequence lengths for key
    cu_seqlens_k: torch.Tensor = None
    # Page table, the index of KV Cache Tables/Blocks
    page_table: torch.Tensor = None


class TRTLLMHAAttnBackend(FlashInferAttnBackend):
    """TRTLLM MHA attention kernel from flashinfer."""

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        kv_indptr_buf: Optional[torch.Tensor] = None,
        kv_last_page_len_buf: Optional[torch.Tensor] = None,
        speculative_step_id: int = 0,
    ):
        # Capture workspace size before super().__init__() to preserve user's
        # SGLANG_FLASHINFER_WORKSPACE_SIZE setting (may be overridden by parent)
        env_var = envs.SGLANG_FLASHINFER_WORKSPACE_SIZE
        workspace_size_bytes = (
            env_var.get()
            if env_var.is_set()
            else DEFAULT_WORKSPACE_SIZE_MB * 1024 * 1024
        )

        super().__init__(
            model_runner, skip_prefill, kv_indptr_buf, kv_last_page_len_buf
        )

        config = model_runner.model_config

        # MHA-specific dimensions
        self.max_context_len = model_runner.model_config.context_len
        self.hidden_size = config.hidden_size

        # Runtime parameters
        self.data_type = model_runner.kv_cache_dtype
        self.q_data_type = model_runner.dtype
        self.page_size = model_runner.page_size
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.device = model_runner.device

        # Workspace allocation
        self.workspace_size = workspace_size_bytes
        # Allocate buffers
        global global_zero_init_workspace_buffer
        if global_zero_init_workspace_buffer is None:
            global_zero_init_workspace_buffer = torch.zeros(
                self.workspace_size,
                dtype=torch.uint8,
                device=model_runner.device,
            )
        self.workspace_buffer = global_zero_init_workspace_buffer

        # CUDA graph state
        self.decode_cuda_graph_metadata = {}

        # Speculative decoding
        # Only support topk <= 1 for now.
        self.topk = model_runner.server_args.speculative_eagle_topk or 0
        self.speculative_step_id = speculative_step_id
        self.target_verify_metadata = {}

        self.speculative_num_draft_tokens = (
            model_runner.server_args.speculative_num_draft_tokens
        )

        # Forward metadata
        self.forward_metadata: Optional[TRTLLMMHAMetadata] = None

    @staticmethod
    def _is_eagle_cg_dbg_enabled() -> bool:
        return get_bool_env_var("EAGLE_CG_DBG")

    @staticmethod
    def _is_stream_capturing() -> bool:
        return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()

    @classmethod
    def _eagle_cg_dbg_log(cls, msg: str):
        global _eagle_cg_dbg_count
        if not cls._is_eagle_cg_dbg_enabled():
            return
        if cls._is_stream_capturing():
            return
        if _eagle_cg_dbg_count >= _EAGLE_CG_DBG_MAX_LOGS:
            return
        _eagle_cg_dbg_count += 1
        logger.info(f"EAGLE_CG_DBG trtllm_mha: {msg}")

    @staticmethod
    def _tensor_summary(tensor: Optional[torch.Tensor], max_items: int = 4) -> str:
        if tensor is None:
            return "None"
        numel = int(tensor.numel())
        if numel == 0:
            return "n=0"
        if tensor.is_cuda and TRTLLMHAAttnBackend._is_stream_capturing():
            return f"shape={tuple(tensor.shape)},n={numel},capture=1"
        flat = tensor.reshape(-1)
        head = flat[: min(max_items, numel)].tolist()
        tail = flat[max(0, numel - max_items) :].tolist()
        return (
            f"n={numel},sum={int(flat.sum().item())},max={int(flat.max().item())},"
            f"head={head},tail={tail}"
        )

    @staticmethod
    def _tensor_sig(tensor: Optional[torch.Tensor], max_sample: int = 256) -> str:
        if tensor is None:
            return "None"
        numel = int(tensor.numel())
        if numel == 0:
            return f"shape={tuple(tensor.shape)},n=0"
        if tensor.is_cuda and TRTLLMHAAttnBackend._is_stream_capturing():
            return f"shape={tuple(tensor.shape)},n={numel},capture=1"

        flat = tensor.reshape(-1)
        if numel > max_sample:
            step = max(1, numel // max_sample)
            sample = flat[::step][:max_sample]
        else:
            sample = flat

        sample_f = sample.float()
        nan_count = int(torch.isnan(sample_f).sum().item())
        inf_count = int(torch.isinf(sample_f).sum().item())
        min_v = float(sample_f.min().item())
        max_v = float(sample_f.max().item())
        sum_v = float(sample_f.sum().item())
        l2_v = float((sample_f * sample_f).sum().item())
        head = sample[: min(4, int(sample.numel()))].tolist()
        return (
            f"shape={tuple(tensor.shape)},n={numel},sample={int(sample.numel())},"
            f"nan={nan_count},inf={inf_count},sum={sum_v:.4e},l2={l2_v:.4e},"
            f"min={min_v:.4e},max={max_v:.4e},head={head}"
        )

    def _log_metadata_state(
        self,
        stage: str,
        forward_mode: ForwardMode,
        bs: int,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        metadata: Optional[TRTLLMMHAMetadata],
        req_pool_indices: Optional[torch.Tensor] = None,
        spec_info: Optional[SpecInput] = None,
    ):
        if self._is_stream_capturing():
            return
        if metadata is None:
            self._eagle_cg_dbg_log(
                f"{stage}: mode={forward_mode}, bs={bs}, metadata=None, "
                f"seq_lens={self._tensor_summary(seq_lens)}, "
                f"seq_lens_cpu={self._tensor_summary(seq_lens_cpu)}"
            )
            return
        req_summary = "None"
        if req_pool_indices is not None:
            req_summary = self._tensor_summary(req_pool_indices)
        page_shape = (
            tuple(metadata.page_table.shape) if metadata.page_table is not None else None
        )
        page_row0_summary = (
            self._tensor_summary(metadata.page_table[0, : min(16, metadata.page_table.shape[1])])
            if metadata.page_table is not None and metadata.page_table.ndim == 2 and metadata.page_table.shape[0] > 0
            else None
        )
        cu_q_last = (
            int(metadata.cu_seqlens_q[-1].item())
            if metadata.cu_seqlens_q is not None and metadata.cu_seqlens_q.numel() > 0
            else None
        )
        cu_k_last = (
            int(metadata.cu_seqlens_k[-1].item())
            if metadata.cu_seqlens_k is not None and metadata.cu_seqlens_k.numel() > 0
            else None
        )
        spec_name = type(spec_info).__name__ if spec_info is not None else None
        self._eagle_cg_dbg_log(
            f"{stage}: mode={forward_mode}, bs={bs}, spec={spec_name}, "
            f"max_seq_len_q={metadata.max_seq_len_q}, max_seq_len_k={metadata.max_seq_len_k}, "
            f"cache={self._tensor_summary(metadata.cache_seqlens_int32)}, "
            f"seq_lens={self._tensor_summary(seq_lens)}, "
            f"seq_lens_cpu={self._tensor_summary(seq_lens_cpu)}, "
            f"cu_q={self._tensor_summary(metadata.cu_seqlens_q)}, "
            f"cu_k={self._tensor_summary(metadata.cu_seqlens_k)}, "
            f"cu_q_last={cu_q_last}, cu_k_last={cu_k_last}, "
            f"page_table_shape={page_shape}, page_table_row0={page_row0_summary}, "
            f"req_pool_indices={req_summary}"
        )

    def init_cuda_graph_state(
        self,
        max_bs: int,
        max_num_tokens: int,
        kv_indices_buf: Optional[torch.Tensor] = None,
    ):
        """Initialize CUDA graph state for TRTLLM MHA."""
        max_num_pages = (self.max_context_len + self.page_size - 1) // self.page_size
        self.decode_cuda_graph_metadata = {
            "cache_seqlens": torch.zeros(max_bs, dtype=torch.int32, device=self.device),
            "page_table": torch.zeros(
                max_bs,
                max_num_pages,
                dtype=torch.int32,
                device=self.device,
            ),
            "strided_indices": torch.arange(
                0, self.max_context_len, self.page_size, device=self.device
            ),
        }

        if (
            self.speculative_num_draft_tokens is not None
            and self.speculative_num_draft_tokens > 0
        ):
            self.decode_cuda_graph_metadata["cu_seqlens_q"] = torch.arange(
                0, max_bs + 1, dtype=torch.int32, device=self.device
            )
            self.decode_cuda_graph_metadata["cu_seqlens_k"] = torch.zeros(
                max_bs + 1, dtype=torch.int32, device=self.device
            )
            self.decode_cuda_graph_metadata["page_table_draft_decode"] = torch.zeros(
                max_bs,
                max_num_pages,
                dtype=torch.int32,
                device=self.device,
            )
            self.target_verify_metadata = {
                "cache_seqlens": torch.zeros(
                    max_bs, dtype=torch.int32, device=self.device
                ),
                "cu_seqlens_q": torch.arange(
                    0,
                    max_bs * self.speculative_num_draft_tokens + 1,
                    step=self.speculative_num_draft_tokens,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "cu_seqlens_k": torch.zeros(
                    max_bs + 1, dtype=torch.int32, device=self.device
                ),
                "page_table": torch.zeros(
                    max_bs,
                    max_num_pages,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "strided_indices": torch.arange(
                    0, self.max_context_len, self.page_size, device=self.device
                ),
            }

            self.draft_extend_metadata = {
                "cache_seqlens": torch.zeros(
                    max_bs, dtype=torch.int32, device=self.device
                ),
                "cu_seqlens_q": torch.zeros(
                    max_bs + 1,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "cu_seqlens_k": torch.zeros(
                    max_bs + 1, dtype=torch.int32, device=self.device
                ),
                "page_table": torch.zeros(
                    max_bs,
                    max_num_pages,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "strided_indices": torch.arange(
                    0, self.max_context_len, self.page_size, device=self.device
                ),
            }

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
    ):
        """Initialize metadata for CUDA graph capture."""
        metadata = TRTLLMMHAMetadata()
        device = seq_lens.device
        seq_lens_for_log = seq_lens[:bs]

        if forward_mode.is_decode_or_idle():
            if spec_info is not None:
                # Draft Decode
                # Here we only support topk = 1 for now.
                metadata.cache_seqlens_int32 = self.decode_cuda_graph_metadata[
                    "cache_seqlens"
                ][:bs]
                metadata.max_seq_len_k = seq_lens.max().item() + (
                    self.speculative_step_id + 1
                )
                metadata.cu_seqlens_q = self.decode_cuda_graph_metadata["cu_seqlens_q"][
                    : bs + 1
                ]
                metadata.cu_seqlens_k = torch.nn.functional.pad(
                    torch.cumsum(
                        metadata.cache_seqlens_int32, dim=0, dtype=torch.int32
                    ),
                    (1, 0),
                )
                metadata.page_table = self.decode_cuda_graph_metadata[
                    "page_table_draft_decode"
                ][:bs, :]
                self.decode_cuda_graph_metadata[bs] = metadata
            else:
                # Normal Decode
                # Get sequence information
                metadata.cache_seqlens_int32 = seq_lens[:bs].to(torch.int32)
                batch_size = len(seq_lens)
                metadata.cu_seqlens_k = torch.nn.functional.pad(
                    torch.cumsum(seq_lens, dim=0, dtype=torch.int32), (1, 0)
                )

                # Precompute maximum sequence length
                metadata.max_seq_len_k = seq_lens.max().item()
                # Precompute cumulative sequence lengths
                metadata.cu_seqlens_q = torch.arange(
                    0, batch_size + 1, dtype=torch.int32, device=device
                )
                # Precompute page table
                metadata.page_table = self.decode_cuda_graph_metadata["page_table"][
                    :bs, :
                ]
                self.decode_cuda_graph_metadata[bs] = metadata
        elif forward_mode.is_target_verify():
            # Target Verify
            # Here we only support topk = 1 for now.
            metadata.cache_seqlens_int32 = self.target_verify_metadata["cache_seqlens"][
                :bs
            ]
            metadata.cache_seqlens_int32.copy_(
                (seq_lens + self.speculative_num_draft_tokens)
            )

            metadata.cu_seqlens_q = torch.arange(
                0,
                bs * self.speculative_num_draft_tokens + 1,
                self.speculative_num_draft_tokens,
                dtype=torch.int32,
                device=device,
            )

            metadata.cu_seqlens_k = self.target_verify_metadata["cu_seqlens_k"][
                : (bs + 1)
            ]

            metadata.max_seq_len_q = self.speculative_num_draft_tokens
            metadata.max_seq_len_k = (
                seq_lens.max().item() + self.speculative_num_draft_tokens
            )

            metadata.page_table = self.target_verify_metadata["page_table"][:bs, :]

            self.target_verify_metadata[bs] = metadata
        elif forward_mode.is_draft_extend():
            metadata.cache_seqlens_int32 = self.draft_extend_metadata["cache_seqlens"][
                :bs
            ]
            metadata.cache_seqlens_int32.copy_(seq_lens)
            num_tokens_per_bs = num_tokens // bs
            metadata.cu_seqlens_q = torch.arange(
                0,
                bs * num_tokens_per_bs + 1,
                num_tokens_per_bs,
                dtype=torch.int32,
                device=device,
            )

            metadata.cu_seqlens_k = self.draft_extend_metadata["cu_seqlens_k"][
                : (bs + 1)
            ]
            num_tokens_per_bs = num_tokens // bs
            metadata.max_seq_len_q = num_tokens_per_bs
            metadata.max_seq_len_k = seq_lens.max().item()

            metadata.page_table = self.draft_extend_metadata["page_table"][:bs, :]

            self.draft_extend_metadata[bs] = metadata
        self.forward_metadata = metadata
        self._log_metadata_state(
            stage="capture_cuda_graph",
            forward_mode=forward_mode,
            bs=bs,
            seq_lens=seq_lens_for_log,
            seq_lens_cpu=None,
            metadata=metadata,
            req_pool_indices=req_pool_indices[:bs],
            spec_info=spec_info,
        )

    @staticmethod
    def _safe_max_seq_len(
        seq_lens_cpu: Optional[torch.Tensor], seq_lens: torch.Tensor
    ) -> int:
        if seq_lens_cpu is not None and seq_lens_cpu.numel() > 0:
            return int(seq_lens_cpu.max().item())
        if seq_lens.numel() > 0:
            return int(seq_lens.max().item())
        return 0

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        """Replay CUDA graph with new inputs."""
        seq_lens = seq_lens[:bs]
        seq_lens_cpu = seq_lens_cpu[:bs] if seq_lens_cpu is not None else None
        req_pool_indices = req_pool_indices[:bs]
        max_len = self._safe_max_seq_len(seq_lens_cpu, seq_lens)
        metadata = None
        if forward_mode.is_decode_or_idle():
            if spec_info is not None:
                # Draft Decode
                # Here we only support topk = 1 for now.
                metadata = self.decode_cuda_graph_metadata[bs]
                metadata.max_seq_len_k = max_len + self.speculative_step_id + 1

                max_seq_pages = (
                    metadata.max_seq_len_k + self.page_size - 1
                ) // self.page_size

                metadata.cache_seqlens_int32.copy_(
                    seq_lens + self.speculative_step_id + 1
                )
            else:
                # Normal Decode
                metadata = self.decode_cuda_graph_metadata[bs]
                max_seq_pages = (max_len + self.page_size - 1) // self.page_size
                metadata.max_seq_len_k = max_len

                metadata.cache_seqlens_int32.copy_(seq_lens)

            metadata.cu_seqlens_k[1:].copy_(
                torch.cumsum(metadata.cache_seqlens_int32, dim=0, dtype=torch.int32)
            )
            page_indices = self.req_to_token[
                req_pool_indices[:, None],
                self.decode_cuda_graph_metadata["strided_indices"][:max_seq_pages][
                    None, :
                ],
            ]
            metadata.page_table[:, :max_seq_pages].copy_(page_indices // self.page_size)
        elif forward_mode.is_target_verify():
            # Here we only support topk = 1 for now.
            metadata = self.target_verify_metadata[bs]
            metadata.cache_seqlens_int32.copy_(
                (seq_lens + self.speculative_num_draft_tokens)
            )

            metadata.max_seq_len_k = (
                max_len + self.speculative_num_draft_tokens
            )
            metadata.cu_seqlens_k[1:].copy_(
                torch.cumsum(metadata.cache_seqlens_int32, dim=0, dtype=torch.int32)
            )
            max_seq_pages = (
                metadata.max_seq_len_k + self.page_size - 1
            ) // self.page_size
            page_indices = self.req_to_token[
                req_pool_indices[:, None],
                self.decode_cuda_graph_metadata["strided_indices"][:max_seq_pages],
            ]
            page_indices //= self.page_size
            metadata.page_table[:, :max_seq_pages].copy_(page_indices)
            metadata.max_seq_len_q = self.speculative_num_draft_tokens
        elif forward_mode.is_draft_extend():
            metadata = self.draft_extend_metadata[bs]
            metadata.cache_seqlens_int32.copy_(seq_lens)

            metadata.max_seq_len_k = max_len
            metadata.cu_seqlens_k[1:].copy_(
                torch.cumsum(metadata.cache_seqlens_int32, dim=0, dtype=torch.int32)
            )
            accept_length = spec_info.accept_length[:bs]
            if spec_info.accept_length_cpu:
                metadata.max_seq_len_q = max(spec_info.accept_length_cpu) + 1
            else:
                metadata.max_seq_len_q = 1

            metadata.cu_seqlens_q[1:].copy_(
                torch.cumsum(accept_length, dim=0, dtype=torch.int32)
            )

            max_seq_pages = (
                metadata.max_seq_len_k + self.page_size - 1
            ) // self.page_size
            page_indices = self.req_to_token[
                req_pool_indices[:, None],
                self.draft_extend_metadata["strided_indices"][:max_seq_pages],
            ]
            metadata.page_table[:, :max_seq_pages].copy_(page_indices // self.page_size)
        self.forward_metadata = metadata
        self._log_metadata_state(
            stage="replay_cuda_graph",
            forward_mode=forward_mode,
            bs=bs,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            metadata=metadata,
            req_pool_indices=req_pool_indices,
            spec_info=spec_info,
        )

    def get_cuda_graph_seq_len_fill_value(self) -> int:
        """Get the fill value for sequence lengths in CUDA graph."""
        return 1

    def _should_use_fused_fp8_path(self, save_kv_cache: bool, k: torch.Tensor) -> bool:
        """Check if we should use the fused FP8 KV cache write path."""
        return save_kv_cache and k is not None and self.data_type == torch.float8_e4m3fn

    def _fused_fp8_set_kv_buffer(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        **kwargs,
    ):
        """Fused FP8 quantization and KV cache write."""
        cache_loc = forward_batch.out_cache_loc

        # Get K/V cache buffers from token_to_kv_pool
        k_cache, v_cache = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)

        fused_fp8_set_kv_buffer(
            k=k,
            v=v,
            k_cache=k_cache,
            v_cache=v_cache,
            cache_loc=cache_loc,
            k_scale=layer.k_scale,  # May be None
            v_scale=layer.v_scale,  # May be None
            page_size=self.page_size,
        )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Initialize the metadata for a forward pass."""

        metadata = TRTLLMMHAMetadata()
        seqlens_in_batch = forward_batch.seq_lens
        batch_size = forward_batch.batch_size
        device = seqlens_in_batch.device
        max_len = self._safe_max_seq_len(forward_batch.seq_lens_cpu, seqlens_in_batch)

        if batch_size == 0:
            metadata.cache_seqlens_int32 = seqlens_in_batch.to(torch.int32)
            metadata.max_seq_len_q = 0
            metadata.max_seq_len_k = 0
            metadata.cu_seqlens_q = torch.zeros(1, dtype=torch.int32, device=device)
            metadata.cu_seqlens_k = torch.zeros(1, dtype=torch.int32, device=device)
            metadata.page_table = forward_batch.req_to_token_pool.req_to_token[
                forward_batch.req_pool_indices, :0
            ]
            self.forward_metadata = metadata
            self._log_metadata_state(
                stage="init_forward",
                forward_mode=forward_batch.forward_mode,
                bs=batch_size,
                seq_lens=seqlens_in_batch,
                seq_lens_cpu=forward_batch.seq_lens_cpu,
                metadata=metadata,
                req_pool_indices=forward_batch.req_pool_indices,
                spec_info=forward_batch.spec_info,
            )
            return

        if forward_batch.forward_mode.is_decode_or_idle():
            if forward_batch.spec_info is not None:
                # Draft Decode
                # Here we only support topk = 1 for now.
                metadata.cache_seqlens_int32 = (
                    seqlens_in_batch + (self.speculative_step_id + 1)
                ).to(torch.int32)
                metadata.max_seq_len_k = max_len + (self.speculative_step_id + 1)
                metadata.cu_seqlens_q = torch.arange(
                    0, batch_size + 1, dtype=torch.int32, device=device
                )
                metadata.cu_seqlens_k = torch.nn.functional.pad(
                    torch.cumsum(
                        metadata.cache_seqlens_int32, dim=0, dtype=torch.int32
                    ),
                    (1, 0),
                )
                metadata.page_table = forward_batch.req_to_token_pool.req_to_token[
                    forward_batch.req_pool_indices, : metadata.max_seq_len_k
                ]
            else:
                # Normal Decode
                metadata.cache_seqlens_int32 = seqlens_in_batch.to(torch.int32)
                metadata.max_seq_len_k = max_len
                metadata.cu_seqlens_q = torch.arange(
                    0, batch_size + 1, dtype=torch.int32, device=device
                )
                metadata.cu_seqlens_k = torch.nn.functional.pad(
                    torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0)
                )
                metadata.page_table = forward_batch.req_to_token_pool.req_to_token[
                    forward_batch.req_pool_indices, : metadata.max_seq_len_k
                ]
        elif forward_batch.forward_mode.is_target_verify():
            # Only support topk = 1 for now.
            metadata.cache_seqlens_int32 = (
                forward_batch.seq_lens + self.speculative_num_draft_tokens
            ).to(torch.int32)
            metadata.max_seq_len_q = self.speculative_num_draft_tokens
            metadata.max_seq_len_k = (
                max_len + self.speculative_num_draft_tokens
            )
            metadata.cu_seqlens_q = torch.arange(
                0,
                batch_size * self.speculative_num_draft_tokens + 1,
                self.speculative_num_draft_tokens,
                dtype=torch.int32,
                device=device,
            )
            metadata.cu_seqlens_k = torch.nn.functional.pad(
                torch.cumsum(metadata.cache_seqlens_int32, dim=0, dtype=torch.int32),
                (1, 0),
            )
            metadata.page_table = forward_batch.req_to_token_pool.req_to_token[
                forward_batch.req_pool_indices, : metadata.max_seq_len_k
            ]

        else:
            metadata.cache_seqlens_int32 = seqlens_in_batch.to(torch.int32)
            metadata.max_seq_len_k = max_len
            metadata.cu_seqlens_k = torch.nn.functional.pad(
                torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0)
            )
            metadata.page_table = forward_batch.req_to_token_pool.req_to_token[
                forward_batch.req_pool_indices, : metadata.max_seq_len_k
            ]

            if any(
                forward_batch.extend_prefix_lens_cpu
            ) or forward_batch.forward_mode.is_draft_extend(include_v2=True):
                extend_seq_lens = forward_batch.extend_seq_lens
                # NOTE: in piecewise CUDA graph warmup, extend_seq_lens_cpu is a torch.Tensor;
                # Python's max() returns a 0-d tensor, but flashinfer expects an int.
                max_q = max(forward_batch.extend_seq_lens_cpu)
                metadata.max_seq_len_q = (
                    int(max_q.item()) if isinstance(max_q, torch.Tensor) else int(max_q)
                )
                metadata.cu_seqlens_q = torch.nn.functional.pad(
                    torch.cumsum(extend_seq_lens, dim=0, dtype=torch.int32), (1, 0)
                )
            else:
                metadata.max_seq_len_q = metadata.max_seq_len_k
                metadata.cu_seqlens_q = metadata.cu_seqlens_k

        # Convert the page table to a strided format
        if self.page_size > 1:
            self.strided_indices = torch.arange(
                0, metadata.page_table.shape[1], self.page_size, device=self.device
            )
            metadata.page_table = (
                metadata.page_table[:, self.strided_indices] // self.page_size
            )

        self.forward_metadata = metadata
        self._log_metadata_state(
            stage="init_forward",
            forward_mode=forward_batch.forward_mode,
            bs=batch_size,
            seq_lens=seqlens_in_batch,
            seq_lens_cpu=forward_batch.seq_lens_cpu,
            metadata=metadata,
            req_pool_indices=forward_batch.req_pool_indices,
            spec_info=forward_batch.spec_info,
        )

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """Run forward for decode using TRTLLM MHA kernel."""
        cache_loc = forward_batch.out_cache_loc

        use_fused_fp8_path = self._should_use_fused_fp8_path(save_kv_cache, k)

        if use_fused_fp8_path:
            # Use fused FP8 quantization + KV cache write path
            self._fused_fp8_set_kv_buffer(
                q=q,
                k=k,
                v=v,
                layer=layer,
                forward_batch=forward_batch,
            )
            k = None
            v = None
        else:
            # Use original set_kv_buffer path
            if save_kv_cache and k is not None:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                )

        if self.data_type == torch.float8_e4m3fn:
            q = q.to(torch.float8_e4m3fn)
        q = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
        k_cache, v_cache = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
        # shape conversion:
        # [num_pages, page_size, num_kv_heads, head_dim] -> [num_pages, num_kv_heads, page_size, head_dim]
        k_cache = k_cache.view(
            -1, self.page_size, layer.tp_k_head_num, layer.head_dim
        ).permute(0, 2, 1, 3)
        v_cache = v_cache.view(
            -1, self.page_size, layer.tp_v_head_num, layer.head_dim
        ).permute(0, 2, 1, 3)
        kv_cache = (k_cache, v_cache)

        # TODO: add support for quantization
        q_scale = 1.0
        k_scale = (
            layer.k_scale_float
            if getattr(layer, "k_scale_float", None) is not None
            else 1.0
        )
        bmm1_scale = q_scale * k_scale * layer.scaling
        bmm2_scale = 1.0
        # sink: additional value per head in the denominator of the softmax.
        attention_sink = kwargs.get("sinks", None)

        # Call TRT-LLM kernel
        # raw_out: like q, [bs, acc_q_len, num_q_heads, head_dim] but with output dtype
        o = flashinfer.decode.trtllm_batch_decode_with_kv_cache(
            query=q,
            kv_cache=kv_cache,
            workspace_buffer=self.workspace_buffer,
            block_tables=self.forward_metadata.page_table,
            seq_lens=self.forward_metadata.cache_seqlens_int32,
            max_seq_len=self.max_context_len,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            window_left=layer.sliding_window_size,
            # TODO: add attention_sink operation or nvfp4 scale factor if needed
            sinks=attention_sink,
            out_dtype=self.q_data_type,  # model_runner.dtype
        )

        return o.view(-1, layer.tp_q_head_num * layer.head_dim)

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        **kwargs,
    ):
        cache_loc = forward_batch.out_cache_loc

        use_fused_fp8_path = self._should_use_fused_fp8_path(save_kv_cache, k)

        if use_fused_fp8_path:
            # Use fused FP8 quantization + KV cache write path
            self._fused_fp8_set_kv_buffer(
                q=q,
                k=k,
                v=v,
                layer=layer,
                forward_batch=forward_batch,
            )
            k = None
            v = None
        else:
            # Use original set_kv_buffer path
            if save_kv_cache and k is not None:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                )

        if self.data_type == torch.float8_e4m3fn:
            q = q.to(torch.float8_e4m3fn)
        q = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
        # [num_pages, page_size, num_kv_heads, head_dim] -> [num_pages, num_kv_heads, page_size, head_dim]
        k_cache, v_cache = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
        k_cache = k_cache.view(
            -1, self.page_size, layer.tp_k_head_num, layer.head_dim
        ).permute(0, 2, 1, 3)
        v_cache = v_cache.view(
            -1, self.page_size, layer.tp_v_head_num, layer.head_dim
        ).permute(0, 2, 1, 3)
        kv_cache = (k_cache, v_cache)

        # sink: additional value per head in the denominator of the softmax.
        attention_sink = kwargs.get("sinks", None)
        # TODO: add support for quantization
        q_scale = 1.0
        k_scale = (
            layer.k_scale_float
            if getattr(layer, "k_scale_float", None) is not None
            else 1.0
        )
        bmm1_scale = q_scale * k_scale * layer.scaling
        bmm2_scale = 1.0

        if forward_batch.forward_mode.is_target_verify():
            self._eagle_cg_dbg_log(
                "verify_pre_kernel: "
                f"layer={layer.layer_id}, q={self._tensor_sig(q)}, "
                f"cache={self._tensor_summary(self.forward_metadata.cache_seqlens_int32)}, "
                f"cu_q={self._tensor_summary(self.forward_metadata.cu_seqlens_q)}, "
                f"cu_k={self._tensor_summary(self.forward_metadata.cu_seqlens_k)}, "
                f"page_table_shape={tuple(self.forward_metadata.page_table.shape) if self.forward_metadata.page_table is not None else None}, "
                f"q_len_per_req={self.forward_metadata.max_seq_len_q}, "
                f"max_seq_len_k={self.forward_metadata.max_seq_len_k}, "
                f"batch_size={forward_batch.batch_size}, "
                f"spec={type(forward_batch.spec_info).__name__ if forward_batch.spec_info is not None else None}"
            )
            o = flashinfer.decode.trtllm_batch_decode_with_kv_cache(
                query=q,
                kv_cache=kv_cache,
                workspace_buffer=self.workspace_buffer,
                block_tables=self.forward_metadata.page_table,
                seq_lens=self.forward_metadata.cache_seqlens_int32,
                max_seq_len=self.max_context_len,
                bmm1_scale=bmm1_scale,
                bmm2_scale=bmm2_scale,
                window_left=layer.sliding_window_size,
                # TODO: add attention_sink operation or nvfp4 scale factor if needed
                sinks=attention_sink,
                out_dtype=self.q_data_type,  # model_runner.dtype
                q_len_per_req=self.forward_metadata.max_seq_len_q,
            )
            self._eagle_cg_dbg_log(
                "verify_post_kernel: "
                f"layer={layer.layer_id}, out={self._tensor_sig(o)}"
            )
        else:

            o = flashinfer.prefill.trtllm_batch_context_with_kv_cache(
                query=q,
                kv_cache=kv_cache,
                workspace_buffer=self.workspace_buffer,
                block_tables=self.forward_metadata.page_table,
                seq_lens=self.forward_metadata.cache_seqlens_int32,
                max_q_len=self.forward_metadata.max_seq_len_q,
                max_kv_len=self.max_context_len,
                bmm1_scale=bmm1_scale,
                bmm2_scale=bmm2_scale,
                batch_size=forward_batch.batch_size,
                cum_seq_lens_q=self.forward_metadata.cu_seqlens_q,
                cum_seq_lens_kv=self.forward_metadata.cu_seqlens_k,
                window_left=layer.sliding_window_size,
                # TODO: add attention_sink operation or nvfp4 scale factor if needed
                sinks=attention_sink,
                out_dtype=self.q_data_type,  # model_runner.dtype
            )

        return o.view(-1, layer.tp_q_head_num * layer.head_dim)


class TRTLLMHAAttnMultiStepDraftBackend(FlashInferMultiStepDraftBackend):
    """Multi-step TRTLLM MHA attention kernel used by EAGLE."""

    def __init__(
        self, model_runner: ModelRunner, topk: int, speculative_num_steps: int
    ):
        super().__init__(model_runner, topk, speculative_num_steps)
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i] = TRTLLMHAAttnBackend(
                model_runner,
                skip_prefill=True,
                kv_indptr_buf=self.kv_indptr[i],
                kv_last_page_len_buf=self.kv_last_page_len,
                speculative_step_id=i,
            )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_forward_metadata(forward_batch)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_cuda_graph_state(max_bs, max_num_tokens)

    def init_forward_metadata_capture_cuda_graph(
        self,
        forward_batch: ForwardBatch,
    ):
        assert forward_batch.spec_info is not None
        assert forward_batch.spec_info.is_draft_input()

        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_forward_metadata_capture_cuda_graph(
                forward_batch.batch_size,
                forward_batch.batch_size * self.topk,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                encoder_lens=forward_batch.encoder_lens,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
            )

    def init_forward_metadata_replay_cuda_graph(
        self, forward_batch: ForwardBatch, bs: int
    ):
        assert forward_batch.spec_info is not None
        assert forward_batch.spec_info.is_draft_input()

        for i in range(self.speculative_num_steps - 1):

            self.attn_backends[i].init_forward_metadata_replay_cuda_graph(
                bs,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.seq_lens_sum,
                encoder_lens=forward_batch.encoder_lens,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
                seq_lens_cpu=forward_batch.seq_lens_cpu,
            )
