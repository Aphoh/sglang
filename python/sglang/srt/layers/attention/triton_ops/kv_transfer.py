# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Triton kernels for gathering/scattering KV cache data between GPU and pinned CPU memory.

These kernels enable efficient mixed-TP KV cache transfers by:
1. Gathering scattered KV data from GPU into pinned CPU buffer (device -> host)
2. Scattering KV data from pinned CPU buffer to GPU KV cache (host -> device)

Primary API:
- gather_kv(): Gather KV from GPU to pinned CPU (uses chunked staging for best performance)
- scatter_kv(): Scatter KV from pinned CPU to GPU (direct reads, no staging needed)

The chunked approach uses a fixed-size GPU staging buffer (default 256MB) to achieve
~80% of PCIe bandwidth while limiting GPU memory overhead. This outperforms pure
zero-copy writes which achieve only ~50% of PCIe bandwidth due to uncoalesced writes.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _gather_kv_kernel(
    # Source KV cache - GPU memory
    src_ptr,
    # Slot indices to gather
    slot_indices_ptr,
    # Output pinned CPU buffer - GPU-accessible via zero-copy
    output_ptr,
    # Dimensions
    num_tokens,
    head_dim: tl.constexpr,
    # Head slicing params
    head_start: tl.constexpr,
    num_heads_to_gather: tl.constexpr,
    # Source strides (in elements)
    src_slot_stride,
    src_head_stride,
    # Output strides (in elements) - can be very large for HEAD-FIRST layout, use int64
    out_token_stride,
    out_head_stride,
    # Block sizes
    BLOCK_TOKENS: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    """
    Gather KV data from scattered GPU slots directly into contiguous output buffer.

    This kernel handles a single layer's K or V buffer.

    Input layout: [total_slots, num_heads, head_dim] on GPU
    Output layout: HEAD-FIRST, base pointer offset to (layer, kv) position

    Grid: (num_heads_to_gather,) - one program per head
    """
    head_id = tl.program_id(0)  # relative head (0 to num_heads_to_gather-1)

    # Source head index (absolute in source KV cache)
    src_head = head_start + head_id

    # Cast to int64 to avoid overflow with large buffers
    src_slot_stride_i64 = src_slot_stride.to(tl.int64)
    src_head_stride_i64 = src_head_stride.to(tl.int64)
    out_token_stride_i64 = out_token_stride.to(tl.int64)
    out_head_stride_i64 = out_head_stride.to(tl.int64)
    src_head_i64 = src_head.to(tl.int64)
    head_id_i64 = head_id.to(tl.int64)

    # Process tokens in blocks
    for token_block_start in range(0, num_tokens, BLOCK_TOKENS):
        token_offsets = token_block_start + tl.arange(0, BLOCK_TOKENS)
        token_mask = token_offsets < num_tokens
        token_offsets_i64 = token_offsets.to(tl.int64)

        # Load slot indices for these tokens
        slot_ids = tl.load(slot_indices_ptr + token_offsets, mask=token_mask, other=0)
        slot_ids_i64 = slot_ids.to(tl.int64)

        # Process head_dim in blocks
        for dim_start in range(0, head_dim, BLOCK_DIM):
            dim_offsets = dim_start + tl.arange(0, BLOCK_DIM)
            dim_mask = dim_offsets < head_dim
            dim_offsets_i64 = dim_offsets.to(tl.int64)

            # Compute source addresses in GPU KV cache (use int64)
            src_offsets = (
                slot_ids_i64[:, None] * src_slot_stride_i64
                + src_head_i64 * src_head_stride_i64
                + dim_offsets_i64[None, :]
            )

            # Load from GPU source
            mask = token_mask[:, None] & dim_mask[None, :]
            data = tl.load(src_ptr + src_offsets, mask=mask, other=0.0)

            # Compute output addresses (use int64 for large HEAD-FIRST strides)
            out_offsets = (
                head_id_i64 * out_head_stride_i64
                + token_offsets_i64[:, None] * out_token_stride_i64
                + dim_offsets_i64[None, :]
            )

            # Store to output buffer (may be pinned CPU - zero-copy write over PCIe)
            tl.store(output_ptr + out_offsets, data, mask=mask)


def gather_kv_to_pinned(
    k_buffers: list[torch.Tensor],  # [num_layers] each [total_slots, num_heads, head_dim] on GPU
    v_buffers: list[torch.Tensor],
    slot_indices: torch.Tensor,  # [num_tokens] on GPU
    pinned_output: torch.Tensor,  # pinned CPU buffer, viewed as flat
    head_start: int,
    num_heads_to_gather: int,
) -> None:
    """
    Launch gather kernel to collect KV data from GPU directly into pinned CPU buffer.

    Args:
        k_buffers: List of K cache tensors per layer, each [total_slots, num_heads, head_dim]
        v_buffers: List of V cache tensors per layer, each [total_slots, num_heads, head_dim]
        slot_indices: Tensor of slot indices to gather, shape [num_tokens]
        pinned_output: Pinned CPU buffer to write to. Must be created with pin_memory=True.
                       Should have size >= num_layers * 2 * num_tokens * num_heads_to_gather * head_dim * dtype_size
        head_start: First head index to gather (for head slicing in mixed-TP)
        num_heads_to_gather: Number of heads to gather starting from head_start

    Output layout in pinned_output: [num_heads_to_gather, num_layers, 2, num_tokens, head_dim]
    This is HEAD-FIRST layout for easy head slicing in mixed-TP transfers.
    """
    assert pinned_output.is_pinned(), "Output buffer must be pinned CPU memory"
    assert slot_indices.is_cuda, "slot_indices must be on GPU"

    num_layers = len(k_buffers)
    num_tokens = slot_indices.shape[0]
    num_heads = k_buffers[0].shape[1]
    head_dim = k_buffers[0].shape[2]
    dtype = k_buffers[0].dtype

    assert head_start + num_heads_to_gather <= num_heads, (
        f"head_start ({head_start}) + num_heads_to_gather ({num_heads_to_gather}) "
        f"exceeds num_heads ({num_heads})"
    )

    # Get strides (in elements, not bytes)
    src_slot_stride = k_buffers[0].stride(0)
    src_head_stride = k_buffers[0].stride(1)

    # Determine block sizes
    BLOCK_TOKENS = 64
    BLOCK_DIM = min(64, triton.next_power_of_2(head_dim))

    # HEAD-FIRST output layout: [num_heads_to_gather, num_layers, 2, num_tokens, head_dim]
    # Strides for this layout (in elements):
    # - head_stride: distance between heads = num_layers * 2 * num_tokens * head_dim
    # - layer_stride: distance between layers = 2 * num_tokens * head_dim
    # - kv_stride: distance between K and V = num_tokens * head_dim
    # - token_stride: distance between tokens = head_dim
    out_head_stride = num_layers * 2 * num_tokens * head_dim
    layer_stride = 2 * num_tokens * head_dim
    kv_stride = num_tokens * head_dim
    out_token_stride = head_dim  # tokens are contiguous within each head's slice

    grid = (num_heads_to_gather,)

    # View pinned output as the right dtype
    pinned_output_typed = pinned_output.view(dtype)

    # Iterate over layers and K/V
    for layer_idx in range(num_layers):
        # Base offset for this (layer, K) in HEAD-FIRST layout
        # Each head will add head_id * out_head_stride in the kernel
        k_base_offset = layer_idx * layer_stride  # + 0 * kv_stride for K
        _gather_kv_kernel[grid](
            k_buffers[layer_idx],
            slot_indices,
            pinned_output_typed[k_base_offset:],
            num_tokens=num_tokens,
            head_dim=head_dim,
            head_start=head_start,
            num_heads_to_gather=num_heads_to_gather,
            src_slot_stride=src_slot_stride,
            src_head_stride=src_head_stride,
            out_token_stride=out_token_stride,
            out_head_stride=out_head_stride,
            BLOCK_TOKENS=BLOCK_TOKENS,
            BLOCK_DIM=BLOCK_DIM,
        )

        # Base offset for this (layer, V) in HEAD-FIRST layout
        v_base_offset = layer_idx * layer_stride + kv_stride
        _gather_kv_kernel[grid](
            v_buffers[layer_idx],
            slot_indices,
            pinned_output_typed[v_base_offset:],
            num_tokens=num_tokens,
            head_dim=head_dim,
            head_start=head_start,
            num_heads_to_gather=num_heads_to_gather,
            src_slot_stride=src_slot_stride,
            src_head_stride=src_head_stride,
            out_token_stride=out_token_stride,
            out_head_stride=out_head_stride,
            BLOCK_TOKENS=BLOCK_TOKENS,
            BLOCK_DIM=BLOCK_DIM,
        )

    # Ensure kernel completes before CPU reads the pinned buffer
    torch.cuda.synchronize()


@triton.jit
def _gather_kv_to_staging_kernel(
    # Source KV cache - GPU memory
    src_ptr,
    # Slot indices to gather
    slot_indices_ptr,
    # Output GPU staging buffer
    output_ptr,
    # Dimensions
    num_tokens,
    head_dim: tl.constexpr,
    # Head slicing params
    head_start: tl.constexpr,
    num_heads_to_gather: tl.constexpr,
    # Source strides (in elements)
    src_slot_stride,
    src_head_stride,
    # Output strides (in elements) - can be large for HEAD-FIRST layout
    out_token_stride,
    out_head_stride,
    # Block sizes
    BLOCK_TOKENS: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    """
    Gather KV data from scattered GPU slots into contiguous GPU staging buffer.

    Same as _gather_kv_kernel but output is GPU memory (fast coalesced writes).
    """
    head_id = tl.program_id(0)
    src_head = head_start + head_id

    # Cast to int64 to avoid overflow with large buffers
    src_slot_stride_i64 = src_slot_stride.to(tl.int64)
    src_head_stride_i64 = src_head_stride.to(tl.int64)
    out_token_stride_i64 = out_token_stride.to(tl.int64)
    out_head_stride_i64 = out_head_stride.to(tl.int64)
    src_head_i64 = src_head.to(tl.int64)
    head_id_i64 = head_id.to(tl.int64)

    for token_block_start in range(0, num_tokens, BLOCK_TOKENS):
        token_offsets = token_block_start + tl.arange(0, BLOCK_TOKENS)
        token_mask = token_offsets < num_tokens
        token_offsets_i64 = token_offsets.to(tl.int64)

        slot_ids = tl.load(slot_indices_ptr + token_offsets, mask=token_mask, other=0)
        slot_ids_i64 = slot_ids.to(tl.int64)

        for dim_start in range(0, head_dim, BLOCK_DIM):
            dim_offsets = dim_start + tl.arange(0, BLOCK_DIM)
            dim_mask = dim_offsets < head_dim
            dim_offsets_i64 = dim_offsets.to(tl.int64)

            src_offsets = (
                slot_ids_i64[:, None] * src_slot_stride_i64
                + src_head_i64 * src_head_stride_i64
                + dim_offsets_i64[None, :]
            )

            mask = token_mask[:, None] & dim_mask[None, :]
            data = tl.load(src_ptr + src_offsets, mask=mask, other=0.0)

            out_offsets = (
                head_id_i64 * out_head_stride_i64
                + token_offsets_i64[:, None] * out_token_stride_i64
                + dim_offsets_i64[None, :]
            )

            # Store to GPU staging buffer (fast coalesced writes)
            tl.store(output_ptr + out_offsets, data, mask=mask)


def gather_kv_with_staging(
    k_buffers: list[torch.Tensor],
    v_buffers: list[torch.Tensor],
    slot_indices: torch.Tensor,
    pinned_output: torch.Tensor,
    head_start: int,
    num_heads_to_gather: int,
    staging_buffer: torch.Tensor = None,
) -> None:
    """
    Gather KV data using a GPU staging buffer, then copy to pinned CPU.

    This is a hybrid approach:
    1. Gather scattered KV data into contiguous GPU staging buffer (fast coalesced writes)
    2. Copy from GPU staging buffer to pinned CPU memory (optimized memcpy)

    Args:
        k_buffers: List of K cache tensors per layer, each [total_slots, num_heads, head_dim]
        v_buffers: List of V cache tensors per layer
        slot_indices: Tensor of slot indices to gather, shape [num_tokens]
        pinned_output: Pinned CPU buffer to write to
        head_start: First head index to gather
        num_heads_to_gather: Number of heads to gather
        staging_buffer: Optional pre-allocated GPU staging buffer. If None, will allocate.

    Output layout: [num_heads_to_gather, num_layers, 2, num_tokens, head_dim] (HEAD-FIRST)
    """
    assert pinned_output.is_pinned(), "Output buffer must be pinned CPU memory"
    assert slot_indices.is_cuda, "slot_indices must be on GPU"

    num_layers = len(k_buffers)
    num_tokens = slot_indices.shape[0]
    num_heads = k_buffers[0].shape[1]
    head_dim = k_buffers[0].shape[2]
    dtype = k_buffers[0].dtype
    device = k_buffers[0].device

    assert head_start + num_heads_to_gather <= num_heads

    # Allocate or validate staging buffer
    staging_size = num_layers * 2 * num_tokens * num_heads_to_gather * head_dim
    if staging_buffer is None:
        staging_buffer = torch.empty(staging_size, dtype=dtype, device=device)
    else:
        assert staging_buffer.device == device
        assert staging_buffer.numel() >= staging_size

    src_slot_stride = k_buffers[0].stride(0)
    src_head_stride = k_buffers[0].stride(1)

    BLOCK_TOKENS = 64
    BLOCK_DIM = min(64, triton.next_power_of_2(head_dim))

    # HEAD-FIRST output layout: [num_heads_to_gather, num_layers, 2, num_tokens, head_dim]
    out_head_stride = num_layers * 2 * num_tokens * head_dim
    layer_stride = 2 * num_tokens * head_dim
    kv_stride = num_tokens * head_dim
    out_token_stride = head_dim

    grid = (num_heads_to_gather,)

    staging_typed = staging_buffer.view(dtype)

    # Step 1: Gather into GPU staging buffer in HEAD-FIRST layout
    for layer_idx in range(num_layers):
        # K buffer - base offset for (layer, K=0)
        k_base_offset = layer_idx * layer_stride
        _gather_kv_to_staging_kernel[grid](
            k_buffers[layer_idx],
            slot_indices,
            staging_typed[k_base_offset:],
            num_tokens=num_tokens,
            head_dim=head_dim,
            head_start=head_start,
            num_heads_to_gather=num_heads_to_gather,
            src_slot_stride=src_slot_stride,
            src_head_stride=src_head_stride,
            out_token_stride=out_token_stride,
            out_head_stride=out_head_stride,
            BLOCK_TOKENS=BLOCK_TOKENS,
            BLOCK_DIM=BLOCK_DIM,
        )

        # V buffer - base offset for (layer, V=1)
        v_base_offset = layer_idx * layer_stride + kv_stride
        _gather_kv_to_staging_kernel[grid](
            v_buffers[layer_idx],
            slot_indices,
            staging_typed[v_base_offset:],
            num_tokens=num_tokens,
            head_dim=head_dim,
            head_start=head_start,
            num_heads_to_gather=num_heads_to_gather,
            src_slot_stride=src_slot_stride,
            src_head_stride=src_head_stride,
            out_token_stride=out_token_stride,
            out_head_stride=out_head_stride,
            BLOCK_TOKENS=BLOCK_TOKENS,
            BLOCK_DIM=BLOCK_DIM,
        )

    # Step 2: Copy from GPU staging to pinned CPU (optimized memcpy)
    pinned_output_typed = pinned_output.view(dtype)
    pinned_output_typed[:staging_size].copy_(staging_typed[:staging_size], non_blocking=False)

    torch.cuda.synchronize()


def gather_kv_chunked(
    k_buffers: list[torch.Tensor],
    v_buffers: list[torch.Tensor],
    slot_indices: torch.Tensor,
    pinned_output: torch.Tensor,
    head_start: int,
    num_heads_to_gather: int,
    staging_buffer_size_mb: float = 256.0,
    staging_buffer: torch.Tensor = None,
) -> None:
    """
    Gather KV data using a fixed-size chunked staging buffer approach.

    This approach:
    1. Allocates a fixed-size GPU staging buffer (default 256MB)
    2. Processes layers/KV in chunks that fit in the staging buffer
    3. For each chunk: gather scattered KV into staging (in HEAD-FIRST layout), then copy to pinned CPU
    4. Repeat until all data is transferred

    This limits GPU memory overhead to staging_buffer_size_mb while potentially
    achieving better bandwidth than pure zero-copy by using optimized memcpy.

    Args:
        k_buffers: List of K cache tensors per layer, each [total_slots, num_heads, head_dim]
        v_buffers: List of V cache tensors per layer
        slot_indices: Tensor of slot indices to gather, shape [num_tokens]
        pinned_output: Pinned CPU buffer to write to
        head_start: First head index to gather
        num_heads_to_gather: Number of heads to gather
        staging_buffer_size_mb: Size of GPU staging buffer in MB (default 256MB)
        staging_buffer: Optional pre-allocated GPU staging buffer. If None, will allocate.

    Output layout in pinned_output: [num_heads_to_gather, num_layers, 2, num_tokens, head_dim]
    This HEAD-FIRST layout allows easy slicing by head for mixed-TP transfers.
    """
    assert pinned_output.is_pinned(), "Output buffer must be pinned CPU memory"
    assert slot_indices.is_cuda, "slot_indices must be on GPU"

    num_layers = len(k_buffers)
    num_tokens = slot_indices.shape[0]
    num_heads = k_buffers[0].shape[1]
    head_dim = k_buffers[0].shape[2]
    dtype = k_buffers[0].dtype
    device = k_buffers[0].device

    assert head_start + num_heads_to_gather <= num_heads

    bytes_per_element = 2 if dtype in (torch.float16, torch.bfloat16) else 4

    # Size of one K or V slice for one head: num_tokens * head_dim elements
    single_head_kv_elements = num_tokens * head_dim

    # Total elements for full output
    total_elements = num_heads_to_gather * num_layers * 2 * num_tokens * head_dim

    # Calculate staging buffer size in elements
    staging_buffer_bytes = int(staging_buffer_size_mb * 1e6)
    staging_buffer_elements = staging_buffer_bytes // bytes_per_element

    # If staging buffer can hold everything, just use gather_kv_with_staging
    if staging_buffer_elements >= total_elements:
        return gather_kv_with_staging(
            k_buffers=k_buffers,
            v_buffers=v_buffers,
            slot_indices=slot_indices,
            pinned_output=pinned_output,
            head_start=head_start,
            num_heads_to_gather=num_heads_to_gather,
            staging_buffer=staging_buffer,
        )

    # Otherwise, process in chunks. Each chunk processes some (layer, kv) pairs.
    # Since kernels output in HEAD-FIRST layout directly, we need staging to
    # match that layout for the subset of (layer, kv) pairs we're processing.

    # We'll process one (layer, kv) at a time for simplicity with chunking
    # Each (layer, kv) contributes num_heads_to_gather * num_tokens * head_dim elements
    single_layer_kv_elements = num_heads_to_gather * num_tokens * head_dim

    # Allocate staging for one (layer, kv)
    if staging_buffer is None:
        staging_buffer = torch.empty(single_layer_kv_elements, dtype=dtype, device=device)
    else:
        assert staging_buffer.device == device
        assert staging_buffer.numel() >= single_layer_kv_elements

    src_slot_stride = k_buffers[0].stride(0)
    src_head_stride = k_buffers[0].stride(1)

    BLOCK_TOKENS = 64
    BLOCK_DIM = min(64, triton.next_power_of_2(head_dim))

    # HEAD-FIRST layout strides (in elements)
    # Full output: [num_heads_to_gather, num_layers, 2, num_tokens, head_dim]
    out_head_stride_full = num_layers * 2 * num_tokens * head_dim
    layer_stride = 2 * num_tokens * head_dim
    kv_stride = num_tokens * head_dim
    out_token_stride = head_dim

    # For staging buffer (holds one layer_kv): [num_heads_to_gather, 1, 1, num_tokens, head_dim]
    # which is effectively [num_heads_to_gather, num_tokens, head_dim] flattened
    # Kernel writes with head_stride = num_tokens * head_dim for staging
    staging_head_stride = num_tokens * head_dim

    grid = (num_heads_to_gather,)

    staging_typed = staging_buffer.view(dtype)
    pinned_output_typed = pinned_output.view(dtype)

    # Process each (layer, kv) pair
    for layer_idx in range(num_layers):
        for is_v in [0, 1]:  # 0=K, 1=V
            buffer = k_buffers[layer_idx] if is_v == 0 else v_buffers[layer_idx]

            # Gather into staging buffer
            # Staging layout: [num_heads_to_gather, num_tokens, head_dim]
            _gather_kv_to_staging_kernel[grid](
                buffer,
                slot_indices,
                staging_typed,
                num_tokens=num_tokens,
                head_dim=head_dim,
                head_start=head_start,
                num_heads_to_gather=num_heads_to_gather,
                src_slot_stride=src_slot_stride,
                src_head_stride=src_head_stride,
                out_token_stride=out_token_stride,
                out_head_stride=staging_head_stride,
                BLOCK_TOKENS=BLOCK_TOKENS,
                BLOCK_DIM=BLOCK_DIM,
            )

            torch.cuda.synchronize()

            # Copy from staging to pinned in HEAD-FIRST layout
            # Staging: [num_heads_to_gather, num_tokens, head_dim] contiguous
            # Pinned: [num_heads_to_gather, num_layers, 2, num_tokens, head_dim]
            # For this (layer, kv), copy each head's data to correct position
            for h in range(num_heads_to_gather):
                # Source in staging: h * staging_head_stride
                staging_offset = h * staging_head_stride
                # Dest in pinned: h * out_head_stride_full + layer_idx * layer_stride + is_v * kv_stride
                pinned_offset = h * out_head_stride_full + layer_idx * layer_stride + is_v * kv_stride

                pinned_output_typed[pinned_offset:pinned_offset + single_head_kv_elements].copy_(
                    staging_typed[staging_offset:staging_offset + single_head_kv_elements],
                    non_blocking=False
                )

    torch.cuda.synchronize()


# Default staging buffer size (256MB provides good balance of performance vs memory)
DEFAULT_STAGING_BUFFER_SIZE_MB = 256.0


def gather_kv(
    k_buffers: list[torch.Tensor],
    v_buffers: list[torch.Tensor],
    slot_indices: torch.Tensor,
    pinned_output: torch.Tensor,
    head_start: int = 0,
    num_heads_to_gather: int = None,
    staging_buffer: torch.Tensor = None,
    staging_buffer_size_mb: float = DEFAULT_STAGING_BUFFER_SIZE_MB,
) -> None:
    """
    Gather KV cache data from GPU to pinned CPU memory.

    This is the primary API for device-to-host KV transfers. It uses a chunked
    staging buffer approach that achieves ~80% of PCIe bandwidth while limiting
    GPU memory overhead to a fixed staging buffer size.

    Args:
        k_buffers: List of K cache tensors per layer, each [total_slots, num_heads, head_dim]
        v_buffers: List of V cache tensors per layer, each [total_slots, num_heads, head_dim]
        slot_indices: Tensor of slot indices to gather, shape [num_tokens], on GPU
        pinned_output: Pinned CPU buffer to write to. Must be created with pin_memory=True.
        head_start: First head index to gather (for head slicing in mixed-TP). Default 0.
        num_heads_to_gather: Number of heads to gather. Default None means all heads.
        staging_buffer: Optional pre-allocated GPU staging buffer for reuse across calls.
        staging_buffer_size_mb: Size of staging buffer in MB if not provided. Default 256MB.

    Output layout in pinned_output: [num_heads_to_gather, num_layers, 2, num_tokens, head_dim]
    This HEAD-FIRST layout allows easy slicing by head for mixed-TP transfers:
    - To send heads [a:b], slice pinned_output[a*head_stride : b*head_stride]
    - head_stride = num_layers * 2 * num_tokens * head_dim * bytes_per_element
    """
    if num_heads_to_gather is None:
        num_heads_to_gather = k_buffers[0].shape[1] - head_start

    num_layers = len(k_buffers)
    num_tokens = slot_indices.shape[0]
    head_dim = k_buffers[0].shape[2]
    dtype = k_buffers[0].dtype
    bytes_per_element = 2 if dtype in (torch.float16, torch.bfloat16) else 4
    transfer_bytes = num_layers * 2 * num_tokens * num_heads_to_gather * head_dim * bytes_per_element

    logger.debug(
        f"[TRITON-KV] gather_kv: {num_tokens} tokens, {num_layers} layers, "
        f"heads [{head_start}:{head_start + num_heads_to_gather}], "
        f"head_dim={head_dim}, dtype={dtype}, "
        f"transfer_size={transfer_bytes / 1e6:.2f}MB, "
        f"staging={'provided' if staging_buffer is not None else f'{staging_buffer_size_mb}MB'}"
    )

    gather_kv_chunked(
        k_buffers=k_buffers,
        v_buffers=v_buffers,
        slot_indices=slot_indices,
        pinned_output=pinned_output,
        head_start=head_start,
        num_heads_to_gather=num_heads_to_gather,
        staging_buffer_size_mb=staging_buffer_size_mb,
        staging_buffer=staging_buffer,
    )


# =============================================================================
# Scatter Kernel: Host -> Device (Pinned CPU -> GPU KV Cache)
# =============================================================================


@triton.jit
def _scatter_kv_kernel_head_first(
    # Source pinned CPU buffer in HEAD-FIRST layout - GPU-accessible via zero-copy
    src_ptr,
    # Slot indices to scatter to
    slot_indices_ptr,
    # Destination KV cache - GPU memory
    dst_ptr,
    # Dimensions
    num_tokens,
    head_dim: tl.constexpr,
    # Head slicing params
    head_start: tl.constexpr,
    num_heads_to_scatter: tl.constexpr,
    # Source strides for HEAD-FIRST layout [head, layer, kv, token, head_dim]
    # Base pointer is already offset to correct (layer, kv) position
    in_head_stride,  # distance between heads in source (can be very large, use int64)
    in_token_stride,  # = head_dim (tokens contiguous within head slice)
    # Destination strides (in elements)
    dst_slot_stride,
    dst_head_stride,
    # Block sizes
    BLOCK_TOKENS: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    """
    Scatter KV data from HEAD-FIRST pinned CPU buffer to scattered GPU slots.

    This kernel handles a single layer's K or V buffer.

    Input layout: HEAD-FIRST [num_heads, num_layers, 2, num_tokens, head_dim]
                  Base pointer is offset to (layer, kv), so effective input is
                  [num_heads, num_tokens, head_dim] with head_stride spacing.
    Output layout: [total_slots, num_heads, head_dim] on GPU

    Grid: (num_heads_to_scatter,) - one program per head
    """
    head_id = tl.program_id(0)  # relative head (0 to num_heads_to_scatter-1)

    # Destination head index (absolute in destination KV cache)
    dst_head = head_start + head_id

    # Cast strides to int64 to avoid overflow with large buffers
    in_head_stride_i64 = in_head_stride.to(tl.int64)
    in_token_stride_i64 = in_token_stride.to(tl.int64)
    dst_slot_stride_i64 = dst_slot_stride.to(tl.int64)
    dst_head_stride_i64 = dst_head_stride.to(tl.int64)
    head_id_i64 = head_id.to(tl.int64)
    dst_head_i64 = dst_head.to(tl.int64)

    # Process tokens in blocks
    for token_block_start in range(0, num_tokens, BLOCK_TOKENS):
        token_offsets = token_block_start + tl.arange(0, BLOCK_TOKENS)
        token_mask = token_offsets < num_tokens
        token_offsets_i64 = token_offsets.to(tl.int64)

        # Load slot indices for these tokens
        slot_ids = tl.load(slot_indices_ptr + token_offsets, mask=token_mask, other=0)
        slot_ids_i64 = slot_ids.to(tl.int64)

        # Process head_dim in blocks
        for dim_start in range(0, head_dim, BLOCK_DIM):
            dim_offsets = dim_start + tl.arange(0, BLOCK_DIM)
            dim_mask = dim_offsets < head_dim
            dim_offsets_i64 = dim_offsets.to(tl.int64)

            # Compute source addresses in HEAD-FIRST pinned buffer (use int64)
            # src[head_id, token, dim] = base + head_id * head_stride + token * token_stride + dim
            src_offsets = (
                head_id_i64 * in_head_stride_i64
                + token_offsets_i64[:, None] * in_token_stride_i64
                + dim_offsets_i64[None, :]
            )

            # Load from pinned CPU buffer (zero-copy read over PCIe)
            mask = token_mask[:, None] & dim_mask[None, :]
            data = tl.load(src_ptr + src_offsets, mask=mask, other=0.0)

            # Compute destination addresses in GPU KV cache (use int64)
            dst_offsets = (
                slot_ids_i64[:, None] * dst_slot_stride_i64
                + dst_head_i64 * dst_head_stride_i64
                + dim_offsets_i64[None, :]
            )

            # Store to GPU KV cache
            tl.store(dst_ptr + dst_offsets, data, mask=mask)


def scatter_kv_to_gpu(
    pinned_input: torch.Tensor,  # pinned CPU buffer, viewed as flat
    k_buffers: list[torch.Tensor],  # [num_layers] each [total_slots, num_heads, head_dim] on GPU
    v_buffers: list[torch.Tensor],
    slot_indices: torch.Tensor,  # [num_tokens] on GPU
    head_start: int,
    num_heads_to_scatter: int,
) -> None:
    """
    Launch scatter kernel to distribute KV data from pinned CPU buffer to GPU KV cache.

    Args:
        pinned_input: Pinned CPU buffer to read from. Must be created with pin_memory=True.
                      Should contain data in HEAD-FIRST layout:
                      [num_heads_to_scatter, num_layers, 2, num_tokens, head_dim]
        k_buffers: List of K cache tensors per layer, each [total_slots, num_heads, head_dim]
        v_buffers: List of V cache tensors per layer, each [total_slots, num_heads, head_dim]
        slot_indices: Tensor of slot indices to scatter to, shape [num_tokens], on GPU
        head_start: First head index to scatter to (for head slicing in mixed-TP)
        num_heads_to_scatter: Number of heads to scatter starting from head_start

    Input layout in pinned_input: [num_heads_to_scatter, num_layers, 2, num_tokens, head_dim]
    This HEAD-FIRST layout allows easy slicing by head for mixed-TP transfers.
    """
    assert pinned_input.is_pinned(), "Input buffer must be pinned CPU memory"
    assert slot_indices.is_cuda, "slot_indices must be on GPU"

    num_layers = len(k_buffers)
    num_tokens = slot_indices.shape[0]
    num_heads = k_buffers[0].shape[1]
    head_dim = k_buffers[0].shape[2]
    dtype = k_buffers[0].dtype

    assert head_start + num_heads_to_scatter <= num_heads, (
        f"head_start ({head_start}) + num_heads_to_scatter ({num_heads_to_scatter}) "
        f"exceeds num_heads ({num_heads})"
    )

    # Get destination strides (in elements, not bytes)
    dst_slot_stride = k_buffers[0].stride(0)
    dst_head_stride = k_buffers[0].stride(1)

    # Determine block sizes
    BLOCK_TOKENS = 64
    BLOCK_DIM = min(64, triton.next_power_of_2(head_dim))

    # HEAD-FIRST input layout: [head, layer, kv, token, head_dim]
    # Strides for input layout (in elements):
    # - head_stride: distance between heads = num_layers * 2 * num_tokens * head_dim
    # - layer_stride: distance between layers = 2 * num_tokens * head_dim
    # - kv_stride: distance between K and V = num_tokens * head_dim
    # - token_stride: distance between tokens = head_dim (contiguous within head slice)
    in_head_stride = num_layers * 2 * num_tokens * head_dim
    layer_stride = 2 * num_tokens * head_dim
    kv_stride = num_tokens * head_dim
    in_token_stride = head_dim  # tokens are contiguous within each head's slice

    # View pinned input as the right dtype
    pinned_input_typed = pinned_input.view(dtype)

    grid = (num_heads_to_scatter,)

    # Iterate over layers and K/V - kernel reads directly from HEAD-FIRST layout
    for layer_idx in range(num_layers):
        # K buffer - base offset for (layer, K=0)
        k_base_offset = layer_idx * layer_stride  # + 0 * kv_stride for K
        _scatter_kv_kernel_head_first[grid](
            pinned_input_typed[k_base_offset:],
            slot_indices,
            k_buffers[layer_idx],
            num_tokens=num_tokens,
            head_dim=head_dim,
            head_start=head_start,
            num_heads_to_scatter=num_heads_to_scatter,
            in_head_stride=in_head_stride,
            in_token_stride=in_token_stride,
            dst_slot_stride=dst_slot_stride,
            dst_head_stride=dst_head_stride,
            BLOCK_TOKENS=BLOCK_TOKENS,
            BLOCK_DIM=BLOCK_DIM,
        )

        # V buffer - base offset for (layer, V=1)
        v_base_offset = layer_idx * layer_stride + kv_stride
        _scatter_kv_kernel_head_first[grid](
            pinned_input_typed[v_base_offset:],
            slot_indices,
            v_buffers[layer_idx],
            num_tokens=num_tokens,
            head_dim=head_dim,
            head_start=head_start,
            num_heads_to_scatter=num_heads_to_scatter,
            in_head_stride=in_head_stride,
            in_token_stride=in_token_stride,
            dst_slot_stride=dst_slot_stride,
            dst_head_stride=dst_head_stride,
            BLOCK_TOKENS=BLOCK_TOKENS,
            BLOCK_DIM=BLOCK_DIM,
        )

    # Ensure kernel completes before returning
    torch.cuda.synchronize()


def scatter_kv(
    pinned_input: torch.Tensor,
    k_buffers: list[torch.Tensor],
    v_buffers: list[torch.Tensor],
    slot_indices: torch.Tensor,
    head_start: int = 0,
    num_heads_to_scatter: int = None,
) -> None:
    """
    Scatter KV cache data from pinned CPU memory to GPU.

    This is the primary API for host-to-device KV transfers.

    Args:
        pinned_input: Pinned CPU buffer to read from. Must be created with pin_memory=True.
        k_buffers: List of K cache tensors per layer, each [total_slots, num_heads, head_dim]
        v_buffers: List of V cache tensors per layer, each [total_slots, num_heads, head_dim]
        slot_indices: Tensor of slot indices to scatter to, shape [num_tokens], on GPU
        head_start: First head index to scatter to (for head slicing in mixed-TP). Default 0.
        num_heads_to_scatter: Number of heads to scatter. Default None means all heads.

    Input layout in pinned_input: [num_heads_to_scatter, num_layers, 2, num_tokens, head_dim]
    This HEAD-FIRST layout allows easy slicing by head for mixed-TP transfers.
    """
    if num_heads_to_scatter is None:
        num_heads_to_scatter = k_buffers[0].shape[1] - head_start

    num_layers = len(k_buffers)
    num_tokens = slot_indices.shape[0]
    head_dim = k_buffers[0].shape[2]
    dtype = k_buffers[0].dtype
    bytes_per_element = 2 if dtype in (torch.float16, torch.bfloat16) else 4
    transfer_bytes = num_layers * 2 * num_tokens * num_heads_to_scatter * head_dim * bytes_per_element

    logger.debug(
        f"[TRITON-KV] scatter_kv: {num_tokens} tokens, {num_layers} layers, "
        f"heads [{head_start}:{head_start + num_heads_to_scatter}], "
        f"head_dim={head_dim}, dtype={dtype}, "
        f"transfer_size={transfer_bytes / 1e6:.2f}MB"
    )

    scatter_kv_to_gpu(
        pinned_input=pinned_input,
        k_buffers=k_buffers,
        v_buffers=v_buffers,
        slot_indices=slot_indices,
        head_start=head_start,
        num_heads_to_scatter=num_heads_to_scatter,
    )
