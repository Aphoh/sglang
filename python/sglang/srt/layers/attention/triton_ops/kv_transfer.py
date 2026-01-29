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
- scatter_kv(): Scatter KV from pinned CPU to GPU (TODO)

The chunked approach uses a fixed-size GPU staging buffer (default 256MB) to achieve
~80% of PCIe bandwidth while limiting GPU memory overhead. This outperforms pure
zero-copy writes which achieve only ~50% of PCIe bandwidth due to uncoalesced writes.
"""

import torch
import triton
import triton.language as tl


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
    # Output strides (in elements) - output is [num_tokens, num_heads_to_gather, head_dim]
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
    Output layout: [num_tokens, num_heads_to_gather, head_dim] (contiguous)

    Grid: (num_heads_to_gather,) - one program per head
    """
    head_id = tl.program_id(0)  # relative head (0 to num_heads_to_gather-1)

    # Source head index (absolute in source KV cache)
    src_head = head_start + head_id

    # Process tokens in blocks
    for token_block_start in range(0, num_tokens, BLOCK_TOKENS):
        token_offsets = token_block_start + tl.arange(0, BLOCK_TOKENS)
        token_mask = token_offsets < num_tokens

        # Load slot indices for these tokens
        slot_ids = tl.load(slot_indices_ptr + token_offsets, mask=token_mask, other=0)

        # Process head_dim in blocks
        for dim_start in range(0, head_dim, BLOCK_DIM):
            dim_offsets = dim_start + tl.arange(0, BLOCK_DIM)
            dim_mask = dim_offsets < head_dim

            # Compute source addresses in GPU KV cache
            # src[slot_id, src_head, dim] = base + slot_id * slot_stride + src_head * head_stride + dim
            src_offsets = (
                slot_ids[:, None] * src_slot_stride
                + src_head * src_head_stride
                + dim_offsets[None, :]
            )

            # Load from GPU source
            mask = token_mask[:, None] & dim_mask[None, :]
            data = tl.load(src_ptr + src_offsets, mask=mask, other=0.0)

            # Compute output addresses
            # out[token, head_id, dim] = base + token * token_stride + head_id * head_stride + dim
            out_offsets = (
                token_offsets[:, None] * out_token_stride
                + head_id * out_head_stride
                + dim_offsets[None, :]
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

    Output layout in pinned_output: [num_layers, 2, num_tokens, num_heads_to_gather, head_dim]
    where dimension 1 is K=0, V=1
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

    # Output strides for [num_tokens, num_heads_to_gather, head_dim] layout
    out_token_stride = num_heads_to_gather * head_dim
    out_head_stride = head_dim

    # Size of one K or V buffer in output: num_tokens * num_heads_to_gather * head_dim elements
    kv_output_size = num_tokens * num_heads_to_gather * head_dim

    grid = (num_heads_to_gather,)

    # View pinned output as the right dtype
    pinned_output_typed = pinned_output.view(dtype)

    # Iterate over layers and K/V
    for layer_idx in range(num_layers):
        # Output offset for this layer
        # Layout: [num_layers, 2, num_tokens, num_heads_to_gather, head_dim]
        layer_output_offset = layer_idx * 2 * kv_output_size

        # K buffer for this layer
        k_output_offset = layer_output_offset
        _gather_kv_kernel[grid](
            k_buffers[layer_idx],
            slot_indices,
            pinned_output_typed[k_output_offset:],
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

        # V buffer for this layer
        v_output_offset = layer_output_offset + kv_output_size
        _gather_kv_kernel[grid](
            v_buffers[layer_idx],
            slot_indices,
            pinned_output_typed[v_output_offset:],
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
    # Output strides (in elements) - output is [num_tokens, num_heads_to_gather, head_dim]
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

    for token_block_start in range(0, num_tokens, BLOCK_TOKENS):
        token_offsets = token_block_start + tl.arange(0, BLOCK_TOKENS)
        token_mask = token_offsets < num_tokens

        slot_ids = tl.load(slot_indices_ptr + token_offsets, mask=token_mask, other=0)

        for dim_start in range(0, head_dim, BLOCK_DIM):
            dim_offsets = dim_start + tl.arange(0, BLOCK_DIM)
            dim_mask = dim_offsets < head_dim

            src_offsets = (
                slot_ids[:, None] * src_slot_stride
                + src_head * src_head_stride
                + dim_offsets[None, :]
            )

            mask = token_mask[:, None] & dim_mask[None, :]
            data = tl.load(src_ptr + src_offsets, mask=mask, other=0.0)

            out_offsets = (
                token_offsets[:, None] * out_token_stride
                + head_id * out_head_stride
                + dim_offsets[None, :]
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

    out_token_stride = num_heads_to_gather * head_dim
    out_head_stride = head_dim
    kv_output_size = num_tokens * num_heads_to_gather * head_dim

    grid = (num_heads_to_gather,)

    staging_typed = staging_buffer.view(dtype)

    # Step 1: Gather into GPU staging buffer (fast)
    for layer_idx in range(num_layers):
        layer_output_offset = layer_idx * 2 * kv_output_size

        # K buffer
        k_output_offset = layer_output_offset
        _gather_kv_to_staging_kernel[grid](
            k_buffers[layer_idx],
            slot_indices,
            staging_typed[k_output_offset:],
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

        # V buffer
        v_output_offset = layer_output_offset + kv_output_size
        _gather_kv_to_staging_kernel[grid](
            v_buffers[layer_idx],
            slot_indices,
            staging_typed[v_output_offset:],
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
    3. For each chunk: gather scattered KV into staging, then copy to pinned CPU
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

    # Size of one K or V buffer: num_tokens * num_heads_to_gather * head_dim elements
    single_kv_elements = num_tokens * num_heads_to_gather * head_dim
    single_kv_bytes = single_kv_elements * bytes_per_element

    # Calculate staging buffer size in elements
    staging_buffer_bytes = int(staging_buffer_size_mb * 1e6)
    staging_buffer_elements = staging_buffer_bytes // bytes_per_element

    # How many K or V buffers fit in the staging buffer?
    # We want to gather at least one full K or V buffer at a time for simplicity
    kvs_per_chunk = max(1, staging_buffer_elements // single_kv_elements)

    # Allocate staging buffer (sized to hold kvs_per_chunk worth of data)
    actual_staging_elements = kvs_per_chunk * single_kv_elements
    if staging_buffer is None:
        staging_buffer = torch.empty(actual_staging_elements, dtype=dtype, device=device)
    else:
        assert staging_buffer.device == device
        assert staging_buffer.numel() >= actual_staging_elements

    src_slot_stride = k_buffers[0].stride(0)
    src_head_stride = k_buffers[0].stride(1)

    BLOCK_TOKENS = 64
    BLOCK_DIM = min(64, triton.next_power_of_2(head_dim))

    out_token_stride = num_heads_to_gather * head_dim
    out_head_stride = head_dim

    grid = (num_heads_to_gather,)

    staging_typed = staging_buffer.view(dtype)
    pinned_output_typed = pinned_output.view(dtype)

    # Build list of (layer_idx, is_v, buffer) tuples for all K/V to transfer
    # is_v: 0 for K, 1 for V
    kv_list = []
    for layer_idx in range(num_layers):
        kv_list.append((layer_idx, 0, k_buffers[layer_idx]))  # K
        kv_list.append((layer_idx, 1, v_buffers[layer_idx]))  # V

    # Process in chunks
    total_kvs = len(kv_list)
    kv_idx = 0

    while kv_idx < total_kvs:
        # Determine how many KVs to process in this chunk
        chunk_size = min(kvs_per_chunk, total_kvs - kv_idx)

        # Gather this chunk into staging buffer
        for i in range(chunk_size):
            layer_idx, is_v, buffer = kv_list[kv_idx + i]
            staging_offset = i * single_kv_elements

            _gather_kv_to_staging_kernel[grid](
                buffer,
                slot_indices,
                staging_typed[staging_offset:],
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

        # Synchronize to ensure gather is complete
        torch.cuda.synchronize()

        # Copy this chunk from staging to pinned CPU
        chunk_elements = chunk_size * single_kv_elements

        # Calculate output offset: each KV is at (layer_idx * 2 + is_v) * single_kv_elements
        for i in range(chunk_size):
            layer_idx, is_v, _ = kv_list[kv_idx + i]
            output_offset = (layer_idx * 2 + is_v) * single_kv_elements
            staging_offset = i * single_kv_elements

            pinned_output_typed[output_offset:output_offset + single_kv_elements].copy_(
                staging_typed[staging_offset:staging_offset + single_kv_elements],
                non_blocking=False
            )

        kv_idx += chunk_size

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

    Output layout in pinned_output: [num_layers, 2, num_tokens, num_heads_to_gather, head_dim]
    where dimension 1 is K=0, V=1
    """
    if num_heads_to_gather is None:
        num_heads_to_gather = k_buffers[0].shape[1] - head_start

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
