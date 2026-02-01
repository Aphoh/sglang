"""
Tests for KV transfer Triton kernels.

Tests the gather and scatter kernels for KV cache transfers:
- gather_kv_to_pinned_all_layers: GPU -> pinned CPU (device to host)
- scatter_kv_with_staging_all_layers: pinned CPU -> GPU (host to device)
"""

import sys
from pathlib import Path

import pytest
import torch

# Add the triton_ops path since it's not in sgl_kernel
triton_ops_path = Path(__file__).parent.parent.parent / "python" / "sglang" / "srt" / "layers" / "attention" / "triton_ops"
sys.path.insert(0, str(triton_ops_path))

from kv_transfer import (
    gather_kv_to_pinned_all_layers,
    scatter_kv_with_staging_all_layers,
)


def reference_gather_kv(
    k_buffers: list[torch.Tensor],
    v_buffers: list[torch.Tensor],
    slot_indices: torch.Tensor,
    head_start: int,
    num_heads_to_gather: int,
) -> torch.Tensor:
    """
    Reference implementation of KV gather using PyTorch operations.

    Returns tensor of shape [num_heads_to_gather, num_layers, 2, num_tokens, head_dim]
    This is the HEAD-FIRST layout for easy head slicing in mixed-TP transfers.
    """
    num_layers = len(k_buffers)
    num_tokens = slot_indices.shape[0]
    head_dim = k_buffers[0].shape[2]
    dtype = k_buffers[0].dtype

    output = torch.zeros(
        (num_heads_to_gather, num_layers, 2, num_tokens, head_dim),
        dtype=dtype,
        device=k_buffers[0].device,
    )

    head_end = head_start + num_heads_to_gather

    for layer_idx in range(num_layers):
        k_data = k_buffers[layer_idx][slot_indices, head_start:head_end, :]
        v_data = v_buffers[layer_idx][slot_indices, head_start:head_end, :]

        for h in range(num_heads_to_gather):
            output[h, layer_idx, 0] = k_data[:, h, :]
            output[h, layer_idx, 1] = v_data[:, h, :]

    return output


def reference_scatter_kv(
    pinned_input: torch.Tensor,
    slot_indices: torch.Tensor,
    num_layers: int,
    num_heads_to_scatter: int,
    head_dim: int,
    total_slots: int,
    num_heads: int,
    head_start: int,
    dtype: torch.dtype,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Reference implementation of KV scatter using PyTorch operations.

    Input is in HEAD-FIRST layout: [num_heads_to_scatter, num_layers, 2, num_tokens, head_dim]
    Returns (k_buffers, v_buffers) with data scattered to the specified slots.
    """
    num_tokens = slot_indices.shape[0]

    k_buffers = [
        torch.zeros(total_slots, num_heads, head_dim, dtype=dtype, device="cuda")
        for _ in range(num_layers)
    ]
    v_buffers = [
        torch.zeros(total_slots, num_heads, head_dim, dtype=dtype, device="cuda")
        for _ in range(num_layers)
    ]

    input_shaped = pinned_input.view(num_heads_to_scatter, num_layers, 2, num_tokens, head_dim)

    for layer_idx in range(num_layers):
        for h in range(num_heads_to_scatter):
            k_buffers[layer_idx][slot_indices, head_start + h, :] = input_shaped[h, layer_idx, 0].cuda()
            v_buffers[layer_idx][slot_indices, head_start + h, :] = input_shaped[h, layer_idx, 1].cuda()

    return k_buffers, v_buffers


def create_pointer_tensors(k_buffers, v_buffers):
    """Helper to create pointer tensors and get strides."""
    k_data_ptrs = torch.tensor(
        [x.data_ptr() for x in k_buffers], dtype=torch.uint64, device="cuda"
    )
    v_data_ptrs = torch.tensor(
        [x.data_ptr() for x in v_buffers], dtype=torch.uint64, device="cuda"
    )
    slot_stride = k_buffers[0].stride(0)
    head_stride = k_buffers[0].stride(1)
    return k_data_ptrs, v_data_ptrs, slot_stride, head_stride


# =============================================================================
# Gather Tests (Device -> Host)
# =============================================================================


@pytest.mark.parametrize("num_layers", [1, 4, 32])
@pytest.mark.parametrize("num_tokens", [1, 64, 512])
@pytest.mark.parametrize("num_heads", [8, 32])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gather_kv_full_heads(num_layers, num_tokens, num_heads, head_dim, dtype):
    """Test gathering all heads (no slicing)."""
    total_slots = 1024

    k_buffers = [
        torch.randn(total_slots, num_heads, head_dim, dtype=dtype, device="cuda")
        for _ in range(num_layers)
    ]
    v_buffers = [
        torch.randn(total_slots, num_heads, head_dim, dtype=dtype, device="cuda")
        for _ in range(num_layers)
    ]

    slot_indices = torch.randperm(total_slots, device="cuda")[:num_tokens].to(torch.int32)

    k_data_ptrs, v_data_ptrs, src_slot_stride, src_head_stride = create_pointer_tensors(k_buffers, v_buffers)

    output_size = num_layers * 2 * num_tokens * num_heads * head_dim
    pinned_output = torch.empty(output_size, dtype=dtype, device="cpu", pin_memory=True)

    gather_kv_to_pinned_all_layers(
        k_data_ptrs=k_data_ptrs,
        v_data_ptrs=v_data_ptrs,
        slot_indices=slot_indices,
        pinned_output=pinned_output,
        head_start=0,
        num_heads_to_gather=num_heads,
        num_layers=num_layers,
        head_dim=head_dim,
        src_slot_stride=src_slot_stride,
        src_head_stride=src_head_stride,
    )

    expected = reference_gather_kv(
        k_buffers, v_buffers, slot_indices.long(),
        head_start=0, num_heads_to_gather=num_heads
    )

    actual = pinned_output.view(num_heads, num_layers, 2, num_tokens, head_dim)
    torch.testing.assert_close(actual, expected.cpu(), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("num_heads", [8, 32])
@pytest.mark.parametrize("head_start,num_heads_to_gather", [(0, 4), (4, 4), (0, 8), (8, 8)])
def test_gather_kv_head_slicing(num_heads, head_start, num_heads_to_gather):
    """Test gathering a subset of heads (for mixed-TP)."""
    if head_start + num_heads_to_gather > num_heads:
        pytest.skip("head slice exceeds num_heads")

    num_layers = 4
    num_tokens = 128
    head_dim = 128
    total_slots = 512
    dtype = torch.float16

    k_buffers = [
        torch.randn(total_slots, num_heads, head_dim, dtype=dtype, device="cuda")
        for _ in range(num_layers)
    ]
    v_buffers = [
        torch.randn(total_slots, num_heads, head_dim, dtype=dtype, device="cuda")
        for _ in range(num_layers)
    ]

    slot_indices = torch.randperm(total_slots, device="cuda")[:num_tokens].to(torch.int32)

    k_data_ptrs, v_data_ptrs, src_slot_stride, src_head_stride = create_pointer_tensors(k_buffers, v_buffers)

    output_size = num_layers * 2 * num_tokens * num_heads_to_gather * head_dim
    pinned_output = torch.empty(output_size, dtype=dtype, device="cpu", pin_memory=True)

    gather_kv_to_pinned_all_layers(
        k_data_ptrs=k_data_ptrs,
        v_data_ptrs=v_data_ptrs,
        slot_indices=slot_indices,
        pinned_output=pinned_output,
        head_start=head_start,
        num_heads_to_gather=num_heads_to_gather,
        num_layers=num_layers,
        head_dim=head_dim,
        src_slot_stride=src_slot_stride,
        src_head_stride=src_head_stride,
    )

    expected = reference_gather_kv(
        k_buffers, v_buffers, slot_indices.long(),
        head_start=head_start, num_heads_to_gather=num_heads_to_gather
    )

    actual = pinned_output.view(num_heads_to_gather, num_layers, 2, num_tokens, head_dim)
    torch.testing.assert_close(actual, expected.cpu(), rtol=1e-3, atol=1e-3)


def test_gather_kv_contiguous_indices():
    """Test with contiguous slot indices (best case for memory access)."""
    num_layers = 4
    num_tokens = 256
    num_heads = 8
    head_dim = 128
    total_slots = 1024
    dtype = torch.float16

    k_buffers = [
        torch.randn(total_slots, num_heads, head_dim, dtype=dtype, device="cuda")
        for _ in range(num_layers)
    ]
    v_buffers = [
        torch.randn(total_slots, num_heads, head_dim, dtype=dtype, device="cuda")
        for _ in range(num_layers)
    ]

    start_slot = 100
    slot_indices = torch.arange(
        start_slot, start_slot + num_tokens, device="cuda", dtype=torch.int32
    )

    k_data_ptrs, v_data_ptrs, src_slot_stride, src_head_stride = create_pointer_tensors(k_buffers, v_buffers)

    output_size = num_layers * 2 * num_tokens * num_heads * head_dim
    pinned_output = torch.empty(output_size, dtype=dtype, device="cpu", pin_memory=True)

    gather_kv_to_pinned_all_layers(
        k_data_ptrs=k_data_ptrs,
        v_data_ptrs=v_data_ptrs,
        slot_indices=slot_indices,
        pinned_output=pinned_output,
        head_start=0,
        num_heads_to_gather=num_heads,
        num_layers=num_layers,
        head_dim=head_dim,
        src_slot_stride=src_slot_stride,
        src_head_stride=src_head_stride,
    )

    expected = reference_gather_kv(
        k_buffers, v_buffers, slot_indices.long(),
        head_start=0, num_heads_to_gather=num_heads
    )

    actual = pinned_output.view(num_heads, num_layers, 2, num_tokens, head_dim)
    torch.testing.assert_close(actual, expected.cpu(), rtol=1e-3, atol=1e-3)


def test_gather_kv_non_pinned_raises():
    """Test that non-pinned output raises an error."""
    k_buffers = [torch.randn(64, 4, 32, dtype=torch.float16, device="cuda")]
    v_buffers = [torch.randn(64, 4, 32, dtype=torch.float16, device="cuda")]
    slot_indices = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device="cuda")

    k_data_ptrs, v_data_ptrs, src_slot_stride, src_head_stride = create_pointer_tensors(k_buffers, v_buffers)

    non_pinned_output = torch.empty(1 * 2 * 4 * 4 * 32, dtype=torch.float16, device="cpu")

    with pytest.raises(AssertionError, match="pinned"):
        gather_kv_to_pinned_all_layers(
            k_data_ptrs=k_data_ptrs,
            v_data_ptrs=v_data_ptrs,
            slot_indices=slot_indices,
            pinned_output=non_pinned_output,
            head_start=0,
            num_heads_to_gather=4,
            num_layers=1,
            head_dim=32,
            src_slot_stride=src_slot_stride,
            src_head_stride=src_head_stride,
        )


@pytest.mark.parametrize("total_slots", [100_000, 500_000])
@pytest.mark.parametrize("num_tokens", [128, 1024, 4096])
def test_gather_kv_large_pool_sparse_access(total_slots, num_tokens):
    """Test sparse access pattern in a large KV cache pool."""
    if num_tokens > total_slots:
        pytest.skip("num_tokens exceeds total_slots")

    num_layers = 4
    num_heads = 8
    head_dim = 128
    dtype = torch.float16

    k_buffers = [
        torch.randn(total_slots, num_heads, head_dim, dtype=dtype, device="cuda")
        for _ in range(num_layers)
    ]
    v_buffers = [
        torch.randn(total_slots, num_heads, head_dim, dtype=dtype, device="cuda")
        for _ in range(num_layers)
    ]

    slot_indices = torch.randperm(total_slots, device="cuda")[:num_tokens].to(torch.int32)

    k_data_ptrs, v_data_ptrs, src_slot_stride, src_head_stride = create_pointer_tensors(k_buffers, v_buffers)

    output_size = num_layers * 2 * num_tokens * num_heads * head_dim
    pinned_output = torch.empty(output_size, dtype=dtype, device="cpu", pin_memory=True)

    gather_kv_to_pinned_all_layers(
        k_data_ptrs=k_data_ptrs,
        v_data_ptrs=v_data_ptrs,
        slot_indices=slot_indices,
        pinned_output=pinned_output,
        head_start=0,
        num_heads_to_gather=num_heads,
        num_layers=num_layers,
        head_dim=head_dim,
        src_slot_stride=src_slot_stride,
        src_head_stride=src_head_stride,
    )

    expected = reference_gather_kv(
        k_buffers, v_buffers, slot_indices.long(),
        head_start=0, num_heads_to_gather=num_heads
    )

    actual = pinned_output.view(num_heads, num_layers, 2, num_tokens, head_dim)
    torch.testing.assert_close(actual, expected.cpu(), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("num_tokens", [8192, 16384])
def test_gather_kv_long_sequence(num_tokens):
    """Test with long sequences (many tokens to gather)."""
    num_layers = 4
    num_heads = 8
    head_dim = 128
    total_slots = num_tokens + 10_000
    dtype = torch.float16

    k_buffers = [
        torch.randn(total_slots, num_heads, head_dim, dtype=dtype, device="cuda")
        for _ in range(num_layers)
    ]
    v_buffers = [
        torch.randn(total_slots, num_heads, head_dim, dtype=dtype, device="cuda")
        for _ in range(num_layers)
    ]

    slot_indices = torch.randperm(total_slots, device="cuda")[:num_tokens].to(torch.int32)

    k_data_ptrs, v_data_ptrs, src_slot_stride, src_head_stride = create_pointer_tensors(k_buffers, v_buffers)

    output_size = num_layers * 2 * num_tokens * num_heads * head_dim
    pinned_output = torch.empty(output_size, dtype=dtype, device="cpu", pin_memory=True)

    gather_kv_to_pinned_all_layers(
        k_data_ptrs=k_data_ptrs,
        v_data_ptrs=v_data_ptrs,
        slot_indices=slot_indices,
        pinned_output=pinned_output,
        head_start=0,
        num_heads_to_gather=num_heads,
        num_layers=num_layers,
        head_dim=head_dim,
        src_slot_stride=src_slot_stride,
        src_head_stride=src_head_stride,
    )

    expected = reference_gather_kv(
        k_buffers, v_buffers, slot_indices.long(),
        head_start=0, num_heads_to_gather=num_heads
    )

    actual = pinned_output.view(num_heads, num_layers, 2, num_tokens, head_dim)
    torch.testing.assert_close(actual, expected.cpu(), rtol=1e-3, atol=1e-3)


# =============================================================================
# Scatter Tests (Host -> Device)
# =============================================================================


@pytest.mark.parametrize("num_layers", [1, 4, 32])
@pytest.mark.parametrize("num_tokens", [1, 64, 512])
@pytest.mark.parametrize("num_heads", [8, 32])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_scatter_kv_full_heads(num_layers, num_tokens, num_heads, head_dim, dtype):
    """Test scattering all heads (no slicing)."""
    total_slots = 1024

    input_size = num_layers * 2 * num_tokens * num_heads * head_dim
    pinned_input = torch.randn(input_size, dtype=dtype, device="cpu", pin_memory=True)

    slot_indices = torch.randperm(total_slots, device="cuda")[:num_tokens].to(torch.int32)

    k_buffers = [
        torch.zeros(total_slots, num_heads, head_dim, dtype=dtype, device="cuda")
        for _ in range(num_layers)
    ]
    v_buffers = [
        torch.zeros(total_slots, num_heads, head_dim, dtype=dtype, device="cuda")
        for _ in range(num_layers)
    ]

    k_data_ptrs, v_data_ptrs, dst_slot_stride, dst_head_stride = create_pointer_tensors(k_buffers, v_buffers)

    scatter_kv_with_staging_all_layers(
        pinned_input=pinned_input,
        k_data_ptrs=k_data_ptrs,
        v_data_ptrs=v_data_ptrs,
        slot_indices=slot_indices,
        head_start=0,
        num_heads_to_scatter=num_heads,
        num_layers=num_layers,
        head_dim=head_dim,
        dst_slot_stride=dst_slot_stride,
        dst_head_stride=dst_head_stride,
    )

    expected_k, expected_v = reference_scatter_kv(
        pinned_input, slot_indices.long(),
        num_layers=num_layers,
        num_heads_to_scatter=num_heads,
        head_dim=head_dim,
        total_slots=total_slots,
        num_heads=num_heads,
        head_start=0,
        dtype=dtype,
    )

    for layer_idx in range(num_layers):
        torch.testing.assert_close(k_buffers[layer_idx], expected_k[layer_idx], rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(v_buffers[layer_idx], expected_v[layer_idx], rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("head_start,num_heads_to_scatter", [
    (0, 4),
    (4, 4),
    (2, 4),
    (0, 8),
])
def test_scatter_kv_head_slicing(head_start, num_heads_to_scatter):
    """Test scattering a subset of heads (for mixed-TP)."""
    num_layers = 4
    num_tokens = 256
    num_heads = 8
    head_dim = 128
    total_slots = 2048
    dtype = torch.float16

    input_size = num_layers * 2 * num_tokens * num_heads_to_scatter * head_dim
    pinned_input = torch.randn(input_size, dtype=dtype, device="cpu", pin_memory=True)

    slot_indices = torch.randperm(total_slots, device="cuda")[:num_tokens].to(torch.int32)

    k_buffers = [
        torch.zeros(total_slots, num_heads, head_dim, dtype=dtype, device="cuda")
        for _ in range(num_layers)
    ]
    v_buffers = [
        torch.zeros(total_slots, num_heads, head_dim, dtype=dtype, device="cuda")
        for _ in range(num_layers)
    ]

    k_data_ptrs, v_data_ptrs, dst_slot_stride, dst_head_stride = create_pointer_tensors(k_buffers, v_buffers)

    scatter_kv_with_staging_all_layers(
        pinned_input=pinned_input,
        k_data_ptrs=k_data_ptrs,
        v_data_ptrs=v_data_ptrs,
        slot_indices=slot_indices,
        head_start=head_start,
        num_heads_to_scatter=num_heads_to_scatter,
        num_layers=num_layers,
        head_dim=head_dim,
        dst_slot_stride=dst_slot_stride,
        dst_head_stride=dst_head_stride,
    )

    expected_k, expected_v = reference_scatter_kv(
        pinned_input, slot_indices.long(),
        num_layers=num_layers,
        num_heads_to_scatter=num_heads_to_scatter,
        head_dim=head_dim,
        total_slots=total_slots,
        num_heads=num_heads,
        head_start=head_start,
        dtype=dtype,
    )

    for layer_idx in range(num_layers):
        torch.testing.assert_close(k_buffers[layer_idx], expected_k[layer_idx], rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(v_buffers[layer_idx], expected_v[layer_idx], rtol=1e-3, atol=1e-3)


def test_scatter_kv_roundtrip():
    """
    Test that gather followed by scatter is identity (roundtrip test).

    Data gathered from GPU to pinned CPU should be correctly scattered back to GPU.
    """
    num_layers = 32
    num_tokens = 1024
    num_heads = 8
    head_dim = 128
    total_slots = 10_000
    dtype = torch.float16

    # Source KV buffers
    k_buffers_src = [
        torch.randn(total_slots, num_heads, head_dim, dtype=dtype, device="cuda")
        for _ in range(num_layers)
    ]
    v_buffers_src = [
        torch.randn(total_slots, num_heads, head_dim, dtype=dtype, device="cuda")
        for _ in range(num_layers)
    ]

    slot_indices = torch.randperm(total_slots, device="cuda")[:num_tokens].to(torch.int32)

    k_data_ptrs_src, v_data_ptrs_src, src_slot_stride, src_head_stride = create_pointer_tensors(k_buffers_src, v_buffers_src)

    # Gather
    output_size = num_layers * 2 * num_tokens * num_heads * head_dim
    pinned_buffer = torch.empty(output_size, dtype=dtype, device="cpu", pin_memory=True)

    gather_kv_to_pinned_all_layers(
        k_data_ptrs=k_data_ptrs_src,
        v_data_ptrs=v_data_ptrs_src,
        slot_indices=slot_indices,
        pinned_output=pinned_buffer,
        head_start=0,
        num_heads_to_gather=num_heads,
        num_layers=num_layers,
        head_dim=head_dim,
        src_slot_stride=src_slot_stride,
        src_head_stride=src_head_stride,
    )

    # Destination KV buffers
    k_buffers_dst = [
        torch.zeros(total_slots, num_heads, head_dim, dtype=dtype, device="cuda")
        for _ in range(num_layers)
    ]
    v_buffers_dst = [
        torch.zeros(total_slots, num_heads, head_dim, dtype=dtype, device="cuda")
        for _ in range(num_layers)
    ]

    k_data_ptrs_dst, v_data_ptrs_dst, dst_slot_stride, dst_head_stride = create_pointer_tensors(k_buffers_dst, v_buffers_dst)

    # Scatter
    scatter_kv_with_staging_all_layers(
        pinned_input=pinned_buffer,
        k_data_ptrs=k_data_ptrs_dst,
        v_data_ptrs=v_data_ptrs_dst,
        slot_indices=slot_indices,
        head_start=0,
        num_heads_to_scatter=num_heads,
        num_layers=num_layers,
        head_dim=head_dim,
        dst_slot_stride=dst_slot_stride,
        dst_head_stride=dst_head_stride,
    )

    # Verify roundtrip
    for layer_idx in range(num_layers):
        src_k = k_buffers_src[layer_idx][slot_indices.long()]
        dst_k = k_buffers_dst[layer_idx][slot_indices.long()]
        torch.testing.assert_close(dst_k, src_k, rtol=1e-3, atol=1e-3)

        src_v = v_buffers_src[layer_idx][slot_indices.long()]
        dst_v = v_buffers_dst[layer_idx][slot_indices.long()]
        torch.testing.assert_close(dst_v, src_v, rtol=1e-3, atol=1e-3)


def test_scatter_kv_large_pool():
    """Test scatter with large pool (high fragmentation)."""
    torch.cuda.empty_cache()

    num_layers = 32
    num_tokens = 2048
    num_heads = 8
    head_dim = 128
    total_slots = 100_000
    dtype = torch.float16

    input_size = num_layers * 2 * num_tokens * num_heads * head_dim
    pinned_input = torch.randn(input_size, dtype=dtype, device="cpu", pin_memory=True)

    slot_indices = torch.randperm(total_slots, device="cuda")[:num_tokens].to(torch.int32)

    k_buffers = [
        torch.zeros(total_slots, num_heads, head_dim, dtype=dtype, device="cuda")
        for _ in range(num_layers)
    ]
    v_buffers = [
        torch.zeros(total_slots, num_heads, head_dim, dtype=dtype, device="cuda")
        for _ in range(num_layers)
    ]

    k_data_ptrs, v_data_ptrs, dst_slot_stride, dst_head_stride = create_pointer_tensors(k_buffers, v_buffers)

    scatter_kv_with_staging_all_layers(
        pinned_input=pinned_input,
        k_data_ptrs=k_data_ptrs,
        v_data_ptrs=v_data_ptrs,
        slot_indices=slot_indices,
        head_start=0,
        num_heads_to_scatter=num_heads,
        num_layers=num_layers,
        head_dim=head_dim,
        dst_slot_stride=dst_slot_stride,
        dst_head_stride=dst_head_stride,
    )

    expected_k, expected_v = reference_scatter_kv(
        pinned_input, slot_indices.long(),
        num_layers=num_layers,
        num_heads_to_scatter=num_heads,
        head_dim=head_dim,
        total_slots=total_slots,
        num_heads=num_heads,
        head_start=0,
        dtype=dtype,
    )

    for layer_idx in range(num_layers):
        actual_k = k_buffers[layer_idx][slot_indices.long()]
        expected_k_at_slots = expected_k[layer_idx][slot_indices.long()]
        torch.testing.assert_close(actual_k, expected_k_at_slots, rtol=1e-3, atol=1e-3)

        actual_v = v_buffers[layer_idx][slot_indices.long()]
        expected_v_at_slots = expected_v[layer_idx][slot_indices.long()]
        torch.testing.assert_close(actual_v, expected_v_at_slots, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
