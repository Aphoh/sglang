"""
Tests for KV transfer Triton kernels.

Tests the gather and scatter kernels for KV cache transfers:
- gather_kv: GPU -> pinned CPU (device to host)
- scatter_kv: pinned CPU -> GPU (host to device)
"""

import sys
from pathlib import Path

import pytest
import torch

# Add the triton_ops path since it's not in sgl_kernel
triton_ops_path = Path(__file__).parent.parent.parent / "python" / "sglang" / "srt" / "layers" / "attention" / "triton_ops"
sys.path.insert(0, str(triton_ops_path))

from kv_transfer import gather_kv_to_pinned, scatter_kv_to_gpu


def reference_gather_kv(
    k_buffers: list[torch.Tensor],
    v_buffers: list[torch.Tensor],
    slot_indices: torch.Tensor,
    head_start: int,
    num_heads_to_gather: int,
) -> torch.Tensor:
    """
    Reference implementation of KV gather using PyTorch operations.

    Returns tensor of shape [num_layers, 2, num_tokens, num_heads_to_gather, head_dim]
    """
    num_layers = len(k_buffers)
    num_tokens = slot_indices.shape[0]
    head_dim = k_buffers[0].shape[2]
    dtype = k_buffers[0].dtype

    output = torch.zeros(
        (num_layers, 2, num_tokens, num_heads_to_gather, head_dim),
        dtype=dtype,
        device=k_buffers[0].device,
    )

    head_end = head_start + num_heads_to_gather

    for layer_idx in range(num_layers):
        output[layer_idx, 0] = k_buffers[layer_idx][slot_indices, head_start:head_end, :]
        output[layer_idx, 1] = v_buffers[layer_idx][slot_indices, head_start:head_end, :]

    return output


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

    output_size = num_layers * 2 * num_tokens * num_heads * head_dim
    pinned_output = torch.empty(output_size, dtype=dtype, device="cpu", pin_memory=True)

    gather_kv_to_pinned(
        k_buffers=k_buffers,
        v_buffers=v_buffers,
        slot_indices=slot_indices,
        pinned_output=pinned_output,
        head_start=0,
        num_heads_to_gather=num_heads,
    )

    expected = reference_gather_kv(
        k_buffers, v_buffers, slot_indices.long(),
        head_start=0, num_heads_to_gather=num_heads
    )

    actual = pinned_output.view(num_layers, 2, num_tokens, num_heads, head_dim)
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

    output_size = num_layers * 2 * num_tokens * num_heads_to_gather * head_dim
    pinned_output = torch.empty(output_size, dtype=dtype, device="cpu", pin_memory=True)

    gather_kv_to_pinned(
        k_buffers=k_buffers,
        v_buffers=v_buffers,
        slot_indices=slot_indices,
        pinned_output=pinned_output,
        head_start=head_start,
        num_heads_to_gather=num_heads_to_gather,
    )

    expected = reference_gather_kv(
        k_buffers, v_buffers, slot_indices.long(),
        head_start=head_start, num_heads_to_gather=num_heads_to_gather
    )

    actual = pinned_output.view(num_layers, 2, num_tokens, num_heads_to_gather, head_dim)
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

    output_size = num_layers * 2 * num_tokens * num_heads * head_dim
    pinned_output = torch.empty(output_size, dtype=dtype, device="cpu", pin_memory=True)

    gather_kv_to_pinned(
        k_buffers=k_buffers,
        v_buffers=v_buffers,
        slot_indices=slot_indices,
        pinned_output=pinned_output,
        head_start=0,
        num_heads_to_gather=num_heads,
    )

    expected = reference_gather_kv(
        k_buffers, v_buffers, slot_indices.long(),
        head_start=0, num_heads_to_gather=num_heads
    )

    actual = pinned_output.view(num_layers, 2, num_tokens, num_heads, head_dim)
    torch.testing.assert_close(actual, expected.cpu(), rtol=1e-3, atol=1e-3)


def test_gather_kv_non_pinned_raises():
    """Test that non-pinned output raises an error."""
    k_buffers = [torch.randn(64, 4, 32, dtype=torch.float16, device="cuda")]
    v_buffers = [torch.randn(64, 4, 32, dtype=torch.float16, device="cuda")]
    slot_indices = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device="cuda")

    non_pinned_output = torch.empty(1 * 2 * 4 * 4 * 32, dtype=torch.float16, device="cpu")

    with pytest.raises(AssertionError, match="pinned"):
        gather_kv_to_pinned(
            k_buffers=k_buffers,
            v_buffers=v_buffers,
            slot_indices=slot_indices,
            pinned_output=non_pinned_output,
            head_start=0,
            num_heads_to_gather=4,
        )


# =============================================================================
# Memory Fragmentation Tests
# =============================================================================
# These tests simulate realistic KV cache fragmentation patterns where tokens
# from a single request are scattered across a large memory pool.


@pytest.mark.parametrize("total_slots", [100_000, 500_000])
@pytest.mark.parametrize("num_tokens", [128, 1024, 4096])
def test_gather_kv_large_pool_sparse_access(total_slots, num_tokens):
    """
    Test sparse access pattern in a large KV cache pool.

    Simulates pulling a small number of tokens from a very large pool,
    which is common when the KV cache has many concurrent requests.
    """
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

    # Random sparse indices across the entire pool
    slot_indices = torch.randperm(total_slots, device="cuda")[:num_tokens].to(torch.int32)

    output_size = num_layers * 2 * num_tokens * num_heads * head_dim
    pinned_output = torch.empty(output_size, dtype=dtype, device="cpu", pin_memory=True)

    gather_kv_to_pinned(
        k_buffers=k_buffers,
        v_buffers=v_buffers,
        slot_indices=slot_indices,
        pinned_output=pinned_output,
        head_start=0,
        num_heads_to_gather=num_heads,
    )

    expected = reference_gather_kv(
        k_buffers, v_buffers, slot_indices.long(),
        head_start=0, num_heads_to_gather=num_heads
    )

    actual = pinned_output.view(num_layers, 2, num_tokens, num_heads, head_dim)
    torch.testing.assert_close(actual, expected.cpu(), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("stride", [2, 7, 64, 256])
def test_gather_kv_strided_access(stride):
    """
    Test strided access pattern (worst case for coalescing).

    Simulates a pathological fragmentation pattern where slots are
    allocated in a strided pattern (e.g., every Nth slot).
    """
    num_layers = 4
    num_heads = 8
    head_dim = 128
    num_tokens = 512
    total_slots = num_tokens * stride + 1000  # Ensure enough slots
    dtype = torch.float16

    k_buffers = [
        torch.randn(total_slots, num_heads, head_dim, dtype=dtype, device="cuda")
        for _ in range(num_layers)
    ]
    v_buffers = [
        torch.randn(total_slots, num_heads, head_dim, dtype=dtype, device="cuda")
        for _ in range(num_layers)
    ]

    # Strided indices: 0, stride, 2*stride, 3*stride, ...
    slot_indices = torch.arange(0, num_tokens * stride, stride, device="cuda", dtype=torch.int32)

    output_size = num_layers * 2 * num_tokens * num_heads * head_dim
    pinned_output = torch.empty(output_size, dtype=dtype, device="cpu", pin_memory=True)

    gather_kv_to_pinned(
        k_buffers=k_buffers,
        v_buffers=v_buffers,
        slot_indices=slot_indices,
        pinned_output=pinned_output,
        head_start=0,
        num_heads_to_gather=num_heads,
    )

    expected = reference_gather_kv(
        k_buffers, v_buffers, slot_indices.long(),
        head_start=0, num_heads_to_gather=num_heads
    )

    actual = pinned_output.view(num_layers, 2, num_tokens, num_heads, head_dim)
    torch.testing.assert_close(actual, expected.cpu(), rtol=1e-3, atol=1e-3)


def test_gather_kv_mixed_contiguous_scattered():
    """
    Test mixed access pattern with some contiguous runs and some scattered.

    Simulates realistic fragmentation where some tokens are contiguous
    (recently allocated together) and some are scattered (from older allocations).
    """
    num_layers = 4
    num_heads = 8
    head_dim = 128
    total_slots = 100_000
    dtype = torch.float16

    k_buffers = [
        torch.randn(total_slots, num_heads, head_dim, dtype=dtype, device="cuda")
        for _ in range(num_layers)
    ]
    v_buffers = [
        torch.randn(total_slots, num_heads, head_dim, dtype=dtype, device="cuda")
        for _ in range(num_layers)
    ]

    # Build mixed pattern:
    # - 3 contiguous runs of 100 tokens each at different locations
    # - 200 randomly scattered tokens
    indices_list = []

    # Contiguous runs at different parts of the pool
    indices_list.append(torch.arange(1000, 1100, device="cuda"))      # Near start
    indices_list.append(torch.arange(50000, 50100, device="cuda"))    # Middle
    indices_list.append(torch.arange(99000, 99100, device="cuda"))    # Near end

    # Scattered tokens
    scattered = torch.randperm(total_slots, device="cuda")[:200]
    indices_list.append(scattered)

    slot_indices = torch.cat(indices_list).to(torch.int32)
    num_tokens = slot_indices.shape[0]

    output_size = num_layers * 2 * num_tokens * num_heads * head_dim
    pinned_output = torch.empty(output_size, dtype=dtype, device="cpu", pin_memory=True)

    gather_kv_to_pinned(
        k_buffers=k_buffers,
        v_buffers=v_buffers,
        slot_indices=slot_indices,
        pinned_output=pinned_output,
        head_start=0,
        num_heads_to_gather=num_heads,
    )

    expected = reference_gather_kv(
        k_buffers, v_buffers, slot_indices.long(),
        head_start=0, num_heads_to_gather=num_heads
    )

    actual = pinned_output.view(num_layers, 2, num_tokens, num_heads, head_dim)
    torch.testing.assert_close(actual, expected.cpu(), rtol=1e-3, atol=1e-3)


def test_gather_kv_boundary_slots():
    """
    Test accessing slots at boundaries of the pool.

    Ensures correct handling of first slot, last slot, and slots
    near the boundaries.
    """
    num_layers = 4
    num_heads = 8
    head_dim = 128
    total_slots = 100_000
    dtype = torch.float16

    k_buffers = [
        torch.randn(total_slots, num_heads, head_dim, dtype=dtype, device="cuda")
        for _ in range(num_layers)
    ]
    v_buffers = [
        torch.randn(total_slots, num_heads, head_dim, dtype=dtype, device="cuda")
        for _ in range(num_layers)
    ]

    # Include boundary slots explicitly
    slot_indices = torch.tensor([
        0,                      # First slot
        1,                      # Second slot
        total_slots // 2,       # Middle
        total_slots - 2,        # Second to last
        total_slots - 1,        # Last slot
    ] + list(range(100, 200))   # Plus a contiguous chunk
    , device="cuda", dtype=torch.int32)

    num_tokens = slot_indices.shape[0]

    output_size = num_layers * 2 * num_tokens * num_heads * head_dim
    pinned_output = torch.empty(output_size, dtype=dtype, device="cpu", pin_memory=True)

    gather_kv_to_pinned(
        k_buffers=k_buffers,
        v_buffers=v_buffers,
        slot_indices=slot_indices,
        pinned_output=pinned_output,
        head_start=0,
        num_heads_to_gather=num_heads,
    )

    expected = reference_gather_kv(
        k_buffers, v_buffers, slot_indices.long(),
        head_start=0, num_heads_to_gather=num_heads
    )

    actual = pinned_output.view(num_layers, 2, num_tokens, num_heads, head_dim)
    torch.testing.assert_close(actual, expected.cpu(), rtol=1e-3, atol=1e-3)


def test_gather_kv_realistic_fragmentation():
    """
    Test a realistic fragmentation pattern from a simulated allocator.

    Simulates what happens when many requests allocate and free slots
    over time, leaving the pool fragmented.
    """
    num_layers = 32  # Realistic layer count
    num_heads = 8
    head_dim = 128
    total_slots = 200_000  # Large pool
    num_tokens = 2048      # Moderate sequence length
    dtype = torch.float16

    k_buffers = [
        torch.randn(total_slots, num_heads, head_dim, dtype=dtype, device="cuda")
        for _ in range(num_layers)
    ]
    v_buffers = [
        torch.randn(total_slots, num_heads, head_dim, dtype=dtype, device="cuda")
        for _ in range(num_layers)
    ]

    # Simulate fragmented allocation:
    # The allocator gave us slots from different "eras" of allocation
    indices_list = []

    # Era 1: Old allocation, slots 5000-5200
    indices_list.append(torch.arange(5000, 5200, device="cuda"))

    # Era 2: Slightly newer, scattered in 20000-40000 range
    era2_base = torch.arange(20000, 40000, 50, device="cuda")  # Every 50th slot
    indices_list.append(era2_base[:200])

    # Era 3: Recent allocation, several small contiguous chunks
    for start in [80000, 82000, 84000, 86000, 88000]:
        indices_list.append(torch.arange(start, start + 100, device="cuda"))

    # Era 4: Very recent, contiguous chunk
    indices_list.append(torch.arange(150000, 150000 + (num_tokens - 1100), device="cuda"))

    slot_indices = torch.cat(indices_list).to(torch.int32)
    actual_num_tokens = slot_indices.shape[0]

    output_size = num_layers * 2 * actual_num_tokens * num_heads * head_dim
    pinned_output = torch.empty(output_size, dtype=dtype, device="cpu", pin_memory=True)

    gather_kv_to_pinned(
        k_buffers=k_buffers,
        v_buffers=v_buffers,
        slot_indices=slot_indices,
        pinned_output=pinned_output,
        head_start=0,
        num_heads_to_gather=num_heads,
    )

    expected = reference_gather_kv(
        k_buffers, v_buffers, slot_indices.long(),
        head_start=0, num_heads_to_gather=num_heads
    )

    actual = pinned_output.view(num_layers, 2, actual_num_tokens, num_heads, head_dim)
    torch.testing.assert_close(actual, expected.cpu(), rtol=1e-3, atol=1e-3)


def test_gather_kv_reverse_order_indices():
    """
    Test with indices in reverse order.

    The output should still be in the order specified by slot_indices,
    not sorted by slot number.
    """
    num_layers = 4
    num_heads = 8
    head_dim = 128
    total_slots = 10_000
    num_tokens = 500
    dtype = torch.float16

    k_buffers = [
        torch.randn(total_slots, num_heads, head_dim, dtype=dtype, device="cuda")
        for _ in range(num_layers)
    ]
    v_buffers = [
        torch.randn(total_slots, num_heads, head_dim, dtype=dtype, device="cuda")
        for _ in range(num_layers)
    ]

    # Indices in reverse order
    slot_indices = torch.arange(num_tokens - 1, -1, -1, device="cuda", dtype=torch.int32) * 10

    output_size = num_layers * 2 * num_tokens * num_heads * head_dim
    pinned_output = torch.empty(output_size, dtype=dtype, device="cpu", pin_memory=True)

    gather_kv_to_pinned(
        k_buffers=k_buffers,
        v_buffers=v_buffers,
        slot_indices=slot_indices,
        pinned_output=pinned_output,
        head_start=0,
        num_heads_to_gather=num_heads,
    )

    expected = reference_gather_kv(
        k_buffers, v_buffers, slot_indices.long(),
        head_start=0, num_heads_to_gather=num_heads
    )

    actual = pinned_output.view(num_layers, 2, num_tokens, num_heads, head_dim)
    torch.testing.assert_close(actual, expected.cpu(), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("num_tokens", [8192, 16384])
def test_gather_kv_long_sequence(num_tokens):
    """
    Test with long sequences (many tokens to gather).

    Tests the kernel's ability to handle large transfers efficiently.
    """
    num_layers = 4
    num_heads = 8
    head_dim = 128
    total_slots = num_tokens + 10_000  # Pool slightly larger than sequence
    dtype = torch.float16

    k_buffers = [
        torch.randn(total_slots, num_heads, head_dim, dtype=dtype, device="cuda")
        for _ in range(num_layers)
    ]
    v_buffers = [
        torch.randn(total_slots, num_heads, head_dim, dtype=dtype, device="cuda")
        for _ in range(num_layers)
    ]

    # Scattered indices
    slot_indices = torch.randperm(total_slots, device="cuda")[:num_tokens].to(torch.int32)

    output_size = num_layers * 2 * num_tokens * num_heads * head_dim
    pinned_output = torch.empty(output_size, dtype=dtype, device="cpu", pin_memory=True)

    gather_kv_to_pinned(
        k_buffers=k_buffers,
        v_buffers=v_buffers,
        slot_indices=slot_indices,
        pinned_output=pinned_output,
        head_start=0,
        num_heads_to_gather=num_heads,
    )

    expected = reference_gather_kv(
        k_buffers, v_buffers, slot_indices.long(),
        head_start=0, num_heads_to_gather=num_heads
    )

    actual = pinned_output.view(num_layers, 2, num_tokens, num_heads, head_dim)
    torch.testing.assert_close(actual, expected.cpu(), rtol=1e-3, atol=1e-3)


# =============================================================================
# Scatter Tests (Host -> Device)
# =============================================================================


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

    Returns (k_buffers, v_buffers) with data scattered to the specified slots.
    """
    num_tokens = slot_indices.shape[0]

    # Initialize empty KV buffers
    k_buffers = [
        torch.zeros(total_slots, num_heads, head_dim, dtype=dtype, device="cuda")
        for _ in range(num_layers)
    ]
    v_buffers = [
        torch.zeros(total_slots, num_heads, head_dim, dtype=dtype, device="cuda")
        for _ in range(num_layers)
    ]

    # Reshape input to expected layout
    input_shaped = pinned_input.view(num_layers, 2, num_tokens, num_heads_to_scatter, head_dim)
    head_end = head_start + num_heads_to_scatter

    for layer_idx in range(num_layers):
        k_buffers[layer_idx][slot_indices, head_start:head_end, :] = input_shaped[layer_idx, 0].cuda()
        v_buffers[layer_idx][slot_indices, head_start:head_end, :] = input_shaped[layer_idx, 1].cuda()

    return k_buffers, v_buffers


@pytest.mark.parametrize("num_layers", [1, 4, 32])
@pytest.mark.parametrize("num_tokens", [1, 64, 512])
@pytest.mark.parametrize("num_heads", [8, 32])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_scatter_kv_full_heads(num_layers, num_tokens, num_heads, head_dim, dtype):
    """Test scattering all heads (no slicing)."""
    total_slots = 1024

    # Create input data in pinned memory
    input_size = num_layers * 2 * num_tokens * num_heads * head_dim
    pinned_input = torch.randn(input_size, dtype=dtype, device="cpu", pin_memory=True)

    slot_indices = torch.randperm(total_slots, device="cuda")[:num_tokens].to(torch.int32)

    # Create empty KV buffers on GPU
    k_buffers = [
        torch.zeros(total_slots, num_heads, head_dim, dtype=dtype, device="cuda")
        for _ in range(num_layers)
    ]
    v_buffers = [
        torch.zeros(total_slots, num_heads, head_dim, dtype=dtype, device="cuda")
        for _ in range(num_layers)
    ]

    scatter_kv_to_gpu(
        pinned_input=pinned_input,
        k_buffers=k_buffers,
        v_buffers=v_buffers,
        slot_indices=slot_indices,
        head_start=0,
        num_heads_to_scatter=num_heads,
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
    (0, 4),   # First 4 of 8 heads
    (4, 4),   # Last 4 of 8 heads
    (2, 4),   # Middle 4 of 8 heads
    (0, 8),   # All 8 heads
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

    scatter_kv_to_gpu(
        pinned_input=pinned_input,
        k_buffers=k_buffers,
        v_buffers=v_buffers,
        slot_indices=slot_indices,
        head_start=head_start,
        num_heads_to_scatter=num_heads_to_scatter,
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

    This is the most important test - data gathered from GPU to pinned CPU
    should be correctly scattered back to GPU KV cache.
    """
    num_layers = 4
    num_tokens = 512
    num_heads = 8
    head_dim = 128
    total_slots = 2048
    dtype = torch.float16

    # Create source KV buffers with known data
    k_buffers_src = [
        torch.randn(total_slots, num_heads, head_dim, dtype=dtype, device="cuda")
        for _ in range(num_layers)
    ]
    v_buffers_src = [
        torch.randn(total_slots, num_heads, head_dim, dtype=dtype, device="cuda")
        for _ in range(num_layers)
    ]

    slot_indices = torch.randperm(total_slots, device="cuda")[:num_tokens].to(torch.int32)

    # Gather from source to pinned
    output_size = num_layers * 2 * num_tokens * num_heads * head_dim
    pinned_buffer = torch.empty(output_size, dtype=dtype, device="cpu", pin_memory=True)

    gather_kv_to_pinned(
        k_buffers=k_buffers_src,
        v_buffers=v_buffers_src,
        slot_indices=slot_indices,
        pinned_output=pinned_buffer,
        head_start=0,
        num_heads_to_gather=num_heads,
    )

    # Create destination KV buffers (all zeros)
    k_buffers_dst = [
        torch.zeros(total_slots, num_heads, head_dim, dtype=dtype, device="cuda")
        for _ in range(num_layers)
    ]
    v_buffers_dst = [
        torch.zeros(total_slots, num_heads, head_dim, dtype=dtype, device="cuda")
        for _ in range(num_layers)
    ]

    # Scatter from pinned to destination
    scatter_kv_to_gpu(
        pinned_input=pinned_buffer,
        k_buffers=k_buffers_dst,
        v_buffers=v_buffers_dst,
        slot_indices=slot_indices,
        head_start=0,
        num_heads_to_scatter=num_heads,
    )

    # Verify: destination[slot_indices] should match source[slot_indices]
    for layer_idx in range(num_layers):
        src_k = k_buffers_src[layer_idx][slot_indices.long()]
        dst_k = k_buffers_dst[layer_idx][slot_indices.long()]
        torch.testing.assert_close(dst_k, src_k, rtol=1e-3, atol=1e-3)

        src_v = v_buffers_src[layer_idx][slot_indices.long()]
        dst_v = v_buffers_dst[layer_idx][slot_indices.long()]
        torch.testing.assert_close(dst_v, src_v, rtol=1e-3, atol=1e-3)


def test_scatter_kv_large_pool_sparse():
    """Test scatter with large pool and sparse access (high fragmentation)."""
    # Clean up memory from previous tests
    torch.cuda.empty_cache()

    num_layers = 32
    num_tokens = 2048
    num_heads = 8
    head_dim = 128
    total_slots = 200_000  # Large pool (reduced from 500K to avoid OOM)
    dtype = torch.float16

    input_size = num_layers * 2 * num_tokens * num_heads * head_dim
    pinned_input = torch.randn(input_size, dtype=dtype, device="cpu", pin_memory=True)

    # Random sparse access
    slot_indices = torch.randperm(total_slots, device="cuda")[:num_tokens].to(torch.int32)

    k_buffers = [
        torch.zeros(total_slots, num_heads, head_dim, dtype=dtype, device="cuda")
        for _ in range(num_layers)
    ]
    v_buffers = [
        torch.zeros(total_slots, num_heads, head_dim, dtype=dtype, device="cuda")
        for _ in range(num_layers)
    ]

    scatter_kv_to_gpu(
        pinned_input=pinned_input,
        k_buffers=k_buffers,
        v_buffers=v_buffers,
        slot_indices=slot_indices,
        head_start=0,
        num_heads_to_scatter=num_heads,
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

    # Only check the slots we wrote to (checking all 500K slots would be slow)
    for layer_idx in range(num_layers):
        actual_k = k_buffers[layer_idx][slot_indices.long()]
        expected_k_at_slots = expected_k[layer_idx][slot_indices.long()]
        torch.testing.assert_close(actual_k, expected_k_at_slots, rtol=1e-3, atol=1e-3)

        actual_v = v_buffers[layer_idx][slot_indices.long()]
        expected_v_at_slots = expected_v[layer_idx][slot_indices.long()]
        torch.testing.assert_close(actual_v, expected_v_at_slots, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
