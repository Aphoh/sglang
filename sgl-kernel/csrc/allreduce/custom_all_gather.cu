// SPDX-License-Identifier: Apache-2.0
// Adapted from https://github.com/Aphoh/flashinfer/blob/
// 1ac01069632461c4a84110d2c9c630a95a4c77e3/csrc/symmetric_allgather.cu.

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/all.h>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <vector>

namespace {

constexpr int kNumBuffers = 3;
constexpr int kMaxWorldSize = 8;
constexpr int kThreads = 256;
constexpr int kInvalidSequenceStatus = 1;
constexpr int kTimeoutStatus = 2;
constexpr int kTimeoutDetailStatus = 3;
constexpr int kStatusWords = 4;
constexpr unsigned long long kTimeoutCycles = 30000000000ULL;

size_t align_up(size_t value, size_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

// Every rank owns one peer-mapped workspace. A call pushes its input into each
// peer's source slot, publishes readiness, copies peer slots into its output,
// then publishes completion. Rotating slots allow a rank to return without a
// tail barrier; a device-side sequence stays fresh across CUDA graph replays.
struct Layout {
  size_t slot_bytes;
  size_t control;
  size_t scratch;
  size_t flags;
  size_t done;
  size_t arrive_copy;
  size_t arrive_done;
  size_t status;
  size_t bytes;
};

Layout make_layout(int64_t max_bytes, int64_t world_size) {
  TORCH_CHECK(max_bytes > 0, "max_bytes must be positive");
  TORCH_CHECK(world_size > 1 && world_size <= kMaxWorldSize, "world_size must be in [2, 8]");

  Layout layout{};
  layout.slot_bytes = align_up(static_cast<size_t>(max_bytes), 128);
  size_t offset = 0;
  layout.control = offset;
  offset += sizeof(uint64_t);
  offset = align_up(offset, 128);
  layout.scratch = offset;
  offset += kNumBuffers * static_cast<size_t>(world_size) * layout.slot_bytes;
  offset = align_up(offset, 128);
  layout.flags = offset;
  offset += kNumBuffers * static_cast<size_t>(world_size) * sizeof(int);
  offset = align_up(offset, 128);
  layout.done = offset;
  offset += kNumBuffers * static_cast<size_t>(world_size) * sizeof(int);
  offset = align_up(offset, 128);
  layout.arrive_copy = offset;
  offset += sizeof(int);
  offset = align_up(offset, 128);
  layout.arrive_done = offset;
  offset += sizeof(int);
  offset = align_up(offset, 128);
  layout.status = offset;
  offset += kStatusWords * sizeof(int);
  layout.bytes = align_up(offset, 4096);
  return layout;
}

struct KernelContext {
  char* local_base;
  const char* input;
  char* output;
  char* peer_bases[kMaxWorldSize];
  size_t bytes;
  size_t slot_bytes;
  size_t scratch;
  size_t flags;
  size_t done;
  size_t arrive_copy;
  size_t arrive_done;
  size_t status;
  int world_size;
  int rank;
  unsigned long long timeout_cycles;
};

__device__ __forceinline__ int load_acquire(const int* value) {
  int result;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(result) : "l"(value));
  return result;
}

__device__ __forceinline__ void store_release(int* value, int update) {
  asm volatile("st.release.sys.global.u32 [%1], %0;" : : "r"(update), "l"(value));
}

__device__ bool wait_until_at_least(int* value, int expected, unsigned long long timeout_cycles, int* status) {
  const unsigned long long start = clock64();
  while (load_acquire(value) < expected) {
    if (load_acquire(status + kInvalidSequenceStatus) != 0) {
      return false;
    }
    if (clock64() - start > timeout_cycles) {
      atomicExch(status + kTimeoutStatus, 1);
      return false;
    }
  }
  return true;
}

__device__ __forceinline__ void
copy_bytes(const char* source, char* destination, size_t bytes, size_t tid, size_t stride) {
  const uintptr_t alignment = reinterpret_cast<uintptr_t>(source) | reinterpret_cast<uintptr_t>(destination) | bytes;
  if ((alignment & (alignof(uint4) - 1)) == 0) {
    const auto* source_vec = reinterpret_cast<const uint4*>(source);
    auto* destination_vec = reinterpret_cast<uint4*>(destination);
    const size_t vector_count = bytes / sizeof(uint4);
    for (size_t index = tid; index < vector_count; index += stride) {
      destination_vec[index] = source_vec[index];
    }
    return;
  }
  for (size_t index = tid; index < bytes; index += stride) {
    destination[index] = source[index];
  }
}

__device__ bool wait_for_all(int* values, int sequence, KernelContext context, int timeout_stage) {
  int* status = reinterpret_cast<int*>(context.local_base + context.status);
  for (int source = 0; source < context.world_size; ++source) {
    if (!wait_until_at_least(values + source, sequence, context.timeout_cycles, status)) {
      if (status[kTimeoutStatus] != 0) {
        atomicCAS(status + kTimeoutDetailStatus, 0, timeout_stage * 100000 + sequence * 100 + source);
      }
      return false;
    }
  }
  return true;
}

__device__ void publish_sequence(KernelContext context, size_t offset, int sequence) {
  for (int destination = 0; destination < context.world_size; ++destination) {
    if (destination == context.rank) {
      continue;
    }
    store_release(reinterpret_cast<int*>(context.peer_bases[destination] + offset), sequence);
  }
  store_release(reinterpret_cast<int*>(context.local_base + offset), sequence);
}

__device__ void all_gather_step(KernelContext context, int sequence) {
  const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  const int buffer = sequence % kNumBuffers;
  int* status = reinterpret_cast<int*>(context.local_base + context.status);
  __shared__ int block_ready;

  int* local_done = reinterpret_cast<int*>(
      context.local_base + context.done + static_cast<size_t>(buffer) * context.world_size * sizeof(int));
  if (threadIdx.x == 0) {
    block_ready = sequence <= kNumBuffers || wait_for_all(local_done, sequence - kNumBuffers, context, 1);
  }
  __syncthreads();
  if (!block_ready) {
    return;
  }

  const size_t target_offset =
      context.scratch + (static_cast<size_t>(buffer) * context.world_size + context.rank) * context.slot_bytes;
  copy_bytes(
      context.input, context.output + static_cast<size_t>(context.rank) * context.bytes, context.bytes, tid, stride);
  for (int destination = 0; destination < context.world_size; ++destination) {
    if (destination == context.rank) {
      continue;
    }
    copy_bytes(context.input, context.peer_bases[destination] + target_offset, context.bytes, tid, stride);
  }

  __syncthreads();
  int* arrive_copy = reinterpret_cast<int*>(context.local_base + context.arrive_copy);
  int* local_flags = reinterpret_cast<int*>(
      context.local_base + context.flags + static_cast<size_t>(buffer) * context.world_size * sizeof(int));
  if (gridDim.x == 1) {
    if (threadIdx.x == 0) {
      __threadfence_system();
      const size_t flag_offset =
          context.flags + (static_cast<size_t>(buffer) * context.world_size + context.rank) * sizeof(int);
      publish_sequence(context, flag_offset, sequence);
    }
    __syncthreads();
  } else if (threadIdx.x == 0) {
    __threadfence_system();
    const int prior = atomicAdd(arrive_copy, 1);
    if (prior == gridDim.x - 1) {
      *arrive_copy = 0;
      const size_t flag_offset =
          context.flags + (static_cast<size_t>(buffer) * context.world_size + context.rank) * sizeof(int);
      publish_sequence(context, flag_offset, sequence);
    }
  }

  if (threadIdx.x == 0) {
    block_ready = wait_for_all(local_flags, sequence, context, 2);
  }
  __syncthreads();
  if (!block_ready) {
    return;
  }

  for (int source = 0; source < context.world_size; ++source) {
    if (source == context.rank) {
      continue;
    }
    const char* source_slot = context.local_base + context.scratch +
                              (static_cast<size_t>(buffer) * context.world_size + source) * context.slot_bytes;
    copy_bytes(source_slot, context.output + static_cast<size_t>(source) * context.bytes, context.bytes, tid, stride);
  }

  __syncthreads();
  int* arrive_done = reinterpret_cast<int*>(context.local_base + context.arrive_done);
  if (gridDim.x == 1) {
    if (threadIdx.x == 0) {
      const size_t done_offset =
          context.done + (static_cast<size_t>(buffer) * context.world_size + context.rank) * sizeof(int);
      publish_sequence(context, done_offset, sequence);
    }
    __syncthreads();
    return;
  }
  if (threadIdx.x == 0) {
    const int prior = atomicAdd(arrive_done, 1);
    if (prior == gridDim.x - 1) {
      *arrive_done = 0;
      const size_t done_offset =
          context.done + (static_cast<size_t>(buffer) * context.world_size + context.rank) * sizeof(int);
      publish_sequence(context, done_offset, sequence);
    }
  }

  if (threadIdx.x == 0) {
    block_ready = wait_until_at_least(local_done + context.rank, sequence, context.timeout_cycles, status);
    if (!block_ready && status[kTimeoutStatus] != 0) {
      atomicCAS(status + kTimeoutDetailStatus, 0, 300000 + sequence * 100 + context.rank);
    }
  }
  __syncthreads();
}

__global__ void all_gather_kernel(KernelContext context, const uint64_t* ticket) {
  const uint64_t sequence = *ticket;
  int* status = reinterpret_cast<int*>(context.local_base + context.status);
  if (sequence == 0 || sequence > static_cast<uint64_t>(INT32_MAX)) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
      atomicExch(status + kInvalidSequenceStatus, 1);
    }
    return;
  }
  all_gather_step(context, static_cast<int>(sequence));
}

__global__ void initialize_control_kernel(uint64_t* next_sequence) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *next_sequence = 1;
  }
}

__global__ void reserve_sequence_kernel(uint64_t* next_sequence, uint64_t* ticket) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *ticket = atomicAdd(reinterpret_cast<unsigned long long*>(next_sequence), 1ULL);
  }
}

void check_anchor(const torch::Tensor& anchor) {
  TORCH_CHECK(anchor.is_cuda(), "all-gather anchor must be a CUDA tensor");
}

}  // namespace

int64_t custom_all_gather_workspace_size(int64_t max_bytes, int64_t world_size) {
  return static_cast<int64_t>(make_layout(max_bytes, world_size).bytes);
}

void custom_all_gather_initialize(torch::Tensor& anchor, int64_t local_ptr, int64_t max_bytes, int64_t world_size) {
  check_anchor(anchor);
  TORCH_CHECK(local_ptr != 0, "null local all-gather workspace pointer");
  const auto device_guard = c10::cuda::OptionalCUDAGuard(device_of(anchor));
  const auto stream = c10::cuda::getCurrentCUDAStream().stream();
  const Layout layout = make_layout(max_bytes, world_size);
  auto* local_base = reinterpret_cast<char*>(local_ptr);
  AT_CUDA_CHECK(cudaMemsetAsync(local_base, 0, layout.bytes, stream));
  initialize_control_kernel<<<1, 1, 0, stream>>>(reinterpret_cast<uint64_t*>(local_base + layout.control));
  AT_CUDA_CHECK(cudaGetLastError());
}

void custom_all_gather(
    torch::Tensor& input,
    torch::Tensor& output,
    torch::Tensor& ticket,
    const std::vector<int64_t>& peer_ptrs,
    int64_t rank,
    int64_t max_bytes) {
  const int world_size = static_cast<int>(peer_ptrs.size());
  TORCH_CHECK(world_size > 1 && world_size <= kMaxWorldSize, "world_size must be in [2, 8]");
  TORCH_CHECK(rank >= 0 && rank < world_size, "invalid rank");
  TORCH_CHECK(input.is_cuda() && output.is_cuda(), "input and output must be CUDA tensors");
  TORCH_CHECK(input.device() == output.device(), "input and output must be on the same device");
  TORCH_CHECK(input.is_contiguous() && output.is_contiguous(), "input and output must be contiguous");
  TORCH_CHECK(input.scalar_type() == output.scalar_type(), "input and output dtype must match");
  TORCH_CHECK(
      input.scalar_type() == torch::kFloat16 || input.scalar_type() == torch::kBFloat16 ||
          input.scalar_type() == torch::kFloat32,
      "all-gather supports float16, bfloat16, and float32");
  TORCH_CHECK(output.numel() == input.numel() * world_size, "invalid all-gather output size");
  TORCH_CHECK(ticket.is_cuda() && ticket.scalar_type() == torch::kUInt64, "ticket must be CUDA uint64");
  TORCH_CHECK(ticket.is_contiguous() && ticket.numel() >= 1, "ticket must contain a contiguous uint64 value");
  TORCH_CHECK(ticket.device() == input.device(), "ticket must be on the input device");
  const size_t input_bytes = static_cast<size_t>(input.numel() * input.element_size());
  TORCH_CHECK(input_bytes > 0 && input_bytes <= static_cast<size_t>(max_bytes), "input exceeds all-gather workspace");

  const auto device_guard = c10::cuda::OptionalCUDAGuard(device_of(input));
  const auto stream = c10::cuda::getCurrentCUDAStream().stream();
  const Layout layout = make_layout(max_bytes, world_size);
  KernelContext context{};
  context.input = static_cast<const char*>(input.data_ptr());
  context.output = static_cast<char*>(output.data_ptr());
  for (int peer = 0; peer < world_size; ++peer) {
    TORCH_CHECK(peer_ptrs[peer] != 0, "null all-gather workspace pointer");
    context.peer_bases[peer] = reinterpret_cast<char*>(peer_ptrs[peer]);
  }
  context.local_base = context.peer_bases[rank];
  context.bytes = input_bytes;
  context.slot_bytes = layout.slot_bytes;
  context.scratch = layout.scratch;
  context.flags = layout.flags;
  context.done = layout.done;
  context.arrive_copy = layout.arrive_copy;
  context.arrive_done = layout.arrive_done;
  context.status = layout.status;
  context.world_size = world_size;
  context.rank = static_cast<int>(rank);
  context.timeout_cycles = kTimeoutCycles;

  auto* sequence_ticket = static_cast<uint64_t*>(ticket.data_ptr());
  reserve_sequence_kernel<<<1, 1, 0, stream>>>(
      reinterpret_cast<uint64_t*>(context.local_base + layout.control), sequence_ticket);
  AT_CUDA_CHECK(cudaGetLastError());
  const int requested_blocks =
      static_cast<int>((input_bytes + sizeof(uint4) * kThreads - 1) / (sizeof(uint4) * kThreads));
  const int blocks =
      std::max(1, std::min(requested_blocks, at::cuda::getCurrentDeviceProperties()->multiProcessorCount));
  all_gather_kernel<<<blocks, kThreads, 0, stream>>>(context, sequence_ticket);
  AT_CUDA_CHECK(cudaGetLastError());
}

std::vector<int64_t>
custom_all_gather_status(torch::Tensor& anchor, int64_t local_ptr, int64_t max_bytes, int64_t world_size) {
  check_anchor(anchor);
  TORCH_CHECK(local_ptr != 0, "null local all-gather workspace pointer");
  const auto device_guard = c10::cuda::OptionalCUDAGuard(device_of(anchor));
  const auto stream = c10::cuda::getCurrentCUDAStream().stream();
  const Layout layout = make_layout(max_bytes, world_size);
  int host_status[kStatusWords]{};
  const auto* local_base = reinterpret_cast<const char*>(local_ptr);
  AT_CUDA_CHECK(
      cudaMemcpyAsync(host_status, local_base + layout.status, sizeof(host_status), cudaMemcpyDeviceToHost, stream));
  AT_CUDA_CHECK(cudaStreamSynchronize(stream));
  return {host_status[0], host_status[1], host_status[2], host_status[3]};
}
