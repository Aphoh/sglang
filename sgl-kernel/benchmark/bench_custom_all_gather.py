"""Compare SGLang's peer-copy all-gather with NCCL.

Run with:
  torchrun --standalone --nproc-per-node=2 \
    benchmark/bench_custom_all_gather.py --cudagraph
"""

import argparse
import ctypes
import os

import torch
import torch.distributed as dist

import sgl_kernel.allreduce as custom_ops
from sglang.srt.distributed.device_communicators.cuda_wrapper import CudaRTLibrary


def allocate_ipc_workspace(size: int) -> list[int]:
    runtime = CudaRTLibrary()
    allocation = runtime.cudaMalloc(size)
    handle = runtime.cudaIpcGetMemHandle(allocation)
    handle_bytes = ctypes.string_at(ctypes.addressof(handle), ctypes.sizeof(handle))
    encoded = torch.tensor(
        list(handle_bytes), dtype=torch.uint8, device=torch.cuda.current_device()
    )
    gathered = [torch.empty_like(encoded) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, encoded)

    pointers = []
    for peer, value in enumerate(gathered):
        if peer == dist.get_rank():
            pointers.append(allocation.value)
            continue
        peer_handle = type(handle)()
        payload = bytes(value.cpu().tolist())
        ctypes.memmove(ctypes.addressof(peer_handle), payload, len(payload))
        pointers.append(runtime.cudaIpcOpenMemHandle(peer_handle).value)
    dist.barrier()
    return pointers


def capture(fn):
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph.replay


def benchmark(fn, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    latency_us = start.elapsed_time(end) * 1000 / iterations

    maximum = torch.tensor(latency_us, dtype=torch.float64, device="cuda")
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    return maximum.item()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--elements",
        type=int,
        nargs="+",
        default=[1, 256, 1024, 2560, 4096, 5120, 16384, 65536, 75968, 262144],
    )
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--cudagraph", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size < 2:
        raise RuntimeError("custom all-gather requires at least two ranks")

    dtype = getattr(torch, args.dtype)
    max_bytes = max(args.elements) * dtype.itemsize
    workspace_bytes = custom_ops.custom_all_gather_workspace_size(max_bytes, world_size)
    workspace_ptrs = allocate_ipc_workspace(workspace_bytes)
    anchor = torch.empty(0, device="cuda")
    custom_ops.custom_all_gather_initialize(
        anchor, workspace_ptrs[rank], max_bytes, world_size
    )
    dist.barrier()

    if rank == 0:
        mode = "CUDA graph" if args.cudagraph else "eager"
        print(f"world_size={world_size} dtype={args.dtype} mode={mode}")
        print("elements  bytes/rank  custom_us  nccl_us  custom/nccl")

    for elements in args.elements:
        inp = torch.full((elements,), rank, dtype=dtype, device="cuda")
        custom_out = torch.empty(elements * world_size, dtype=dtype, device="cuda")
        nccl_out = torch.empty_like(custom_out)
        ticket = torch.empty(1, dtype=torch.uint64, device="cuda")

        def custom():
            custom_ops.custom_all_gather(
                inp, custom_out, ticket, workspace_ptrs, rank, max_bytes
            )

        def nccl():
            dist.all_gather_into_tensor(nccl_out, inp)

        if args.cudagraph:
            custom = capture(custom)
            nccl = capture(nccl)

        custom_us = benchmark(custom, args.warmup, args.iterations)
        nccl_us = benchmark(nccl, args.warmup, args.iterations)
        torch.testing.assert_close(custom_out, nccl_out, rtol=0, atol=0)
        if rank == 0:
            print(
                f"{elements:>8}  {elements * dtype.itemsize:>10}  "
                f"{custom_us:>9.2f}  {nccl_us:>7.2f}  {custom_us / nccl_us:>11.2f}x"
            )
        if args.cudagraph:
            del custom, nccl

    status = custom_ops.custom_all_gather_status(
        anchor, workspace_ptrs[rank], max_bytes, world_size
    )
    if status != [0, 0, 0, 0]:
        raise RuntimeError(f"custom all-gather failed: {status}")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
