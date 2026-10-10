"""Fused FP4 prefill indexer against the DeepGEMM dense tiles plus top-k on
one SM100 device: 4096 query rows of one request at several cached lengths.

    python benchmark/kernels/attention/bench_fp4_indexer_fused.py --cached 65536 262144
"""

import argparse
import importlib.util
import os
import statistics

import torch

from sglang.srt.layers.attention.dsv4.v41_indexer.scoring import (
    dense_prefill_topk,
    fused_prefill_topk,
)

_TEST = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "..",
    "..",
    "test",
    "registered",
    "kernels",
    "ops",
    "attention",
    "test_fp4_indexer_fused.py",
)


def _load_test_module():
    spec = importlib.util.spec_from_file_location("test_fp4_indexer_fused", _TEST)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def time_cuda(fn, iters: int) -> float:
    """Median wall time of ``fn`` in microseconds."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000)
    return statistics.median(times)


def kernel_split(fn, iters: int) -> list:
    """CUDA time per kernel name in microseconds, averaged over ``iters`` calls
    of ``fn``, largest first."""
    fn()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    per_kernel = {}
    for event in prof.events():
        if event.device_type == torch.autograd.DeviceType.CUDA:
            per_kernel[event.name] = (
                per_kernel.get(event.name, 0.0) + event.device_time / iters
            )
    return sorted(per_kernel.items(), key=lambda item: -item[1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument(
        "--cached", type=int, nargs="+", default=[65536, 131072, 262144]
    )
    parser.add_argument("--ratio", type=int, default=2)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument(
        "--kernels",
        type=int,
        default=0,
        help="also print the CUDA time of the N largest kernels of each path",
    )
    args = parser.parse_args()
    tests = _load_test_module()
    print(f"rows={args.rows} ratio={args.ratio} device={torch.cuda.get_device_name()}")
    print(f"{'cached':>8} {'dense_us':>10} {'fused_us':>10} {'fused/dense':>12}")
    for cached in args.cached:
        case = tests.make_case([(args.rows, cached)], ratio=args.ratio, seed=cached)
        data = case.data
        rows, device = data.num_rows, data.q_fp4.device
        ref = torch.full((rows, tests.TOPK), -1, dtype=torch.int32, device=device)
        got = torch.full((rows, tests.TOPK), -1, dtype=torch.int32, device=device)
        slots = torch.full((rows, tests.TOPK), -1, dtype=torch.int32, device=device)

        def dense():
            dense_prefill_topk(data, case.kv, out=ref)
            data_slots = data.k_slots[(ref + data.request_starts[:, None]).clamp_min(0)]
            torch.where(
                ref >= 0, data_slots.to(torch.int32), slots.new_tensor(-1), out=slots
            )

        def fused():
            fused_prefill_topk(
                data,
                k_cache=case.k_cache,
                kv_page_table=case.kv_page_table,
                page_size=tests.PAGE_SIZE,
                compress_ratio=case.ratio,
                out_positions=got,
                out_slots=slots,
            )

        dense_us = time_cuda(dense, args.iters)
        fused_us = time_cuda(fused, args.iters)
        print(
            f"{cached:>8} {dense_us:>10.0f} {fused_us:>10.0f} {fused_us / dense_us:>12.3f}"
        )
        for label, fn in (("dense", dense), ("fused", fused)):
            for name, us in kernel_split(fn, args.iters)[: args.kernels]:
                print(f"{'':>8} {label} {us:>8.0f} us  {name[:90]}")
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
