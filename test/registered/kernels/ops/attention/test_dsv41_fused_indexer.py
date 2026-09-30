"""Correctness tests for the opt-in DeepSeek V4.1 fused indexer."""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

HEADS = 32
HEAD_DIM = 128
PAGE_TOKENS = 128
PAGE_BYTES = 8704
PAYLOAD_BYTES = PAGE_TOKENS * (HEAD_DIM // 2)
TOPK = 512
GROUP = 32
E2M1 = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ]
)
OUTPUT_SENTINEL = 0x55555555


def _dequantize(payload: torch.Tensor, exponents: torch.Tensor) -> torch.Tensor:
    payload = payload.view(torch.uint8)
    nibbles = torch.stack((payload & 0x0F, payload >> 4), dim=-1).reshape(
        payload.shape[0], HEAD_DIM
    )
    values = E2M1.to(payload.device)[nibbles.long()]
    scales = torch.ldexp(
        torch.ones_like(exponents, dtype=torch.float32), exponents.int() - 127
    )
    return values * scales.repeat_interleave(GROUP, dim=1)


def _bf16(value: torch.Tensor) -> torch.Tensor:
    return value.to(torch.bfloat16).to(torch.float32)


def _scores(case: dict[str, torch.Tensor]) -> torch.Tensor:
    rows = case["q_fp4"].shape[0]
    cache = case["k_cache"].reshape(-1, PAGE_BYTES)
    keys = _dequantize(
        cache[:, :PAYLOAD_BYTES].reshape(-1, HEAD_DIM // 2),
        cache[:, PAYLOAD_BYTES:].reshape(-1, 4),
    ).view(cache.shape[0], PAGE_TOKENS, HEAD_DIM)
    queries = _dequantize(
        case["q_fp4"].reshape(-1, HEAD_DIM // 2),
        case["q_sf"].view(torch.uint8).reshape(-1, 4),
    ).view(rows, HEADS, HEAD_DIM)
    width = int(case["lengths"].max())
    scores = torch.full((rows, width), -1.0, device=cache.device)
    for row in range(rows):
        length = int(case["lengths"][row])
        if not length:
            continue
        page_count = (length + PAGE_TOKENS - 1) // PAGE_TOKENS
        physical = case["page_table"][row, :page_count].long()
        row_keys = keys[physical].reshape(-1, HEAD_DIM)[:length]
        dots = (queries[row].double() @ row_keys.double().T).float().clamp_min(0)
        weights = _bf16(case["weights"][row])
        products = _bf16(_bf16(dots) * weights[:, None])
        total = torch.zeros(length, device=row_keys.device, dtype=torch.float32)
        for head in range(HEADS):
            total = total + products[head]
        scores[row, :length] = _bf16(total)
    return scores


def _reference(
    case: dict[str, torch.Tensor], scores: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = scores.shape[0]
    raw = torch.full((rows, TOPK), -1, dtype=torch.int32, device=scores.device)
    mapped = torch.full_like(raw, -1)
    for row in range(rows):
        length = int(case["lengths"][row])
        eligible = torch.nonzero(scores[row, :length] >= 0).flatten()
        order = torch.argsort(scores[row, eligible], descending=True, stable=True)
        tokens = torch.sort(eligible[order[:TOPK]]).values
        count = tokens.numel()
        raw[row, :count] = tokens.to(torch.int32)
        mapped[row, :count] = case["page_table"][
            row, tokens // PAGE_TOKENS
        ] * PAGE_TOKENS + tokens.remainder(PAGE_TOKENS).to(torch.int32)
    return raw, mapped


def _bf16_steps(scores: torch.Tensor) -> torch.Tensor:
    bits = (scores.float().view(torch.int32) >> 16).long()
    return torch.where(bits < 0, -(bits & 0x7FFF), bits)


def _check_row(
    want: torch.Tensor, got: torch.Tensor, scores: torch.Tensor, score_steps: int = 1
) -> str:
    want, got = want[want >= 0], got[got != -1]
    if got.numel() != want.numel():
        return f"selected {got.numel()} tokens, expected {want.numel()}"
    if got.numel() == 0:
        return ""
    if (
        (got < 0).any()
        or (got >= scores.numel()).any()
        or (scores[got.long()] < 0).any()
    ):
        return "a selected entry is not a token below the row length"
    if got.numel() > 1 and (got[1:] <= got[:-1]).any():
        return "the selected tokens are not in increasing order"
    extra = got[~torch.isin(got, want)]
    missing = want[~torch.isin(want, got)]
    if extra.numel() == 0:
        return ""
    low = scores[extra.long()].min()
    high = scores[missing.long()].max()
    if _bf16_steps(high) - _bf16_steps(low) > score_steps:
        return (
            f"selected score {low.item():g} instead of "
            f"{high.item():g} at the top-k boundary"
        )
    if (scores[extra.long()] == high).any() and extra[
        scores[extra.long()] == high
    ].min() > missing[scores[missing.long()] == high].min():
        return "a tie went to the higher token"
    return ""


def _make_case(
    rows: int, max_length: int, seed: int, *, ties: bool = False
) -> dict[str, torch.Tensor]:
    from sglang.kernels.ops.attention.dsv4.fp4_indexer import (
        quantize_fp4_indexer_tensor,
    )

    torch.manual_seed(seed)
    device = torch.device("cuda")
    page_count = (max_length + PAGE_TOKENS - 1) // PAGE_TOKENS
    pool_pages = rows * page_count + 3
    if max_length == 600:
        lengths = torch.tensor(
            [0, 1, 73, 511, 512, 513, 599, 600], dtype=torch.int32, device=device
        )
    elif rows == 1 or ties:
        lengths = torch.full((rows,), max_length, dtype=torch.int32, device=device)
    else:
        lengths = torch.linspace(
            max(512, max_length // 3), max_length, rows, device=device
        ).to(torch.int32)

    page_table = torch.stack(
        [torch.randperm(pool_pages - 3)[:page_count] for _ in range(rows)]
    ).to(device=device, dtype=torch.int32)
    k_fp4, k_sf = quantize_fp4_indexer_tensor(
        torch.randn(
            pool_pages * PAGE_TOKENS,
            HEAD_DIM,
            device=device,
            dtype=torch.bfloat16,
        ),
        rne=True,
    )
    k_cache = torch.cat(
        [
            k_fp4.view(torch.uint8).reshape(pool_pages, PAGE_TOKENS * 64),
            k_sf.view(torch.uint8).reshape(pool_pages, PAGE_TOKENS * 4),
        ],
        dim=1,
    ).view(pool_pages, PAGE_TOKENS, 1, 68)
    if ties:
        one_token = torch.randint(
            0, 256, (HEAD_DIM // 2,), device=device, dtype=torch.uint8
        )
        k_cache.view(pool_pages, PAGE_BYTES)[:, :PAYLOAD_BYTES] = one_token.repeat(
            PAGE_TOKENS
        )
        k_cache.view(pool_pages, PAGE_BYTES)[:, PAYLOAD_BYTES:] = 127

    q_fp4, q_sf = quantize_fp4_indexer_tensor(
        torch.randn(rows * HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16),
        rne=True,
    )
    q_fp4 = q_fp4.view(rows, HEADS, HEAD_DIM // 2)
    q_sf = q_sf.view(rows, HEADS)
    weights = torch.rand(rows, HEADS, device=device)
    return {
        "q_fp4": q_fp4,
        "q_sf": q_sf,
        "weights": weights,
        "k_cache": k_cache,
        "lengths": lengths,
        "page_table": page_table,
    }


def _run_kernel(case: dict[str, torch.Tensor], with_raw: bool):
    from sglang.kernels.ops.attention.dsv4.fused_indexer_sm100 import (
        fused_indexer_topk,
    )

    page_indices, raw_indices = _output_buffers(case, with_raw)
    fused_indexer_topk(
        case["q_fp4"],
        case["q_sf"],
        case["weights"],
        case["k_cache"].view(-1, PAGE_BYTES),
        case["lengths"],
        case["page_table"],
        page_indices,
        raw_indices,
    )
    return page_indices, raw_indices


def _output_buffers(case: dict[str, torch.Tensor], with_raw: bool):
    rows = case["q_fp4"].shape[0]
    output_width = TOPK + 19
    page_indices = torch.full(
        (rows, output_width),
        OUTPUT_SENTINEL,
        dtype=torch.int32,
        device=case["q_fp4"].device,
    )
    raw_indices = torch.full_like(page_indices, OUTPUT_SENTINEL) if with_raw else None
    return page_indices, raw_indices


def _raw_from_pages(
    case: dict[str, torch.Tensor], page_indices: torch.Tensor
) -> torch.Tensor:
    raw = torch.full_like(page_indices[:, :TOPK], -1)
    for row in range(page_indices.shape[0]):
        inverse = torch.full(
            (case["k_cache"].shape[0],),
            -1,
            dtype=torch.int32,
            device=page_indices.device,
        )
        inverse[case["page_table"][row].long()] = torch.arange(
            case["page_table"].shape[1], device=page_indices.device
        )
        valid = page_indices[row, :TOPK] >= 0
        mapped = page_indices[row, :TOPK][valid]
        raw[row, valid] = inverse[
            mapped.long() // PAGE_TOKENS
        ] * PAGE_TOKENS + mapped.remainder(PAGE_TOKENS)
    return raw


def _assert_case(
    test: CustomTestCase,
    case: dict[str, torch.Tensor],
    page_indices: torch.Tensor,
    raw_indices: torch.Tensor | None,
    scores: torch.Tensor,
    expected_raw: torch.Tensor,
    *,
    ties: bool = False,
) -> None:
    inferred_raw = _raw_from_pages(case, page_indices)
    if raw_indices is not None:
        test.assertTrue(
            torch.equal(raw_indices[:, :TOPK], inferred_raw),
            "raw positions do not correspond to the mapped page indices",
        )
        test.assertTrue(
            torch.equal(
                raw_indices[:, TOPK:],
                torch.full_like(raw_indices[:, TOPK:], OUTPUT_SENTINEL),
            )
        )
        got_raw = raw_indices[:, :TOPK]
    else:
        got_raw = inferred_raw
    test.assertTrue(
        torch.equal(
            page_indices[:, TOPK:],
            torch.full_like(page_indices[:, TOPK:], OUTPUT_SENTINEL),
        ),
        "columns after top-512 were modified",
    )
    for row in range(page_indices.shape[0]):
        error = _check_row(expected_raw[row], got_raw[row], scores[row])
        test.assertEqual(error, "", f"row {row}: {error}")
    if ties:
        expected_tied_positions = torch.arange(TOPK, device=page_indices.device)
        test.assertTrue(
            torch.equal(got_raw, expected_tied_positions.expand_as(got_raw))
        )
    selected = got_raw >= 0
    remapped = torch.full_like(page_indices[:, :TOPK], -1)
    for row in range(page_indices.shape[0]):
        positions = got_raw[row, selected[row]].long()
        remapped[row, selected[row]] = case["page_table"][
            row, positions // PAGE_TOKENS
        ] * PAGE_TOKENS + positions.remainder(PAGE_TOKENS).to(torch.int32)
    test.assertTrue(torch.equal(page_indices[:, :TOPK], remapped))


class TestDSV41FusedIndexerSplits(CustomTestCase):
    def test_rhino_split_rule(self):
        from sglang.kernels.ops.attention.dsv4.fused_indexer_sm100 import (
            fused_indexer_splits,
        )

        cases = [
            (1, 512, 148),
            (4, 64, 37),
            (4, 5, 5),
            (4096, 8192, 1),
        ]
        for rows, page_count, expected in cases:
            with self.subTest(rows=rows, page_count=page_count):
                self.assertEqual(fused_indexer_splits(rows, page_count, 148), expected)


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10,
    "the fused indexer needs SM100",
)
class TestDSV41FusedIndexer(CustomTestCase):
    @torch.inference_mode()
    def test_correctness_and_page_mapping(self):
        cases = [
            (4, 8192, False),
            (1, 65536, False),
            (8, 600, False),
            (256, 4096, False),
            (4, 4096, True),
        ]
        for rows, max_length, ties in cases:
            case = _make_case(rows, max_length, seed=rows + max_length, ties=ties)
            scores = _scores(case)
            expected_raw, _ = _reference(case, scores)
            for with_raw in (False, True):
                with self.subTest(
                    rows=rows, max_length=max_length, ties=ties, with_raw=with_raw
                ):
                    page_indices, raw_indices = _run_kernel(case, with_raw)
                    _assert_case(
                        self,
                        case,
                        page_indices,
                        raw_indices,
                        scores,
                        expected_raw,
                        ties=ties,
                    )

    @torch.inference_mode()
    def test_cuda_graph_capture_and_replay(self):
        from sglang.kernels.ops.attention.dsv4.fused_indexer_sm100 import (
            fused_indexer_topk,
        )

        case = _make_case(4, 8192, seed=17)
        scores = _scores(case)
        expected_raw, _ = _reference(case, scores)
        page_indices, raw_indices = _output_buffers(case, with_raw=True)
        inputs = (
            case["q_fp4"],
            case["q_sf"],
            case["weights"],
            case["k_cache"].view(-1, PAGE_BYTES),
            case["lengths"],
            case["page_table"],
        )
        fused_indexer_topk(*inputs, page_indices, raw_indices)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fused_indexer_topk(*inputs, page_indices, raw_indices)
        graph.replay()
        _assert_case(
            self,
            case,
            page_indices,
            raw_indices,
            scores,
            expected_raw,
        )


if __name__ == "__main__":
    unittest.main()
