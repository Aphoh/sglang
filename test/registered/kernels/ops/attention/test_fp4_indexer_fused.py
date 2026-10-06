"""The fused FP4 prefill indexer against the DeepGEMM dense tiles plus top-k:
per row the selected positions, their pool slots, and a candidate source's block
ids as sets, equal except where the scores at the selection floor tie exactly."""

import math
import unittest
from typing import List, NamedTuple, Tuple

import torch

from sglang.kernels.ops.attention.dsv4 import topk_transform_ragged_v2
from sglang.kernels.ops.attention.dsv4.candidate_blocks import (
    amax8_varlen,
    candidate_row_lens,
)
from sglang.kernels.ops.attention.dsv4.candidate_table import CANDIDATE_BLOCK_SIZE
from sglang.kernels.ops.attention.dsv4.fp4_indexer import quantize_fp4_indexer_tensor
from sglang.kernels.ops.attention.dsv4.fp4_indexer_fused import (
    CANDIDATE_BLOCKS,
    INDEX_PAGE_SLOTS,
    TOPK,
    fused_indexer_available,
)
from sglang.srt.layers.attention.dsv4.metadata import expand_index_page_table
from sglang.srt.layers.attention.dsv4.v41_indexer.scoring import (
    DeepGEMMPrefillData,
    dense_prefill_topk,
    fused_prefill_topk,
    score_tiles,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=400, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

HEADS, DIM = 32, 128
PAGE_SIZE = 256  # KV page; the index-K pool pages at INDEX_PAGE_SLOTS positions
PREFILL_ROWS, PREFILL_CACHED = 4096, 131072


class Case(NamedTuple):
    data: DeepGEMMPrefillData
    kv: Tuple[torch.Tensor, torch.Tensor]  # the flattened index K, column order
    k_cache: torch.Tensor  # [pages, INDEX_PAGE_SLOTS * 68] uint8, the index-K pool
    kv_page_table: torch.Tensor  # [rows, kv pages] int32
    ratio: int


def make_case(requests: List[Tuple[int, int]], ratio: int, seed: int) -> Case:
    """``requests`` as (query rows, cached tokens): a request's rows are its
    newest tokens. Its K sits in randomly placed pool pages, as in serving."""
    torch.manual_seed(seed)
    dev = "cuda"
    slots_per_kv_page = PAGE_SIZE // ratio
    lens_per_request = [(cached + rows) // ratio for rows, cached in requests]
    rows_per_request = [rows for rows, _ in requests]
    kv_pages_per_request = [
        math.ceil((cached + rows) / PAGE_SIZE) for rows, cached in requests
    ]
    total_kv_pages = sum(kv_pages_per_request)
    perm = torch.randperm(total_kv_pages, device=dev).to(torch.int32)
    width = max(kv_pages_per_request)
    request_table = torch.zeros(len(requests), width, dtype=torch.int32, device=dev)
    offset = 0
    for r, n in enumerate(kv_pages_per_request):
        request_table[r, :n] = perm[offset : offset + n]
        offset += n
    index_table = expand_index_page_table(
        request_table,
        full_page_size=PAGE_SIZE,
        compress_ratio=ratio,
        index_page_size=INDEX_PAGE_SLOTS,
    )
    columns = sum(lens_per_request)
    k_fp4, k_sf = quantize_fp4_indexer_tensor(
        torch.randn(columns, DIM, device=dev, dtype=torch.bfloat16), rne=True
    )
    request_of_column = torch.repeat_interleave(
        torch.arange(len(requests), device=dev),
        torch.tensor(lens_per_request, device=dev),
    )
    starts = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(lens_per_request), 0)[:-1]),
        device=dev,
    )
    position = torch.arange(columns, device=dev) - starts[request_of_column]
    k_slots = (
        index_table[request_of_column, position // INDEX_PAGE_SLOTS].to(torch.int64)
        * INDEX_PAGE_SLOTS
        + position % INDEX_PAGE_SLOTS
    )
    n_pages = total_kv_pages * (slots_per_kv_page // INDEX_PAGE_SLOTS) + 3
    payload = torch.zeros(n_pages, INDEX_PAGE_SLOTS, 64, dtype=torch.uint8, device=dev)
    scales = torch.zeros(n_pages, INDEX_PAGE_SLOTS, dtype=torch.int32, device=dev)
    payload[k_slots // INDEX_PAGE_SLOTS, k_slots % INDEX_PAGE_SLOTS] = k_fp4.view(
        torch.uint8
    )
    scales[k_slots // INDEX_PAGE_SLOTS, k_slots % INDEX_PAGE_SLOTS] = k_sf
    k_cache = torch.cat(
        [payload.reshape(n_pages, -1), scales.view(torch.uint8).reshape(n_pages, -1)],
        1,
    )
    rows = sum(rows_per_request)
    positions = torch.cat(
        [torch.arange(cached, cached + n, device=dev) for (n, cached) in requests]
    )
    q_fp4, q_sf = quantize_fp4_indexer_tensor(
        torch.randn(rows * HEADS, DIM, device=dev, dtype=torch.bfloat16), rne=True
    )
    rows_dev = torch.tensor(rows_per_request, device=dev)
    return Case(
        data=DeepGEMMPrefillData(
            k_slots=k_slots,
            request_starts=torch.repeat_interleave(starts, rows_dev).to(torch.int32),
            lens_per_request=lens_per_request,
            rows_per_request=rows_per_request,
            compress_lens=((positions + 1) // ratio).to(torch.int32),
            q_fp4=q_fp4.view(rows, HEADS, 64),
            q_sf=q_sf.view(rows, HEADS),
            weights=torch.rand(rows, HEADS, device=dev),
        ),
        kv=(k_fp4, k_sf),
        k_cache=k_cache,
        kv_page_table=torch.repeat_interleave(request_table, rows_dev, dim=0),
        ratio=ratio,
    )


def row_scores(case: Case, row: int) -> torch.Tensor:
    """fp32 scores of one row against its request's compressed positions."""
    d = case.data
    one = DeepGEMMPrefillData(
        k_slots=d.k_slots,
        request_starts=d.request_starts[row : row + 1],
        lens_per_request=[int(d.compress_lens[row])],
        rows_per_request=[1],
        compress_lens=d.compress_lens[row : row + 1],
        q_fp4=d.q_fp4[row : row + 1],
        q_sf=d.q_sf[row : row + 1],
        weights=d.weights[row : row + 1],
    )
    (_, logits), *_ = score_tiles(one, case.kv, width_align=8)
    return logits[0, : int(d.compress_lens[row])]


def block_scores(scores: torch.Tensor) -> torch.Tensor:
    """The candidate key of each block: its max score, +inf for the newest."""
    n = scores.numel()
    padded = torch.full(
        (
            (n + CANDIDATE_BLOCK_SIZE - 1)
            // CANDIDATE_BLOCK_SIZE
            * CANDIDATE_BLOCK_SIZE,
        ),
        float("-inf"),
        device=scores.device,
    )
    padded[:n] = scores
    keys = padded.view(-1, CANDIDATE_BLOCK_SIZE).amax(1)
    keys[-1] = float("inf")
    return keys


def reference_blocks(case: Case, topk_blocks: int) -> torch.Tensor:
    """The candidate source's block selection on the DeepGEMM path."""
    data = case.data
    rows, device = data.num_rows, data.q_fp4.device
    nblocks, _ = candidate_row_lens(data.compress_lens, topk_blocks)
    blocks = torch.full((rows, topk_blocks), -1, dtype=torch.int32, device=device)
    zero = torch.zeros(rows, dtype=torch.int32, device=device)
    for tile, logits in score_tiles(data, case.kv, width_align=8):
        width_blocks = (
            logits.shape[1] + CANDIDATE_BLOCK_SIZE - 1
        ) // CANDIDATE_BLOCK_SIZE
        keys = logits.new_empty(logits.shape[0], (width_blocks + 3) // 4 * 4)
        amax8_varlen(logits, data.compress_lens[tile], out=keys)
        topk_transform_ragged_v2(
            keys, nblocks[tile], out_offsets=zero[tile], out_indices=blocks[tile]
        )
        del logits
    return blocks


class Comparison(NamedTuple):
    rows_differing: int
    ties: int  # positions swapped between the two selections, all at one score


def compare_selections(
    ref: torch.Tensor,
    got: torch.Tensor,
    counts: torch.Tensor,
    scores_of,
    test: unittest.TestCase,
) -> Comparison:
    """Both ``[rows, k]`` int32, -1 padded, unordered: the same count of picks
    per row and the same sets, except for swaps between positions whose scores
    round to the same bf16 value (the fused keys are bf16)."""
    test.assertEqual(ref.shape, got.shape)
    test.assertTrue(torch.equal((ref >= 0).sum(1), counts), "pick counts")
    test.assertTrue(torch.equal((got >= 0).sum(1), counts), "fused pick counts")
    ref_sorted, got_sorted = ref.sort(1).values, got.sort(1).values
    differing = (ref_sorted != got_sorted).any(1).nonzero().flatten().tolist()
    ties = 0
    for row in differing:
        a = set(ref[row].tolist()) - {-1}
        b = set(got[row].tolist()) - {-1}
        only_ref, only_got = a - b, b - a
        test.assertEqual(len(only_ref), len(only_got), f"row {row}")
        scores = scores_of(row).to(torch.bfloat16)
        swapped = scores[
            torch.tensor(sorted(only_ref | only_got), device=scores.device)
        ]
        test.assertTrue(
            bool((swapped == swapped[0]).all()),
            f"row {row}: {len(only_ref)} picks differ at scores {swapped.tolist()}",
        )
        ties += len(only_ref)
    return Comparison(len(differing), ties)


@unittest.skipUnless(
    torch.cuda.is_available() and fused_indexer_available(),
    "the fused indexer needs an SM100 device",
)
class TestFusedIndexer(CustomTestCase):
    def run_full(self, case: Case, topk: int = TOPK) -> Comparison:
        data = case.data
        rows, device = data.num_rows, data.q_fp4.device
        ref = torch.full((rows, topk), -1, dtype=torch.int32, device=device)
        dense_prefill_topk(data, case.kv, out=ref)
        got = torch.full((rows, topk), -1, dtype=torch.int32, device=device)
        slots = torch.full((rows, topk), -1, dtype=torch.int32, device=device)
        fused_prefill_topk(
            data,
            k_cache=case.k_cache,
            kv_page_table=case.kv_page_table,
            page_size=PAGE_SIZE,
            compress_ratio=case.ratio,
            out_positions=got,
            out_slots=slots,
        )
        torch.cuda.synchronize()
        counts = data.compress_lens.clamp_max(topk)
        # the slot output is the position output through the pool's slot map
        chosen = got >= 0
        expected = torch.where(
            chosen,
            data.k_slots[(got + data.request_starts[:, None]).clamp_min(0)].to(
                torch.int32
            ),
            -1,
        )
        self.assertTrue(torch.equal(slots, expected), "pool slots of the picks")
        return compare_selections(
            ref, got, counts, lambda row: row_scores(case, row), self
        )

    def run_candidates(self, case: Case) -> Tuple[Comparison, Comparison]:
        data = case.data
        rows, device = data.num_rows, data.q_fp4.device
        ref_pos = torch.full((rows, TOPK), -1, dtype=torch.int32, device=device)
        dense_prefill_topk(data, case.kv, out=ref_pos)
        ref_blocks = reference_blocks(case, CANDIDATE_BLOCKS)
        got_pos = torch.full((rows, TOPK), -1, dtype=torch.int32, device=device)
        got_blocks = fused_prefill_topk(
            data,
            k_cache=case.k_cache,
            kv_page_table=case.kv_page_table,
            page_size=PAGE_SIZE,
            compress_ratio=case.ratio,
            out_positions=got_pos,
            topk_blocks=CANDIDATE_BLOCKS,
        )
        torch.cuda.synchronize()
        nblocks, _ = candidate_row_lens(data.compress_lens, CANDIDATE_BLOCKS)
        positions = compare_selections(
            ref_pos,
            got_pos,
            data.compress_lens.clamp_max(TOPK),
            lambda row: row_scores(case, row),
            self,
        )
        blocks = compare_selections(
            ref_blocks,
            got_blocks,
            nblocks.clamp_max(CANDIDATE_BLOCKS),
            lambda row: block_scores(row_scores(case, row)),
            self,
        )
        return positions, blocks

    def test_prefill_ratio_2(self):
        case = make_case([(PREFILL_ROWS, PREFILL_CACHED)], ratio=2, seed=1)
        c = self.run_full(case)
        print(f"ratio 2: {c.rows_differing} rows differ on {c.ties} tied picks")

    def test_prefill_ratio_1(self):
        case = make_case([(PREFILL_ROWS, PREFILL_CACHED)], ratio=1, seed=2)
        c = self.run_full(case)
        print(f"ratio 1: {c.rows_differing} rows differ on {c.ties} tied picks")

    def test_mixed_requests(self):
        # rows not a multiple of the CTA's four, a request that fits one page,
        # and rows that see fewer positions than the top-k
        case = make_case([(7, 3000), (301, 20000), (6, 0), (130, 200)], ratio=2, seed=3)
        c = self.run_full(case)
        print(f"mixed: {c.rows_differing} rows differ on {c.ties} tied picks")

    def test_candidate_source(self):
        case = make_case([(PREFILL_ROWS, PREFILL_CACHED)], ratio=2, seed=4)
        positions, blocks = self.run_candidates(case)
        print(
            f"candidates: positions {positions.rows_differing} rows / "
            f"{positions.ties} ties, blocks {blocks.rows_differing} rows / "
            f"{blocks.ties} ties"
        )

    def test_candidate_source_short_context(self):
        # fewer blocks than the budget: every block is a candidate
        case = make_case([(512, 12000), (64, 500)], ratio=1, seed=5)
        positions, blocks = self.run_candidates(case)
        print(
            f"short candidates: positions {positions.rows_differing} rows / "
            f"{positions.ties} ties, blocks {blocks.rows_differing} rows / "
            f"{blocks.ties} ties"
        )


if __name__ == "__main__":
    unittest.main()
