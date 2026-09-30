def fused_indexer_splits(rows: int, page_count: int, sm_count: int) -> int:
    block_q = 1 if rows <= 128 else 4
    ctas = max((rows + block_q - 1) // block_q, 1)
    return min(max(sm_count // ctas, 1), max(page_count, 1))
