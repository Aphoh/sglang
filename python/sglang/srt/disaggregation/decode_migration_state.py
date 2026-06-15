"""Pure state helpers for decode migration frontiers."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DecodeMigrationFrontier:
    committed_input_ids: list[int]
    pending_input_ids: list[int]
    unforwarded_committed_output_ids: list[int]
    prompt_len: int
    committed_len: int
    logical_len: int
    output_tokens_seen: int


def build_decode_migration_frontier(
    prompt_ids: list[int],
    output_ids: list[int],
    committed_len: int,
    output_tokens_seen: int,
) -> DecodeMigrationFrontier:
    """Build the exact committed-KV and sampled-token frontiers.

    The non-speculative prototype requires exactly one sampled token beyond the
    committed KV frontier. ``output_tokens_seen`` is a frontend stream watermark,
    not a KV watermark.
    """
    logical_ids = prompt_ids + output_ids
    logical_len = len(logical_ids)
    if logical_len != committed_len + 1:
        raise ValueError(
            "Expected exactly one sampled token beyond committed KV, got "
            f"logical_len={logical_len}, committed_len={committed_len}"
        )

    prompt_len = len(prompt_ids)
    committed_output_count = max(0, committed_len - prompt_len)
    seen = min(max(0, output_tokens_seen), len(output_ids))
    unforwarded_end = min(committed_output_count, len(output_ids))
    return DecodeMigrationFrontier(
        committed_input_ids=logical_ids[:committed_len],
        pending_input_ids=logical_ids[committed_len:],
        unforwarded_committed_output_ids=output_ids[
            min(seen, unforwarded_end) : unforwarded_end
        ],
        prompt_len=prompt_len,
        committed_len=committed_len,
        logical_len=logical_len,
        output_tokens_seen=seen,
    )
