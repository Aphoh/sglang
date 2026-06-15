import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).parents[3]
    / "python/sglang/srt/disaggregation/decode_migration_state.py"
)
SPEC = importlib.util.spec_from_file_location("decode_migration_state", MODULE_PATH)
STATE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(STATE)


class DecodeMigrationFrontierTests(unittest.TestCase):
    def test_streamed_pending_token_is_not_replayed_as_committed_tail(self):
        frontier = STATE.build_decode_migration_frontier(
            prompt_ids=[10, 11],
            output_ids=[20, 21, 22, 23],
            committed_len=5,
            output_tokens_seen=4,
        )
        self.assertEqual(frontier.committed_input_ids, [10, 11, 20, 21, 22])
        self.assertEqual(frontier.pending_input_ids, [23])
        self.assertEqual(frontier.unforwarded_committed_output_ids, [])

    def test_stream_interval_tail_is_returned_from_committed_range(self):
        frontier = STATE.build_decode_migration_frontier(
            prompt_ids=[10, 11],
            output_ids=[20, 21, 22, 23],
            committed_len=5,
            output_tokens_seen=1,
        )
        self.assertEqual(frontier.unforwarded_committed_output_ids, [21, 22])
        self.assertEqual(frontier.pending_input_ids, [23])

    def test_frontend_watermark_is_clamped(self):
        frontier = STATE.build_decode_migration_frontier(
            prompt_ids=[10],
            output_ids=[20, 21],
            committed_len=2,
            output_tokens_seen=100,
        )
        self.assertEqual(frontier.output_tokens_seen, 2)
        self.assertEqual(frontier.unforwarded_committed_output_ids, [])

    def test_rejects_ambiguous_kv_frontier(self):
        with self.assertRaisesRegex(ValueError, "exactly one sampled token"):
            STATE.build_decode_migration_frontier(
                prompt_ids=[10, 11],
                output_ids=[20, 21, 22],
                committed_len=3,
                output_tokens_seen=0,
            )


if __name__ == "__main__":
    unittest.main()
