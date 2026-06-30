import unittest

from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDecodeMigrationServerArgs(unittest.TestCase):
    def test_dp_attention_requires_local_control_broadcast(self):
        with self.assertRaisesRegex(
            ValueError, "enable-dp-attention-local-control-broadcast"
        ):
            ServerArgs(
                model_path="dummy",
                enable_decode_migration=True,
                enable_dp_attention=True,
                dp_size=2,
            )

    def test_dp_attention_allows_local_control_broadcast(self):
        server_args = ServerArgs(
            model_path="dummy",
            enable_decode_migration=True,
            enable_dp_attention=True,
            enable_dp_attention_local_control_broadcast=True,
            dp_size=2,
        )

        self.assertTrue(server_args.enable_decode_migration)


if __name__ == "__main__":
    unittest.main()
