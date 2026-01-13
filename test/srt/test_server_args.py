import json
import unittest
from unittest.mock import patch

from sglang.srt.server_args import PortArgs, ServerArgs, prepare_server_args
from sglang.test.test_utils import CustomTestCase


class TestPrepareServerArgs(CustomTestCase):
    def test_prepare_server_args(self):
        server_args = prepare_server_args(
            [
                "--model-path",
                "meta-llama/Meta-Llama-3.1-8B-Instruct",
                "--json-model-override-args",
                '{"rope_scaling": {"factor": 2.0, "rope_type": "linear"}}',
            ]
        )
        self.assertEqual(
            server_args.model_path, "meta-llama/Meta-Llama-3.1-8B-Instruct"
        )
        self.assertEqual(
            json.loads(server_args.json_model_override_args),
            {"rope_scaling": {"factor": 2.0, "rope_type": "linear"}},
        )


class TestPortArgs(unittest.TestCase):
    @patch("sglang.srt.server_args.is_port_available")
    @patch("sglang.srt.server_args.tempfile.NamedTemporaryFile")
    def test_init_new_standard_case(self, mock_temp_file, mock_is_port_available):
        mock_is_port_available.return_value = True
        mock_temp_file.return_value.name = "temp_file"

        server_args = ServerArgs(model_path="dummy")
        server_args.port = 30000
        server_args.nccl_port = None
        server_args.enable_dp_attention = False

        port_args = PortArgs.init_new(server_args)

        self.assertTrue(port_args.tokenizer_ipc_name.startswith("ipc://"))
        self.assertTrue(port_args.scheduler_input_ipc_name.startswith("ipc://"))
        self.assertTrue(port_args.detokenizer_ipc_name.startswith("ipc://"))
        self.assertIsInstance(port_args.nccl_port, int)

    @patch("sglang.srt.server_args.is_port_available")
    def test_init_new_with_single_node_dp_attention(self, mock_is_port_available):
        mock_is_port_available.return_value = True

        server_args = ServerArgs(model_path="dummy")
        server_args.port = 30000
        server_args.nccl_port = None
        server_args.enable_dp_attention = True
        server_args.nnodes = 1
        server_args.dist_init_addr = None

        port_args = PortArgs.init_new(server_args)

        self.assertTrue(port_args.tokenizer_ipc_name.startswith("tcp://127.0.0.1:"))
        self.assertTrue(
            port_args.scheduler_input_ipc_name.startswith("tcp://127.0.0.1:")
        )
        self.assertTrue(port_args.detokenizer_ipc_name.startswith("tcp://127.0.0.1:"))
        self.assertIsInstance(port_args.nccl_port, int)

    @patch("sglang.srt.server_args.is_port_available")
    def test_init_new_with_dp_rank(self, mock_is_port_available):
        mock_is_port_available.return_value = True

        server_args = ServerArgs(model_path="dummy")
        server_args.port = 30000
        server_args.nccl_port = None
        server_args.enable_dp_attention = True
        server_args.nnodes = 1
        server_args.dist_init_addr = "192.168.1.1:25000"

        worker_ports = [25006, 25007, 25008, 25009]
        port_args = PortArgs.init_new(server_args, dp_rank=2, worker_ports=worker_ports)

        self.assertTrue(port_args.scheduler_input_ipc_name.endswith(":25008"))

        self.assertTrue(port_args.tokenizer_ipc_name.startswith("tcp://192.168.1.1:"))
        self.assertTrue(port_args.detokenizer_ipc_name.startswith("tcp://192.168.1.1:"))
        self.assertIsInstance(port_args.nccl_port, int)

    @patch("sglang.srt.server_args.is_port_available")
    def test_init_new_with_ipv4_address(self, mock_is_port_available):
        mock_is_port_available.return_value = True

        server_args = ServerArgs(model_path="dummy")
        server_args.port = 30000

        server_args.nccl_port = None

        server_args.enable_dp_attention = True
        server_args.nnodes = 2
        server_args.dist_init_addr = "192.168.1.1:25000"

        port_args = PortArgs.init_new(server_args)

        self.assertTrue(port_args.tokenizer_ipc_name.startswith("tcp://192.168.1.1:"))
        self.assertTrue(
            port_args.scheduler_input_ipc_name.startswith("tcp://192.168.1.1:")
        )
        self.assertTrue(port_args.detokenizer_ipc_name.startswith("tcp://192.168.1.1:"))
        self.assertIsInstance(port_args.nccl_port, int)

    @patch("sglang.srt.server_args.is_port_available")
    def test_init_new_with_malformed_ipv4_address(self, mock_is_port_available):
        mock_is_port_available.return_value = True

        server_args = ServerArgs(model_path="dummy")
        server_args.port = 30000
        server_args.nccl_port = None

        server_args.enable_dp_attention = True
        server_args.nnodes = 2
        server_args.dist_init_addr = "192.168.1.1"

        with self.assertRaises(AssertionError) as context:
            PortArgs.init_new(server_args)

        self.assertIn(
            "please provide --dist-init-addr as host:port", str(context.exception)
        )

    @patch("sglang.srt.server_args.is_port_available")
    def test_init_new_with_malformed_ipv4_address_invalid_port(
        self, mock_is_port_available
    ):
        mock_is_port_available.return_value = True

        server_args = ServerArgs(model_path="dummy")
        server_args.port = 30000
        server_args.nccl_port = None

        server_args.enable_dp_attention = True
        server_args.nnodes = 2
        server_args.dist_init_addr = "192.168.1.1:abc"

        with self.assertRaises(ValueError) as context:
            PortArgs.init_new(server_args)

    @patch("sglang.srt.server_args.is_port_available")
    @patch("sglang.srt.server_args.is_valid_ipv6_address", return_value=True)
    def test_init_new_with_ipv6_address(
        self, mock_is_valid_ipv6, mock_is_port_available
    ):
        mock_is_port_available.return_value = True

        server_args = ServerArgs(model_path="dummy")
        server_args.port = 30000
        server_args.nccl_port = None

        server_args.enable_dp_attention = True
        server_args.nnodes = 2
        server_args.dist_init_addr = "[2001:db8::1]:25000"

        port_args = PortArgs.init_new(server_args)

        self.assertTrue(port_args.tokenizer_ipc_name.startswith("tcp://[2001:db8::1]:"))
        self.assertTrue(
            port_args.scheduler_input_ipc_name.startswith("tcp://[2001:db8::1]:")
        )
        self.assertTrue(
            port_args.detokenizer_ipc_name.startswith("tcp://[2001:db8::1]:")
        )
        self.assertIsInstance(port_args.nccl_port, int)

    @patch("sglang.srt.server_args.is_port_available")
    @patch("sglang.srt.server_args.is_valid_ipv6_address", return_value=False)
    def test_init_new_with_invalid_ipv6_address(
        self, mock_is_valid_ipv6, mock_is_port_available
    ):
        mock_is_port_available.return_value = True

        server_args = ServerArgs(model_path="dummy")
        server_args.port = 30000
        server_args.nccl_port = None

        server_args.enable_dp_attention = True
        server_args.nnodes = 2
        server_args.dist_init_addr = "[invalid-ipv6]:25000"

        with self.assertRaises(ValueError) as context:
            PortArgs.init_new(server_args)

        self.assertIn("invalid IPv6 address", str(context.exception))

    @patch("sglang.srt.server_args.is_port_available")
    def test_init_new_with_malformed_ipv6_address_missing_bracket(
        self, mock_is_port_available
    ):
        mock_is_port_available.return_value = True

        server_args = ServerArgs(model_path="dummy")
        server_args.port = 30000
        server_args.nccl_port = None

        server_args.enable_dp_attention = True
        server_args.nnodes = 2
        server_args.dist_init_addr = "[2001:db8::1:25000"

        with self.assertRaises(ValueError) as context:
            PortArgs.init_new(server_args)

        self.assertIn("invalid IPv6 address format", str(context.exception))

    @patch("sglang.srt.server_args.is_port_available")
    @patch("sglang.srt.server_args.is_valid_ipv6_address", return_value=True)
    def test_init_new_with_malformed_ipv6_address_missing_port(
        self, mock_is_valid_ipv6, mock_is_port_available
    ):
        mock_is_port_available.return_value = True

        server_args = ServerArgs(model_path="dummy")
        server_args.port = 30000
        server_args.nccl_port = None

        server_args.enable_dp_attention = True
        server_args.nnodes = 2
        server_args.dist_init_addr = "[2001:db8::1]"

        with self.assertRaises(ValueError) as context:
            PortArgs.init_new(server_args)

        self.assertIn(
            "a port must be specified in IPv6 address", str(context.exception)
        )

    @patch("sglang.srt.server_args.is_port_available")
    @patch("sglang.srt.server_args.is_valid_ipv6_address", return_value=True)
    def test_init_new_with_malformed_ipv6_address_invalid_port(
        self, mock_is_valid_ipv6, mock_is_port_available
    ):
        mock_is_port_available.return_value = True

        server_args = ServerArgs(model_path="dummy")
        server_args.port = 30000
        server_args.nccl_port = None

        server_args.enable_dp_attention = True
        server_args.nnodes = 2
        server_args.dist_init_addr = "[2001:db8::1]:abcde"

        with self.assertRaises(ValueError) as context:
            PortArgs.init_new(server_args)

        self.assertIn("invalid port in IPv6 address", str(context.exception))

    @patch("sglang.srt.server_args.is_port_available")
    @patch("sglang.srt.server_args.is_valid_ipv6_address", return_value=True)
    def test_init_new_with_malformed_ipv6_address_wrong_separator(
        self, mock_is_valid_ipv6, mock_is_port_available
    ):
        mock_is_port_available.return_value = True

        server_args = ServerArgs(model_path="dummy")
        server_args.port = 30000
        server_args.nccl_port = None

        server_args.enable_dp_attention = True
        server_args.nnodes = 2
        server_args.dist_init_addr = "[2001:db8::1]#25000"

        with self.assertRaises(ValueError) as context:
            PortArgs.init_new(server_args)

        self.assertIn("expected ':' after ']'", str(context.exception))


class TestServerArgsValidation(unittest.TestCase):
    def test_max_reqs_per_dp_worker_requires_shortest_queue(self):
        """Test that max_reqs_per_dp_worker requires shortest_queue load balancing."""
        server_args = ServerArgs(model_path="dummy")
        server_args.dp_size = 2
        server_args.max_reqs_per_dp_worker = 10
        server_args.load_balance_method = "round_robin"

        with self.assertRaises(ValueError) as context:
            server_args.check_server_args()

        self.assertIn("--max-reqs-per-dp-worker requires --load-balance-method shortest_queue", str(context.exception))
        self.assertIn("round_robin", str(context.exception))

    def test_max_reqs_per_dp_worker_with_shortest_queue_passes(self):
        """Test that max_reqs_per_dp_worker with shortest_queue passes validation."""
        server_args = ServerArgs(model_path="dummy")
        server_args.dp_size = 2
        server_args.max_reqs_per_dp_worker = 10
        server_args.load_balance_method = "shortest_queue"

        # Should not raise an error for this specific check
        # Note: check_server_args may still fail for other reasons (like missing model config)
        # but we're testing that this specific validation passes
        try:
            server_args.check_server_args()
        except ValueError as e:
            # Make sure the error is NOT about max_reqs_per_dp_worker
            self.assertNotIn("--max-reqs-per-dp-worker", str(e))
        except Exception:
            # Other errors are fine - we just care that our validation passed
            pass

    def test_max_reqs_per_dp_worker_with_dp_size_1_passes(self):
        """Test that max_reqs_per_dp_worker is allowed with dp_size=1 (no DP)."""
        server_args = ServerArgs(model_path="dummy")
        server_args.dp_size = 1
        server_args.max_reqs_per_dp_worker = 10
        server_args.load_balance_method = "round_robin"

        # Should not raise the max_reqs_per_dp_worker error
        try:
            server_args.check_server_args()
        except ValueError as e:
            self.assertNotIn("--max-reqs-per-dp-worker", str(e))
        except Exception:
            pass


if __name__ == "__main__":
    unittest.main()
