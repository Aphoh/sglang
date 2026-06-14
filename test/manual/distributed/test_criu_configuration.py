from types import SimpleNamespace

import pytest

from sglang.srt.distributed.criu_process_groups import (
    validate_checkpoint_configuration,
)


def make_server_args(**overrides):
    values = {
        "pp_size": 1,
        "dp_size": 1,
        "ep_size": 1,
        "moe_dp_size": 1,
        "attn_cp_size": 1,
        "enable_dp_attention": False,
        "moe_a2a_backend": "none",
        "disaggregation_mode": "null",
        "enable_hierarchical_cache": False,
        "hicache_storage_backend": None,
        "disable_radix_cache": True,
        "disable_custom_all_reduce": True,
        "enable_symm_mem": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def make_model_config(**hf_values):
    return SimpleNamespace(hf_text_config=SimpleNamespace(**hf_values))


def test_dense_tp_configuration_is_supported():
    validate_checkpoint_configuration(make_server_args(), make_model_config())


@pytest.mark.parametrize(
    ("override", "value", "message"),
    [
        ("pp_size", 2, "pipeline parallelism"),
        ("dp_size", 2, "data parallelism"),
        ("ep_size", 2, "expert parallelism"),
        ("moe_a2a_backend", "flashinfer", "MoE all-to-all"),
        ("disaggregation_mode", "decode", "PD disaggregation"),
        ("enable_hierarchical_cache", True, "hierarchical cache"),
        ("disable_radix_cache", False, "radix cache"),
        ("disable_custom_all_reduce", False, "custom all-reduce"),
    ],
)
def test_unsupported_configuration_is_rejected(override, value, message):
    with pytest.raises(RuntimeError, match=message):
        validate_checkpoint_configuration(
            make_server_args(**{override: value}),
            make_model_config(),
        )


@pytest.mark.parametrize(
    "hf_values",
    [
        {"n_routed_experts": 128},
        {"num_local_experts": 8},
        {"moe_num_experts": [8, 8]},
    ],
)
def test_moe_models_are_rejected(hf_values):
    with pytest.raises(RuntimeError, match="MoE model"):
        validate_checkpoint_configuration(
            make_server_args(),
            make_model_config(**hf_values),
        )
