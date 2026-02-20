# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import logging
import math
from typing import Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.fused_moe_triton.layer import get_moe_runner_backend
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz
from sglang.srt.utils import (
    cpu_has_amx_support,
    get_bool_env_var,
    get_device_sm,
    is_cpu,
    is_cuda,
    is_gfx95_supported,
    is_hip,
    is_npu,
    is_nvidia_cublas_version_ge_12_9,
)

_is_hip = is_hip()
_is_cuda = is_cuda()
_is_npu = is_npu()
_is_fp8_fnuz = is_fp8_fnuz()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
_is_cpu_amx_available = cpu_has_amx_support()
_is_cpu = is_cpu()
_device_sm = get_device_sm()
_is_gfx95_supported = is_gfx95_supported()
_use_aiter_gfx95 = _use_aiter and _is_gfx95_supported


_is_cublas_ge_129 = is_nvidia_cublas_version_ge_12_9()

logger = logging.getLogger(__name__)

NVFP4_CKPT_FP8_ATTN_QUANT_MODULES = ["q_b_proj"]

FORWARD_ABSORB_CORE_ATTENTION_BACKENDS = [
    "fa3",
    "nsa",
    "flashinfer",
    "cutlass_mla",
    "trtllm_mla",
    "ascend",
]


def awq_dequantize_func():
    """
    Get the AWQ dequantize function for the current device

    Return:
        - The AWQ dequantize function for the current device.
        - None if the current device is not supported.
    """
    if _is_cuda:
        from sgl_kernel import awq_dequantize

        return awq_dequantize
    elif _is_hip:
        from sglang.srt.layers.quantization.awq_triton import (
            awq_dequantize_triton as awq_dequantize,
        )

        return awq_dequantize
    elif _is_npu:
        from sglang.srt.layers.quantization.awq_triton import (
            awq_dequantize_decomposition as awq_dequantize,
        )

        return awq_dequantize
    else:
        return None


def is_nextn_moe_checkpoint_fp8(model_path: str) -> bool:
    """Detect if the nextn/MTP MoE layer is pre-quantized to FP8 in the checkpoint."""
    import json
    import os

    # Fast path: dedicated mtp-fp8 file (GLM-5 style)
    if os.path.exists(os.path.join(model_path, "mtp-fp8.safetensors")):
        return True
    # General: inspect safetensors index for weight_scale_inv in MoE layers
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        return False
    with open(index_path) as f:
        data = json.load(f)
    return any(
        "weight_scale_inv" in k and ".mlp.experts." in k
        for k in data.get("weight_map", {})
    )


def enable_nextn_moe_fp8(
    quant_config: Optional[QuantizationConfig],
    model_path: Optional[str] = None,
) -> bool:
    """Whether to build the nextn MoE layer with FP8 quantization.

    True in two cases:
    - Checkpoint already has FP8 nextn weights (any runner backend), OR
    - BF16 checkpoint that will be quantized on-the-fly (deep_gemm only).
    """
    if not (
        envs.SGLANG_NVFP4_CKPT_FP8_NEXTN_MOE.get()
        and quant_config is not None
        and quant_config.get_name() == "modelopt_fp4"
    ):
        return False
    if model_path is not None and is_nextn_moe_checkpoint_fp8(model_path):
        return True  # pre-quantized FP8, any runner works
    return get_moe_runner_backend().is_deep_gemm()  # BF16 on-the-fly, deep_gemm only


def enable_nextn_moe_bf16_cast_to_fp8(
    quant_config: Optional[QuantizationConfig],
) -> bool:
    """Whether on-the-fly BF16->UE8M0 quantization was applied to nextn MoE weights.

    Only True when the checkpoint is BF16 and deep_gemm is the runner. Used
    to gate the UE8M0 format marking after loading; not for layer init or
    partial_names population (use enable_nextn_moe_fp8 for those).
    """
    return (
        envs.SGLANG_NVFP4_CKPT_FP8_NEXTN_MOE.get()
        and quant_config is not None
        and quant_config.get_name() == "modelopt_fp4"
        and get_moe_runner_backend().is_deep_gemm()
    )


def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def _get_llama_4_scaling(
    original_max_position_embeddings: int, scaling_beta: float, positions: torch.Tensor
) -> torch.Tensor:
    scaling = 1 + scaling_beta * torch.log(
        1 + torch.floor(positions / original_max_position_embeddings)
    )
    return scaling[..., None, None]
