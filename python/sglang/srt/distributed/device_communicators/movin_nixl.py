import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch

logger = logging.getLogger(__name__)


def _is_weak_contiguous(tensor: torch.Tensor) -> bool:
    expected_stride = 1
    for size, stride in zip(reversed(tensor.shape), reversed(tensor.stride())):
        if size > 1 and stride != expected_stride:
            return False
        expected_stride *= size
    return True


class FlashInferSymmetricAllGather:
    """SGLang adapter for FlashInfer's checkpointable symmetric all-gather."""

    def __init__(
        self,
        group: Any,
        device: torch.device,
        group_name: str,
        *,
        max_elems: int,
        restore_probe: bool = False,
    ):
        from flashinfer.comm import SymmetricAllGatherWorkspace
        from sglang.srt.layers.moe.token_dispatcher.flashinfer_utils import (
            TorchDistributedCommBackend,
        )

        self.control_group = group
        self.device = device
        self.group_name = group_name
        self.restore_probe = restore_probe
        self.workspace = SymmetricAllGatherWorkspace(
            max_elems=max_elems,
            world_size=group.size(),
            rank=group.rank(),
            comm_backend=TorchDistributedCommBackend(group),
            dtype=torch.bfloat16,
        )

    @contextmanager
    def capture(self):
        yield

    def should_all_gather(
        self,
        input_: torch.Tensor,
        output: torch.Tensor | None = None,
    ) -> bool:
        return (
            input_.dim() > 0
            and input_.dtype == self.workspace.dtype
            and input_.device == self.device
            and _is_weak_contiguous(input_)
            and input_.numel() <= self.workspace.max_elems
            and (
                output is None
                or (
                    output.dtype == input_.dtype
                    and output.device == input_.device
                    and output.is_contiguous()
                    and output.numel() == input_.numel() * self.workspace.world_size
                )
            )
        )

    def all_gather(
        self,
        input_: torch.Tensor,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.should_all_gather(input_, output):
            raise RuntimeError("FlashInfer all-gather does not support this input")
        return self.workspace.all_gather(input_.contiguous(), output)

    def prepare_checkpoint(self) -> None:
        self.workspace.prepare_checkpoint()

    def set_control_group(self, group: Any) -> None:
        self.control_group = group

    def restore_after_checkpoint(self) -> None:
        if self.control_group is None:
            raise RuntimeError("FlashInfer all-gather has no restored control group")
        from sglang.srt.layers.moe.token_dispatcher.flashinfer_utils import (
            TorchDistributedCommBackend,
        )

        self.workspace.restore_after_checkpoint(
            TorchDistributedCommBackend(self.control_group)
        )
        if self.restore_probe:
            self._run_restore_probe()

    def status(self) -> list[int]:
        return self.workspace.status()

    def close(self) -> None:
        self.workspace.destroy()

    def _run_restore_probe(self) -> None:
        input_ = torch.zeros(8, dtype=self.workspace.dtype, device=self.device)
        output = self.workspace.all_gather(input_)
        torch.cuda.synchronize(self.device)
        if torch.count_nonzero(output).item():
            raise RuntimeError("restored FlashInfer all-gather probe was nonzero")
        status = self.workspace.status()
        if status[1:] != [0, 0, 0]:
            raise RuntimeError(f"FlashInfer all-gather failed: {status}")


@dataclass(frozen=True)
class MovinCollectiveConfig:
    all_reduce_backend: str | None
    enable_all_gather: bool
    all_reduce_max_elems: int
    all_gather_max_elems: int
    checkpointable: bool
    force_nixl_after_restore: bool
    restore_probe: bool

    @classmethod
    def from_env(cls) -> "MovinCollectiveConfig":
        from sglang.srt.environ import envs

        backend = envs.SGLANG_TP_ALL_REDUCE_BACKEND.get() or None
        if backend not in (None, "flashinfer", "movin_nixl"):
            raise ValueError(
                "SGLANG_TP_ALL_REDUCE_BACKEND must be flashinfer or movin_nixl "
                f"for Movin, got {backend!r}"
            )
        config = cls(
            all_reduce_backend=backend,
            enable_all_gather=envs.SGLANG_MOVIN_NIXL_ENABLE_ALLGATHER.get(),
            all_reduce_max_elems=envs.SGLANG_MOVIN_NIXL_MAX_ELEMS.get(),
            all_gather_max_elems=envs.SGLANG_MOVIN_NIXL_ALLGATHER_MAX_ELEMS.get(),
            checkpointable=envs.SGLANG_CRIU_SUSPEND_DEVICE_PROCESS_GROUP.get(),
            force_nixl_after_restore=(
                envs.SGLANG_MOVIN_FORCE_NIXL_ALLREDUCE_AFTER_RESTORE.get()
            ),
            restore_probe=envs.SGLANG_MOVIN_RESTORE_PROBE.get(),
        )
        if config.all_reduce_max_elems <= 0:
            raise ValueError("SGLANG_MOVIN_NIXL_MAX_ELEMS must be positive")
        if config.all_gather_max_elems <= 0:
            raise ValueError("SGLANG_MOVIN_NIXL_ALLGATHER_MAX_ELEMS must be positive")
        return config


def create_movin_collectives(
    group: Any,
    device: torch.device,
    group_name: str,
    config: MovinCollectiveConfig | None = None,
):
    """Build the optional Movin collective set for one TP coordinator."""
    if group_name not in ("tp", "attention_tp"):
        return None
    config = config or MovinCollectiveConfig.from_env()
    if (
        config.all_reduce_backend is None
        and not config.enable_all_gather
        and not config.checkpointable
    ):
        return None

    from movin import (
        CollectiveManager,
        TorchDistributedNixlAllReduce,
    )

    all_reduce = None
    if config.all_reduce_backend == "movin_nixl":
        all_reduce = TorchDistributedNixlAllReduce(
            group=group,
            device=device,
            group_name=group_name,
            max_elems=config.all_reduce_max_elems,
            checkpointable=config.checkpointable,
            force_nixl_after_restore=config.force_nixl_after_restore,
            restore_probe=config.restore_probe,
        )
    elif config.all_reduce_backend == "flashinfer":
        from sglang.srt.layers.flashinfer_comm_fusion import (
            create_flashinfer_raw_allreduce,
        )

        all_reduce = create_flashinfer_raw_allreduce(group)

    all_gather = None
    if config.enable_all_gather:
        all_gather = FlashInferSymmetricAllGather(
            group=group,
            device=device,
            group_name=group_name,
            max_elems=config.all_gather_max_elems,
            restore_probe=config.restore_probe,
        )

    participants = {}
    if config.checkpointable:
        from sglang.srt.layers.flashinfer_comm_fusion import (
            get_flashinfer_checkpoint_participants,
        )

        participants = get_flashinfer_checkpoint_participants(group_name)

    if all_reduce is None and all_gather is None and not participants:
        return None
    logger.info(
        "Movin collectives enabled for %s: all_reduce=%s all_gather=%s",
        group_name,
        type(all_reduce).__name__ if all_reduce is not None else "disabled",
        type(all_gather).__name__ if all_gather is not None else "disabled",
    )
    return CollectiveManager(
        all_reduce=all_reduce,
        all_gather=all_gather,
        participants=participants,
    )
