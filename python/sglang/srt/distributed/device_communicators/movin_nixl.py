import logging
from dataclasses import dataclass
from typing import Any

import torch

logger = logging.getLogger(__name__)



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
        TorchDistributedSymmetricAllGather,
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
        all_gather = TorchDistributedSymmetricAllGather(
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
