from __future__ import annotations

import contextlib
import gc
import time
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import torch


@dataclass(frozen=True)
class SuspendedDeviceProcessGroup:
    backend: str
    rank: int
    world_size: int
    store_path: str


_suspended_device_group: SuspendedDeviceProcessGroup | None = None
_device_group_generation = 0
_registered_groups: weakref.WeakValueDictionary[str, Any] = (
    weakref.WeakValueDictionary()
)


def register_group(group: Any) -> None:
    _registered_groups[group.unique_name] = group


def unregister_group(group: Any) -> None:
    _registered_groups.pop(group.unique_name, None)


def registered_groups() -> list[Any]:
    return list(_registered_groups.values())


def _model_uses_moe(model_config: Any) -> bool:
    config = model_config.hf_text_config
    for name in (
        "n_routed_experts",
        "num_local_experts",
        "num_experts",
        "num_moe_experts",
        "moe_num_experts",
    ):
        value = getattr(config, name, None)
        if isinstance(value, (list, tuple)):
            if any(item for item in value if isinstance(item, int)):
                return True
        elif isinstance(value, int) and value > 0:
            return True
    ffn_config = getattr(config, "ffn_config", None)
    return bool(ffn_config and getattr(ffn_config, "moe_num_experts", 0))


def validate_checkpoint_configuration(
    server_args: Any,
    model_config: Any,
) -> None:
    """Reject runtime modes outside the currently validated dense-TP scope."""
    checks = (
        ("pipeline parallelism", server_args.pp_size != 1),
        ("data parallelism", server_args.dp_size != 1),
        ("expert parallelism", server_args.ep_size != 1),
        ("MoE data parallelism", server_args.moe_dp_size != 1),
        ("context parallelism", server_args.attn_cp_size != 1),
        ("DP attention", server_args.enable_dp_attention),
        ("MoE all-to-all", server_args.moe_a2a_backend != "none"),
        ("PD disaggregation", server_args.disaggregation_mode != "null"),
        (
            "hierarchical cache",
            server_args.enable_hierarchical_cache
            or server_args.hicache_storage_backend is not None,
        ),
        ("radix cache", not server_args.disable_radix_cache),
        ("custom all-reduce", not server_args.disable_custom_all_reduce),
        ("symmetric memory", server_args.enable_symm_mem),
    )
    unsupported = [name for name, enabled in checks if enabled]
    if _model_uses_moe(model_config):
        unsupported.append("MoE model")
    if unsupported:
        raise RuntimeError(
            "CRIU checkpoint mode currently supports dense TP with PP=DP=EP=CP=1, "
            "radix cache disabled, and checkpointable Movin/FlashInfer collectives; "
            f"unsupported configuration: {', '.join(unsupported)}"
        )


def validate_checkpoint_topology(groups: Iterable[Any]) -> None:
    """Reject process groups that the dense-TP checkpoint path cannot renew."""
    if not torch.distributed.is_initialized():
        raise RuntimeError("Default process group is not initialized")
    default_group = torch.distributed.group.WORLD
    unsupported = []
    for group in groups:
        device_group = group.device_group
        if (
            device_group is not None
            and group._owns_device_group
            and not group._device_group_is_cpu_alias
            and device_group is not default_group
        ):
            unsupported.append(group.unique_name)
    if unsupported:
        raise RuntimeError(
            "CRIU currently supports dense TP groups that reuse WORLD plus "
            f"CPU-only singleton groups; owned device subgroups are unsupported: "
            f"{unsupported}"
        )
    for group in groups:
        group.validate_checkpoint_lifecycle()


def suspend_device_process_group(
    groups: Iterable[Any],
    *,
    enabled: bool,
    store_base: str | None,
) -> None:
    if not enabled:
        return

    global _suspended_device_group
    global _device_group_generation
    if _suspended_device_group is not None:
        return
    if not torch.distributed.is_initialized():
        raise RuntimeError("Default process group is not initialized")
    if not store_base:
        raise RuntimeError(
            "SGLANG_CRIU_DEVICE_STORE is required to suspend the device group"
        )

    default_group = torch.distributed.group.WORLD
    backend = str(torch.distributed.get_backend(default_group))
    _device_group_generation += 1
    store_path = f"{store_base}.{_device_group_generation}"
    resume_path = Path(f"{store_path}.resume")
    rank = torch.distributed.get_rank(group=default_group)
    world_size = torch.distributed.get_world_size(group=default_group)

    if rank == 0:
        for path in (Path(store_path), resume_path):
            with contextlib.suppress(FileNotFoundError):
                path.unlink()

    rebound = 0
    for group in groups:
        if group.device_group is default_group:
            group.device_group = None
            group._device_group_suspended = True
            rebound += 1
    if rebound == 0:
        raise RuntimeError("No SGLang groups borrow the default process group")

    if backend == "nccl":
        default_group.abort()
    elif hasattr(default_group, "shutdown"):
        default_group.shutdown()
    torch.distributed.destroy_process_group()
    gc.collect()
    _suspended_device_group = SuspendedDeviceProcessGroup(
        backend=backend,
        rank=rank,
        world_size=world_size,
        store_path=store_path,
    )


def wait_for_process_group_teardown(timeout: float = 120.0) -> None:
    state = _suspended_device_group
    if state is None:
        return

    marker_prefix = f"{state.store_path}.ready"
    Path(f"{marker_prefix}.{state.rank}").touch()
    deadline = time.monotonic() + timeout
    expected = [Path(f"{marker_prefix}.{rank}") for rank in range(state.world_size)]
    while not all(path.exists() for path in expected):
        if time.monotonic() >= deadline:
            missing = [str(path) for path in expected if not path.exists()]
            raise TimeoutError(
                f"Timed out waiting for process-group teardown: {missing}"
            )
        time.sleep(0.01)


def resume_device_process_group(
    groups: Iterable[Any],
    *,
    timeout,
    pg_options_factory: Callable[[], Any],
) -> None:
    global _suspended_device_group
    state = _suspended_device_group
    if state is None:
        return
    if torch.distributed.is_initialized():
        raise RuntimeError("Default process group was unexpectedly initialized")

    store = torch.distributed.FileStore(state.store_path, state.world_size)
    torch.distributed.init_process_group(
        backend=state.backend,
        store=store,
        rank=state.rank,
        world_size=state.world_size,
        timeout=timeout,
        pg_options=pg_options_factory(),
    )
    default_group = torch.distributed.group.WORLD
    rebound = 0
    for group in groups:
        if group._device_group_suspended:
            group.device_group = default_group
            group._device_group_suspended = False
            rebound += 1
    if rebound == 0:
        raise RuntimeError("No SGLang groups were rebound to the new process group")
    _suspended_device_group = None


def suspend_cpu_group(group: Any) -> None:
    if group._cpu_group_suspended:
        return
    cpu_group = group.cpu_group
    if cpu_group is None:
        raise RuntimeError(f"CPU group {group.unique_name} is not initialized")

    if group.mq_broadcaster is not None:
        group.mq_broadcaster.close()
        group.mq_broadcaster = None
    if group._device_group_is_cpu_alias:
        group.device_group = None
    group.cpu_group = None
    group._set_communicator_cpu_group(None)

    cpu_group_ref = weakref.ref(cpu_group)
    cpu_group.shutdown()
    try:
        torch.distributed.destroy_process_group(cpu_group)
    except ValueError as exc:
        if "Invalid process group specified" not in str(exc):
            raise
    del cpu_group
    gc.collect()
    if cpu_group_ref() is not None:
        referrer_types = [
            type(referrer).__name__ for referrer in gc.get_referrers(cpu_group_ref())
        ]
        raise RuntimeError(
            f"CPU group {group.unique_name} is still referenced after shutdown: "
            f"{referrer_types}"
        )
    group._cpu_group_suspended = True


def resume_cpu_group(group: Any, *, model_parallel_timeout) -> None:
    if not group._cpu_group_suspended:
        return

    active_cpu_group = None
    for ranks in group._group_ranks:
        active_ranks_cpu = torch.ones(len(ranks), dtype=torch.int32)
        if "mooncake" in group._torch_distributed_backend:
            from mooncake.ep import MooncakeBackendOptions

            cpu_group = torch.distributed.new_group(
                ranks,
                backend="mooncake-cpu",
                pg_options=MooncakeBackendOptions(
                    active_ranks_cpu,
                    group._recovered_rank,
                ),
                timeout=model_parallel_timeout,
            )
        else:
            cpu_group = torch.distributed.new_group(
                ranks,
                backend="gloo",
                timeout=group._gloo_timeout,
            )
        if group.rank in ranks:
            active_cpu_group = cpu_group
            group.active_ranks_cpu = active_ranks_cpu

    if active_cpu_group is None:
        raise RuntimeError(f"Rank {group.rank} has no subgroup in {group.unique_name}")
    group.cpu_group = active_cpu_group
    if group._device_group_is_cpu_alias:
        group.device_group = active_cpu_group
    group._set_communicator_cpu_group(active_cpu_group)
    if (
        group.use_message_queue_broadcaster
        and group.world_size > 1
        and not group._recovered_rank
    ):
        from sglang.srt.distributed.device_communicators.shm_broadcast import (
            MessageQueue,
        )

        group.mq_broadcaster = MessageQueue.create_from_process_group(
            active_cpu_group,
            1 << 22,
            6,
        )
    group._cpu_group_suspended = False
    group._cpu_group_generation += 1
