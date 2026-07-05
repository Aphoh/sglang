"""Worker-local composition of checkpointable process-group lifecycles."""

from __future__ import annotations

from datetime import timedelta
from enum import Enum, auto
from typing import Callable, Iterable, Protocol

from sglang.srt.distributed.cpu_group_lifecycle import CpuGroupTransaction
from sglang.srt.distributed.device_group_lifecycle import (
    DefaultDeviceGroupTransaction,
)


class CheckpointLifecycleError(RuntimeError):
    pass


class CheckpointState(Enum):
    READY = auto()
    SUSPENDING = auto()
    SUSPENDED = auto()
    RESTORING = auto()
    FAILED = auto()


class CheckpointGroup(Protocol):
    cpu_group_lifecycle: object


class CheckpointLifecycle:
    """Order CPU and default-device group renewal around an external checkpoint."""

    def __init__(
        self,
        groups: Iterable[CheckpointGroup],
        *,
        store_prefix: str,
        timeout: timedelta = timedelta(minutes=30),
        pg_options_factory: Callable[[], object] | None = None,
        synchronize: Callable[[], None] | None = None,
    ) -> None:
        groups = tuple({id(group): group for group in groups}.values())
        if not groups:
            raise ValueError("checkpoint lifecycle requires at least one group")
        bindings = tuple(
            {
                id(group.cpu_group_lifecycle): group.cpu_group_lifecycle
                for group in groups
            }.values()
        )
        self._cpu = CpuGroupTransaction(bindings)
        self._device = DefaultDeviceGroupTransaction(
            groups,
            store_prefix=store_prefix,
            timeout=timeout,
            pg_options_factory=pg_options_factory,
            synchronize=synchronize,
        )
        self._state = CheckpointState.READY

    @property
    def state(self) -> CheckpointState:
        return self._state

    def suspend(self) -> None:
        if self._state is not CheckpointState.READY:
            raise CheckpointLifecycleError(
                f"cannot suspend checkpoint lifecycle from {self._state.name}"
            )
        self._device.preflight()
        self._state = CheckpointState.SUSPENDING
        try:
            self._cpu.suspend()
            self._device.suspend()
        except Exception as exc:
            self._state = CheckpointState.FAILED
            raise CheckpointLifecycleError(
                "checkpoint suspend failed; worker restart required"
            ) from exc
        self._state = CheckpointState.SUSPENDED

    def resume(self) -> None:
        if self._state is not CheckpointState.SUSPENDED:
            raise CheckpointLifecycleError(
                f"cannot resume checkpoint lifecycle from {self._state.name}"
            )
        self._state = CheckpointState.RESTORING
        try:
            self._device.resume()
            self._cpu.resume()
        except Exception as exc:
            self._state = CheckpointState.FAILED
            raise CheckpointLifecycleError(
                "checkpoint restore failed; worker restart required"
            ) from exc
        self._state = CheckpointState.READY
