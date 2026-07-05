"""Transactional renewal of the default device process group."""

from __future__ import annotations

from datetime import timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

import torch.distributed as dist
from torch.distributed import ProcessGroup


class DeviceGroupLifecycleError(RuntimeError):
    pass


class DeviceGroupState(Enum):
    ACTIVE = auto()
    SUSPENDED = auto()
    FAILED = auto()


class DeviceGroupBorrower(Protocol):
    unique_name: str
    device_group: ProcessGroup | None
    owns_device_group: bool
    device_group_is_cpu_alias: bool


class DefaultDeviceGroupTransaction:
    """Destroy and recreate WORLD while preserving its SGLang borrowers."""

    def __init__(
        self,
        groups: Iterable[DeviceGroupBorrower],
        *,
        store_prefix: str,
        timeout: timedelta = timedelta(minutes=30),
        pg_options_factory: Callable[[], Any] | None = None,
        synchronize: Callable[[], None] | None = None,
    ) -> None:
        if not store_prefix:
            raise ValueError("store_prefix must not be empty")
        self._groups = tuple({id(group): group for group in groups}.values())
        self._store_prefix = store_prefix
        self._timeout = timeout
        self._pg_options_factory = pg_options_factory
        self._synchronize = synchronize or (lambda: None)
        self._state = DeviceGroupState.ACTIVE
        self._generation = 0
        self._borrowers: tuple[DeviceGroupBorrower, ...] = ()
        self._recipe: tuple[str, int, int, str] | None = None

    @property
    def state(self) -> DeviceGroupState:
        return self._state

    def preflight(self) -> tuple[DeviceGroupBorrower, ...]:
        if self._state is not DeviceGroupState.ACTIVE:
            raise DeviceGroupLifecycleError(
                f"cannot suspend device WORLD from {self._state.name}"
            )
        if not dist.is_initialized():
            raise DeviceGroupLifecycleError("default process group is unavailable")

        world = dist.group.WORLD
        borrowers = self._preflight(world)
        self._check_manifest(borrowers)
        return borrowers

    def suspend(self) -> None:
        borrowers = self.preflight()
        world = dist.group.WORLD
        backend = str(dist.get_backend(world))
        rank = dist.get_rank(world)
        world_size = dist.get_world_size(world)
        self._generation += 1
        store_path = f"{self._store_prefix}.{self._generation}"
        if rank == 0:
            Path(store_path).unlink(missing_ok=True)
        dist.barrier(group=world)
        self._synchronize()

        self._borrowers = borrowers
        self._recipe = (backend, rank, world_size, store_path)
        for group in borrowers:
            group.device_group = None
        try:
            if backend == "nccl" and hasattr(world, "abort"):
                world.abort()
            elif hasattr(world, "shutdown"):
                world.shutdown()
            dist.destroy_process_group()
        except Exception as exc:
            self._state = DeviceGroupState.FAILED
            raise DeviceGroupLifecycleError(
                "device WORLD teardown failed; worker restart required"
            ) from exc
        self._state = DeviceGroupState.SUSPENDED

    def resume(self) -> None:
        if self._state is not DeviceGroupState.SUSPENDED or self._recipe is None:
            raise DeviceGroupLifecycleError(
                f"cannot resume device WORLD from {self._state.name}"
            )
        if dist.is_initialized():
            raise DeviceGroupLifecycleError("default process group is already active")

        backend, rank, world_size, store_path = self._recipe
        try:
            store = dist.FileStore(store_path, world_size)
            pg_options = (
                None if self._pg_options_factory is None else self._pg_options_factory()
            )
            dist.init_process_group(
                backend=backend,
                store=store,
                rank=rank,
                world_size=world_size,
                timeout=self._timeout,
                pg_options=pg_options,
            )
            world = dist.group.WORLD
            for group in self._borrowers:
                group.device_group = world
            dist.barrier(group=world)
        except Exception as exc:
            self._state = DeviceGroupState.FAILED
            raise DeviceGroupLifecycleError(
                "device WORLD restore failed; worker restart required"
            ) from exc
        self._borrowers = ()
        self._recipe = None
        self._state = DeviceGroupState.ACTIVE

    def _preflight(self, world: ProcessGroup) -> tuple[DeviceGroupBorrower, ...]:
        borrowers = []
        unsupported = []
        for group in sorted(self._groups, key=lambda item: item.unique_name):
            if group.device_group is world:
                borrowers.append(group)
            elif group.device_group_is_cpu_alias:
                continue
            elif group.device_group is not None or group.owns_device_group:
                unsupported.append(group.unique_name)
        if unsupported:
            raise DeviceGroupLifecycleError(
                f"owned or foreign device groups cannot be renewed: {unsupported}"
            )
        if not borrowers:
            raise DeviceGroupLifecycleError("no SGLang group borrows device WORLD")
        return tuple(borrowers)

    @staticmethod
    def _check_manifest(borrowers: tuple[DeviceGroupBorrower, ...]) -> None:
        local = tuple(group.unique_name for group in borrowers)
        manifests = [None] * dist.get_world_size()
        dist.all_gather_object(manifests, local)
        if any(manifest != manifests[0] for manifest in manifests[1:]):
            raise DeviceGroupLifecycleError(
                "device WORLD borrower manifest differs across ranks"
            )
