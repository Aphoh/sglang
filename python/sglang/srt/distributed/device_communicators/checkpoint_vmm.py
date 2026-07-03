"""Checkpoint-renewable peer mappings for native CUDA collectives."""

from __future__ import annotations

import array
import os
import socket
import struct
import tempfile
import threading
from enum import Enum, auto
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.srt.utils import get_cuda_driver_bindings


class MappingState(Enum):
    ATTACHED = auto()
    DETACHING = auto()
    DETACHED = auto()
    RESTORING = auto()
    CLOSED = auto()


def _check(result: Any, operation: str, driver: Any) -> Any:
    if not isinstance(result, tuple):
        result = (result,)
    error, *values = result
    if error != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{operation} failed: {error}")
    if not values:
        return None
    return values[0] if len(values) == 1 else tuple(values)


def _exchange_fd(
    group: ProcessGroup,
    rank: int,
    world_size: int,
    local_fd: int,
) -> dict[int, int]:
    """Exchange one CUDA allocation fd per rank using SCM_RIGHTS."""
    directory = tempfile.mkdtemp(prefix="sglang_collective_fd_")
    path = os.path.join(directory, "socket")
    kind = getattr(socket, "SOCK_SEQPACKET", socket.SOCK_STREAM)
    server = socket.socket(socket.AF_UNIX, kind)
    server.settimeout(120)
    server.bind(path)
    server.listen(world_size)
    paths = [None] * world_size
    dist.all_gather_object(paths, path, group=group)

    received: dict[int, int] = {}
    errors: list[BaseException] = []

    def receive() -> None:
        try:
            for _ in range(world_size - 1):
                connection, _ = server.accept()
                with connection:
                    payload, ancillary, _, _ = connection.recvmsg(
                        8, socket.CMSG_SPACE(array.array("i").itemsize)
                    )
                    if len(payload) != 8:
                        raise RuntimeError("received truncated CUDA fd metadata")
                    source = struct.unpack("<Q", payload)[0]
                    fds = array.array("i")
                    for level, message_type, data in ancillary:
                        if (
                            level == socket.SOL_SOCKET
                            and message_type == socket.SCM_RIGHTS
                        ):
                            fds.frombytes(data[: len(data) - len(data) % fds.itemsize])
                    if len(fds) != 1 or source in received:
                        for fd in fds:
                            os.close(fd)
                        raise RuntimeError("invalid CUDA fd exchange payload")
                    received[int(source)] = int(fds[0])
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=receive, daemon=True)
    thread.start()
    try:
        for peer, peer_path in enumerate(paths):
            if peer == rank:
                continue
            with socket.socket(socket.AF_UNIX, kind) as client:
                client.settimeout(120)
                client.connect(peer_path)
                fds = array.array("i", [local_fd])
                sent = client.sendmsg(
                    [struct.pack("<Q", rank)],
                    [(socket.SOL_SOCKET, socket.SCM_RIGHTS, fds.tobytes())],
                )
                if sent != 8:
                    raise RuntimeError("failed to send CUDA allocation fd")
        thread.join(120)
        if thread.is_alive():
            raise TimeoutError("timed out exchanging CUDA allocation fds")
        if errors:
            raise RuntimeError("failed to receive CUDA allocation fd") from errors[0]
        expected = set(range(world_size)) - {rank}
        if set(received) != expected:
            raise RuntimeError(
                f"CUDA fd exchange mismatch: got={sorted(received)}, "
                f"expected={sorted(expected)}"
            )
        return received
    except Exception:
        for fd in received.values():
            os.close(fd)
        raise
    finally:
        server.close()
        try:
            os.unlink(path)
            os.rmdir(directory)
        except OSError:
            pass


class CheckpointableVmmBuffer:
    """A same-node shared buffer whose virtual addresses never change."""

    def __init__(
        self,
        size: int,
        group: ProcessGroup,
        device: torch.device,
    ) -> None:
        if size <= 0:
            raise ValueError("VMM buffer size must be positive")
        self.group = group
        self.device = device
        self.rank = dist.get_rank(group=group)
        self.world_size = dist.get_world_size(group=group)
        self.driver = get_cuda_driver_bindings()
        self._prop = self._allocation_prop()
        granularity_flag = (
            self.driver.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED
        )
        self.granularity = int(
            _check(
                self.driver.cuMemGetAllocationGranularity(self._prop, granularity_flag),
                "cuMemGetAllocationGranularity",
                self.driver,
            )
        )
        self.size = (size + self.granularity - 1) // self.granularity * self.granularity
        self.span_size = self.size * self.world_size
        self.base = int(
            _check(
                self.driver.cuMemAddressReserve(self.span_size, self.granularity, 0, 0),
                "cuMemAddressReserve",
                self.driver,
            )
        )
        self.ptrs = tuple(
            self.base + rank * self.size for rank in range(self.world_size)
        )
        self.handles: list[Any] = []
        self.state = MappingState.DETACHED
        try:
            self.restore()
        except Exception:
            _check(
                self.driver.cuMemAddressFree(self.base, self.span_size),
                "cuMemAddressFree",
                self.driver,
            )
            self.state = MappingState.CLOSED
            raise

    @property
    def local_ptr(self) -> int:
        return self.ptrs[self.rank]

    @property
    def attached(self) -> bool:
        return self.state is MappingState.ATTACHED

    def set_control_group(self, group: ProcessGroup | None) -> None:
        self.group = group

    def detach(self) -> None:
        if self.state is MappingState.DETACHED:
            return
        if self.state is not MappingState.ATTACHED:
            raise RuntimeError(f"cannot detach VMM buffer from {self.state.name}")
        self.state = MappingState.DETACHING
        try:
            for pointer in self.ptrs:
                _check(
                    self.driver.cuMemUnmap(pointer, self.size),
                    "cuMemUnmap(checkpoint buffer)",
                    self.driver,
                )
            for handle in self.handles:
                _check(
                    self.driver.cuMemRelease(handle),
                    "cuMemRelease(checkpoint buffer)",
                    self.driver,
                )
            self.handles.clear()
        except Exception:
            self.state = MappingState.DETACHING
            raise
        self.state = MappingState.DETACHED

    def restore(self) -> None:
        if self.state is MappingState.ATTACHED:
            return
        if self.state is not MappingState.DETACHED:
            raise RuntimeError(f"cannot restore VMM buffer from {self.state.name}")
        if self.group is None:
            raise RuntimeError("cannot restore VMM buffer without a control group")
        self.state = MappingState.RESTORING
        handles: list[Any] = []
        mapped: list[int] = []
        received_fds: dict[int, int] = {}
        local_fd = None
        local_handle = None
        try:
            local_handle = _check(
                self.driver.cuMemCreate(self.size, self._prop, 0),
                "cuMemCreate(checkpoint buffer)",
                self.driver,
            )
            handle_type = (
                self.driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
            )
            local_fd = int(
                _check(
                    self.driver.cuMemExportToShareableHandle(
                        local_handle, handle_type, 0
                    ),
                    "cuMemExportToShareableHandle",
                    self.driver,
                )
            )
            received_fds = _exchange_fd(
                self.group, self.rank, self.world_size, local_fd
            )
            for peer in range(self.world_size):
                if peer == self.rank:
                    handle = local_handle
                else:
                    duplicate = os.dup(received_fds[peer])
                    try:
                        handle = _check(
                            self.driver.cuMemImportFromShareableHandle(
                                duplicate, handle_type
                            ),
                            "cuMemImportFromShareableHandle",
                            self.driver,
                        )
                    finally:
                        os.close(duplicate)
                handles.append(handle)
                _check(
                    self.driver.cuMemMap(self.ptrs[peer], self.size, 0, handle, 0),
                    "cuMemMap(checkpoint buffer)",
                    self.driver,
                )
                mapped.append(peer)
            access = self.driver.CUmemAccessDesc()
            access.location.type = (
                self.driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
            )
            access.location.id = self.device.index
            access.flags = (
                self.driver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
            )
            _check(
                self.driver.cuMemSetAccess(self.base, self.span_size, [access], 1),
                "cuMemSetAccess(checkpoint buffer)",
                self.driver,
            )
            _check(
                self.driver.cuMemsetD8(self.local_ptr, 0, self.size),
                "cuMemsetD8(checkpoint buffer)",
                self.driver,
            )
        except Exception:
            for peer in reversed(mapped):
                self.driver.cuMemUnmap(self.ptrs[peer], self.size)
            for handle in reversed(handles):
                self.driver.cuMemRelease(handle)
            if local_handle is not None and local_handle not in handles:
                self.driver.cuMemRelease(local_handle)
            self.state = MappingState.DETACHED
            raise
        finally:
            if local_fd is not None:
                os.close(local_fd)
            for fd in received_fds.values():
                os.close(fd)
        self.handles = handles
        self.state = MappingState.ATTACHED

    def close(self) -> None:
        if self.state is MappingState.CLOSED:
            return
        if self.state is MappingState.ATTACHED:
            self.detach()
        if self.state is not MappingState.DETACHED:
            raise RuntimeError(f"cannot close VMM buffer from {self.state.name}")
        _check(
            self.driver.cuMemAddressFree(self.base, self.span_size),
            "cuMemAddressFree(checkpoint buffer)",
            self.driver,
        )
        self.state = MappingState.CLOSED

    def _allocation_prop(self):
        prop = self.driver.CUmemAllocationProp()
        prop.type = self.driver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = self.driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = self.device.index
        prop.requestedHandleTypes = (
            self.driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
        )
        return prop
