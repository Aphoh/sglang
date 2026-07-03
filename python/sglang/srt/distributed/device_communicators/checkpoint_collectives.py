"""Native SGLang collectives used across a device-process-group checkpoint."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import torch
import torch.distributed as dist

import sglang.srt.distributed.device_communicators.custom_all_reduce_ops as ops
from sglang.srt.distributed.device_communicators.checkpoint_vmm import (
    CheckpointableVmmBuffer,
)
from sglang.srt.environ import envs


class SymmetricAllGather:
    def __init__(
        self,
        group: Any,
        device: torch.device,
        *,
        max_elems: int,
    ) -> None:
        if max_elems <= 0:
            raise ValueError("all-gather max_elems must be positive")
        self.control_group = group
        self.device = device
        self.rank = dist.get_rank(group=group)
        self.world_size = dist.get_world_size(group=group)
        self.max_elems = max_elems
        self.max_bytes = max_elems * 4
        workspace_size = ops.custom_all_gather_workspace_size(
            self.max_bytes, self.world_size
        )
        self.memory = CheckpointableVmmBuffer(workspace_size, group, device)
        self._default_ticket = torch.empty(2, dtype=torch.uint64, device=device)
        self._capture_tickets: list[torch.Tensor] = []
        self._initialize_protocol()
        dist.barrier(group=group)

    def should_all_gather(
        self,
        input_: torch.Tensor,
        output: torch.Tensor | None = None,
    ) -> bool:
        return (
            input_.dim() > 0
            and input_.dtype in (torch.float16, torch.bfloat16, torch.float32)
            and input_.device == self.device
            and input_.is_contiguous()
            and 0 < input_.numel() <= self.max_elems
            and (
                output is None
                or (
                    output.dtype == input_.dtype
                    and output.device == input_.device
                    and output.is_contiguous()
                    and output.numel() == input_.numel() * self.world_size
                )
            )
        )

    def all_gather(
        self,
        input_: torch.Tensor,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.should_all_gather(input_, output):
            raise RuntimeError("symmetric all-gather does not support this input")
        if output is None:
            output = torch.empty(
                (input_.shape[0] * self.world_size, *input_.shape[1:]),
                dtype=input_.dtype,
                device=input_.device,
            )
        if torch.cuda.is_current_stream_capturing():
            ticket = torch.empty(2, dtype=torch.uint64, device=self.device)
            self._capture_tickets.append(ticket)
        else:
            ticket = self._default_ticket
        ops.custom_all_gather(
            input_,
            output,
            ticket,
            list(self.memory.ptrs),
            self.rank,
            self.max_bytes,
        )
        return output

    def prepare_checkpoint(self) -> None:
        self.memory.detach()

    def set_control_group(self, group: Any) -> None:
        self.control_group = group
        self.memory.set_control_group(group)

    def restore_after_checkpoint(self) -> None:
        self.memory.restore()
        self._initialize_protocol()

    def status(self) -> list[int]:
        return ops.custom_all_gather_status(
            self._default_ticket,
            self.memory.local_ptr,
            self.max_bytes,
            self.world_size,
        )

    def close(self) -> None:
        self._capture_tickets.clear()
        self._default_ticket = None
        self.memory.close()

    def _initialize_protocol(self) -> None:
        ops.custom_all_gather_initialize(
            self._default_ticket,
            self.memory.local_ptr,
            self.max_bytes,
            self.world_size,
        )


class NativeCheckpointCollectives:
    """One lifecycle boundary for a coordinator's native collectives."""

    def __init__(self, all_reduce: Any, all_gather: SymmetricAllGather) -> None:
        self.all_reduce = all_reduce
        self.all_gather_impl = all_gather

    @property
    def has_all_reduce(self) -> bool:
        return self.all_reduce is not None and not self.all_reduce.disabled

    @property
    def has_all_gather(self) -> bool:
        return self.all_gather_impl is not None

    @contextmanager
    def capture(self):
        yield

    def should_all_gather(self, input_: torch.Tensor, output=None) -> bool:
        return self.all_gather_impl.should_all_gather(input_, output)

    def all_gather(self, input_: torch.Tensor, output=None) -> torch.Tensor:
        return self.all_gather_impl.all_gather(input_, output)

    def prepare_checkpoint(self) -> None:
        if not self.has_all_reduce:
            raise RuntimeError("native checkpointable all-reduce is unavailable")
        self.all_reduce.prepare_checkpoint()
        self.all_gather_impl.prepare_checkpoint()

    def restore_after_checkpoint(self) -> None:
        if not self.has_all_reduce:
            raise RuntimeError("native checkpointable all-reduce is unavailable")
        self.all_reduce.restore_after_checkpoint()
        self.all_gather_impl.restore_after_checkpoint()
        dist.barrier(group=self.all_reduce.control_group)

    def set_control_group(self, group: Any) -> None:
        self.all_reduce.set_control_group(group)
        self.all_gather_impl.set_control_group(group)

    def status(self) -> dict[str, list[int]]:
        return {
            "all_reduce": self.all_reduce.status(),
            "all_gather": self.all_gather_impl.status(),
        }

    def close(self) -> None:
        self.all_gather_impl.close()


def create_checkpoint_collectives(
    group: Any,
    device: torch.device,
    group_name: str,
    all_reduce: Any,
) -> NativeCheckpointCollectives | None:
    if not envs.SGLANG_CRIU_SUSPEND_DEVICE_PROCESS_GROUP.get():
        return None
    if group_name not in ("tp", "attention_tp"):
        return None
    all_gather = SymmetricAllGather(
        group,
        device,
        max_elems=envs.SGLANG_CRIU_ALL_GATHER_MAX_ELEMS.get(),
    )
    return NativeCheckpointCollectives(all_reduce, all_gather)
