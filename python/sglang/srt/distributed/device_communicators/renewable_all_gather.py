"""CUDA all-gather ownership across CPU process-group renewal."""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

import sglang.srt.distributed.device_communicators.custom_all_reduce_ops as ops
from sglang.srt.distributed.cpu_group_lifecycle import CpuGroupBinding
from sglang.srt.distributed.device_communicators.renewable_vmm import (
    RenewableVmmBuffer,
)


class RenewableAllGather:
    """Own the stable virtual addresses used by sgl-kernel all-gather."""

    name = "custom_all_gather"

    def __init__(
        self,
        binding: CpuGroupBinding,
        device: torch.device,
        *,
        max_bytes: int,
    ) -> None:
        if not ops.IS_CUSTOM_AG_AVAILABLE:
            raise RuntimeError("sgl-kernel custom all-gather is unavailable")
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")

        group = binding.group
        self.device = device
        self.rank = dist.get_rank(group)
        self.world_size = dist.get_world_size(group)
        self.max_bytes = max_bytes
        self._ticket = torch.empty(1, dtype=torch.uint64, device=device)
        self._capture_tickets: list[torch.Tensor] = []
        self._closed = False

        binding.add_blocker(self.name)
        try:
            workspace_size = ops.custom_all_gather_workspace_size(
                max_bytes, self.world_size
            )
            self.memory = RenewableVmmBuffer(workspace_size, group, device)
            self._initialize()
            dist.barrier(group=group)
            binding.register_participant(self)
        except Exception:
            memory = getattr(self, "memory", None)
            if memory is not None:
                memory.close()
            binding.remove_blocker(self.name)
            raise

    def should_all_gather(
        self, input_: torch.Tensor, output: torch.Tensor | None = None
    ) -> bool:
        return (
            not self._closed
            and self.memory.attached
            and input_.dim() > 0
            and input_.dtype in (torch.float16, torch.bfloat16, torch.float32)
            and input_.device == self.device
            and input_.is_contiguous()
            and 0 < input_.numel() * input_.element_size() <= self.max_bytes
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
        self, input_: torch.Tensor, output: torch.Tensor | None = None
    ) -> torch.Tensor:
        if not self.should_all_gather(input_, output):
            raise RuntimeError("custom all-gather does not support this input")
        if output is None:
            output = torch.empty(
                (input_.shape[0] * self.world_size, *input_.shape[1:]),
                dtype=input_.dtype,
                device=input_.device,
            )
        ticket = self._ticket
        if torch.cuda.is_current_stream_capturing():
            ticket = torch.empty(1, dtype=torch.uint64, device=self.device)
            self._capture_tickets.append(ticket)
        ops.custom_all_gather(
            input_,
            output,
            ticket,
            list(self.memory.ptrs),
            self.rank,
            self.max_bytes,
        )
        return output

    def status(self) -> list[int]:
        if self._closed or not self.memory.attached:
            raise RuntimeError("custom all-gather is not attached")
        return ops.custom_all_gather_status(
            self._ticket,
            self.memory.local_ptr,
            self.max_bytes,
            self.world_size,
        )

    def preflight(self, group: ProcessGroup) -> None:
        if self._closed:
            return
        if not self.memory.attached or self.memory.control_group is not group:
            raise RuntimeError("custom all-gather is not attached to the CPU group")

    def suspend(self) -> None:
        if self._closed:
            return
        torch.cuda.synchronize(self.device)
        self.memory.detach()
        self.memory.set_control_group(None)

    def resume(self, group: ProcessGroup) -> None:
        if self._closed:
            return
        if (
            dist.get_rank(group) != self.rank
            or dist.get_world_size(group) != self.world_size
        ):
            raise RuntimeError("restored all-gather group geometry changed")
        self.memory.set_control_group(group)
        self.memory.restore()
        self._initialize()
        dist.barrier(group=group)

    def close(self) -> None:
        if self._closed:
            return
        torch.cuda.synchronize(self.device)
        self._capture_tickets.clear()
        self.memory.close()
        self.memory.set_control_group(None)
        self._closed = True

    def _initialize(self) -> None:
        ops.custom_all_gather_initialize(
            self._ticket,
            self.memory.local_ptr,
            self.max_bytes,
            self.world_size,
        )
