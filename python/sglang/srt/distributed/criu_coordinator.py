from __future__ import annotations

import logging
import json
import os
import time
from enum import Enum, auto
from pathlib import Path
from typing import Any, Iterable

import torch
import zmq

from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import RpcReqInput

logger = logging.getLogger(__name__)


class CheckpointState(Enum):
    READY = auto()
    PREPARING = auto()
    PREPARED = auto()
    RESTORING = auto()
    FAILED = auto()


class CriuCheckpointCoordinator:
    """Own the validated dense-TP lifecycle around an external checkpoint."""

    PREPARE_RPC = "prepare_criu"
    RESTORE_RPC = "restore_after_criu"

    def __init__(self, *, device: torch.device, groups: Iterable[Any]):
        self.device = device
        self.groups = tuple(dict.fromkeys(groups))
        self.state = CheckpointState.READY
        self.rank = torch.distributed.get_rank()
        self.world_size = torch.distributed.get_world_size()
        self._checkpoint_rpc_generation = 0
        if envs.SGLANG_CRIU_SUSPEND_DEVICE_PROCESS_GROUP.get():
            from sglang.srt.distributed.criu_process_groups import (
                registered_groups,
                validate_checkpoint_topology,
            )

            validate_checkpoint_topology(registered_groups())

    def collective_status(
        self,
        phase: str,
        *,
        require_clean: bool,
    ) -> dict[str, list[int]]:
        statuses = {}
        for index, collectives in enumerate(self._collective_sets()):
            for name, status in collectives.status().items():
                statuses[f"{index}.{name}"] = status
        logger.info("Movin collective status phase=%s: %s", phase, statuses)
        failures = {
            name: status
            for name, status in statuses.items()
            if len(status) < 3 or status[1] != 0 or status[2] != 0
        }
        if require_clean and failures:
            raise RuntimeError(f"Movin collective failure during {phase}: {failures}")
        return statuses

    def prepare(self) -> None:
        from sglang.srt.distributed.parallel_state import (
            suspend_cpu_process_groups,
            suspend_device_process_group,
            wait_for_process_group_teardown,
        )
        from sglang.srt.layers.flashinfer_comm_fusion import (
            prepare_flashinfer_workspaces_for_criu,
        )
        from sglang.srt.distributed.criu_process_groups import (
            registered_groups,
            validate_checkpoint_topology,
        )

        if self.state is not CheckpointState.READY:
            raise RuntimeError(f"cannot prepare CRIU from state {self.state.name}")
        validate_checkpoint_topology(registered_groups())
        self.collective_status("preflight", require_clean=True)
        self.state = CheckpointState.PREPARING
        try:
            torch.cuda.synchronize(self.device)
            prepare_flashinfer_workspaces_for_criu()
            for collectives in self._collective_sets():
                collectives.prepare_criu()
            torch.cuda.synchronize(self.device)
            suspend_cpu_process_groups()
            suspend_device_process_group()
            wait_for_process_group_teardown()
        except Exception:
            self.state = CheckpointState.FAILED
            logger.exception("CRIU prepare failed; worker lifecycle is terminal")
            raise
        self.state = CheckpointState.PREPARED

    def restore(self) -> None:
        from sglang.srt.distributed.parallel_state import (
            resume_cpu_process_groups,
            resume_device_process_group,
        )
        from sglang.srt.layers.flashinfer_comm_fusion import (
            restore_flashinfer_workspaces_after_criu,
        )

        if self.state is not CheckpointState.PREPARED:
            raise RuntimeError(f"cannot restore CRIU from state {self.state.name}")
        self.state = CheckpointState.RESTORING
        try:
            resume_device_process_group()
            resume_cpu_process_groups()
            restore_flashinfer_workspaces_after_criu()
            for collectives in self._collective_sets():
                collectives.restore_after_criu()
            torch.cuda.synchronize(self.device)
            self.collective_status("restored", require_clean=True)
        except Exception:
            self.state = CheckpointState.FAILED
            logger.exception("CRIU restore failed; worker lifecycle is terminal")
            raise
        self.state = CheckpointState.READY

    def should_barrier_after_rpc(self, method: str, *, success: bool) -> bool:
        return success and method != self.PREPARE_RPC

    def converge_rpc_result(
        self,
        method: str,
        *,
        success: bool,
        error: str,
        timeout: float = 120.0,
    ) -> tuple[bool, str]:
        """Give every rank the same checkpoint RPC outcome without a PG."""
        if method not in (self.PREPARE_RPC, self.RESTORE_RPC):
            return success, error
        store_base = envs.SGLANG_CRIU_DEVICE_STORE.get()
        if not store_base:
            raise RuntimeError(
                "SGLANG_CRIU_DEVICE_STORE is required for checkpoint RPC consensus"
            )

        self._checkpoint_rpc_generation += 1
        prefix = Path(f"{store_base}.rpc.{self._checkpoint_rpc_generation}.{method}")
        result_path = Path(f"{prefix}.result.{self.rank}.json")
        temporary = Path(f"{result_path}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps({"success": success, "error": error}) + "\n")
        temporary.replace(result_path)

        result_paths = [
            Path(f"{prefix}.result.{rank}.json") for rank in range(self.world_size)
        ]
        self._wait_for_paths(result_paths, timeout, "checkpoint RPC results")
        results = [json.loads(path.read_text()) for path in result_paths]
        failures = [
            f"rank {rank}: {result['error'] or 'unknown failure'}"
            for rank, result in enumerate(results)
            if not result["success"]
        ]

        ack_path = Path(f"{prefix}.ack.{self.rank}")
        ack_path.touch()
        if self.rank == 0:
            ack_paths = [
                Path(f"{prefix}.ack.{rank}") for rank in range(self.world_size)
            ]
            self._wait_for_paths(ack_paths, timeout, "checkpoint RPC acknowledgements")
            for path in (*result_paths, *ack_paths):
                path.unlink(missing_ok=True)

        if failures:
            return False, "; ".join(failures)
        return True, ""

    @staticmethod
    def _wait_for_paths(
        paths: list[Path],
        timeout: float,
        description: str,
    ) -> None:
        deadline = time.monotonic() + timeout
        while not all(path.exists() for path in paths):
            if time.monotonic() >= deadline:
                missing = [str(path) for path in paths if not path.exists()]
                raise TimeoutError(f"Timed out waiting for {description}: {missing}")
            time.sleep(0.01)

    def _collective_sets(self):
        seen = set()
        for group in self.groups:
            collectives = getattr(group, "movin_collectives", None)
            if collectives is not None and id(collectives) not in seen:
                seen.add(id(collectives))
                yield collectives


def receive_restore_request(
    *,
    tp_group: Any,
    recv_from_rpc: Any,
) -> list[RpcReqInput]:
    """Bridge the restore RPC while Gloo groups and their sockets are absent."""
    store_base = envs.SGLANG_CRIU_DEVICE_STORE.get()
    if not store_base:
        raise RuntimeError(
            "SGLANG_CRIU_DEVICE_STORE is required while CPU groups are suspended"
        )
    generation = tp_group._cpu_group_generation + 1
    resume_path = Path(f"{store_base}.{generation}.resume")
    resume_req = None
    if tp_group.is_first_rank and recv_from_rpc is not None:
        try:
            resume_req = recv_from_rpc.recv_pyobj(zmq.NOBLOCK)
        except zmq.ZMQError:
            pass
        if resume_req is not None:
            if not (
                isinstance(resume_req, RpcReqInput)
                and resume_req.method == CriuCheckpointCoordinator.RESTORE_RPC
            ):
                raise RuntimeError(
                    "Only restore_after_criu is accepted while CPU process "
                    "groups are suspended"
                )
            resume_path.touch()

    if not resume_path.exists():
        return []
    if resume_req is None:
        resume_req = RpcReqInput(
            method=CriuCheckpointCoordinator.RESTORE_RPC,
            parameters={},
        )
    return [resume_req]
