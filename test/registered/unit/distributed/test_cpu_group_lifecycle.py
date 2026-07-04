"""CPU-only coverage for transactional process-group renewal."""

from __future__ import annotations

import sys
import time
import types
from datetime import timedelta
from functools import partial

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.distributed.cpu_group_lifecycle import (
    CpuGroupBinding,
    CpuGroupLifecycleError,
    CpuGroupRecipe,
    CpuGroupState,
    CpuGroupTransaction,
)
from sglang.srt.utils.network import get_free_port
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

_PG_TIMEOUT = timedelta(seconds=8)
_PROCESS_TIMEOUT = 40.0


def _recipe(group_ranks=((0,),)) -> CpuGroupRecipe:
    return CpuGroupRecipe(
        group_ranks=tuple(tuple(ranks) for ranks in group_ranks),
        torch_distributed_backend="gloo",
        gloo_timeout=_PG_TIMEOUT,
        model_parallel_timeout=None,
    )


class _Participant:
    name = "test_resource"

    def __init__(self, *, fail_resume=False):
        self.fail_resume = fail_resume
        self.suspend_calls = 0

    def preflight(self, _group):
        return None

    def suspend(self):
        self.suspend_calls += 1

    def resume(self, _group):
        if self.fail_resume:
            raise RuntimeError("attach failed")


def _local_consensus(_phase, payload):
    return [payload]


def _transaction(*, participant=None, consensus=_local_consensus):
    old_group, candidate = object(), object()
    binding = CpuGroupBinding(
        name="test:0",
        recipe=_recipe(),
        group=old_group,
        active_ranks=torch.ones(1, dtype=torch.int32),
    )
    if participant is not None:
        binding.add_blocker(participant.name)
        binding.register_participant(participant)
    destroyed = []
    transaction = CpuGroupTransaction(
        [binding],
        rank=0,
        create_group=lambda _recipe, _rank: (
            candidate,
            torch.ones(1, dtype=torch.int32),
        ),
        destroy_group=destroyed.append,
        consensus=consensus,
    )
    return binding, transaction, destroyed, old_group, candidate


def test_attach_failure_is_fail_closed_and_cleans_candidate():
    participant = _Participant(fail_resume=True)
    binding, transaction, destroyed, old_group, candidate = _transaction(
        participant=participant
    )

    transaction.suspend()
    assert destroyed == [old_group]

    with pytest.raises(CpuGroupLifecycleError, match="attach failed"):
        transaction.resume()

    assert destroyed == [old_group, candidate]
    assert binding.state is CpuGroupState.TERMINAL
    assert participant.suspend_calls == 2
    with pytest.raises(CpuGroupLifecycleError, match="TERMINAL"):
        _ = binding.group


def test_rank_local_create_failure_requires_restart_and_skips_group_teardown():
    def peer_fails(phase, payload):
        if phase == "resume.test:0.group":
            return [payload, (False, None, "peer create failed")]
        return [payload, payload]

    binding, transaction, destroyed, old_group, _candidate = _transaction(
        consensus=peer_fails
    )
    transaction.suspend()

    with pytest.raises(CpuGroupLifecycleError, match="restart required"):
        transaction.resume()
    assert destroyed == [old_group]

    binding.close()
    assert destroyed == [old_group]
    assert binding.state is CpuGroupState.CLOSED


def test_message_queue_readiness_handshake_is_bounded():
    from sglang.srt.distributed.device_communicators.shm_broadcast import MessageQueue

    class NeverReadySocket:
        def poll(self, timeout):
            return 0

    message_queue = MessageQueue.__new__(MessageQueue)
    message_queue._is_writer = True
    message_queue.n_local_reader = 1
    message_queue.n_remote_reader = 0
    message_queue.local_socket = NeverReadySocket()

    with pytest.raises(TimeoutError, match="readiness handshake"):
        message_queue.wait_until_ready(timeout=0.01)


def _assert_full_group_collective(group, rank, cycle):
    value = torch.tensor([rank + cycle], dtype=torch.int64)
    dist.all_reduce(value, group=group)
    assert value.item() == 1 + 2 * cycle


def _install_optional_communicator_stubs():
    class Stub(types.ModuleType):
        def __getattr__(self, name):
            if name.startswith("__"):
                raise AttributeError(name)
            return lambda *args, **kwargs: None

    prefix = "sglang.srt.distributed.device_communicators."
    names = (
        "custom_all_reduce pymscclpp pynccl pynccl_allocator torch_symm_mem "
        "hpu_communicator npu_communicator xpu_communicator"
    ).split()
    for name in names:
        sys.modules[prefix + name] = Stub(prefix + name)
    sys.modules["sglang.srt.layers.dp_attention"] = Stub("dp_attention")


def _run_lifecycle_rank(rank, port):
    coordinators = []
    try:
        dist.init_process_group(
            "gloo",
            init_method=f"tcp://127.0.0.1:{port}",
            rank=rank,
            world_size=2,
            timeout=_PG_TIMEOUT,
        )
        _install_optional_communicator_stubs()

        import sglang.srt.distributed.parallel_state as parallel_state
        from sglang.srt.distributed.parallel_state import GroupCoordinator

        parallel_state.is_cuda_alike = lambda: False
        parallel_state._is_npu = False
        parallel_state._is_xpu = False
        parallel_state._is_musa = False

        coordinator = partial(
            GroupCoordinator,
            local_rank=rank,
            torch_distributed_backend="gloo",
            use_pynccl=False,
            use_pymscclpp=False,
            use_custom_allreduce=False,
            use_torch_symm_mem_all_reduce=False,
            use_hpu_communicator=False,
            use_xpu_communicator=False,
            use_npu_communicator=False,
            gloo_timeout=_PG_TIMEOUT,
        )
        full = coordinator(
            group_ranks=[[0, 1]],
            group_name="renew_full",
            use_message_queue_broadcaster=True,
        )
        coordinators.append(full)
        singleton = coordinator(group_ranks=[[0], [1]], group_name="renew_singleton")
        coordinators.append(singleton)

        world = dist.group.WORLD
        device_groups = (full.device_group, singleton.device_group)
        bindings = [full.cpu_group_lifecycle, singleton.cpu_group_lifecycle]
        if rank:
            bindings.reverse()  # Transaction order must not depend on callers.
        transaction = CpuGroupTransaction(bindings)

        old_cpu = full.cpu_group
        if rank == 0:
            full.cpu_group_lifecycle.add_blocker("rank_zero_blocker")
        with pytest.raises(CpuGroupLifecycleError) as exc_info:
            transaction.suspend()
        assert "rank 0" in str(exc_info.value)
        assert "rank_zero_blocker" in str(exc_info.value)
        if rank == 0:
            full.cpu_group_lifecycle.remove_blocker("rank_zero_blocker")

        assert all(binding.state is CpuGroupState.ACTIVE for binding in bindings)
        assert full.cpu_group is old_cpu
        _assert_full_group_collective(full.cpu_group, rank, 0)

        for cycle in range(1, 3):
            old_groups = (full.cpu_group, singleton.cpu_group)
            old_queue = full.mq_broadcaster
            assert old_queue is not None
            payload = {"cycle": cycle} if rank == 0 else None
            assert full.broadcast_object(payload) == {"cycle": cycle}

            transaction.suspend()
            assert all(binding.state is CpuGroupState.SUSPENDED for binding in bindings)
            assert dist.is_initialized() and dist.group.WORLD is world
            assert (full.device_group, singleton.device_group) == device_groups
            assert old_queue.local_socket is None
            assert old_queue.remote_socket is None
            assert old_queue.buffer is None
            old_queue.close()  # Explicit resource cleanup must be idempotent.

            transaction.resume()
            assert all(binding.state is CpuGroupState.ACTIVE for binding in bindings)
            assert full.cpu_group is not old_groups[0]
            assert singleton.cpu_group is not old_groups[1]
            assert full.mq_broadcaster is not None
            assert full.mq_broadcaster is not old_queue
            _assert_full_group_collective(full.cpu_group, rank, cycle)

            singleton_value = torch.tensor([rank], dtype=torch.int64)
            dist.all_reduce(singleton_value, group=singleton.cpu_group)
            assert singleton_value.item() == rank
    finally:
        for coordinator in reversed(coordinators):
            coordinator.destroy()
        if dist.is_initialized():
            dist.destroy_process_group()


def test_two_rank_gloo_lifecycle_is_converged_and_repeatable():
    port = get_free_port()
    context = mp.spawn(_run_lifecycle_rank, args=(port,), nprocs=2, join=False)
    deadline = time.monotonic() + _PROCESS_TIMEOUT
    try:
        while not context.join(timeout=1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("two-rank lifecycle test exceeded watchdog")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=3)
            if process.is_alive():
                process.kill()
                process.join(timeout=3)
