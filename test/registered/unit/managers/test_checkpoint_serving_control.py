from __future__ import annotations

from types import SimpleNamespace

import pytest

import sglang.srt.managers.scheduler as scheduler_module
from sglang.srt.distributed.checkpoint_lifecycle import CheckpointState
from sglang.srt.managers.io_struct import RpcReqInput, RpcReqOutput
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Lifecycle:
    def __init__(self, state=CheckpointState.READY):
        self.state = state
        self.calls = []

    def suspend(self):
        self.calls.append("suspend")
        self.state = CheckpointState.SUSPENDED

    def resume(self):
        self.calls.append("resume")
        self.state = CheckpointState.READY


class _Socket:
    def __init__(self, request=None):
        self.request = request

    def recv_pyobj(self, flags):
        if self.request is None:
            raise scheduler_module.zmq.Again()
        request, self.request = self.request, None
        return request


def _scheduler(tmp_path, *, state=CheckpointState.READY, request=None):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.checkpoint_lifecycle = _Lifecycle(state)
    scheduler._checkpoint_resume_path = tmp_path / "resume"
    scheduler.ipc_channels = SimpleNamespace(recv_from_rpc=_Socket(request))
    scheduler.is_fully_idle = lambda: True
    return scheduler


def test_criu_server_args_are_dense_tp_only():
    args = ServerArgs(model_path="dummy", enable_criu_checkpoint=True)
    assert args.criu_store_prefix

    with pytest.raises(ValueError, match="pipeline parallelism"):
        ServerArgs(
            model_path="dummy",
            enable_criu_checkpoint=True,
            pp_size=2,
        )


def test_suspend_requires_idle_and_clears_stale_resume_marker(monkeypatch, tmp_path):
    scheduler = _scheduler(tmp_path)
    scheduler._checkpoint_resume_path.touch()
    barriers = []
    monkeypatch.setattr(scheduler_module.torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(scheduler_module, "barrier", lambda: barriers.append(True))

    scheduler.suspend_checkpoint()

    assert scheduler.checkpoint_lifecycle.calls == ["suspend"]
    assert barriers == [True]
    assert not scheduler._checkpoint_resume_path.exists()

    scheduler.checkpoint_lifecycle.state = CheckpointState.READY
    scheduler.is_fully_idle = lambda: False
    with pytest.raises(RuntimeError, match="idle scheduler"):
        scheduler.suspend_checkpoint()


def test_suspended_loop_accepts_only_resume(monkeypatch, tmp_path):
    request = RpcReqInput(method="save_remote_model")
    scheduler = _scheduler(
        tmp_path,
        state=CheckpointState.SUSPENDED,
        request=request,
    )
    outputs = []
    monkeypatch.setattr(
        scheduler_module, "sock_send", lambda _socket, output: outputs.append(output)
    )

    assert scheduler._poll_checkpoint_resume()
    assert outputs == [
        RpcReqOutput(
            success=False,
            message="only resume_checkpoint is accepted while suspended",
        )
    ]
    assert scheduler.checkpoint_lifecycle.state is CheckpointState.SUSPENDED


def test_resume_request_wakes_every_rank_via_marker(monkeypatch, tmp_path):
    request = RpcReqInput(method="resume_checkpoint")
    scheduler = _scheduler(
        tmp_path,
        state=CheckpointState.SUSPENDED,
        request=request,
    )
    outputs = []
    monkeypatch.setattr(
        scheduler_module, "sock_send", lambda _socket, output: outputs.append(output)
    )
    scheduler.handle_rpc_request = lambda req: RpcReqOutput(
        success=req.method == "resume_checkpoint", message=""
    )

    assert scheduler._poll_checkpoint_resume()
    assert scheduler._checkpoint_resume_path.exists()
    assert outputs == [RpcReqOutput(success=True, message="")]

    follower = _scheduler(tmp_path, state=CheckpointState.SUSPENDED)
    seen = []
    follower.handle_rpc_request = lambda req: seen.append(req.method) or RpcReqOutput(
        success=True, message=""
    )
    assert follower._poll_checkpoint_resume()
    assert seen == ["resume_checkpoint"]
