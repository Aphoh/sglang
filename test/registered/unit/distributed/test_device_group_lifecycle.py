from __future__ import annotations

from types import SimpleNamespace

import pytest

import sglang.srt.distributed.device_group_lifecycle as lifecycle
from sglang.srt.distributed.device_group_lifecycle import (
    DefaultDeviceGroupTransaction,
    DeviceGroupLifecycleError,
    DeviceGroupState,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _Group:
    def __init__(
        self,
        name,
        device_group,
        *,
        owns=False,
        cpu_alias=False,
    ):
        self.unique_name = name
        self.device_group = device_group
        self.owns_device_group = owns
        self.device_group_is_cpu_alias = cpu_alias


class _Dist:
    def __init__(self):
        self.world = SimpleNamespace(abort=lambda: None)
        self.group = SimpleNamespace(WORLD=self.world)
        self.initialized = True
        self.manifests = None
        self.init_kwargs = None

    def is_initialized(self):
        return self.initialized

    def get_backend(self, _group):
        return "nccl"

    def get_rank(self, _group=None):
        return 0

    def get_world_size(self, _group=None):
        return 2

    def all_gather_object(self, output, value):
        values = self.manifests or [value, value]
        output[:] = values

    def barrier(self, group=None):
        return None

    def destroy_process_group(self):
        self.initialized = False

    def FileStore(self, path, world_size):
        return (path, world_size)

    def init_process_group(self, **kwargs):
        self.init_kwargs = kwargs
        self.initialized = True
        self.world = object()
        self.group.WORLD = self.world


def _transaction(monkeypatch, groups, fake_dist=None):
    fake_dist = fake_dist or _Dist()
    monkeypatch.setattr(lifecycle, "dist", fake_dist)
    transaction = DefaultDeviceGroupTransaction(
        groups,
        store_prefix="/tmp/device-group-lifecycle-test",
        pg_options_factory=lambda: "options",
    )
    return fake_dist, transaction


def test_suspend_resume_rebinds_world_borrowers(monkeypatch):
    fake_dist = _Dist()
    first = _Group("tp:0", fake_dist.world)
    second = _Group("world:0", fake_dist.world)
    fake_dist, transaction = _transaction(
        monkeypatch, [first, second, first], fake_dist
    )

    transaction.suspend()
    assert transaction.state is DeviceGroupState.SUSPENDED
    assert first.device_group is None and second.device_group is None
    assert not fake_dist.initialized

    transaction.resume()
    assert transaction.state is DeviceGroupState.ACTIVE
    assert first.device_group is fake_dist.world
    assert second.device_group is fake_dist.world
    assert fake_dist.init_kwargs["pg_options"] == "options"


def test_owned_device_group_fails_before_mutation(monkeypatch):
    fake_dist = _Dist()
    borrower = _Group("tp:0", fake_dist.world)
    owned = _Group("pp:0", object(), owns=True)
    _, transaction = _transaction(monkeypatch, [borrower, owned], fake_dist)

    with pytest.raises(DeviceGroupLifecycleError, match="owned or foreign"):
        transaction.suspend()

    assert transaction.state is DeviceGroupState.ACTIVE
    assert borrower.device_group is fake_dist.world
    assert fake_dist.initialized


def test_rank_divergent_manifest_fails_before_mutation(monkeypatch):
    fake_dist = _Dist()
    borrower = _Group("tp:0", fake_dist.world)
    fake_dist.manifests = [("tp:0",), ("different:0",)]
    _, transaction = _transaction(monkeypatch, [borrower], fake_dist)

    with pytest.raises(DeviceGroupLifecycleError, match="manifest differs"):
        transaction.suspend()

    assert transaction.state is DeviceGroupState.ACTIVE
    assert borrower.device_group is fake_dist.world
