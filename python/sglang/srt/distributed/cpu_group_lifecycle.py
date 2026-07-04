"""Transactional CPU-group lifecycle using the live WORLD group as control."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from enum import Enum, auto
from functools import partial
from typing import Callable, Iterable, Protocol

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup


class CpuGroupLifecycleError(RuntimeError):
    pass


class CpuGroupState(Enum):
    ACTIVE = auto()
    SUSPENDING = auto()
    SUSPENDED = auto()
    RESTORING = auto()
    FAILED = auto()
    TERMINAL = auto()
    CLOSED = auto()


class CpuGroupParticipant(Protocol):
    """A resource that retains or is constructed from a CPU process group."""

    name: str

    def preflight(self, group: ProcessGroup) -> None: ...
    def suspend(self) -> None: ...
    def resume(self, group: ProcessGroup) -> None: ...


CreateGroup = Callable[["CpuGroupRecipe", int], tuple[ProcessGroup, torch.Tensor]]
DestroyGroup = Callable[[ProcessGroup], None]
Consensus = Callable[[str, object], list[object]]


@dataclass(frozen=True)
class CpuGroupRecipe:
    """Immutable inputs needed to recreate one coordinator's CPU group."""

    group_ranks: tuple[tuple[int, ...], ...]
    torch_distributed_backend: str
    gloo_timeout: timedelta
    model_parallel_timeout: timedelta | None
    recovered_rank: bool = False

    def create_group(self, ranks: tuple[int, ...]) -> tuple[ProcessGroup, torch.Tensor]:
        """Create one subgroup while preserving the caller's global order."""
        ranks_active = torch.ones(len(ranks), dtype=torch.int32)
        if "mooncake" in self.torch_distributed_backend:
            from mooncake.ep import MooncakeBackendOptions

            group = dist.new_group(
                list(ranks),
                backend="mooncake-cpu",
                pg_options=MooncakeBackendOptions(ranks_active, self.recovered_rank),
                timeout=self.model_parallel_timeout,
            )
        else:
            group = dist.new_group(
                list(ranks), backend="gloo", timeout=self.gloo_timeout
            )
        return group, ranks_active

    def create_for_rank(self, rank: int) -> tuple[ProcessGroup, torch.Tensor]:
        active_group = None
        active_ranks = None
        for ranks in self.group_ranks:
            group, ranks_active = self.create_group(ranks)
            if rank in ranks:
                active_group, active_ranks = group, ranks_active
        if active_group is None or active_ranks is None:
            raise RuntimeError(f"rank {rank} has no configured CPU subgroup")
        return active_group, active_ranks


class CpuGroupBinding:
    """Stable, fail-closed access to one renewable CPU process group."""

    def __init__(
        self,
        *,
        name: str,
        recipe: CpuGroupRecipe,
        group: ProcessGroup,
        active_ranks: torch.Tensor,
    ) -> None:
        self.name = name
        self.recipe = recipe
        self._group: ProcessGroup | None = group
        self._active_ranks: torch.Tensor | None = active_ranks
        self._participants: dict[str, CpuGroupParticipant] = {}
        self._blockers: set[str] = set()
        self._state = CpuGroupState.ACTIVE

    @property
    def state(self) -> CpuGroupState:
        return self._state

    @property
    def group(self) -> ProcessGroup:
        if self._state is not CpuGroupState.ACTIVE or self._group is None:
            raise CpuGroupLifecycleError(
                f"CPU group {self.name} is unavailable ({self._state.name})"
            )
        return self._group

    @property
    def active_ranks(self) -> torch.Tensor:
        if self._state is not CpuGroupState.ACTIVE or self._active_ranks is None:
            raise CpuGroupLifecycleError(
                f"CPU group {self.name} is unavailable ({self._state.name})"
            )
        return self._active_ranks

    def register_participant(self, participant: CpuGroupParticipant) -> None:
        if self._state is not CpuGroupState.ACTIVE:
            raise CpuGroupLifecycleError("participants require an active CPU group")
        if not participant.name or participant.name in self._participants:
            raise ValueError(f"duplicate or empty participant: {participant.name!r}")
        self._participants[participant.name] = participant
        self._blockers.discard(participant.name)

    def add_blocker(self, name: str) -> None:
        if not name:
            raise ValueError("CPU-group blocker name must not be empty")
        self._blockers.add(name)

    def remove_blocker(self, name: str) -> None:
        self._blockers.discard(name)

    def _manifest(self) -> tuple[object, ...]:
        return (
            self.name,
            self.recipe,
            tuple(sorted(self._participants)),
            tuple(sorted(self._blockers)),
        )

    def close(self) -> None:
        """Destroy held groups during ordinary coordinator teardown."""
        if self._state is CpuGroupState.CLOSED:
            return
        errors = []
        for participant in reversed(self._ordered_participants()):
            try:
                participant.suspend()
            except Exception as exc:
                errors.append(f"{participant.name}: {_format_error(exc)}")

        # Divergent process groups can only be cleaned safely by process exit.
        if self._state is CpuGroupState.TERMINAL:
            self._group = self._active_ranks = None
            self._state = CpuGroupState.CLOSED
            if errors:
                raise CpuGroupLifecycleError("; ".join(errors))
            return

        current_group = self._group
        if current_group is not None:
            try:
                dist.destroy_process_group(current_group)
            except Exception as exc:
                errors.append(_format_error(exc))
            else:
                self._group = self._active_ranks = None

        if errors:
            self._state = CpuGroupState.FAILED
            raise CpuGroupLifecycleError("; ".join(errors))
        self._state = CpuGroupState.CLOSED

    def _preflight_suspend(self) -> None:
        assert self._group is not None
        if self._group is dist.group.WORLD:
            raise CpuGroupLifecycleError(f"CPU group {self.name} is the control group")
        if self._blockers:
            blockers = ", ".join(sorted(self._blockers))
            raise CpuGroupLifecycleError(
                f"CPU group {self.name} has blockers: {blockers}"
            )
        for participant in self._ordered_participants():
            participant.preflight(self._group)

    def _ordered_participants(self) -> tuple[CpuGroupParticipant, ...]:
        return tuple(self._participants[name] for name in sorted(self._participants))


def _format_error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


class CpuGroupTransaction:
    """Renew CPU groups; post-mutation failure requires worker restart."""

    def __init__(
        self,
        bindings: Iterable[CpuGroupBinding],
        *,
        rank: int | None = None,
        create_group: CreateGroup | None = None,
        destroy_group: DestroyGroup | None = None,
        consensus: Consensus | None = None,
    ) -> None:
        self._bindings = tuple(bindings)
        self._rank = dist.get_rank() if rank is None else rank
        self._create = create_group or (
            lambda recipe, group_rank: recipe.create_for_rank(group_rank)
        )
        self._destroy = destroy_group or dist.destroy_process_group
        self._consensus = consensus or self._distributed_consensus

    def suspend(self) -> None:
        bindings = self._preflight(CpuGroupState.ACTIVE, "suspend")
        for binding in bindings:
            binding._state = CpuGroupState.SUSPENDING
        for binding in bindings:
            for participant in binding._ordered_participants():
                self._exchange(
                    f"suspend.{binding.name}.participant.{participant.name}",
                    participant.suspend,
                )
            self._exchange(
                f"suspend.{binding.name}.group",
                partial(self._suspend, binding),
            )

    def resume(self) -> None:
        bindings = self._preflight(CpuGroupState.SUSPENDED, "resume")
        for binding in bindings:
            binding._state = CpuGroupState.RESTORING
        candidates: dict[CpuGroupBinding, tuple[ProcessGroup, torch.Tensor]] = {}
        attempted: list[CpuGroupParticipant] = []
        try:
            for binding in bindings:
                candidates[binding] = self._create_candidate(binding)
            for binding in bindings:
                group = candidates[binding][0]
                for participant in binding._ordered_participants():
                    attempted.append(participant)
                    self._exchange(
                        f"resume.{binding.name}.participant.{participant.name}",
                        partial(participant.resume, group),
                    )
        except Exception as exc:
            errors = self._cleanup_restore(candidates, attempted)
            message = str(exc)
            if errors:
                message += "; restore cleanup: " + "; ".join(errors)
            raise CpuGroupLifecycleError(message) from exc
        for binding in bindings:
            binding._group, binding._active_ranks = candidates[binding]
            binding._state = CpuGroupState.ACTIVE

    def _preflight(
        self, expected: CpuGroupState, operation: str
    ) -> tuple[CpuGroupBinding, ...]:
        bindings = tuple(sorted(self._bindings, key=lambda binding: binding.name))

        def local_manifest() -> tuple[object, ...]:
            names = [binding.name for binding in bindings]
            if len(names) != len(set(names)):
                raise CpuGroupLifecycleError("CPU group names must be unique")
            for binding in bindings:
                if binding.state is not expected:
                    raise CpuGroupLifecycleError(
                        f"CPU group {binding.name} is {binding.state.name}, "
                        f"expected {expected.name}"
                    )
                if operation == "suspend":
                    binding._preflight_suspend()
            return tuple(binding._manifest() for binding in bindings)

        manifests = self._exchange(
            f"{operation}.preflight", local_manifest, mutate_started=False
        )
        if any(manifest != manifests[0] for manifest in manifests[1:]):
            raise CpuGroupLifecycleError(f"{operation} manifest differs across ranks")
        return bindings

    def _exchange(
        self,
        phase: str,
        action: Callable[[], object],
        *,
        mutate_started: bool = True,
    ) -> list[object]:
        try:
            payload: object = (True, action(), "")
        except Exception as exc:
            payload = (False, None, _format_error(exc))
        try:
            gathered = self._consensus(phase, payload)
        except Exception as exc:
            if mutate_started:
                self._fail()
                raise CpuGroupLifecycleError(
                    f"{phase} consensus failed; worker restart required"
                ) from exc
            raise
        failures = [
            f"rank {rank}: {item[2]}"
            for rank, item in enumerate(gathered)
            if not item[0]
        ]
        if failures:
            if mutate_started:
                self._fail()
            suffix = "; worker restart required" if mutate_started else ""
            raise CpuGroupLifecycleError(
                f"{phase} failed: {'; '.join(failures)}{suffix}"
            )
        return [item[1] for item in gathered]

    def _distributed_consensus(self, _phase: str, payload: object) -> list[object]:
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, payload)
        return gathered

    def _suspend(self, binding: CpuGroupBinding) -> None:
        group = binding._group
        if group is None:
            raise CpuGroupLifecycleError(f"CPU group {binding.name} is unavailable")
        self._destroy(group)
        binding._group = None
        binding._state = CpuGroupState.SUSPENDED

    def _create_candidate(
        self, binding: CpuGroupBinding
    ) -> tuple[ProcessGroup, torch.Tensor]:
        candidate = None

        def create() -> None:
            nonlocal candidate
            candidate = self._create(binding.recipe, self._rank)

        self._exchange(f"resume.{binding.name}.group", create)
        assert candidate is not None
        return candidate

    def _cleanup_restore(
        self,
        candidates: dict[CpuGroupBinding, tuple[ProcessGroup, torch.Tensor]],
        attempted: list[CpuGroupParticipant],
    ) -> list[str]:
        errors = []
        for participant in reversed(attempted):
            try:
                participant.suspend()
            except Exception as exc:
                errors.append(f"{participant.name}: {_format_error(exc)}")
        for binding, (group, _) in reversed(candidates.items()):
            try:
                self._destroy(group)
            except Exception as exc:
                errors.append(f"{binding.name}: {_format_error(exc)}")
        return errors

    def _fail(self) -> None:
        for binding in self._bindings:
            binding._state = CpuGroupState.TERMINAL
