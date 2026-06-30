"""Request-local state for decode migration."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.disaggregation.base import BaseKVSender
    from sglang.srt.managers.io_struct import (
        PrepareDecodeMigrationReqInput,
        QuiesceDecodeMigrationReqInput,
    )
    from sglang.srt.managers.schedule_batch import Req


class DecodeMigrationState(str, Enum):
    BOOTSTRAPPING = "bootstrapping"
    TRANSFERRING = "transferring"


@dataclass
class PreparedDecodeMigrationSource:
    request: PrepareDecodeMigrationReqInput
    sender: BaseKVSender
    pending_quiesce: QuiesceDecodeMigrationReqInput | None = None

    @property
    def migration_id(self) -> str:
        return self.request.migration_id

    @property
    def rid(self) -> str:
        return self.request.rid


@dataclass
class DecodeMigrationTransfer:
    migration_id: str
    req: Req
    sender: BaseKVSender
    metadata_buffer_index: int
    committed_len: int
    logical_len: int
    output_tokens_seen: int
    created_at: float
    state: DecodeMigrationState = DecodeMigrationState.BOOTSTRAPPING

    @property
    def rid(self) -> str:
        return self.req.rid


@dataclass(frozen=True)
class AwaitingStaleDecodeResult:
    migration_id: str
    req: Req

    @property
    def rid(self) -> str:
        return self.req.rid


DecodeMigrationEntry = (
    PreparedDecodeMigrationSource | DecodeMigrationTransfer | AwaitingStaleDecodeResult
)


@dataclass
class DecodeMigrationRegistry:
    """Own source migration lifecycle state indexed by migration and request."""

    _by_migration_id: dict[str, DecodeMigrationEntry] = field(default_factory=dict)
    _migration_id_by_rid: dict[str, str] = field(default_factory=dict)

    def get(self, migration_id: str) -> DecodeMigrationEntry | None:
        return self._by_migration_id.get(migration_id)

    def get_for_rid(self, rid: str) -> DecodeMigrationEntry | None:
        migration_id = self._migration_id_by_rid.get(rid)
        return self.get(migration_id) if migration_id is not None else None

    def prepare(
        self, request: PrepareDecodeMigrationReqInput, sender: BaseKVSender
    ) -> PreparedDecodeMigrationSource:
        existing = self.get(request.migration_id)
        if existing is not None:
            assert existing.rid == request.rid
            assert isinstance(existing, PreparedDecodeMigrationSource)
            return existing
        assert self.get_for_rid(request.rid) is None
        prepared = PreparedDecodeMigrationSource(request, sender)
        self._by_migration_id[prepared.migration_id] = prepared
        self._migration_id_by_rid[prepared.rid] = prepared.migration_id
        return prepared

    def activate(self, record: DecodeMigrationTransfer) -> None:
        existing = self.get(record.migration_id)
        if existing is not None:
            assert existing.rid == record.req.rid
        by_rid = self.get_for_rid(record.req.rid)
        if by_rid is not None:
            assert by_rid.migration_id == record.migration_id
        self._by_migration_id[record.migration_id] = record
        self._migration_id_by_rid[record.req.rid] = record.migration_id

    def await_stale_result(self, record: DecodeMigrationTransfer) -> None:
        self._by_migration_id[record.migration_id] = AwaitingStaleDecodeResult(
            record.migration_id, record.req
        )

    def discard(self, migration_id: str) -> DecodeMigrationEntry | None:
        entry = self._by_migration_id.pop(migration_id, None)
        if (
            entry is not None
            and self._migration_id_by_rid.get(entry.rid) == migration_id
        ):
            self._migration_id_by_rid.pop(entry.rid, None)
        return entry

    def prepared_sources(self) -> tuple[PreparedDecodeMigrationSource, ...]:
        return tuple(
            entry
            for entry in self._by_migration_id.values()
            if isinstance(entry, PreparedDecodeMigrationSource)
        )

    def transfers(self) -> tuple[DecodeMigrationTransfer, ...]:
        return tuple(
            entry
            for entry in self._by_migration_id.values()
            if isinstance(entry, DecodeMigrationTransfer)
        )

    def kv_owner_reqs(self) -> tuple[Req, ...]:
        """Return parked requests whose KV remains owned by migration state."""
        return tuple(
            entry.req
            for entry in self._by_migration_id.values()
            if isinstance(entry, (DecodeMigrationTransfer, AwaitingStaleDecodeResult))
        )

    def has_active_transfers(self) -> bool:
        return any(
            isinstance(entry, DecodeMigrationTransfer)
            for entry in self._by_migration_id.values()
        )

    def get_result_owner_for_req(
        self, req: Req
    ) -> DecodeMigrationTransfer | AwaitingStaleDecodeResult | None:
        entry = self.get_for_rid(req.rid)
        return (
            entry
            if isinstance(entry, (DecodeMigrationTransfer, AwaitingStaleDecodeResult))
            and entry.req is req
            else None
        )


@dataclass(frozen=True)
class DecodeMigrationFrontier:
    committed_input_ids: list[int]
    pending_input_id: int
    unforwarded_committed_output_ids: list[int]
    prompt_len: int
    committed_len: int
    logical_len: int
    output_tokens_seen: int


def build_decode_migration_frontier(
    prompt_ids: list[int],
    output_ids: list[int],
    committed_len: int,
    output_tokens_seen: int,
) -> DecodeMigrationFrontier:
    """Build the exact committed-KV and sampled-token frontiers.

    The non-speculative prototype requires exactly one sampled token beyond the
    committed KV frontier. ``output_tokens_seen`` is a frontend stream watermark,
    not a KV watermark.
    """
    logical_ids = prompt_ids + output_ids
    logical_len = len(logical_ids)
    if logical_len != committed_len + 1:
        raise ValueError(
            "Expected exactly one sampled token beyond committed KV, got "
            f"logical_len={logical_len}, committed_len={committed_len}"
        )

    prompt_len = len(prompt_ids)
    committed_output_count = max(0, committed_len - prompt_len)
    seen = min(max(0, output_tokens_seen), len(output_ids))
    unforwarded_end = min(committed_output_count, len(output_ids))
    return DecodeMigrationFrontier(
        committed_input_ids=logical_ids[:committed_len],
        pending_input_id=logical_ids[committed_len],
        unforwarded_committed_output_ids=output_ids[
            min(seen, unforwarded_end) : unforwarded_end
        ],
        prompt_len=prompt_len,
        committed_len=committed_len,
        logical_len=logical_len,
        output_tokens_seen=seen,
    )
