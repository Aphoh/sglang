from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

from sglang.srt.managers.schedule_batch import Req, ScheduleBatch


class ResultDisposition(str, Enum):
    PROCESS = "process"
    DISCARD = "discard"


@dataclass(frozen=True)
class ResultDispositionHandler:
    """Apply request-local result decisions without exposing their owner."""

    get_disposition: Optional[Callable[[Req], ResultDisposition]] = None

    def should_discard(self, req: Req) -> bool:
        return (
            self.get_disposition is not None
            and self.get_disposition(req) is ResultDisposition.DISCARD
        )

    @staticmethod
    def without_discarded(reqs: list[Req], discarded_reqs: list[Req]) -> list[Req]:
        return [
            req
            for req in reqs
            if all(req is not discarded_req for discarded_req in discarded_reqs)
        ]

    @staticmethod
    def batch_without_discarded(
        batch: ScheduleBatch, discarded_reqs: list[Req]
    ) -> ScheduleBatch:
        if not discarded_reqs:
            return batch
        filtered_batch = batch.copy()
        filtered_batch.filter_batch(
            keep_indices=[
                i
                for i, req in enumerate(batch.reqs)
                if all(req is not discarded_req for discarded_req in discarded_reqs)
            ]
        )
        return filtered_batch
