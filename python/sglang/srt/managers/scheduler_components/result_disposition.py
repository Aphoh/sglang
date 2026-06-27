from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

import torch

from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.observability.metrics_collector import DPCooperationInfo


class ResultDisposition(str, Enum):
    PROCESS = "process"
    DISCARD = "discard"


@dataclass(frozen=True)
class DecodeMetricsBatchView:
    reqs: list[Req]
    seq_lens_cpu: torch.Tensor | None
    dp_cooperation_info: DPCooperationInfo | None
    forward_iter: int | None

    def batch_size(self) -> int:
        return len(self.reqs)


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
    def decode_metrics_view(
        batch: ScheduleBatch, discarded_reqs: list[Req]
    ) -> ScheduleBatch | DecodeMetricsBatchView:
        if not discarded_reqs:
            return batch
        keep_indices = [
            i
            for i, req in enumerate(batch.reqs)
            if all(req is not discarded_req for discarded_req in discarded_reqs)
        ]
        return DecodeMetricsBatchView(
            reqs=[batch.reqs[i] for i in keep_indices],
            seq_lens_cpu=(
                batch.seq_lens_cpu[keep_indices]
                if batch.seq_lens_cpu is not None
                else None
            ),
            dp_cooperation_info=batch.dp_cooperation_info,
            forward_iter=batch.forward_iter,
        )
