"""Opt-in structured timing events for decode migration."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)


def enabled() -> bool:
    return os.environ.get("DYNAMO_DECODE_MIGRATION_TRACE", "").lower() in {
        "1",
        "true",
        "yes",
    }


def trace_scheduler(
    scheduler: "Scheduler",
    role: str,
    stage: str,
    *,
    rid: str,
    migration_id: Optional[str] = None,
    **fields,
) -> None:
    """Emit a machine-readable migration event from one TP rank per DP group.

    Wall time joins logs across the frontend and worker pods. Local durations
    should use the monotonic timing fields included in the event payload.
    """

    if not enabled():
        return
    parallel_state = scheduler.ps
    if getattr(parallel_state, "tp_rank", 0) != 0:
        return
    logger.info(
        "decode_migration_trace role=%s stage=%s rid=%s migration_id=%s "
        "wall_time_ns=%d mono_time_ns=%d dp_rank=%s tp_rank=%s fields=%s",
        role,
        stage,
        rid,
        migration_id or "unknown",
        time.time_ns(),
        time.monotonic_ns(),
        getattr(parallel_state, "dp_rank", 0),
        getattr(parallel_state, "tp_rank", 0),
        json.dumps(fields, separators=(",", ":"), sort_keys=True),
    )


def trace_destination_request(
    scheduler: "Scheduler",
    stage: str,
    req: Any,
    **fields,
) -> None:
    """Trace only requests created by the decode-migration destination path."""

    if not getattr(req, "is_decode_migration_destination", False):
        return
    trace_scheduler(
        scheduler,
        "destination_receiver",
        stage,
        rid=req.rid,
        bootstrap_room=req.bootstrap_room,
        **fields,
    )
