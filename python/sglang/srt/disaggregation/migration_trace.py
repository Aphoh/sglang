"""Low-overhead, opt-in structured timing for decode migration.

Events are retained in process memory and dumped as JSONL at orderly process
exit. This keeps the migration hot path free of stdout/stderr locks and file
I/O. Send SIGUSR1 to a worker process to flush its current buffer without
stopping it.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import signal
import socket
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)

_TRACE_ENV = "DYNAMO_DECODE_MIGRATION_TRACE"
_TRACE_DIR_ENV = "DYNAMO_DECODE_MIGRATION_TRACE_DIR"
_TRACE_MAX_EVENTS_ENV = "DYNAMO_DECODE_MIGRATION_TRACE_MAX_EVENTS"
_TRACE_DIR_DEFAULT = "/tmp/dynamo-decode-migration-trace"

_events: list[dict[str, Any]] = []
_events_lock = threading.Lock()
_dropped_events = 0


def enabled() -> bool:
    return os.environ.get(_TRACE_ENV, "").lower() in {"1", "true", "yes"}


def _max_events() -> int:
    try:
        return max(1, int(os.environ.get(_TRACE_MAX_EVENTS_ENV, "8192")))
    except ValueError:
        return 8192


def _append(event: dict[str, Any]) -> None:
    """Append without serializing or performing I/O on the scheduler hot path."""

    global _dropped_events
    with _events_lock:
        if len(_events) < _max_events():
            _events.append(event)
        else:
            _dropped_events += 1


def _trace_path() -> Path:
    trace_dir = Path(os.environ.get(_TRACE_DIR_ENV, _TRACE_DIR_DEFAULT))
    return trace_dir / f"decode-migration-{socket.gethostname()}-{os.getpid()}.jsonl"


def flush() -> None:
    """Persist the current buffer. Safe to call repeatedly and from SIGUSR1."""

    global _dropped_events
    if not enabled():
        return
    with _events_lock:
        if not _events and not _dropped_events:
            return
        events = list(_events)
        _events.clear()
        dropped_events = _dropped_events
        _dropped_events = 0
    if dropped_events:
        events.append(
            {
                "event": "decode_migration_trace",
                "role": "trace_buffer",
                "stage": "events_dropped",
                "wall_time_ns": time.time_ns(),
                "mono_time_ns": time.monotonic_ns(),
                "fields": {"count": dropped_events},
            }
        )
    try:
        path = _trace_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", buffering=1024 * 1024) as trace_file:
            trace_file.write("".join(json.dumps(event, separators=(",", ":"), sort_keys=True) + "\n" for event in events))
    except Exception:
        # Do not turn tracing persistence failures into inference failures.
        logger.exception("Failed to flush decode migration trace buffer")


def _install_flush_hook() -> None:
    atexit.register(flush)
    if not hasattr(signal, "SIGUSR1"):
        return
    try:
        previous = signal.getsignal(signal.SIGUSR1)
        if previous not in (signal.SIG_DFL, signal.SIG_IGN):
            return

        def _flush_on_usr1(_signum, _frame) -> None:
            flush()

        signal.signal(signal.SIGUSR1, _flush_on_usr1)
    except (ValueError, OSError):
        # Signal handlers can only be installed in a process main thread.
        pass


def trace_nixl_bootstrap(
    stage: str,
    *,
    bootstrap_room: int,
    **fields,
) -> None:
    """Trace an opaque-room NIXL bootstrap-server event.

    The connection layer does not own a request ID. Consumers join this event
    to scheduler events via the opaque bootstrap room, never by rank identity.
    """

    if not enabled():
        return
    _append(
        {
            "event": "decode_migration_trace",
            "role": "source_nixl_bootstrap",
            "stage": stage,
            "bootstrap_room": bootstrap_room,
            "wall_time_ns": time.time_ns(),
            "mono_time_ns": time.monotonic_ns(),
            "fields": fields,
        }
    )


def trace_scheduler(
    scheduler: "Scheduler",
    role: str,
    stage: str,
    *,
    rid: str,
    migration_id: Optional[str] = None,
    **fields,
) -> None:
    """Buffer one machine-readable scheduler event from each owning rank."""

    if not enabled():
        return
    parallel_state = scheduler.ps
    _append(
        {
            "event": "decode_migration_trace",
            "role": role,
            "stage": stage,
            "rid": rid,
            "migration_id": migration_id or "unknown",
            "wall_time_ns": time.time_ns(),
            "mono_time_ns": time.monotonic_ns(),
            "dp_rank": getattr(parallel_state, "dp_rank", 0),
            "tp_rank": getattr(parallel_state, "tp_rank", 0),
            "fields": fields,
        }
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


_install_flush_hook()
