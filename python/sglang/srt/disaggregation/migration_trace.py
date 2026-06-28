"""Low-overhead, opt-in structured timing for decode migration.

Events are appended to bounded in-memory buffers on the worker hot path. A
daemon writer snapshots and persists them as JSONL in batches, so neither
stdout/stderr locks nor synchronous file I/O are part of each trace event.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
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
_TRACE_FLUSH_EVENTS_ENV = "DYNAMO_DECODE_MIGRATION_TRACE_FLUSH_EVENTS"
_TRACE_FLUSH_INTERVAL_MS_ENV = "DYNAMO_DECODE_MIGRATION_TRACE_FLUSH_INTERVAL_MS"
_TRACE_DIR_DEFAULT = "/tmp/dynamo-decode-migration-trace"

_events: list[dict[str, Any]] = []
_events_lock = threading.Lock()
_dropped_events = 0
_writer_pid = 0
_writer_start_lock = threading.Lock()
_writer_wakeup = threading.Event()


def enabled() -> bool:
    return os.environ.get(_TRACE_ENV, "").lower() in {"1", "true", "yes"}


def _positive_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, str(default))))
    except ValueError:
        return default


def _max_events() -> int:
    return _positive_int(_TRACE_MAX_EVENTS_ENV, 8192)


def _flush_events() -> int:
    return _positive_int(_TRACE_FLUSH_EVENTS_ENV, 256)


def _flush_interval_s() -> float:
    return _positive_int(_TRACE_FLUSH_INTERVAL_MS_ENV, 250) / 1000


def _trace_path() -> Path:
    trace_dir = Path(os.environ.get(_TRACE_DIR_ENV, _TRACE_DIR_DEFAULT))
    return trace_dir / f"decode-migration-{socket.gethostname()}-{os.getpid()}.jsonl"


def _drain() -> list[dict[str, Any]]:
    global _dropped_events
    with _events_lock:
        if not _events and not _dropped_events:
            return []
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
    return events


def _write(events: list[dict[str, Any]]) -> None:
    if not events:
        return
    try:
        path = _trace_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = "".join(
            json.dumps(event, separators=(",", ":"), sort_keys=True) + "\n"
            for event in events
        )
        with path.open("a", encoding="utf-8", buffering=1024 * 1024) as trace_file:
            trace_file.write(payload)
    except Exception:
        # Tracing must never break inference. This is at most one error per batch.
        logger.exception("Failed to flush decode migration trace buffer")


def flush() -> None:
    """Synchronously persist the current process buffer at orderly exit."""

    if enabled():
        _write(_drain())


def _writer_loop(wakeup: threading.Event) -> None:
    while True:
        wakeup.wait(_flush_interval_s())
        wakeup.clear()
        _write(_drain())


def _ensure_writer() -> None:
    """Start one daemon writer per process, including forked scheduler children."""

    global _writer_pid, _writer_wakeup
    if not enabled() or _writer_pid == os.getpid():
        return
    with _writer_start_lock:
        if _writer_pid == os.getpid():
            return
        _writer_pid = os.getpid()
        _writer_wakeup = threading.Event()
        threading.Thread(
            target=_writer_loop,
            args=(_writer_wakeup,),
            name="decode-migration-trace-writer",
            daemon=True,
        ).start()


def _append(event: dict[str, Any]) -> None:
    """Append without serialization or synchronous I/O on the scheduler path."""

    global _dropped_events
    _ensure_writer()
    should_wake = False
    with _events_lock:
        if len(_events) < _max_events():
            _events.append(event)
            should_wake = len(_events) >= _flush_events()
        else:
            _dropped_events += 1
    if should_wake:
        _writer_wakeup.set()


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


atexit.register(flush)
