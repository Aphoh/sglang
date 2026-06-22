# Decode-to-Decode Migration Roadmap

## Ownership

Dynamo selects workers and decides when to migrate. SGLang owns request-local
parking, the exact KV frontier, NIXL transfer, destination admission, and
cleanup.

Every worker started with `--enable-decode-migration` can send and receive.
`decode/fast`, `decode/slow`, and benchmark-specific roles are Dynamo taints,
not SGLang engine modes.

## Current One-Shot Flow

1. Dynamo selects and reserves a destination.
2. Dynamo arms the source with a sequence-length trigger and the destination's
   bootstrap address, opaque room, and explicit rank metadata.
3. The normal overlap result path detects the trigger and removes only that
   request from scheduling.
4. SGLang exports the requested committed/logical frontier. If overlap execution
   advanced internally beyond the trigger, unstreamed extra results are excluded
   from the exported KV and token state.
5. The destination arms its continuation receiver and reports any cached prefix.
6. NIXL transfers only the missing committed KV range.
7. Dynamo attaches the destination stream, activates the destination, and commits
   the source after valid destination output.

The source retains the parked request and KV until commit or cancellation, but
there is no resume-after-failed-handoff protocol. Failure aborts the transaction
and performs cleanup.

## Required Invariants

- The engine is never paused globally.
- Overlap scheduling remains enabled.
- Unrelated requests and independent migrations continue running.
- The exported frontier is numeric and exact; stream chunk boundaries do not
  define KV ownership.
- The destination receives committed KV plus exactly one pending sampled token.
- Tokens already emitted by the source are never replayed.
- Requests that finish before parking are not migrated.
- Cancellation releases request rows, KV pages, metadata buffers, receivers,
  senders, and Dynamo reservation state.
- Source and destination model, page size, dtype, KV layout, PP layout, and
  transfer protocol must be compatible.

## Validation Status

Focused tests cover frontier construction, overlap overshoot, destination
admission, stream intervals, finish races, cancellation, rollback, and cleanup.
The source/destination test set passes 17/17 tests; Dynamo migration tests pass
24/24.

Live validation includes TP1 -> TP1, TP4 -> TP1, MLA transfer, and TP2 -> TP1.
The current static Qwen3-32B confirmation completed 256/256 measured migrations
at an exact 307/308 frontier with overlap enabled.

The confirmed local Pareto point compares pure TP4 with TP2 -> TP1 migration:

```text
Pure TP4:       1.300 req/s/GPU, P95 TTFNT 7.646 s
TP2 -> TP1:     1.692 req/s/GPU, P95 TTFNT 7.424 s
```

See `measurement_plan.md` and `measurement_results_qwen3_32b.md`.

## Next Engineering Steps

### Hard destination leases

Replace advisory reservation state with request-pool and KV-capacity leases.
Leases need capacity, rank, compatibility fingerprint, TTL, and idempotent
reserve/grow/arm/activate/abort transitions.

### Incremental KV synchronization

Reuse the same destination lease, migration ID, bootstrap room, and explicit rank
mapping. While source decode continues, copy monotonically increasing stable KV
ranges. At the trigger, park the request and transfer only the final delta. The
current activation, stream reconciliation, and commit protocol should remain
unchanged.

### Compatibility and DP

Publish cache-layout capabilities and reject incompatible pairs before arming the
source. Add multi-DP tests proving that control RPCs and NIXL handshakes reach the
selected rank.

### Fault injection and observability

Add phase timings, byte/page counts, lease age, and terminal-state metrics. Test
allocation failure, transfer timeout, lost control responses, worker loss,
lease expiry, cancellation at every phase, and rank mismatch.

### Production policy

Use measured reasoning/visible token distributions to choose fast/slow GPU
allocations. Pre-reserve only when predicted boundary timing and KV headroom make
it worthwhile. Incremental transfer is justified only when its TTFNT reduction
exceeds added transfer traffic and reserved-KV cost.
