# Decode-to-Decode Migration Roadmap

## Scope

SGLang provides the engine-side transaction for moving a live decode request
between ordinary workers. Dynamo chooses the source, destination, and trigger;
SGLang owns exact request quiescence, KV transfer, destination admission, and
cleanup.

Every worker started with `--enable-decode-migration` can send and receive.
Fast and slow are Dynamo scheduling taints, not SGLang worker roles.

## Current Transaction

The one-shot prototype uses SGLang's decode-side NIXL path:

1. The destination reserves a migration record and opaque bootstrap room.
2. The source pauses at a scheduler boundary and captures the exact request
   frontier.
3. The destination arms a continuation receiver with explicit source rank and
   frontier data.
4. NIXL transfers the missing committed KV range.
5. After the destination produces valid output, Dynamo activates it and commits
   the source.

The source retains its request row, KV pages, radix ownership, sampling state,
and pending token until `commit`, `resume`, or `cancel`. Before commit, a failed
handoff can therefore resume the original request.

The worker control operations are:

- `migration_prepare`: reserve, then arm the destination continuation.
- `migration_sync`: describe the source or quiesce it for transfer.
- `migration_finalize`: activate/abort the destination or
  commit/resume/cancel the source.

The destination reports its cached prefix, so the source transfers only the
missing committed range. A cold destination receives prompt and generated KV;
a warmed destination can receive only the delta.

## Correctness Rules

1. The source is authoritative until destination output is valid and source
   commit succeeds.
2. KV ownership uses numeric token frontiers, not streamed chunk boundaries.
3. The destination starts from the sampled pending token after receiving all
   committed KV.
4. Source tokens committed but not emitted are forwarded exactly once.
5. Destination replay of already emitted tokens is trimmed by position.
6. A request carrying a finish reason is never migrated.
7. Destination receive admission must preserve the KV allocation established by
   transfer preparation; it must not perform a second radix match.
8. Cancellation and rollback release request rows, KV pages, transfer state,
   receiver state, and reservations.

## Prototype Limits

- Exact quiescence requires `--disable-overlap-schedule`.
- Quiescence pauses the source scheduler and allows one active source migration
  per worker. Dynamo serializes concurrent migrations per source rank.
- Transfer is one-shot.
- Destination reservation is advisory handler state, not a hard KV lease.
- Live coverage is DP=1; multi-DP routing still needs validation.
- Speculative decoding, beam search, multiple return sequences, guided decoding,
  multimodal continuation state, and sessions are unsupported.
- Source and destination must have compatible model, page size, KV layout and
  dtype, PP layout, and transfer protocol. Heterogeneous TP works only through a
  supported NIXL direct or staging layout.

## Upstream Work

### Per-request parking

Replace the scheduler-wide pause with an engine-owned parked state for one
request. Unrelated requests and independent migrations must continue while the
parked request retains all decode and KV ownership.

### Hard destination leases

Reserve request-pool and KV capacity before source quiescence. Leases need a
capacity grant, destination rank, compatibility fingerprint, TTL, and
idempotent reserve/grow/arm/activate/abort transitions.

### Incremental synchronization

Reuse one destination lease and migration ID. Copy monotonically increasing
stable ranges while source decode continues, then park the source and copy only
the final delta at the trigger. The existing activation, stream reconciliation,
and commit protocol should remain unchanged.

### Compatibility and DP

Publish cache-layout capabilities and reject incompatible pairs before
quiescence. Add multi-DP tests proving that control RPCs and NIXL handshakes
reach the selected rank.

### Observability and fault injection

Record phase latency, token ranges, bytes/pages, terminal outcomes, and cleanup.
Add failures for allocation, transfer timeout, lost control responses, worker
loss, lease expiry, and rank mismatch.

## Validation

Focused tests cover frontier construction, radix ownership, destination
admission, stream intervals 1 and 4, finish races, cancellation, rollback, and
cleanup. Live tests cover Qwen3-0.6B TP1-to-TP1 and Qwen3-8B TP4-to-TP1 through
the heterogeneous-TP staging path.

A 200-sample Qwen3-8B GSM8K run at temperature 1.0 and 32K context completed
200/200 migrations with no busy rejection, rollback, invalid response, or
length truncation. Accuracy was 96.0% without migration and 96.5% with
migration; the paired McNemar exact test gave `p=1.0`.

Performance evaluation is defined in `measurement_plan.md`.
