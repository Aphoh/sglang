# Decode-to-Decode Migration Roadmap

## Objective

Support transactional migration of a live decode request from one ordinary
SGLang worker to another. Dynamo owns routing and trigger policy. SGLang owns
request quiescence, exact KV frontiers, NIXL transfer, destination admission,
and rollback-safe cleanup.

Every worker started with `--enable-decode-migration` can be a source or a
destination. Fast and slow are deployment taints in Dynamo, not SGLang worker
roles. Workers keep their normal model registration and generation endpoint.

## Implemented Prototype

The prototype is one-shot and uses the normal SGLang decode-side NIXL machinery.
It implements:

- source request lookup and exact quiescence at a scheduler iteration boundary;
- retained source request, KV allocation, radix ownership, and sampling state
  until Dynamo commits the migration;
- a destination receiver on ordinary aggregated workers, without requiring the
  worker to run globally in P/D decode mode;
- opaque random bootstrap rooms that do not encode rank;
- explicit migration request state, including source rank and whether a request
  is the prepared destination continuation;
- equal-TP transfer and the existing heterogeneous-TP staging path;
- exact stream-frontier data for `--stream-interval > 1` reconciliation;
- source `commit`, `resume`, and `cancel` cleanup;
- destination receive admission that preserves the KV ownership established by
  decode preallocation and avoids a second radix-cache match.

The current source barrier is intentionally coarse: it pauses the scheduler on
that worker and permits one source migration at a time. This is sufficient to
prove transaction correctness, but it is not the production concurrency model.

## Endpoint Model

Dynamo exposes these worker RPCs on every migration-enabled worker:

- `generate`
- `migration_prepare`
- `migration_sync`
- `migration_finalize`

SGLang itself implements the engine operations reached by those handlers.

### `migration_sync` on the source

`phase=describe` returns the source NIXL bootstrap address. It does not alter the
request.

`phase=quiesce` is the source-side transactional boundary:

1. Pause between scheduler iterations.
2. Find the exact live request.
3. Reject migration if the request already finished or is finishing.
4. Compute prompt, committed-KV, logical-token, pending-token, and emitted-token
   frontiers.
5. Create the NIXL sender for the destination's opaque room.
6. Retain the request and all KV state until `commit`, `resume`, or `cancel`.

For the current non-speculative path, the sampled pending token is not yet in KV:

```text
logical_len = committed_len + 1
```

The response includes the committed input IDs, pending input ID, and any
committed output tokens that Dynamo has not yet forwarded.

### `migration_prepare` on the destination

The worker handler currently has two idempotent phases:

1. `reserved`: allocate a migration record and opaque room.
2. `ready`: start the destination `async_generate` receiver with the exact source
   frontier and retain that prepared stream for the subsequent `generate` RPC.

Important limitation: the first phase is a logical reservation. It records
`reserve_tokens`, but does not yet acquire a hard KV-capacity lease. Actual KV
indices are allocated when the second phase starts the normal decode receiver.
A production implementation needs an engine-owned reservation with TTL,
capacity accounting, and deterministic release.

A destination continuation is marked explicitly. If its prepared stream is
missing or consumed twice, generation fails closed instead of silently running a
fresh prefill.

### `migration_finalize`

Source actions:

- `commit`: allowed only after NIXL reports transfer success; abort the retained
  source through the normal scheduler finish path.
- `resume`: discard transfer state and continue the untouched source request.
- `cancel`: discard transfer state and terminate the retained source request.

Destination actions:

- `activate`: mark the prepared continuation authoritative.
- `abort`: abort the receive/generation request and release handler state.

Control calls are keyed by `migration_id`. The current implementation is robust
to repeated prepare and cleanup calls used by normal retry paths, but complete
persistent idempotency across process failure is out of scope.

## Data-Plane Sequence

```text
source generate
  -> source describe
  -> destination prepare(reserve logical session and room)
  -> source sync(quiesce and create sender)
  -> destination prepare(arm exact continuation and receiver)
  -> NIXL sends [destination prefix, committed_len)
  -> destination produces its first valid output
  -> destination activate
  -> source commit
  -> destination continues generation
```

The destination receiver reports its cached prefix to the source. The source
therefore sends only the missing committed range, including prompt KV when the
destination has no warm prefix.

## Correctness Invariants

1. The source remains authoritative until destination output is valid and source
   commit succeeds.
2. A pre-commit failure aborts the destination and resumes or cancels the
   retained source according to client ownership.
3. KV ownership is based on numeric token frontiers, never stream chunk
   boundaries.
4. The destination receives committed KV and starts from the sampled pending
   token.
5. Tokens already emitted by the source are trimmed from destination replay.
6. Source tokens committed but not emitted are forwarded once before handoff.
7. A request with a finish reason is never migrated.
8. Destination admission must not rematch radix cache after receive
   preallocation; doing so can leak or double-protect transferred KV pages.
9. Cancellation releases source transfer metadata, destination receive state,
   request rows, KV pages, and handler reservations.

## Current Constraints

- `--disable-overlap-schedule` is required for an exact source frontier.
- The source worker is scheduler-paused during transfer.
- One source migration may be active per worker.
- Transfer is one-shot.
- The live test topology uses DP=1. Rank is no longer encoded in the room, but
  multi-DP endpoint routing and rank validation still need dedicated work.
- Speculative decoding, beam search, multiple return sequences, guided decoding,
  multimodal continuation state, and session migration are not supported.
- Model revision, KV layout/dtype, page size, PP layout, and transfer protocol
  must match. Heterogeneous TP is supported only where the existing NIXL direct
  or staging layout supports it.
- Destination capacity reservation is not yet hard or durable.

## Next Implementation Steps

### 1. Per-request source parking

Replace `_engine_paused` with a scheduler-owned migration state on one request.
The parked request must retain its request-pool row, KV pages, radix locks,
sampling state, output IDs, and pending token while unrelated requests continue.

Required races:

- finish before and during park;
- normal abort and client cancellation;
- preemption while migration is requested;
- two independent migrations on the same worker;
- rollback after destination arm or transfer failure.

### 2. Engine-owned destination reservation

Move reservation state out of the Dynamo Python handler and into SGLang. Reserve
request-pool and KV capacity before quiescing the source. Add:

- requested, granted, and consumed token capacity;
- TTL and expiry cleanup;
- idempotent reserve, grow, arm, activate, and abort transitions;
- admission backpressure rather than late allocation failure;
- metrics for reserved KV-token-seconds and stranded capacity.

### 3. Incremental KV synchronization

Keep one destination session while the source continues decoding. Extend sync to
accept a monotonically increasing range:

```text
migration_sync(quiesce=false, from_token, through_token)
```

For each increment:

1. Snapshot a stable source frontier without parking the request.
2. Transfer only `[transferred_len, stable_len)` into destination-owned KV slots.
3. Advance the destination session watermark after NIXL completion.
4. At the configured trigger, park the source and transfer the final delta.
5. Reuse the existing activation, stream reconciliation, and source commit path.

Non-final ranges should be page-aligned where required by the transfer backend.
The final range may be exact. Each transfer generation needs an unambiguous room
or generation identifier; rank remains a separate field.

### 4. Compatibility and DP validation

Expose a structured transfer capability derived from the actual cache layout and
validate source/destination pairs before quiescence. Add multi-DP tests that prove
control calls and NIXL handshakes reach the selected rank.

### 5. Observability

Emit phase timings and terminal outcomes for reserve, quiesce, receiver arm,
bytes/pages transferred, first destination output, commit, rollback, cancel, and
cleanup. These are required for the Pareto experiments in `measurement_plan.md`.

## Verification Matrix

Implemented tests cover:

- frontier construction and radix lock ownership;
- destination receive admission and cleanup;
- source finish before trigger and during quiescence;
- cancellation before and after handoff;
- stream intervals 1 and 4;
- finish immediately after handoff;
- concurrent trigger attempts under the coarse source barrier;
- deterministic source-only versus migrated output;
- Qwen3-8B TP4 source to TP1 destination using NIXL staging;
- paired Qwen3 thinking-boundary GSM8K checks.

The June 12, 2026 Qwen3-8B TP4-to-TP1 run collected 20 committed migrations
after skipping one completion that never emitted `</think>`. The fast-only
baseline scored 19/20 and the migrated path scored 18/20, with 90% extracted
answer agreement. No scheduler exception or KV-pool leak signature was observed.
This passes the configured one-regression smoke gate, but it is not evidence of
accuracy neutrality; a larger run and a same-TP control are required to separate
normal TP-layout numerical divergence from migration-specific defects.

Before upstreaming, add fault injection for receiver allocation failure, NIXL
failure/timeout, lost control responses, process loss, reservation expiry, and
multi-DP rank mismatch.
