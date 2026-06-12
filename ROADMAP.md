# Decode-to-Decode Migration Roadmap

## Goal

Implement live SGLang decode-to-decode migration behind Dynamo without giving
workers special public/private identities. Every migration-enabled decode
worker keeps its normal model deployment card and exposes the same endpoints:

- `generate`
- `migration_prepare`
- `migration_sync`
- `migration_finalize`

Worker metadata describes scheduling policy (`fast` or `slow`) and transfer
compatibility. It does not remove workers from normal discovery and does not
claim that only some workers can send or receive KV. Every compatible worker
can be either side of a migration.

## Core Decisions

### Destination owns preparation

`migration_prepare` is always called on the selected destination. It creates a
destination migration session, reserves KV capacity, chooses an opaque NIXL
bootstrap room, and records the selected source's bootstrap endpoint.

Preparation is intentionally separate from copying KV. Reasoning length may be
unknown when a destination is selected, so the reservation has an estimated or
bounded size and may grow before the final handoff.

### Source owns synchronization

`migration_sync` is always called on the exact source instance that owns the
live request. It snapshots a stable KV frontier and sends a requested range to
the destination transfer ticket. With `quiesce=true`, it parks the source at a
decode iteration boundary and returns the final request and stream frontiers.

### Exact receiver arming is a separate state

The destination cannot infer the exact committed KV frontier from client-visible
stream chunks, especially when `--stream-interval > 1`. The one-shot protocol
therefore distinguishes capacity reservation from receiver arming:

1. Destination `migration_prepare` reserves capacity and returns a session and
   opaque room.
2. Source `migration_sync(quiesce=true, ticket)` returns the exact committed,
   logical, and emitted frontiers, creates a pending sender for the ticket, and
   retains ownership. The sender waits for receiver metadata before copying.
3. Destination `migration_prepare` is retried idempotently with that exact
   frontier. It allocates destination indices, starts the NIXL receiver, and
   returns a ready transfer ticket using the same session and room.
4. Destination receiver arming supplies its allocated indices through the NIXL
   bootstrap service. The pending source sync then transfers the stable range.

The implementation may later combine steps 2 and 3 with a control-plane callback,
but the state machine and ownership boundaries must remain explicit.

### Rank is explicit

`bootstrap_room` is an opaque random 63-bit rendezvous identifier. It must not
encode a DP rank through modulo arithmetic. Transfer tickets carry source and
destination DP ranks explicitly. SGLang's NIXL bootstrap path must use those
fields directly when selecting the participating rank.

### Migration and room IDs are different

- `migration_id` is a UUID-like control-plane transaction identity used for
  idempotency, retries, logging, cleanup, commit, and abort.
- `bootstrap_room` is a data-plane rendezvous identity. A migration may use
  multiple rooms as incremental synchronization is added.

## Compatibility Metadata

All workers publish their normal MDC. Migration metadata should minimally carry:

```json
{
  "decode_migration": {
    "protocol_version": 1,
    "decode_class": "fast",
    "transport": "nixl",
    "compatibility_id": "model-revision-layout-page-size-kv-dtype-pp-protocol"
  }
}
```

`decode_class` is a routing hint, not an engine capability. A worker can be fast
in one deployment and slow in another without changing SGLang.

The compatibility ID must account for at least model revision, KV dtype/layout,
page size, PP topology, and transfer protocol version. TP size is separate
topology metadata: equal TP is the simplest case, but NIXL can transform certain
heterogeneous TP layouts. DP rank and live capacity are dynamic routing inputs,
not compatibility identity.

The router must validate a source/destination TP pair against transfer
capabilities. Non-MLA GQA/MHA uses head slicing or the GPU staging path. MLA KV is
replicated across TP ranks and can use the direct transfer path.

## One-Shot Protocol

```text
GENERATING(source)
  -> destination PREPARED (capacity lease, room allocated)
  -> source QUIESCED (exact frontiers captured, rollback retained)
  -> destination ARMED (KV indices registered with source bootstrap service)
  -> source TRANSFERRING
  -> destination RECEIVED
  -> destination ACTIVE
  -> source RELEASED
```

Failure before destination activation aborts the destination session and resumes
the retained source. Failure after destination activation is handled as normal
destination request failure; source release is only sent after activation has
been acknowledged.

## Endpoint Contracts

### `migration_prepare` on destination

First call creates the reservation:

```json
{
  "migration_id": "...",
  "request_id": "...",
  "source": {
    "instance_id": 17,
    "bootstrap_host": "10.0.0.17",
    "bootstrap_port": 8998,
    "dp_rank": 2
  },
  "reserve_tokens": 4096,
  "compatibility_id": "..."
}
```

It returns `status=reserved`, the destination instance/rank, and an opaque room.

An idempotent call with exact source state arms the receiver:

```json
{
  "migration_id": "...",
  "request_id": "...",
  "source_state": {
    "committed_input_ids": [],
    "pending_input_ids": [],
    "committed_len": 0,
    "logical_len": 0
  }
}
```

It returns `status=ready` only after destination KV indices and NIXL agent
metadata have been registered with the source bootstrap service.

### `migration_sync` on source

The quiescent sync captures exact state and installs a sender that waits for the
destination receiver:

```json
{
  "migration_id": "...",
  "request_id": "...",
  "output_tokens_seen": 12,
  "bootstrap_room": 123,
  "quiesce": true
}
```

A transfer call supplies the destination ticket and range:

```json
{
  "migration_id": "...",
  "request_id": "...",
  "transfer_ticket": {
    "bootstrap_room": 123,
    "destination_dp_rank": 0
  },
  "from_token": 0,
  "through_token": 1600,
  "quiesce": true
}
```

It returns monotonically increasing source-generated, KV-stable, transferred,
and client-emitted watermarks.

### `migration_finalize` on either side

Destination actions:

- `activate`: verify receive completion, install exact request state, and make
  the parked request runnable.
- `abort`: release reservation, receiver, request row, and KV allocation.

Source actions:

- `release`: abort the retained source request through normal cleanup after the
  destination is active.
- `resume`: discard transfer state and make the parked source runnable again.
- `cancel`: release source state because the client disconnected.

All control operations are idempotent by `(migration_id, action)`.

## Stream Correctness

The protocol tracks independent numeric frontiers:

- `logical_len`: prompt plus all sampled output tokens.
- `committed_len`: tokens represented in stable source KV.
- `transferred_len`: stable KV installed at destination.
- `output_tokens_seen`: output token positions already forwarded to the client.

For the initial non-speculative implementation:

```text
logical_len == committed_len + 1
```

The sampled-but-uncommitted token becomes the destination's first decode input.
Committed output tokens hidden by `--stream-interval` are emitted exactly once
by the coordinator before destination output. Stream chunk boundaries never
define KV ownership.

## Trigger Policy Boundary

SGLang does not interpret reasoning syntax or decide when a request should move.
Dynamo may trigger on a generated-token count, Qwen3's `</think>` token, an SLA
signal, or a router policy. Once triggered, `migration_sync` captures the same
exact committed/logical frontier regardless of why it was called.

The current coordinator forwards the matching boundary token before quiescing.
For `--stream-interval > 1`, any later tokens coalesced in the same source chunk
are handled by the same emitted and duplicate-trimming watermarks. A request
that has already reached a finish reason is not migrated.

Incremental KV movement should extend `migration_sync` with successive stable
ranges on the existing destination session. The final policy boundary performs
a quiescent delta and activation; it must not require a separate reasoning-aware
engine path.

## Implementation Phases

### Phase 1: reshape the working prototype

- Retain the known-good one-shot NIXL transfer and exact frontier helper.
- Rename source-side prepare to `migration_sync`; keep snapshot and asynchronous
  send as explicit states within the same idempotent sync operation.
- Add destination reservation/session records and receiver-ready status.
- Remove DP-rank encoding from room generation and carry ranks explicitly.
- Keep source state until destination activation succeeds.
- Keep the current scheduler-wide pause only as a prototype barrier.

### Phase 2: normal Dynamo discovery

- Remove `internal_decode_migration_worker` and MDC suppression.
- Publish role and compatibility metadata through the normal runtime config.
- Route migration-enabled requests through a coordinator/operator using request
  metadata and constrained worker selection.
- Preserve ordinary generation for requests that do not opt into migration.

### Phase 3: per-request parking

- Replace the scheduler-wide pause with a typed parked-request state.
- Keep request-pool row, KV pages, radix locks, sampling state, and output state.
- Allow unrelated requests and independent migrations to proceed.
- Integrate finish and cancellation races with parked-request cleanup.

### Phase 4: incremental synchronization

- Reuse the destination session and capacity lease.
- Arm successive transfer ranges with new opaque rooms or resettable receiver
  generations.
- Copy only `[transferred_len, stable_len)` while source decode continues.
- Perform a final quiescent delta, activate destination, then release source.

## Required Tests

Unit tests:

- opaque room generation independent of rank;
- explicit rank serialization and routing;
- prepare/sync/finalize idempotency;
- reservation expiry and cleanup;
- exact frontier calculations for multiple stream intervals;
- request finish before and during each migration state;
- cancellation in reserved, quiesced, transferring, received, and active states;
- retry after lost control responses;
- destination failure followed by source resume.

Live tests with two Qwen3-0.6B workers and Dynamo:

- deterministic source-only output equals migrated output;
- token-count and semantic-token migration triggers;
- `--stream-interval=1` and a value greater than one;
- request finishing just before and just after the handoff threshold;
- injected destination preparation and activation failures;
- client disconnect during handoff followed by successful worker reuse;
- logs prove source instance targeting, destination reservation, NIXL transfer,
  destination activation, and source release.

## Verified Prototype Status

The one-shot engine prototype meets the original completion criteria. It has
passed the stream, finish-race, rollback, cancellation, and post-cancellation
recovery scenarios, plus Qwen3-8B TP4-to-TP1 migration at the semantic
`</think>` boundary. In a 20-example paired GSM8K smoke run, all requests
completed the NIXL handoff, baseline and migrated accuracy were both 95%, hidden
reasoning matched 20/20, and extracted answers matched 20/20.

The remaining engine work for a production PR is per-request parking instead of
a scheduler-wide pause, lease expiry and cleanup under concurrent migrations,
idempotency coverage for lost control responses, incremental stable-range sync,
and transfer/SLA measurement at realistic reasoning lengths.
