# Decode Migration Measurement Plan

## Goal

Measure whether fast hidden reasoning followed by migration to a slower visible
decode tier improves serving efficiency without violating user-visible latency
or correctness.

The primary Pareto plot is:

```text
x = P95 time to first non-thinking token (TTFNT)
y = SLO-compliant completed requests / second / allocated GPU
```

A point is useful only if it is not dominated by another configuration on both
axes.

## Metrics

### TTFNT

Measure from frontend admission to delivery of the first token after the
reasoning boundary. For Qwen3, `</think>` is the boundary and is not counted as
visible output.

TTFNT includes queueing, prefill, fast decode, destination selection and
reservation, final KV transfer, activation, and the first slow decode step.
Record these components separately.

### Visible-stage SLO

The slow stage must sustain at least 20 visible tokens/second, equivalent to
mean TPOT no greater than 50 ms/token. Use this gate:

```text
P95 per-request visible generation rate >= 20 tokens/second
```

Also report P50/P95/P99 TPOT, the longest visible-token gap, and the fraction of
requests below the target. Exclude one-token visible outputs from rate
percentiles but retain them in TTFNT.

### Goodput per GPU

```text
goodput/GPU =
  successful requests satisfying correctness and latency gates
  / measurement duration
  / all allocated GPUs
```

Count prefill, fast, slow, and dedicated transfer GPUs. Also report admitted and
completed request rate, tokens/second/GPU by tier, failures, cancellations, SLO
violations, GPU utilization, and KV occupancy.

Correctness is a gate: malformed boundaries, lost or duplicated tokens, or task
accuracy regressions do not count as goodput.

## Rate Matching

Treat the deployment as a tandem queue:

```text
arrival -> prefill -> fast reasoning -> migration -> slow visible decode
```

Let `R` and `V` be expected reasoning and visible tokens per request, and
`C_fast` and `C_slow` the sustainable token rates per GPU. A starting allocation
is:

```text
G_fast / G_slow ~= (E[R] / C_fast) / (E[V] / C_slow)
```

Sustainable request rate is bounded by the minimum capacity of prefill, fast
decode, slow decode, and transfer. Use expected remaining token work and
predicted boundary arrivals as scaling signals, not request count alone.

Keep slow-tier headroom because reasoning completions are bursty and slow-tier
overload immediately harms visible TPOT. Track:

- predicted boundaries in the next 0.5, 1, 2, and 5 seconds;
- queued reasoning and visible token work;
- prepared destinations and reservation age;
- free and reserved KV blocks;
- queued transfer bytes and recent migration latency;
- reasoning- and visible-length prediction error.

## Policies to Compare

1. **Late binding:** choose and reserve the destination at `</think>`.
2. **Predicted reservation:** reserve shortly before the predicted boundary.
3. **Prompt warming:** place prompt KV early and transfer the reasoning delta at
   handoff.
4. **Incremental sync:** periodically copy stable reasoning KV, then transfer the
   final delta at the boundary.
5. **Immediate pairing:** reserve both workers at admission as a
   low-handoff-latency, high-resource-cost reference.

For prepared capacity, start with:

```text
prepared slow slots =
  predicted boundaries over the next W seconds + burst margin
```

Measure reservation hit rate, expiry, KV-token-seconds, idle reserved capacity,
and TTFNT benefit.

## Experiment Matrix

Use synthetic distributions and real traces. Vary:

- prompt, reasoning, and visible lengths;
- reasoning-length variance and heavy-tail frequency;
- arrival rate and burstiness;
- cancellation and pre-trigger completion rates;
- total GPUs and fast/slow split;
- fast and slow TP, batch limits, and KV fractions;
- stream interval and migration policy.

For an eight-GPU system, start with shared-prefill splits around `4/4`, `5/3`,
and `3/5`, accounting for prefill interference.

Required baselines:

1. fast-only;
2. slow-only;
3. static request routing;
4. immediate pairing;
5. one-shot late-bound migration;
6. predicted reservation;
7. prompt warming;
8. incremental synchronization.

Keep prompts, model, sampling, scoring, and arrival trace identical.

## Procedure

1. Characterize isolated prefill, fast decode, slow decode, and NIXL transfer.
2. Warm model and transfer paths.
3. Use an open-loop arrival generator.
4. Increase offered load until latency, correctness, or reliability fails.
5. Refine around the latency knee.
6. Repeat with multiple seeds or trace offsets.
7. Report bootstrap confidence intervals; reject points whose interval crosses
   an SLO boundary.

Run long enough to include multiple heavy-tail requests and reach stationary
queues. Plot the primary Pareto, offered-load curves, GPU split, reservation
tradeoff, transfer strategy, and reasoning-length sensitivity.

## Instrumentation

Each request needs stable request and migration IDs plus timestamps for:

- admission, prefill, and fast-decode start/end;
- boundary observation and destination selection;
- reservation, receiver arm, and source quiescence;
- each transfer range, bytes, and completion;
- destination activation and source release;
- first visible token and terminal outcome.

Capture worker queue depth, batch size, KV usage/reservations, token rate, GPU
utilization, HBM use, NIXL throughput/latency, and failures. A request counts as
migrated only when reservation, arm, transfer, activation, and source release
events agree.

## Acceptance Gates

Reject a configuration for:

- meaningful task-accuracy regression;
- malformed reasoning boundaries or token loss/duplication;
- migration/request failures above the configured limit;
- leaked reservations or failed cancellation cleanup;
- worker restart, OOM, or unstable queue growth.

Adopt a policy only if it improves goodput/GPU at equal P95 TTFNT, or lowers
P95 TTFNT at equal goodput/GPU, while meeting the 20 token/second visible-stage
SLO. Incremental sync is worthwhile only when reduced handoff latency exceeds
its transfer, KV-residency, and control-plane costs.
