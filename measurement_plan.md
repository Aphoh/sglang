# Decode Migration Measurement Plan

## Objective

Build a Pareto frontier with:

```text
x = P95 time to first non-thinking token (TTFNT)
y = SLO-compliant completed requests / second / allocated GPU
```

A migration point is useful when it improves throughput/GPU at equal TTFNT, or
reduces TTFNT at equal throughput/GPU. Correctness and visible-stage latency are
hard gates.

## Request Model

For a controlled static experiment, fix:

- input length;
- total output length `O`;
- fast-stage fraction `f`;
- migration sequence length `ISL + round(f * O)`;
- model, sampling, seeds, context length, page size, and stream interval.

The June 17, 2026 local check uses ISL 1, OSL 512, and `f=0.60`. The source
produces 307 output tokens. TTFNT is the arrival of output token 308.

For real traces, derive the boundary from the model's thinking terminator and
report the observed reasoning/visible length distribution.

## Load Generation

Use open-loop constant or trace-driven arrivals. Do not measure a start-empty
burst as steady state.

Each rate point must contain three continuous phases at the same arrival rate:

1. warmup load;
2. measured requests;
3. cooldown load.

Cooldown traffic is required because stopping arrivals after the last measured
request removes prefill interference and artificially improves its latency.
Report early-half versus late-half P50 TTFNT; large drift invalidates the point.

Use offered SLO-compliant request rate for the primary throughput axis. Also
report completion throughput including drain time, but do not use finite-run
drain time to define the steady-state Pareto.

## Metrics

Record per request:

- source-boundary arrival;
- first non-thinking token arrival;
- handoff gap between those two events;
- visible token rate and longest visible-token gap;
- completion length, finish reason, token IDs, and terminal migration state.

Record per worker:

- running and queued requests;
- generated tokens/second;
- KV allocation and reservation;
- migration prepare, transfer-complete, activate, and commit counts;
- transfer bytes and latency.

The visible stage must sustain at least 20 tokens/second. Every request must
complete with the expected token count and no lost, duplicated, or fallback
stream chunks.

## Rate Matching

For reasoning tokens `R`, visible tokens `V`, and sustainable tier capacities
`C_fast` and `C_slow`, start from:

```text
G_fast / G_slow ~= (R / C_fast) / (V / C_slow)
```

Then sweep arrival rate. Token-load equivalence is only a starting estimate;
per-request prefill and control overhead can shift the actual knee. Inspect
worker logs to confirm that queues are stationary and destination traffic is
balanced.

Compare topology choices, not only TP4 source plus extra slow GPUs. A lower-TP
fast tier can improve throughput/GPU if it still satisfies TTFNT.

## Correctness Gates

Reject a point for:

- incomplete requests or wrong output length;
- malformed reasoning boundaries;
- migration prepare, transfer, activation, or commit mismatch;
- a non-exact exported KV frontier;
- cancellation or reservation leaks;
- visible rate below 20 tokens/second;
- unstable queue growth, OOM, or worker restart;
- meaningful task-accuracy regression.

For overlap scheduling, log both the internal source length and the exported
migration frontier. Internal execution may advance beyond the trigger, but the
exported committed/logical frontier must remain exact.

## Repetition

Refine around candidate Pareto points, then confirm the best pair with at least
256 measured requests plus warmup and cooldown load. Repeat with multiple seeds
or trace offsets when sampling or prompt distributions are non-deterministic.

The runnable static client is
`benchmark/decode_migration/static_decode_pareto.py`. Current Qwen3-32B results
are in `measurement_results_qwen3_32b.md`.
