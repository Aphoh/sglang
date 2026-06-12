# Decode Migration Measurement Plan

## Objective

Measure the Pareto frontier of a serving design that performs hidden reasoning on
a fast decode tier, migrates the live request and KV state at the reasoning
boundary, and produces user-visible output on a slower decode tier.

The experiment should answer:

1. How much SLO-compliant throughput per GPU does migration provide?
2. What is the resulting time to the first non-thinking token?
3. How should GPUs be divided between fast and slow decode tiers?
4. When should the slow destination be selected, reserved, and warmed?
5. When does one-shot transfer stop being sufficient and incremental transfer
   become worthwhile?

## Primary Pareto Frontier

Plot each deployment configuration as:

```text
x = P95 time to first non-thinking token

y = maximum SLO-compliant completed requests / second / allocated GPU
```

Only include requests that complete successfully and satisfy the visible-stage
latency requirement. This makes the efficiency axis **SLO-constrained goodput
per GPU**, rather than raw admitted request throughput.

A point dominates another point if it has both:

- lower or equal P95 time to first non-thinking token; and
- higher or equal SLO-compliant goodput per GPU;

with at least one strict improvement.

Report P50, P90, P95, and P99 latency even though P95 defines the primary plot.

## Latency Definition

### Time to first non-thinking token

Start the clock when the request is accepted by the serving frontend. Stop it
when the first user-visible token after the reasoning boundary is available to
the client.

For Qwen3, `</think>` is the reasoning boundary and is not itself counted as a
non-thinking token.

The metric includes:

```text
frontend and router queueing
+ prefill
+ fast-tier reasoning decode
+ destination selection and reservation
+ final KV synchronization
+ destination activation
+ first slow-tier decode step
+ frontend stream delivery
```

Record the following decomposition for every request:

- admission-to-prefill-start;
- prefill duration;
- prefill-to-fast-decode-start;
- fast decode duration;
- generated reasoning-token count;
- boundary-to-destination-reserved;
- destination reserve-to-receiver-ready;
- final KV transfer duration;
- transfer-complete-to-destination-active;
- destination-active-to-first-visible-token;
- total time to first non-thinking token.

### Visible-stage SLO

The slow stage must sustain at least 20 user-visible tokens per second.
Equivalently, its mean visible TPOT must not exceed 50 ms/token.

Use a tail constraint as the experiment gate rather than only an average:

```text
P95 per-request visible generation rate >= 20 tokens/second
```

Also report:

- P50/P95/P99 visible TPOT;
- fraction of requests below 20 tokens/second;
- longest visible inter-token gap;
- visible-stage stall time caused by migration or queueing.

Requests with only one visible token cannot produce a stable rate estimate. Keep
them in the time-to-first-non-thinking-token metric, but exclude them from the
per-request visible-rate distribution and report their count separately.

## Throughput Definition

The primary efficiency metric is:

```text
SLO-compliant goodput per GPU =
  successfully completed requests satisfying all SLOs
  / measurement duration
  / total allocated GPUs
```

Count every GPU allocated to the deployment, including:

- prefill GPUs;
- fast decode GPUs;
- slow decode GPUs;
- dedicated transfer or routing GPUs, if any.

Report supporting metrics:

- admitted requests/second;
- completed requests/second;
- reasoning tokens/second/GPU on the fast tier;
- visible tokens/second/GPU on the slow tier;
- total tokens/second/GPU;
- useful model tokens versus replayed or duplicated tokens;
- rejected, cancelled, failed, and SLO-violating requests;
- GPU utilization and memory occupancy by tier.

Correctness is a gate. A request is not goodput if migration changes its scored
answer or produces malformed reasoning/content boundaries.

## Queueing and Rate-Matching Model

Treat the deployment as a tandem queue:

```text
arrival -> prefill -> fast reasoning decode -> migration -> slow visible decode
```

Define:

```text
R = expected reasoning tokens per request
V = expected visible tokens per request
C_fast = sustainable reasoning tokens/second/GPU
C_slow = sustainable visible tokens/second/GPU while meeting the TPOT SLO
G_fast = GPUs assigned to fast decode
G_slow = GPUs assigned to slow decode
```

The approximate tier capacities are:

```text
lambda_fast = G_fast * C_fast / E[R]
lambda_slow = G_slow * C_slow / E[V]
```

The deployment's sustainable request rate is bounded by:

```text
lambda_system <= min(lambda_prefill,
                     lambda_fast,
                     lambda_slow,
                     lambda_transfer)
```

A first-order allocation rule is:

```text
G_fast / G_slow ~= (E[R] / C_fast) / (E[V] / C_slow)
```

This is only a starting point. Reasoning completions can be bursty, and
reasoning length is usually heavy-tailed. Average-rate matching can therefore
leave the slow tier overloaded even when mean capacity appears balanced.

Initially target approximately:

- 75-85% fast-tier utilization;
- 60-75% slow-tier utilization;
- transfer utilization comfortably below saturation.

The slow tier should carry more headroom because overload immediately degrades
user-visible TPOT.

## Work-Based Scaling Signals

Do not scale tiers from request counts alone. Track expected remaining token
work.

Fast-tier pressure:

```text
queued and active predicted remaining reasoning tokens
/ aggregate fast-tier token capacity
```

Slow-tier pressure:

```text
queued and active predicted visible tokens
/ aggregate slow-tier token capacity
```

Additional leading indicators:

- predicted fast-tier completions in the next 0.5, 1, 2, and 5 seconds;
- count of prepared slow destinations;
- destination reservation age;
- free and reserved KV blocks;
- warmed prompt-KV bytes;
- queued migration bytes;
- recent P95 migration duration;
- recent visible-stage TPOT;
- reasoning-length prediction error;
- visible-length prediction error.

Scale slow capacity before fast requests reach the reasoning boundary. Scale it
down only after active visible generation, reservations, and warmed KV state
have drained or been safely evicted.

## Destination Selection and Reservation Policies

Compare the following policies.

### Late binding

Select and reserve the slow destination only when `</think>` is observed.

Expected properties:

- highest KV and scheduling efficiency;
- simplest capacity accounting;
- migration and cold-start latency directly affect time to first non-thinking
  token.

### Fixed lead-time reservation

Predict reasoning completion and reserve a destination a fixed time before it.
Sweep lead times such as:

```text
0.25 s, 0.5 s, 1 s, 2 s, 5 s
```

Measure prediction misses, stranded reservations, and latency benefit.

### Prompt-KV warming

Select a slow worker early and install only prompt KV while fast reasoning
continues. Retain the prompt KV until handoff, then transfer the reasoning delta.

Measure:

- prompt-KV residency cost;
- reduction in final transfer bytes and latency;
- lost cache capacity due to requests with long reasoning phases;
- cost of changing the selected slow destination.

### Incremental reasoning-KV synchronization

Reuse one destination session and periodically copy newly stable reasoning KV.
At `</think>`, quiesce the source, transfer the final delta, and activate the
destination.

Sweep synchronization policies by:

- token interval;
- byte interval;
- time interval;
- predicted time remaining to the reasoning boundary;
- transfer-queue pressure.

Account for extra transfer traffic, synchronization overhead, and destination KV
residency. Compare against one-shot transfer at equal offered load.

### Immediate pairing

Select and reserve both workers at admission. Use this as a low-handoff-latency,
high-resource-cost reference point, not as the expected production policy.

## Prepared-Capacity Control

A useful policy should maintain enough prepared slow capacity for near-future
fast completions without pairing every request at admission.

A starting controller is:

```text
prepared slow slots =
  predicted fast completions over the next W seconds
  + burst safety margin
```

Sweep `W` and the safety margin. Report:

- reservation hit rate;
- requests that reach `</think>` without prepared capacity;
- reservation occupancy time;
- reservation expiry rate;
- stranded KV-token-seconds;
- slow GPU idle time while holding reservations;
- TTFNT reduction relative to late binding.

## Experimental Matrix

### Workload dimensions

Use both controlled synthetic distributions and real traces.

Sweep:

- prompt length;
- reasoning length;
- visible output length;
- reasoning-to-visible token ratio;
- coefficient of variation of reasoning length;
- heavy-tail frequency;
- concurrency and arrival burstiness;
- cancellation probability;
- requests that finish before a migration boundary;
- agentic workloads with multiple tool or search phases.

At minimum include:

1. Short reasoning, short visible output.
2. Long reasoning, short visible output.
3. Long reasoning, long visible output.
4. Fixed mean reasoning length with increasing variance.
5. A heavy-tailed reasoning distribution.
6. Bursty arrivals with the same average rate as a Poisson workload.

### Deployment dimensions

Sweep:

- total GPU count;
- prefill/fast/slow GPU split;
- fast-tier TP size;
- slow-tier TP size;
- fast and slow batch limits;
- KV-cache memory fractions;
- stream interval;
- migration policy;
- reservation lead time;
- incremental synchronization interval;
- admission-control threshold.

For the current eight-GPU system, useful initial allocations include:

```text
prefill/fast/slow
1 / 4 / 3
1 / 5 / 2
1 / 3 / 4
0 or shared / 4 / 4
```

If prefill is shared with fast decode, account for its GPU time and interference
rather than treating it as free.

### Offered-load sweep

For every deployment and policy:

1. Warm all model and transfer paths.
2. Start below saturation.
3. Increase offered load in small steps.
4. Hold each step long enough to reach steady state.
5. Stop increasing when any SLO or reliability gate fails.
6. Refine around the latency knee with smaller load steps.

The Pareto point for that configuration is the highest offered load whose
confidence interval still satisfies all gates.

Use an open-loop arrival generator for capacity measurements. A closed-loop
client hides queue growth by reducing offered load when latency increases.

## Required Comparisons

Measure at least these baselines:

1. Fast-only serving with no migration.
2. Slow-only serving with no migration.
3. Static request-level routing between fast and slow workers.
4. Immediate fast/slow pairing.
5. One-shot late-bound migration at `</think>`.
6. Predicted reservation plus one-shot final transfer.
7. Prompt warming plus one-shot reasoning delta.
8. Incremental reasoning-KV synchronization.

Keep model, sampling configuration, prompt set, correctness scoring, and arrival
trace identical across comparisons.

## Core Plots

Produce these plots for each workload family.

### Primary Pareto

```text
P95 TTFNT vs SLO-compliant requests/second/GPU
```

Label points by GPU split and migration policy.

### Offered-load curve

Plot offered load against:

- achieved goodput;
- P50/P95/P99 TTFNT;
- P95 visible TPOT;
- fast and slow queue depth;
- migration queue depth.

This identifies the first bottleneck and latency knee.

### GPU allocation curve

At a fixed total GPU budget, plot every feasible fast/slow split. Show both
throughput and latency to expose mismatched capacity.

### Reservation tradeoff

Plot reservation lead time or prepared-pool size against:

- TTFNT;
- goodput/GPU;
- reservation hit rate;
- stranded KV-token-seconds;
- slow-tier idle capacity.

### Transfer strategy

Plot one-shot versus incremental transfer against:

- final handoff duration;
- total bytes transferred;
- duplicate bytes;
- destination KV residency time;
- goodput/GPU;
- TTFNT.

### Length and skew sensitivity

Plot results by reasoning-length bucket and by workload coefficient of variation.
Report whether the tail is caused by generation, reservation misses, transfer,
or slow-tier queueing.

## Instrumentation Requirements

Every request should carry a stable request ID and migration ID. Emit structured
timestamps for:

- frontend admission;
- route decision;
- prefill start/end;
- fast decode start;
- reasoning boundary observed;
- destination selected;
- reservation created;
- receiver armed;
- source quiesced;
- each incremental sync start/end and token range;
- final transfer start/end and bytes;
- destination activated;
- source released;
- first non-thinking token emitted;
- request finished or cancelled.

Also capture per-worker time series:

- active and queued requests;
- active reservations;
- free/used/reserved KV blocks;
- prompt and reasoning KV residency;
- batch size;
- decode tokens/second;
- GPU SM utilization;
- HBM usage and bandwidth;
- NIXL bytes/second and operation latency;
- transfer failures and retries.

The benchmark must prove migration occurred rather than inferring it from output.
Require matching reservation, arm, transfer-complete, activation, and source
release events for every request counted as migrated goodput.

## Correctness and Reliability Gates

A configuration is invalid if it exceeds any configured limit for:

- answer-quality regression;
- missing or malformed reasoning/content boundaries;
- token loss or duplication;
- migration failure;
- request failure;
- leaked KV reservations;
- failed cancellation cleanup;
- worker restart or out-of-memory events.

For deterministic paired tests, separately report:

- exact hidden-reasoning match;
- exact visible-output match;
- extracted-answer match;
- scored task accuracy.

Numerical differences between heterogeneous TP layouts may change wording while
preserving the answer. Treat answer accuracy as the correctness gate, but retain
exact-match metrics as diagnostics.

## Statistical Method

For each steady-state point:

- discard warmup requests;
- run for a fixed minimum duration and minimum completed-request count;
- use the same arrival trace across compared configurations;
- repeat with multiple random seeds or arrival-trace offsets;
- report bootstrap confidence intervals for goodput and latency percentiles.

Do not accept an SLO point whose confidence interval crosses the SLO boundary.

Long reasoning requests can make short experiments misleading. The measurement
window must be long enough to include multiple heavy-tail requests and allow all
tiers to reach stationary queue behavior.

## Initial Experiment Sequence

### Phase 1: characterize isolated stages

Measure prefill, fast reasoning decode, slow visible decode, and NIXL transfer in
isolation across sequence lengths, batch sizes, and TP layouts. Estimate
`C_fast`, `C_slow`, and transfer capacity.

### Phase 2: establish one-shot Pareto

Run late-bound one-shot migration over all feasible fast/slow GPU splits and an
offered-load sweep. Identify whether fast decode, slow decode, KV capacity, or
transfer is the first bottleneck.

### Phase 3: reservation timing

Sweep prepared-pool size and reservation lead time. Determine how much slow-tier
headroom is required to absorb bursty reasoning completions while maintaining
20 visible tokens/second.

### Phase 4: workload skew

Repeat the best configurations with increasing reasoning-length variance,
heavy-tailed requests, and bursty arrivals. Evaluate prediction error and
reservation misses.

### Phase 5: incremental synchronization

Add prompt warming and incremental reasoning-KV transfer. Compare them against
the best one-shot policy at equal total GPU budget and offered load.

### Phase 6: scaling controller

Implement work-based tier scaling or worker activation using predicted remaining
tokens and near-future fast completions. Evaluate scale-up lag, scale-down safety,
and oscillation under workload changes.

## Decision Criteria

Adopt a migration policy only if it produces a meaningful Pareto improvement
over fast-only, slow-only, and static routing baselines.

A useful policy should demonstrate:

- higher SLO-compliant goodput/GPU at equal P95 TTFNT; or
- lower P95 TTFNT at equal SLO-compliant goodput/GPU;
- visible-stage throughput of at least 20 tokens/second at P95;
- no meaningful task-accuracy regression;
- bounded KV reservation and transfer overhead;
- stable behavior under reasoning-length skew and arrival bursts.

The expected leading candidate is late destination selection with a small pool
of predicted prepared capacity. Incremental synchronization should be adopted
only when its reduction in final handoff latency outweighs its additional
transfer traffic, KV residency, and control-plane complexity.
