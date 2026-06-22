# Qwen3-32B Decode Migration Results

## Controlled Static Workload

The final June 17, 2026 experiment isolates decode and migration behavior:

- model: `Qwen/Qwen3-32B-FP8`;
- context length: 32K;
- logical ISL: 1 token;
- fixed OSL: 512 tokens;
- fast stage: 307 output tokens (60%);
- TTFNT: arrival of output token 308;
- `--stream-interval 1`;
- overlap scheduling enabled;
- deterministic sampling, `temperature=0`;
- open-loop arrivals with 45 seconds of warmup and cooldown traffic;
- 256 measured requests;
- minimum slow-stage rate: 20 tokens/second.

The workers were not collocated. Pure TP4 used physical GPUs 0-3. Migration
used a TP2 source on GPUs 4-5 and a TP1 destination on GPU 6. Both modes used
the same Dynamo frontend, model, request shape, and measurement window.

## Confirmed Pareto Point

| Metric | Pure TP4 | TP2 -> TP1 migration |
| --- | ---: | ---: |
| Arrival rate | 5.200 req/s | 5.075 req/s |
| Allocated GPUs | 4 | 3 |
| Completed | 256/256 | 256/256 |
| Offered goodput/GPU | 1.300 req/s | 1.692 req/s |
| Completion throughput/GPU including drain | 1.047 req/s | 1.430 req/s |
| P95 source-boundary latency | 7.632 s | 7.267 s |
| P95 handoff gap | 0.090 s | 0.192 s |
| P95 TTFNT | 7.646 s | 7.424 s |
| P50 TTFNT drift, late minus early | +0.129 s | -0.079 s |
| Minimum visible rate | 39.1 tok/s | 79.4 tok/s |

At matched TTFNT, migration improves offered throughput/GPU by 30.1% and lowers
P95 TTFNT by 2.9%. The secondary completion-throughput metric, which includes
finite-run drain time, improves by 36.6%.

All measured requests returned exactly 512 completion tokens with no fallback
stream chunks. Across the continuous migration window, all 1,428 TP-rank
prepare events had the exact exported frontier:

```text
committed=307 logical=308 actual_logical=308 seen=307
```

Prepare, transfer-complete, and source-commit counts were all 1,428. The exact
boundary gate also passed 32/32 TP-rank transfers.

## Refinement Sweep

| Pure TP4 rate | TP4 P95 TTFNT | TP4 req/s/GPU | Migration rate | Migration P95 TTFNT | Migration req/s/GPU |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5.20 | 7.115 s | 1.300 | 5.05 | 7.037 s | 1.683 |
| 5.60 | 7.772 s | 1.400 | 5.35 | 7.966 s | 1.783 |
| 6.00 | 8.362 s | 1.500 | 5.50 | 8.114 s | 1.833 |

Every point completed 64/64 measured requests and met the visible-rate gate.
The confirmation run increased the first pair to 256 measured requests.

## What Did Not Work

TP4 source plus one TP1 destination used five GPUs and did not beat pure TP4 per
GPU. Adding a second TP1 destination also did not help because the source tier,
not destination compute, was limiting the tested rates.

Start-empty finite runs produced misleading migration P95 values. Requests
formed large cohorts, and the last requests accelerated when arrivals stopped.
Continuous warmup and cooldown traffic removed both artifacts.

## Accuracy Context

This static run validates transport and stream correctness, not task accuracy.
Separate checks remain relevant:

- Qwen3-8B GSM8K, 200 samples: 96.0% without migration and 96.5% with migration;
- Qwen3-32B Natural Questions proxy scoring, 128 paired outputs: reference-passage
  token F1 differed by +0.0014 in favor of migration.

Future performance points should retain a task-accuracy gate with the exact
model, TP topology, and sampling configuration under test.
