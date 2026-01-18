# Tokenizer Backpressure Bug (Decode TTFT)

## Summary
Requests are decoded and tokens are generated quickly, but **TokenizerManager processes
the output ~20–25s later**, causing extremely long TTFT. This is **not** a KV transfer
or scheduling delay. The stall is **between scheduler output send and tokenizer
handling**, consistent with **TokenizerManager backlog/backpressure**.

## Repro Run
- Log dir: `repro_logs_20260117_221857/`
- Env:
  - `DYN_LOG=debug`
  - `SGLANG_TRACE_BATCH_RIDS=6`
  - `SGLANG_TRACE_BATCH_RIDS_EVERY=1`
  - `SGLANG_TOKENIZER_TRACE_RECV=1`
  - `SGLANG_TOKENIZER_TRACE_RECV_EVERY=1`

## Example Request (around 60s, max TTFT)
- `rid`: `5b920f43-bd1e-4e84-a4ce-45ca3ccdecf0`
- `frontend recv`: `2026-01-18T06:21:19.008614Z`
- **TTFT** (frontend → first token received): **23.922s**

### Stage Timeline
```
transfer_done          06:21:19.083017
waiting_to_running     06:21:19.083052
first_token_generated  06:21:19.095685
first_output_stream    06:21:19.091650
tokenizer_first_token  06:21:42.930328
first_token_received   06:21:42.930440
```

### Key Deltas
- `waiting_to_running → first_token_generated`: **0.013s**
- `first_output_stream → tokenizer_first_token`: **23.839s**
- `tokenizer_first_token → first_token_received`: **~0s**

**Conclusion:** Token is generated and sent immediately. It sits ~24s before
TokenizerManager processes it.

## Evidence (Representative Logs)
### Decode side (token generated + output sent quickly)
```
[TRACE disagg] rid=5b920f... waiting_to_running ...
[TRACE disagg] rid=5b920f... first_token_generated stage=process_batch_result_decode ...
[TRACE disagg] rid=5b920f... first_output_stream stage=stream_output_generation ...
[TRACE disagg] output_send stage=send_to_detokenizer batch_rids=...
```

### TokenizerManager backlog (recv loop shows huge pending states)
```
[TOK recv] type=BatchTokenIDOutput rids=15 states=850 recv_wait_ms=0.2 more=True
[TOK recv] type=BatchTokenIDOutput rids=13 states=850 recv_wait_ms=0.1 more=True
```
- `states=850` indicates heavy backlog.
- `more=True` indicates ZMQ still has buffered messages ready.
- `recv_wait_ms` is tiny → not waiting on recv, but overwhelmed by queued work.

### TokenizerManager first token (late)
```
[TRACE output] rid=5b920f... tokenizer_first_token output_len=1 pending=256
[TRACE output] rid=5b920f... tokenizer_event_set output_len=1 pending=256
```

## What’s Occurring
1. Decode receives request, prealloc/transfer completes quickly.
2. Scheduler admits request to running and generates first token within milliseconds.
3. Scheduler sends `BatchTokenIDOutput` immediately.
4. TokenizerManager has a large backlog (`states=850`), so it processes this output
   ~24s later.
5. The decode handler only receives the first token once TokenizerManager signals
   the async event, so TTFT is dominated by output-path backlog.

## Code Flow (first_output_stream → tokenizer_first_token)
1. Scheduler stream path: `scheduler_output_processor_mixin.py`
   - `stream_output_generation()` logs `first_output_stream`
   - sends `BatchTokenIDOutput` via `send_to_detokenizer`
2. TokenizerManager receives: `tokenizer_manager.py`
   - `handle_loop()` awaits `recv_pyobj()` (ZMQ PULL)
   - `_handle_batch_output()` extends `state.output_ids`
   - logs `tokenizer_first_token` and `tokenizer_event_set`
   - sets `state.event`, unblocking the async generator

## Why This Is Likely “Backpressure”
The output queue between scheduler and TokenizerManager is saturated. The
TokenizerManager loop processes many pending requests, and the specific request’s
first token is not handled until much later, even though decoding completed.

## Profiling Evidence (2026-01-18 07:54–07:55)
### Repro + Env
- Log dir: `repro_logs_20260117_235319/`
- Profile path: `repro_profile.pstats`
  - Source file: `/tmp/tokenizer_profile_20260118_075528_states1024.pstats`
- Env (excerpt):
  - `SGLANG_TOKENIZER_PROFILE=1`
  - `SGLANG_TOKENIZER_PROFILE_SECONDS=45`
  - `SGLANG_TOKENIZER_PROFILE_TRIGGER_STATES=800`
  - `SGLANG_TOKENIZER_DISABLE_METRICS=1`
  - `SGLANG_TOKENIZER_LOG_INTERVAL_S=5`
  - `SGLANG_TOKENIZER_HIGH_CPU_PCT=80`

### Evidence of TTFT gap inside the profile window
Profile window: **07:54:43.324878 → 07:55:28.349519** (45s).

Within that window (1200 rids whose `first_output_stream` fell inside the window):
- **median:** 0.232s
- **p95:** 33.205s
- **p99:** 41.176s
- **max:** 48.835s

Example RID with max gap inside the profile window:
```
2026-01-18T07:55:00.330146Z [TRACE disagg] rid=20cba755... first_output_stream ...
2026-01-18T07:55:49.165298Z [TRACE output] rid=20cba755... tokenizer_first_token ... pending=1014
```

### CPU saturation during backlog
During the same window, TokenizerManager reported sustained high CPU:
```
[TOK cpu_high] proc=225.0 sys=7.1 states=1024
```
This indicates the process is CPU-bound (not waiting on ZMQ) while backlog is high.

## cProfile Breakdown (metrics disabled)
This profile isolates non-metrics overhead in the output path.

### High-level hot functions
Top cumulative time from the profile:
- `decode_handler.generate` → **10.469s** (409,000 calls)
- `decode_handler._process_token_stream` → **9.369s** (408,755 calls)
- `decode_handler.decode_stream` → **6.635s** (408,509 calls)
- `TokenizerManager._handle_batch_output` → **5.606s** (541 calls)
- `TokenizerManager._wait_one_response` → **5.656s** (410,098 calls)
- `TokenizerManager.generate_request` → **6.742s** (408,754 calls)
- ZMQ `recv_pyobj` → **0.295s** (541 calls)

### Low-level hot functions inside `_handle_batch_output`
Most time is in *very small per-call costs, repeated at high volume*:
- `isinstance` → **0.795s / 780,489 calls ≈ 1.0µs/call**
- `asyncio.Event.set` → **0.418s / 259,833 calls ≈ 1.6µs/call**
- `list.copy` (`output_ids.copy`) → **0.366s / 259,833 calls ≈ 1.4µs/call**
- `getattr` → **0.226s / 779,499 calls ≈ 0.29µs/call**
- dict `get` / dict `update` / list `extend` / list `append` → **~0.33s combined**

**Interpretation:** per-call costs are reasonable, but the call volume is enormous.
The backpressure is therefore driven by sheer per-token Python work in
TokenizerManager + Dynamo decode handler, not by ZMQ IO latency.

## Next Steps (if needed)
- Quantify distribution of `first_output_stream → tokenizer_first_token` across
  all requests to correlate TTFT with `states=...`.
- Instrument TokenizerManager dispatch time per batch to see if specific
  operations are slow.
- Check for excessive `rid_to_state` growth (requests not cleared).
