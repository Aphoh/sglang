# Repro cProfile Summary (Tokenizer Backpressure)

## Files
- Profile: `repro_profile.pstats`
- Source run: `repro_logs_20260117_235319/` (profile window 45s)
- Run flags: `SGLANG_TOKENIZER_DISABLE_METRICS=1`

## How to inspect
```
python -m pstats repro_profile.pstats
```
Suggested commands inside pstats:
```
sort cumtime
stats 40
stats tokenizer_manager.py
```

## What is slow (metrics disabled)
The hot path shifts to **per-token Python overhead** in both the tokenizer
output handler and the Dynamo decode handler.

Top cumulative callers (from profile):
- `decode_handler.generate` / `_process_token_stream` / `decode_stream`
- `TokenizerManager._handle_batch_output`
  - `builtins.isinstance` → **~0.80s**
  - `asyncio.Event.set` → **~0.42s**
  - `list.copy` (from `output_ids.copy`) → **~0.37s**
  - `builtins.getattr` / dict `get` / dict `update` → **~0.45s combined**
  - list `extend` / `append` → **~0.12s combined**

ZMQ receive (`recv_pyobj`) is ~0.30s in this window, so it is **not** the
dominant cost relative to per-request bookkeeping.

## Note on Dynamo decode handler
With metrics disabled, decode handler functions become the top cumulative
costs (`generate`, `decode_stream`, `_process_token_stream`). This suggests
per-token processing in Dynamo is a significant contributor once tokenizer
metrics are removed.

## Verification toggle
To compare metrics-on vs metrics-off:
```
SGLANG_TOKENIZER_DISABLE_METRICS=1
```

