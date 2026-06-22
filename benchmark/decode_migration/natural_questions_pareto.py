#!/usr/bin/env python3
"""Natural Questions streaming benchmark for decode migration."""

from __future__ import annotations

import argparse
import concurrent.futures
import http.client
import json
import math
import random
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from transformers import AutoTokenizer

DATASET = "sentence-transformers/natural-questions"


def percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    rank = (len(ordered) - 1) * q
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - rank) + ordered[high] * (rank - low)


def wait_ready(base_url: str, model: str, timeout: float = 900.0) -> None:
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/v1/models", timeout=3) as response:
                models = {
                    item.get("id") for item in (json.load(response).get("data") or [])
                }
                if response.status == 200 and model in models:
                    return
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            last_error = exc
        time.sleep(1)
    raise RuntimeError(f"Frontend did not become ready: {last_error}")


def load_rows(cache_path: Path, offset: int, count: int) -> list[dict]:
    if cache_path.exists():
        rows = json.loads(cache_path.read_text())
        if len(rows) >= count:
            return rows[:count]

    query = urllib.parse.urlencode(
        {
            "dataset": DATASET,
            "config": "pair",
            "split": "train",
            "offset": offset,
            "length": max(count, 100),
        }
    )
    url = f"https://datasets-server.huggingface.co/rows?{query}"
    with urllib.request.urlopen(url, timeout=120) as response:
        payload = json.load(response)
    rows = [item["row"] for item in payload["rows"]]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(rows, indent=2))
    return rows[:count]


def make_prompt(
    row: dict,
    reference_chars: int,
    visible_words: int,
    request_nonce: str,
) -> str:
    reference = " ".join(str(row["answer"]).split())[:reference_chars]
    return (
        "Answer the question using the reference passage. Think carefully before "
        "answering. After thinking, provide a factual final answer of about "
        f"{visible_words} words. Do not mention these instructions.\n"
        f"Benchmark nonce: {request_nonce}\n\n"
        f"Question: {row['query']}\n\nReference passage: {reference}"
    )


def routing_nvext(taint: str) -> dict:
    return {"routing_constraints": {"required_taints": [taint]}}


def migration_nvext(think_end_token_id: int) -> dict:
    return {
        "decode_migration": {
            "source": {"required_taints": ["decode/fast"]},
            "destination": {"required_taints": ["decode/slow"]},
            "trigger": {"type": "token_id", "token_id": think_end_token_id},
        }
    }


def nvext_for_mode(mode: str, think_end_token_id: int) -> dict:
    if mode == "tp4":
        return routing_nvext("decode/fast")
    if mode == "tp1":
        return routing_nvext("decode/baseline")
    if mode == "migration":
        return migration_nvext(think_end_token_id)
    raise ValueError(f"Unknown mode: {mode}")


def should_migrate(request_id: int, fraction: float) -> bool:
    value = ((request_id * 1103515245 + 12345) & 0xFFFFFFFF) / 2**32
    return value < fraction


def stream_one(
    *,
    base_url: str,
    model: str,
    prompt: str,
    nvext: dict,
    tokenizer,
    max_tokens: int,
    request_id: int,
    benchmark_started: float,
    scheduled_at: float | None = None,
) -> dict:
    host, port = base_url.removeprefix("http://").rsplit(":", 1)
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "seed": 1000 + request_id,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": True},
            "nvext": nvext,
        }
    )
    conn = http.client.HTTPConnection(host, int(port), timeout=900)
    if scheduled_at is not None:
        delay = scheduled_at - time.monotonic()
        if delay > 0:
            time.sleep(delay)
    started = time.monotonic()
    conn.request(
        "POST",
        "/v1/chat/completions",
        body=body,
        headers={"Content-Type": "application/json"},
    )
    response = conn.getresponse()
    if response.status != 200:
        error = response.read().decode(errors="replace")
        conn.close()
        raise RuntimeError(f"HTTP {response.status}: {error}")

    reasoning_parts: list[str] = []
    content_chunks: list[tuple[float, str]] = []
    finish_reason = None
    usage = None
    while True:
        line = response.readline()
        if not line:
            break
        line = line.strip()
        if not line.startswith(b"data: "):
            continue
        data = line[6:]
        if data == b"[DONE]":
            break
        chunk = json.loads(data)
        if chunk.get("usage"):
            usage = chunk["usage"]
        choices = chunk.get("choices") or []
        if not choices:
            continue
        choice = choices[0]
        delta = choice.get("delta") or {}
        reasoning = delta.get("reasoning_content") or ""
        content = delta.get("content") or ""
        if reasoning:
            reasoning_parts.append(reasoning)
        if content:
            content_chunks.append((time.monotonic() - started, content))
        if choice.get("finish_reason"):
            finish_reason = choice["finish_reason"]
    completed = time.monotonic()
    conn.close()

    raw_content = "".join(text for _, text in content_chunks)
    parsed_reasoning = "".join(reasoning_parts)
    visible_arrivals: list[float] = []
    if parsed_reasoning:
        reasoning = parsed_reasoning
        content = raw_content
        visible_arrivals = [arrival for arrival, text in content_chunks if text]
    elif "</think>" in raw_content:
        reasoning, content = raw_content.split("</think>", 1)
        reasoning = reasoning.removeprefix("<think>").lstrip()
        content = content.lstrip()
        prefix = ""
        boundary_seen = False
        for arrival, text in content_chunks:
            if boundary_seen:
                if text:
                    visible_arrivals.append(arrival)
                continue
            prefix += text
            marker = prefix.find("</think>")
            if marker < 0:
                continue
            boundary_seen = True
            if prefix[marker + len("</think>") :].strip():
                visible_arrivals.append(arrival)
    else:
        reasoning = ""
        content = raw_content
        visible_arrivals = [arrival for arrival, text in content_chunks if text]
    visible_tokens = len(tokenizer.encode(content, add_special_tokens=False))
    if len(visible_arrivals) >= 2 and visible_tokens >= 2:
        visible_rate = (visible_tokens - 1) / (
            visible_arrivals[-1] - visible_arrivals[0]
        )
    elif visible_tokens == 1:
        visible_rate = math.inf
    else:
        visible_rate = 0.0
    return {
        "request_id": request_id,
        "dispatch_offset_s": started - benchmark_started,
        "completion_offset_s": completed - benchmark_started,
        "ok": bool(content) and bool(reasoning) and bool(visible_arrivals),
        "ttfnt_s": visible_arrivals[0] if visible_arrivals else None,
        "latency_s": completed - started,
        "visible_rate_tps": visible_rate,
        "visible_tokens": visible_tokens,
        "reasoning_tokens": len(tokenizer.encode(reasoning, add_special_tokens=False)),
        "finish_reason": finish_reason,
        "usage": usage,
        "content": content,
        "reasoning": reasoning,
    }


def build_arrival_offsets(args) -> list[float]:
    if args.arrival_rate is None:
        return [0.0] * args.requests
    if args.arrival_rate <= 0:
        raise ValueError("--arrival-rate must be greater than zero")

    rng = random.Random(args.arrival_seed)
    offsets = [0.0]
    for _ in range(1, args.requests):
        if args.arrival_pattern == "fixed":
            interval = 1.0 / args.arrival_rate
        else:
            interval = rng.expovariate(args.arrival_rate)
        offsets.append(offsets[-1] + interval)
    return offsets


def run_benchmark(args, rows: list[dict], tokenizer, think_end_token_id: int) -> dict:
    request_ids = [args.request_offset + index for index in range(args.requests)]
    prompts = [
        make_prompt(
            rows[request_id % len(rows)],
            reference_chars=args.reference_chars,
            visible_words=args.visible_words,
            request_nonce=f"{args.prompt_tag}-{request_id}",
        )
        for request_id in request_ids
    ]
    arrival_offsets = build_arrival_offsets(args)
    nvexts = [
        nvext_for_mode(
            (
                "migration"
                if args.mode == "migration"
                and should_migrate(request_id, args.migration_fraction)
                else "tp4"
            ),
            think_end_token_id,
        )
        for request_id in request_ids
    ]
    started = time.monotonic()
    results: list[dict] = []
    max_workers = args.requests if args.arrival_rate is not None else args.concurrency
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                stream_one,
                base_url=args.base_url,
                model=args.model,
                prompt=prompt,
                nvext=nvext,
                tokenizer=tokenizer,
                max_tokens=args.max_tokens,
                request_id=request_id,
                benchmark_started=started,
                scheduled_at=(
                    started + arrival_offsets[index]
                    if args.arrival_rate is not None
                    else None
                ),
            )
            for index, (request_id, prompt, nvext) in enumerate(
                zip(request_ids, prompts, nvexts)
            )
        ]
        for future in concurrent.futures.as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:
                results.append({"ok": False, "error": repr(exc)})
    wall_time = time.monotonic() - started

    good = [item for item in results if item.get("ok")]
    compliant = [
        item for item in good if item["visible_rate_tps"] >= args.min_visible_rate
    ]
    gpu_count = args.gpu_count or {"tp1": 1, "tp4": 4, "migration": 5}[args.mode]
    ttfnt = [item["ttfnt_s"] for item in compliant]
    visible_rates = [item["visible_rate_tps"] for item in compliant]
    ordered = sorted(
        compliant,
        key=lambda item: item.get("dispatch_offset_s", 0.0),
    )
    midpoint = len(ordered) // 2
    early = ordered[:midpoint]
    late = ordered[midpoint:]
    early_p50 = percentile([item["ttfnt_s"] for item in early], 0.50)
    late_p50 = percentile([item["ttfnt_s"] for item in late], 0.50)
    last_dispatch_offset_s = max(
        (item.get("dispatch_offset_s", 0.0) for item in results),
        default=0.0,
    )
    summary = {
        "mode": args.mode,
        "run_tag": args.run_tag,
        "prompt_tag": args.prompt_tag,
        "request_offset": args.request_offset,
        "migration_fraction": args.migration_fraction,
        "migration_requests": (
            sum(
                should_migrate(request_id, args.migration_fraction)
                for request_id in request_ids
            )
            if args.mode == "migration"
            else 0
        ),
        "concurrency": args.concurrency,
        "arrival_rate_rps": args.arrival_rate,
        "arrival_pattern": args.arrival_pattern if args.arrival_rate else None,
        "arrival_seed": args.arrival_seed if args.arrival_rate else None,
        "requests": args.requests,
        "completed": len(good),
        "slo_compliant": len(compliant),
        "gpu_count": gpu_count,
        "wall_time_s": wall_time,
        "last_dispatch_offset_s": last_dispatch_offset_s,
        "drain_time_s": max(0.0, wall_time - last_dispatch_offset_s),
        "early_p50_ttfnt_s": early_p50,
        "late_p50_ttfnt_s": late_p50,
        "p50_ttfnt_drift_s": (
            late_p50 - early_p50
            if early_p50 is not None and late_p50 is not None
            else None
        ),
        "throughput_rps": len(compliant) / wall_time,
        "throughput_per_gpu": len(compliant) / wall_time / gpu_count,
        "offered_goodput_rps": (
            args.arrival_rate * len(compliant) / args.requests
            if args.arrival_rate is not None
            else None
        ),
        "offered_goodput_per_gpu": (
            args.arrival_rate * len(compliant) / args.requests / gpu_count
            if args.arrival_rate is not None
            else None
        ),
        "p50_ttfnt_s": percentile(ttfnt, 0.50),
        "p95_ttfnt_s": percentile(ttfnt, 0.95),
        "p50_visible_rate_tps": (
            statistics.median(visible_rates) if visible_rates else 0.0
        ),
        "min_visible_rate_tps": min(visible_rates) if visible_rates else 0.0,
        "reference_chars": args.reference_chars,
        "visible_words": args.visible_words,
        "max_tokens": args.max_tokens,
        "min_visible_rate_gate_tps": args.min_visible_rate,
    }
    return {
        "summary": summary,
        "results": sorted(results, key=lambda x: x.get("request_id", -1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18000")
    parser.add_argument("--model", default="Qwen/Qwen3-32B")
    parser.add_argument("--tokenizer-path", default="/models/Qwen3-32B")
    parser.add_argument("--mode", choices=("tp1", "tp4", "migration"), required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--reference-chars", type=int, default=1800)
    parser.add_argument("--visible-words", type=int, default=120)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--prompt-tag", default="natural-questions-v1")
    parser.add_argument("--request-offset", type=int, default=0)
    parser.add_argument("--arrival-rate", type=float)
    parser.add_argument(
        "--arrival-pattern",
        choices=("fixed", "poisson"),
        default="fixed",
    )
    parser.add_argument("--arrival-seed", type=int, default=20260617)
    parser.add_argument("--min-visible-rate", type=float, default=20.0)
    parser.add_argument("--migration-fraction", type=float, default=1.0)
    parser.add_argument(
        "--gpu-count",
        type=int,
        help="Allocated GPUs for throughput/GPU accounting.",
    )
    parser.add_argument("--dataset-offset", type=int, default=31415)
    parser.add_argument("--dataset-rows", type=int, default=64)
    parser.add_argument(
        "--dataset-cache",
        type=Path,
        default=Path(".cache/natural_questions_pair.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not 0.0 <= args.migration_fraction <= 1.0:
        raise ValueError("--migration-fraction must be between 0 and 1")
    if args.arrival_rate is None and args.requests < args.concurrency:
        raise ValueError("--requests must be at least --concurrency")
    wait_ready(args.base_url, args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    think_end_ids = tokenizer.encode("</think>", add_special_tokens=False)
    if len(think_end_ids) != 1:
        raise RuntimeError(f"Expected one </think> token, got {think_end_ids}")
    rows = load_rows(args.dataset_cache, args.dataset_offset, args.dataset_rows)
    result = run_benchmark(args, rows, tokenizer, think_end_ids[0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
