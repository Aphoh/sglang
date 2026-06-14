"""Checkpoint/restore workload for Qwen3 TP2 with Movin collectives."""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


GSM8K_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"


def wait_for(path: Path, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for {path}")
        time.sleep(0.1)


def assert_no_io_uring() -> None:
    rings = []
    for fd_path in Path("/proc/self/fd").iterdir():
        try:
            if os.readlink(fd_path) == "anon_inode:[io_uring]":
                rings.append(fd_path.name)
        except FileNotFoundError:
            continue
    if rings:
        raise RuntimeError(f"CRIU-incompatible io_uring descriptors remain: {rings}")


def load_gsm8k_cases(
    data_path: str,
    num_questions: int,
    num_shots: int,
    tokenizer_path: str,
    use_chat_template: bool,
    enable_thinking: bool,
) -> tuple[list[str], list[int]]:
    repo_root = str(Path(__file__).resolve().parents[3])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from benchmark.gsm8k.bench_sglang import (
        INVALID,
        get_answer_value,
        get_few_shot_examples,
        get_one_example,
    )
    from sglang.utils import download_and_cache_file, read_jsonl

    if not os.path.isfile(data_path):
        data_path = download_and_cache_file(GSM8K_URL)
    lines = list(read_jsonl(data_path))
    few_shot_examples = get_few_shot_examples(lines, num_shots)
    tokenizer = None
    if use_chat_template:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
        )
    prompts = []
    for index in range(min(num_questions, len(lines))):
        prompt = few_shot_examples + get_one_example(lines, index, False)
        if tokenizer is not None:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        prompts.append(prompt)
    labels = [get_answer_value(lines[i]["answer"]) for i in range(len(prompts))]
    if not prompts:
        raise ValueError("GSM8K correctness test requires at least one question")
    if any(label == INVALID for label in labels):
        raise ValueError("GSM8K dataset contains an answer the benchmark cannot parse")
    return prompts, labels


def run_gsm8k(
    engine: Any,
    prompts: list[str],
    labels: list[int],
    max_new_tokens: int,
) -> tuple[dict[str, Any], list[str]]:
    from benchmark.gsm8k.bench_sglang import INVALID, get_answer_value

    sampling = {
        "temperature": 0,
        "top_p": 1.0,
        "max_new_tokens": max_new_tokens,
        "stop": ["Question", "Assistant:", "<|separator|>"],
    }
    started = time.perf_counter()
    outputs = engine.generate(prompts, sampling)
    latency = time.perf_counter() - started
    texts = [output["text"] for output in outputs]
    predictions = [get_answer_value(text) for text in texts]
    correct = [prediction == label for prediction, label in zip(predictions, labels)]
    completion_tokens = sum(
        output["meta_info"]["completion_tokens"] for output in outputs
    )
    result = {
        "accuracy": sum(correct) / len(correct),
        "completion_tokens": completion_tokens,
        "invalid_rate": sum(prediction == INVALID for prediction in predictions)
        / len(predictions),
        "labels": labels,
        "latency_seconds": latency,
        "num_questions": len(prompts),
        "output_throughput": completion_tokens / latency,
        "predictions": predictions,
    }
    return result, texts


def log_movin_status(engine: Any, phase: str) -> None:
    print(f"requesting_movin_status={phase}", flush=True)
    engine.collective_rpc("log_movin_nixl_status", phase=phase, require_clean=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--checkpoint-backend", default="none")
    parser.add_argument("--rendezvous", type=Path)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--gsm8k-data-path", default="test.jsonl")
    parser.add_argument("--gsm8k-num-questions", type=int, default=1)
    parser.add_argument("--gsm8k-num-shots", type=int, default=5)
    parser.add_argument("--gsm8k-max-new-tokens", type=int, default=64)
    parser.add_argument("--gsm8k-min-accuracy", type=float, default=0.0)
    parser.add_argument("--max-total-tokens", type=int, default=4096)
    parser.add_argument(
        "--disable-flashinfer-allreduce-fusion",
        action="store_true",
    )
    parser.add_argument(
        "--gsm8k-chat-template",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--gsm8k-enable-thinking", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpus)
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ.setdefault("UCX_TLS", "cuda_ipc,cuda_copy,sm,self")
    os.environ.setdefault("UV_USE_IO_URING", "0")
    os.environ.setdefault("USE_LIBUV", "0")
    if args.checkpoint_backend != "none":
        if args.rendezvous is None:
            raise ValueError("--rendezvous is required for checkpoint mode")
        os.environ.setdefault("SGLANG_CRIU_SUSPEND_DEVICE_PROCESS_GROUP", "1")
        os.environ.setdefault(
            "SGLANG_CRIU_DEVICE_STORE",
            str(args.rendezvous / "torch-device-store"),
        )

    import sglang as sgl

    gsm8k_prompts, gsm8k_labels = load_gsm8k_cases(
        args.gsm8k_data_path,
        args.gsm8k_num_questions,
        args.gsm8k_num_shots,
        args.model,
        args.gsm8k_chat_template,
        args.gsm8k_enable_thinking,
    )

    engine = sgl.Engine(
        model_path=args.model,
        tp_size=args.tp_size,
        dtype="bfloat16",
        mem_fraction_static=0.15,
        max_total_tokens=args.max_total_tokens,
        max_running_requests=1,
        cuda_graph_max_bs_decode=1,
        disable_prefill_cuda_graph=True,
        disable_overlap_schedule=True,
        disable_radix_cache=True,
        disable_custom_all_reduce=True,
        enable_flashinfer_allreduce_fusion=(
            not args.disable_flashinfer_allreduce_fusion
        ),
        enforce_disable_flashinfer_allreduce_fusion=(
            args.disable_flashinfer_allreduce_fusion
        ),
        random_seed=7,
        log_level="info",
    )
    log_movin_status(engine, "after_engine_init")
    prompt = "The first three prime numbers are"
    sampling = {"temperature": 0, "max_new_tokens": 4}
    baseline = engine.generate(prompt, sampling)["text"]
    print(f"baseline={baseline!r}", flush=True)
    log_movin_status(engine, "after_smoke")

    assert_no_io_uring()
    after_quiesce = engine.generate(prompt, sampling)["text"]
    if not after_quiesce:
        raise AssertionError("generation after loader quiesce was empty")
    if after_quiesce != baseline:
        raise AssertionError(
            f"smoke generation changed before checkpoint: {baseline!r} -> "
            f"{after_quiesce!r}"
        )
    gsm8k_before, gsm8k_text_before = run_gsm8k(
        engine,
        gsm8k_prompts,
        gsm8k_labels,
        args.gsm8k_max_new_tokens,
    )
    print(f"gsm8k_before={json.dumps(gsm8k_before, sort_keys=True)}", flush=True)
    log_movin_status(engine, "before_checkpoint")

    if args.checkpoint_backend == "none":
        restored = after_quiesce
    else:
        args.rendezvous.mkdir(parents=True, exist_ok=True)
        engine.collective_rpc("prepare_criu")
        (args.rendezvous / "controller-pid").write_text(f"{os.getpid()}\n")
        (args.rendezvous / "worker-pids.json").write_text(
            json.dumps(engine.get_all_child_pids())
        )
        (args.rendezvous / "job-ready").touch()
        wait_for(args.rendezvous / "phase", args.timeout)
        engine.collective_rpc("restore_after_criu")
        log_movin_status(engine, "after_restore_before_generation")
        restored = engine.generate(prompt, sampling)["text"]

    print(f"restored={restored!r}", flush=True)
    if args.checkpoint_backend != "none":
        log_movin_status(engine, "after_restore_smoke")
    if not restored:
        raise AssertionError("post-restore generation was empty")
    if restored != after_quiesce:
        raise AssertionError(
            f"smoke generation changed after restore: {after_quiesce!r} -> {restored!r}"
        )
    if args.checkpoint_backend == "none":
        gsm8k_after = gsm8k_before
        gsm8k_text_after = gsm8k_text_before
    else:
        gsm8k_after, gsm8k_text_after = run_gsm8k(
            engine,
            gsm8k_prompts,
            gsm8k_labels,
            args.gsm8k_max_new_tokens,
        )
        log_movin_status(engine, "after_restore_gsm8k")
    mismatched_predictions = [
        index
        for index, (before, after) in enumerate(
            zip(gsm8k_before["predictions"], gsm8k_after["predictions"])
        )
        if before != after
    ]
    mismatched_text = [
        index
        for index, (before, after) in enumerate(
            zip(gsm8k_text_before, gsm8k_text_after)
        )
        if before != after
    ]
    gsm8k_result = {
        "after": gsm8k_after,
        "before": gsm8k_before,
        "mismatched_prediction_indices": mismatched_predictions,
        "mismatched_text_indices": mismatched_text,
    }
    print(f"gsm8k_after={json.dumps(gsm8k_after, sort_keys=True)}", flush=True)
    if mismatched_text:
        raise AssertionError(
            "GSM8K generated text changed after restore at question indices "
            f"{mismatched_text}"
        )
    print(
        "gsm8k_comparison=" + json.dumps(gsm8k_result, sort_keys=True),
        flush=True,
    )
    if mismatched_predictions:
        raise AssertionError(
            "GSM8K predictions changed after restore at question indices "
            f"{mismatched_predictions}"
        )
    if gsm8k_after["accuracy"] < args.gsm8k_min_accuracy:
        raise AssertionError(
            f"post-restore GSM8K accuracy {gsm8k_after['accuracy']:.3f} is below "
            f"the required {args.gsm8k_min_accuracy:.3f}"
        )
    if args.rendezvous is not None:
        (args.rendezvous / "gsm8k-result.json").write_text(
            json.dumps(gsm8k_result, indent=2, sort_keys=True) + "\n"
        )
        (args.rendezvous / "passed").touch()
    engine.shutdown()


if __name__ == "__main__":
    main()
