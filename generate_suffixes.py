#!/usr/bin/env python3
"""Generate suffixes by continuing from truncated prefixes.

For each prefix, constructs a prompt with the original problem and the prefix
as a partial assistant response, then lets the model continue generating.

Usage:
    python generate_suffixes.py --model qwen3-235b-thinking --start 0 --end 1
    python generate_suffixes.py --model qwen3-235b-thinking --num-suffixes 8
    python generate_suffixes.py --model qwen3-235b-thinking --start 0 --end 100 --concurrency 8
"""

import asyncio
import json
import os
import argparse
import time
from pathlib import Path

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as atqdm

# ── Model registry ──────────────────────────────────────────────────────────
MODELS = {
    "qwen3-235b-thinking": {
        "model_id": "qwen/qwen3-235b-a22b-thinking-2507",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "temperature": 0.6,
    },
    "gpt5-mini": {
        "model_id": "gpt-5-mini",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "temperature": None,
    },
}

# Models that use <think> tags for reasoning
THINK_TAG_MODELS = {"qwen3-235b-thinking"}

# Models where suffix generation is not supported
UNSUPPORTED_MODELS = {"gpt5-mini"}

# ── Token budgets by problem type ────────────────────────────────────────────
INITIAL_MAX_TOKENS_MATH = 32768
INITIAL_MAX_TOKENS_PROOF = 76800
MAX_BUDGET_ESCALATIONS = 3  # how many times to double the budget on truncation
ABSOLUTE_MAX_TOKENS = 131072  # qwen3-235b-thinking context length


def load_records(path: Path) -> dict[int, dict]:
    """Load JSONL records keyed by index."""
    records = {}
    if path.exists():
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    records[record["index"]] = record
                except (json.JSONDecodeError, KeyError):
                    continue
    return records


def load_completed_suffixes(path: Path) -> dict[int, list[dict]]:
    """Load completed suffixes grouped by index."""
    completed = {}
    if path.exists():
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    idx = record["index"]
                    if idx not in completed:
                        completed[idx] = []
                    completed[idx].append(record)
                except (json.JSONDecodeError, KeyError):
                    continue
    return completed


def extract_original_suffix(prefix_record: dict, trace_record: dict, model: str) -> dict | None:
    """Try to extract suffix 0 from the original trace.

    Returns a suffix record if the original trace was complete (finish_reason=stop
    and response is not None), otherwise returns None (needs regeneration).
    """
    if trace_record.get("finish_reason") != "stop":
        return None
    if trace_record.get("response") is None:
        return None

    prefix = prefix_record["prefix"]

    # Extract the suffix portion of the reasoning (everything after the prefix)
    if model in THINK_TAG_MODELS:
        full_reasoning = trace_record.get("reasoning", "")
        # Find where the prefix ends in the full reasoning
        prefix_pos = full_reasoning.find(prefix)
        if prefix_pos == -1:
            return None
        suffix_reasoning = full_reasoning[prefix_pos + len(prefix):]
    else:
        suffix_reasoning = None

    return {
        "index": prefix_record["index"],
        "suffix_num": 0,
        "problem": prefix_record["problem"],
        "answer": prefix_record.get("answer"),
        "source": prefix_record.get("source"),
        "mean_reward": prefix_record.get("mean_reward"),
        "prefix": prefix,
        "prefix_type": prefix_record.get("prefix_type"),
        "prefix_end_index": prefix_record.get("prefix_end_index"),
        "num_thoughts": prefix_record.get("num_thoughts"),
        "suffix_response": trace_record.get("response"),
        "suffix_reasoning": suffix_reasoning,
        "finish_reason": "stop",
        "budget_used": 0,
        "escalation": 0,
        "model": trace_record.get("model"),
        "usage": trace_record.get("usage"),
        "from_original_trace": True,
    }


def build_messages(prefix_record: dict, model: str) -> list[dict]:
    """Build the message list for suffix generation.

    For thinking models (Qwen3): prefill the assistant's <think> block with the prefix.
    """
    problem = prefix_record["problem"]
    prefix = prefix_record["prefix"]
    answer = prefix_record.get("answer")

    # Use the same prompt format as generate_traces.py
    is_proof = answer is None or str(answer).strip() in ("", "None")
    if is_proof:
        user_prompt = f"Generate a rigorous proof to the following question: \n\n{problem}"
    else:
        user_prompt = f"{problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."

    if model in THINK_TAG_MODELS:
        # Prefill assistant with open <think> tag + prefix reasoning
        # The model will continue reasoning, close </think>, and write the answer
        assistant_prefill = f"<think>\n{prefix}"
        return [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "prefix": True, "content": assistant_prefill},
        ]
    else:
        raise NotImplementedError(
            f"Suffix generation not supported for model '{model}': "
            f"internal chain-of-thought is not exposed by the API, "
            f"so prefix continuation is not possible."
        )


async def generate_one_suffix(
    client: AsyncOpenAI,
    model_id: str,
    model: str,
    prefix_record: dict,
    suffix_num: int,
    semaphore: asyncio.Semaphore,
    write_lock: asyncio.Lock,
    output_file,
    temperature: float,
    max_retries: int = 5,
):
    """Generate a single suffix with budget escalation on truncation."""
    async with semaphore:
        messages = build_messages(prefix_record, model)
        index = prefix_record["index"]
        answer = prefix_record.get("answer")
        is_proof = answer is None or str(answer).strip() in ("", "None")
        base_budget = INITIAL_MAX_TOKENS_PROOF if is_proof else INITIAL_MAX_TOKENS_MATH

        last_error = None
        last_result = None

        # Try with escalating budgets
        for escalation in range(MAX_BUDGET_ESCALATIONS + 1):
            budget = min(base_budget * (2 ** escalation), ABSOLUTE_MAX_TOKENS)

            for attempt in range(max_retries):
                try:
                    kwargs = dict(
                        model=model_id,
                        messages=messages,
                        max_completion_tokens=budget,
                    )
                    if temperature is not None:
                        kwargs["temperature"] = temperature

                    response = await client.chat.completions.create(**kwargs)
                    choice = response.choices[0]

                    # Extract reasoning/thinking if present
                    reasoning = None
                    content = choice.message.content
                    if hasattr(choice.message, "reasoning_content") and choice.message.reasoning_content:
                        reasoning = choice.message.reasoning_content
                    elif hasattr(choice.message, "reasoning") and choice.message.reasoning:
                        reasoning = choice.message.reasoning

                    result = {
                        "index": index,
                        "suffix_num": suffix_num,
                        "problem": prefix_record["problem"],
                        "answer": prefix_record.get("answer"),
                        "source": prefix_record.get("source"),
                        "mean_reward": prefix_record.get("mean_reward"),
                        "prefix": prefix_record["prefix"],
                        "prefix_type": prefix_record.get("prefix_type"),
                        "prefix_end_index": prefix_record.get("prefix_end_index"),
                        "num_thoughts": prefix_record.get("num_thoughts"),
                        "suffix_response": content,
                        "suffix_reasoning": reasoning,
                        "finish_reason": choice.finish_reason,
                        "budget_used": budget,
                        "escalation": escalation,
                        "model": response.model,
                        "usage": {
                            "prompt_tokens": response.usage.prompt_tokens,
                            "completion_tokens": response.usage.completion_tokens,
                            "total_tokens": response.usage.total_tokens,
                        }
                        if response.usage
                        else None,
                    }

                    # If finished, write and return
                    if choice.finish_reason == "stop":
                        async with write_lock:
                            output_file.write(json.dumps(result) + "\n")
                            output_file.flush()
                        return result

                    # Truncated — save as last_result, try higher budget
                    last_result = result
                    break  # break retry loop, escalate budget

                except Exception as e:
                    last_error = e
                    wait = min(2**attempt * 2, 60)
                    if attempt < max_retries - 1:
                        await asyncio.sleep(wait)
            else:
                # All retries exhausted at this budget level — write error
                result = {
                    "index": index,
                    "suffix_num": suffix_num,
                    "problem": prefix_record["problem"],
                    "answer": prefix_record.get("answer"),
                    "prefix": prefix_record["prefix"],
                    "suffix_response": None,
                    "finish_reason": None,
                    "budget_used": budget,
                    "error": str(last_error),
                    "error_type": type(last_error).__name__,
                }
                async with write_lock:
                    output_file.write(json.dumps(result) + "\n")
                    output_file.flush()
                return result

            # If we already hit absolute max, stop escalating
            if budget >= ABSOLUTE_MAX_TOKENS:
                break

        # Exhausted all budget escalations — write the truncated result
        if last_result:
            async with write_lock:
                output_file.write(json.dumps(last_result) + "\n")
                output_file.flush()
            return last_result

        # Shouldn't reach here, but just in case
        result = {
            "index": index,
            "suffix_num": suffix_num,
            "problem": prefix_record["problem"],
            "answer": prefix_record.get("answer"),
            "prefix": prefix_record["prefix"],
            "suffix_response": None,
            "finish_reason": None,
            "budget_used": None,
            "error": str(last_error) if last_error else "Unknown error",
            "error_type": type(last_error).__name__ if last_error else "Unknown",
        }
        async with write_lock:
            output_file.write(json.dumps(result) + "\n")
            output_file.flush()
        return result


async def main():
    parser = argparse.ArgumentParser(description="Generate suffixes from prefixes")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODELS.keys()),
        help="Model to use for suffix generation",
    )
    parser.add_argument("--trace-dir", type=str, default="traces")
    parser.add_argument("--prefix-dir", type=str, default="prefixes")
    parser.add_argument("--output-dir", type=str, default="suffixes")
    parser.add_argument("--num-suffixes", "-n", type=int, default=7,
                        help="Number of suffixes to generate per prefix")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=None,
                        help="Override temperature")
    parser.add_argument("--start", type=int, default=0, help="Start index (inclusive)")
    parser.add_argument("--end", type=int, default=None, help="End index (exclusive)")
    args = parser.parse_args()

    model = args.model

    if model in UNSUPPORTED_MODELS:
        raise NotImplementedError(
            f"Suffix generation not supported for model '{model}': "
            f"internal chain-of-thought is not exposed by the API, "
            f"so prefix continuation is not possible."
        )

    model_cfg = MODELS[model]
    model_id = model_cfg["model_id"]

    # Temperature: CLI > per-model > default 0.6
    if args.temperature is not None:
        temperature = args.temperature
    elif "temperature" in model_cfg:
        temperature = model_cfg["temperature"]
    else:
        temperature = 0.6

    trace_path = Path(args.trace_dir) / f"{model}.jsonl"
    prefix_path = Path(args.prefix_dir) / f"{model}.jsonl"
    output_path = Path(args.output_dir) / f"{model}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not prefix_path.exists():
        raise FileNotFoundError(f"Prefix file not found: {prefix_path}")

    # Load prefixes and original traces
    prefixes = load_records(prefix_path)
    traces = load_records(trace_path) if trace_path.exists() else {}

    # Determine index range
    all_indices = sorted(prefixes.keys())
    start = args.start
    end = args.end if args.end is not None else max(all_indices) + 1
    indices = [i for i in all_indices if start <= i < end]

    # Filter out skipped prefixes (no content to continue from)
    indices = [i for i in indices if prefixes[i].get("prefix") is not None]

    # Resume support: check which (index, suffix_num) pairs are done
    completed = load_completed_suffixes(output_path)

    # Handle suffix 0 (original trace) separately — write directly, no API call
    n_original_written = 0
    with open(output_path, "a") as f:
        for idx in indices:
            existing = completed.get(idx, [])
            existing_nums = {r["suffix_num"] for r in existing}
            if 0 in existing_nums:
                continue
            if idx in traces:
                original = extract_original_suffix(prefixes[idx], traces[idx], model)
                if original is not None:
                    f.write(json.dumps(original) + "\n")
                    f.flush()
                    # Track it as completed
                    if idx not in completed:
                        completed[idx] = []
                    completed[idx].append(original)
                    n_original_written += 1

    if n_original_written:
        print(f"Wrote {n_original_written} original trace suffixes (suffix 0)")

    # Build remaining tasks: suffix 0 that needs regeneration + suffixes 1..N-1
    tasks_to_run = []
    for idx in indices:
        existing = completed.get(idx, [])
        existing_nums = {r["suffix_num"] for r in existing}
        for s in range(args.num_suffixes):
            if s not in existing_nums:
                tasks_to_run.append((idx, s))

    n_total = len(indices) * args.num_suffixes
    n_done = n_total - len(tasks_to_run)

    print(f"Model:       {model_id}")
    print(f"Prefixes:    {prefix_path} ({len(prefixes)} records)")
    print(f"Range:       [{start}, {end})")
    print(f"Indices:     {len(indices)} (with valid prefix)")
    print(f"Suffixes/prefix: {args.num_suffixes}")
    print(f"Total jobs:  {n_total}")
    print(f"Completed:   {n_done}")
    print(f"Remaining:   {len(tasks_to_run)}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Temperature: {temperature}")
    print(f"Output:      {output_path}")
    print()

    if not tasks_to_run:
        print("All suffixes already generated!")
        return

    # Setup client
    api_key_env = model_cfg["api_key_env"]
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"{api_key_env} environment variable not set")

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=model_cfg["base_url"],
        max_retries=0,
        timeout=3600,
    )

    semaphore = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()

    t0 = time.time()
    with open(output_path, "a") as f:
        coros = [
            generate_one_suffix(
                client,
                model_id,
                model,
                prefixes[idx],
                suffix_num,
                semaphore,
                write_lock,
                f,
                temperature,
            )
            for idx, suffix_num in tasks_to_run
        ]

        results = []
        for coro in atqdm(
            asyncio.as_completed(coros),
            total=len(coros),
            desc=f"Suffixes ({model})",
        ):
            result = await coro
            results.append(result)

    elapsed = time.time() - t0
    n_stop = sum(1 for r in results if r.get("finish_reason") == "stop")
    n_length = sum(1 for r in results if r.get("finish_reason") == "length")
    n_error = sum(1 for r in results if r.get("suffix_response") is None)

    print()
    print(f"Done in {elapsed:.1f}s")
    print(f"  Finished (stop):    {n_stop}")
    print(f"  Truncated (length): {n_length}")
    print(f"  Errors:             {n_error}")
    print(f"  Output:             {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
