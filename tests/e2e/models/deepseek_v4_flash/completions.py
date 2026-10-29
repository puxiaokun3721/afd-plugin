# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Cancellable ten-request acceptance with per-request diagnostics."""

from __future__ import annotations

import asyncio
import json
import math
import statistics
import time
from pathlib import Path

import httpx

from tests.e2e.models.deepseek_v4_flash.config import (
    DSV4_COMPLETION_MAX_TOKENS,
    DSV4_CONCURRENT_REQUESTS,
    DSV4_PROMPT_FIRST_OPERAND,
    DSV4_PROMPT_SECOND_OPERAND,
    DSV4_REQUEST_TIMEOUT_S,
)

PREFILL_MAX_TOKENS = 1
FIRST_TOKEN_LOGPROB_ATOL = 0.05


def evaluate_completions(
    *,
    url: str,
    model: str,
    output_path: Path,
    check_answer: bool = True,
    prefill_only: bool = False,
    reference_path: Path | None = None,
) -> None:
    """Run the ten concurrent requests and check every response.

    ``check_answer`` requires the content to be exactly the expected sum and the
    generation to finish with ``stop``, which is what the asynchronous case
    answers. A profile whose host does not return reliable answers yet passes
    ``check_answer=False``: the ten requests must still be served together and
    each must return a nonempty answer that finished, which is the plumbing this
    smoke case exists to cover. The exact check stays on for every host that has
    it.
    """
    results: list[dict] = [
        {
            "index": index,
            "prompt": (
                f"Compute {DSV4_PROMPT_FIRST_OPERAND + index} + "
                f"{DSV4_PROMPT_SECOND_OPERAND}. Reply with just the number."
            ),
            "error": "Request cancelled before completion",
        }
        for index in range(DSV4_CONCURRENT_REQUESTS)
    ]

    reference_cases: dict[str, dict] = {}
    if prefill_only:
        if reference_path is None:
            raise ValueError("prefill accuracy requires a native reference file")
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        if reference["quantization"] != "W4A8-per-channel":
            raise ValueError(
                "prefill reference must use the same per-channel W4A8 math"
            )
        if not reference["model_revision"]:
            raise ValueError("prefill reference must identify its checkpoint revision")
        reference_cases = {case["prompt"]: case for case in reference["cases"]}
        if set(reference_cases) != {item["prompt"] for item in results}:
            raise ValueError(
                "prefill reference must cover exactly these request prompts"
            )

    async def request(client: httpx.AsyncClient, item: dict) -> None:
        item["started_at"] = time.monotonic()
        try:
            response = await client.post(
                url,
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": item["prompt"]}],
                    "temperature": 0.0,
                    "max_tokens": PREFILL_MAX_TOKENS
                    if prefill_only
                    else DSV4_COMPLETION_MAX_TOKENS,
                    **({"logprobs": True, "top_logprobs": 5} if prefill_only else {}),
                    "chat_template_kwargs": {"thinking": False},
                },
            )
            item["status_code"] = response.status_code
            # Keep even non-JSON error bodies for diagnosing a failed server.
            item["response_body"] = response.text
            response.raise_for_status()
            item["response"] = response.json()
            if prefill_only:
                validate_prefill_response(
                    item["response"], reference_cases[item["prompt"]]
                )
            else:
                validate_response(item["response"], check_answer=check_answer)
            expected = (
                DSV4_PROMPT_FIRST_OPERAND + item["index"] + DSV4_PROMPT_SECOND_OPERAND
            )
            content = item["response"]["choices"][0]["message"]["content"].strip()
            if check_answer and not prefill_only and content != str(expected):
                raise RuntimeError(
                    f"wrong answer: expected {expected}, got {content!r}"
                )
            del item["error"]
        except (httpx.HTTPError, ValueError, RuntimeError, KeyError, TypeError) as exc:
            item["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            item["finished_at"] = time.monotonic()
            item["latency_seconds"] = item["finished_at"] - item["started_at"]

    async def run_requests() -> None:
        # All requests are scheduled before yielding. Async socket operations
        # cancel immediately, so SIGTERM can reach the runner's service cleanup
        # without waiting for a thread blocked in a 300-second HTTP read.
        async with httpx.AsyncClient(timeout=DSV4_REQUEST_TIMEOUT_S) as client:
            await asyncio.gather(*(request(client, item) for item in results))

    try:
        asyncio.run(run_requests())
    finally:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))

    errors = [
        f"Request {item['index']}: {item['error']}"
        for item in results
        if "error" in item
    ]
    if errors:
        raise RuntimeError("Concurrent completions failed:\n" + "\n".join(errors))
    # Scheduling together is insufficient: retain timings and require every
    # request to start before the first one finishes.
    if max(item["started_at"] for item in results) >= min(
        item["finished_at"] for item in results
    ):
        raise RuntimeError("The ten completion requests did not overlap")
    for item in results:
        content = item["response"]["choices"][0]["message"]["content"]
        print(f"Request {item['index'] + 1}: {item['prompt']} -> {content}")
    print(f"Concurrent completions: {len(results)}/{DSV4_CONCURRENT_REQUESTS} passed")
    if prefill_only:
        latencies = [item["latency_seconds"] for item in results]
        print(
            "First-token HTTP latency seconds: "
            f"median={statistics.median(latencies):.6f} "
            f"min={min(latencies):.6f} max={max(latencies):.6f}"
        )


def validate_response(result: dict, *, check_answer: bool = True) -> None:
    choices = result.get("choices") if isinstance(result, dict) else None
    if not isinstance(choices, list) or len(choices) != 1:
        raise RuntimeError("must return one choice")
    choice = choices[0]
    if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
        raise RuntimeError("returned an invalid message")
    content = choice["message"].get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("returned empty content")
    finish_reason = choice.get("finish_reason")
    if not check_answer:
        # A profile that does not check the answer still requires the request to
        # have finished; anything without a terminal reason did not.
        if not isinstance(finish_reason, str) or not finish_reason:
            raise RuntimeError("did not finish normally")
        return
    if finish_reason != "stop":
        raise RuntimeError("did not finish normally")


def validate_prefill_response(result: dict, reference: dict) -> None:
    """Check exactly one generated token and finite same-checkpoint logprob."""
    validate_response(result, check_answer=False)
    if result["usage"]["completion_tokens"] != PREFILL_MAX_TOKENS:
        raise RuntimeError("prefill-only request must generate exactly one token")
    choice = result["choices"][0]
    if choice["finish_reason"] not in ("length", "stop"):
        raise RuntimeError("prefill-only request did not finish normally")
    token_scores = choice["logprobs"]["content"]
    if len(token_scores) != 1:
        raise RuntimeError("prefill-only response must contain one token logprob")
    actual = token_scores[0]
    expected_score, actual_score = float(reference["logprob"]), float(actual["logprob"])
    if not math.isfinite(expected_score) or not math.isfinite(actual_score):
        raise RuntimeError("non-finite first-token logprob")
    if actual["token"] != reference["token"]:
        raise RuntimeError("first-token identity differs from native reference")
    if abs(actual_score - expected_score) > FIRST_TOKEN_LOGPROB_ATOL:
        raise RuntimeError("first-token logprob exceeds native-reference tolerance")
