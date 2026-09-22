#!/usr/bin/env python3
"""Controlled configurable-concurrency decode benchmark for ExLlamaV3 A/B tests.

The server-side log remains the source of truth for decode tok/s. This script
keeps prompt, cache warmup, concurrency, sampling, and output length fixed.

Environment:
  OPENAI_BASE_URL   default http://127.0.0.1:8084/v1
  MODEL             default qwen3.8-flash-next-layer
  PROMPT_REPEAT     default 1000 (~40k tokens on the validation tokenizer)
  MAX_TOKENS        default 900
  CONCURRENCY       default 3
"""

import concurrent.futures
import json
import os
import time
import urllib.request

BASE = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8084/v1").rstrip("/")
URL = BASE + "/chat/completions"
MODEL = os.environ.get("MODEL", "qwen3.8-flash-next-layer")
PROMPT_REPEAT = int(os.environ.get("PROMPT_REPEAT", "1000"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "900"))
CONCURRENCY = int(os.environ.get("CONCURRENCY", "3"))

unit = (
    "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi "
    "omicron pi rho sigma tau upsilon phi chi psi omega 0123456789. "
)
corpus = unit * PROMPT_REPEAT

messages = [
    {
        "role": "system",
        "content": (
            "You are running a deterministic throughput benchmark. "
            "Follow the requested output format exactly."
        ),
    },
    {
        "role": "user",
        "content": (
            corpus
            + "\n\nTask: Output the integers 1 through 2000, one integer per line, "
            "with no prose or explanation. Continue until stopped by the token limit."
        ),
    },
]


def call(max_tokens: int, tag: str) -> dict:
    body = json.dumps(
        {
            "model": MODEL,
            "messages": messages,
            "temperature": 0,
            "top_p": 1,
            "max_tokens": max_tokens,
            "stream": False,
        }
    ).encode()

    req = urllib.request.Request(
        URL,
        data=body,
        headers={"Content-Type": "application/json"},
    )

    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=600) as response:
            raw = response.read()
        parsed = json.loads(raw)
        return {
            "tag": tag,
            "ok": True,
            "elapsed": time.time() - start,
            "finish_reason": parsed.get("choices", [{}])[0].get("finish_reason"),
            "usage": parsed.get("usage"),
        }
    except Exception as exc:
        return {
            "tag": tag,
            "ok": False,
            "elapsed": time.time() - start,
            "error": repr(exc),
        }


def main() -> None:
    warm = call(8, "warm")
    print("WARM", json.dumps(warm, ensure_ascii=False), flush=True)
    if not warm["ok"]:
        raise SystemExit(2)

    time.sleep(1)

    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        futures = [
            pool.submit(call, MAX_TOKENS, f"c{CONCURRENCY}-{index}")
            for index in range(1, CONCURRENCY + 1)
        ]
        for future in concurrent.futures.as_completed(futures):
            print(
                "RESULT",
                json.dumps(future.result(), ensure_ascii=False),
                flush=True,
            )


if __name__ == "__main__":
    main()
