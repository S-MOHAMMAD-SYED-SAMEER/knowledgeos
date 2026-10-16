"""One real generation call, to check that a Gemini model answers, and what it costs.

    python scripts/gemini_smoke.py [--model ID] [--temperature T] [--record]

Makes exactly ONE request, with no retries, so the number of requests used is known.
Needs GEMINI_API_KEY (or GOOGLE_API_KEY) in the environment and a model ID, from --model or
KNOWLEDGEOS_LLM_MODEL. Prints the model, the temperature (or "service default"), tokens, finish
reason and latency; never the key. With --record it saves the same facts to docs/results/.
Exits 1 on any failure and 3 on a rate limit, so a caller can stop instead of looping.
"""

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from app.config import get_settings
from app.providers.gemini_llm import GeminiLLMProvider
from app.providers.llm import LLMError, LLMRateLimited

RESULTS_DIR = Path(__file__).resolve().parent.parent / "docs" / "results"
SYSTEM = "You are a connectivity check. Answer in one short sentence."
USER = "Reply with the words: smoke test ok."


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=None, help="model ID (default: KNOWLEDGEOS_LLM_MODEL)")
    parser.add_argument("--temperature", type=float, default=None, help="default: the service's own")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--record", action="store_true", help="save the result under docs/results/")
    args = parser.parse_args(argv)

    settings = get_settings()
    model = args.model or settings.llm_model
    if not model:
        print("no model: pass --model or set KNOWLEDGEOS_LLM_MODEL", file=sys.stderr)
        return 1
    temperature = args.temperature if args.temperature is not None else settings.llm_temperature

    print(f"Expected requests: 1 generation request, no retries. Model: {model}. "
          f"Temperature: {temperature if temperature is not None else 'service default (unset)'}.", flush=True)
    llm = GeminiLLMProvider(model, timeout_seconds=settings.llm_timeout_seconds, max_retries=0,
                            temperature=temperature)
    started = time.monotonic()
    try:
        result = llm.complete(system=SYSTEM, user=USER, max_tokens=args.max_tokens)
    except LLMRateLimited as exc:
        print(f"STOP: rate limited ({exc}); retry_after={exc.retry_after}", file=sys.stderr)
        return 3
    except LLMError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    latency = time.monotonic() - started

    record = {
        "kind": "generation-smoke",
        "date": datetime.now(UTC).strftime("%Y-%m-%d"),
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "provider": "gemini",
        "model": model,
        "temperature": temperature,
        "temperature_note": None if temperature is not None else "unset: the model's own default",
        "max_output_tokens": args.max_tokens,
        "requests": 1,
        "retries_allowed": 0,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "finish_reason": result.finish_reason,
        "latency_seconds": round(latency, 2),
        "reply": result.text,
    }
    print(json.dumps(record, indent=2))
    if args.record:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        path = RESULTS_DIR / f"{record['date']}-smoke-{model}.json"
        path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(f"saved {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
