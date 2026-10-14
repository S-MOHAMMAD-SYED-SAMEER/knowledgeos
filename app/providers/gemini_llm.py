"""The real generation provider, run against Google Gemini.

**Provider deviation, explained in full.** The specification's stack list
(§3) and milestone 8 scope line (§17) name "Anthropic SDK" / "Anthropic
generation". This project uses Google Gemini instead — an explicit,
authorized, documented deviation from that *wording*. §9 — the section that
actually specifies what generation must do (versioned prompt, structured
output, citation validation, grounding, abstention) — names no vendor
anywhere. Nothing in `app/generation/` or in the `LLMProvider` interface
(`app/providers/llm.py`) is Anthropic- or Gemini-shaped; either vendor's SDK
could sit behind this one file, and only this file. An Anthropic adapter is
not implemented — deferred, the same way §1 defers a Voyage embeddings
adapter without building one. The README's milestone 8 section documents
this decision plainly, including the SPEC sections it deviates from.

Same two deliberate properties as `app/providers/bge.py` and
`app/providers/cross_encoder.py`, for the same reasons.

**The import is lazy.** `google.genai` is only imported inside `_load()`,
never at module load, so importing this module — or collecting the test
suite — never requires the package to be installed, let alone reach the
network.

**The client is built once and kept.** Cheap to build here (unlike loading
a local model), but the pattern stays consistent with every other provider
in this package.

**The API key is never read by this module directly.** `google.genai.Client()`
resolves `GEMINI_API_KEY` (or `GOOGLE_API_KEY`, which takes precedence) from
the process environment itself — this file never calls `os.environ` for it,
and the key never appears in `Settings`, a log line, or an error message. A
missing key raises inside the SDK's own client construction; this module
catches that specific failure and wraps it in `LLMError` without repeating
the SDK's message, in case a future SDK version ever echoes an attempted
value into it.

**The model ID is never chosen here from memory.** It is read from
`Settings.llm_model`, which has no default (`app/config.py`) — the model
must be supplied by whoever has actually verified it against the live
`client.models.list()` listing, because no Gemini credential was available
in the environment this adapter was written in to verify one.

**What happens when a call goes wrong** (each case is a distinct, tested
outcome; none leaks the prompt, the evidence, a key or the SDK's own message):

- *A call that takes too long* is cut off after `timeout_seconds` and, like
  any other transient failure, retried.
- *A transient failure* — a timeout, a connection error, an HTTP 429 or a 5xx —
  is retried up to `max_retries` times, waiting `Retry-After` when the service
  says how long and an exponential back-off with jitter when it does not.
- *A wait that would be too long* (the service asks for more than
  `retry_max_wait_seconds`, which is what a spent daily quota looks like) is not
  waited out: `LLMRateLimited` is raised at once, carrying the delay, so the
  caller can tell the person to come back later.
- *A request the service rejects outright* (400, 401, 403, 404: a wrong model
  ID, a bad key) or any other failure that is not transient is never retried;
  the message carries the HTTP status.
- *A reply the model refused to give* (a blocked prompt, or a finish reason
  such as `SAFETY` with no text) is `LLMBlocked`, not an outage.
- *A reply cut off at the token limit* is returned with
  `finish_reason="MAX_TOKENS"`, so the generator can say why the JSON is
  incomplete instead of reporting a mystery.

**Tokens.** Thinking models spend output tokens on reasoning before they write
the answer, and bill them as output. `thoughts_token_count` is therefore added
to `output_tokens`, so the cost the application computes is the cost incurred.
"""

import logging
import random
import re
import time
from collections.abc import Callable

from app.providers.llm import (
    LLMBlocked,
    LLMError,
    LLMRateLimited,
    LLMResult,
    LLMTimeout,
)

logger = logging.getLogger(__name__)

# Finish reasons that mean the model produced what it was going to produce.
_NORMAL_FINISH_REASONS = frozenset({None, "STOP", "MAX_TOKENS", "FINISH_REASON_UNSPECIFIED"})

_RETRY_DELAY_PATTERN = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*s\s*$")


class GeminiLLMProvider:
    """Google Gemini, through the `google-genai` SDK.

    Construction never fails — the same lazy-failure discipline
    `BgeEmbeddingProvider` and `CrossEncoderRerankProvider` follow, and for
    the same structural reason here: this provider is built while FastAPI
    resolves the `llm_provider` dependency, *before* `POST /query`'s own
    try/except around `complete()` ever runs. A provider that raised in
    `__init__` would raise there instead, outside every error-mapping this
    application has — an unhandled 500 with a traceback, not the honest
    503 the specification's error-response rule requires. A missing model
    is therefore checked at `complete()` time, exactly like a missing
    credential or an unreachable model is.

    `client`, `sleep` and `jitter` exist so a test can script the service's
    behaviour (errors, delays) with no network and no waiting.
    """

    def __init__(
        self,
        model_name: str | None,
        *,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        retry_max_wait_seconds: float = 15.0,
        temperature: float | None = None,
        client=None,
        sleep: Callable[[float], None] = time.sleep,
        jitter: Callable[[], float] = random.random,
    ) -> None:
        self._model_name = model_name
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._retry_max_wait_seconds = retry_max_wait_seconds
        self._temperature = temperature
        self._client = client
        self._sleep = sleep
        self._jitter = jitter

    @property
    def model_name(self) -> str | None:
        return self._model_name

    def complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int,
        json_schema: dict | None = None,
    ) -> LLMResult:
        if not self._model_name:
            raise LLMError(
                "no Gemini model is configured (KNOWLEDGEOS_LLM_MODEL is "
                "unset); verify an available model against the live API "
                "(client.models.list()) before setting one — never guess"
            )

        client = self._load()
        config = self._config(system=system, max_tokens=max_tokens, json_schema=json_schema)

        attempt = 0
        while True:
            try:
                response = client.models.generate_content(
                    model=self._model_name, contents=user, config=config
                )
                break
            except Exception as exc:  # noqa: BLE001 - classified just below
                retry_after = self._classify(exc)  # raises for the cases not worth retrying
                if attempt >= self._max_retries:
                    raise self._final_error(exc, retry_after) from exc
                wait = self._wait_for(attempt, retry_after)
                logger.warning(
                    "Gemini call failed (%s); retrying in %.1f s (retry %d of %d).",
                    _describe(exc),
                    wait,
                    attempt + 1,
                    self._max_retries,
                )
                self._sleep(wait)
                attempt += 1

        return self._to_result(response)

    # --- request -------------------------------------------------------

    def _config(self, *, system: str, max_tokens: int, json_schema: dict | None):
        from google.genai import types

        settings: dict = {"system_instruction": system, "max_output_tokens": max_tokens}
        if self._temperature is not None:
            settings["temperature"] = self._temperature
        if json_schema is not None:
            # JSON mode: the service itself is asked for a single JSON document that
            # matches the schema. What comes back is still parsed and validated in
            # Python — this makes malformed output rarer, not impossible.
            settings["response_mime_type"] = "application/json"
            settings["response_json_schema"] = json_schema
        return types.GenerateContentConfig(**settings)

    # --- failures ------------------------------------------------------

    def _classify(self, exc: Exception) -> float | None:
        """Return the service's requested delay (or None) for a failure worth
        retrying; raise for one that is not."""
        if isinstance(exc, LLMError):
            raise exc

        status = getattr(exc, "code", None)
        transient = (
            (isinstance(status, int) and (status == 429 or status >= 500))
            or _is_timeout(exc)
            or _is_connection_error(exc)
        )
        if not transient:
            # A rejected request (400, 401, 403, 404 ...) or a bug: retrying repeats it.
            raise LLMError(f"generation failed ({_describe(exc)})") from exc

        retry_after = _retry_after_seconds(exc)
        if status == 429 and retry_after is not None and retry_after > self._retry_max_wait_seconds:
            raise LLMRateLimited(
                f"the model is rate limited; the service asks for {retry_after:.0f} s",
                retry_after=retry_after,
            ) from exc
        return retry_after

    def _final_error(self, exc: Exception, retry_after: float | None) -> LLMError:
        status = getattr(exc, "code", None)
        if status == 429:
            return LLMRateLimited(
                "the model is rate limited (HTTP 429)", retry_after=retry_after
            )
        if _is_timeout(exc):
            return LLMTimeout(f"the model did not answer within {self._timeout_seconds:g} s")
        return LLMError(f"generation failed ({_describe(exc)})")

    def _wait_for(self, attempt: int, retry_after: float | None) -> float:
        if retry_after is not None:
            return retry_after
        # 1 s, 2 s, 4 s ... with up to half a second of jitter, never past the cap.
        backoff = 1.0 * (2**attempt) + 0.5 * self._jitter()
        return min(backoff, self._retry_max_wait_seconds)

    # --- response ------------------------------------------------------

    def _to_result(self, response) -> LLMResult:
        feedback = getattr(response, "prompt_feedback", None)
        block_reason = getattr(feedback, "block_reason", None) if feedback else None
        if block_reason:
            raise LLMBlocked(f"the prompt was blocked ({_name(block_reason)})", reason=_name(block_reason))

        candidates = getattr(response, "candidates", None) or []
        finish_reason = _name(getattr(candidates[0], "finish_reason", None)) if candidates else None

        text = response.text
        if not text:
            if finish_reason not in _NORMAL_FINISH_REASONS:
                raise LLMBlocked(
                    f"the model gave no answer (finish reason {finish_reason})", reason=finish_reason
                )
            raise LLMError("the model returned no text content")

        usage = response.usage_metadata
        input_tokens = int(getattr(usage, "prompt_token_count", None) or 0)
        output_tokens = int(getattr(usage, "candidates_token_count", None) or 0) + int(
            getattr(usage, "thoughts_token_count", None) or 0
        )
        return LLMResult(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            finish_reason=finish_reason,
        )

    def _load(self):
        """Build the client once, resolving credentials from the process
        environment exactly as the SDK does — this module never reads the
        key itself, and never repeats it.
        """
        if self._client is not None:
            return self._client

        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise LLMError("google-genai is not installed") from exc

        try:
            logger.info("Constructing the Gemini client for model %s.", self._model_name)
            self._client = genai.Client(
                http_options=types.HttpOptions(timeout=int(self._timeout_seconds * 1000))
            )
        except ValueError as exc:
            # The SDK's own "no API key was provided" construction-time
            # failure. Never the SDK's message verbatim.
            raise LLMError("no Gemini credential is configured") from exc
        except Exception as exc:  # noqa: BLE001
            raise LLMError(
                f"the Gemini client could not be constructed ({type(exc).__name__})"
            ) from exc

        return self._client


# --- helpers ---------------------------------------------------------------


def _name(value) -> str | None:
    """An enum or a string, as its plain name ("STOP", "SAFETY"), or None."""
    if value is None:
        return None
    return getattr(value, "name", None) or str(value)


def _is_timeout(exc: Exception) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    try:
        import httpx
    except ImportError:  # pragma: no cover - google-genai depends on it
        return False
    return isinstance(exc, httpx.TimeoutException)


def _is_connection_error(exc: Exception) -> bool:
    if isinstance(exc, ConnectionError):
        return True
    try:
        import httpx
    except ImportError:  # pragma: no cover - google-genai depends on it
        return False
    return isinstance(exc, httpx.TransportError)


def _describe(exc: Exception) -> str:
    status = getattr(exc, "code", None)
    if isinstance(status, int):
        return f"HTTP {status}"
    return "timeout" if _is_timeout(exc) else type(exc).__name__


def _retry_after_seconds(exc: Exception) -> float | None:
    """How long the service asked us to wait, from the `Retry-After` header or
    the `RetryInfo` detail Google puts in a 429 body, or None."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is not None:
        try:
            raw = headers.get("retry-after")
        except Exception:  # noqa: BLE001
            raw = None
        if raw is not None:
            try:
                return max(0.0, float(raw))
            except (TypeError, ValueError):
                pass

    details = getattr(exc, "details", None)
    if isinstance(details, dict):
        for item in (details.get("error") or {}).get("details") or []:
            if isinstance(item, dict) and str(item.get("@type", "")).endswith("RetryInfo"):
                match = _RETRY_DELAY_PATTERN.match(str(item.get("retryDelay", "")))
                if match:
                    return float(match.group(1))
    return None


__all__ = ["GeminiLLMProvider"]
