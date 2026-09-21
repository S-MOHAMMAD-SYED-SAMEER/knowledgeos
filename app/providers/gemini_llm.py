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
"""

import logging

from app.providers.llm import LLMError, LLMResult

logger = logging.getLogger(__name__)


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
    """

    def __init__(self, model_name: str | None) -> None:
        self._model_name = model_name
        self._client = None

    @property
    def model_name(self) -> str | None:
        return self._model_name

    def complete(self, *, system: str, user: str, max_tokens: int) -> LLMResult:
        if not self._model_name:
            raise LLMError(
                "no Gemini model is configured (KNOWLEDGEOS_LLM_MODEL is "
                "unset); verify an available model against the live API "
                "(client.models.list()) before setting one — never guess"
            )

        client = self._load()
        try:
            from google.genai import types

            response = client.models.generate_content(
                model=self._model_name,
                contents=user,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    max_output_tokens=max_tokens,
                ),
            )
        except Exception as exc:  # noqa: BLE001 - any transport/API failure is one case
            raise LLMError(f"generation failed ({type(exc).__name__})") from exc

        text = response.text
        if text is None:
            raise LLMError("the model returned no text content")

        usage = response.usage_metadata
        input_tokens = int(getattr(usage, "prompt_token_count", None) or 0)
        output_tokens = int(getattr(usage, "candidates_token_count", None) or 0)

        return LLMResult(text=text, input_tokens=input_tokens, output_tokens=output_tokens)

    def _load(self):
        """Build the client once, resolving credentials from the process
        environment exactly as the SDK does — this module never reads the
        key itself, and never repeats it.
        """
        if self._client is not None:
            return self._client

        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise LLMError("google-genai is not installed") from exc

        try:
            logger.info("Constructing the Gemini client for model %s.", self._model_name)
            self._client = genai.Client()
        except ValueError as exc:
            # The SDK's own "no API key was provided" construction-time
            # failure. Never the SDK's message verbatim.
            raise LLMError("no Gemini credential is configured") from exc
        except Exception as exc:  # noqa: BLE001
            raise LLMError(
                f"the Gemini client could not be constructed ({type(exc).__name__})"
            ) from exc

        return self._client


__all__ = ["GeminiLLMProvider"]
