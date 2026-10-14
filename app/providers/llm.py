"""The generation boundary.

`LLMProvider.complete(...)` is deliberately provider-agnostic: nothing in
this interface, or in anything that calls it, names a vendor. The
specification's own §9 (generation, citations, grounding, abstention)
describes what the model must be instructed to do and what its output must
contain; it names no vendor anywhere in that section. The vendor commitment
lives in exactly one place — `app/providers/gemini_llm.py` — and everything
above this interface (generation orchestration, citation validation,
grounding, abstention) is written against this `Protocol` alone.

**Provider deviation, recorded here because this is the interface every
provider must satisfy.** The specification's stack list (§3) and its
milestone-8 scope line (§17) name "Anthropic generation" / "Anthropic SDK".
This project uses Google Gemini instead — an explicit, authorized deviation
from that *wording*, not from §9's *behavioural* requirements, none of which
mention a vendor. `app/providers/gemini_llm.py` carries the full
explanation and the README's milestone 8 section documents it plainly. An
Anthropic adapter is not implemented; it is deferred, the same way §1
defers a Voyage embeddings adapter without building it.

The provider returns **raw text plus token usage** — never a parsed answer.
Parsing and validating the model's structured output happens in
`app/generation/generator.py`, in Python, per the specification's own words
("Parse and validate in Python"). A provider that pre-parsed its own output
would make the specification's mandatory malformed-response test fixtures
untestable: there would be nothing left for the validation layer to catch
failing.
"""

from typing import NamedTuple, Protocol, runtime_checkable


class LLMError(RuntimeError):
    """Generation could not be completed.

    The message names what failed structurally — a missing credential, an
    unreachable model, a rate limit, a malformed transport response — and
    never the prompt, the evidence text, or the model's raw output.
    """


class LLMRateLimited(LLMError):
    """The service said to slow down (HTTP 429) and retrying did not clear it.

    `retry_after` is how long the service asked for, in seconds, when it said.
    A spent daily quota shows up as a long `retry_after`, which is not waited out.
    """

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class LLMTimeout(LLMError):
    """The model did not answer within the configured time, even after retries."""


class LLMBlocked(LLMError):
    """The model declined to answer: a blocked prompt, or a safety stop with no text.

    Not an outage, so it is not reported as one. `reason` is the service's own
    label ("SAFETY", "PROHIBITED_CONTENT", ...), never the prompt or the output.
    """

    def __init__(self, message: str, *, reason: str | None = None) -> None:
        super().__init__(message)
        self.reason = reason


class LLMResult(NamedTuple):
    """One generation call's raw output: text, plus what it cost.

    `text` is exactly what the model returned, unparsed. Turning it into
    `answer`/`citations`/`sufficient_evidence` is
    `app/generation/generator.py`'s job, not the provider's.
    """

    text: str
    input_tokens: int
    output_tokens: int  # includes any reasoning ("thinking") tokens, which are billed as output
    finish_reason: str | None = None  # "STOP", "MAX_TOKENS", ...; None when the provider has none


@runtime_checkable
class LLMProvider(Protocol):
    """Turns a system instruction and a user prompt into text."""

    @property
    def model_name(self) -> str:
        """Which model produced the text, for the record."""
        ...

    def complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int,
        json_schema: dict | None = None,
    ) -> LLMResult:
        """One generation call.

        `json_schema`, when given, asks the provider for a single JSON document
        of that shape. A provider that cannot do so ignores it; the caller
        validates what comes back either way.

        Raises `LLMError` (or a subclass: `LLMRateLimited`, `LLMTimeout`,
        `LLMBlocked`) rather than returning something the caller would have to
        guess about.
        """
        ...


__all__ = [
    "LLMBlocked",
    "LLMError",
    "LLMProvider",
    "LLMRateLimited",
    "LLMResult",
    "LLMTimeout",
]
