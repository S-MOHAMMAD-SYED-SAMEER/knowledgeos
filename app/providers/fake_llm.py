"""A scripted generation provider, for tests.

Unlike `FakeEmbeddingProvider` and `FakeRerankProvider`, this fake is not
hash-derived: those two must produce *plausible* numbers, but the
specification requires the generation fake to produce *specific* failure
shapes — "scripted fake returning fixture responses, including malformed
and invalid-citation responses so the validation layer is tested against
the failure it exists for." A hash cannot script that; a list can.

Construct one with a list of responses (each a raw text string, a full
`LLMResult`, or an `LLMError` instance to raise instead of returning), and
each call to `complete()` returns or raises the next one in order. Running
out of scripted responses is itself a test bug — it raises `LLMError`
rather than silently returning something plausible.

Nothing in the application ever selects this provider. Only a test does.
"""

from app.providers.llm import LLMError, LLMResult

FAKE_LLM_MODEL_NAME = "fake-scripted-llm"

ScriptedItem = str | LLMResult | LLMError


class FakeLLMProvider:
    """Returns (or raises) exactly what it was told to, in order."""

    def __init__(
        self,
        responses: list[ScriptedItem] | None = None,
        *,
        model_name: str = FAKE_LLM_MODEL_NAME,
    ) -> None:
        self._responses: list[ScriptedItem] = list(responses or [])
        self._index = 0
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    def complete(self, *, system: str, user: str, max_tokens: int) -> LLMResult:
        if self._index >= len(self._responses):
            raise LLMError(
                f"the fake has no scripted response left for call {self._index + 1}"
            )
        item = self._responses[self._index]
        self._index += 1

        if isinstance(item, LLMError):
            raise item
        if isinstance(item, LLMResult):
            return item
        # A bare string: wrap it with token counts derived from word counts,
        # never from the real tokenizer this fake deliberately has none of.
        return LLMResult(
            text=item,
            input_tokens=len(user.split()) + len(system.split()),
            output_tokens=len(item.split()),
        )

    @property
    def call_count(self) -> int:
        """How many scripted responses have been consumed. For tests that
        need to prove the provider was — or was not — called."""
        return self._index


__all__ = ["FAKE_LLM_MODEL_NAME", "FakeLLMProvider", "ScriptedItem"]
