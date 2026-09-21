"""The generation provider interface, the scripted fake, and the Gemini
adapter's structural behavior — no network, no credential."""

import os

import pytest

from app.providers.fake_llm import FAKE_LLM_MODEL_NAME, FakeLLMProvider
from app.providers.gemini_llm import GeminiLLMProvider
from app.providers.llm import LLMError, LLMProvider, LLMResult


def test_llm_provider_is_a_protocol_the_fake_satisfies() -> None:
    assert isinstance(FakeLLMProvider(), LLMProvider)


def test_llm_provider_is_a_protocol_the_gemini_adapter_satisfies() -> None:
    assert isinstance(GeminiLLMProvider("some-verified-model"), LLMProvider)


# --- the scripted fake ------------------------------------------------


def test_the_fake_returns_scripted_strings_in_order() -> None:
    fake = FakeLLMProvider(["first", "second"])

    a = fake.complete(system="sys", user="u1", max_tokens=100)
    b = fake.complete(system="sys", user="u2", max_tokens=100)

    assert a.text == "first"
    assert b.text == "second"


def test_the_fake_derives_token_counts_from_word_counts_not_a_real_tokenizer() -> None:
    fake = FakeLLMProvider(["one two three"])

    result = fake.complete(system="a b", user="c d e", max_tokens=100)

    assert result.output_tokens == 3
    assert result.input_tokens == 5  # "a b" (2) + "c d e" (3)


def test_the_fake_can_return_a_full_llm_result_directly() -> None:
    fake = FakeLLMProvider([LLMResult(text="exact", input_tokens=7, output_tokens=9)])

    result = fake.complete(system="s", user="u", max_tokens=100)

    assert result == LLMResult(text="exact", input_tokens=7, output_tokens=9)


def test_the_fake_can_raise_a_scripted_llm_error() -> None:
    fake = FakeLLMProvider([LLMError("scripted provider failure")])

    with pytest.raises(LLMError, match="scripted provider failure"):
        fake.complete(system="s", user="u", max_tokens=100)


def test_the_fake_raises_when_it_runs_out_of_scripted_responses() -> None:
    fake = FakeLLMProvider(["only one"])
    fake.complete(system="s", user="u", max_tokens=100)

    with pytest.raises(LLMError, match="no scripted response"):
        fake.complete(system="s", user="u", max_tokens=100)


def test_the_fake_tracks_how_many_calls_were_made() -> None:
    fake = FakeLLMProvider(["a", "b"])
    assert fake.call_count == 0

    fake.complete(system="s", user="u", max_tokens=100)
    assert fake.call_count == 1


def test_the_fake_has_its_own_model_name_by_default() -> None:
    assert FakeLLMProvider().model_name == FAKE_LLM_MODEL_NAME


def test_the_fake_model_name_is_overridable() -> None:
    assert FakeLLMProvider(model_name="custom").model_name == "custom"


def test_nothing_in_the_application_selects_the_fake() -> None:
    """The same static guard `test_embeddings.py` applies to
    `FakeEmbeddingProvider` and `test_reranking.py` applies to
    `FakeRerankProvider` — nothing under `app/` may select the fake. Only a
    test may choose it."""
    import ast
    import pathlib

    app_dir = pathlib.Path(__file__).resolve().parent.parent / "app"
    for module in app_dir.rglob("*.py"):
        if module.name == "fake_llm.py" or module.parent.name == "providers":
            continue
        tree = ast.parse(module.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                assert node.id != "FakeLLMProvider", module.name


# --- the Gemini adapter: structural behavior, no network ---------------


def test_gemini_provider_construction_never_fails_even_with_no_model() -> None:
    """Construction must never raise -- it happens while FastAPI resolves
    the `llm_provider` dependency, before `POST /query`'s own error
    mapping runs at all. See `GeminiLLMProvider`'s own docstring."""
    GeminiLLMProvider(None)  # must not raise
    GeminiLLMProvider("")  # must not raise


def test_gemini_provider_refuses_to_complete_with_no_model_configured() -> None:
    provider = GeminiLLMProvider(None)
    with pytest.raises(LLMError, match="no Gemini model is configured"):
        provider.complete(system="s", user="u", max_tokens=100)


def test_gemini_provider_refuses_to_complete_with_an_empty_model() -> None:
    provider = GeminiLLMProvider("")
    with pytest.raises(LLMError, match="no Gemini model is configured"):
        provider.complete(system="s", user="u", max_tokens=100)


def test_gemini_provider_never_chose_a_model_id_itself() -> None:
    """The model name is exactly what was passed in — never a default the
    adapter picked from memory."""
    provider = GeminiLLMProvider("models/whatever-was-verified")
    assert provider.model_name == "models/whatever-was-verified"


def test_gemini_provider_error_message_never_contains_a_real_looking_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no credential in the environment, constructing the real client
    fails — and the wrapped error never repeats a key, per
    `app/providers/gemini_llm.py`'s own discipline."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    provider = GeminiLLMProvider("some-model")
    with pytest.raises(LLMError) as exc_info:
        provider.complete(system="s", user="u", max_tokens=100)

    message = str(exc_info.value)
    assert "sk-" not in message
    assert "AIza" not in message
    assert "key" not in message.lower() or "credential" in message.lower()


@pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"),
    reason=(
        "no GEMINI_API_KEY/GOOGLE_API_KEY in this environment; the real "
        "Gemini model has NOT been exercised. Set "
        "KNOWLEDGEOS_RUN_GENERATION_SMOKE_TEST=1 and a real key to run it."
    ),
)
@pytest.mark.skipif(
    os.environ.get("KNOWLEDGEOS_RUN_GENERATION_SMOKE_TEST") != "1",
    reason="opt-in only: set KNOWLEDGEOS_RUN_GENERATION_SMOKE_TEST=1",
)
def test_the_real_gemini_model_can_generate() -> None:
    """The only test in this suite that touches the network or a real
    model. Everything else uses the fake, and this one skipping is
    reported rather than hidden. Requires `KNOWLEDGEOS_LLM_MODEL` to be
    set to a model already verified against the live API — this test does
    not verify one itself and does not invent one."""
    from app.config import get_settings

    model = get_settings().llm_model
    if not model:
        pytest.skip("KNOWLEDGEOS_LLM_MODEL is not set; no model to exercise")

    provider = GeminiLLMProvider(model)
    result = provider.complete(
        system="Respond with the single word: OK",
        user="Respond now.",
        max_tokens=16,
    )
    assert result.text
    assert result.input_tokens > 0
    assert result.output_tokens > 0
