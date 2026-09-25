"""`Settings.demo_mode` and the one place it is decided,
`app/api/query.py::llm_provider` — the M2 config seam.

No database needed: everything here is either a `Settings()` construction
or a direct call to the `llm_provider` dependency function.
"""

import pytest

from app.api.query import llm_provider
from app.config import Settings, get_settings
from app.providers.demo_llm import DemoLLMProvider
from app.providers.gemini_llm import GeminiLLMProvider


def test_demo_mode_defaults_to_false() -> None:
    assert Settings().demo_mode is False


def test_demo_mode_reads_from_the_documented_env_var(settings_env, monkeypatch) -> None:
    monkeypatch.setenv("KNOWLEDGEOS_DEMO_MODE", "true")
    get_settings.cache_clear()

    assert get_settings().demo_mode is True


def test_llm_provider_resolves_to_the_demo_provider_when_demo_mode_is_on(
    settings_env, monkeypatch
) -> None:
    monkeypatch.setenv("KNOWLEDGEOS_DEMO_MODE", "true")
    get_settings.cache_clear()

    provider = llm_provider()

    assert isinstance(provider, DemoLLMProvider)


def test_llm_provider_resolves_to_the_real_gemini_adapter_when_demo_mode_is_off(
    settings_env, monkeypatch
) -> None:
    """Normal-mode regression check: demo mode off must leave the
    pre-M2 behaviour exactly as it was — the real, cached Gemini adapter,
    never the demo provider."""
    monkeypatch.setenv("KNOWLEDGEOS_DEMO_MODE", "false")
    get_settings.cache_clear()

    provider = llm_provider()

    assert isinstance(provider, GeminiLLMProvider)
    assert not isinstance(provider, DemoLLMProvider)


def test_gemini_adapter_is_never_constructed_when_demo_mode_is_on(
    settings_env, monkeypatch
) -> None:
    """Not merely "prefers the demo provider": the real accessor is never
    even called when demo mode is on, so a missing `GEMINI_API_KEY` can
    never surface in demo mode either."""
    import app.api.query as query_module

    def _fail_if_called() -> None:
        raise AssertionError(
            "get_llm_provider() must not be called when demo_mode is True"
        )

    monkeypatch.setattr(query_module, "get_llm_provider", _fail_if_called)
    monkeypatch.setenv("KNOWLEDGEOS_DEMO_MODE", "true")
    get_settings.cache_clear()

    provider = llm_provider()

    assert isinstance(provider, DemoLLMProvider)


@pytest.mark.parametrize("flag_value", ["false", "0"])
def test_demo_mode_off_variants_all_resolve_to_the_real_provider(
    settings_env, monkeypatch, flag_value
) -> None:
    monkeypatch.setenv("KNOWLEDGEOS_DEMO_MODE", flag_value)
    get_settings.cache_clear()

    assert isinstance(llm_provider(), GeminiLLMProvider)
