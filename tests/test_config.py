"""Configuration: the defaults, the overrides, and what is deliberately absent.

The last test is the one that will matter later. A fresh clone of this project
must be runnable with no credential anywhere — the specification requires the
whole suite to pass with `ANTHROPIC_API_KEY` unset — and the cheapest way to
keep that true is to assert it from milestone 1, before there is a key to
forget about.
"""

import pytest

from app.config import DEFAULT_DATABASE_URL, Settings, get_settings


def test_the_defaults_are_what_milestone_one_says(monkeypatch) -> None:
    monkeypatch.delenv("KNOWLEDGEOS_DATABASE_URL", raising=False)
    settings = Settings(_env_file=None)

    assert settings.app_name == "KnowledgeOS"
    assert settings.environment == "local"
    assert settings.debug is False
    assert settings.database_url == DEFAULT_DATABASE_URL


def test_the_environment_overrides_a_setting(monkeypatch) -> None:
    monkeypatch.setenv("KNOWLEDGEOS_ENVIRONMENT", "staging")

    assert Settings(_env_file=None).environment == "staging"


def test_the_database_url_comes_from_the_environment(monkeypatch) -> None:
    """The one setting a deployment always has to supply."""
    monkeypatch.setenv(
        "KNOWLEDGEOS_DATABASE_URL",
        "postgresql+psycopg://someone:secret@db.example:5432/knowledgeos",
    )

    assert "db.example" in Settings(_env_file=None).database_url


def test_a_value_that_is_not_a_boolean_is_refused(monkeypatch) -> None:
    monkeypatch.setenv("KNOWLEDGEOS_DEBUG", "maybe")

    with pytest.raises(ValueError):
        Settings(_env_file=None)


def test_settings_are_built_once(settings_env) -> None:
    assert get_settings() is get_settings()


def test_no_credential_is_needed_to_configure_this_application() -> None:
    """A fresh clone runs with nothing set. That must stay true.

    Credential-shaped names, not the bare word "token": milestone 3's
    `chunk_size_tokens` counts words in a chunk and is nobody's secret.
    """
    settings = Settings(_env_file=None)

    credentials = (
        "api_key",
        "apikey",
        "secret",
        "password",
        "auth_token",
        "access_token",
        "credential",
    )
    for field in Settings.model_fields:
        for shape in credentials:
            assert shape not in field.lower(), field
    assert settings.app_name


# --- the generation provider's settings -------------------------------------


def test_the_llm_resilience_defaults() -> None:
    settings = Settings(_env_file=None)

    assert settings.llm_max_output_tokens == 4096  # room for reasoning tokens as well as the answer
    assert settings.llm_timeout_seconds == 30
    assert settings.llm_max_retries == 2
    assert settings.llm_retry_max_wait_seconds == 15
    # Explicit, and the model's own default (1.0), which is what Google says to keep for Gemini 3.
    assert settings.llm_temperature == 1.0
    assert settings.llm_model == "gemini-3.5-flash-lite"


def test_the_llm_settings_can_be_overridden(monkeypatch) -> None:
    monkeypatch.setenv("KNOWLEDGEOS_LLM_TIMEOUT_SECONDS", "5.5")
    monkeypatch.setenv("KNOWLEDGEOS_LLM_MAX_RETRIES", "0")
    monkeypatch.setenv("KNOWLEDGEOS_LLM_TEMPERATURE", "0")

    settings = Settings(_env_file=None)

    assert settings.llm_timeout_seconds == 5.5
    assert settings.llm_max_retries == 0
    assert settings.llm_temperature == 0.0


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("KNOWLEDGEOS_LLM_TIMEOUT_SECONDS", "0"),
        ("KNOWLEDGEOS_LLM_MAX_RETRIES", "-1"),
        ("KNOWLEDGEOS_LLM_RETRY_MAX_WAIT_SECONDS", "0"),
        ("KNOWLEDGEOS_LLM_TEMPERATURE", "3"),
        ("KNOWLEDGEOS_LLM_TEMPERATURE", "-0.1"),
    ],
)
def test_nonsense_llm_settings_are_refused(monkeypatch, name: str, value: str) -> None:
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError):
        Settings(_env_file=None)
