"""Configuration: the defaults, the overrides, and what is deliberately absent.

The last test is the one that will matter later. A fresh clone of this project
must be runnable with no credential anywhere — the specification requires the
whole suite to pass with `ANTHROPIC_API_KEY` unset — and the cheapest way to
keep that true is to assert it from milestone 1, before there is a key to
forget about.
"""

import pytest

from app.config import DEFAULT_DATABASE_URL, Settings, get_settings


def test_the_defaults_are_what_milestone_one_says() -> None:
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
