"""The Gemini adapter's behaviour when the service misbehaves, with no network and no waiting.

A scripted client stands in for `google.genai.Client`: each call to `generate_content` returns
or raises the next item in a list. The errors are the SDK's own classes (`ClientError`,
`ServerError`) and httpx's own timeouts, so the adapter's classification is tested against the
real shapes it will meet. `sleep` is a recorder, so back-off is asserted, not waited for.
"""

from types import SimpleNamespace

import httpx
import pytest
from google.genai import errors

from app.providers.gemini_llm import GeminiLLMProvider
from app.providers.llm import LLMBlocked, LLMError, LLMRateLimited, LLMTimeout

SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}}}


class ScriptedClient:
    """`client.models.generate_content(...)`, returning or raising what it was told to."""

    def __init__(self, *script) -> None:
        self._script = list(script)
        self.calls: list[dict] = []
        self.models = self

    def generate_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        if not self._script:
            raise AssertionError("the adapter called the service more often than the test expected")
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def reply(text="{}", *, prompt=10, candidates=5, thoughts=0, finish="STOP", block_reason=None):
    return SimpleNamespace(
        text=text,
        usage_metadata=SimpleNamespace(
            prompt_token_count=prompt, candidates_token_count=candidates, thoughts_token_count=thoughts
        ),
        candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name=finish))] if finish else [],
        prompt_feedback=SimpleNamespace(block_reason=SimpleNamespace(name=block_reason)) if block_reason else None,
    )


def server_error(code=503, message="high demand"):
    return errors.ServerError(code, {"error": {"code": code, "message": message, "status": "UNAVAILABLE"}})


def client_error(code, *, retry_delay=None, retry_after_header=None, message="nope"):
    body = {"error": {"code": code, "message": message, "status": "X", "details": []}}
    if retry_delay is not None:
        body["error"]["details"].append(
            {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry_delay}
        )
    headers = {"retry-after": str(retry_after_header)} if retry_after_header is not None else {}
    return errors.ClientError(code, body, httpx.Response(code, headers=headers, json=body))


class Sleeps:
    def __init__(self) -> None:
        self.waited: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.waited.append(seconds)


def provider(client, sleeps=None, **overrides):
    settings = {
        "client": client,
        "sleep": sleeps if sleeps is not None else Sleeps(),
        "jitter": lambda: 0.0,
        "max_retries": 2,
        "retry_max_wait_seconds": 15.0,
        "timeout_seconds": 30.0,
    }
    settings.update(overrides)
    return GeminiLLMProvider("some-verified-model", **settings)


def ask(p, **kwargs):
    return p.complete(system="sys", user="usr", max_tokens=100, **kwargs)


# --- the successful call -------------------------------------------------------


def test_a_good_reply_comes_back_with_its_text_and_finish_reason() -> None:
    result = ask(provider(ScriptedClient(reply('{"a": 1}'))))

    assert result.text == '{"a": 1}'
    assert result.finish_reason == "STOP"


def test_thinking_tokens_are_counted_as_output_tokens() -> None:
    result = ask(provider(ScriptedClient(reply(prompt=120, candidates=30, thoughts=900))))

    assert result.input_tokens == 120
    assert result.output_tokens == 930  # 30 written + 900 reasoned, both billed as output


def test_missing_usage_counts_are_zero_not_a_crash() -> None:
    bare = SimpleNamespace(
        text="{}",
        usage_metadata=SimpleNamespace(prompt_token_count=None, candidates_token_count=None, thoughts_token_count=None),
        candidates=[],
        prompt_feedback=None,
    )

    result = ask(provider(ScriptedClient(bare)))

    assert (result.input_tokens, result.output_tokens) == (0, 0)


def test_a_reply_cut_off_at_the_token_limit_says_so() -> None:
    result = ask(provider(ScriptedClient(reply('{"answer": "cut', finish="MAX_TOKENS"))))

    assert result.finish_reason == "MAX_TOKENS"


# --- what is sent -------------------------------------------------------------


def test_the_request_carries_the_model_prompt_system_instruction_and_cap() -> None:
    client = ScriptedClient(reply())

    ask(provider(client))

    call = client.calls[0]
    assert call["model"] == "some-verified-model"
    assert call["contents"] == "usr"
    assert call["config"].system_instruction == "sys"
    assert call["config"].max_output_tokens == 100


def test_json_mode_and_the_schema_are_requested_when_a_schema_is_given() -> None:
    client = ScriptedClient(reply())

    ask(provider(client), json_schema=SCHEMA)

    config = client.calls[0]["config"]
    assert config.response_mime_type == "application/json"
    assert config.response_json_schema == SCHEMA


def test_json_mode_is_not_requested_without_a_schema() -> None:
    client = ScriptedClient(reply())

    ask(provider(client))

    config = client.calls[0]["config"]
    assert config.response_mime_type is None
    assert config.response_json_schema is None


def test_temperature_is_left_to_the_model_unless_set() -> None:
    unset, set_to_zero = ScriptedClient(reply()), ScriptedClient(reply())

    ask(provider(unset))
    ask(provider(set_to_zero, temperature=0.0))

    assert unset.calls[0]["config"].temperature is None
    assert set_to_zero.calls[0]["config"].temperature == 0.0


def test_the_client_is_built_with_the_configured_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    import google.genai as genai

    built: dict = {}

    class RecordingClient:
        def __init__(self, **kwargs) -> None:
            built.update(kwargs)
            self.models = ScriptedClient(reply())

    monkeypatch.setattr(genai, "Client", RecordingClient)

    p = GeminiLLMProvider("m", timeout_seconds=12.5)
    p.complete(system="s", user="u", max_tokens=10)

    assert built["http_options"].timeout == 12500  # the SDK counts milliseconds


# --- retries on transient failures ----------------------------------------------


def test_a_503_is_retried_after_a_backoff_and_then_succeeds() -> None:
    client, sleeps = ScriptedClient(server_error(), reply('{"ok": true}')), Sleeps()

    result = ask(provider(client, sleeps))

    assert result.text == '{"ok": true}'
    assert len(client.calls) == 2
    assert sleeps.waited == [1.0]


def test_backoff_doubles_between_retries() -> None:
    client, sleeps = ScriptedClient(server_error(), server_error(), reply()), Sleeps()

    ask(provider(client, sleeps))

    assert sleeps.waited == [1.0, 2.0]


def test_backoff_includes_jitter_and_never_exceeds_the_cap() -> None:
    sleeps = Sleeps()
    p = provider(ScriptedClient(server_error(), server_error(), reply()), sleeps, jitter=lambda: 1.0, retry_max_wait_seconds=2.2)

    ask(p)

    assert sleeps.waited == [1.5, 2.2]  # 1 + 0.5 jitter; then 2 + 0.5 capped at 2.2


def test_persistent_503s_give_up_after_the_configured_retries() -> None:
    client, sleeps = ScriptedClient(server_error(), server_error(), server_error()), Sleeps()

    with pytest.raises(LLMError, match="HTTP 503") as raised:
        ask(provider(client, sleeps))

    assert len(client.calls) == 3  # the first try and two retries
    assert not isinstance(raised.value, (LLMRateLimited, LLMTimeout))


def test_zero_retries_means_one_attempt() -> None:
    client = ScriptedClient(server_error())

    with pytest.raises(LLMError):
        ask(provider(client, max_retries=0))

    assert len(client.calls) == 1


def test_a_429_honours_the_retry_after_header() -> None:
    client, sleeps = ScriptedClient(client_error(429, retry_after_header=7), reply()), Sleeps()

    ask(provider(client, sleeps))

    assert sleeps.waited == [7.0]


def test_a_429_honours_the_retry_delay_in_the_error_body() -> None:
    client, sleeps = ScriptedClient(client_error(429, retry_delay="4s"), reply()), Sleeps()

    ask(provider(client, sleeps))

    assert sleeps.waited == [4.0]


def test_a_429_asking_for_a_long_wait_is_not_waited_out() -> None:
    """What a spent daily quota looks like: tell the person at once instead of stalling a request."""
    client, sleeps = ScriptedClient(client_error(429, retry_delay="33s")), Sleeps()

    with pytest.raises(LLMRateLimited) as raised:
        ask(provider(client, sleeps))

    assert raised.value.retry_after == 33.0
    assert len(client.calls) == 1
    assert sleeps.waited == []


def test_a_429_that_keeps_coming_back_ends_as_rate_limited() -> None:
    client = ScriptedClient(*[client_error(429, retry_delay="1s") for _ in range(3)])

    with pytest.raises(LLMRateLimited) as raised:
        ask(provider(client, Sleeps()))

    assert raised.value.retry_after == 1.0
    assert len(client.calls) == 3


def test_a_429_with_no_stated_delay_backs_off_like_any_transient_failure() -> None:
    client, sleeps = ScriptedClient(client_error(429), reply()), Sleeps()

    ask(provider(client, sleeps))

    assert sleeps.waited == [1.0]


# --- timeouts and connection failures ---------------------------------------------


def test_a_timeout_is_retried_and_then_reported_as_a_timeout() -> None:
    client = ScriptedClient(*[httpx.ReadTimeout("slow") for _ in range(3)])

    with pytest.raises(LLMTimeout, match="30 s"):
        ask(provider(client, Sleeps()))

    assert len(client.calls) == 3


def test_a_timeout_followed_by_success_is_a_success() -> None:
    result = ask(provider(ScriptedClient(httpx.ConnectTimeout("slow"), reply("{}")), Sleeps()))

    assert result.text == "{}"


def test_a_connection_failure_is_retried() -> None:
    client = ScriptedClient(httpx.ConnectError("no route"), reply())

    ask(provider(client, Sleeps()))

    assert len(client.calls) == 2


# --- failures that are not worth retrying -----------------------------------------


@pytest.mark.parametrize("code", [400, 401, 403, 404])
def test_a_rejected_request_is_never_retried(code: int) -> None:
    client, sleeps = ScriptedClient(client_error(code)), Sleeps()

    with pytest.raises(LLMError, match=f"HTTP {code}"):
        ask(provider(client, sleeps))

    assert len(client.calls) == 1
    assert sleeps.waited == []


def test_an_unexpected_exception_is_not_retried_and_not_echoed() -> None:
    client = ScriptedClient(ValueError("AIzaSyDoNotLeakThisKey"))

    with pytest.raises(LLMError) as raised:
        ask(provider(client, Sleeps()))

    assert len(client.calls) == 1
    assert "ValueError" in str(raised.value)
    assert "AIza" not in str(raised.value)


def test_error_messages_never_repeat_what_the_service_said() -> None:
    client = ScriptedClient(server_error(message="Your key AIzaSyDoNotLeakThisKey is over quota"))

    with pytest.raises(LLMError) as raised:
        ask(provider(client, max_retries=0))

    assert "AIza" not in str(raised.value)
    assert "quota" not in str(raised.value)


# --- the model declining, and empty replies ------------------------------------------


def test_a_blocked_prompt_is_a_block_not_an_outage() -> None:
    blocked = reply(text=None, finish=None, block_reason="PROHIBITED_CONTENT")

    with pytest.raises(LLMBlocked) as raised:
        ask(provider(ScriptedClient(blocked)))

    assert raised.value.reason == "PROHIBITED_CONTENT"


def test_a_safety_stop_with_no_text_is_a_block() -> None:
    with pytest.raises(LLMBlocked) as raised:
        ask(provider(ScriptedClient(reply(text=None, finish="SAFETY"))))

    assert raised.value.reason == "SAFETY"


def test_a_block_is_not_retried() -> None:
    client = ScriptedClient(reply(text=None, finish="SAFETY"))

    with pytest.raises(LLMBlocked):
        ask(provider(client, Sleeps()))

    assert len(client.calls) == 1


def test_an_empty_reply_that_finished_normally_is_an_error_not_a_block() -> None:
    with pytest.raises(LLMError, match="no text content") as raised:
        ask(provider(ScriptedClient(reply(text=None, finish="STOP"))))

    assert not isinstance(raised.value, LLMBlocked)


def test_an_empty_string_reply_is_an_error() -> None:
    with pytest.raises(LLMError, match="no text content"):
        ask(provider(ScriptedClient(reply(text="", finish="STOP"))))


# --- wiring from the settings ----------------------------------------------------------


def test_the_application_builds_the_provider_from_the_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.config import get_settings
    from app.generation.provider import get_llm_provider, reset_llm_provider

    monkeypatch.setenv("KNOWLEDGEOS_LLM_MODEL", "a-verified-model")
    monkeypatch.setenv("KNOWLEDGEOS_LLM_TIMEOUT_SECONDS", "12")
    monkeypatch.setenv("KNOWLEDGEOS_LLM_MAX_RETRIES", "5")
    monkeypatch.setenv("KNOWLEDGEOS_LLM_RETRY_MAX_WAIT_SECONDS", "9")
    monkeypatch.setenv("KNOWLEDGEOS_LLM_TEMPERATURE", "0")
    get_settings.cache_clear()
    reset_llm_provider()
    try:
        built = get_llm_provider()
        assert built.model_name == "a-verified-model"
        assert built._timeout_seconds == 12
        assert built._max_retries == 5
        assert built._retry_max_wait_seconds == 9
        assert built._temperature == 0
    finally:
        get_settings.cache_clear()
        reset_llm_provider()
