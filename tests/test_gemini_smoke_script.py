"""scripts/gemini_smoke.py, with the provider replaced: no network, no key."""

import importlib.util
import json
from pathlib import Path

import pytest

from app.providers.llm import LLMError, LLMRateLimited, LLMResult

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "gemini_smoke.py"


@pytest.fixture
def smoke(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("gemini_smoke_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "RESULTS_DIR", tmp_path)
    return module


def _provider(outcome):
    class Fake:
        constructed = {}

        def __init__(self, model, **kwargs):
            Fake.constructed = {"model": model, **kwargs}

        def complete(self, **_):
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

    return Fake


def test_records_model_temperature_tokens_and_uses_no_retries(smoke, monkeypatch, tmp_path, capsys):
    fake = _provider(LLMResult(text="Smoke test ok.", input_tokens=23, output_tokens=4, finish_reason="STOP"))
    monkeypatch.setattr(smoke, "GeminiLLMProvider", fake)

    assert smoke.main(["--model", "some-model", "--record"]) == 0

    assert fake.constructed["max_retries"] == 0  # exactly one request
    capsys.readouterr()
    (path,) = tmp_path.glob("*-smoke-some-model.json")
    saved = json.loads(path.read_text("utf-8"))
    assert saved["model"] == "some-model"
    assert saved["temperature"] is None and "default" in saved["temperature_note"]
    assert (saved["input_tokens"], saved["output_tokens"], saved["requests"]) == (23, 4, 1)


def test_a_rate_limit_exits_3_and_writes_nothing(smoke, monkeypatch, tmp_path):
    monkeypatch.setattr(smoke, "GeminiLLMProvider", _provider(LLMRateLimited("slow down", retry_after=51)))
    assert smoke.main(["--model", "m", "--record"]) == 3
    assert list(tmp_path.iterdir()) == []


def test_any_other_failure_exits_1_and_writes_nothing(smoke, monkeypatch, tmp_path):
    monkeypatch.setattr(smoke, "GeminiLLMProvider", _provider(LLMError("generation failed (HTTP 404)")))
    assert smoke.main(["--model", "m", "--record"]) == 1
    assert list(tmp_path.iterdir()) == []


def test_no_model_is_an_error(smoke, monkeypatch):
    monkeypatch.delenv("KNOWLEDGEOS_LLM_MODEL", raising=False)
    monkeypatch.setattr(smoke, "get_settings", lambda: type("S", (), {"llm_model": None, "llm_temperature": None, "llm_timeout_seconds": 30.0})())
    assert smoke.main([]) == 1
