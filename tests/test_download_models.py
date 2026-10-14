"""`scripts/download_models.py`: the one place the two models are fetched.

No network: the dry run only prints, and the other tests read the module."""

import importlib.util
import pathlib

from app.providers.cross_encoder import MODEL_NAME as RERANK_MODEL_NAME
from app.providers.embeddings import MODEL_NAME as EMBEDDING_MODEL_NAME

SCRIPT = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "download_models.py"


def _load():
    spec = importlib.util.spec_from_file_location("download_models", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_it_fetches_exactly_the_models_the_providers_load() -> None:
    assert _load().MODELS == (EMBEDDING_MODEL_NAME, RERANK_MODEL_NAME)


def test_a_dry_run_names_both_models_and_downloads_nothing(capsys, monkeypatch) -> None:
    import huggingface_hub

    def _must_not_download(*args, **kwargs):
        raise AssertionError("a dry run must not download")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", _must_not_download)

    assert _load().main(["--dry-run"]) == 0

    out = capsys.readouterr().out
    assert EMBEDDING_MODEL_NAME in out
    assert RERANK_MODEL_NAME in out
    assert "Hugging Face cache:" in out


def test_a_failed_download_is_reported_and_exits_non_zero(capsys, monkeypatch) -> None:
    import huggingface_hub

    def _offline(*args, **kwargs):
        raise ConnectionError("no network")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", _offline)

    assert _load().main([]) == 1

    assert "FAILED" in capsys.readouterr().err


def test_the_application_itself_never_downloads_a_model() -> None:
    """The reason this script exists: both providers load with `local_files_only=True`."""
    root = SCRIPT.parent.parent / "app" / "providers"
    for name in ("bge.py", "cross_encoder.py"):
        assert "local_files_only=True" in (root / name).read_text(encoding="utf-8"), name
