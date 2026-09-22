"""P3 (PROJECT_PLAN.md): the demo generation provider
(`demo.llm.DemoLLMProvider`), which replays `demo/fixtures/answers.yaml`
as deterministic, marker-compliant JSON.

No database needed. Evidence is built with the real, unmodified
`app.generation.evidence.select_evidence` and
`app.generation.prompt.serialize_evidence` -- the exact functions
`generate_answer()` itself calls -- and several tests here drive the
real, unmodified `generate_answer()` end to end, so these prove the
provider's output survives the real citation-validation and grounding
gate, not a hand-rolled approximation of it.
"""

import ast
import json
import pathlib
import re

import pytest
import yaml

from app.generation.abstention import REASON_SUFFICIENT_EVIDENCE_FALSE
from app.generation.citations import extract_all_inline_citations, split_sentences
from app.generation.evidence import select_evidence
from app.generation.generator import generate_answer
from app.generation.prompt import serialize_evidence
from app.providers.llm import LLMError
from demo.llm import ANSWERS_FIXTURE_PATH, DEMO_LLM_MODEL_NAME, DemoLLMProvider

from .generation_fixtures import make_reranked

FLAGSHIP_QUESTION_IDS = ["da001", "cs001", "cv001", "md001", "ie001"]


@pytest.fixture(scope="module")
def fixture_entries() -> dict[str, dict]:
    raw = yaml.safe_load(ANSWERS_FIXTURE_PATH.read_text(encoding="utf-8"))
    return {entry["id"]: entry for entry in raw}


def _user_prompt_and_candidates(entry: dict):
    """A real `user` prompt -- built with the real `serialize_evidence`
    -- carrying one candidate per fixture citation (or one arbitrary
    candidate for an abstaining entry with none), all within the real
    selection limit so every one is marked selected."""
    uids = entry["citations"] or ["00000000000000000000000000000000"]
    candidates = [make_reranked(i + 1, uid=uid) for i, uid in enumerate(uids)]
    selected = select_evidence(candidates)
    evidence_block = serialize_evidence(selected)
    user = f"Question:\n{entry['question']}\n\nEvidence:\n{evidence_block}"
    return user, candidates


# --- DemoLLMProvider.complete(): replay correctness ------------------------


def test_model_name_is_distinct_from_any_real_vendor() -> None:
    provider = DemoLLMProvider()
    assert provider.model_name == DEMO_LLM_MODEL_NAME
    for vendor_word in ("gemini", "anthropic", "gpt", "claude"):
        assert vendor_word not in provider.model_name.lower()


@pytest.mark.parametrize("question_id", FLAGSHIP_QUESTION_IDS)
def test_all_five_flagship_questions_resolve(question_id, fixture_entries) -> None:
    entry = fixture_entries[question_id]
    user, _ = _user_prompt_and_candidates(entry)
    provider = DemoLLMProvider()

    result = provider.complete(system="irrelevant", user=user, max_tokens=2048)
    parsed = json.loads(result.text)

    assert set(parsed) == {"answer", "citations", "sufficient_evidence"}
    assert parsed["citations"] == entry["citations"]
    assert parsed["sufficient_evidence"] == entry["sufficient_evidence"]


@pytest.mark.parametrize("question_id", FLAGSHIP_QUESTION_IDS)
def test_fixture_prose_is_preserved_word_for_word(question_id, fixture_entries) -> None:
    """The only difference between the fixture's own `answer_text` and
    what this provider returns is the appended `[<chunk_uid>]`
    marker(s) -- never a reworded claim."""
    entry = fixture_entries[question_id]
    user, _ = _user_prompt_and_candidates(entry)
    provider = DemoLLMProvider()

    result = provider.complete(system="s", user=user, max_tokens=2048)
    answer = json.loads(result.text)["answer"]

    stripped = re.sub(r"\s*\[[0-9a-f]{32}\]", "", answer)
    assert stripped == " ".join(split_sentences(entry["answer_text"]))


@pytest.mark.parametrize("question_id", FLAGSHIP_QUESTION_IDS)
def test_citation_markers_are_exactly_the_fixtures_declared_citations(
    question_id, fixture_entries
) -> None:
    """Every marker refers to a fixture citation, and every fixture
    citation is marked -- no fabrication either direction."""
    entry = fixture_entries[question_id]
    user, _ = _user_prompt_and_candidates(entry)
    provider = DemoLLMProvider()

    result = provider.complete(system="s", user=user, max_tokens=2048)
    answer = json.loads(result.text)["answer"]

    assert extract_all_inline_citations(answer) == frozenset(entry["citations"])


def test_output_is_deterministic_across_repeated_calls(fixture_entries) -> None:
    entry = fixture_entries["md001"]
    user, _ = _user_prompt_and_candidates(entry)
    provider = DemoLLMProvider()

    first = provider.complete(system="s", user=user, max_tokens=2048)
    second = provider.complete(system="s", user=user, max_tokens=2048)

    assert first.text == second.text


def test_an_unrecognized_question_raises() -> None:
    provider = DemoLLMProvider()
    user = (
        "Question:\nA question nobody ever precomputed a demo answer for.\n\n"
        'Evidence:\n<evidence chunk_uid="' + "0" * 32 + '" document="x" version="1">\n'
        "some text\n</evidence>"
    )

    with pytest.raises(LLMError):
        provider.complete(system="s", user=user, max_tokens=2048)


def test_a_fixture_citation_missing_from_the_shown_evidence_raises(fixture_entries) -> None:
    """No fabricated citation IDs: if the evidence this call was shown
    does not actually include a chunk_uid the fixture declares, the
    provider refuses rather than citing something it was never given."""
    entry = fixture_entries["cs001"]
    user = (
        f"Question:\n{entry['question']}\n\nEvidence:\n"
        '<evidence chunk_uid="' + "1" * 32 + '" document="Unrelated" version="1">\n'
        "unrelated text\n</evidence>"
    )
    provider = DemoLLMProvider()

    with pytest.raises(LLMError):
        provider.complete(system="s", user=user, max_tokens=2048)


def test_the_error_never_carries_evidence_text() -> None:
    provider = DemoLLMProvider()
    secret_text = "SUPERSECRET production access policy details"
    user = (
        "Question:\nan unrecognized question\n\nEvidence:\n"
        '<evidence chunk_uid="' + "0" * 32 + f'" document="x" version="1">\n{secret_text}\n</evidence>'
    )

    with pytest.raises(LLMError) as raised:
        provider.complete(system="s", user=user, max_tokens=2048)

    assert secret_text not in str(raised.value)


# --- static guards: no network, no real provider, no app/generation/ edit --


def test_the_demo_provider_never_imports_a_real_or_network_provider() -> None:
    import demo.llm as module

    tree = ast.parse(pathlib.Path(module.__file__).read_text())
    top_level = {
        alias.name.split(".")[0]
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module.split(".")[0]
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module
    }

    forbidden = {
        "google",
        "genai",
        "anthropic",
        "openai",
        "requests",
        "httpx",
        "urllib",
        "urllib3",
        "socket",
    }
    assert not (top_level & forbidden)


def test_the_demo_provider_never_references_the_real_gemini_provider() -> None:
    import demo.llm as module

    source = pathlib.Path(module.__file__).read_text()
    assert "GeminiLLMProvider" not in source
    assert "gemini_llm" not in source


def test_the_demo_provider_never_imports_generation_orchestration_or_validation() -> None:
    """`app.generation.citations.split_sentences` is a pure function this
    module reuses; the orchestration (`generator.py`), the validation
    gate itself (`validate_citations`), and grounding (`grounding.py`)
    must never be *imported* here -- this provider only ever plugs into
    them from the outside, through the real `LLMProvider` interface.
    (The module's own docstring names them in prose, which is exactly
    why this checks actual import statements, not the file's raw text.)
    """
    import demo.llm as module

    tree = ast.parse(pathlib.Path(module.__file__).read_text())
    imported_names = {
        alias.asname or alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    forbidden = {"generate_answer", "validate_citations", "deterministic_grounding"}
    assert not (imported_names & forbidden)


# --- end-to-end: the real, unmodified generate_answer() accepts this ------


@pytest.mark.parametrize("question_id", FLAGSHIP_QUESTION_IDS)
def test_generate_answer_accepts_the_demo_provider_end_to_end(question_id, fixture_entries) -> None:
    """The real, unmodified `generate_answer()` /
    `validate_citations()` / `deterministic_grounding()` pipeline --
    never bypassed -- accepts this provider's output and reproduces the
    fixture's own declared citations and abstention state."""
    entry = fixture_entries[question_id]
    _, candidates = _user_prompt_and_candidates(entry)
    provider = DemoLLMProvider()

    result = generate_answer(
        query_text=entry["question"],
        candidates=candidates,
        llm=provider,
        abstention_threshold=None,
        max_tokens=2048,
    )

    assert result.citations == entry["citations"]
    assert result.abstained == entry["abstained"]
    assert result.sufficient_evidence == entry["sufficient_evidence"]
    if entry["abstained"]:
        assert result.abstain_reason == REASON_SUFFICIENT_EVIDENCE_FALSE
    else:
        assert result.citation_valid is True
        assert result.grounded is True
