"""`DemoLLMProvider`: matching, JSON shape, and structural isolation from
the real Gemini adapter. No network, no credential, no database."""

import ast
import json
import pathlib

import pytest

from app.generation.demo_scenarios import DEMO_SCENARIOS
from app.providers.demo_llm import (
    DEMO_MODEL_NAME,
    DemoAnswerNotAvailable,
    DemoLLMProvider,
)
from app.providers.llm import LLMProvider


def _user_prompt(question_text: str, evidence: str = "<evidence>irrelevant</evidence>") -> str:
    """The exact shape `app.generation.generator.generate_answer` builds —
    see that module's `user_prompt = f"Question:\\n{query_text}\\n\\n
    Evidence:\\n{evidence_block}"` line, which this test intentionally
    stays coupled to."""
    return f"Question:\n{question_text}\n\nEvidence:\n{evidence}"


def test_demo_llm_provider_is_a_protocol_the_provider_satisfies() -> None:
    assert isinstance(DemoLLMProvider(), LLMProvider)


def test_model_name_is_a_distinct_demo_constant() -> None:
    assert DemoLLMProvider().model_name == DEMO_MODEL_NAME
    assert "gemini" not in DEMO_MODEL_NAME.lower()


@pytest.mark.parametrize("scenario", DEMO_SCENARIOS, ids=lambda s: s.question_id)
def test_each_curated_scenario_returns_its_hand_verified_answer(scenario) -> None:
    provider = DemoLLMProvider()

    result = provider.complete(
        system="irrelevant", user=_user_prompt(scenario.question_text), max_tokens=999
    )
    parsed = json.loads(result.text)

    assert parsed == {
        "answer": scenario.answer_text,
        "citations": list(scenario.citations),
        "sufficient_evidence": scenario.sufficient_evidence,
    }


def test_ie001_reports_insufficient_evidence_with_the_real_abstention_text() -> None:
    from app.generation.abstention import DEFAULT_ABSTENTION_TEXT
    from app.generation.demo_scenarios import find_scenario

    scenario = find_scenario(
        "What is the company's policy on using generative AI tools for writing code?"
    )
    assert scenario is not None
    assert scenario.answer_text == DEFAULT_ABSTENTION_TEXT
    assert scenario.sufficient_evidence is False
    assert scenario.citations == ()


def test_an_unmatched_question_raises_rather_than_answering() -> None:
    provider = DemoLLMProvider()

    with pytest.raises(DemoAnswerNotAvailable):
        provider.complete(
            system="s", user=_user_prompt("What is the weather today?"), max_tokens=100
        )


def test_a_near_miss_does_not_accidentally_match() -> None:
    """Deterministic exact-after-normalization matching only — a question
    that merely resembles a curated one must not match it."""
    provider = DemoLLMProvider()
    close_but_not_it = (
        DEMO_SCENARIOS[0].question_text.rstrip("?") + ", roughly speaking?"
    )

    with pytest.raises(DemoAnswerNotAvailable):
        provider.complete(system="s", user=_user_prompt(close_but_not_it), max_tokens=100)


def test_matching_tolerates_case_and_whitespace_but_nothing_fuzzier() -> None:
    provider = DemoLLMProvider()
    scenario = DEMO_SCENARIOS[0]
    noisy = "  " + scenario.question_text.upper().replace(" ", "   ") + "  "

    result = provider.complete(system="s", user=_user_prompt(noisy), max_tokens=100)

    assert json.loads(result.text)["answer"] == scenario.answer_text


def test_malformed_prompt_shape_is_treated_as_no_match_not_a_crash() -> None:
    """If `generate_answer`'s prompt format ever changed shape, this
    provider fails closed (no scenario matches) rather than raising an
    unrelated exception or guessing at a question."""
    provider = DemoLLMProvider()

    with pytest.raises(DemoAnswerNotAvailable):
        provider.complete(system="s", user="not the expected shape at all", max_tokens=100)


def test_question_extraction_matches_the_real_prompt_format() -> None:
    """Guards the coupling `app/providers/demo_llm.py` documents: this
    provider parses `user` assuming `app.generation.generator.
    generate_answer`'s exact `f"Question:\\n{query_text}\\n\\nEvidence:\\n
    {evidence_block}"` shape. If that f-string ever changes, this test
    should be the one that catches it."""
    scenario = DEMO_SCENARIOS[0]
    from app.generation.evidence import select_evidence
    from app.generation.prompt import serialize_evidence
    from tests.generation_fixtures import make_candidates

    candidates = make_candidates(1)
    selected = select_evidence(candidates)
    real_shaped_user = (
        f"Question:\n{scenario.question_text}\n\nEvidence:\n"
        f"{serialize_evidence(selected)}"
    )

    result = DemoLLMProvider().complete(system="s", user=real_shaped_user, max_tokens=1)

    assert json.loads(result.text)["answer"] == scenario.answer_text


def test_token_counts_are_derived_never_fabricated() -> None:
    provider = DemoLLMProvider()
    scenario = DEMO_SCENARIOS[0]
    user = _user_prompt(scenario.question_text)

    result = provider.complete(system="a b c", user=user, max_tokens=100)

    assert result.input_tokens == len(user.split()) + len("a b c".split())
    assert result.output_tokens == len(result.text.split())


# --- structural isolation from the real Gemini adapter -----------------


def test_demo_provider_module_never_imports_the_gemini_adapter() -> None:
    module_path = (
        pathlib.Path(__file__).resolve().parent.parent
        / "app"
        / "providers"
        / "demo_llm.py"
    )
    tree = ast.parse(module_path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert "gemini" not in node.module.lower(), node.module
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "gemini" not in alias.name.lower(), alias.name


def test_nothing_outside_the_demo_seam_selects_the_demo_provider() -> None:
    """The same static guard `test_llm_provider.py` applies to
    `FakeLLMProvider` — `DemoLLMProvider` may be named only by its own
    module and by `app/api/query.py::llm_provider`, the one place M2
    wires demo mode in."""
    app_dir = pathlib.Path(__file__).resolve().parent.parent / "app"
    allowed = {"demo_llm.py", "query.py"}
    for module in app_dir.rglob("*.py"):
        if module.name in allowed:
            continue
        tree = ast.parse(module.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                assert node.id != "DemoLLMProvider", module
