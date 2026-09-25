"""The 8 curated demo scenarios, proved correct against the REAL
`generate_answer` orchestration — real citation validation
(`app.generation.citations.validate_citations`) and real grounding
(`app.generation.grounding.deterministic_grounding`), not a re-implementation
of either. No network, no database: candidates are built with
`tests/generation_fixtures.make_reranked`, one per real cited `chunk_uid`,
so this proves the *answer text and citation shape* are valid independent of
retrieval quality — a concern this file leaves to
`tests/test_demo_query_api.py`.
"""

import pytest

from app.generation.abstention import DEFAULT_ABSTENTION_TEXT
from app.generation.demo_scenarios import DEMO_SCENARIOS
from app.generation.generator import generate_answer
from app.providers.demo_llm import DemoLLMProvider
from tests.generation_fixtures import make_reranked


def _candidates_for(scenario):
    """One reranked candidate per real cited chunk_uid, ranked in citation
    order, plus a couple of uncited noise candidates — the same shape a
    real top-8 selection would have: not every retrieved chunk is cited."""
    cited = [
        make_reranked(rank, uid=uid)
        for rank, uid in enumerate(scenario.citations, start=1)
    ]
    noise_start = len(cited) + 1
    noise = [
        make_reranked(rank, uid=f"{'f' * 31}{rank:x}")
        for rank in range(noise_start, noise_start + 2)
    ]
    return cited + noise


@pytest.mark.parametrize("scenario", DEMO_SCENARIOS, ids=lambda s: s.question_id)
def test_scenario_answer_passes_real_citation_validation_and_grounding(scenario) -> None:
    result = generate_answer(
        query_text=scenario.question_text,
        candidates=_candidates_for(scenario),
        llm=DemoLLMProvider(),
        abstention_threshold=None,
        max_tokens=512,
    )

    assert result.answer_text == scenario.answer_text
    assert result.citations == list(scenario.citations)
    assert result.citation_valid is True


@pytest.mark.parametrize(
    "scenario",
    [s for s in DEMO_SCENARIOS if s.question_id != "ie001"],
    ids=lambda s: s.question_id,
)
def test_supported_scenarios_are_grounded_and_not_abstained(scenario) -> None:
    result = generate_answer(
        query_text=scenario.question_text,
        candidates=_candidates_for(scenario),
        llm=DemoLLMProvider(),
        abstention_threshold=None,
        max_tokens=512,
    )

    assert result.abstained is False
    assert result.grounded is True
    assert result.sufficient_evidence is True


def test_ie001_abstains_through_the_real_post_llm_path_with_the_real_text() -> None:
    scenario = next(s for s in DEMO_SCENARIOS if s.question_id == "ie001")

    result = generate_answer(
        query_text=scenario.question_text,
        candidates=_candidates_for(scenario),
        llm=DemoLLMProvider(),
        abstention_threshold=None,
        max_tokens=512,
    )

    assert result.abstained is True
    assert result.answer_text == DEFAULT_ABSTENTION_TEXT
    assert result.citations == []
    assert result.sufficient_evidence is False
    assert result.citation_valid is True


def test_da001_cites_only_the_remote_work_chunk() -> None:
    """A named example of the multi-scenario check above, pinned so a
    future scenario-text edit that accidentally drops or adds a citation
    fails loudly and specifically, not just as one row of a parametrize."""
    scenario = next(s for s in DEMO_SCENARIOS if s.question_id == "da001")

    result = generate_answer(
        query_text=scenario.question_text,
        candidates=_candidates_for(scenario),
        llm=DemoLLMProvider(),
        abstention_threshold=None,
        max_tokens=512,
    )

    assert result.citations == ["5a04c330c14efaf8f4da83ebf6d6854f"]
    assert result.grounded is True


def test_md001_cites_both_the_incident_and_the_standard_sop_chunks() -> None:
    scenario = next(s for s in DEMO_SCENARIOS if s.question_id == "md001")

    result = generate_answer(
        query_text=scenario.question_text,
        candidates=_candidates_for(scenario),
        llm=DemoLLMProvider(),
        abstention_threshold=None,
        max_tokens=512,
    )

    assert result.citations == [
        "f9cbee4e0939d52e95eebe93a063a85c",
        "da1cdb5e0e87ad69728e9d2781239d13",
    ]
    assert result.grounded is True
