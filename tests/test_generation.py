"""Generation orchestration: prompt -> provider -> parse -> validate ->
ground, over the scripted fake. No database, no network.

Covers every scenario the specification's §16 names by name: valid answer,
malformed output, invalid JSON, a citation to a chunk that was never
retrieved, a citation to a retrieved-but-not-selected chunk, a citation
coverage failure, `sufficient_evidence=false`, and provider failure.
"""

import pytest

from app.generation.citations import CitationError
from app.generation.generator import GenerationError, generate_answer
from app.generation.prompt import load_prompt
from app.providers.fake_llm import FakeLLMProvider
from app.providers.llm import LLMError

from .generation_fixtures import cited_answer_text, make_candidates, valid_json

PROMPT = load_prompt()  # loaded once; generation tests never touch the network


def _candidates(count: int) -> list:
    return make_candidates(count)


# --- pre-LLM abstention: zero candidates ----------------------------------


def test_zero_candidates_abstains_without_calling_the_model() -> None:
    fake = FakeLLMProvider([])  # no scripted responses -- would raise if called

    answer = generate_answer(
        query_text="anything",
        candidates=[],
        llm=fake,
        abstention_threshold=None,
        max_tokens=100,
        prompt=PROMPT,
    )

    assert answer.abstained
    assert answer.abstain_reason == "no_candidates"
    assert answer.citations == []
    assert answer.model_name is None
    assert fake.call_count == 0


def test_pre_llm_abstention_produces_a_valid_grounded_answer() -> None:
    fake = FakeLLMProvider([])

    answer = generate_answer(
        query_text="anything",
        candidates=[],
        llm=fake,
        abstention_threshold=None,
        max_tokens=100,
        prompt=PROMPT,
    )

    assert answer.citation_valid
    assert answer.grounded


# --- pre-LLM abstention: rerank score threshold ----------------------------


def test_a_configured_threshold_can_abstain_before_calling_the_model() -> None:
    candidates = _candidates(3)  # top rerank_score = 100 - 1 = 99.0
    fake = FakeLLMProvider([])

    answer = generate_answer(
        query_text="anything",
        candidates=candidates,
        llm=fake,
        abstention_threshold=200.0,  # above every candidate's score
        max_tokens=100,
        prompt=PROMPT,
    )

    assert answer.abstained
    assert answer.abstain_reason == "rerank_score_below_threshold"
    assert fake.call_count == 0


def test_a_none_threshold_never_abstains_before_calling_the_model() -> None:
    candidates = _candidates(1)
    uid = candidates[0].evidence.chunk_uid
    fake = FakeLLMProvider([valid_json(cited_answer_text(uid), [uid])])

    answer = generate_answer(
        query_text="anything",
        candidates=candidates,
        llm=fake,
        abstention_threshold=None,
        max_tokens=100,
        prompt=PROMPT,
    )

    assert not answer.abstained
    assert fake.call_count == 1


# --- the valid path ---------------------------------------------------


def test_a_valid_well_formed_answer_is_accepted() -> None:
    candidates = _candidates(3)
    uids = [c.evidence.chunk_uid for c in candidates]
    fake = FakeLLMProvider([valid_json(cited_answer_text(*uids), uids)])

    answer = generate_answer(
        query_text="what is the policy",
        candidates=candidates,
        llm=fake,
        abstention_threshold=None,
        max_tokens=100,
        prompt=PROMPT,
    )

    assert not answer.abstained
    assert answer.citation_valid
    assert answer.grounded
    assert set(answer.citations) == set(uids)
    assert answer.model_name == fake.model_name
    assert answer.prompt_version == "knowledge_answer_v1"
    assert answer.input_tokens > 0
    assert answer.output_tokens > 0


def test_the_selected_set_is_exactly_the_top_eight() -> None:
    candidates = _candidates(12)
    uids = [c.evidence.chunk_uid for c in candidates[:8]]
    fake = FakeLLMProvider([valid_json(cited_answer_text(*uids), uids)])

    answer = generate_answer(
        query_text="q",
        candidates=candidates,
        llm=fake,
        abstention_threshold=None,
        max_tokens=100,
        prompt=PROMPT,
    )

    selected_ranks = {item.candidate.final_rank for item in answer.selected if item.selected}
    assert selected_ranks == set(range(1, 9))


# --- malformed / invalid output ---------------------------------------


def test_invalid_json_raises_generation_error() -> None:
    candidates = _candidates(1)
    fake = FakeLLMProvider(["this is not { valid json at all"])

    with pytest.raises(GenerationError, match="not valid JSON"):
        generate_answer(
            query_text="q",
            candidates=candidates,
            llm=fake,
            abstention_threshold=None,
            max_tokens=100,
            prompt=PROMPT,
        )


def test_malformed_shape_missing_a_required_field_raises_generation_error() -> None:
    import json

    candidates = _candidates(1)
    malformed = json.dumps({"answer": "text only, no citations field"})
    fake = FakeLLMProvider([malformed])

    with pytest.raises(GenerationError, match="expected exactly"):
        generate_answer(
            query_text="q",
            candidates=candidates,
            llm=fake,
            abstention_threshold=None,
            max_tokens=100,
            prompt=PROMPT,
        )


def test_malformed_shape_wrong_type_raises_generation_error() -> None:
    import json

    candidates = _candidates(1)
    malformed = json.dumps(
        {"answer": "ok", "citations": "not-a-list", "sufficient_evidence": True}
    )
    fake = FakeLLMProvider([malformed])

    with pytest.raises(GenerationError, match="must be a list"):
        generate_answer(
            query_text="q",
            candidates=candidates,
            llm=fake,
            abstention_threshold=None,
            max_tokens=100,
            prompt=PROMPT,
        )


def test_no_silent_repair_of_malformed_output() -> None:
    """A JSON object with an extra, unexpected field is rejected outright
    -- never coerced by dropping the extra field silently."""
    import json

    candidates = _candidates(1)
    extra_field = json.dumps(
        {
            "answer": "ok",
            "citations": [],
            "sufficient_evidence": True,
            "confidence": 0.9,
        }
    )
    fake = FakeLLMProvider([extra_field])

    with pytest.raises(GenerationError, match="expected exactly"):
        generate_answer(
            query_text="q",
            candidates=candidates,
            llm=fake,
            abstention_threshold=None,
            max_tokens=100,
            prompt=PROMPT,
        )


# --- invalid citations ---------------------------------------------------


def test_a_citation_to_a_never_retrieved_chunk_is_rejected() -> None:
    candidates = _candidates(1)
    fake_uid = "f" * 32
    fake = FakeLLMProvider([valid_json(cited_answer_text(fake_uid), [fake_uid])])

    with pytest.raises(CitationError, match="not in the retrieved evidence"):
        generate_answer(
            query_text="q",
            candidates=candidates,
            llm=fake,
            abstention_threshold=None,
            max_tokens=100,
            prompt=PROMPT,
        )


def test_a_citation_to_a_retrieved_but_unselected_chunk_is_rejected() -> None:
    candidates = _candidates(12)  # only the first 8 are selected
    unselected_uid = candidates[9].evidence.chunk_uid  # final_rank 10
    fake = FakeLLMProvider(
        [valid_json(cited_answer_text(unselected_uid), [unselected_uid])]
    )

    with pytest.raises(CitationError, match="not selected"):
        generate_answer(
            query_text="q",
            candidates=candidates,
            llm=fake,
            abstention_threshold=None,
            max_tokens=100,
            prompt=PROMPT,
        )


def test_a_citation_coverage_failure_is_rejected() -> None:
    candidates = _candidates(1)
    uid = candidates[0].evidence.chunk_uid
    text = f"Cited fact [{uid}]. Uncited fact with no marker at all."
    fake = FakeLLMProvider([valid_json(text, [uid])])

    with pytest.raises(CitationError, match="no citation"):
        generate_answer(
            query_text="q",
            candidates=candidates,
            llm=fake,
            abstention_threshold=None,
            max_tokens=100,
            prompt=PROMPT,
        )


# --- sufficient_evidence=false: post-LLM abstention -------------------


def test_sufficient_evidence_false_abstains_after_calling_the_model() -> None:
    candidates = _candidates(3)
    fake = FakeLLMProvider(
        [valid_json("There is not enough information to answer this.", [], sufficient_evidence=False)]
    )

    answer = generate_answer(
        query_text="q",
        candidates=candidates,
        llm=fake,
        abstention_threshold=None,
        max_tokens=100,
        prompt=PROMPT,
    )

    assert answer.abstained
    assert answer.abstain_reason == "sufficient_evidence_false"
    assert answer.citations == []
    assert answer.model_name == fake.model_name  # the model WAS called
    assert fake.call_count == 1


def test_sufficient_evidence_false_with_leftover_citations_is_rejected() -> None:
    """The model claiming insufficient evidence while still citing
    something is an inconsistent, invalid response -- rejected, not
    silently repaired by dropping the citations."""
    candidates = _candidates(1)
    uid = candidates[0].evidence.chunk_uid
    fake = FakeLLMProvider(
        [valid_json(f"Partial fact [{uid}].", [uid], sufficient_evidence=False)]
    )

    with pytest.raises(CitationError, match="must carry no citations"):
        generate_answer(
            query_text="q",
            candidates=candidates,
            llm=fake,
            abstention_threshold=None,
            max_tokens=100,
            prompt=PROMPT,
        )


# --- provider failure ---------------------------------------------------


def test_provider_failure_propagates_as_llm_error() -> None:
    candidates = _candidates(1)
    fake = FakeLLMProvider([LLMError("the model could not be reached")])

    with pytest.raises(LLMError, match="could not be reached"):
        generate_answer(
            query_text="q",
            candidates=candidates,
            llm=fake,
            abstention_threshold=None,
            max_tokens=100,
            prompt=PROMPT,
        )


def test_provider_failure_leaves_no_generated_answer_to_persist() -> None:
    """A raised LLMError means `generate_answer` never returns -- there is
    nothing for a caller to accidentally persist."""
    candidates = _candidates(1)
    fake = FakeLLMProvider([LLMError("down")])

    try:
        generate_answer(
            query_text="q",
            candidates=candidates,
            llm=fake,
            abstention_threshold=None,
            max_tokens=100,
            prompt=PROMPT,
        )
        raised = False
    except LLMError:
        raised = True

    assert raised
