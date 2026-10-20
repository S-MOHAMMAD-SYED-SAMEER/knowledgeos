"""Generation orchestration: prompt -> provider -> parse -> validate ->
ground, over the scripted fake. No database, no network.

Covers every scenario the specification's §16 names by name: valid answer,
malformed output, invalid JSON, a citation to a chunk that was never
retrieved, a citation to a retrieved-but-not-selected chunk, a citation
coverage failure, `sufficient_evidence=false`, and provider failure.

A rejected reply is never shown and never repaired. It is retried once; if the second
reply is rejected too, the result is the fixed "I couldn't verify this answer" reply
(`verification_failure` set). Each rejection scenario below is therefore scripted twice.
"""

import json
import logging

import pytest

from app.generation.generator import (
    ANSWER_JSON_SCHEMA,
    FAILURE_CITATION_INVALID,
    FAILURE_MALFORMED_OUTPUT,
    FAILURE_MODEL_BLOCKED,
    MAX_GENERATION_ATTEMPTS,
    UNVERIFIED_ANSWER_TEXT,
    generate_answer,
)
from app.generation.prompt import load_prompt
from app.providers.fake_llm import FakeLLMProvider
from app.providers.llm import (
    LLMBlocked,
    LLMError,
    LLMRateLimited,
    LLMResult,
    LLMTimeout,
)

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
    assert answer.prompt_version == "knowledge_answer_v2"
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


def _ask(candidates, fake):
    return generate_answer(
        query_text="q",
        candidates=candidates,
        llm=fake,
        abstention_threshold=None,
        max_tokens=100,
        prompt=PROMPT,
    )


def _assert_unverified(answer, fake, kind: str) -> None:
    """What every rejected-twice scenario ends in: nothing the model wrote, the fixed reply."""
    assert answer.verification_failure == kind
    assert answer.answer_text == UNVERIFIED_ANSWER_TEXT
    assert answer.citations == []
    assert answer.abstained is False  # not an abstention: the evidence was never judged
    assert answer.grounded is False
    assert answer.attempts == MAX_GENERATION_ATTEMPTS
    assert fake.call_count == MAX_GENERATION_ATTEMPTS
    assert answer.grounding_detail["generation"] == {
        "verified": False,
        "attempts": MAX_GENERATION_ATTEMPTS,
        "failure": kind,
    }


def test_invalid_json_twice_gives_the_unverified_reply(caplog) -> None:
    fake = FakeLLMProvider(["this is not { valid json at all"] * 2)

    with caplog.at_level(logging.WARNING):
        answer = _ask(_candidates(1), fake)

    _assert_unverified(answer, fake, FAILURE_MALFORMED_OUTPUT)
    assert "not valid JSON" in caplog.text
    assert "this is not" not in caplog.text  # what the model wrote is never logged


def test_malformed_shape_missing_a_required_field_gives_the_unverified_reply(caplog) -> None:
    malformed = json.dumps({"answer": "text only, no citations field"})
    fake = FakeLLMProvider([malformed] * 2)

    with caplog.at_level(logging.WARNING):
        answer = _ask(_candidates(1), fake)

    _assert_unverified(answer, fake, FAILURE_MALFORMED_OUTPUT)
    assert "expected exactly" in caplog.text


def test_malformed_shape_wrong_type_gives_the_unverified_reply(caplog) -> None:
    malformed = json.dumps({"answer": "ok", "citations": "not-a-list", "sufficient_evidence": True})
    fake = FakeLLMProvider([malformed] * 2)

    with caplog.at_level(logging.WARNING):
        answer = _ask(_candidates(1), fake)

    _assert_unverified(answer, fake, FAILURE_MALFORMED_OUTPUT)
    assert "must be a list" in caplog.text


def test_no_silent_repair_of_malformed_output(caplog) -> None:
    """A JSON object with an extra, unexpected field is rejected outright
    -- never coerced by dropping the extra field silently."""
    extra_field = json.dumps(
        {"answer": "ok", "citations": [], "sufficient_evidence": True, "confidence": 0.9}
    )
    fake = FakeLLMProvider([extra_field] * 2)

    with caplog.at_level(logging.WARNING):
        answer = _ask(_candidates(1), fake)

    _assert_unverified(answer, fake, FAILURE_MALFORMED_OUTPUT)
    assert "expected exactly" in caplog.text
    assert answer.answer_text != "ok"


def test_a_reply_wrapped_in_code_fences_is_not_repaired(caplog) -> None:
    """The model returns the right JSON inside a markdown fence. It is rejected, not unwrapped."""
    candidates = _candidates(1)
    uid = candidates[0].evidence.chunk_uid
    fenced = "```json\n" + valid_json(cited_answer_text(uid), [uid]) + "\n```"
    fake = FakeLLMProvider([fenced] * 2)

    with caplog.at_level(logging.WARNING):
        answer = _ask(candidates, fake)

    _assert_unverified(answer, fake, FAILURE_MALFORMED_OUTPUT)
    assert "not valid JSON" in caplog.text


def test_a_reply_cut_off_at_the_token_limit_is_reported_as_truncated(caplog) -> None:
    cut_off = LLMResult(text='{"answer": "The policy says', input_tokens=10, output_tokens=100, finish_reason="MAX_TOKENS")
    fake = FakeLLMProvider([cut_off, cut_off])

    with caplog.at_level(logging.WARNING):
        answer = _ask(_candidates(1), fake)

    _assert_unverified(answer, fake, FAILURE_MALFORMED_OUTPUT)
    assert "cut off at the token limit" in caplog.text


# --- invalid citations ---------------------------------------------------


def test_a_citation_to_a_never_retrieved_chunk_gives_the_unverified_reply(caplog) -> None:
    fake_uid = "f" * 32
    reply = valid_json(cited_answer_text(fake_uid), [fake_uid])
    fake = FakeLLMProvider([reply] * 2)

    with caplog.at_level(logging.WARNING):
        answer = _ask(_candidates(1), fake)

    _assert_unverified(answer, fake, FAILURE_CITATION_INVALID)
    assert "not in the retrieved evidence" in caplog.text


def test_a_citation_to_a_retrieved_but_unselected_chunk_gives_the_unverified_reply(caplog) -> None:
    candidates = _candidates(12)  # only the first 8 are selected
    unselected_uid = candidates[9].evidence.chunk_uid  # final_rank 10
    reply = valid_json(cited_answer_text(unselected_uid), [unselected_uid])
    fake = FakeLLMProvider([reply] * 2)

    with caplog.at_level(logging.WARNING):
        answer = _ask(candidates, fake)

    _assert_unverified(answer, fake, FAILURE_CITATION_INVALID)
    assert "not selected" in caplog.text


def test_a_citation_coverage_failure_gives_the_unverified_reply(caplog) -> None:
    candidates = _candidates(1)
    uid = candidates[0].evidence.chunk_uid
    reply = valid_json(f"Cited fact [{uid}]. Uncited fact with no marker at all.", [uid])
    fake = FakeLLMProvider([reply] * 2)

    with caplog.at_level(logging.WARNING):
        answer = _ask(candidates, fake)

    _assert_unverified(answer, fake, FAILURE_CITATION_INVALID)
    assert "no citation" in caplog.text


# --- the retry ---------------------------------------------------------


def test_a_bad_first_reply_is_retried_and_a_good_second_reply_is_shown() -> None:
    candidates = _candidates(1)
    uid = candidates[0].evidence.chunk_uid
    good = valid_json(cited_answer_text(uid), [uid])
    fake = FakeLLMProvider(["not json", good])

    answer = _ask(candidates, fake)

    assert answer.verification_failure is None
    assert answer.citation_valid and answer.grounded
    assert answer.attempts == 2
    assert fake.call_count == 2
    assert answer.grounding_detail["generation"] == {"verified": True, "attempts": 2}


def test_a_good_first_reply_is_not_retried_and_adds_nothing_to_the_detail() -> None:
    candidates = _candidates(1)
    uid = candidates[0].evidence.chunk_uid
    fake = FakeLLMProvider([valid_json(cited_answer_text(uid), [uid])])

    answer = _ask(candidates, fake)

    assert answer.attempts == 1
    assert fake.call_count == 1
    assert "generation" not in answer.grounding_detail


def test_tokens_from_every_attempt_are_summed() -> None:
    candidates = _candidates(1)
    uid = candidates[0].evidence.chunk_uid
    good = valid_json(cited_answer_text(uid), [uid])
    fake = FakeLLMProvider(
        [
            LLMResult(text="not json", input_tokens=100, output_tokens=7),
            LLMResult(text=good, input_tokens=100, output_tokens=30),
        ]
    )

    answer = _ask(candidates, fake)

    assert answer.input_tokens == 200
    assert answer.output_tokens == 37


def test_the_unverified_reply_still_counts_the_tokens_that_were_spent() -> None:
    fake = FakeLLMProvider(
        [LLMResult(text="x", input_tokens=50, output_tokens=5), LLMResult(text="y", input_tokens=50, output_tokens=6)]
    )

    answer = _ask(_candidates(1), fake)

    assert (answer.input_tokens, answer.output_tokens) == (100, 11)


def test_a_model_that_declines_is_not_asked_again() -> None:
    fake = FakeLLMProvider([LLMBlocked("blocked", reason="SAFETY")])

    answer = _ask(_candidates(1), fake)

    assert answer.verification_failure == FAILURE_MODEL_BLOCKED
    assert answer.answer_text == UNVERIFIED_ANSWER_TEXT
    assert fake.call_count == 1


@pytest.mark.parametrize(
    "error",
    [LLMRateLimited("slow down", retry_after=30), LLMTimeout("too slow"), LLMError("down")],
)
def test_a_provider_failure_is_not_softened_into_a_reply(error) -> None:
    fake = FakeLLMProvider([error])

    with pytest.raises(type(error)):
        _ask(_candidates(1), fake)

    assert fake.call_count == 1  # and never retried here: the adapter already did


def test_the_provider_is_asked_for_json_in_the_answer_shape() -> None:
    candidates = _candidates(1)
    uid = candidates[0].evidence.chunk_uid
    fake = FakeLLMProvider([valid_json(cited_answer_text(uid), [uid])])

    _ask(candidates, fake)

    assert fake.last_json_schema == ANSWER_JSON_SCHEMA
    assert set(ANSWER_JSON_SCHEMA["required"]) == {"answer", "citations", "sufficient_evidence"}


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


def test_sufficient_evidence_false_with_leftover_citations_gives_the_unverified_reply(caplog) -> None:
    """The model claiming insufficient evidence while still citing
    something is an inconsistent, invalid response -- rejected, not
    silently repaired by dropping the citations."""
    candidates = _candidates(1)
    uid = candidates[0].evidence.chunk_uid
    reply = valid_json(f"Partial fact [{uid}].", [uid], sufficient_evidence=False)
    fake = FakeLLMProvider([reply] * 2)

    with caplog.at_level(logging.WARNING):
        answer = _ask(candidates, fake)

    _assert_unverified(answer, fake, FAILURE_CITATION_INVALID)
    assert "must carry no citations" in caplog.text


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
