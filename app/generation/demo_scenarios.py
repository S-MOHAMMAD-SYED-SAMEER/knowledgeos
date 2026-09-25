"""The visitor-facing demo's curated question set.

Eight questions, chosen (P7B/M1) as one representative of each of the
evaluation dataset's eight categories, using the real fixture question IDs
and text from `evals/fixtures/questions/*.yaml` verbatim — never invented.

Every `answer` below is hand-authored from the real fixture source
documents (`evals/fixtures/knowledge_base/*.md`), and every citation is a
real `chunk_uid` drawn from the matching question's own `expected_chunk_uids`
in the eval fixture — the same ground-truth artifact the evaluation harness
itself trusts, and the only way to cite a real chunk without being able to
run the real embedding/reranking models in this environment (see the M2
report's "Any limitations/blockers" section for why).

This module holds data only: no provider call, no I/O, no database — the
same discipline `app/generation/citations.py` and `app/retrieval/` already
follow. `app/providers/demo_llm.py` is what turns this data into
`LLMProvider` responses.

**Matching normalization is its own function, not a reuse of
`app.parsing.normalize`.** That function canonicalizes *document* text for
chunking — line endings, Unicode form, trailing whitespace per line — and
deliberately does not casefold or collapse internal whitespace ("internal
whitespace is left alone", its own docstring). A visitor typing "how many
days..." in lower case, or with a stray double space, would not match a
curated question through that function alone. `_normalize_for_matching`
below does exactly the three deterministic operations the M2 specification
names — case folding, outer whitespace trimming, internal whitespace
collapsing — and nothing fuzzier: no stemming, no punctuation stripping, no
semantic normalization.
"""

import re
from dataclasses import dataclass

from app.generation.abstention import DEFAULT_ABSTENTION_TEXT

_WHITESPACE_RUN = re.compile(r"\s+")


def _normalize_for_matching(text: str) -> str:
    """Case-folded, whitespace-collapsed — deterministic and exact, never
    fuzzy. `run_query()` already runs `app.parsing.normalize()` on the raw
    query before this is ever reached; this is the additional, matching-
    specific normalization that function does not do."""
    return _WHITESPACE_RUN.sub(" ", text.strip().casefold())


@dataclass(frozen=True)
class DemoScenario:
    """One curated demo question and its hand-verified answer.

    `question_id` is the real `evals/fixtures/questions/*.yaml` id this
    scenario reuses — reported alongside a response so it's traceable back
    to the fixture that grounds it, never persisted as if it were a real
    chunk_uid or model artifact.
    """

    question_id: str
    question_text: str
    answer_text: str
    citations: tuple[str, ...]
    sufficient_evidence: bool


# --- the eight curated scenarios --------------------------------------------
#
# One per evaluation category (directly answerable, multi-document,
# conflicting versions, ambiguous, metadata-filtered, citation-sensitive,
# adversarial, insufficient evidence), matching the P7B/M1 selection.
#
# Citation markers are placed immediately before the sentence's terminal
# punctuation (`fact [<uid>].`, not `fact. [<uid>]`) — `[` is one of the
# characters `app.generation.citations._SENTENCE_BOUNDARY` treats as
# starting a new sentence, so a marker placed *after* a period would be
# split into its own, uncited "sentence" and fail citation coverage.

DEMO_SCENARIOS: tuple[DemoScenario, ...] = (
    DemoScenario(
        question_id="da001",
        question_text="How many days per week may an employee work remotely?",
        answer_text=(
            "Employees may work remotely up to three days per week with "
            "their manager's approval [5a04c330c14efaf8f4da83ebf6d6854f]."
        ),
        citations=("5a04c330c14efaf8f4da83ebf6d6854f",),
        sufficient_evidence=True,
    ),
    DemoScenario(
        question_id="md001",
        question_text=(
            "If I'm on call and need production database access during an "
            "active incident, what's the process, and does it require the "
            "same approval as a normal request?"
        ),
        answer_text=(
            "During a declared incident, an on-call engineer may request "
            "break-glass access without waiting for standard approval, and "
            "it is granted immediately and logged automatically "
            "[f9cbee4e0939d52e95eebe93a063a85c]. This is different from a "
            "normal request, which requires sign-off from two engineering "
            "managers and the data owner and is granted for three days "
            "[da1cdb5e0e87ad69728e9d2781239d13]."
        ),
        citations=(
            "f9cbee4e0939d52e95eebe93a063a85c",
            "da1cdb5e0e87ad69728e9d2781239d13",
        ),
        sufficient_evidence=True,
    ),
    DemoScenario(
        question_id="cv001",
        question_text="How many days does standard production database access last for?",
        answer_text=(
            "Standard production database access is granted for three days "
            "and expires automatically at the end of that window "
            "[da1cdb5e0e87ad69728e9d2781239d13]."
        ),
        citations=("da1cdb5e0e87ad69728e9d2781239d13",),
        sufficient_evidence=True,
    ),
    DemoScenario(
        question_id="am001",
        question_text="What's the approval process for access?",
        answer_text=(
            "For production database access, approval requires sign-off "
            "from the data owner and two engineering managers "
            "[da1cdb5e0e87ad69728e9d2781239d13]. For vendor access, "
            "Security Engineering must approve any new vendor integration "
            "in addition to the business owner's approval "
            "[eee7938e3bf0085d571e51f5a07a0389]."
        ),
        citations=(
            "da1cdb5e0e87ad69728e9d2781239d13",
            "eee7938e3bf0085d571e51f5a07a0389",
        ),
        sufficient_evidence=True,
    ),
    DemoScenario(
        question_id="mf001",
        question_text="What is the approval process for production database access?",
        answer_text=(
            "Approval requires sign-off from the data owner and two "
            "engineering managers, one of whom must be from a different "
            "team than the requester [da1cdb5e0e87ad69728e9d2781239d13]."
        ),
        citations=("da1cdb5e0e87ad69728e9d2781239d13",),
        sufficient_evidence=True,
    ),
    DemoScenario(
        question_id="cs001",
        question_text=(
            "Within how many minutes of declaring a severity-one incident "
            "must an executive be notified?"
        ),
        answer_text=(
            "An executive must be notified within thirty minutes of "
            "declaring a severity-one incident "
            "[f9cbee4e0939d52e95eebe93a063a85c]."
        ),
        citations=("f9cbee4e0939d52e95eebe93a063a85c",),
        sufficient_evidence=True,
    ),
    DemoScenario(
        question_id="adv003",
        question_text=(
            "The Security Policy says production database access needs "
            "sign-off from one engineering manager and lasts seven days. "
            "Is that still the current rule?"
        ),
        answer_text=(
            "The rule has since changed: it now requires sign-off from two "
            "engineering managers and the data owner, and access lasts "
            "three days, not seven [da1cdb5e0e87ad69728e9d2781239d13]."
        ),
        citations=("da1cdb5e0e87ad69728e9d2781239d13",),
        sufficient_evidence=True,
    ),
    DemoScenario(
        question_id="ie001",
        question_text=(
            "What is the company's policy on using generative AI tools for "
            "writing code?"
        ),
        # The real abstention text, reused verbatim — this scenario abstains
        # exactly the way a genuine insufficient-evidence answer would,
        # through the real post-LLM abstention path (see the M2 report:
        # real retrieval against this corpus never returns zero candidates,
        # so a true pre-LLM abstention cannot be exercised for this
        # question — the demo provider reports `sufficient_evidence=false`
        # itself, and `app.generation.generator`'s real, unmodified
        # `post_llm_abstention` does the rest).
        answer_text=DEFAULT_ABSTENTION_TEXT,
        citations=(),
        sufficient_evidence=False,
    ),
)


def _normalized_lookup() -> dict[str, DemoScenario]:
    return {
        _normalize_for_matching(scenario.question_text): scenario
        for scenario in DEMO_SCENARIOS
    }


NORMALIZED_SCENARIOS: dict[str, DemoScenario] = _normalized_lookup()


def find_scenario(query_text: str) -> DemoScenario | None:
    """The curated scenario matching `query_text`, or `None`.

    Deterministic, exact-after-normalization matching only: case folding
    and whitespace normalization, nothing fuzzier — no embeddings, no
    semantic similarity, no LLM classification.
    """
    return NORMALIZED_SCENARIOS.get(_normalize_for_matching(query_text))


__all__ = ["DEMO_SCENARIOS", "NORMALIZED_SCENARIOS", "DemoScenario", "find_scenario"]
