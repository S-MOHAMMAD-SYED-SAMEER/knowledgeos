"""The demo's generation provider: real, human-verified answer fixtures,
replayed as deterministic, marker-compliant JSON through the unmodified
real generation pipeline.

Not a fake in the sense `FakeLLMProvider` is ("nothing in the application
ever selects this provider; only a test does"). Every claim and every
citation this provider returns is the exact, hand-verified content of
`demo/fixtures/answers.yaml` -- P1's own curated, real answer fixture,
never invented here. The only thing this module adds is mechanical:
`answers.yaml`'s `answer_text` values are plain prose written for direct
display, with no inline `[<chunk_uid>]` markers, and
`app/generation/citations.py`'s deterministic validation gate -- frozen,
unmodified, and still the sole authority here -- requires every sentence
of a non-abstained answer to carry one. This provider closes that gap the
same way `tests/generation_fixtures.py::cited_answer_text` already proves
works against the real gate: split the fixture's own `answer_text` into
sentences with the real `split_sentences` function, append the fixture's
own declared citation marker(s) to every sentence, and change not one
word of the fixture's actual prose.

Because every marker comes from the fixture's own `citations` list and
every sentence gets at least one, `generate_answer()`'s real
`validate_citations()` -- coverage, declared/inline agreement, retrieved
and selected membership -- runs against this output exactly as it would
against real model output, and genuinely passes or fails on its own
merits. Nothing here bypasses it; this module never imports
`app.generation`'s validation, grounding, or orchestration code at all,
only the plain-function sentence splitter it already exposes.
"""

import hashlib
import json
import re
from pathlib import Path

import yaml

from app.generation.citations import split_sentences
from app.providers.llm import LLMError, LLMResult

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
ANSWERS_FIXTURE_PATH = FIXTURES_DIR / "answers.yaml"

# Distinct from any real vendor's model name: a log line or an audit
# reading this value should never conclude a live model ran.
DEMO_LLM_MODEL_NAME = "demo-fixture-replay (no live model)"

# The exact shape `app/generation/generator.py::generate_answer` builds
# the user prompt with -- frozen, unmodified. Extracting the question
# back out of it is how this provider identifies which fixture entry to
# replay without `app/generation/` ever needing to change or know this
# provider exists.
_QUESTION_PATTERN = re.compile(r"^Question:\n(.*?)\n\nEvidence:\n", re.DOTALL)

# The exact attribute `app/generation/prompt.py::serialize_evidence` emits
# per selected chunk -- the same one `tests/generation_fixtures.py`'s
# `AutoCitingLLMProvider` already reads out of the prompt it is given.
_CHUNK_UID_ATTR = re.compile(r'chunk_uid="([0-9a-f]{32})"')


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class DemoLLMProvider:
    """Real, curated demo answers, replayed as marker-compliant JSON --
    for exactly the five flagship questions `demo/fixtures/answers.yaml`
    pins."""

    def __init__(self, fixture_path: Path = ANSWERS_FIXTURE_PATH) -> None:
        self._fixture_path = fixture_path
        self._entries: dict[str, dict] | None = None

    @property
    def model_name(self) -> str:
        return DEMO_LLM_MODEL_NAME

    def complete(self, *, system: str, user: str, max_tokens: int) -> LLMResult:
        del max_tokens  # the fixture's answer never depends on it

        match = _QUESTION_PATTERN.match(user)
        if not match:
            raise LLMError(
                "the demo provider could not find a question in the "
                "prompt -- the generation orchestration's prompt format "
                "may have changed"
            )
        question = match.group(1)

        entry = self._entries_by_hash().get(_sha256(question))
        if entry is None:
            raise LLMError(
                "no precomputed demo answer exists for this question -- "
                "DemoLLMProvider only serves the five pinned flagship "
                "questions, never arbitrary input"
            )

        available_uids = frozenset(_CHUNK_UID_ATTR.findall(user))
        missing = [uid for uid in entry["citations"] if uid not in available_uids]
        if missing:
            raise LLMError(
                "the demo fixture's declared citation(s) are not present "
                "in the evidence this call was actually shown -- the "
                "retrieval/reranking result for this query no longer "
                "matches what the fixture was written against"
            )

        answer_text = _mark_up(entry["answer_text"], entry["citations"])
        payload = {
            "answer": answer_text,
            "citations": list(entry["citations"]),
            "sufficient_evidence": entry["sufficient_evidence"],
        }
        text = json.dumps(payload)

        return LLMResult(
            text=text,
            input_tokens=len(system.split()) + len(user.split()),
            output_tokens=len(text.split()),
        )

    def _entries_by_hash(self) -> dict[str, dict]:
        if self._entries is not None:
            return self._entries

        raw = yaml.safe_load(self._fixture_path.read_text(encoding="utf-8"))
        self._entries = {_sha256(entry["question"]): entry for entry in raw}
        return self._entries


def _mark_up(answer_text: str, citations: list[str]) -> str:
    """Every sentence of `answer_text`, word for word, with every one of
    `citations`' markers inserted just before its closing punctuation --
    never a new claim, never a citation absent from the fixture's own
    declared list. An abstaining entry (`citations == []`) is returned
    untouched, with no markers added, the only shape
    `abstention_flag_consistent` accepts.

    Before the closing punctuation, not after: `split_sentences` (the
    same real function `app/generation/citations.py` re-applies to this
    provider's own output during validation) treats `. [` as a sentence
    boundary -- a marker placed after the period would be split onto its
    own "sentence" and orphaned from the one it was meant to cover.
    Placing it before the period keeps the real period the last
    character, which is the boundary the real function is actually
    looking for.
    """
    if not citations:
        return answer_text

    markers = " " + " ".join(f"[{uid}]" for uid in citations)
    sentences = split_sentences(answer_text)
    marked = [
        f"{sentence[:-1]}{markers}{sentence[-1]}"
        if sentence and sentence[-1] in ".!?"
        else f"{sentence}{markers}"
        for sentence in sentences
    ]
    return " ".join(marked)


__all__ = ["DEMO_LLM_MODEL_NAME", "ANSWERS_FIXTURE_PATH", "DemoLLMProvider"]
