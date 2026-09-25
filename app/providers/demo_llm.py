"""The visitor-facing demo's generation provider.

Distinct from `app/providers/fake_llm.py` on purpose: `FakeLLMProvider` is
explicitly test-only ("nothing in the application ever selects this
provider — only a test does"). `DemoLLMProvider` is the opposite: it is the
provider `app/api/query.py::llm_provider` selects for real, whenever
`Settings.demo_mode` is true — a real, application-selected generation
path, never a test double.

It never calls a real model. It matches the visitor's question, after the
same normalization `app.parsing.normalize` already applies, against the
curated set in `app/generation/demo_scenarios.py`, and returns the matching
hand-verified answer as this provider's `complete()` output. A question
that does not match raises `DemoAnswerNotAvailable` rather than inventing
anything — `app/api/query.py` is what turns that into an honest, distinct
"no verified demo answer" response; this provider's job stops at refusing
to guess.

This module imports nothing from `app.providers.gemini_llm` and nothing
that would let it reach a real model — `tests/test_demo_llm_provider.py`
asserts this structurally, the same way `tests/test_llm_provider.py`
already asserts no application module other than `gemini_llm.py` imports
the Anthropic-shaped... er, Gemini SDK.
"""

import json
import re

from app.generation.demo_scenarios import DemoScenario, find_scenario
from app.providers.llm import LLMResult

DEMO_MODEL_NAME = "demo-fixture"

# `app.generation.generator.generate_answer` always builds its `user` prompt
# as `f"Question:\n{query_text}\n\nEvidence:\n{evidence_block}"` — this is
# the one place that format is depended on outside that function itself.
# `tests/test_demo_llm_provider.py::test_question_extraction_matches_the_
# real_prompt_format` guards this coupling so a future change to that
# f-string is caught here rather than silently misrouting every question.
_QUESTION_PATTERN = re.compile(r"^Question:\n(.*?)\n\nEvidence:\n", re.DOTALL)


class DemoAnswerNotAvailable(Exception):
    """The visitor's question does not match a curated demo scenario.

    Deliberately not an `LLMError` subclass: `app/api/query.py` catches
    this separately from `LLMError` so an unmatched question gets the
    honest, distinct "no verified demo answer" response `M1` specifies —
    not the same 503 a real provider outage would return.
    """


def _extract_question(user: str) -> str:
    match = _QUESTION_PATTERN.match(user)
    if match is None:
        # Same fail-closed instinct as the rest of this codebase: an
        # unrecognised prompt shape is treated as "no scenario matches",
        # never as a reason to guess at a question.
        return ""
    return match.group(1)


class DemoLLMProvider:
    """Deterministic, hand-verified answers for a curated question set.

    Implements the same `LLMProvider` protocol every other adapter does
    (`model_name`, `complete`) so it can be injected at exactly the seam
    `app/api/query.py::llm_provider` already exposes — no second query
    implementation, no new interface.
    """

    model_name = DEMO_MODEL_NAME

    def complete(self, *, system: str, user: str, max_tokens: int) -> LLMResult:
        del max_tokens  # Matching depends only on the question.

        question_text = _extract_question(user)
        scenario = find_scenario(question_text)
        if scenario is None:
            raise DemoAnswerNotAvailable(
                "no curated demo scenario matches this question"
            )

        text = _scenario_json(scenario)
        return LLMResult(
            text=text,
            input_tokens=len(user.split()) + len(system.split()),
            output_tokens=len(text.split()),
        )


def _scenario_json(scenario: DemoScenario) -> str:
    return json.dumps(
        {
            "answer": scenario.answer_text,
            "citations": list(scenario.citations),
            "sufficient_evidence": scenario.sufficient_evidence,
        }
    )


__all__ = ["DEMO_MODEL_NAME", "DemoAnswerNotAvailable", "DemoLLMProvider"]
