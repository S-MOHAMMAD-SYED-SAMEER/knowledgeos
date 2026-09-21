"""Generation: turning milestone 6's reranked evidence into a grounded,
cited answer — or an honest abstention.

`app/generation/citations.py` contains no provider calls and no I/O beyond
the database, per the specification's own layering rule (§4), exactly like
`app/retrieval/` and `app/chunking/`. Everything else here is orchestration:
`evidence.py` selects which reranked chunks reach the model, `prompt.py`
loads the versioned prompt and serializes evidence into it, `generator.py`
calls the provider and parses/validates its output, `abstention.py` decides
whether to answer at all, `grounding.py` reports the deterministic checks
(and honestly reports the semantic layer as unexercised — see its own
docstring), and `persistence.py` writes the result.

The generation provider is Google Gemini, not Anthropic — an explicit,
documented deviation from the specification's stack wording (§3, §17); see
`app/providers/gemini_llm.py` and the README's milestone 8 section for the
full explanation. Nothing in this package is Gemini-shaped: it is written
against `app.providers.llm.LLMProvider` alone.
"""

from app.generation.provider import get_llm_provider, reset_llm_provider

__all__ = ["get_llm_provider", "reset_llm_provider"]
