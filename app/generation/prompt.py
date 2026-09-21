"""Loading, versioning, and evidence-serializing the generation prompt.

The prompt is a file, never a string literal in Python — the specification
is explicit: "Never inline, never unversioned." Its version is its filename
stem, `knowledge_answer_v1`, and a content hash travels alongside it as an
independently-computable proof that the file on disk is the one that
produced a given answer — the same "recompute rather than trust" instinct
this project already applies to `chunk_uid`. A content change without a
version bump changes the hash even though the version string did not,
which is exactly the drift this exists to catch.

Evidence serialization is this module's other job: turning selected chunks
into the delimited, attributed `<evidence>` blocks the prompt instructs the
model to read as data, never as instructions.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path

from app.generation.evidence import SelectedEvidence

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
DEFAULT_PROMPT_PATH = PROMPT_DIR / "knowledge_answer_v1.md"


class PromptError(RuntimeError):
    """The prompt file could not be loaded."""


@dataclass(frozen=True)
class LoadedPrompt:
    """One versioned prompt, read from disk."""

    version: str
    content_hash: str
    text: str


def load_prompt(path: Path = DEFAULT_PROMPT_PATH) -> LoadedPrompt:
    """Read the versioned prompt file.

    `version` is the filename stem (`knowledge_answer_v1`); `content_hash`
    is a sha256 prefix of the file's exact bytes, so a content edit is
    detectable independently of whether the filename was ever bumped.
    """
    if not path.exists():
        raise PromptError(f"prompt file not found: {path.name}")
    raw = path.read_bytes()
    return LoadedPrompt(
        version=path.stem,
        content_hash=hashlib.sha256(raw).hexdigest()[:16],
        text=raw.decode("utf-8"),
    )


def serialize_evidence(selected: list[SelectedEvidence]) -> str:
    """The selected chunks, as delimited, attributed `<evidence>` blocks —
    one per chunk, `chunk_uid` as an attribute, the chunk's own text last
    inside the element, ordered by `final_rank`. Unselected chunks are
    never serialized: they were retrieved but did not reach the model, so
    a citation to one of them is invalid by rule 3, not merely absent from
    the prompt by omission.
    """
    blocks = [
        _one_block(item)
        for item in sorted(selected, key=lambda s: s.candidate.final_rank)
        if item.selected
    ]
    return "\n\n".join(blocks)


def _one_block(item: SelectedEvidence) -> str:
    evidence = item.candidate.evidence
    return (
        f'<evidence chunk_uid="{evidence.chunk_uid}" '
        f'document="{_escape(evidence.document_title)}" '
        f'version="{evidence.version_number}">\n'
        f"{evidence.text}\n"
        f"</evidence>"
    )


def _escape(value: str) -> str:
    """Minimal XML attribute escaping — this project has no XML dependency
    and needs none for two characters."""
    return value.replace("&", "&amp;").replace('"', "&quot;")


__all__ = [
    "DEFAULT_PROMPT_PATH",
    "PROMPT_DIR",
    "LoadedPrompt",
    "PromptError",
    "load_prompt",
    "serialize_evidence",
]
