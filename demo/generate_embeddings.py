"""Generates `demo/fixtures/embeddings.json` from the real BGE model.

Run once, in an environment with `BAAI/bge-small-en-v1.5` available in the
local sentence-transformers cache::

    python -m demo.generate_embeddings

The demo corpus is exactly the evaluation fixture corpus
(`evals/fixtures/knowledge_base/manifest.yaml`), chunked the same way the
real ingestion pipeline chunks it -- so every chunk_uid produced here is
one the real pipeline would also produce, and demo seeding
(`demo/seed.py`) can reuse `evals.retrieval.corpus.ensure_corpus_seeded`
unchanged. Both versions of `access-sop` are included (eleven versions
across ten documents), which is what makes the corpus exactly 33 chunks
rather than the 30 an active-versions-only corpus would give -- the extra
three are what let the conflicting-versions flagship scenario demonstrate
a real superseded chunk existing, and not being served by default
retrieval.

A fixed handful of flagship demo query texts are embedded alongside the
chunks, under the same real model in the same run, so the
flagship-scenario tests in `tests/test_demo_fixtures.py` never need a
fake or a live-called vector either.
"""

import hashlib
import json
import uuid
from dataclasses import dataclass
from pathlib import Path

from app.chunking import chunk_document
from app.ingestion.pipeline import BLOCK_SEPARATOR
from app.ingestion.validation import extension_of
from app.parsing import normalize, parser_for
from evals.retrieval.corpus import DEFAULT_MANIFEST, load_manifest

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
OUTPUT_PATH = FIXTURES_DIR / "embeddings.json"

EXPECTED_CHUNK_COUNT = 33

# The five scenarios that demonstrate the product goal chain end to end:
#   retrieved correctly -> ranked correctly -> answered from evidence ->
#   cited correctly -> abstained when evidence is insufficient.
# Every id/text pair is reused verbatim from `evals/fixtures/questions/`'s
# own frozen, already-verified question set rather than invented, so the
# "expected" side of each flagship test is ground truth this project
# already trusts.
FLAGSHIP_QUERIES: tuple[tuple[str, str], ...] = (
    ("da001", "How many days per week may an employee work remotely?"),
    (
        "cs001",
        "Within how many minutes of declaring a severity-one incident "
        "must an executive be notified?",
    ),
    ("cv001", "How many days does standard production database access last for?"),
    (
        "md001",
        "If I'm on call and need production database access during an "
        "active incident, what's the process, and does it require the "
        "same approval as a normal request?",
    ),
    (
        "ie001",
        "What is the company's policy on using generative AI tools for "
        "writing code?",
    ),
)


@dataclass(frozen=True)
class ManifestChunk:
    """One chunk the real pipeline would produce for one manifest version."""

    document_key: str
    file: str
    version_id: uuid.UUID
    sequence: int
    chunk_uid: str
    text: str


def iter_manifest_chunks(manifest_path: Path = DEFAULT_MANIFEST) -> list[ManifestChunk]:
    """Every chunk the real ingestion pipeline would produce for every
    document version pinned in the evaluation fixture manifest.

    Parsing, normalization and chunking here call the exact same
    `app.parsing` / `app.chunking` functions `app/ingestion/pipeline.py`
    calls (`parse_and_normalize` / `build_chunks`), just without a
    database session or a `DocumentVersion` row in between -- so the
    resulting chunk_uids are identical to what seeding the manifest
    through the real pipeline produces.
    """
    manifest = load_manifest(manifest_path)
    base_dir = manifest_path.parent
    chunks: list[ManifestChunk] = []

    for entry in manifest["documents"]:
        for version_entry in entry["versions"]:
            version_id = uuid.UUID(str(version_entry["version_id"]))
            file_name = version_entry["file"]
            parser = parser_for(extension_of(file_name))
            with open(base_dir / file_name, "rb") as handle:
                parsed = parser.parse(handle)

            pieces: list[str] = []
            blocks = []
            offsets: list[tuple[int, int]] = []
            cursor = 0
            for block in parsed.blocks:
                body = normalize(block.text)
                if not body:
                    continue
                if pieces:
                    cursor += len(BLOCK_SEPARATOR)
                pieces.append(body)
                offsets.append((cursor, cursor + len(body)))
                blocks.append(block)
                cursor += len(body)
            text = BLOCK_SEPARATOR.join(pieces)

            document_chunks = chunk_document(
                document_version_id=version_id,
                text=text,
                blocks=tuple(blocks),
                block_offsets=tuple(offsets),
                size=manifest["chunk_size_tokens"],
                overlap=manifest["chunk_overlap_tokens"],
            )
            for chunk in document_chunks:
                chunks.append(
                    ManifestChunk(
                        document_key=entry["key"],
                        file=file_name,
                        version_id=version_id,
                        sequence=chunk.sequence,
                        chunk_uid=chunk.chunk_uid,
                        text=chunk.text,
                    )
                )

    return chunks


def sha256_of(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def generate(output_path: Path = OUTPUT_PATH) -> None:
    """Embed the demo corpus and the flagship queries with the real
    local BGE model, and write the fixture the demo replays offline."""
    from app.providers.bge import BgeEmbeddingProvider
    from app.providers.embeddings import DIMENSIONS, MODEL_NAME, check_shape

    manifest = load_manifest(DEFAULT_MANIFEST)
    manifest_chunks = iter_manifest_chunks()
    if len(manifest_chunks) != EXPECTED_CHUNK_COUNT:
        raise RuntimeError(
            f"the demo corpus produced {len(manifest_chunks)} chunk(s), "
            f"expected exactly {EXPECTED_CHUNK_COUNT} -- the manifest or "
            f"the pinned chunk configuration has changed"
        )

    provider = BgeEmbeddingProvider()

    chunk_texts = [c.text for c in manifest_chunks]
    chunk_vectors = provider.embed(chunk_texts)
    check_shape(chunk_vectors, chunk_texts, DIMENSIONS)

    query_texts = [text for _, text in FLAGSHIP_QUERIES]
    query_vectors = provider.embed(query_texts)
    check_shape(query_vectors, query_texts, DIMENSIONS)

    payload = {
        "model_name": MODEL_NAME,
        "dimensions": DIMENSIONS,
        "manifest_chunk_size_tokens": manifest["chunk_size_tokens"],
        "manifest_chunk_overlap_tokens": manifest["chunk_overlap_tokens"],
        "chunks": [
            {
                "chunk_uid": chunk.chunk_uid,
                "document_key": chunk.document_key,
                "file": chunk.file,
                "version_id": str(chunk.version_id),
                "sequence": chunk.sequence,
                "text_sha256": sha256_of(chunk.text),
                "embedding": vector,
            }
            for chunk, vector in zip(manifest_chunks, chunk_vectors, strict=True)
        ],
        "queries": [
            {
                "id": query_id,
                "text": text,
                "text_sha256": sha256_of(text),
                "embedding": vector,
            }
            for (query_id, text), vector in zip(
                FLAGSHIP_QUERIES, query_vectors, strict=True
            )
        ],
    }

    output_path.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    generate()
