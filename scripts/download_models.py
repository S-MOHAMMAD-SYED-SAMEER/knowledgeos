"""Put the two local models in the Hugging Face cache, once.

    python scripts/download_models.py             # download both
    python scripts/download_models.py --dry-run   # say what would be downloaded and where

KnowledgeOS loads BGE (embeddings) and the cross-encoder (reranking) with
`local_files_only=True`: the application never downloads anything while it runs, so a
machine that has not been given the models fails every ingest and query with "the
embedding model is unavailable". This script is the one place that fetches them. It is
outside `app/` on purpose, and the model names come from the providers themselves so the
two can never disagree.

The cache is Hugging Face's usual one (`~/.cache/huggingface/hub`, or `HF_HOME` /
`HF_HUB_CACHE` if you set them). About 200 MB in all.
"""

import argparse
import sys

from app.providers.cross_encoder import MODEL_NAME as RERANK_MODEL_NAME
from app.providers.embeddings import MODEL_NAME as EMBEDDING_MODEL_NAME

MODELS = (EMBEDDING_MODEL_NAME, RERANK_MODEL_NAME)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download the two local models into the Hugging Face cache.")
    parser.add_argument("--dry-run", action="store_true", help="only print what would be downloaded")
    args = parser.parse_args(argv)

    from huggingface_hub import constants, snapshot_download

    print(f"Hugging Face cache: {constants.HF_HUB_CACHE}")
    for model in MODELS:
        if args.dry_run:
            print(f"  would download {model}")
            continue
        print(f"  downloading {model} ...")
        try:
            path = snapshot_download(repo_id=model)
        except Exception as exc:  # noqa: BLE001 - a network problem is the usual cause
            print(f"  FAILED: {model} ({type(exc).__name__}). Check the network, then run this again.", file=sys.stderr)
            return 1
        print(f"  ok: {path}")

    if not args.dry_run:
        print("Done. The application can now load both models with no network.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
