"""The frozen gold set, and the hash that says it has not moved.

"The gold" is everything under `evals/fixtures/`: the 52 questions with their expected chunks and
abstention flags (`questions/*.yaml`), the dev/test split (`splits.yaml`), the provenance record
(`PROVENANCE.yaml`), and the fixture corpus the expected chunk ids refer to (`knowledge_base/`).

`python -m evals.gold` prints the digest. `docs/EVAL.md` records it as "frozen", and
`tests/test_gold_frozen.py` fails if the files no longer hash to it. Changing the gold after a model
has been run against it is exactly what that test exists to catch, so a deliberate change means a new
version of the gold, a new digest in `docs/EVAL.md`, and a note saying why.

Line endings are normalised before hashing, so a Windows checkout and a Linux one agree.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def gold_files() -> list[Path]:
    """Every file in the gold set, in a fixed order."""
    # Sorted by the posix path as a plain string: `Path` ordering is case-insensitive on Windows and
    # case-sensitive on Linux, which would hash the same files in a different order on each.
    found = (path for path in FIXTURES.rglob("*") if path.is_file() and "__pycache__" not in path.parts)
    return sorted(found, key=lambda path: path.relative_to(FIXTURES).as_posix())


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def gold_digest() -> tuple[str, dict[str, str]]:
    """The combined sha256 of the gold set, and the sha256 of each file in it."""
    per_file = {path.relative_to(FIXTURES).as_posix(): file_digest(path) for path in gold_files()}
    combined = hashlib.sha256("".join(f"{name}\0{digest}\n" for name, digest in per_file.items()).encode()).hexdigest()
    return combined, per_file


def provenance() -> dict:
    """`dataset_version` and `label` from `fixtures/PROVENANCE.yaml`."""
    import yaml

    data = yaml.safe_load((FIXTURES / "PROVENANCE.yaml").read_text(encoding="utf-8"))
    return {"dataset_version": data["dataset_version"], "label": data["label"]}


def main() -> None:
    combined, per_file = gold_digest()
    for name, digest in per_file.items():
        print(f"{digest}  {name}")
    print(f"\ngold-sha256: {combined}")


if __name__ == "__main__":
    main()
