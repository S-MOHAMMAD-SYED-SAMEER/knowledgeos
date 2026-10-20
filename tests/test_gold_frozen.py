"""The gold set is frozen: it hashes to the digest recorded in docs/EVAL.md, it carries a provenance entry for every
question, and Gold Review 2 is applied exactly as decided.

If `test_the_gold_has_not_moved` fails, something under `evals/fixtures/` changed. If that was deliberate it is a new
gold version: change `PROVENANCE.yaml`'s `dataset_version`, put the new digest in docs/EVAL.md, and say why there.
"""

import re
from pathlib import Path

import yaml

from evals.gold import FIXTURES, gold_digest, gold_files, provenance
from evals.retrieval.questions import load_question_set

REPO = Path(__file__).resolve().parents[1]
EVAL_MD = (REPO / "docs" / "EVAL.md").read_text(encoding="utf-8")


def _recorded_digest() -> str:
    match = re.search(r"^gold-sha256: ([0-9a-f]{64})$", EVAL_MD, re.MULTILINE)
    assert match, "docs/EVAL.md does not record a gold-sha256"
    return match.group(1)


def test_the_gold_has_not_moved() -> None:
    combined, _ = gold_digest()
    assert combined == _recorded_digest(), (
        "evals/fixtures/ no longer hashes to the frozen digest in docs/EVAL.md; "
        "a change to the gold after a model run is a new gold version"
    )


def test_eval_md_says_frozen_and_names_the_version_and_the_label() -> None:
    assert "**Status: frozen.**" in EVAL_MD
    assert f"gold-version: {provenance()['dataset_version']}" in EVAL_MD
    assert provenance()["label"] in EVAL_MD.replace("> **", "").replace("**", "")


def test_the_digest_covers_every_fixture_file_and_ignores_line_endings(tmp_path: Path) -> None:
    from evals.gold import file_digest

    names = {path.relative_to(FIXTURES).as_posix() for path in gold_files()}
    assert "PROVENANCE.yaml" in names and "splits.yaml" in names
    assert any(name.startswith("questions/") for name in names)
    assert any(name.startswith("knowledge_base/") for name in names)

    lf = tmp_path / "a.yaml"
    crlf = tmp_path / "b.yaml"
    lf.write_bytes(b"a: 1\nb: 2\n")
    crlf.write_bytes(b"a: 1\r\nb: 2\r\n")
    assert file_digest(lf) == file_digest(crlf)


def test_every_question_has_a_provenance_entry_with_the_original_authorship() -> None:
    questions = load_question_set()
    record = yaml.safe_load((FIXTURES / "PROVENANCE.yaml").read_text(encoding="utf-8"))
    assert record["dataset_version"] == "gold-v2"
    assert set(record["questions"]) == {q.id for q in questions}
    for qid, entry in record["questions"].items():
        kinds = [event["kind"] for event in entry["events"]]
        assert kinds[0] == "claude-written", qid
        assert set(kinds) <= {"claude-written", "changed-by-sameer", "decided-by-sameer", "fact-checked"}, qid


def test_gold_review_2_is_applied_exactly_as_decided() -> None:
    by_id = {q.id: q for q in load_question_set()}
    record = yaml.safe_load((FIXTURES / "PROVENANCE.yaml").read_text(encoding="utf-8"))["questions"]

    def kinds(qid: str) -> list[str]:
        return [event["kind"] for event in record[qid]["events"]]

    # Reworded
    assert by_id["da005"].text == (
        "What is the typical contractual window for a vendor to notify the company of a security incident "
        "on the vendor's side?"
    )
    assert by_id["mf002"].text == (
        "How long are database backups retained before older backups are automatically deleted?"
    )
    # Expected chunks reduced to the one the decision names
    assert by_id["md003"].expected_chunk_uids == frozenset({"ac735f6387ca659b3a009e78d98f0118"})  # Data Retention chunk 0
    assert by_id["md004"].expected_chunk_uids == frozenset({"1520a2137d9557a94fc05784da189fc0"})  # Handbook chunk 1
    # Kept as deliberate tests: not abstain, and recorded as a decision
    for qid in ("am001", "am002", "am003", "am004", "am005", "adv004", "adv005"):
        assert by_id[qid].expected_abstain is False, qid
        assert "decided-by-sameer" in kinds(qid), qid
    for qid in ("da005", "md003", "md004", "mf002"):
        assert "changed-by-sameer" in kinds(qid), qid
    # The filter on mf002 stays the Policy category
    filtered = yaml.safe_load((FIXTURES / "questions" / "metadata-filtered.yaml").read_text(encoding="utf-8"))
    assert next(q for q in filtered if q["id"] == "mf002")["filters"] == {"category": "Policy"}
    # Nothing else was decided or changed
    changed = {qid for qid in record if "changed-by-sameer" in kinds(qid)}
    assert changed == {"da005", "md003", "md004", "mf002"}


def test_every_filter_admits_the_document_that_holds_its_expected_chunk() -> None:
    """The metadata fact check, repeated as a test: the filter on each filtered question matches the manifest."""
    manifest = yaml.safe_load((FIXTURES / "knowledge_base" / "manifest.yaml").read_text(encoding="utf-8"))
    documents = manifest["documents"]
    filtered = yaml.safe_load((FIXTURES / "questions" / "metadata-filtered.yaml").read_text(encoding="utf-8"))
    expected_documents = {
        "mf001": "access-sop",
        "mf002": "data-retention",
        "mf003": "on-call-rotation",
        "mf004": "security-policy",
        "mf005": "vendor-access",
        "mf006": "remote-work",
        "mf007": "access-sop",
    }
    by_key = {doc["key"]: doc for doc in documents}
    for question in filtered:
        doc = by_key[expected_documents[question["id"]]]
        for field, wanted in question["filters"].items():
            if field == "tags":
                assert set(wanted) <= set(doc["tags"]), question["id"]
            elif field == "document_id":
                assert wanted == doc["document_id"], question["id"]
            else:
                assert doc[field] == wanted, question["id"]
