"""Files on disk: exactly the bytes given, exactly where expected, nowhere else.

Two groups. The first is that a stored file comes back byte for byte and that
a write is all-or-nothing. The second is that a key cannot address anything
outside the root — which in practice no key does, because they are built from
UUIDs, but the check is what makes that a guarantee rather than a habit.
"""

import io
import os
from pathlib import Path

import pytest

from app.storage import LocalStorage, StorageError, UnsafeKey
from app.storage.local import TEMPORARY_PREFIX

KEY = "documents/a/b.pdf"
CONTENT = b"%PDF-1.7\n" + bytes(range(256)) * 40


@pytest.fixture
def storage(tmp_path: Path) -> LocalStorage:
    return LocalStorage(tmp_path)


# --- writing and reading ---------------------------------------------------


def test_a_file_is_written_where_the_key_says(storage: LocalStorage) -> None:
    storage.write(KEY, io.BytesIO(CONTENT))

    assert (storage.root / "documents" / "a" / "b.pdf").is_file()


def test_the_bytes_come_back_exactly(storage: LocalStorage) -> None:
    """Not "roughly": a checksum in a later milestone is computed over these."""
    storage.write(KEY, io.BytesIO(CONTENT))

    with storage.open(KEY) as handle:
        assert handle.read() == CONTENT


def test_the_write_reports_the_size_it_stored(storage: LocalStorage) -> None:
    stored = storage.write(KEY, io.BytesIO(CONTENT))

    assert stored.size == len(CONTENT)
    assert stored.key == KEY


def test_an_empty_file_is_storable(storage: LocalStorage) -> None:
    """Storage does not judge content. Refusing an empty upload is the
    validator's job, and it happens before anything reaches here."""
    assert storage.write(KEY, io.BytesIO(b"")).size == 0


def test_a_larger_file_survives_being_copied_in_chunks(storage: LocalStorage) -> None:
    """Bigger than one copy chunk, so the loop runs more than once."""
    from app.storage.local import COPY_CHUNK_BYTES

    payload = os.urandom(COPY_CHUNK_BYTES * 3 + 17)
    storage.write(KEY, io.BytesIO(payload))

    with storage.open(KEY) as handle:
        assert handle.read() == payload


def test_the_same_key_is_overwritten_not_duplicated(storage: LocalStorage) -> None:
    storage.write(KEY, io.BytesIO(b"first"))
    storage.write(KEY, io.BytesIO(b"second"))

    with storage.open(KEY) as handle:
        assert handle.read() == b"second"


# --- existence and removal -------------------------------------------------


def test_exists_is_false_before_and_true_after(storage: LocalStorage) -> None:
    assert storage.exists(KEY) is False
    storage.write(KEY, io.BytesIO(CONTENT))
    assert storage.exists(KEY) is True


def test_exists_is_false_for_a_directory(storage: LocalStorage) -> None:
    storage.write(KEY, io.BytesIO(CONTENT))

    assert storage.exists("documents/a") is False


def test_delete_removes_the_file(storage: LocalStorage) -> None:
    storage.write(KEY, io.BytesIO(CONTENT))
    storage.delete(KEY)

    assert storage.exists(KEY) is False


def test_deleting_something_that_is_not_there_is_not_an_error(
    storage: LocalStorage
) -> None:
    """The compensating delete on a failed upload must not raise on top of
    the failure it is cleaning up after."""
    storage.delete("documents/never/written.pdf")


def test_opening_something_that_is_not_there_raises_a_storage_error(
    storage: LocalStorage
) -> None:
    with pytest.raises(StorageError):
        storage.open("documents/never/written.pdf")


# --- atomicity -------------------------------------------------------------


def test_a_failed_write_leaves_nothing_at_the_key(storage: LocalStorage) -> None:
    """The reason for the temporary file: a reader must never find a
    half-written document at a real key."""

    class _Breaks(io.RawIOBase):
        def __init__(self) -> None:
            self._sent = False

        def read(self, size: int = -1) -> bytes:
            if not self._sent:
                self._sent = True
                return b"the beginning of a file"
            raise OSError("the connection went away")

    with pytest.raises(StorageError):
        storage.write(KEY, _Breaks())

    assert storage.exists(KEY) is False


def test_a_failed_write_leaves_no_temporary_file(storage: LocalStorage) -> None:
    class _Breaks(io.RawIOBase):
        def read(self, size: int = -1) -> bytes:
            raise OSError("gone")

    with pytest.raises(StorageError):
        storage.write(KEY, _Breaks())

    directory = storage.root / "documents" / "a"
    leftovers = list(directory.glob(f"{TEMPORARY_PREFIX}*")) if directory.exists() else []
    assert leftovers == []


def test_a_failed_write_does_not_destroy_what_was_already_there(
    storage: LocalStorage
) -> None:
    """A rename that never happens cannot damage the previous file."""

    class _Breaks(io.RawIOBase):
        def read(self, size: int = -1) -> bytes:
            raise OSError("gone")

    storage.write(KEY, io.BytesIO(b"the original"))
    with pytest.raises(StorageError):
        storage.write(KEY, _Breaks())

    with storage.open(KEY) as handle:
        assert handle.read() == b"the original"


# --- keys cannot leave the root --------------------------------------------


@pytest.mark.parametrize(
    "key",
    [
        "/etc/passwd",
        "../escaped.pdf",
        "documents/../../escaped.pdf",
        "documents/a/../../../escaped.pdf",
        "..",
        "",
        "   ",
        "\\windows\\system32",
        "C:/windows/system32",
        "documents//double.pdf",
        "./documents/a.pdf",
        "documents/a.pdf/",
    ],
)
def test_an_unsafe_key_is_refused(storage: LocalStorage, key: str) -> None:
    with pytest.raises(UnsafeKey):
        storage.write(key, io.BytesIO(CONTENT))


@pytest.mark.parametrize("key", ["/etc/passwd", "../escaped.pdf", ".."])
def test_every_method_refuses_an_unsafe_key(storage: LocalStorage, key: str) -> None:
    """Not only `write`: a traversal on read or delete is the same bug."""
    for call in (storage.open, storage.delete, storage.exists):
        with pytest.raises(UnsafeKey):
            call(key)


def test_nothing_is_written_outside_the_root(tmp_path: Path) -> None:
    """Measured, not argued: the parent directory is checked afterwards."""
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.pdf"
    storage = LocalStorage(root)

    with pytest.raises(UnsafeKey):
        storage.write("../outside.pdf", io.BytesIO(CONTENT))

    assert outside.exists() is False


def test_a_symlinked_directory_cannot_lead_out_of_the_root(tmp_path: Path) -> None:
    """The second check earns its place here: the key itself is innocent."""
    root = tmp_path / "root"
    root.mkdir()
    (tmp_path / "elsewhere").mkdir()
    (root / "documents").symlink_to(tmp_path / "elsewhere")
    storage = LocalStorage(root)

    with pytest.raises(UnsafeKey):
        storage.write("documents/escaped.pdf", io.BytesIO(CONTENT))

    assert (tmp_path / "elsewhere" / "escaped.pdf").exists() is False
