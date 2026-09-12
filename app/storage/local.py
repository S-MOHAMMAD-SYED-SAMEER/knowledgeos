"""Files on the local disk.

The only `Storage` implementation, and the only module in the application that
touches the filesystem.

Two properties are worth the code they cost.

**A write is atomic.** The bytes go to a temporary file in the destination
directory, are flushed and `fsync`ed, and only then are renamed into place with
`os.replace`. Same directory, so the rename is a same-filesystem operation and
therefore atomic; `fsync` first, so "written" means on the disk rather than in
a buffer. A crash mid-upload leaves a stray temporary file — never a truncated
file at the real key, which a later parser would read as a corrupt document.

**A key cannot escape the root.** Keys are generated from UUIDs by this
application, so in practice none of them is hostile. The check is here anyway,
because it is four lines and the failure it prevents is writing outside the
root.
"""

import os
import tempfile
from pathlib import Path, PurePosixPath
from typing import BinaryIO

from app.storage.base import StorageError, StoredFile, UnsafeKey

# How much is moved at a time. Large enough not to make a system call per
# kilobyte, small enough that memory does not follow the size of the upload.
COPY_CHUNK_BYTES = 64 * 1024

TEMPORARY_PREFIX = ".incoming-"


class LocalStorage:
    """Stored files under one root directory."""

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root).resolve()

    @property
    def root(self) -> Path:
        return self._root

    def write(self, key: str, source: BinaryIO) -> StoredFile:
        destination = self._resolve(key)
        destination.parent.mkdir(parents=True, exist_ok=True)

        # In the destination directory, so the rename below cannot cross a
        # filesystem boundary and stop being atomic.
        handle, temporary_name = tempfile.mkstemp(
            dir=destination.parent, prefix=TEMPORARY_PREFIX
        )
        temporary = Path(temporary_name)
        size = 0
        try:
            with os.fdopen(handle, "wb") as target:
                while chunk := source.read(COPY_CHUNK_BYTES):
                    target.write(chunk)
                    size += len(chunk)
                target.flush()
                os.fsync(target.fileno())
            os.replace(temporary, destination)
        except Exception as exc:
            # Nothing was renamed, so there is nothing at `key`. Clear the
            # temporary file so a failed upload leaves no residue at all.
            temporary.unlink(missing_ok=True)
            raise StorageError(f"could not store {key!r}") from exc

        return StoredFile(key=key, size=size)

    def open(self, key: str) -> BinaryIO:
        path = self._resolve(key)
        try:
            return path.open("rb")
        except OSError as exc:
            raise StorageError(f"could not read {key!r}") from exc

    def delete(self, key: str) -> None:
        self._resolve(key).unlink(missing_ok=True)

    def exists(self, key: str) -> bool:
        return self._resolve(key).is_file()

    # --- the one place a key becomes a path --------------------------------

    def _resolve(self, key: str) -> Path:
        """Turn a key into a path inside the root, or refuse.

        Checked twice on purpose. The first check reads the key as written —
        a `..` segment or a leading slash is refused whatever the filesystem
        would have made of it. The second checks the result, which is what
        catches anything the first did not think of, including a symlinked
        parent directory pointing out of the root.
        """
        if not key or key != key.strip():
            raise UnsafeKey("a storage key must be a non-empty path")

        pure = PurePosixPath(key)
        if pure.is_absolute() or key.startswith("\\") or ":" in key:
            raise UnsafeKey("a storage key must be relative")
        if any(part == ".." for part in pure.parts):
            raise UnsafeKey("a storage key must not contain '..'")
        # Keys are built by this application, so one that is not already in
        # canonical form — a doubled slash, a `.` segment, a trailing slash —
        # is a bug in the caller rather than a file to store.
        if str(pure) != key:
            raise UnsafeKey("a storage key must be a normalised relative path")

        resolved = (self._root / pure).resolve()
        if resolved != self._root and self._root not in resolved.parents:
            raise UnsafeKey("a storage key must stay inside the storage root")
        return resolved


__all__ = ["COPY_CHUNK_BYTES", "LocalStorage"]
