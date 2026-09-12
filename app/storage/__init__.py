"""Where uploaded files are kept.

The only package in the application that touches the filesystem. Everything
above it deals in keys.
"""

from functools import lru_cache

from app.config import get_settings
from app.storage.base import Storage, StorageError, StoredFile, UnsafeKey
from app.storage.local import LocalStorage


@lru_cache
def get_storage() -> Storage:
    """The storage this process uses, built once."""
    return LocalStorage(get_settings().storage_root)


def reset_storage() -> None:
    """Drop the cached storage. For tests, and for a root that changed."""
    get_storage.cache_clear()


__all__ = [
    "LocalStorage",
    "Storage",
    "StorageError",
    "StoredFile",
    "UnsafeKey",
    "get_storage",
    "reset_storage",
]
